//
//   Copyright 2018 DreamWorks Animation LLC.
//
//   Licensed under the Apache License, Version 2.0 (the "Apache License")
//   with the following modification; you may not use this file except in
//   compliance with the Apache License and the following modification to it:
//   Section 6. Trademarks. is deleted and replaced with:
//
//   6. Trademarks. This License does not grant permission to use the trade
//      names, trademarks, service marks, or product names of the Licensor
//      and its affiliates, except as required to comply with Section 4(c) of
//      the License and to reproduce the content of the NOTICE file.
//
//   You may obtain a copy of the Apache License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the Apache License with the above modification is
//   distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
//   KIND, either express or implied. See the Apache License for the specific
//   language governing permissions and limitations under the Apache License.
//

#include "../far/patchBuilder.h"
#include "../far/catmarkPatchBuilder.h"
#include "../far/loopPatchBuilder.h"
#include "../far/bilinearPatchBuilder.h"
#include "../vtr/level.h"
#include "../vtr/fvarLevel.h"
#include "../vtr/refinement.h"
#include "../vtr/stackBuffer.h"

#include <cassert>
#include <cstdio>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

using Vtr::Array;
using Vtr::ConstArray;
using Vtr::internal::Level;
using Vtr::internal::FVarLevel;
using Vtr::internal::Refinement;
using Vtr::internal::StackBuffer;

namespace Far {

//
//  Local helper functions for topology queries:
//
namespace {
    //
    //  Local helper functions for identifying the subset of a ring around a
    //  corner that contributes to a patch -- parameterized by a mask that
    //  indicates what kind of edge is to delimit the span.
    //
    //  Note that the two main methods need both face-verts and face-edges
    //  for each corner, and that we don't really need the face-index once
    //  we have them -- consider passing the fVerts and fEdges as arguments
    //  as they will otherwise be retrieved repeatedly for each corner.
    //
    //  (As these mature it is likely they will be moved to Vtr, as a method
    //  to identify a VSpan would complement the existing method to gather
    //  the vertex/values associated with it.  The manifold vs non-manifold
    //  choice would then also be encapsulated -- provided both remain free
    //  of PatchTable-specific logic.)
    //
    inline Level::ETag
    getSingularEdgeMask(bool includeAllInfSharpEdges = false) {

        Level::ETag eTagMask;
        eTagMask.clear();
        eTagMask._boundary = true;
        eTagMask._nonManifold = true;
        eTagMask._infSharp = includeAllInfSharpEdges;
        return eTagMask;
    }

    inline bool
    isEdgeSingular(Level const & level, FVarLevel const * fvarLevel,
                   Index eIndex, Level::ETag eTagMask)
    {
        Level::ETag eTag = level.getEdgeTag(eIndex);
        if (fvarLevel) {
            eTag = fvarLevel->getEdgeTag(eIndex).combineWithLevelETag(eTag);
        }

        Level::ETag::ETagSize * iTag  =
                reinterpret_cast<Level::ETag::ETagSize*>(&eTag);
        Level::ETag::ETagSize * iMask =
                reinterpret_cast<Level::ETag::ETagSize*>(&eTagMask);

        return (*iTag & *iMask) > 0;
    }

    void
    identifyManifoldCornerSpan(Level const & level, Index fIndex,
                               int fCorner, Level::ETag eTagMask,
                               Level::VSpan & vSpan, int fvc = -1)
    {
        FVarLevel const * fvarLevel = (fvc < 0) ? 0 : &level.getFVarLevel(fvc);

        ConstIndexArray fVerts = level.getFaceVertices(fIndex);
        ConstIndexArray fEdges = level.getFaceEdges(fIndex);

        ConstIndexArray vEdges = level.getVertexEdges(fVerts[fCorner]);
        int             nEdges = vEdges.size();

        int iLeadingStart  = vEdges.FindIndex(fEdges[fCorner]);
        int iTrailingStart = (iLeadingStart + 1) % nEdges;

        vSpan.clear();
        vSpan._numFaces = 1;

        int iLeading  = iLeadingStart;
        while (! isEdgeSingular(level, fvarLevel, vEdges[iLeading], eTagMask)) {
            ++vSpan._numFaces;
            iLeading = (iLeading + nEdges - 1) % nEdges;
            if (iLeading == iTrailingStart) break;
        }

        int iTrailing = iTrailingStart;
        while (!isEdgeSingular(level, fvarLevel, vEdges[iTrailing], eTagMask)) {
            ++vSpan._numFaces;
            iTrailing = (iTrailing + 1) % nEdges;
            if (iTrailing == iLeadingStart) break;
        }
        vSpan._startFace = (LocalIndex) iLeading;
    }

    void
    identifyNonManifoldCornerSpan(Level const & level, Index fIndex,
                                  int fCorner, Level::ETag /* eTagMask */,
                                  Level::VSpan & vSpan, int /* fvc */ = -1)
    {
        //  For now, non-manifold patches revert to regular patches -- just
        //  identify the single face now for a sharp corner patch.
        //
        //  Remember that the face may be incident the vertex multiple times
        //  when non-manifold, so make sure the local index of the corner
        //  vertex in the face identified additionally matches the corner.
        //
        //FVarLevel * fvarLevel = (fvc < 0) ? 0 : &level.getFVarChannel(fvc);

        Index vIndex = level.getFaceVertices(fIndex)[fCorner];

        ConstIndexArray      vFaces  = level.getVertexFaces(vIndex);
        ConstLocalIndexArray vInFace = level.getVertexFaceLocalIndices(vIndex);

        vSpan.clear();
        for (int i = 0; i < vFaces.size(); ++i) {
            if ((vFaces[i] == fIndex) && ((int)vInFace[i] == fCorner)) {
                vSpan._startFace = (LocalIndex) i;
                vSpan._numFaces = 1;
                vSpan._sharp = true;
                break;
            }
        }
        assert(vSpan._numFaces == 1);
    }
} // namespace anon


//
//  Factory method and constructor:
//
PatchBuilder*
PatchBuilder::Create(TopologyRefiner const& refiner, Options const& options) {

    switch (refiner.GetSchemeType()) {
    case Sdc::SCHEME_BILINEAR:
        return new BilinearPatchBuilder(refiner, options);
    case Sdc::SCHEME_CATMARK:
        return new CatmarkPatchBuilder(refiner, options);
    case Sdc::SCHEME_LOOP:
        return new LoopPatchBuilder(refiner, options);
    }
    assert("Unrecognized Sdc::SchemeType for PatchBuilder construction" == 0);
    return 0;
}

PatchBuilder::PatchBuilder(
    TopologyRefiner const& refiner, Options const& options) :
        _refiner(refiner), _options(options) {

    //
    //  Initialize members with properties of the subdivision scheme and patch
    //  choices for quick retrieval:
    //
    _schemeType         = refiner.GetSchemeType();
    _schemeRegFaceSize  = Sdc::SchemeTypeTraits::GetRegularFaceSize(_schemeType);
    _schemeNeighborhood = Sdc::SchemeTypeTraits::GetLocalNeighborhoodSize(_schemeType);

    //  Initialization of members involving patch types is deferred to the
    //  subclass for the scheme
}

PatchBuilder::~PatchBuilder() {
}

//
//  This inline assertion is invoked in code that is intended to support
//  triangular patches, but has not yet been fully validated as doing so --
//  typically the public methods used by clients.
//
//  This is in contrast to code that is never intended to support triangles,
//  i.e. code for which a test for quads is a pre-condition, where the
//  explicit assertion should be used instead of this method (as it will be
//  completely removed once planned support for triangular patches is done).
//
inline void
PatchBuilder::assertTriangularPatchesNotYetSupportedHere() const {

    assert(_schemeRegFaceSize == 4);
}

//
//  Topology inspections methods for a particular face in the hierarchy:
//
bool
PatchBuilder::IsFaceAPatch(int levelIndex, Index faceIndex) const {

    Level const & level = _refiner.getLevel(levelIndex);
    ConstIndexArray fVerts = level.getFaceVertices(faceIndex);

    //  Fail if the face is a hole (i.e. no limit surface)
    if (level.isFaceHole(faceIndex)) return false;

    //  Fail if the face is irregular
    if (fVerts.size() != _schemeRegFaceSize) return false;

    //  Fail if the face lacks its complete neighborhood of support
    if (level.getFaceCompositeVTag(fVerts)._incomplete) return false;

    return true;
}

bool
PatchBuilder::IsFaceALeaf(int levelIndex, Index faceIndex) const {

    //  All faces in the last level are leaves
    if (levelIndex < _refiner.GetMaxLevel()) {
        //  Faces selected for further refinement are not leaves
        if (_refiner.getRefinement(levelIndex).
                        getParentFaceSparseTag(faceIndex)._selected) {
            return false;
        }
    }
    return true;
}

bool
PatchBuilder::IsPatchRegular(int levelIndex, Index faceIndex,
    int fvarChannel) const {

    assertTriangularPatchesNotYetSupportedHere();

    if (_schemeNeighborhood == 0) {
        //  The previous face-is-a-patch test precludes an irregular patch
        return true;
    }

    Level const & level = _refiner.getLevel(levelIndex);

    //  Retrieve the composite VTag for the four corners:
    Level::VTag fCompVTag = level.getFaceCompositeVTag(faceIndex, fvarChannel);

    //  All patches around non-manifold features are currently regular:
    bool isRegular = ! fCompVTag._xordinary || fCompVTag._nonManifold;

    //  Reconsider when using inf-sharp patches at inf-sharp features:
    if (!_options.approxInfSharpWithSmooth &&
            (fCompVTag._infSharp || fCompVTag._infSharpEdges)) {

        if (fCompVTag._nonManifold || !fCompVTag._infIrregular) {
            isRegular = true;
        } else if (!fCompVTag._infSharpEdges) {
            isRegular = false;
        } else {
            //
            //   This is unfortunately a relatively complex case to determine...
            //   if a corner vertex has been tagged has having an inf-sharp
            //   irregularity about it, the neighborhood of the corner is
            //   partitioned into both regular and irregular regions and the
            //   face must be more closely inspected to determine in which it
            //   lies.
            //
            //   There could be a simpler test here to quickly accept/reject
            //   regularity given how local it is -- involving no more than
            //   one or two (in the case of Loop) adjacent faces -- but it will
            //   likely be messy and will need to inspect adjacent faces and/or
            //   edges.  In the meantime, gathering and inspecting the subset
            //   of the neighborhood delimited by inf-sharp edges will suffice
            //   (and be comparable in all but cases of high valence)
            //
            Level::VTag vTags[4];
            level.getFaceVTags(faceIndex, vTags, fvarChannel);

            Level::VSpan vSpan;
            Level::ETag eMask = getSingularEdgeMask(true);

            isRegular = true;
            for (int i = 0; i < 4; ++i) {
                if (vTags[i]._infIrregular) {
                    identifyManifoldCornerSpan(
                        level, faceIndex, i, eMask, vSpan, fvarChannel);

                    isRegular = (vSpan._numFaces ==
                                    (vTags[i]._infSharpCrease ? 2 : 1));
                    if (!isRegular) break;
                }
            }
        }

        //  When inf-sharp and extra-ordinary features are not isolated, need
        //  to inspect more closely -- any smooth extra-ordinary corner makes
        //  the patch irregular:
        if (fCompVTag._xordinary && (levelIndex < 2)) {
            Level::VTag vTags[4];
            level.getFaceVTags(faceIndex, vTags, fvarChannel);
            for (int i = 0; i < 4; ++i) {
                if (vTags[i]._xordinary &&
                       (vTags[i]._rule == Sdc::Crease::RULE_SMOOTH)) {
                    isRegular = false;
                }
            }
        }
    }

    //  Legacy option -- reinterpret smooth corner as sharp if specified:
    if (!isRegular && _options.approxSmoothCornerWithSharp) {
        if (fCompVTag._xordinary && fCompVTag._boundary && !fCompVTag._nonManifold) {
            isRegular = isPatchSmoothCorner(levelIndex, faceIndex, fvarChannel);
        }
    }
    return isRegular;
}

bool
PatchBuilder::isPatchSmoothCorner(int levelIndex, Index faceIndex,
    int fvarChannel) const {

    Level const & level = _refiner.getLevel(levelIndex);

    ConstIndexArray fVerts = level.getFaceVertices(faceIndex);
    if (fVerts.size() != 4) return false;

    Level::VTag vTags[4];
    level.getFaceVTags(faceIndex, vTags, fvarChannel);

    //
    //  Test the subdivision rules for the corners, rather than just the
    //  boundary/interior tags, to ensure that inf-sharp vertices or edges
    //  are properly accounted for (and the cases appropriately excluded)
    //  if inf-sharp patches are enabled:
    //
    int boundaryCount = 0;
    if (!_options.approxInfSharpWithSmooth) {
        boundaryCount =
            (vTags[0]._infSharpEdges && (vTags[0]._rule == Sdc::Crease::RULE_CREASE)) +
            (vTags[1]._infSharpEdges && (vTags[1]._rule == Sdc::Crease::RULE_CREASE)) +
            (vTags[2]._infSharpEdges && (vTags[2]._rule == Sdc::Crease::RULE_CREASE)) +
            (vTags[3]._infSharpEdges && (vTags[3]._rule == Sdc::Crease::RULE_CREASE));
    } else {
        boundaryCount =
            (vTags[0]._boundary && (vTags[0]._rule == Sdc::Crease::RULE_CREASE)) +
            (vTags[1]._boundary && (vTags[1]._rule == Sdc::Crease::RULE_CREASE)) +
            (vTags[2]._boundary && (vTags[2]._rule == Sdc::Crease::RULE_CREASE)) +
            (vTags[3]._boundary && (vTags[3]._rule == Sdc::Crease::RULE_CREASE));
    }
    int xordinaryCount = vTags[0]._xordinary
                       + vTags[1]._xordinary
                       + vTags[2]._xordinary
                       + vTags[3]._xordinary;
    
    if ((boundaryCount == 3) && (xordinaryCount == 1)) {
        //  This must be an isolated xordinary corner above level 1, otherwise
        //  we still need to assure the xordinary vertex is opposite a smooth
        //  interior vertex:
        //
        if (levelIndex > 1) return true;
    
        if (vTags[0]._xordinary) return (vTags[2]._rule == Sdc::Crease::RULE_SMOOTH);
        if (vTags[1]._xordinary) return (vTags[3]._rule == Sdc::Crease::RULE_SMOOTH);
        if (vTags[2]._xordinary) return (vTags[0]._rule == Sdc::Crease::RULE_SMOOTH);
        if (vTags[3]._xordinary) return (vTags[1]._rule == Sdc::Crease::RULE_SMOOTH);
    }
    return false;
}

int
PatchBuilder::GetRegularPatchBoundaryMask(int levelIndex, Index faceIndex,
    int fvarChannel) const {

    assertTriangularPatchesNotYetSupportedHere();

    if (_schemeNeighborhood == 0) {
        //  Boundaries for patches not dependent on the 1-ring are ignored
        return 0;
    }

    Level const & level = _refiner.getLevel(levelIndex);

    //  Gather the VTags for the four corners.  Regardless of the options
    //  for treating non-manifold or inf-sharp patches, for a regular patch
    //  we can infer all that we need need from tags for the corner vertices:
    //
    Level::VTag vTags[4];
    level.getFaceVTags(faceIndex, vTags, fvarChannel);

    Level::VTag fTag = Level::VTag::BitwiseOr(vTags);

    //
    //  Identify vertex tags for inf-sharp edges and/or boundaries, depending
    //  on whether or not inf-sharp patches are in use:
    //
    int vBoundaryMask = 0;
    if (fTag._infSharpEdges) {
        if (!_options.approxInfSharpWithSmooth) {
            vBoundaryMask |= (vTags[0]._infSharpEdges << 0) |
                             (vTags[1]._infSharpEdges << 1) |
                             (vTags[2]._infSharpEdges << 2) |
                             (vTags[3]._infSharpEdges << 3);
        } else if (fTag._boundary) {
            vBoundaryMask |= (vTags[0]._boundary << 0) |
                             (vTags[1]._boundary << 1) |
                             (vTags[2]._boundary << 2) |
                             (vTags[3]._boundary << 3);
        }
    }

    //
    //  Non-manifold patches have been historically represented as regular
    //  in all cases -- when a non-manifold vertex is sharp, it requires a
    //  regular corner patch, and so both of its neighboring corners need to
    //  be re-interpreted as boundaries.
    //
    //  With the introduction of sharp irregular patches, we are now better
    //  off using irregular patches where appropriate, which will simplify
    //  the following when this patch was already determined to be regular.
    //
    if (fTag._nonManifold) {
        if (vTags[0]._nonManifold)
            vBoundaryMask |= (1 << 0) | (vTags[0]._infSharp ? 10 : 0);
        if (vTags[1]._nonManifold)
            vBoundaryMask |= (1 << 1) | (vTags[1]._infSharp ?  5 : 0);
        if (vTags[2]._nonManifold)
            vBoundaryMask |= (1 << 2) | (vTags[2]._infSharp ? 10 : 0);
        if (vTags[3]._nonManifold)
            vBoundaryMask |= (1 << 3) | (vTags[3]._infSharp ?  5 : 0);

        //  Force adjacent edges as boundaries if only one vertex in the
        //  resulting mask (which would be an irregular boundary for Catmark,
        //  but not Loop):
        if ((vBoundaryMask == (1 << 0)) || (vBoundaryMask == (1 << 2))) {
            vBoundaryMask |= 10;
        } else if ((vBoundaryMask == (1 << 1)) || (vBoundaryMask == (1 << 3))) {
            vBoundaryMask |= 5;
        }
    }

    //  Convert directly from a vertex- to edge-mask (no need to inspect edges):
    int eBoundaryMask = 0;
    if (vBoundaryMask) {
        static int const vBoundaryMaskToEMask[16] =
                { 0, -1, -1, 1, -1, -1, 2, 3, -1, 8, -1, 9, 4, 12, 6, -1 };
        eBoundaryMask = vBoundaryMaskToEMask[vBoundaryMask];
        assert(eBoundaryMask != -1);
    }
    return eBoundaryMask;
}

void
PatchBuilder::GetIrregularPatchCornerSpans(int levelIndex, Index faceIndex,
        Level::VSpan cornerSpans[4], int fvarChannel) const {

    assertTriangularPatchesNotYetSupportedHere();

    Level const & level = _refiner.getLevel(levelIndex);

    //  Retrieve tags and identify other information for the corner vertices:
    Level::VTag vTags[4];
    level.getFaceVTags(faceIndex, vTags, fvarChannel);

    FVarLevel::ValueTag fvarTags[4];
    if (fvarChannel >= 0) {
        level.getFVarLevel(fvarChannel).getFaceValueTags(faceIndex, fvarTags);
    }

    //
    //  For each corner vertex, use the complete neighborhood when possible
    //  (which does not require a search, otherwise identify the span of
    //  interest around the vertex:
    //
    ConstIndexArray fVerts = level.getFaceVertices(faceIndex);

    Level::ETag singularEdgeMask =
        getSingularEdgeMask(!_options.approxInfSharpWithSmooth);

    for (int i = 0; i < fVerts.size(); ++i) {
        bool noFVarMisMatch = (fvarChannel < 0) || !fvarTags[i]._mismatch;

        bool testInfSharp = !_options.approxInfSharpWithSmooth &&
                            vTags[i]._infSharpEdges &&
                            (vTags[i]._rule != Sdc::Crease::RULE_DART);

        if (noFVarMisMatch && !testInfSharp) {
            cornerSpans[i].clear();
        } else {
            if (!vTags[i]._nonManifold) {
                identifyManifoldCornerSpan(level, faceIndex,
                        i, singularEdgeMask, cornerSpans[i], fvarChannel);
            } else {
                identifyNonManifoldCornerSpan(level, faceIndex,
                        i, singularEdgeMask, cornerSpans[i], fvarChannel);
            }
        }
        if (vTags[i]._corner) {
            cornerSpans[i]._sharp = true;
        } else if (!_options.approxInfSharpWithSmooth) {
            cornerSpans[i]._sharp = vTags[i]._infIrregular &&
                                   (vTags[i]._rule == Sdc::Crease::RULE_CORNER);
        }

        //  Legacy option -- reinterpret smooth corner as sharp if specified:
        if (!cornerSpans[i]._sharp && _options.approxSmoothCornerWithSharp) {
            if (vTags[i]._xordinary && vTags[i]._boundary && !vTags[i]._nonManifold) {
                int nFaces = cornerSpans[i].isAssigned()
                           ? cornerSpans[i]._numFaces
                           : level.getVertexFaces(fVerts[i]).size();
                cornerSpans[i]._sharp = (nFaces == 1);
            }
        }
    }
}

int
PatchBuilder::getRegularFacePoints(int levelIndex, Index faceIndex,
        Index patchPoints[], int fvarChannel) const {

    Level const & level = _refiner.getLevel(levelIndex);

    ConstIndexArray facePoints = (fvarChannel < 0)
                               ? level.getFaceVertices(faceIndex)
                               : level.getFaceFVarValues(faceIndex, fvarChannel);

    for (int i = 0; i < facePoints.size(); ++i) {
        patchPoints[i] = facePoints[i];
    }
    return facePoints.size();
}

int
PatchBuilder::getQuadRegularPatchPoints(int levelIndex, Index faceIndex,
        int regBoundaryMask, Index patchPoints[],
        int fvarChannel) const {

    if (regBoundaryMask < 0) {
        regBoundaryMask = GetRegularPatchBoundaryMask(levelIndex, faceIndex);
    }

    int bType  = 0;
    int bIndex = 0;
    if (regBoundaryMask) {
        static int const boundaryEdgeMaskToType[16] =
            { 0, 1, 1, 2, 1, -1, 2, -1, 1, 2, -1, -1, 2, -1, -1, -1 };
        static int const boundaryEdgeMaskToFeature[16] =
            { -1, 0, 1, 1, 2, -1, 2, -1, 3, 0, -1, -1, 3, -1, -1,-1 };

        bType  = boundaryEdgeMaskToType[regBoundaryMask];
        bIndex = boundaryEdgeMaskToFeature[regBoundaryMask];
    }

    Level const & level = _refiner.getLevel(levelIndex);

    Index sourcePoints[16];

    int const * permutation = 0;
    if (bType == 0) {
        static int const permuteRegular[16] =
            { 5, 6, 7, 8, 4, 0, 1, 9, 15, 3, 2, 10, 14, 13, 12, 11 };
        permutation = permuteRegular;
        level.gatherQuadRegularInteriorPatchPoints(
                faceIndex, sourcePoints, /*rotation=*/0, fvarChannel);
    } else if (bType == 1) {
        // Expand boundary patch vertices and rotate to
        // restore correct orientation.
        static int const permuteBoundary[4][16] = {
            { -1, -1, -1, -1, 11, 3, 0, 4, 10, 2, 1, 5, 9, 8, 7, 6 },
            { 9, 10, 11, -1, 8, 2, 3, -1, 7, 1, 0, -1, 6, 5, 4, -1 },
            { 6, 7, 8, 9, 5, 1, 2, 10, 4, 0, 3, 11, -1, -1, -1, -1 },
            { -1, 4, 5, 6, -1, 0, 1, 7, -1, 3, 2, 8, -1, 11, 10, 9 } };
        permutation = permuteBoundary[bIndex];
        level.gatherQuadRegularBoundaryPatchPoints(
                faceIndex, sourcePoints, bIndex, fvarChannel);
    } else if (bType == 2) {
        // Expand corner patch vertices and rotate to
        // restore correct orientation.
        static int const permuteCorner[4][16] = {
            { -1, -1, -1, -1, -1, 0, 1, 4, -1, 3, 2, 5, -1, 8, 7, 6 },
            { -1, -1, -1, -1, 8, 3, 0, -1, 7, 2, 1, -1, 6, 5, 4, -1 },
            { 6, 7, 8, -1, 5, 2, 3, -1, 4, 1, 0, -1, -1, -1, -1, -1 },
            { -1, 4, 5, 6, -1, 1, 2, 7, -1, 0, 3, 8, -1, -1, -1, -1 } };
        permutation = permuteCorner[bIndex];
        level.gatherQuadRegularCornerPatchPoints(
                faceIndex, sourcePoints, bIndex, fvarChannel);
    }
    assert(permutation != 0);

    //  Re-orient the points into a row-wise order and (optionally) fill in any
    //  missing boundary points with a known point (first of the source points)
    if (regBoundaryMask == 0) {
        for (int i = 0; i < 16; ++i) {
            patchPoints[i] = sourcePoints[permutation[i]];
        }
    } else {
        for (int i = 0; i < 16; ++i) {
            if (permutation[i] >= 0) {
                patchPoints[i] = sourcePoints[permutation[i]];
            } else if (_options.fillMissingBoundaryPoints) {
                patchPoints[i] = sourcePoints[0];
            } else {
                patchPoints[i] = INDEX_INVALID;
            }
        }
    }
    return 16;
}

int
PatchBuilder::GetRegularPatchPoints(int levelIndex, Index faceIndex,
        int regBoundaryMask, Index patchPoints[],
        int fvarChannel) const {

    if (_schemeNeighborhood == 0) {
        return getRegularFacePoints(
            levelIndex, faceIndex, patchPoints, fvarChannel);
    } else if (_schemeRegFaceSize == 4) {
        return getQuadRegularPatchPoints(
            levelIndex, faceIndex, regBoundaryMask, patchPoints, fvarChannel);
    } else {
        assertTriangularPatchesNotYetSupportedHere();
        //  return getTriRegularPatchPoints(
        //      levelIndex, faceIndex, regBoundaryMask, patchPoints, fvarChannel);
    }
    return 0;
}

int
PatchBuilder::assembleIrregularSourcePatch(
        int levelIndex, Index faceIndex, Level::VSpan const cornerSpans[],
        SourcePatch & sourcePatch) const {

    //
    //  Initialize the four Patch corners and finalize the patch:
    //
    Level const & level = _refiner.getLevel(levelIndex);

    ConstIndexArray fVerts = level.getFaceVertices(faceIndex);

    for (int corner = 0; corner < 4; ++corner) {
        ConstIndexArray vFaces = level.getVertexFaces(fVerts[corner]);

        //
        //  Identify the face for the patch within the given ring or sub-ring.
        //
        //  Note also that specifying a sub-ring in a VSpan currently implies
        //  it is a boundary or dart (if all faces present) -- we need a
        //  better way of specifying corner properties such as boundary, dart,
        //  sharp, etc. (possibly a VTag for each corner in addition to the
        //  VSpan)
        // 
        Level::VTag vTag = level.getVertexTag(fVerts[corner]);

        int  numFaces   = 0;
        int  firstFace  = 0;
        bool isBoundary = false;

        if (cornerSpans[corner]._numFaces == 0) {
            numFaces   = vFaces.size();
            firstFace  = 0;
            isBoundary = vTag._boundary;
        } else {
            numFaces   = cornerSpans[corner]._numFaces;
            firstFace  = cornerSpans[corner]._startFace;
            isBoundary = true;
        }

        int patchFace = 0;
        for ( ; patchFace < numFaces; ++patchFace) {
            int vFaceIndex = (firstFace + patchFace) % vFaces.size();
            if (vFaces[vFaceIndex] == faceIndex) {
                break;
            }
        }
        assert(patchFace < numFaces);

        sourcePatch._corners[corner]._boundary  = isBoundary;
        sourcePatch._corners[corner]._sharp     = cornerSpans[corner]._sharp;
        sourcePatch._corners[corner]._dart      = (vTag._rule == Sdc::Crease::RULE_DART);
        sourcePatch._corners[corner]._numFaces  = numFaces;
        sourcePatch._corners[corner]._patchFace = patchFace;
    }
    sourcePatch.Finalize();
    return sourcePatch.GetNumSourcePoints();
}


//
//  Gather patch points from around the face of a level given a previously
//  initialized SourcePatch.  This is historically specific to an irregular
//  patch and still relies on the cornerSpans (which may or may not have been
//  initialized when the SourcePatch was created) rather than inspecting the
//  corners of the SourcePatch.
//
//  We need temporary/local space for rings around each corner -- both for
//  the Vtr::Level and the corresponding rings of the patch.
//
//  Get the corresponding rings from the Vtr::Level and the patch descriptor:
//  the values of the latter will be indices for points[] whose values will
//  come from values of former, i.e. points[localRing[i]] = sourceRing[i].
//  Points that overlap will be assigned multiple times, but messy logic to
//  deal with overlap while determining the correspondence is avoided.
//
int
PatchBuilder::gatherIrregularSourcePoints(
        int levelIndex, Index faceIndex,
        Level::VSpan const cornerSpans[4], SourcePatch & sourcePatch,
        Index patchVerts[], int fvarChannel) const

{
    //
    //  Allocate temporary space for rings around the corners in both the Level
    //  and the Patch, then retrieve corresponding rings and assign the source
    //  vertices to the given array of patch points
    //
    int numSourceVerts = sourcePatch.GetNumSourcePoints();

    StackBuffer<Index,64,true> sourceRingVertices(sourcePatch.GetMaxRingSize());
    StackBuffer<Index,64,true> patchRingPoints(sourcePatch.GetMaxRingSize());

    //  Debugging -- all entries should be assigned (see test after assignment)
    const int debugUnassignedPointIndex = -1;
    for (int i = 0; i < numSourceVerts; ++i) {
        patchVerts[i] = debugUnassignedPointIndex;
    }

    Level const & level = _refiner.getLevel(levelIndex);

    ConstIndexArray faceVerts = level.getFaceVertices(faceIndex);
    for (int corner = 0; corner < 4; ++corner) {
        Index cornerVertex = faceVerts[corner];
        
        //  Gather the ring of source points from the Vtr level:
        int sourceRingSize = 0;
        if (!cornerSpans[corner].isAssigned()) {
            sourceRingSize = level.gatherQuadRegularRingAroundVertex(
                cornerVertex, sourceRingVertices,
                fvarChannel);
        } else {
            sourceRingSize = level.gatherQuadRegularPartialRingAroundVertex(
                cornerVertex, cornerSpans[corner], sourceRingVertices,
                fvarChannel);
        }

        //  Gather the ring of local points from the patch:
        int patchRingSize = sourcePatch.GetCornerRingPoints(
                corner, patchRingPoints);
        assert(patchRingSize == sourceRingSize);

        //  Identify source points for corresponding local patch points of ring:
        for (int i = 0; i < patchRingSize; ++i) {
            assert(patchRingPoints[i] < numSourceVerts);
            patchVerts[patchRingPoints[i]] = sourceRingVertices[i];
        }
    }

    //  Debugging -- all entries should be assigned (see pre-assignment above)
    for (int i = 0; i < numSourceVerts; ++i) {
        assert(patchVerts[i] != debugUnassignedPointIndex);
    }
    return numSourceVerts;
}

int
PatchBuilder::GetIrregularPatchSourcePoints(
        int levelIndex, Index faceIndex, Level::VSpan const cornerSpans[],
        Index sourcePoints[], int fvarChannel) const {

    assertTriangularPatchesNotYetSupportedHere();

    SourcePatch sourcePatch;
    assembleIrregularSourcePatch(
            levelIndex, faceIndex, cornerSpans, sourcePatch);

    return gatherIrregularSourcePoints(levelIndex, faceIndex,
        cornerSpans, sourcePatch, sourcePoints, fvarChannel);
}

int
PatchBuilder::GetIrregularPatchConversionMatrix(
        int levelIndex, Index faceIndex,
        Level::VSpan const cornerSpans[],
        SparseMatrix<float> & conversionMatrix) const {

    assertTriangularPatchesNotYetSupportedHere();

    SourcePatch sourcePatch;
    assembleIrregularSourcePatch(
            levelIndex, faceIndex, cornerSpans, sourcePatch);

    return convertToPatchType(
        sourcePatch, GetIrregularPatchType(), conversionMatrix);
}


bool
PatchBuilder::IsRegularSingleCreasePatch(int levelIndex, Index faceIndex,
        SingleCreaseInfo & creaseInfo) const {

    assertTriangularPatchesNotYetSupportedHere();

    Level const & level = _refiner.getLevel(levelIndex);

    return level.isSingleCreasePatch(faceIndex,
                &creaseInfo.creaseSharpness, &creaseInfo.creaseEdgeInFace);
}   

PatchParam
PatchBuilder::ComputePatchParam(int levelIndex, Index faceIndex,
        PtexIndices const& ptexIndices,
        int boundaryMask, bool computeTransitionMask) const {

    // Move up the hierarchy accumulating u,v indices to the coarse level:
    int depth = levelIndex;
    int childIndexInParent = 0,
        u = 0,
        v = 0,
        ofs = 1;

    int regFaceSize = _schemeRegFaceSize;

    bool irregular =
        _refiner.GetLevel(depth).GetFaceVertices(faceIndex).size() !=
        regFaceSize;

    // For triangle refinement, the parameterization is rotated at
    // the fourth triangle subface at each level. The u and v values
    // computed for rotated triangles will be negative while we are
    // walking through the refinement levels.
    bool rotatedTriangle = false;

    int childFaceIndex = faceIndex;
    for (int i = depth; i > 0; --i) {
        Refinement const& refinement  = _refiner.getRefinement(i-1);
        Level const&      parentLevel = _refiner.getLevel(i-1);

        Index parentFaceIndex =
            refinement.getChildFaceParentFace(childFaceIndex);

        irregular =
            parentLevel.getFaceVertices(parentFaceIndex).size() !=
            regFaceSize;

        if (_schemeRegFaceSize == 3) {
            // For now, we don't consider irregular faces for
            // triangle refinement.

            childIndexInParent =
                refinement.getChildFaceInParentFace(childFaceIndex);

            if (rotatedTriangle) {
                switch ( childIndexInParent ) {
                    case 0 :                     break;
                    case 1 : { u-=ofs;         } break;
                    case 2 : {         v-=ofs; } break;
                    case 3 : { u+=ofs; v+=ofs; rotatedTriangle = false; } break;
                }
            } else {
                switch ( childIndexInParent ) {
                    case 0 :                     break;
                    case 1 : { u+=ofs;         } break;
                    case 2 : {         v+=ofs; } break;
                    case 3 : { u-=ofs; v-=ofs; rotatedTriangle = true; } break;
                }
            }
            ofs = (unsigned short)(ofs << 1);
        } else if (!irregular) {
            childIndexInParent =
                refinement.getChildFaceInParentFace(childFaceIndex);

            switch ( childIndexInParent ) {
                case 0 :                     break;
                case 1 : { u+=ofs;         } break;
                case 2 : { u+=ofs; v+=ofs; } break;
                case 3 : {         v+=ofs; } break;
            }
            ofs = (unsigned short)(ofs << 1);
        } else {
            // If the root face is not a quad, we need to offset the ptex index
            // CCW to match the correct child face
            ConstIndexArray children =
                refinement.getFaceChildFaces(parentFaceIndex);

            for (int j=0; j<children.size(); ++j) {
                if (children[j] == childFaceIndex) {
                    childIndexInParent = j;
                    break;
                }
            }
        }
        childFaceIndex = parentFaceIndex;
    }
    if (rotatedTriangle) {
        // If the triangle is tagged as rotated at this point then the
        // computed u and v parameters will both be negative and we map
        // them onto positive values in the opposite diagonal of the
        // parameter space.
        u += ofs;
        v += ofs;
    }
    int baseFaceIndex = childFaceIndex;

    //  Need to store ptex index from base face and child of an irregular face:
    Index ptexIndex = ptexIndices.GetFaceId(baseFaceIndex);
    assert(ptexIndex != -1);
    if (irregular) {
        ptexIndex += childIndexInParent;
    }

    //  Compute/identify the transition mask if requested, otherwise leave it 0:
    int transitionMask = 0;
    if (computeTransitionMask && (levelIndex < _refiner.GetMaxLevel())) {
        transitionMask = _refiner.getRefinement(levelIndex).
                            getParentFaceSparseTag(faceIndex)._transitional;
    }

    PatchParam param;
    param.Set(ptexIndex, (short)u, (short)v, (unsigned short) depth, irregular,
              (unsigned short) boundaryMask, (unsigned short) transitionMask);
    return param;
}


//
//  SourcePatch
//
//  This class allows the full topological specification of the neighborhood
//  of vertices and edges around a face that collectively define a rectangular
//  piece of surface corresponding to that face.  All components are declared
//  in terms of local indices and explicitly avoid any references/indices to
//  an external representation.  It is assembled by specifying the topology
//  of each corner (number of faces/edges, boundary, etc.) and finalization
//  determines a set of source vertices in a canonical orientation relative
//  to the face and any patch which may be derived from it.
//
//  Any/all corners can be arbitrarily irregular.  Information for each corner
//  is similar to what is provided in a Vtr::Level::VSpan but does not require
//  any Vtr dependent orientation (e.g. the leading edge) and also requires
//  identification of the incident face that corresponds to the patch (i.e.
//  the "patch face").
//
//  The set of local source vertices begins with the corner vertices of the
//  face corresponding to the patch.  Since the 1-rings of the corner vertices
//  overlap, a subset of the 1-rings is identified as the "local ring points"
//  for a corner, which are the points most associated with the corner.  While
//  one of these local ring points may overlap with an adjacent corner, the
//  local ring points for each corner are indexed successively for each corner.
//
//  The cumulative set of source points forming the 1-ring around the patch
//  face is indexed successively in a counter-clockwise orientation beginning
//  with the first edge-vertex of the first corner, e.g. for a patch with
//  three regular corners and an irregular boundary:
//
//     13          12       11         10
//        x--------x--------x--------x
//        |        |        |        |
//        |        |        |        |
//        |        |        |        |
//        |        |        |        |
//     14 x--------x--------x--------x 9
//        |        |3      2|        |
//        |        |        |        |
//        |        |        |        |
//        |        |0      1|        |
//      4 x--------x--------x--------x 8
//        |        |        |
//        |        |        |
//        |        |        |
//        |        |        |
//        x--------x--------x
//      5          6         7
//
//  The set of source points consists of the corner points {0,1,2,3} followed
//  by the four sets of points {4,5,6}, {7,8}, {9,10,11} and {12,13,14} --
//  each being the exterior subset of the points of the one-ring for the
//  corresponding corner point.
//
//  The 1-ring for each corner is made available for both assembling the points
//  of the resulting Gregory patch and for defining correspondence between the
//  original source of the vertices.  All 1-rings are oriented counter-clockwise
//  and begin with a vertex at the end of an edge.  The 1-rings for boundaries
//  begin/end with vertices at the ends of the leading/trailing edges.  Interior
//  1-rings are ordered such that the patch-face vertices occur as specified.
//

//
//  SourcePatch method to initialize other internal members once required
//  members of all corners have been explicitly initialized.  This deals with
//  all of the awkward ways in which rings of vertices around each corner
//  overlap in order to define the canonical ordering of vertices (and avoiding
//  have the same vertex twice).
//
//  Note:  Considering passing Corner[4] to a constructor so that this is all
//  dealt with in the constructor.
//
void
SourcePatch::Finalize() {

    //
    //  Determine the sizes of the rings and the total number of points
    //  involved.  In the process, identify which corners share ring points
    //  with their neighbors and accumulate maximal ring sizes and valence:
    //
    _maxValence = 0;
    _maxRingSize = 0;
    _numSourcePoints = 4;

    for (int cIndex = 0; cIndex < 4; ++cIndex) {
        //
        //  Need valence-2 information for neighbors as it affects sizing:
        //
        int cPrev = (cIndex + 3) & 0x3;
        int cNext = (cIndex + 1) & 0x3;

        bool prevIsVal2Interior = ((_corners[cPrev]._numFaces == 2) &&
                                   !_corners[cPrev]._boundary);
        bool thisIsVal2Interior = ((_corners[cIndex]._numFaces == 2) &&
                                   !_corners[cIndex]._boundary);
        bool nextIsVal2Interior = ((_corners[cNext]._numFaces == 2) &&
                                   !_corners[cNext]._boundary);

        _corners[cIndex]._val2Interior = thisIsVal2Interior;
        _corners[cIndex]._val2Adjacent = prevIsVal2Interior || nextIsVal2Interior;

        //
        //  General cases are >= 3-face interior and >= 2-face boundary:
        //
        Corner & corner = _corners[cIndex];

        if ((corner._numFaces + corner._boundary) > 2) {
            if (corner._boundary) {
                corner._sharesWithPrev = (corner._patchFace != (corner._numFaces - 1));
                corner._sharesWithNext = (corner._patchFace != 0);
            } else if (corner._dart) {
                corner._sharesWithPrev = !_corners[cPrev]._boundary;
                corner._sharesWithNext = !_corners[cNext]._boundary;
            } else {
                corner._sharesWithPrev = true;
                corner._sharesWithNext = true;
            }

            _ringSizes[cIndex]      = corner._numFaces * 2 + corner._boundary;
            _localRingSizes[cIndex] = _ringSizes[cIndex] - 3
                                    - corner._sharesWithPrev - corner._sharesWithNext
                                    - prevIsVal2Interior - nextIsVal2Interior;
        } else {
            corner._sharesWithPrev = false;
            corner._sharesWithNext = false;

            //  Single-face boundary and valence-2 interior:
            if (corner._numFaces == 1) {
                _ringSizes[cIndex]      = 3;
                _localRingSizes[cIndex] = 0;
            } else {
                _ringSizes[cIndex]      = 4;
                _localRingSizes[cIndex] = 1;
            }
        }
        _localRingOffsets[cIndex] = _numSourcePoints;

        _maxValence  = std::max(_maxValence, corner._numFaces + corner._boundary);
        _maxRingSize = std::max(_maxRingSize, _ringSizes[cIndex]);

        _numSourcePoints += _localRingSizes[cIndex];
    }
}

int
SourcePatch::GetCornerRingPoints(int corner, int ringPoints[]) const {

    int cNext = (corner + 1) & 0x3;
    int cOpp  = (corner + 2) & 0x3;
    int cPrev = (corner + 3) & 0x3;

    //
    //  Assemble the ring in a canonical ordering beginning with the points of
    //  the three other corners of the face followed by the local ring -- with
    //  any shared or compensating points (for valence-2 interior) preceding
    //  and following the points local to the ring.
    //
    int ringSize = 0;

    ringPoints[ringSize++] = cNext;
    ringPoints[ringSize++] = cOpp;
    ringPoints[ringSize++] = cPrev;

    if (_corners[cPrev]._val2Interior) {
        ringPoints[ringSize++] = cOpp;
    }
    if (_corners[corner]._sharesWithPrev) {
        ringPoints[ringSize++] = _localRingOffsets[cPrev] + _localRingSizes[cPrev] - 1;
    }

    for (int i = 0; i < _localRingSizes[corner]; ++i) {
        ringPoints[ringSize++] = _localRingOffsets[corner] + i;
    }

    if (_corners[corner]._sharesWithNext) {
        ringPoints[ringSize++] = _localRingOffsets[cNext];
    }
    if (_corners[cNext]._val2Interior) {
        ringPoints[ringSize++] = cOpp;
    }
    assert(ringSize == _ringSizes[corner]);

    //  The assembled ordering matches the desired ordering if the patch-face
    //  is first, so rotate the assembled ring if that's not the case:
    //
    if (_corners[corner]._patchFace) {
        int rotationOffset = ringSize - 2*_corners[corner]._patchFace;
        std::rotate(ringPoints, ringPoints + rotationOffset, ringPoints + ringSize);
    }
    return ringSize;
}

} // end namespace Far

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv
