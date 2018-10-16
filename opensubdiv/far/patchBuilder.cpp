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
    //  Inline methods to encapsulate fast and specific modulus operations.
    //  The mod-4 case is trivial.  For mod-N, since we are always doing
    //  the test (i + 1) % N, or (i - 1 + N) % N, the subtraction suffices.
    //
    inline int fastMod4(int x) {
        return x & 0x3;
    }
    inline int fastMod3(int x) {
        static int const mod3Array[] = { 0, 1, 2, 0, 1, 2 };
        return mod3Array[x];
    }
    inline int fastModN(int x, int N) {
        return (x < N) ? x : (x - N);
    }

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
        int iTrailingStart = fastModN(iLeadingStart + 1, nEdges);

        vSpan.clear();
        vSpan._numFaces = 1;

        int iLeading  = iLeadingStart;
        while (! isEdgeSingular(level, fvarLevel, vEdges[iLeading], eTagMask)) {
            ++vSpan._numFaces;
            iLeading = fastModN(iLeading + nEdges - 1, nEdges);
            if (iLeading == iTrailingStart) break;
        }

        int iTrailing = iTrailingStart;
        while (!isEdgeSingular(level, fvarLevel, vEdges[iTrailing], eTagMask)) {
            ++vSpan._numFaces;
            iTrailing = fastModN(iTrailing + 1, nEdges);
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


    //
    //  Gathering the one-ring of vertices from triangles surrounding a vertex:
    //      - the neighborhood of the vertex is assumed to be tri-regular (manifold)
    //
    //  Ordering of resulting vertices:
    //      The surrounding one-ring follows the ordering of the incident faces.  For each
    //  incident tri, the vertex opposite its leading edge is added.  If the vertex is on a
    //  boundary, a second vertex on the boundary edge will be contributed from the last face.
    //
    int
    gatherTriRegularRingAroundVertex(Level const& level,
        Index vIndex, int ringPoints[], int fvarChannel) {

        ConstIndexArray vEdges = level.getVertexEdges(vIndex);

        ConstIndexArray vFaces = level.getVertexFaces(vIndex);
        ConstLocalIndexArray vInFaces = level.getVertexFaceLocalIndices(vIndex);

        bool isBoundary = (vEdges.size() > vFaces.size());

        int ringIndex = 0;
        for (int i = 0; i < vFaces.size(); ++i) {
            //
            //  For each tri, we want the the vertex at the end of the leading edge:
            //
            ConstIndexArray fPoints = (fvarChannel < 0)
                                    ? level.getFaceVertices(vFaces[i])
                                    : level.getFaceFVarValues(vFaces[i], fvarChannel);

            int vInThisFace = vInFaces[i];

            ringPoints[ringIndex++] = fPoints[fastMod3(vInThisFace + 1)];

            if (isBoundary && (i == (vFaces.size() - 1))) {
                ringPoints[ringIndex++] = fPoints[fastMod3(vInThisFace + 2)];
            }
        }
        return ringIndex;
    }

    int
    gatherTriRegularPartialRingAroundVertex(Level const& level,
        Index vIndex, Level::VSpan const & span, int ringPoints[], int fvarChannel) {

        assert(! level.isVertexNonManifold(vIndex));

        ConstIndexArray      vFaces   = level.getVertexFaces(vIndex);
        ConstLocalIndexArray vInFaces = level.getVertexFaceLocalIndices(vIndex);

        int nFaces    = span._numFaces;
        int startFace = span._startFace;

        int ringIndex = 0;
        for (int i = 0; i < nFaces; ++i) {
            //
            //  For each tri, we want the the vertex at the end of the leading edge:
            //
            int fIncident = fastModN(startFace + i, vFaces.size());

            ConstIndexArray fPoints = (fvarChannel < 0)
                                    ? level.getFaceVertices(vFaces[fIncident])
                                    : level.getFaceFVarValues(vFaces[fIncident], fvarChannel);

            int vInThisFace = vInFaces[fIncident];

            ringPoints[ringIndex++] = fPoints[fastMod3(vInThisFace + 1)];

            if ((i == nFaces - 1) && !span._periodic) {
                ringPoints[ringIndex++] = fPoints[fastMod3(vInThisFace + 2)];
            }
        }
        return ringIndex;
    }

    //
    //  Functions to encode/decode the 5-bit boundary mask for a triangular patch
    //  from the two 3-bit boundary vertex and bounday edge masks.  When referring
    //  to a "boundary vertex" in the encoded bits, we are referring to a vertex on
    //  a boundary while its incident edges of the triangle are not boundaries --
    //  topologically distinct from a vertex at the end of a boundary edge.
    //
    //  The 5-bit encoding is as follows:
    //
    //      - the upper 2 bits indicate how to interpret the lower 3 bits:
    //          0 - as boundary edges only (all boundary vertices are implicit)
    //          1 - as "boundary vertices" only (no boundary edges)
    //          2 - a single boundary edge with opposite "boundary vertex"
    //
    //      - the lower 3 bits are set according to boundary features present
    //
    //  There are a total of 18 possible boundary configurations:
    //
    //      - no boundaries at all (1 case)
    //      - one boundary edge (3 cases)
    //      - two boundary edges (3 cases)
    //      - three boundary edges (1 case)
    //      - one boundary vertex (3 cases)
    //      - two boundary vertices (3 cases)
    //      - three boundarey vertices (1 case)
    //      - one boundary edge with opposite boundary vertex (3 cases)
    //
    inline int unpackTriBoundaryMaskLower(int mask) { return mask & 0x7; }
    inline int unpackTriBoundaryMaskUpper(int mask) { return (mask >> 3) & 0x3; }

    inline int packTriBoundaryMask(int upper, int lower) { return (upper << 3) | lower; }

    int
    encodeTriBoundaryMask(int eBits, int vBits)
    {
        int upperBits = 0;
        int lowerBits = eBits;

        if (vBits) {
            if (eBits == 0) {
                upperBits = 1;
                lowerBits = vBits;
            } else if ((vBits == 7) && ((eBits == 1) || (eBits == 2) || (eBits == 4))) {
                upperBits = 2;
                lowerBits = eBits;
            }
        }
        return packTriBoundaryMask(upperBits, lowerBits);
    }
    void
    decodeTriBoundaryMask(int mask, int & eBits, int & vBits)
    {
        static int const eBitsToVBits[] = { 0, 3, 6, 7, 5, 7, 7, 7 };

        int lowerBits = unpackTriBoundaryMaskLower(mask);
        int upperBits = unpackTriBoundaryMaskUpper(mask);

        switch (upperBits) {
        case 0:
            eBits = lowerBits;
            vBits = eBitsToVBits[eBits];
            break;
        case 1:
            eBits = 0;
            vBits = lowerBits;
            break;
        case 2:
            eBits = lowerBits;
            vBits = 0x7;
            break;
        }
    }

    inline Index
    getNextFaceInVertFaces(Level const & level, int thisFaceInVFaces,
                           ConstIndexArray const & vFaces,
                           ConstLocalIndexArray const & vInFaces,
                           bool manifold, int & vInNextFace) {

        Index nextFace;
        if (manifold) {
            int nextFaceInVFaces = fastModN(thisFaceInVFaces + 1, vFaces.size());

            nextFace    = vFaces[nextFaceInVFaces];
            vInNextFace = vInFaces[nextFaceInVFaces];
        } else {
            Index thisFace    = vFaces[thisFaceInVFaces];
            int   vInThisFace = vInFaces[thisFaceInVFaces];

            ConstIndexArray fEdges = level.getFaceEdges(thisFace);

            Index nextEdge = fEdges[fastModN((vInThisFace + fEdges.size() - 1), fEdges.size())];

            ConstIndexArray eFaces = level.getEdgeFaces(nextEdge);
            assert(eFaces.size() == 2);

            nextFace = (eFaces[0] == thisFace) ? eFaces[1] : eFaces[0];

            int edgeInNextFace = level.getFaceEdges(nextFace).FindIndex(nextEdge);

            vInNextFace = edgeInNextFace;
        }
        return nextFace;
    }
    inline Index
    getPrevFaceInVertFaces(Level const & level, int thisFaceInVFaces,
                           ConstIndexArray const & vFaces,
                           ConstLocalIndexArray const & vInFaces,
                           bool manifold, int & vInPrevFace) {

        Index prevFace;
        if (manifold) {
            int prevFaceInVFaces = (thisFaceInVFaces ? thisFaceInVFaces : vFaces.size()) - 1;

            prevFace    = vFaces[prevFaceInVFaces];
            vInPrevFace = vInFaces[prevFaceInVFaces];
        } else {
            Index thisFace    = vFaces[thisFaceInVFaces];
            int   vInThisFace = vInFaces[thisFaceInVFaces];

            ConstIndexArray fEdges = level.getFaceEdges(thisFace);

            Index prevEdge = fEdges[vInThisFace];

            ConstIndexArray eFaces = level.getEdgeFaces(prevEdge);
            assert(eFaces.size() == 2);

            prevFace = (eFaces[0] == thisFace) ? eFaces[1] : eFaces[0];

            int edgeInPrevFace = level.getFaceEdges(prevFace).FindIndex(prevEdge);

            vInPrevFace = fastModN(edgeInPrevFace + 1, fEdges.size());
        }
        return prevFace;
    }

    inline ConstIndexArray
    getFacePoints(Level const& level, Index faceIndex, int fvarChannel)
    {
        return (fvarChannel < 0) ? level.getFaceVertices(faceIndex)
                                 : level.getFaceFVarValues(faceIndex, fvarChannel);
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
    //      - preserving the historical test for 4-sided faces
    if ((_schemeRegFaceSize == 4) || (levelIndex == 0)) {
        if (level.getFaceCompositeVTag(fVerts)._incomplete) return false;
    } else {
        if (_refiner.getRefinement(levelIndex - 1).getChildFaceTag(faceIndex)._incomplete) return false;
    }
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
            int regBoundaryFaces = 2 + (_schemeRegFaceSize == 3);

            Level::VTag vTags[4];
            level.getFaceVTags(faceIndex, vTags, fvarChannel);

            Level::VSpan vSpan;
            Level::ETag eMask = getSingularEdgeMask(true);

            isRegular = true;
            for (int i = 0; i < _schemeRegFaceSize; ++i) {
                if (vTags[i]._infIrregular) {
                    identifyManifoldCornerSpan(
                        level, faceIndex, i, eMask, vSpan, fvarChannel);

                    isRegular = (vSpan._numFaces ==
                                    (vTags[i]._infSharpCrease ? regBoundaryFaces : 1));
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
            for (int i = 0; i < _schemeRegFaceSize; ++i) {
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
    bool isQuad = (fVerts.size() == 4);

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
            (vTags[2]._infSharpEdges && (vTags[2]._rule == Sdc::Crease::RULE_CREASE));
        if (isQuad) {
            boundaryCount +=
                (vTags[3]._infSharpEdges && (vTags[3]._rule == Sdc::Crease::RULE_CREASE));
        }
    } else {
        boundaryCount =
            (vTags[0]._boundary && (vTags[0]._rule == Sdc::Crease::RULE_CREASE)) +
            (vTags[1]._boundary && (vTags[1]._rule == Sdc::Crease::RULE_CREASE)) +
            (vTags[2]._boundary && (vTags[2]._rule == Sdc::Crease::RULE_CREASE));
        if (isQuad) {
            boundaryCount +=
                (vTags[3]._boundary && (vTags[3]._rule == Sdc::Crease::RULE_CREASE));
        }
    }
    int xordinaryCount = vTags[0]._xordinary + vTags[1]._xordinary + vTags[2]._xordinary;
    if (isQuad) {
        xordinaryCount += vTags[3]._xordinary;
    }
    
    if ((boundaryCount == 3) && (xordinaryCount == 1)) {
        //  This must be an isolated xordinary corner above level 1, otherwise
        //  we still need to assure the xordinary vertex is opposite a smooth
        //  interior vertex:
        //
        if (!isQuad || (levelIndex > 1)) return true;
    
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
    Level::ETag eTags[4];

    level.getFaceVTags(faceIndex, vTags, fvarChannel);
    level.getFaceETags(faceIndex, eTags, fvarChannel);

    Level::VTag fTag = Level::VTag::BitwiseOr(vTags, _schemeRegFaceSize);

    //
    //  For quads it is sufficient to inspect only edge tags.  For tris,
    //  vertices may lie on boundaries with no incident boundary edges in
    //  the tri itself.
    //
    bool isQuad = ( _schemeRegFaceSize == 4);

    int vBits = 0;
    int eBits = 0;
    if (fTag._infSharpEdges) {
        if (!_options.approxInfSharpWithSmooth) {
            eBits = (eTags[0]._infSharp << 0) |
                    (eTags[1]._infSharp << 1) |
                    (eTags[2]._infSharp << 2);
            if (isQuad) eBits |= (eTags[3]._infSharp << 3);
        } else {
            eBits = (eTags[0]._boundary << 0) |
                    (eTags[1]._boundary << 1) |
                    (eTags[2]._boundary << 2);
            if (isQuad) eBits |= (eTags[3]._boundary << 3);
        }

        if (!isQuad) {
            if (!_options.approxInfSharpWithSmooth) {
                vBits = (vTags[0]._infSharpEdges << 0) |
                        (vTags[1]._infSharpEdges << 1) |
                        (vTags[2]._infSharpEdges << 2);
            } else if (fTag._boundary) {
                vBits = (vTags[0]._boundary << 0) |
                        (vTags[1]._boundary << 1) |
                        (vTags[2]._boundary << 2);
            }
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
        eBits |= (eTags[0]._nonManifold << 0) |
                 (eTags[1]._nonManifold << 1) |
                 (eTags[2]._nonManifold << 2);

        if (isQuad) {
            eBits |= (eTags[3]._nonManifold << 3);

            if (vTags[0]._nonManifold && vTags[0]._infSharp) eBits |= 9;
            if (vTags[1]._nonManifold && vTags[1]._infSharp) eBits |= 3;
            if (vTags[2]._nonManifold && vTags[2]._infSharp) eBits |= 6;
            if (vTags[3]._nonManifold && vTags[3]._infSharp) eBits |= 12;

            //  This handles faces that touch a non-manifold edge without
            //  its edges being incident that non-manifold edge -- these
            //  are essentially irregular boundary vertices and will be
            //  dealt with differently in future (not considered regular)
            //
            if (eBits == 0) {
                if (vTags[0]._nonManifold) eBits |= 9;
                if (vTags[1]._nonManifold) eBits |= 3;
                if (vTags[2]._nonManifold) eBits |= 6;
                if (vTags[3]._nonManifold) eBits |= 12;
            }
        } else {
            if (vTags[0]._nonManifold && vTags[0]._infSharp) eBits |= 5;
            if (vTags[1]._nonManifold && vTags[1]._infSharp) eBits |= 3;
            if (vTags[2]._nonManifold && vTags[2]._infSharp) eBits |= 6;
        }
    }

    int boundaryMask = 0;
    if (eBits || vBits) {
        if (isQuad) {
            boundaryMask = eBits;
        } else {
            boundaryMask = encodeTriBoundaryMask(eBits, vBits);
        }
    }
    return boundaryMask;
}

void
PatchBuilder::GetIrregularPatchCornerSpans(int levelIndex, Index faceIndex,
        Level::VSpan cornerSpans[4], int fvarChannel) const {

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
    bool interiorPatch = (regBoundaryMask == 0);

    static int const patchPointsPerCorner[4][4] = { {  5,  4,  0,  1 },
                                                    {  6,  2,  3,  7 },
                                                    { 10, 11, 15, 14 },
                                                    {  9, 13, 12,  8 } };

    int eMask = regBoundaryMask;

    Level const & level = _refiner.getLevel(levelIndex);

    ConstIndexArray fVerts  = level.getFaceVertices(faceIndex);
    ConstIndexArray fPoints = getFacePoints(level, faceIndex, fvarChannel);

    Index boundaryPoint = INDEX_INVALID;
    if (!interiorPatch && _options.fillMissingBoundaryPoints) {
        boundaryPoint = fPoints[0];
    }

    for (int i = 0; i < 4; ++i) {
        Index v = fVerts[i];

        const int* cornerPointIndices = patchPointsPerCorner[i];

        ConstIndexArray      vFaces   = level.getVertexFaces(v);
        ConstLocalIndexArray vInFaces = level.getVertexFaceLocalIndices(v);

        Index f = faceIndex;
        int fInVFaces = vFaces.FindIndex(f);
        assert(vInFaces[fInVFaces] == i);  // Beware non-manifold vert in face twice...

        bool interiorCorner = interiorPatch || (((eMask & (1 << i)) |
                                                 (eMask & (1 << fastMod4(i+3)))) == 0);
        if (interiorCorner) {
            int fOppInVFaces = fastMod4(fInVFaces + 2);

            Index fOpp = vFaces[fOppInVFaces];
            int vInFOpp = vInFaces[fOppInVFaces];
            ConstIndexArray fOppPoints = getFacePoints(level, fOpp, fvarChannel);

            patchPoints[cornerPointIndices[1]] = fOppPoints[fastMod4(vInFOpp + 1)];
            patchPoints[cornerPointIndices[2]] = fOppPoints[fastMod4(vInFOpp + 2)];
            patchPoints[cornerPointIndices[3]] = fOppPoints[fastMod4(vInFOpp + 3)];
        } else if ((eMask & (1 << i)) && (eMask & (1 << fastMod4(i+3)))) {
            //  Two indicent boundary edges -- no incident faces
            patchPoints[cornerPointIndices[1]] = boundaryPoint;
            patchPoints[cornerPointIndices[2]] = boundaryPoint;
            patchPoints[cornerPointIndices[3]] = boundaryPoint;
        } else if (eMask & (1 << i)) {
            //  Leading/outgoing boundary edge -- need next face:
            int vInFNext;
            Index fNext = getNextFaceInVertFaces(level, fInVFaces, vFaces, vInFaces,
                                   !level.getVertexTag(v)._nonManifold, vInFNext);

            ConstIndexArray fNextPoints = getFacePoints(level, fNext, fvarChannel);

            patchPoints[cornerPointIndices[1]] = fNextPoints[fastMod4(vInFNext + 3)];
            patchPoints[cornerPointIndices[2]] = boundaryPoint;
            patchPoints[cornerPointIndices[3]] = boundaryPoint;
        } else {
            //  Trailing/incoming boundary edge -- need previous face:
            int vInFPrev;
            Index fPrev = getPrevFaceInVertFaces(level, fInVFaces, vFaces, vInFaces,
                                   !level.getVertexTag(v)._nonManifold, vInFPrev);

            ConstIndexArray fPrevPoints = getFacePoints(level, fPrev, fvarChannel);

            patchPoints[cornerPointIndices[1]] = boundaryPoint;
            patchPoints[cornerPointIndices[2]] = boundaryPoint;
            patchPoints[cornerPointIndices[3]] = fPrevPoints[fastMod4(vInFPrev + 1)];
        }
        patchPoints[cornerPointIndices[0]] = fPoints[i];
    }
    return 16;
}

int
PatchBuilder::getTriRegularPatchPoints(int levelIndex, Index faceIndex,
        int regBoundaryMask, Index patchPoints[],
        int fvarChannel) const {

    if (regBoundaryMask < 0) {
        regBoundaryMask = GetRegularPatchBoundaryMask(levelIndex, faceIndex);
    }
    bool interiorPatch = (regBoundaryMask == 0);

    static int const patchPointsPerCorner[3][4] = { { 4, 7, 3, 0 },
                                                    { 5, 1, 2, 6 },
                                                    { 8, 9, 11, 10 } };

    int vMask = 0;
    int eMask = 0;
    if (!interiorPatch) {
        decodeTriBoundaryMask(regBoundaryMask, eMask, vMask);
    }

    Level const & level = _refiner.getLevel(levelIndex);

    ConstIndexArray fVerts  = level.getFaceVertices(faceIndex);
    ConstIndexArray fPoints = getFacePoints(level, faceIndex, fvarChannel);

    Index boundaryPoint = INDEX_INVALID;
    if (!interiorPatch && _options.fillMissingBoundaryPoints) {
        boundaryPoint = fPoints[0];
    }

    for (int i = 0; i < 3; ++i) {
        Index v = fVerts[i];

        const int* cornerPointIndices = patchPointsPerCorner[i];

        ConstIndexArray      vFaces   = level.getVertexFaces(v);
        ConstLocalIndexArray vInFaces = level.getVertexFaceLocalIndices(v);

        Index f = faceIndex;
        int fInVFaces = vFaces.FindIndex(f);

        bool interiorCorner = interiorPatch || ((vMask & (1 << i)) == 0);
        if (interiorCorner) {
            int f2InVFaces = fastModN(fInVFaces + 2, 6);
            int f3InVFaces = fastModN(fInVFaces + 3, 6);

            Index f2 = vFaces[f2InVFaces];
            Index f3 = vFaces[f3InVFaces];

            int vInf2 = vInFaces[f2InVFaces];
            int vInf3 = vInFaces[f3InVFaces];

            ConstIndexArray f2Points = getFacePoints(level, f2, fvarChannel);
            ConstIndexArray f3Points = getFacePoints(level, f3, fvarChannel);

            patchPoints[cornerPointIndices[1]] = f2Points[fastMod3(vInf2 + 1)];
            patchPoints[cornerPointIndices[2]] = f3Points[fastMod3(vInf3 + 1)];
            patchPoints[cornerPointIndices[3]] = f3Points[fastMod3(vInf3 + 2)];
        } else if ((eMask & (1 << i)) && (eMask & (1 << fastMod3(i+2)))) {
            //  Test for two indicent boundary edges -- no incident faces

            patchPoints[cornerPointIndices[1]] = boundaryPoint;
            patchPoints[cornerPointIndices[2]] = boundaryPoint;
            patchPoints[cornerPointIndices[3]] = boundaryPoint;
        } else if (eMask & (1 << i)) {
            //  Test for leading/outgoing boundary edge:
            int f2InVFaces = fastModN(fInVFaces + 2, vFaces.size());

            Index f2 = vFaces[f2InVFaces];
            int vInf2 = vInFaces[f2InVFaces];
            ConstIndexArray f2Points = getFacePoints(level, f2, fvarChannel);

            patchPoints[cornerPointIndices[1]] = f2Points[fastMod3(vInf2 + 1)];
            patchPoints[cornerPointIndices[2]] = f2Points[fastMod3(vInf2 + 2)];
            patchPoints[cornerPointIndices[3]] = boundaryPoint;
        } else if (eMask & (1 << fastMod3(i+2))) {
            //  Test for trailing/incoming boundary edge:
            int f0InVFaces = fastModN(fInVFaces + vFaces.size() - 2, vFaces.size());

            Index f0 = vFaces[f0InVFaces];
            int vInf0 = vInFaces[f0InVFaces];
            ConstIndexArray f0Points = getFacePoints(level, f0, fvarChannel);

            patchPoints[cornerPointIndices[1]] = boundaryPoint;
            patchPoints[cornerPointIndices[2]] = boundaryPoint;
            patchPoints[cornerPointIndices[3]] = f0Points[fastMod3(vInf0 + 1)];
        } else {
            //  Test for boundary vertex:
            int f2InVFaces = fastModN(fInVFaces + 1, vFaces.size());

            Index f2 = vFaces[f2InVFaces];
            int vInf2 = vInFaces[f2InVFaces];
            ConstIndexArray f2Points = getFacePoints(level, f2, fvarChannel);

            patchPoints[cornerPointIndices[1]] = f2Points[fastMod3(vInf2 + 2)];
            patchPoints[cornerPointIndices[2]] = boundaryPoint;
            patchPoints[cornerPointIndices[3]] = boundaryPoint;
        }
        patchPoints[cornerPointIndices[0]] = fPoints[i];
    }
    return 12;
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
        return getTriRegularPatchPoints(
            levelIndex, faceIndex, regBoundaryMask, patchPoints, fvarChannel);
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

    for (int corner = 0; corner < fVerts.size(); ++corner) {
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
            int vFaceIndex = fastModN(firstFace + patchFace, vFaces.size());
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
    sourcePatch.Finalize(fVerts.size());

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
        Index patchVerts[], int fvarChannel) const {

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
    for (int corner = 0; corner < sourcePatch._numCorners; ++corner) {
        Index cornerVertex = faceVerts[corner];
        
        //  Gather the ring of source points from the Vtr level:
        int sourceRingSize = 0;
        if (!cornerSpans[corner].isAssigned()) {
            if (sourcePatch._numCorners == 4) {
                sourceRingSize = level.gatherQuadRegularRingAroundVertex(
                    cornerVertex, sourceRingVertices,
                    fvarChannel);
            } else {
                sourceRingSize = gatherTriRegularRingAroundVertex(level,
                    cornerVertex, sourceRingVertices,
                    fvarChannel);
            }
        } else {
            if (sourcePatch._numCorners == 4) {
                sourceRingSize = level.gatherQuadRegularPartialRingAroundVertex(
                    cornerVertex, cornerSpans[corner], sourceRingVertices,
                    fvarChannel);
            } else {
                sourceRingSize = gatherTriRegularPartialRingAroundVertex(level,
                    cornerVertex, cornerSpans[corner], sourceRingVertices,
                    fvarChannel);
            }
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

    SourcePatch sourcePatch;
    assembleIrregularSourcePatch(
            levelIndex, faceIndex, cornerSpans, sourcePatch);

    return gatherIrregularSourcePoints(levelIndex, faceIndex,
        cornerSpans, sourcePatch, sourcePoints, fvarChannel);
}

//
//  Template conversion methods for the matrix type -- explicit instantiation
//  for float and double is required and follows the definition:
//
template <typename REAL>
int
PatchBuilder::GetIrregularPatchConversionMatrix(
        int levelIndex, Index faceIndex,
        Level::VSpan const cornerSpans[],
        SparseMatrix<REAL> & conversionMatrix) const {

    SourcePatch sourcePatch;
    assembleIrregularSourcePatch(
            levelIndex, faceIndex, cornerSpans, sourcePatch);

    return convertToPatchType(
        sourcePatch, GetIrregularPatchType(), conversionMatrix);
}
template int PatchBuilder::GetIrregularPatchConversionMatrix<float>(
        int levelIndex, Index faceIndex, Level::VSpan const cornerSpans[],
        SparseMatrix<float> & conversionMatrix) const;
template int PatchBuilder::GetIrregularPatchConversionMatrix<double>(
        int levelIndex, Index faceIndex, Level::VSpan const cornerSpans[],
        SparseMatrix<double> & conversionMatrix) const;


bool
PatchBuilder::IsRegularSingleCreasePatch(int levelIndex, Index faceIndex,
        SingleCreaseInfo & creaseInfo) const {

    if (_schemeRegFaceSize != 4) return false;

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

    bool irregBase =
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

        irregBase =
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
        } else if (!irregBase) {
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
    if (irregBase) {
        ptexIndex += childIndexInParent;
    }

    //  Compute/identify the transition mask if requested, otherwise leave it 0:
    int transitionMask = 0;
    if (computeTransitionMask && (levelIndex < _refiner.GetMaxLevel())) {
        transitionMask = _refiner.getRefinement(levelIndex).
                            getParentFaceSparseTag(faceIndex)._transitional;
    }

    PatchParam param;
    param.Set(ptexIndex, (short)u, (short)v, (unsigned short) depth, irregBase,
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
SourcePatch::Finalize(int size) {

    //
    //  Determine the sizes of the rings and the total number of points
    //  involved.  In the process, identify which corners share ring points
    //  with their neighbors and accumulate maximal ring sizes and valence:
    //
    bool isQuad = (size == 4);

    _numCorners = size;

    _maxValence = 0;
    _maxRingSize = 0;
    _numSourcePoints = _numCorners;

    for (int cIndex = 0; cIndex < _numCorners; ++cIndex) {
        //
        //  Need valence-2 information for neighbors as it affects sizing:
        //
        int cPrev = fastModN(cIndex + 2 + isQuad, _numCorners);
        int cNext = fastModN(cIndex + 1,          _numCorners);

        bool prevIsVal2Interior = ((_corners[cPrev]._numFaces == 2) &&
                                   !_corners[cPrev]._boundary);
        bool thisIsVal2Interior = ((_corners[cIndex]._numFaces == 2) &&
                                   !_corners[cIndex]._boundary);
        bool nextIsVal2Interior = ((_corners[cNext]._numFaces == 2) &&
                                   !_corners[cNext]._boundary);

        Corner & corner = _corners[cIndex];

        corner._val2Interior = thisIsVal2Interior;
        corner._val2Adjacent = prevIsVal2Interior || nextIsVal2Interior;

        //
        //  General cases are >= 3-face interior and >= 2-face boundary:
        //
        if ((corner._numFaces + corner._boundary) > 2) {
            //
            //  Quads generally share with both prev and next, but triangles
            //  never share with prev because of the necessary asymmetry of
            //  the local ring points.
            //
            if (corner._boundary) {
                corner._sharesWithPrev = isQuad && (corner._patchFace != (corner._numFaces - 1));
                corner._sharesWithNext = (corner._patchFace != 0);
            } else if (corner._dart) {
                corner._sharesWithPrev = isQuad && !_corners[cPrev]._boundary;
                corner._sharesWithNext = !_corners[cNext]._boundary;
            } else {
                corner._sharesWithPrev = isQuad;
                corner._sharesWithNext = true;
            }

            _ringSizes[cIndex]      = corner._numFaces * (1 + isQuad) + corner._boundary;
            _localRingSizes[cIndex] = _ringSizes[cIndex] - (_numCorners - 1)
                                    - corner._sharesWithPrev - corner._sharesWithNext;

            if (corner._val2Adjacent) {
                _localRingSizes[cIndex] -= prevIsVal2Interior;
                _localRingSizes[cIndex] -= (nextIsVal2Interior && isQuad);
            }
        } else {
            corner._sharesWithPrev = false;
            corner._sharesWithNext = false;

            //  Single-face boundary/corner and valence-2 interior:
            if (corner._numFaces == 1) {
                _ringSizes[cIndex]      = _numCorners - 1;
                _localRingSizes[cIndex] = 0;
            } else {
                _ringSizes[cIndex]      = 2 * (1 + isQuad);
                _localRingSizes[cIndex] = isQuad;
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

    bool isQuad = (_numCorners == 4);

    int cNext = fastModN(corner + 1,          _numCorners);
    int cOpp  = fastModN(corner + 1 + isQuad, _numCorners);
    int cPrev = fastModN(corner + 2 + isQuad, _numCorners);

    //
    //  Assemble the ring in a canonical ordering beginning with the points of
    //  the 2 or 3 other corners of the face followed by the local ring -- with
    //  any shared or compensating points (for valence-2 interior) preceding
    //  and following the points local to the ring.
    //
    int ringSize = 0;

    //  The adjacent corner points:
    ringPoints[ringSize++] = cNext;
    if (isQuad) {
        ringPoints[ringSize++] = cOpp;
    }
    ringPoints[ringSize++] = cPrev;

    //  Shared points preceding the local ring points:
    if (_corners[cPrev]._val2Interior) {
        ringPoints[ringSize++] = isQuad ? cOpp : cNext;
    }
    if (_corners[corner]._sharesWithPrev) {
        ringPoints[ringSize++] = _localRingOffsets[cPrev] + _localRingSizes[cPrev] - 1;
    }

    //  The local ring points:
    for (int i = 0; i < _localRingSizes[corner]; ++i) {
        ringPoints[ringSize++] = _localRingOffsets[corner] + i;
    }

    //  Shared points following the local ring points:
    if (isQuad) {
        if (_corners[corner]._sharesWithNext) {
            ringPoints[ringSize++] = _localRingOffsets[cNext];
        }
        if (_corners[cNext]._val2Interior) {
            ringPoints[ringSize++] = cOpp;
        }
    } else {
        if (_corners[corner]._sharesWithNext) {
            ringPoints[ringSize++] = _corners[cNext]._val2Interior
                                   ? cPrev : _localRingOffsets[cNext];
        }
    }
    assert(ringSize == _ringSizes[corner]);

    //  The assembled ordering matches the desired ordering if the patch-face
    //  is first, so rotate the assembled ring if that's not the case:
    //
    if (_corners[corner]._patchFace) {
        int rotationOffset = ringSize - (1 + isQuad) * _corners[corner]._patchFace;
        std::rotate(ringPoints, ringPoints + rotationOffset, ringPoints + ringSize);
    }
    return ringSize;
}

} // end namespace Far

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv
