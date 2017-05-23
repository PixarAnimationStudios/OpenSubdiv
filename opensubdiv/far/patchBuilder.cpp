//
//   Copyright 2016 NVIDIA CORPORATION
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

#include "../far/error.h"
#include "../far/patchBuilder.h"
#include "../far/topologyRefiner.h"
#include "../vtr/fvarLevel.h"

#include <algorithm>
#include <cassert>
#include <cstring>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {


namespace Far {

using Vtr::internal::Refinement;
using Vtr::internal::Level;
using Vtr::internal::FVarLevel;

namespace internal {

//
//  Local helper functions for identifying the subset of a ring around a
//  corner that contributes to a patch -- parameterized by a mask that
//  indicates what kind of edge is to delimit the span.
//
//  Note that the two main methods need both face-verts and face-edges for
//  each corner, and that we don't really need the face-index once we have
//  them -- consider passing the fVerts and fEdges as arguments as they
//  will otherwise be retrieved repeatedly for each corner.
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
isEdgeSingular(Level const & level,
    FVarLevel const * fvarLevel, Index eIndex, Level::ETag eTagMask) {

    Level::ETag eTag = level.getEdgeTag(eIndex);

    if (fvarLevel) {
        eTag = fvarLevel->getEdgeTag(eIndex).combineWithLevelETag(eTag);
    }

    Level::ETag::ETagSize * iTag  = reinterpret_cast<Level::ETag::ETagSize*>(&eTag);
    Level::ETag::ETagSize * iMask = reinterpret_cast<Level::ETag::ETagSize*>(&eTagMask);
    return (*iTag & *iMask) > 0;
}

void
identifyManifoldCornerSpan(Level const & level, Index fIndex, int fCorner,
    Level::ETag eTagMask, Level::VSpan & vSpan, int fvc = -1) {

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
    while (! isEdgeSingular(level, fvarLevel, vEdges[iTrailing], eTagMask)) {
        ++vSpan._numFaces;
        iTrailing = (iTrailing + 1) % nEdges;
        if (iTrailing == iLeadingStart) break;
    }
    vSpan._startFace = (LocalIndex) iLeading;
}

void
identifyNonManifoldCornerSpan(Level const & level, Index fIndex, int fCorner,
    Level::ETag /* eTagMask */, Level::VSpan & vSpan, int /* fvc */ = -1) {

    //  For now, non-manifold patches revert to regular patches -- just identify
    //  the single face now for a sharp corner patch.
    //
    //  Remember that the face may be incident the vertex multiple times when
    //  non-manifold, so make sure the local index of the corner vertex in the
    //  face identified additionally matches the corner.
    //
    //FVarLevel const * fvarLevel = (fvc < 0) ? 0 : &level.getFVarChannel(fvc);

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

PatchBuilder::PatchBuilder(TopologyRefiner const & refiner,
    int numFVarChannels, int const * fvarChannelIndices,
        bool useInfSharpPatch, bool generateLegacySharpCornerPatches) :
            _refiner(refiner),
            _useInfSharpPatch(useInfSharpPatch),
            _generateLegacySharpCornerPatches(_generateLegacySharpCornerPatches) {

    // If client-code does not select specific channels, default to all
    // the channels in the refiner.
    if (numFVarChannels==-1) {
        _fvarChannelIndices.resize(refiner.GetNumFVarChannels());
        for (int fvc=0;fvc<(int)_fvarChannelIndices.size(); ++fvc) {
            _fvarChannelIndices[fvc] = fvc; // std::iota
        }
    } else {
        _fvarChannelIndices.assign(
            fvarChannelIndices, fvarChannelIndices+numFVarChannels);
    }
}

Vtr::internal::Level const &
PatchBuilder::GetVtrLevel(int levelIndex) const {
    return _refiner.getLevel(levelIndex);
}

int
PatchBuilder::GetRefinerFVarChannel(int fvcFactory) const {
    return (fvcFactory >= 0) ? _fvarChannelIndices[fvcFactory] : -1;
}


bool
PatchBuilder::DoesFaceVaryingPatchMatch(
    int levelIndex, Index faceIndex, int fvcFactory) const {

    return _refiner.getLevel(levelIndex).doesFaceFVarTopologyMatch(
        faceIndex, GetRefinerFVarChannel(fvcFactory));
}

int
PatchBuilder::GetDistinctRefinerFVarChannel(
    int levelIndex, Index faceIndex, int fvcFactory) const {

    return ((fvcFactory >= 0) &&
            !DoesFaceVaryingPatchMatch(levelIndex, faceIndex, fvcFactory)) ?
        GetRefinerFVarChannel(fvcFactory) : -1;
}

int
PatchBuilder::GetTransitionMask(int levelIndex, Index faceIndex) const {
    return (levelIndex == _refiner.GetMaxLevel()) ? 0 :
        _refiner.getRefinement(levelIndex).
                   getParentFaceSparseTag(faceIndex)._transitional;
}


bool
PatchBuilder::IsPatchEligible(int levelIndex, Index faceIndex) const {

    //
    //  Patches that are not eligible correpond to the following faces:
    //      - holes
    //      - those in intermediate levels that are further refined (not leaves)
    //      - those without complete neighborhoods that supporting other patches
    //
    Level const & level = _refiner.getLevel(levelIndex);

    if (level.isFaceHole(faceIndex)) {
        return false;
    }

    if (levelIndex < _refiner.GetMaxLevel()) {
        if (_refiner.getRefinement(levelIndex).getParentFaceSparseTag(faceIndex)._selected) {
            return false;
        }
    }

    //
    //  Faces that exist solely to support faces intended for patches will not have
    //  their full neighborhood available and so are considered "incomplete":
    //
    Vtr::ConstIndexArray fVerts = level.getFaceVertices(faceIndex);
    assert(fVerts.size() == 4);

    if (level.getFaceCompositeVTag(fVerts)._incomplete) {
        return false;
    }
    return true;
}

bool
PatchBuilder::IsPatchSmoothCorner(
    int levelIndex, Index faceIndex, int fvcFactory) const {

    Level const & level = _refiner.getLevel(levelIndex);

    //  Ignore the face-varying channel if the topology for the face is not distinct
    int fvcRefiner = GetDistinctRefinerFVarChannel(levelIndex, faceIndex, fvcFactory);

    Vtr::ConstIndexArray fVerts = level.getFaceVertices(faceIndex);
    if (fVerts.size() != 4) return false;

    Level::VTag vTags[4];
    level.getFaceVTags(faceIndex, vTags, fvcRefiner);

    //
    //  Test the subdivision rules for the corners, rather than just the boundary/interior
    //  tags, to ensure that inf-sharp vertices or edges are properly accounted for (and
    //  the cases appropriately excluded) if inf-sharp patches are enabled:
    //
    int boundaryCount = 0;
    if (_useInfSharpPatch) {
        boundaryCount = (vTags[0]._infSharpEdges && (vTags[0]._rule == Sdc::Crease::RULE_CREASE))
                      + (vTags[1]._infSharpEdges && (vTags[1]._rule == Sdc::Crease::RULE_CREASE))
                      + (vTags[2]._infSharpEdges && (vTags[2]._rule == Sdc::Crease::RULE_CREASE))
                      + (vTags[3]._infSharpEdges && (vTags[3]._rule == Sdc::Crease::RULE_CREASE));
    } else {
        boundaryCount = (vTags[0]._boundary && (vTags[0]._rule == Sdc::Crease::RULE_CREASE))
                      + (vTags[1]._boundary && (vTags[1]._rule == Sdc::Crease::RULE_CREASE))
                      + (vTags[2]._boundary && (vTags[2]._rule == Sdc::Crease::RULE_CREASE))
                      + (vTags[3]._boundary && (vTags[3]._rule == Sdc::Crease::RULE_CREASE));
    }
    int xordinaryCount = vTags[0]._xordinary
                       + vTags[1]._xordinary
                       + vTags[2]._xordinary
                       + vTags[3]._xordinary;

    if ((boundaryCount == 3) && (xordinaryCount == 1)) {
        //  This must be an isolated xordinary corner above level 1, otherwise we still
        //  need to assure the xordinary vertex is opposite a smooth interior vertex:
        //
        if (levelIndex > 1) return true;

        if (vTags[0]._xordinary) return (vTags[2]._rule == Sdc::Crease::RULE_SMOOTH);
        if (vTags[1]._xordinary) return (vTags[3]._rule == Sdc::Crease::RULE_SMOOTH);
        if (vTags[2]._xordinary) return (vTags[0]._rule == Sdc::Crease::RULE_SMOOTH);
        if (vTags[3]._xordinary) return (vTags[1]._rule == Sdc::Crease::RULE_SMOOTH);
    }
    return false;
}

bool
PatchBuilder::IsPatchRegular(
    int levelIndex, Index faceIndex, int fvcFactory) const {

    Level const & level = _refiner.getLevel(levelIndex);

    //  Ignore the face-varying channel if the topology for the face is not distinct
    int fvcRefiner = GetDistinctRefinerFVarChannel(levelIndex, faceIndex, fvcFactory);

    //  Retrieve the composite VTag for the four corners:
    Level::VTag fCompVTag = level.getFaceCompositeVTag(faceIndex, fvcRefiner);

    //
    //  Patches around non-manifold features are currently regular -- will need to revise
    //  this when infinitely sharp patches are introduced later:
    //
    bool isRegular = ! fCompVTag._xordinary || fCompVTag._nonManifold;

    //  Reconsider when using inf-sharp patches in presence of inf-sharp features:
    if (_useInfSharpPatch && (fCompVTag._infSharp || fCompVTag._infSharpEdges)) {
        if (fCompVTag._nonManifold || !fCompVTag._infIrregular) {
            isRegular = true;
        } else if (!fCompVTag._infSharpEdges) {
            isRegular = false;
        } else {
            //
            //   This is unfortunately a relatively complex case to determine... if a corner
            //   vertex has been tagged has having an inf-sharp irregularity about it, the
            //   neighborhood of the corner is partitioned into both regular and irregular
            //   regions and the face must be more closely inspected to determine in which
            //   it lies.
            //
            //   There could be a simpler test here to quickly accept/reject regularity given
            //   how local it is -- involving no more than one or two (in the case of Loop)
            //   adjacent faces -- but it will likely be messy and will need to inspect
            //   adjacent faces and/or edges.  In the meantime, gathering and inspecting the
            //   subset of the neighborhood delimited by inf-sharp edges will suffice (and
            //   be comparable in all but cases of high valence)
            //
            Level::VTag vTags[4];
            level.getFaceVTags(faceIndex, vTags, fvcRefiner);

            Level::VSpan vSpan;
            Level::ETag eMask = getSingularEdgeMask(true);

            isRegular = true;
            for (int i = 0; i < 4; ++i) {
                if (vTags[i]._infIrregular) {

                    identifyManifoldCornerSpan(level, faceIndex, i, eMask, vSpan, fvcRefiner);

                    isRegular = (vSpan._numFaces == (vTags[i]._infSharpCrease ? 2 : 1));
                    if (!isRegular) break;
                }
            }
        }

        //  When inf-sharp and extra-ordinary features are not isolated, need to inspect more
        //  closely -- any smooth extra-ordinary corner makes the patch irregular:
        if (fCompVTag._xordinary && (levelIndex < 2)) {
            Level::VTag vTags[4];
            level.getFaceVTags(faceIndex, vTags, fvcRefiner);
            for (int i = 0; i < 4; ++i) {
                if (vTags[i]._xordinary && (vTags[i]._rule == Sdc::Crease::RULE_SMOOTH)) {
                    isRegular = false;
                }
            }
        }
    }

    //  Legacy option -- reinterpret an irregular smooth corner as sharp if specified:
    if (!isRegular && _generateLegacySharpCornerPatches) {
        if (fCompVTag._xordinary && fCompVTag._boundary && !fCompVTag._nonManifold) {
            isRegular = IsPatchSmoothCorner(levelIndex, faceIndex, fvcRefiner);
        }
    }
    return isRegular;
}

void
PatchBuilder::GetIrregularPatchCornerSpans(
    int levelIndex, Index faceIndex, Level::VSpan cornerSpans[4], int fvcFactory) const {

    Level const & level = _refiner.getLevel(levelIndex);

    //  Ignore the face-varying channel if the topology for the face is not distinct
    int fvcRefiner = GetDistinctRefinerFVarChannel(levelIndex, faceIndex, fvcFactory);

    //  Retrieve tags and identify other information for the corner vertices:
    Level::VTag vTags[4];
    level.getFaceVTags(faceIndex, vTags, fvcRefiner);

    FVarLevel::ValueTag fvarTags[4];
    if (fvcRefiner >= 0) {
        level.getFVarLevel(fvcRefiner).getFaceValueTags(faceIndex, fvarTags);
    }

    //
    //  For each corner vertex, use the complete neighborhood when possible (which
    //  does not require a search, otherwise identify the span of interest around
    //  the vertex:
    //
    ConstIndexArray fVerts = level.getFaceVertices(faceIndex);

    Level::ETag singularEdgeMask = getSingularEdgeMask(_useInfSharpPatch);

    for (int i = 0; i < fVerts.size(); ++i) {
        bool noFVarMisMatch = (fvcRefiner < 0) || !fvarTags[i]._mismatch;
        bool testInfSharp   = _useInfSharpPatch &&
                                (vTags[i]._infSharpEdges && (vTags[i]._rule != Sdc::Crease::RULE_DART));

        if (noFVarMisMatch && !testInfSharp) {
            cornerSpans[i].clear();
        } else {
            if (!vTags[i]._nonManifold) {
                identifyManifoldCornerSpan(
                    level, faceIndex, i, singularEdgeMask, cornerSpans[i], fvcRefiner);
            } else {
                identifyNonManifoldCornerSpan(
                    level, faceIndex, i, singularEdgeMask, cornerSpans[i], fvcRefiner);
            }
        }
        if (vTags[i]._corner) {
            cornerSpans[i]._sharp = true;
        } else if (_useInfSharpPatch) {
            cornerSpans[i]._sharp = vTags[i]._infIrregular && (vTags[i]._rule == Sdc::Crease::RULE_CORNER);
        }

        //  Legacy option -- reinterpret an irregular smooth corner as sharp if specified:
        if (!cornerSpans[i]._sharp && _generateLegacySharpCornerPatches) {
            if (vTags[i]._xordinary && vTags[i]._boundary && !vTags[i]._nonManifold) {
                int nFaces = cornerSpans[i].isAssigned() ? cornerSpans[i]._numFaces
                           : level.getVertexFaces(fVerts[i]).size();
                cornerSpans[i]._sharp = (nFaces == 1);
            }
        }
    }
}

int
PatchBuilder::GetRegularPatchBoundaryMask(
    int levelIndex, Index faceIndex, int fvcFactory) const {

    Level const & level = _refiner.getLevel(levelIndex);

    //  Ignore the face-varying channel if the topology for the face is not distinct
    int fvcRefiner = GetDistinctRefinerFVarChannel(levelIndex, faceIndex, fvcFactory);

    //  Gather the VTags for the four corners.  Regardless of the options for
    //  treating non-manifold or inf-sharp patches, for a regular patch we can
    //  infer all that we need need from tags for the corner vertices:
    //
    Level::VTag vTags[4];
    level.getFaceVTags(faceIndex, vTags, fvcRefiner);

    Level::VTag fTag = Level::VTag::BitwiseOr(vTags);

    //
    //  Identify vertex tags for inf-sharp edges and/or boundaries, depending on
    //  whether or not inf-sharp patches are in use:
    //
    int vBoundaryMask = 0;
    if (fTag._infSharpEdges) {
        if (_useInfSharpPatch) {
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
    //  Non-manifold patches have been historically represented as regular in all
    //  cases -- when a non-manifold vertex is sharp, it requires a regular corner
    //  patch, and so both of its neighboring corners need to be re-interpreted as
    //  boundaries.
    //
    //  With the introduction of sharp irregular patches, we are now better off
    //  using irregular patches where appropriate, which will simplify the following
    //  when this patch was already determined to be regular...
    //
    if (fTag._nonManifold) {
        if (vTags[0]._nonManifold) vBoundaryMask |= (1 << 0) | (vTags[0]._infSharp ? 10 : 0);
        if (vTags[1]._nonManifold) vBoundaryMask |= (1 << 1) | (vTags[1]._infSharp ?  5 : 0);
        if (vTags[2]._nonManifold) vBoundaryMask |= (1 << 2) | (vTags[2]._infSharp ? 10 : 0);
        if (vTags[3]._nonManifold) vBoundaryMask |= (1 << 3) | (vTags[3]._infSharp ?  5 : 0);

        //  Force adjacent edges as boundaries if only one vertex in the resulting mask
        //  (which would be an irregular boundary for Catmark, but not Loop):
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


} // end namespace internal

} // end namespace Far

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv

