//
//   Copyright 2013 Pixar
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
#include "../far/patchTableFactory.h"
#include "../far/error.h"
#include "../far/ptexIndices.h"
#include "../far/topologyRefiner.h"
#include "../vtr/level.h"
#include "../vtr/fvarLevel.h"
#include "../vtr/refinement.h"
#include "../vtr/stackBuffer.h"
#include "../far/endCapBSplineBasisPatchFactory.h"
#include "../far/endCapGregoryBasisPatchFactory.h"
#include "../far/endCapLegacyGregoryPatchFactory.h"

#include <algorithm>
#include <cassert>
#include <cstring>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {
namespace Far {

using Vtr::internal::Refinement;
using Vtr::internal::Level;
using Vtr::internal::FVarLevel;

namespace {

//  Helpers for compiler warnings and floating point equality tests
#ifdef __INTEL_COMPILER
#pragma warning (push)
#pragma warning disable 1572
#endif

inline bool isSharpnessEqual(float s1, float s2) { return (s1 == s2); }

#ifdef __INTEL_COMPILER
#pragma warning (pop)
#endif

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
isEdgeSingular(Level const & level, FVarLevel const * fvarLevel, Index eIndex,
               Level::ETag eTagMask)
{
    Level::ETag eTag = level.getEdgeTag(eIndex);
    if (fvarLevel) {
        eTag = fvarLevel->getEdgeTag(eIndex).combineWithLevelETag(eTag);
    }

    Level::ETag::ETagSize * iTag  = reinterpret_cast<Level::ETag::ETagSize*>(&eTag);
    Level::ETag::ETagSize * iMask = reinterpret_cast<Level::ETag::ETagSize*>(&eTagMask);
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
    while (! isEdgeSingular(level, fvarLevel, vEdges[iTrailing], eTagMask)) {
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

//
//  Additional anonymous helper functions:
//
static inline void
offsetAndPermuteIndices(Far::Index const indices[], int count,
                        Far::Index offset, int const permutation[],
                        Far::Index result[]) {

    // The patch vertices for boundary and corner patches
    // are assigned index values even though indices will
    // be undefined along boundary and corner edges.
    // When the resulting patch table is going to be used
    // as indices for drawing, it is convenient for invalid
    // indices to be replaced with known good values, such
    // as the first un-permuted index, which is the index
    // of the first vertex of the patch face.
    Far::Index knownGoodIndex = indices[0];

    if (permutation) {
        for (int i = 0; i < count; ++i) {
            if (permutation[i] < 0) {
                result[i] = offset + knownGoodIndex;
            } else {
                result[i] = offset + indices[permutation[i]];
            }
        }
    } else if (offset) {
        for (int i = 0; i < count; ++i) {
            result[i] = offset + indices[i];
        }
    } else {
        std::memcpy(result, indices, count * sizeof(Far::Index));
    }
}

inline int
assignSharpnessIndex(float sharpness, std::vector<float> & sharpnessValues) {

    // linear search
    for (int i=0; i<(int)sharpnessValues.size(); ++i) {
        if (isSharpnessEqual(sharpnessValues[i], sharpness)) {
            return i;
        }
    }
    sharpnessValues.push_back(sharpness);
    return (int)sharpnessValues.size()-1;
}

} // namespace anon


//
// Builder Context
//
// Helper class aggregating transient contextual data structures during
// the creation of a patch table.
// This helps keeping the factory class stateless.
//
// Note : struct members are not re-entrant nor are they intended to be !
//
struct PatchTableFactory::BuilderContext {

public:
    // Simple struct to store <level,face> (and more?) info for a patch:
    struct PatchTuple {
        PatchTuple()
            : faceIndex(Vtr::INDEX_INVALID), levelIndex(-1) { }
        PatchTuple(PatchTuple const & p)
            : faceIndex(p.faceIndex), levelIndex(p.levelIndex) { }
        PatchTuple(Index faceIndex, int levelIndex)
            : faceIndex(faceIndex), levelIndex(levelIndex) { }

        Index faceIndex;
        int   levelIndex;
    };
    typedef std::vector<PatchTuple> PatchTupleVector;

public:
    BuilderContext(TopologyRefiner const & refiner, Options options);

    // Methods to query patch properties for classification and construction.
    bool IsPatchEligible(int levelIndex, Index faceIndex) const;

    bool IsPatchSmoothCorner(int levelIndex, Index faceIndex,
                             int fvcFactory = -1) const;

    bool IsPatchRegular(int levelIndex, Index faceIndex,
                        int fvcFactory = -1) const;

    int GetRegularPatchBoundaryMask(int levelIndex, Index faceIndex,
                                    int fvcFactory = -1) const;

    void GetIrregularPatchCornerSpans(int levelIndex, Index faceIndex,
                                      Level::VSpan cornerSpans[4],
                                      int fvcFactory = -1) const;

    // Methods to gather points associated with the different patch types
    int GatherLinearPatchPoints(Index * iptrs,
                                PatchTuple const & patch,
                                int fvcFactory = -1) const;
    int GatherRegularPatchPoints(Index * iptrs,
                                 PatchTuple const & patch,
                                 int boundaryMask,
                                 int fvcFactory = -1) const;
    template <class END_CAP_FACTORY_TYPE>
    int GatherIrregularPatchPoints(END_CAP_FACTORY_TYPE *endCapFactory,
                                   Index * iptrs,
                                   PatchTuple const & patch,
                                   Level::VSpan cornerSpans[4],
                                   int fvcFactory = -1) const;

    // Additional simple queries -- most regarding face-varying channels that hide
    // the mapping between channels in the source Refiner and corresponding channels
    // in the Factory and PatchTable
    //
    // True if face-varying patches need to be generated for this topology
    bool RequiresFVarPatches() const {
        return (! fvarChannelIndices.empty());
    }

    int GetRefinerFVarChannel(int fvcFactory) const {
        return (fvcFactory >= 0) ? fvarChannelIndices[fvcFactory] : -1;
    }

    bool DoesFaceVaryingPatchMatch(int levelIndex, Index faceIndex, int fvcFactory) const {
        return refiner.getLevel(levelIndex).doesFaceFVarTopologyMatch(faceIndex,
                        GetRefinerFVarChannel(fvcFactory));
    }

    int GetDistinctRefinerFVarChannel(int levelIndex, int faceIndex, int fvcFactory) const {
        return ((fvcFactory >= 0) &&
                !DoesFaceVaryingPatchMatch(levelIndex, faceIndex, fvcFactory)) ?
            GetRefinerFVarChannel(fvcFactory) : -1;
    }

    int GetTransitionMask(int levelIndex, Index faceIndex) const {
        return (levelIndex == refiner.GetMaxLevel()) ? 0 :
            refiner.getRefinement(levelIndex).
                       getParentFaceSparseTag(faceIndex)._transitional;
    }

public:
    TopologyRefiner const & refiner;

    Options const options;

    PtexIndices const ptexIndices;

    // Counters accumulating each type of patch during topology traversal
    int numRegularPatches;
    int numIrregularPatches;
    int numIrregularBoundaryPatches;

    // Tuple for each patch identified during topology traversal
    PatchTupleVector patches;

    std::vector<int> levelVertOffsets;
    std::vector< std::vector<int> > levelFVarValueOffsets;

    // These are the indices of face-varying channels in the refiner
    // or empty if we are not populating face-varying data.
    std::vector<int> fvarChannelIndices;
};

// Constructor
PatchTableFactory::BuilderContext::BuilderContext(
    TopologyRefiner const & ref, Options opts) :
    refiner(ref), options(opts), ptexIndices(refiner),
    numRegularPatches(0), numIrregularPatches(0),
    numIrregularBoundaryPatches(0) {

    if (options.generateFVarTables) {
        // If client-code does not select specific channels, default to all
        // the channels in the refiner.
        if (options.numFVarChannels==-1) {
            fvarChannelIndices.resize(refiner.GetNumFVarChannels());
            for (int fvc=0;fvc<(int)fvarChannelIndices.size(); ++fvc) {
                fvarChannelIndices[fvc] = fvc; // std::iota
            }
        } else {
            fvarChannelIndices.assign(
                options.fvarChannelIndices,
                options.fvarChannelIndices+options.numFVarChannels);
        }
    }
}

int
PatchTableFactory::BuilderContext::GatherLinearPatchPoints(
        Index * iptrs, PatchTuple const & patch, int fvcFactory) const {

    Level const & level = refiner.getLevel(patch.levelIndex);

    int levelVertOffset = (fvcFactory < 0)
                        ? levelVertOffsets[patch.levelIndex]
                        : levelFVarValueOffsets[fvcFactory][patch.levelIndex];

    int fvcRefiner = GetRefinerFVarChannel(fvcFactory);

    ConstIndexArray cvs = (fvcRefiner < 0)
                        ? level.getFaceVertices(patch.faceIndex)
                        : level.getFaceFVarValues(patch.faceIndex, fvcRefiner);

    for (int i = 0; i < cvs.size(); ++i) iptrs[i] = levelVertOffset + cvs[i];
    return cvs.size();
}

int
PatchTableFactory::BuilderContext::GatherRegularPatchPoints(
        Index * iptrs, PatchTuple const & patch, int boundaryMask,
        int fvcFactory) const {

    Level const & level = refiner.getLevel(patch.levelIndex);

    int levelVertOffset = (fvcFactory < 0)
                        ? levelVertOffsets[patch.levelIndex]
                        : levelFVarValueOffsets[fvcFactory][patch.levelIndex];

    int fvcRefiner = GetRefinerFVarChannel(fvcFactory);

    Index patchVerts[16];

    int bType  = 0;
    int bIndex = 0;
    if (boundaryMask) {
        static int const boundaryEdgeMaskToType[16] =
            { 0, 1, 1, 2, 1, -1, 2, -1, 1, 2, -1, -1, 2, -1, -1, -1 };
        static int const boundaryEdgeMaskToFeature[16] =
            { -1, 0, 1, 1, 2, -1, 2, -1, 3, 0, -1, -1, 3, -1, -1,-1 };

        bType  = boundaryEdgeMaskToType[boundaryMask];
        bIndex = boundaryEdgeMaskToFeature[boundaryMask];
    }

    int const * permutation = 0;

    if (bType == 0) {
        static int const permuteRegular[16] =
            { 5, 6, 7, 8, 4, 0, 1, 9, 15, 3, 2, 10, 14, 13, 12, 11 };
        permutation = permuteRegular;
        level.gatherQuadRegularInteriorPatchPoints(
                patch.faceIndex, patchVerts, /*rotation=*/0, fvcRefiner);
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
                patch.faceIndex, patchVerts, bIndex, fvcRefiner);
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
                patch.faceIndex, patchVerts, bIndex, fvcRefiner);
    } else {
        assert(bType <= 2);
    }

    offsetAndPermuteIndices( patchVerts, 16, levelVertOffset, permutation, iptrs);
    return 16;
}

template <class END_CAP_FACTORY_TYPE>
int
PatchTableFactory::BuilderContext::GatherIrregularPatchPoints(
        END_CAP_FACTORY_TYPE *endCapFactory,
        Index * iptrs, PatchTuple const & patch,
        Level::VSpan cornerSpans[4],
        int fvcFactory) const {

    Level const & level = refiner.getLevel(patch.levelIndex);

    int levelVertOffset = (fvcFactory < 0)
                        ? levelVertOffsets[patch.levelIndex]
                        : levelFVarValueOffsets[fvcFactory][patch.levelIndex];

    int fvcRefiner = GetRefinerFVarChannel(fvcFactory);

    ConstIndexArray cvs = endCapFactory->GetPatchPoints(
        &level, patch.faceIndex, cornerSpans, levelVertOffset, fvcRefiner);

    for (int i = 0; i < cvs.size(); ++i) iptrs[i] = cvs[i];
    return cvs.size();
}

bool
PatchTableFactory::BuilderContext::IsPatchEligible(
        int levelIndex, Index faceIndex) const {

    //
    //  Patches that are not eligible correpond to the following faces:
    //      - holes
    //      - those in intermediate levels that are further refined (not leaves)
    //      - those without complete neighborhoods that supporting other patches
    //
    Level const & level = refiner.getLevel(levelIndex);

    if (level.isFaceHole(faceIndex)) {
        return false;
    }

    if (levelIndex < refiner.GetMaxLevel()) {
        if (refiner.getRefinement(levelIndex).getParentFaceSparseTag(faceIndex)._selected) {
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
PatchTableFactory::BuilderContext::IsPatchSmoothCorner(
        int levelIndex, Index faceIndex, int fvcFactory) const
{
    Level const & level = refiner.getLevel(levelIndex);

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
    if (options.useInfSharpPatch) {
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
PatchTableFactory::BuilderContext::IsPatchRegular(
        int levelIndex, Index faceIndex, int fvcFactory) const
{
    Level const & level = refiner.getLevel(levelIndex);

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
    if (options.useInfSharpPatch && (fCompVTag._infSharp || fCompVTag._infSharpEdges)) {
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
    if (!isRegular && options.generateLegacySharpCornerPatches) {
        if (fCompVTag._xordinary && fCompVTag._boundary && !fCompVTag._nonManifold) {
            isRegular = IsPatchSmoothCorner(levelIndex, faceIndex, fvcRefiner);
        }
    }
    return isRegular;
}

int
PatchTableFactory::BuilderContext::GetRegularPatchBoundaryMask(
        int levelIndex, Index faceIndex, int fvcFactory) const
{
    Level const & level = refiner.getLevel(levelIndex);

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
        if (options.useInfSharpPatch) {
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

void
PatchTableFactory::BuilderContext::GetIrregularPatchCornerSpans(
        int levelIndex, Index faceIndex,
        Level::VSpan cornerSpans[4],
        int fvcFactory) const
{
    Level const & level = refiner.getLevel(levelIndex);

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

    Level::ETag singularEdgeMask = getSingularEdgeMask(options.useInfSharpPatch);

    for (int i = 0; i < fVerts.size(); ++i) {
        bool noFVarMisMatch = (fvcRefiner < 0) || !fvarTags[i]._mismatch;
        bool testInfSharp   = options.useInfSharpPatch &&
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
        } else if (options.useInfSharpPatch) {
            cornerSpans[i]._sharp = vTags[i]._infIrregular && (vTags[i]._rule == Sdc::Crease::RULE_CORNER);
        }

        //  Legacy option -- reinterpret an irregular smooth corner as sharp if specified:
        if (!cornerSpans[i]._sharp && options.generateLegacySharpCornerPatches) {
            if (vTags[i]._xordinary && vTags[i]._boundary && !vTags[i]._nonManifold) {
                    int nFaces = cornerSpans[i].isAssigned() ? cornerSpans[i]._numFaces
                               : level.getVertexFaces(fVerts[i]).size();
                    cornerSpans[i]._sharp = (nFaces == 1);
            }
        }
    }
}

//
//  Reserves tables based on the contents of the PatchArrayVector in the PatchTable:
//
void
PatchTableFactory::allocateVertexTables(
        BuilderContext const & context, PatchTable * table) {

    int ncvs = 0, npatches = 0;
    for (int i=0; i<table->GetNumPatchArrays(); ++i) {
        npatches += table->GetNumPatches(i);
        ncvs += table->GetNumControlVertices(i);
    }

    if (ncvs==0 || npatches==0)
        return;

    table->_patchVerts.resize( ncvs );

    table->_paramTable.resize( npatches );

    if (! context.refiner.IsUniform()) {
        table->allocateVaryingVertices(
            PatchDescriptor(PatchDescriptor::QUADS), npatches);
    }

    if (context.options.useSingleCreasePatch) {
        table->_sharpnessIndices.resize( npatches, Vtr::INDEX_INVALID );
    }
}

//
//  Allocate face-varying tables
//
void
PatchTableFactory::allocateFVarChannels(
        BuilderContext const & context, PatchTable * table) {

    TopologyRefiner const & refiner = context.refiner;

    int npatches = table->GetNumPatchesTotal();

    table->allocateFVarPatchChannels((int)context.fvarChannelIndices.size());

    // Initialize each channel
    for (int fvc=0; fvc<(int)context.fvarChannelIndices.size(); ++fvc) {
        int refinerChannel = context.fvarChannelIndices[fvc];

        Sdc::Options::FVarLinearInterpolation interpolation =
            refiner.GetFVarLinearInterpolation(refinerChannel);

        table->setFVarPatchChannelLinearInterpolation(interpolation, fvc);

        if (refiner.IsUniform()) {
            PatchDescriptor::Type uniformType = context.options.triangulateQuads
                ? PatchDescriptor::TRIANGLES
                : PatchDescriptor::QUADS;

            table->allocateFVarPatchChannelValues(
                PatchDescriptor(uniformType), npatches, fvc);

        } else {
            bool allLinear = context.options.generateFVarLegacyLinearPatches ||
                (interpolation == Sdc::Options::FVAR_LINEAR_ALL);

            PatchDescriptor::Type adaptiveType = allLinear
                    ? PatchDescriptor::QUADS
                    : ((context.options.GetEndCapType() == Options::ENDCAP_GREGORY_BASIS)
                        ? PatchDescriptor::GREGORY_BASIS
                        : PatchDescriptor::REGULAR);

            table->allocateFVarPatchChannelValues(
                PatchDescriptor(adaptiveType), npatches, fvc);
        }
    }
}

//
//  Populates the PatchParam for the given face, returning
//  a pointer to the next entry
//
PatchParam
PatchTableFactory::computePatchParam(
    BuilderContext const & context,
    int depth, Index faceIndex,
    int boundaryMask, int transitionMask) {

    TopologyRefiner const & refiner = context.refiner;

    // Move up the hierarchy accumulating u,v indices to the coarse level:
    int childIndexInParent = 0,
        u = 0,
        v = 0,
        ofs = 1;

    bool nonquad = (refiner.GetLevel(depth).GetFaceVertices(faceIndex).size() != 4);

    for (int i = depth; i > 0; --i) {
        Refinement const& refinement  = refiner.getRefinement(i-1);
        Level const&      parentLevel = refiner.getLevel(i-1);

        Index parentFaceIndex = refinement.getChildFaceParentFace(faceIndex);

        if (parentLevel.getFaceVertices(parentFaceIndex).size() == 4) {
            childIndexInParent = refinement.getChildFaceInParentFace(faceIndex);
            switch ( childIndexInParent ) {
                case 0 :                     break;
                case 1 : { u+=ofs;         } break;
                case 2 : { u+=ofs; v+=ofs; } break;
                case 3 : {         v+=ofs; } break;
            }
            ofs = (unsigned short)(ofs << 1);
        } else {
            nonquad = true;
            // If the root face is not a quad, we need to offset the ptex index
            // CCW to match the correct child face
            Vtr::ConstIndexArray children = refinement.getFaceChildFaces(parentFaceIndex);
            for (int j=0; j<children.size(); ++j) {
                if (children[j]==faceIndex) {
                    childIndexInParent = j;
                    break;
                }
            }
        }
        faceIndex = parentFaceIndex;
    }

    Index ptexIndex = context.ptexIndices.GetFaceId(faceIndex);
    assert(ptexIndex!=-1);

    if (nonquad) {
        ptexIndex+=childIndexInParent;
    }

    PatchParam param;
    param.Set(ptexIndex, (short)u, (short)v, (unsigned short) depth, nonquad,
              (unsigned short) boundaryMask, (unsigned short) transitionMask);
    return param;
}

//
//  We should be able to use a single Create() method for both the adaptive and uniform
//  cases.  In the past, more additional arguments were passed to the uniform version,
//  but that may no longer be necessary (see notes in the uniform version below)...
//
PatchTable *
PatchTableFactory::Create(TopologyRefiner const & refiner, Options options) {

    if (refiner.IsUniform()) {
        return createUniform(refiner, options);
    } else {
        return createAdaptive(refiner, options);
    }
}

PatchTable *
PatchTableFactory::createUniform(TopologyRefiner const & refiner, Options options) {

    assert(refiner.IsUniform());

    BuilderContext context(refiner, options);

    // ensure that triangulateQuads is only set for quadrilateral schemes
    options.triangulateQuads &= (refiner.GetSchemeType()==Sdc::SCHEME_BILINEAR ||
                                 refiner.GetSchemeType()==Sdc::SCHEME_CATMARK);

    // level=0 may contain n-gons, which are not supported in PatchTable.
    // even if generateAllLevels = true, we start from level 1.

    int maxvalence = refiner.GetMaxValence(),
        maxlevel = refiner.GetMaxLevel(),
        firstlevel = options.generateAllLevels ? 1 : maxlevel,
        nlevels = maxlevel-firstlevel+1;

    PatchDescriptor::Type ptype = PatchDescriptor::NON_PATCH;
    if (options.triangulateQuads) {
        ptype = PatchDescriptor::TRIANGLES;
    } else {
        switch (refiner.GetSchemeType()) {
            case Sdc::SCHEME_BILINEAR :
            case Sdc::SCHEME_CATMARK  : ptype = PatchDescriptor::QUADS; break;
            case Sdc::SCHEME_LOOP     : ptype = PatchDescriptor::TRIANGLES; break;
        }
    }
    assert(ptype!=PatchDescriptor::NON_PATCH);

    //
    //  Create the instance of the table and allocate and initialize its members.
    PatchTable * table = new PatchTable(maxvalence);

    table->_numPtexFaces = context.ptexIndices.GetNumFaces();

    table->reservePatchArrays(nlevels);

    PatchDescriptor desc(ptype);

    // generate patch arrays
    for (int level=firstlevel, poffset=0, voffset=0; level<=maxlevel; ++level) {

        TopologyLevel const & refLevel = refiner.GetLevel(level);

        int npatches = refLevel.GetNumFaces();
        if (refiner.HasHoles()) {
            for (int i = npatches - 1; i >= 0; --i) {
                npatches -= refLevel.IsFaceHole(i);
            }
        }
        assert(npatches>=0);

        if (options.triangulateQuads)
            npatches *= 2;

        table->pushPatchArray(desc, npatches, &voffset, &poffset, 0);
    }

    // Allocate various tables
    allocateVertexTables(context, table);

    if (context.RequiresFVarPatches()) {
        allocateFVarChannels(context, table);
    }

    //
    //  Now populate the patches:
    //

    Index          * iptr = &table->_patchVerts[0];
    PatchParam     * pptr = &table->_paramTable[0];
    Index         ** fptr = 0;

    // we always skip level=0 vertices (control cages)
    Index levelVertOffset = refiner.GetLevel(0).GetNumVertices();

    Index * levelFVarVertOffsets = 0;
    if (context.RequiresFVarPatches()) {

        levelFVarVertOffsets = (Index *)alloca(context.fvarChannelIndices.size()*sizeof(Index));
        memset(levelFVarVertOffsets, 0, context.fvarChannelIndices.size()*sizeof(Index));

        fptr = (Index **)alloca(context.fvarChannelIndices.size()*sizeof(Index *));
        for (int fvc=0; fvc<(int)context.fvarChannelIndices.size(); ++fvc) {
            fptr[fvc] = table->getFVarValues(fvc).begin();
        }
    }

    for (int level=1; level<=maxlevel; ++level) {

        TopologyLevel const & refLevel = refiner.GetLevel(level);

        int nfaces = refLevel.GetNumFaces();
        if (level>=firstlevel) {
            for (int face=0; face<nfaces; ++face) {

                if (refiner.HasHoles() && refLevel.IsFaceHole(face)) {
                    continue;
                }

                ConstIndexArray fverts = refLevel.GetFaceVertices(face);
                for (int vert=0; vert<fverts.size(); ++vert) {
                    *iptr++ = levelVertOffset + fverts[vert];
                }

                *pptr++ = computePatchParam(context, level, face, /*boundary*/0, /*transition*/0);

                if (context.RequiresFVarPatches()) {
                    for (int fvc=0; fvc<(int)context.fvarChannelIndices.size(); ++fvc) {
                        int refinerChannel = context.fvarChannelIndices[fvc];

                        ConstIndexArray fvalues = refLevel.GetFaceFVarValues(face, refinerChannel);
                        for (int vert=0; vert<fvalues.size(); ++vert) {
                            assert((levelVertOffset + fvalues[vert]) < (int)table->getFVarValues(fvc).size());
                            fptr[fvc][vert] = levelFVarVertOffsets[fvc] + fvalues[vert];
                        }
                        fptr[fvc]+=fvalues.size();
                    }
                }

                if (options.triangulateQuads) {
                    // Triangulate the quadrilateral: {v0,v1,v2,v3} -> {v0,v1,v2},{v3,v0,v2}.
                    *iptr = *(iptr - 4); // copy v0 index
                    ++iptr;
                    *iptr = *(iptr - 3); // copy v2 index
                    ++iptr;

                    *pptr = *(pptr - 1); // copy first patch param
                    ++pptr;

                    if (context.RequiresFVarPatches()) {
                        for (int fvc=0; fvc<(int)context.fvarChannelIndices.size(); ++fvc) {
                            *fptr[fvc] = *(fptr[fvc]-4); // copy fv0 index
                            ++fptr[fvc];
                            *fptr[fvc] = *(fptr[fvc]-3); // copy fv2 index
                            ++fptr[fvc];
                        }
                    }
                }
            }
        }

        if (options.generateAllLevels) {
            levelVertOffset += refiner.GetLevel(level).GetNumVertices();
            for (int fvc=0; fvc<(int)context.fvarChannelIndices.size(); ++fvc) {
                int refinerChannel = context.fvarChannelIndices[fvc];
                levelFVarVertOffsets[fvc] += refiner.GetLevel(level).GetNumFVarValues(refinerChannel);
            }
        }
    }
    return table;
}

PatchTable *
PatchTableFactory::createAdaptive(TopologyRefiner const & refiner, Options options) {

    assert(! refiner.IsUniform());

    BuilderContext context(refiner, options);

    //
    //  First identify the patches -- accumulating an inventory of
    //  information about each resulting patch:
    //
    identifyAdaptivePatches(context);

    //
    //  Create and initialize the instance of the table:
    //
    int maxValence = refiner.GetMaxValence();

    PatchTable * table = new PatchTable(maxValence);

    table->_numPtexFaces = context.ptexIndices.GetNumFaces();

    //
    //  Now populate the patches:
    //
    populateAdaptivePatches(context, table);

    return table;
}

//
//  Identify all patches required for faces at all levels -- accumulating the number of patches
//  for each type, and retaining enough information for the patch for each face to populate it
//  later with no additional analysis.
//
void
PatchTableFactory::identifyAdaptivePatches(BuilderContext & context) {

    TopologyRefiner const & refiner = context.refiner;

    //
    //  Iterate through the levels of refinement to inspect and tag components with information
    //  relative to patch generation.  We allocate all of the tags locally and use them to
    //  populate the patches once a complete inventory has been taken and all tables appropriately
    //  allocated and initialized:
    //
    //  The first Level may have no Refinement if it is the only level -- similarly the last Level
    //  has no Refinement, so a single level is effectively the last, but with less information
    //  available in some cases, as it was not generated by refinement.
    //
    int reservePatches = refiner.GetNumFacesTotal();
    context.patches.reserve(reservePatches);

    context.levelVertOffsets.push_back(0);
    context.levelFVarValueOffsets.resize(context.fvarChannelIndices.size());
    for (int fvc=0; fvc<(int)context.fvarChannelIndices.size(); ++fvc) {
        context.levelFVarValueOffsets[fvc].push_back(0);
    }

    for (int levelIndex=0; levelIndex<refiner.GetNumLevels(); ++levelIndex) {
        Level const & level = refiner.getLevel(levelIndex);

        context.levelVertOffsets.push_back(
                context.levelVertOffsets.back() + level.getNumVertices());

        for (int fvc=0; fvc<(int)context.fvarChannelIndices.size(); ++fvc) {
            int refinerChannel = context.fvarChannelIndices[fvc];
            context.levelFVarValueOffsets[fvc].push_back(
                context.levelFVarValueOffsets[fvc].back()
                + level.getNumFVarValues(refinerChannel));
        }

        for (int faceIndex = 0; faceIndex < level.getNumFaces(); ++faceIndex) {

            if (context.IsPatchEligible(levelIndex, faceIndex)) {

                context.patches.push_back(BuilderContext::PatchTuple(faceIndex, levelIndex));

                // Count the patches here to simplify subsequent allocation.
                if (context.IsPatchRegular(levelIndex, faceIndex)) {
                    ++context.numRegularPatches;
                } else {
                    ++context.numIrregularPatches;

                    // For legacy gregory patches we need to know how many
                    // irregular patches are also boundary patches.
                    if (context.options.GetEndCapType() == Options::ENDCAP_LEGACY_GREGORY) {
                        bool isBoundaryPatch = level.getFaceCompositeVTag(faceIndex)._boundary;
                        context.numIrregularBoundaryPatches += isBoundaryPatch;
                    }
                }
            }
        }
    }
}

//
//  Populate adaptive patches that we've previously identified.
//
void
PatchTableFactory::populateAdaptivePatches(
    BuilderContext & context, PatchTable * table) {

    TopologyRefiner const & refiner = context.refiner;

    // State needed to populate an array in the patch table.
    // Pointers in this structure are initialized after the patch array
    // data buffers have been allocated and are then incremented as we
    // populate data into the patch table. Currently, we'll have at
    // most 3 patch arrays: Regular, Irregular, and IrregularBoundary.
    struct PatchArrayBuilder {
        PatchArrayBuilder()
            : patchType(PatchDescriptor::REGULAR), numPatches(0)
            , iptr(NULL), pptr(NULL), sptr(NULL) { }

        PatchDescriptor::Type patchType;
        int numPatches;

        Far::Index *iptr;
        Far::PatchParam *pptr;
        Far::Index *sptr;
        Vtr::internal::StackBuffer<Far::Index*,1> fptr;
        Vtr::internal::StackBuffer<Far::PatchParam*,1> fpptr;

    private:
        // Non-copyable
        PatchArrayBuilder(PatchArrayBuilder const &) {}
        PatchArrayBuilder & operator=(PatchArrayBuilder const &) {return *this;}

    } arrayBuilders[3];
    int R = 0, IR = 1, IRB = 2; // Regular, Irregular, IrregularBoundary

    // Regular patches patches will be packed into the first patch array
    arrayBuilders[R].patchType = PatchDescriptor::REGULAR;
    arrayBuilders[R].numPatches = context.numRegularPatches;
    int numPatchArrays = (arrayBuilders[R].numPatches > 0);

    switch(context.options.GetEndCapType()) {
    case Options::ENDCAP_BSPLINE_BASIS:
        // Irregular patches are converted to bspline basis and
        // will be packed into the same patch array as regular patches
        IR = IRB = R;
        arrayBuilders[R].numPatches += context.numIrregularPatches;
        // Make sure we've counted this array even when
        // there are no regular patches.
        numPatchArrays = (arrayBuilders[R].numPatches > 0);
        break;
    case Options::ENDCAP_GREGORY_BASIS:
        // Irregular patches (both interior and boundary) are converted
        // to Gregory basis and will be packed into an additional patch array
        IR = IRB = numPatchArrays;
        arrayBuilders[IR].patchType = PatchDescriptor::GREGORY_BASIS;
        arrayBuilders[IR].numPatches += context.numIrregularPatches;
        numPatchArrays += (arrayBuilders[IR].numPatches > 0);
        break;
    case Options::ENDCAP_LEGACY_GREGORY:
        // Irregular interior and irregular boundary patches each will be
        // packed into separate additional patch arrays.
        IR = numPatchArrays;
        arrayBuilders[IR].patchType = PatchDescriptor::GREGORY;
        arrayBuilders[IR].numPatches = context.numIrregularPatches
                                     - context.numIrregularBoundaryPatches;
        numPatchArrays += (arrayBuilders[IR].numPatches > 0);

        IRB = numPatchArrays;
        arrayBuilders[IRB].patchType = PatchDescriptor::GREGORY_BOUNDARY;
        arrayBuilders[IRB].numPatches = context.numIrregularBoundaryPatches;
        numPatchArrays += (arrayBuilders[IRB].numPatches > 0);
        break;
    default:
        break;
    }

    // Create patch arrays
    table->reservePatchArrays(numPatchArrays);

    int voffset=0, poffset=0, qoffset=0;
    for (int arrayIndex=0; arrayIndex<numPatchArrays; ++arrayIndex) {
        PatchArrayBuilder & arrayBuilder = arrayBuilders[arrayIndex];
        table->pushPatchArray(PatchDescriptor(arrayBuilder.patchType),
            arrayBuilder.numPatches, &voffset, &poffset, &qoffset );
    }

    // Allocate patch array data buffers
    bool hasSharpness = context.options.useSingleCreasePatch;
    allocateVertexTables(context, table);

    if (context.RequiresFVarPatches()) {
        allocateFVarChannels(context, table);
    }

    // Initialize pointers used while populating patch array data buffers
    for (int arrayIndex=0; arrayIndex<numPatchArrays; ++arrayIndex) {
        PatchArrayBuilder & arrayBuilder = arrayBuilders[arrayIndex];

        arrayBuilder.iptr = table->getPatchArrayVertices(arrayIndex).begin();
        arrayBuilder.pptr = table->getPatchParams(arrayIndex).begin();
        if (hasSharpness) {
            arrayBuilder.sptr = table->getSharpnessIndices(arrayIndex);
        }

        if (context.RequiresFVarPatches()) {
            arrayBuilder.fptr.SetSize((int)context.fvarChannelIndices.size());
            arrayBuilder.fpptr.SetSize((int)context.fvarChannelIndices.size());

            for (int fvc=0; fvc<(int)context.fvarChannelIndices.size(); ++fvc) {

                PatchDescriptor desc = table->GetFVarPatchDescriptor(fvc);

                Index pidx = table->getPatchIndex(arrayIndex, 0);
                int   ofs  = pidx * desc.GetNumControlVertices();

                arrayBuilder.fptr[fvc] = &table->getFVarValues(fvc)[ofs];
                arrayBuilder.fpptr[fvc] = &table->getFVarPatchParams(fvc)[pidx];
            }
        }
    }

    // endcap factories
    // XXX
    EndCapBSplineBasisPatchFactory *endCapBSpline = NULL;
    EndCapGregoryBasisPatchFactory *endCapGregoryBasis = NULL;
    EndCapLegacyGregoryPatchFactory *endCapLegacyGregory = NULL;
    Vtr::internal::StackBuffer<EndCapBSplineBasisPatchFactory*,1> fvarEndCapBSpline;
    Vtr::internal::StackBuffer<EndCapGregoryBasisPatchFactory*,1> fvarEndCapGregoryBasis;

    StencilTable *localPointStencils = NULL;
    StencilTable *localPointVaryingStencils = NULL;
    Vtr::internal::StackBuffer<StencilTable*,1> localPointFVarStencils;

    switch(context.options.GetEndCapType()) {
    case Options::ENDCAP_GREGORY_BASIS:
        localPointStencils = new StencilTable(0);
        localPointVaryingStencils = new StencilTable(0);
        endCapGregoryBasis = new EndCapGregoryBasisPatchFactory(
            refiner,
            localPointStencils,
            localPointVaryingStencils,
            context.options.shareEndCapPatchPoints);
        break;
    case Options::ENDCAP_BSPLINE_BASIS:
        localPointStencils = new StencilTable(0);
        localPointVaryingStencils = new StencilTable(0);
        endCapBSpline = new EndCapBSplineBasisPatchFactory(
            refiner,
            localPointStencils,
            localPointVaryingStencils);
        break;
    case Options::ENDCAP_LEGACY_GREGORY:
        endCapLegacyGregory = new EndCapLegacyGregoryPatchFactory(refiner);
        break;
    default:
        break;
    }

    if (context.RequiresFVarPatches()) {
        fvarEndCapBSpline.SetSize((int)context.fvarChannelIndices.size());
        fvarEndCapGregoryBasis.SetSize((int)context.fvarChannelIndices.size());
        localPointFVarStencils.SetSize((int)context.fvarChannelIndices.size());

        for (int fvc=0; fvc<(int)context.fvarChannelIndices.size(); ++fvc) {
            switch(context.options.GetEndCapType()) {
            case Options::ENDCAP_GREGORY_BASIS:
                localPointFVarStencils[fvc] = new StencilTable(0);
                fvarEndCapGregoryBasis[fvc] = new EndCapGregoryBasisPatchFactory(
                    refiner,
                    localPointFVarStencils[fvc],
                    NULL,
                    context.options.shareEndCapPatchPoints);
                break;
            case Options::ENDCAP_BSPLINE_BASIS:
                localPointFVarStencils[fvc] = new StencilTable(0);
                fvarEndCapBSpline[fvc] = new EndCapBSplineBasisPatchFactory(
                    refiner,
                    localPointFVarStencils[fvc],
                    NULL);
                break;
            default:
                break;
            }
        }
    }

    // Populate patch data buffers
    for (int patchIndex=0; patchIndex<(int)context.patches.size(); ++patchIndex) {

        BuilderContext::PatchTuple const & patch = context.patches[patchIndex];

        Level const & level = refiner.getLevel(patch.levelIndex);

        Level::VTag faceVTags = level.getFaceCompositeVTag(patch.faceIndex);

        PatchArrayBuilder * arrayBuilder = 0;

        // Properties to potentially be shared across vertex and face-varying patches:
        int          regBoundaryMask = 0;
        bool         isRegSingleCrease = false;
        Level::VSpan irregCornerSpans[4];
        float        sharpness = 0.0f;

        bool isRegular = context.IsPatchRegular(patch.levelIndex, patch.faceIndex);
        if (isRegular) {
            // Build the regular patch array
            arrayBuilder = &arrayBuilders[R];

            regBoundaryMask = context.GetRegularPatchBoundaryMask(patch.levelIndex, patch.faceIndex);

            // Test regular interior patches for a single-crease patch when specified:
            if (hasSharpness && (regBoundaryMask == 0) && (faceVTags._semiSharpEdges ||
                                                           faceVTags._infSharpEdges)) {
                float edgeSharpness = 0.0f;
                int   edgeInFace = 0;
                if (level.isSingleCreasePatch(patch.faceIndex, &edgeSharpness, &edgeInFace)) {
                    // cap sharpness to the max isolation level
                    edgeSharpness = std::min(edgeSharpness,
                        float(context.options.maxIsolationLevel - patch.levelIndex));

                    if (edgeSharpness > 0.0f) {
                        isRegSingleCrease = true;
                        regBoundaryMask = (1 << edgeInFace);
                        sharpness = edgeSharpness;
                    }
                }
            }
            
            //  The single-crease patch is an interior patch so ignore boundary mask when gathering:
            if (isRegSingleCrease) {
                arrayBuilder->iptr +=
                    context.GatherRegularPatchPoints(arrayBuilder->iptr, patch, 0);
            } else {
                arrayBuilder->iptr +=
                    context.GatherRegularPatchPoints(arrayBuilder->iptr, patch, regBoundaryMask);
            }
        } else {
            // Build the irregular patch array
            arrayBuilder = &arrayBuilders[IR];

            context.GetIrregularPatchCornerSpans(patch.levelIndex, patch.faceIndex, irregCornerSpans);

            // switch endcap patch type by option
            switch(context.options.GetEndCapType()) {
            case Options::ENDCAP_GREGORY_BASIS:
                arrayBuilder->iptr +=
                    context.GatherIrregularPatchPoints(
                        endCapGregoryBasis, arrayBuilder->iptr, patch, irregCornerSpans);
                break;
            case Options::ENDCAP_BSPLINE_BASIS:
                arrayBuilder->iptr +=
                    context.GatherIrregularPatchPoints(
                        endCapBSpline, arrayBuilder->iptr, patch, irregCornerSpans);
                break;
            case Options::ENDCAP_LEGACY_GREGORY:
                // For legacy gregory patches we may need to switch to
                // the irregular boundary patch array.
                if (!faceVTags._boundary) {
                    arrayBuilder->iptr +=
                        context.GatherIrregularPatchPoints(
                            endCapLegacyGregory, arrayBuilder->iptr, patch, irregCornerSpans);
                } else {
                    arrayBuilder = &arrayBuilders[IRB];
                    arrayBuilder->iptr +=
                        context.GatherIrregularPatchPoints(
                            endCapLegacyGregory, arrayBuilder->iptr, patch, irregCornerSpans);
                }
                break;
            case Options::ENDCAP_BILINEAR_BASIS:
                // not implemented yet
                assert(false);
                break;
            default:
                // no endcap
                break;
            }
        }

        // Assign the patch param (why is transition mask 0 if not regular?)
        int paramBoundaryMask = regBoundaryMask;
        int paramTransitionMask = isRegular ?
                context.GetTransitionMask(patch.levelIndex, patch.faceIndex) : 0;

        PatchParam patchParam =
            computePatchParam(context,
                              patch.levelIndex, patch.faceIndex,
                              paramBoundaryMask, paramTransitionMask);
        *arrayBuilder->pptr++ = patchParam;

        if (hasSharpness) {
            *arrayBuilder->sptr++ =
                assignSharpnessIndex(sharpness, table->_sharpnessValues);
        }

        if (context.RequiresFVarPatches()) {
            for (int fvc=0; fvc<(int)context.fvarChannelIndices.size(); ++fvc) {

                BuilderContext::PatchTuple fvarPatch(patch);

                PatchDescriptor desc = table->GetFVarPatchDescriptor(fvc);

                PatchParam fvarPatchParam = patchParam;

                // Deal with the linear cases trivially first
                if (desc.GetType() == PatchDescriptor::QUADS) {
                    arrayBuilder->fptr[fvc] +=
                        context.GatherLinearPatchPoints(
                            arrayBuilder->fptr[fvc], fvarPatch, fvc);
                    *arrayBuilder->fpptr[fvc]++ = fvarPatchParam;
                    continue;
                }

                // For non-linear patches, reuse patch information when the topology
                // of the face in face-varying space matches the original patch:
                //
                bool fvarTopologyMatches = context.DoesFaceVaryingPatchMatch(
                        patch.levelIndex, patch.faceIndex, fvc);

                bool fvarIsRegular = fvarTopologyMatches ? isRegular :
                        context.IsPatchRegular(patch.levelIndex, patch.faceIndex, fvc);

                int fvarBoundaryMask = 0;
                if (fvarIsRegular) {
                    fvarBoundaryMask = fvarTopologyMatches ? regBoundaryMask :
                        context.GetRegularPatchBoundaryMask(patch.levelIndex, patch.faceIndex, fvc);

                    if (isRegSingleCrease && fvarTopologyMatches) {
                        context.GatherRegularPatchPoints(
                                arrayBuilder->fptr[fvc], fvarPatch, 0, fvc);
                    } else {
                        context.GatherRegularPatchPoints(
                                arrayBuilder->fptr[fvc], fvarPatch, fvarBoundaryMask, fvc);
                    }
                } else {
                    Level::VSpan  localCornerSpans[4];
                    Level::VSpan* fvarCornerSpans = localCornerSpans;
                    if (fvarTopologyMatches) {
                        fvarCornerSpans = irregCornerSpans;
                    } else {
                        context.GetIrregularPatchCornerSpans(
                                patch.levelIndex, patch.faceIndex, fvarCornerSpans, fvc);
                    }

                    if (desc.GetType() == PatchDescriptor::REGULAR) {
                        context.GatherIrregularPatchPoints(
                                fvarEndCapBSpline[fvc],
                                arrayBuilder->fptr[fvc], fvarPatch, fvarCornerSpans, fvc);
                    } else if (desc.GetType() == PatchDescriptor::GREGORY_BASIS) {
                        context.GatherIrregularPatchPoints(
                                fvarEndCapGregoryBasis[fvc],
                                arrayBuilder->fptr[fvc], fvarPatch, fvarCornerSpans, fvc);
                    } else {
                        assert("Unknown Descriptor for FVar patch" == 0);
                    }
                }
                arrayBuilder->fptr[fvc] += desc.GetNumControlVertices();

                fvarPatchParam.Set(
                    patchParam.GetFaceId(),
                    patchParam.GetU(), patchParam.GetV(),
                    patchParam.GetDepth(),
                    patchParam.NonQuadRoot(),
                    (fvarIsRegular ? fvarBoundaryMask : 0),
                    patchParam.GetTransition(),
                    fvarIsRegular);
                *arrayBuilder->fpptr[fvc]++ = fvarPatchParam;
            }
        }
    }

    table->populateVaryingVertices();

    // finalize end patches
    if (localPointStencils && localPointStencils->GetNumStencils() > 0) {
        localPointStencils->finalize();
    } else {
        delete localPointStencils;
        localPointStencils = NULL;
    }

    if (localPointVaryingStencils && localPointVaryingStencils->GetNumStencils() > 0) {
        localPointVaryingStencils->finalize();
    } else {
        delete localPointVaryingStencils;
        localPointVaryingStencils = NULL;
    }

    switch(context.options.GetEndCapType()) {
    case Options::ENDCAP_GREGORY_BASIS:
        table->_localPointStencils = localPointStencils;
        table->_localPointVaryingStencils = localPointVaryingStencils;
        delete endCapGregoryBasis;
        break;
    case Options::ENDCAP_BSPLINE_BASIS:
        table->_localPointStencils = localPointStencils;
        table->_localPointVaryingStencils = localPointVaryingStencils;
        delete endCapBSpline;
        break;
    case Options::ENDCAP_LEGACY_GREGORY:
        endCapLegacyGregory->Finalize(
            table->GetMaxValence(),
            &table->_quadOffsetsTable,
            &table->_vertexValenceTable);
        delete endCapLegacyGregory;
        break;
    default:
        break;
    }

    if (context.RequiresFVarPatches()) {
        table->_localPointFaceVaryingStencils.resize(
                                        context.fvarChannelIndices.size());

        for (int fvc=0; fvc<(int)context.fvarChannelIndices.size(); ++fvc) {
            if (localPointFVarStencils[fvc]->GetNumStencils() > 0) {
                localPointFVarStencils[fvc]->finalize();
            } else {
                delete localPointFVarStencils[fvc];
                localPointFVarStencils[fvc] = NULL;
            }

            switch(context.options.GetEndCapType()) {
            case Options::ENDCAP_GREGORY_BASIS:
                delete fvarEndCapGregoryBasis[fvc];
                break;
            case Options::ENDCAP_BSPLINE_BASIS:
                delete fvarEndCapBSpline[fvc];
                break;
            default:
                break;
            }

            table->_localPointFaceVaryingStencils[fvc] =
                                        localPointFVarStencils[fvc];
        }
    }
}

//
//  Implementation of the PatchFaceTag:
//
void
PatchTableFactory::PatchFaceTag::clear() {
    std::memset(this, 0, sizeof(*this));
}

void
PatchTableFactory::PatchFaceTag::assignTransitionPropertiesFromEdgeMask(int tMask) {
    _transitionMask = tMask;
}

void
PatchTableFactory::PatchFaceTag::assignBoundaryPropertiesFromEdgeMask(int eMask) {

    static int const edgeMaskToCount[16] =
        { 0, 1, 1, 2, 1, -1, 2, -1, 1, 2, -1, -1, 2, -1, -1, -1 };
    static int const edgeMaskToIndex[16] =
        { -1, 0, 1, 1, 2, -1, 2, -1, 3, 0, -1, -1, 3, -1, -1,-1 };

    assert(edgeMaskToCount[eMask] != -1);
    assert(edgeMaskToIndex[eMask] != -1);

    _boundaryMask    = eMask;
    _hasBoundaryEdge = (eMask > 0);

    _boundaryCount = edgeMaskToCount[eMask];
    _boundaryIndex = edgeMaskToIndex[eMask];
}

void
PatchTableFactory::PatchFaceTag::assignBoundaryPropertiesFromVertexMask(int vMask) {

    // This is only intended to support the case of a single boundary vertex with no
    // boundary edges, which can only occur with an irregular vertex

    static int const singleBitVertexMaskToCount[16] =
        { 0, 1, 1, -1, 1, -1 , -1, -1, 1, -1 , -1, -1, -1, -1 , -1, -1 };
    static int const singleBitVertexMaskToIndex[16] =
        { 0, 0, 1, -1, 2, -1 , -1, -1, 3, -1 , -1, -1, -1, -1 , -1, -1 };

    assert(_hasBoundaryEdge == false);
    assert(singleBitVertexMaskToCount[vMask] != -1);
    assert(singleBitVertexMaskToIndex[vMask] != -1);

    _boundaryMask = vMask;

    _boundaryCount = singleBitVertexMaskToCount[vMask];
    _boundaryIndex = singleBitVertexMaskToIndex[vMask];
}

} // end namespace Far

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv

