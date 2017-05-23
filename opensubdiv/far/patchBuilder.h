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

#ifndef OPENSUBDIV3_FAR_PATCH_BUILDER_H
#define OPENSUBDIV3_FAR_PATCH_BUILDER_H

#include "../version.h"

#include "../far/types.h"
#include "../vtr/level.h"

#include <vector>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Far {

class TopologyRefiner;

namespace internal {

// PatchBuilder
//
// Private helper class aggregating transient contextual data structures during
// the creation of feature-adaptive patches (this helps keeping the factory
// classes stateless).
//
class PatchBuilder {

public:

    PatchBuilder(TopologyRefiner const & refiner,
        int numFVarChannels, int const * fvarChannelIndices,
            bool useInfSharpPatch, bool generateLegacySharpCornerPatches);

    TopologyRefiner const & GetTopologyRefiner() const {
        return _refiner;
    }

    std::vector<int> const & GetFVarChannelsIndices() const {
        return _fvarChannelIndices;
    }
    bool UseInfSharpPatch() const {
        return _useInfSharpPatch;
    }

    bool GenerateLegacySharpCornerPatches() const {
        return _generateLegacySharpCornerPatches;
    }

    Vtr::internal::Level const & GetVtrLevel(int levelIndex) const;


public:

    // Methods to query patch properties for classification and construction.
    bool IsPatchEligible(int levelIndex, Index faceIndex) const;

    bool IsPatchSmoothCorner(int levelIndex, Index faceIndex, int fvcFactory = -1) const;

    bool IsPatchRegular(int levelIndex, Index faceIndex, int fvcFactory = -1) const;

    int GetRegularPatchBoundaryMask(int levelIndex, Index faceIndex, int fvcFactory = -1) const;

    void GetIrregularPatchCornerSpans(int levelIndex, Index faceIndex,
        Vtr::internal::Level::VSpan cornerSpans[4], int fvcFactory = -1) const;

public:

    // Additional simple queries -- most regarding face-varying channels that hide
    // the mapping between channels in the source Refiner and corresponding channels
    // in the Factory and PatchTable

    // True if face-varying patches need to be generated for this topology
    bool RequiresFVarPatches() const { return (!_fvarChannelIndices.empty()); }

    int GetRefinerFVarChannel(int fvcFactory) const;

    bool DoesFaceVaryingPatchMatch(int levelIndex, Index faceIndex, int fvcFactory) const;

    int GetDistinctRefinerFVarChannel(int levelIndex, Index faceIndex, int fvcFactory) const;

    int GetTransitionMask(int levelIndex, Index faceIndex) const;

public:

    static inline void OffsetAndPermuteIndices(Index const indices[],
        int count, Index offset, int const permutation[], Index result[]);

protected:

    TopologyRefiner const & _refiner;

    // These are the indices of face-varying channels in the refiner
    // or empty if we are not populating face-varying data.
    std::vector<int> _fvarChannelIndices;

    bool _useInfSharpPatch,                  // Use infinitely-sharp patch
         _generateLegacySharpCornerPatches;  // Generate sharp regular patches at smooth corners (legacy)
};

void
PatchBuilder::OffsetAndPermuteIndices(Index const indices[], int count,
                        Index offset, int const permutation[],
                        Index result[]) {

    // The patch vertices for boundary and corner patches
    // are assigned index values even though indices will
    // be undefined along boundary and corner edges.
    // When the resulting patch table is going to be used
    // as indices for drawing, it is convenient for invalid
    // indices to be replaced with known good values, such
    // as the first un-permuted index, which is the index
    // of the first vertex of the patch face.
    Index knownGoodIndex = indices[0];

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
        std::memcpy(result, indices, count * sizeof(Index));
    }
}

} // end namespace internal

} // end namespace Far

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv


#endif /* OPENSUBDIV3_FAR_PATCH_FACE_TAG_H */
