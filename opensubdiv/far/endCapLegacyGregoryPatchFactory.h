//
//   Copyright 2015 Pixar
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

#ifndef OPENSUBDIV3_FAR_END_CAP_LEGACY_GREGORY_PATCH_FACTORY_H
#define OPENSUBDIV3_FAR_END_CAP_LEGACY_GREGORY_PATCH_FACTORY_H

#include "../far/patchTableFactory.h"
#include "../vtr/level.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {


namespace Far {

class PatchTable;
class TopologyRefiner;

/// \brief    This factory generates legacy (OpenSubdiv 2.x) gregory patches.
///
/// note: This is an internal use class in PatchTableFactory.
///       will be deprecated at some point.
///
class EndCapLegacyGregoryPatchFactory {
public:
    EndCapLegacyGregoryPatchFactory(TopologyRefiner const & refiner);

    /// \brief Returns end patch point indices for \a faceIndex of \a level.
    ///        Note that legacy gregory patch points exist in the max level
    ///        of subdivision in the topologyRefiner.
    ///        The returning indices are offsetted by levelVertOffset
    ///
    /// @param level            vtr refinement level
    ///
    /// @param faceIndex        vtr faceIndex at the level
    ///
    /// @param levelPatchTags   Array of patchTags for all faces in the level
    ///
    /// @param levelVertOffset  relative offset of patch vertex indices
    ///
    ConstIndexArray GetPatchPoints(Vtr::internal::Level const * level, Index faceIndex,
                                   PatchTableFactory::PatchFaceTag const * levelPatchTags,
                                   int levelVertOffset);

    void Finalize(int maxValence, 
                  PatchTable::QuadOffsetsTable *quadOffsetsTable,
                  PatchTable::VertexValenceTable *vertexValenceTable);


private:
    TopologyRefiner const &_refiner;
    std::vector<Index> _gregoryTopology;
    std::vector<Index> _gregoryBoundaryTopology;
    std::vector<Index> _gregoryFaceIndices;
    std::vector<Index> _gregoryBoundaryFaceIndices;
};

} // end namespace Far

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv

#endif  // OPENSUBDIV3_FAR_END_CAP_LEGACY_GREGORY_PATCH_FACTORY_H
