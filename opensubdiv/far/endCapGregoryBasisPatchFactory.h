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

#ifndef OPENSUBDIV3_FAR_END_CAP_GREGORY_BASIS_PATCH_FACTORY_H
#define OPENSUBDIV3_FAR_END_CAP_GREGORY_BASIS_PATCH_FACTORY_H

#include "../far/patchTableFactory.h"
#include "../far/gregoryBasis.h"
#include "../vtr/level.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Far {

class TopologyRefiner;

/// \brief A specialized factory to gather Gregory basis control vertices
///
/// note: This is an internal use class in PatchTableFactory, and
///       will be replaced with SdcSchemeWorker for mask coefficients
///       and Vtr::Level for topology traversal.
///
class EndCapGregoryBasisPatchFactory {

public:

    //
    // Single patch GregoryBasis basis factory
    //

    /// \brief Instantiates a GregoryBasis from a TopologyRefiner that has been
    ///        refined adaptively for a given face.
    ///
    /// @param refiner     The TopologyRefiner containing the topology
    ///
    /// @param faceIndex   The index of the face (level is assumed to be MaxLevel)
    ///
    /// @param fvarChannel Index of face-varying channel topology (default -1)
    ///
    static GregoryBasis const * Create(TopologyRefiner const & refiner,
        Index faceIndex, int fvarChannel=-1);

public:

    ///
    /// Multi-patch Gregory stencils factory
    ///

    // XXXX need to add support for face-varying channel stencils

    /// \brief This factory accumulates vertex for Gregory basis patch
    ///
    /// @param refiner                TopologyRefiner from which to generate patches
    ///
    /// @param shareBoundaryVertices  Use same boundary vertices for neighboring
    ///                               patches. It reduces the number of stencils
    ///                               to be used.
    ///
    EndCapGregoryBasisPatchFactory(TopologyRefiner const & refiner,
                                   bool shareBoundaryVertices=true);

    /// \brief Returns end patch point indices for \a faceIndex of \a level.
    ///        Note that end patch points are not included in the vertices in
    ///        the topologyRefiner, they're expected to come after the end.
    ///        The returning indices are offsetted by refiner->GetNumVerticesTotal.
    ///
    /// @param level            vtr refinement level
    ///
    /// @param faceIndex        vtr faceIndex at the level
    ///
    /// @param levelPatchTags   Array of patchTags for all faces in the level
    ///
    /// @param levelVertOffset  relative offset of patch vertex indices
    ///
    ConstIndexArray GetPatchPoints(
        Vtr::internal::Level const * level, Index faceIndex,
        PatchTableFactory::PatchFaceTag const * levelPatchTags,
        int levelVertOffset);

    /// \brief Create a StencilTable for end patch points, relative to the max
    ///        subdivision level.
    ///
    StencilTable* CreateVertexStencilTable() const {
        return GregoryBasis::CreateStencilTable(_vertexStencils);
    }

    /// \brief Create a StencilTable for end patch varying primvar.
    ///        This table is used as a convenient way to get varying primvars
    ///        populated on end patch points along with positions.
    ///
    StencilTable* CreateVaryingStencilTable() const {
        return GregoryBasis::CreateStencilTable(_varyingStencils);
    }

private:

    /// Creates a basis for the vertices specified in mask on the face and
    /// accumates it
    bool addPatchBasis(Index faceIndex, bool newVerticesMask[4][5],
                       int levelVertOffset);

    GregoryBasis::PointsVector _vertexStencils;
    GregoryBasis::PointsVector _varyingStencils;

    TopologyRefiner const *_refiner;
    bool _shareBoundaryVertices;
    int _numGregoryBasisVertices;
    int _numGregoryBasisPatches;
    std::vector<Index> _faceIndices;
    std::vector<Index> _patchPoints;
};

} // end namespace Far

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv

#endif  // OPENSUBDIV3_FAR_END_CAP_GREGORY_BASIS_PATCH_FACTORY_H
