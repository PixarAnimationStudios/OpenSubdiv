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

#include "../far/gregoryBasis.h"
#include "../far/stencilTable.h"
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

    ///
    /// Multi-patch Gregory stencils factory
    ///

    /// \brief This factory accumulates vertex for Gregory basis patch
    ///
    /// @param refiner                TopologyRefiner from which to generate patches
    ///
    /// @param vertexStencils         Output stencil table for the patch points
    ///                               (vertex interpolation)
    ///
    /// @param varyingStencils        Output stencil table for the patch points
    ///                               (varying interpolation)
    ///
    /// @param shareBoundaryVertices  Use same boundary vertices for neighboring
    ///                               patches. It reduces the number of stencils
    ///                               to be used.
    ///
    EndCapGregoryBasisPatchFactory(TopologyRefiner const & refiner,
                                   StencilTable *vertexStencils,
                                   StencilTable *varyingStencils,
                                   bool shareBoundaryVertices=true);

    /// \brief Returns end patch point indices for \a faceIndex of \a level.
    ///        Note that end patch points are not included in the vertices in
    ///        the topologyRefiner, they're expected to come after the end.
    ///        The returned indices are offset by refiner->GetNumVerticesTotal.
    ///
    /// @param level            vtr refinement level
    ///
    /// @param faceIndex        vtr faceIndex at the level
    //
    /// @param cornerSpans      information about topology for each corner of patch
    /// @param levelVertOffset  relative offset of patch vertex indices
    ///
    /// @param fvarChannel      face-varying channel index
    ///
    ConstIndexArray GetPatchPoints(
        Vtr::internal::Level const * level, Index faceIndex,
        Vtr::internal::Level::VSpan const cornerSpans[],
        int levelVertOffset, int fvarChannel = -1);

private:

    /// Creates a basis for the vertices specified in mask on the face and
    /// accumulates it
    bool addPatchBasis(Vtr::internal::Level const & level, Index faceIndex,
                       Vtr::internal::Level::VSpan const cornerSpans[],
                       bool newVerticesMask[4][5],
                       int levelVertOffset, int fvarChannel);

    StencilTable *_vertexStencils;
    StencilTable *_varyingStencils;

    TopologyRefiner const *_refiner;
    bool _shareBoundaryVertices;
    int _numGregoryBasisVertices;
    int _numGregoryBasisPatches;
    std::vector<Index> _patchPoints;

    //  Only used when sharing vertices:
    std::vector<unsigned int> _levelAndFaceIndices;
};

} // end namespace Far

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv

#endif  // OPENSUBDIV3_FAR_END_CAP_GREGORY_BASIS_PATCH_FACTORY_H
