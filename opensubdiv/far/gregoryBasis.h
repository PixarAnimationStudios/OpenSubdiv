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

#ifndef FAR_GREGORY_BASIS_H
#define FAR_GREGORY_BASIS_H

#include "../far/protoStencil.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Far {

class TopologyRefiner;

/// \brief Container for gregory basis stencils
///
class GregoryBasis {

public:

    /// \brief Updates point values based on the control values
    ///
    /// \note The destination buffers are assumed to have allocated at least
    ///       \c GetNumStencils() elements.
    ///
    /// @param controlValues  Buffer with primvar data for the control vertices
    ///
    /// @param values         Destination buffer for the interpolated primvar
    ///                       data
    ///
    template <class T, class U>
    void Evaluate(T const & controlValues, U values[20]) const {

        Index const * indices = &_indices.at(0);
        float const * weights = &_weights.at(0);

        for (int i=0; i<20; ++i) {
            values[i].Clear();
            for (int j=0; j<_sizes[i]; ++j, ++indices, ++weights) {
                values[i].AddWithWeight(controlValues[*indices], *weights);
            }
        }
    }

private:

    friend class GregoryBasisFactory;

    int _sizes[20],
        _offsets[20];

    std::vector<Index> _indices;
    std::vector<float> _weights;
};

/// \brief A specialized factory to gather Gregory basis control vertices
///
class GregoryBasisFactory {

public:

    //
    // Single patch GregoryBasis basis factory
    //

    /// \brief Instantiates a GregoryBasis from a TopologyRefiner that has been
    ///        refined adaptively for a given face.
    ///
    /// @param refiner    The TopologyRefiner containing the topology
    ///
    /// @param faceIndex  The index of the face (level is assumed to be MaxLevel)
    ///
    static GregoryBasis const * Create(TopologyRefiner const & refiner, Index faceIndex);

    /// \brief Returns the maximum valence of a vertex in the mesh that the
    ///        Gregory patches can handle
    static int GetMaxValence();

public:

    //
    // Multi-patch Gregory stencils factory
    //

    // This factory accumulates Gregory patch basis into StencilTables
    //
    // Note: the TopologyRefiner and StencilTables references are held for the
    //       lifespan of the factory - neither can be deleted or modified while
    //       this factory is active.
    //
    GregoryBasisFactory(TopologyRefiner const & refiner,
        StencilTables const & stencils, int numpatches, int maxvalence);

    // Creates a basis for the face and adds it to the stencil pool allocator
    bool AddPatchBasis(Index faceIndex);

    // After all the patches have been collected, create the final table
    StencilTables const * CreateStencilTables(int const permute[20]=0);

private:

    int _currentStencil;

    TopologyRefiner const & _refiner; // XXXX these should be smart pointers !

    Index _stencilsOffset;

    StencilTables const & _stencils;
    StencilAllocator _alloc;
};

} // end namespace Far

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv

#endif /* FAR_GREGORY_BASIS_H */
