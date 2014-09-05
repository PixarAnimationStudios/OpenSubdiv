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

#ifndef FAR_STENCILTABLES_H
#define FAR_STENCILTABLES_H

#include "../version.h"

#include <cassert>
#include <vector>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Far {

/// \brief Vertex stencil descriptor
///
/// Allows access and manipulation of a single stencil in a StencilTables.
///
class Stencil {

public:

    /// \brief Default constructor
    Stencil() {}

    /// \brief Constructor
    ///
    /// @param size     Table pointer to the size of the stencil
    ///
    /// @param indices  Table pointer to the vertex indices of the stencil
    ///
    /// @param weights  Table pointer to the vertex weights of the stencil
    ///
    Stencil(unsigned char * size,
               int * indices,
               float * weights)
        : _size(size),
          _indices(indices),
          _weights(weights) {
    }

    /// \brief Copy constructor
    Stencil(Stencil const & other) {
        _size = other._size;
        _indices = other._indices;
        _weights = other._weights;
    }

    int GetSize() const {
        return *_size;
    }

    /// \brief Returns the control vertices indices
    int const * GetVertexIndices() const {
        return _indices;
    }

    /// \brief Returns the interpolation weights
    float const * GetWeights() const {
        return _weights;
    }

    /// \brief Advance to the next stencil in the table
    void Next() {
        int stride = *_size;
        ++_size;
        _indices += stride;
        _weights += stride;
    }

protected:
    friend class StencilTablesFactory;

    unsigned char * _size;
    int           * _indices;
    float         * _weights;
};

/// \brief Table of subdivision stencils.
///
/// Stencils are the most direct methods of evaluation of locations on the limit
/// of a surface. Every point of a limit surface can be computed by linearly
/// blending a collection of coarse control vertices.
///
/// A stencil assigns a series of control vertex indices with a blending weight
/// that corresponds to a unique parametric location of the limit surface. When
/// the control vertices move in space, the limit location can be very efficiently
/// recomputed simply by applying the blending weights to the series of coarse
/// control vertices.
///
class StencilTables {

public:

    /// \brief Returns the number of stencils in the table
    int GetNumStencils() const {
        return (int)_sizes.size();
    }

    int GetNumControlVertices() const {
        return _numControlVertices;
    }

    /// \brief Returns a Stencil at index i in the tables
    Stencil GetStencil(int i) const;

    /// \brief Returns the number of control vertices of each stencil in the table
    std::vector<unsigned char> const & GetSizes() const {
        return _sizes;
    }

    /// \brief Returns the offset to a given stencil (factory may leave empty)
    std::vector<int> const & GetOffsets() const {
        return _offsets;
    }

    /// \brief Returns the indices of the control vertices
    std::vector<int> const & GetControlIndices() const {
        return _indices;
    }

    /// \brief Returns the stencil interpolation weights
    std::vector<float> const & GetWeights() const {
        return _weights;
    }

    /// \brief Updates point values based on the control values
    ///
    /// \note The destination buffers ('uderivs' & 'vderivs') are assumed to
    ///       have allocated at least \c GetNumStencils() elements.
    ///
    /// @param controlValues  Buffer with primvar data for the control vertices
    ///
    /// @param values         Destination buffer for the interpolated primvar
    ///                       data
    ///
    /// @param start          (skip to )index of first value to update
    ///
    /// @param end            Index of last value to update
    ///    
    template <class T>
    void UpdateValues(T const *controlValues, T *values, int start=-1, int end=-1) const {

        _Update(controlValues, values, _weights, start, end);
    }

private:

    // Update values by appling cached stencil weights to new control values
    template <class T> void _Update( T const *controlValues, T *values,
        std::vector<float> const & valueWeights, int start, int end) const;

private:

    friend class StencilTablesFactory;

    int _numControlVertices;              // number of control vertices

    std::vector<unsigned char> _sizes;    // number of coeffiecient for each stencil
    std::vector<int>           _offsets,  // offset to the start of each stencil
                               _indices;  // indices of contributing coarse vertices
    std::vector<float>         _weights;  // stencil weight coefficients
};


/// \brief Limit point stencil descriptor
///
class LimitStencil : public Stencil {

public:

    /// \brief Constructor
    ///
    /// @param size       Table pointer to the size of the stencil
    ///
    /// @param indices    Table pointer to the vertex indices of the stencil
    ///
    /// @param weights    Table pointer to the vertex weights of the stencil
    ///
    /// @param duWeights  Table pointer to the 'u' derivative weights
    ///
    /// @param dvWeights Table pointer to the 'v' derivative weights
    ///
    LimitStencil( unsigned char * size,
                     int * indices,
                     float * weights,
                     float * duWeights,
                     float * dvWeights )
        : Stencil(size, indices, weights),
          _duWeights(duWeights),
          _dvWeights(dvWeights) {
    }

    /// \brief
    float const * GetDuWeights() const {
        return _duWeights;
    }

    /// \brief
    float const * GetDvWeights() const {
        return _dvWeights;
    }

    /// \brief Advance to the next stencil in the table
    void Next() {
       int stride = *_size;
       ++_size;
       _indices += stride;
       _weights += stride;
       _duWeights += stride;
       _dvWeights += stride;
    }

private:
    float * _duWeights,  // pointer to stencil u derivative limit weights
          * _dvWeights;  // pointer to stencil v derivative limit weights
};

/// \brief Table of limit subdivision stencils.
///
///
class LimitStencilTables : public StencilTables {

public:

    /// \brief Returns the 'u' derivative stencil interpolation weights
    std::vector<float> const & GetDuWeights() const {
        return _duWeights;
    }

    /// \brief Returns the 'v' derivative stencil interpolation weights
    std::vector<float> const & GetDvWeights() const {
        return _dvWeights;
    }

    /// \brief Updates derivative values based on the control values
    ///
    /// \note The destination buffers ('uderivs' & 'vderivs') are assumed to
    ///       have allocated at least \c GetNumStencils() elements.
    ///
    /// @param controlValues  Buffer with primvar data for the control vertices
    ///
    /// @param uderivs        Destination buffer for the interpolated 'u'
    ///                       derivative primvar data
    ///
    /// @param vderivs        Destination buffer for the interpolated 'v'
    ///                       derivative primvar data
    ///
    /// @param start          (skip to )index of first value to update
    ///
    /// @param end            Index of last value to update
    ///
    template <class T>
    void UpdateDerivs(T const *controlValues, T *uderivs, T *vderivs,
        int start=-1, int end=-1) const {

        _Update(controlValues, uderivs, _duWeights, start, end);
        _Update(controlValues, vderivs, _dvWeights, start, end);
    }


private:
    std::vector<float>  _duWeights,  // u derivative limit stencil weights
                        _dvWeights;  // v derivative limit stencil weights
};


// Update values by appling cached stencil weights to new control values
template <class T> void
StencilTables::_Update(T const *controlValues, T *values,
    std::vector<float> const &valueWeights, int start, int end) const {

    int const * indices = &_indices.at(0);
    float const * weights = &valueWeights.at(0);

    if (start>0) {
        assert(start<(int)_offsets.size());
        indices += _offsets[start];
        weights += _offsets[start];
        values += start;
    }

    if (end<start or end<0) {
        end = GetNumStencils();
    }

    int nstencils = end - std::max(0, start);
    for (int i=0; i<nstencils; ++i) {

        // Zero out the result accumulators
        values[i].Clear();

        // For each element in the array, add the coefs contribution
        for (int j=0; j<_sizes[i]; ++j, ++indices, ++weights) {
            values[i].AddWithWeight( controlValues[*indices], *weights );
        }
    }
}

// Returns a Stencil at index i in the table
inline Stencil
StencilTables::GetStencil(int i) const {

    int ofs = _offsets[i];

    return Stencil( const_cast<unsigned char *>(&_sizes[i]),
                       const_cast<int *>(&_indices[ofs]),
                       const_cast<float *>(&_weights[ofs]) );
}

} // end namespace Far

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif // FAR_STENCILTABLES_H
