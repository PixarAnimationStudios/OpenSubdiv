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

#include <vector>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {


/// \brief Stencil descriptor
///
/// Allows access and manipulation of a single stencil in a FarStencilTables.
///
class FarStencil {
public:

    /// \brief Returns the size of the stencil (number of control vertices)
    int GetSize() const {
        return *_size;
    }

    /// \brief Returns the control vertices indices
    int const * GetVertexIndices() const {
        return _indices;
    }

    /// \brief Returns the interpolation weights
    float const * GetValueWeights() const {
        return _point;
    }

    /// \brief Returns U derivative interpolation weights
    float const * GetUDerivWeights() const {
        return _uderiv;
    }

    /// \brief Returns V derivative interpolation weights
    float const * GetVDerivWeights() const {
        return _vderiv;
    }

    /// \brief Increment to the next stencil in the array
    /// Note : there is no array boundary check !
    void Increment() {
        int stride = *_size;
        ++_size;
        _indices += stride;
        _point   += stride;
        _uderiv  += stride;
        _vderiv  += stride;
    }

private:

     FarStencil( int * size,
                 int * indices,
                 float * point,
                 float * uderiv,
                 float * vderiv )
         : _size(size),
           _indices(indices),
           _point(point),
           _uderiv(uderiv),
           _vderiv(vderiv) {
     }

    friend class FarStencilTables;
    template <class T> friend class FarStencilTablesFactory;

    int * _size,
        * _indices;

    float * _point,
          * _uderiv,
          * _vderiv;

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
class FarStencilTables {

public:

    /// \brief Clears the stencils from the table
    void Clear() {
        _sizes.clear();
        _offsets.clear();
        _indices.clear();
        _point.clear();
        _uderiv.clear();
        _vderiv.clear();
    }

    /// \brief Returns the number of stencils in the tables
    int GetNumStencils() const {
        return (int)_sizes.size();
    }

    /// \brief Updates point values based on the control values
    ///
    /// \note The values array is assumed to be at least as big as the
    /// result of \c GetNumStencils().
    template <class T>
    void UpdateValues( T const *controlValues, T *values, int stride=0 ) const {
        _Update( controlValues, &_point.at(0), values, stride );
    }

    /// \brief Updates derivative values based on the control values
    ///
    /// \note The values array is assumed to be at least as big as the
    /// result of \c GetNumStencils().
    template <class T>
    void UpdateDerivs( T const *controlValues, T *uderivs,
                                               T *vderivs, int stride=0 ) const {
        _Update( controlValues, &_uderiv.at(0), uderivs, stride );
        _Update( controlValues, &_vderiv.at(0), vderivs, stride );
    }

    /// \brief Returns a FarStencil at index i in the tables
    FarStencil GetStencil(int i) const;

    /// \brief Returns the number of control vertices of each stencil in the table
    std::vector<int> const & GetSizes() const {
        return _sizes;
    }

    /// \brief Returns the offset to a given stencil
    std::vector<int> const & GetOffsets() const {
        return _offsets;
    }

    /// \brief Returns the indices of the control vertices
    std::vector<int> const & GetControlIndices() const {
        return _indices;
    }

    /// \brief Returns the stencils interpolation weights
    std::vector<float> const & GetWeights() const {
        return _point;
    }

    /// \brief Returns the stencils U deriv weights
    std::vector<float> const & GetDuWeights() const {
        return _uderiv;
    }

    /// \brief Returns the stencils U deriv weights
    std::vector<float> const & GetDvWeights() const {
        return _vderiv;
    }
    
private:

    template <class T> friend class FarStencilTablesFactory;

    // Update values by appling cached stencil weights to new control values
    template <class T> void _Update( T const *controlValues,
                                     float const * weights,
                                     T *values,
                                     int stride ) const;

    std::vector<int>    _sizes;   // number of coeffiecient for each stencil
    std::vector<int>    _offsets; // offset to the start of each stencil
    std::vector<int>    _indices;

    std::vector<float> _point,       // weight coefficients (value & derivatives)
                       _uderiv,
                       _vderiv;

};

template <class T> void
FarStencilTables::_Update( T const *controlValues,
                           float const * weights,
                           T *values,
                           int stride ) const {

    int const * index = &_indices.at(0);

    for (int i=0; i<GetNumStencils(); ++i) {

        // Zero out the result accumulators
        values[i].Clear();

        // For each element in the array, add the coefs contribution
        for (int j=0; j<_sizes[i]; ++j, ++index, ++weights) {
            values[i].AddWithWeight( controlValues[*index], *weights );
        }
    }
}

inline FarStencil
FarStencilTables::GetStencil(int i) const {

    int ofs = _offsets[i];

    return FarStencil( const_cast<int *>(&_sizes[i]),
                       const_cast<int *>(&_indices[ofs]),
                       const_cast<float *>(&_point[ofs]),
                       const_cast<float *>(&_uderiv[ofs]),
                       const_cast<float *>(&_vderiv[ofs]) );
}


} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif // FAR_STENCILTABLES_H
