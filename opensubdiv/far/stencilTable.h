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

#ifndef OPENSUBDIV3_FAR_STENCILTABLE_H
#define OPENSUBDIV3_FAR_STENCILTABLE_H

#include "../version.h"

#include "../far/types.h"

#include <cassert>
#include <cstring>
#include <vector>
#include <iostream>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Far {

/// \brief Vertex stencil descriptor
///
/// Allows access and manipulation of a single stencil in a StencilTable.
///

template<class FD> class StencilTableFactoryG;
template<class FD> class LimitStencilTableFactoryG;
template<class FD> class StencilTableFactoryG;
template<class FD> class PatchTableFactoryG;
template<class FD> class GregoryBasisG;
template<class FD> class EndCapBSplineBasisPatchFactoryG;
template<class FD> class EndCapGregoryBasisPatchFactoryG;

template<class FD>
class StencilG {

public:

    /// \brief Default constructor
    StencilG() {}

    /// \brief Constructor
    ///
    /// @param size     Table pointer to the size of the stencil
    ///
    /// @param indices  Table pointer to the vertex indices of the stencil
    ///
    /// @param weights  Table pointer to the vertex weights of the stencil
    ///
    StencilG(int * size,
            Index * indices,
            FD * weights)
        : _size(size),
          _indices(indices),
          _weights(weights) {
    }

    /// \brief Copy constructor
    StencilG(StencilG const & other) {
        _size = other._size;
        _indices = other._indices;
        _weights = other._weights;
    }

    /// \brief Returns the size of the stencil
    int GetSize() const {
        return *_size;
    }

    /// \brief Returns the size of the stencil as a pointer
    int * GetSizePtr() const {
        return _size;
    }

    /// \brief Returns the control vertices' indices
    Index const * GetVertexIndices() const {
        return _indices;
    }

    /// \brief Returns the interpolation weights
    FD const * GetWeights() const {
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
    friend StencilTableFactoryG<FD>;
    friend LimitStencilTableFactoryG<FD>;

    int * _size;
    Index         * _indices;
    FD         * _weights;
};

/// \brief Table of subdivision stencils.
///
/// Stencils are the most direct method of evaluation of locations on the limit
/// of a surface. Every point of a limit surface can be computed by linearly
/// blending a collection of coarse control vertices.
///
/// A stencil assigns a series of control vertex indices with a blending weight
/// that corresponds to a unique parametric location of the limit surface. When
/// the control vertices move in space, the limit location can be very efficiently
/// recomputed simply by applying the blending weights to the series of coarse
/// control vertices.
///
template<class FD>
class StencilTableG {
    StencilTableG(int numControlVerts,
                    std::vector<int> const& offsets,
                    std::vector<int> const& sizes,
                    std::vector<int> const& sources,
                    std::vector<FD> const& weights,
                    bool includeCoarseVerts,
                    size_t firstOffset);

public:

    virtual ~StencilTableG() {};
    
    /// \brief Returns the number of stencils in the table
    int GetNumStencils() const {
        return (int)_sizes.size();
    }

    /// \brief Returns the number of control vertices indexed in the table
    int GetNumControlVertices() const {
        return _numControlVertices;
    }

    /// \brief Returns a Stencil at index i in the table
    StencilG<FD> GetStencil(Index i) const;

    /// \brief Returns the number of control vertices of each stencil in the table
    std::vector<int> const & GetSizes() const {
        return _sizes;
    }

    /// \brief Returns the offset to a given stencil (factory may leave empty)
    std::vector<Index> const & GetOffsets() const {
        return _offsets;
    }

    /// \brief Returns the indices of the control vertices
    std::vector<Index> const & GetControlIndices() const {
        return _indices;
    }

    /// \brief Returns the stencil interpolation weights
    std::vector<FD> const & GetWeights() const {
        return _weights;
    }

    /// \brief Returns the stencil at index i in the table
    StencilG<FD> operator[] (Index index) const;

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
    /// @param start          index of first value to update
    ///
    /// @param end            Index of last value to update
    ///
    template <class T>
    void UpdateValues(T const *controlValues, T *values, Index start=-1, Index end=-1) const {
        update(controlValues, values, _weights, start, end);
    }

    /// \brief Clears the stencils from the table
    void Clear();

protected:

    // Update values by applying cached stencil weights to new control values
    template <class T> void update( T const *controlValues, T *values,
        std::vector<FD> const & valueWeights, Index start, Index end) const;

    // Populate the offsets table from the stencil sizes in _sizes (factory helper)
    void generateOffsets();

    // Resize the table arrays (factory helper)
    void resize(int nstencils, int nelems);

    // Reserves the table arrays (factory helper)
    void reserve(int nstencils, int nelems);

    // Reallocates the table arrays to remove excess capacity (factory helper)
    void shrinkToFit();

    // Performs any final operations on internal tables (factory helper)
    void finalize();

protected:
    StencilTableG() : _numControlVertices(0) {}
    StencilTableG(int numControlVerts)
        : _numControlVertices(numControlVerts) 
    { }

    friend StencilTableFactoryG<FD>;
    friend PatchTableFactoryG<FD>;
    // XXX: temporarily, GregoryBasis class will go away.
    friend GregoryBasisG<FD>;
    // XXX: needed to call reserve().
    friend EndCapBSplineBasisPatchFactoryG<FD>;
    friend EndCapGregoryBasisPatchFactoryG<FD>;

    int _numControlVertices;              // number of control vertices

    std::vector<int>           _sizes;    // number of coefficients for each stencil
    std::vector<Index>         _offsets,  // offset to the start of each stencil
                               _indices;  // indices of contributing coarse vertices
    std::vector<FD>         _weights;  // stencil weight coefficients
};


/// \brief Limit point stencil descriptor
///
template<class FD>
class LimitStencilG : public StencilG<FD> {

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
    /// @param dvWeights  Table pointer to the 'v' derivative weights
    ///
    /// @param duuWeights Table pointer to the 'uu' derivative weights
    ///
    /// @param duvWeights Table pointer to the 'uv' derivative weights
    ///
    /// @param dvvWeights Table pointer to the 'vv' derivative weights
    ///
    LimitStencilG( int* size,
                  Index * indices,
                  FD * weights,
                  FD * duWeights=0,
                  FD * dvWeights=0,
                  FD * duuWeights=0,
                  FD * duvWeights=0,
                  FD * dvvWeights=0)
        : StencilG<FD>(size, indices, weights),
          _duWeights(duWeights),
          _dvWeights(dvWeights),
          _duuWeights(duuWeights),
          _duvWeights(duvWeights),
          _dvvWeights(dvvWeights) {
    }

    /// \brief Returns the u derivative weights
    FD const * GetDuWeights() const {
        return _duWeights;
    }

    /// \brief Returns the v derivative weights
    FD const * GetDvWeights() const {
        return _dvWeights;
    }

    /// \brief Returns the uu derivative weights
    FD const * GetDuuWeights() const {
        return _duuWeights;
    }

    /// \brief Returns the uv derivative weights
    FD const * GetDuvWeights() const {
        return _duvWeights;
    }

    /// \brief Returns the vv derivative weights
    FD const * GetDvvWeights() const {
        return _dvvWeights;
    }

    /// \brief Advance to the next stencil in the table
    void Next() {
       int stride = *StencilG<FD>::_size;
       ++StencilG<FD>::_size;
       StencilG<FD>::_indices += stride;
       StencilG<FD>::_weights += stride;
       if (_duWeights) _duWeights += stride;
       if (_dvWeights) _dvWeights += stride;
       if (_duuWeights) _duuWeights += stride;
       if (_duvWeights) _duvWeights += stride;
       if (_dvvWeights) _dvvWeights += stride;
    }

private:

    friend StencilTableFactoryG<FD>;
    friend LimitStencilTableFactoryG<FD>;

    FD * _duWeights,  // pointer to stencil u derivative limit weights
       * _dvWeights,  // pointer to stencil v derivative limit weights
       * _duuWeights, // pointer to stencil uu derivative limit weights
       * _duvWeights, // pointer to stencil uv derivative limit weights
       * _dvvWeights; // pointer to stencil vv derivative limit weights
};

/// \brief Table of limit subdivision stencils.
///
///
template<class FD>
class LimitStencilTableG : public StencilTableG<FD> {
    LimitStencilTableG(int numControlVerts,
                    std::vector<int> const& offsets,
                    std::vector<int> const& sizes,
                    std::vector<int> const& sources,
                    std::vector<FD> const& weights,
                    std::vector<FD> const& duWeights,
                    std::vector<FD> const& dvWeights,
                    std::vector<FD> const& duuWeights,
                    std::vector<FD> const& duvWeights,
                    std::vector<FD> const& dvvWeights,
                    bool includeCoarseVerts,
                    size_t firstOffset);

public:

    /// \brief Returns a LimitStencil at index i in the table
    LimitStencilG<FD> GetLimitStencil(Index i) const;

    /// \brief Returns the limit stencil at index i in the table
    LimitStencilG<FD> operator[] (Index index) const;

    /// \brief Returns the 'u' derivative stencil interpolation weights
    std::vector<FD> const & GetDuWeights() const {
        return _duWeights;
    }

    /// \brief Returns the 'v' derivative stencil interpolation weights
    std::vector<FD> const & GetDvWeights() const {
        return _dvWeights;
    }

    /// \brief Returns the 'uu' derivative stencil interpolation weights
    std::vector<FD> const & GetDuuWeights() const {
        return _duuWeights;
    }

    /// \brief Returns the 'uv' derivative stencil interpolation weights
    std::vector<FD> const & GetDuvWeights() const {
        return _duvWeights;
    }

    /// \brief Returns the 'vv' derivative stencil interpolation weights
    std::vector<FD> const & GetDvvWeights() const {
        return _dvvWeights;
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
    /// @param start          index of first value to update
    ///
    /// @param end            Index of last value to update
    ///
    template <class T>
    void UpdateDerivs(T const *controlValues, T *uderivs, T *vderivs,
        int start=-1, int end=-1) const {

        this->update(controlValues, uderivs, _duWeights, start, end);
        this->update(controlValues, vderivs, _dvWeights, start, end);
    }

    /// \brief Updates 2nd derivative values based on the control values
    ///
    /// \note The destination buffers ('uuderivs', 'uvderivs', & 'vderivs') are
    ///       assumed to have allocated at least \c GetNumStencils() elements.
    ///
    /// @param controlValues  Buffer with primvar data for the control vertices
    ///
    /// @param uuderivs       Destination buffer for the interpolated 'uu'
    ///                       derivative primvar data
    ///
    /// @param uvderivs       Destination buffer for the interpolated 'uv'
    ///                       derivative primvar data
    ///
    /// @param vvderivs       Destination buffer for the interpolated 'vv'
    ///                       derivative primvar data
    ///
    /// @param start          index of first value to update
    ///
    /// @param end            Index of last value to update
    ///
    template <class T>
    void Update2ndDerivs(T const *controlValues, T *uuderivs, T *uvderivs, T *vvderivs,
        int start=-1, int end=-1) const {

        this->update(controlValues, uuderivs, _duuWeights, start, end);
        this->update(controlValues, uvderivs, _duvWeights, start, end);
        this->update(controlValues, vvderivs, _dvvWeights, start, end);
    }

    /// \brief Clears the stencils from the table
    void Clear();

private:
    friend LimitStencilTableFactoryG<FD>;

    // Resize the table arrays (factory helper)
    void resize(int nstencils, int nelems);

private:
    std::vector<FD>  _duWeights,   // u  derivative limit stencil weights
                        _dvWeights,   // v  derivative limit stencil weights
                        _duuWeights,  // uu derivative limit stencil weights
                        _duvWeights,  // uv derivative limit stencil weights
                        _dvvWeights;  // vv derivative limit stencil weights

    using StencilTableG<FD>::_offsets;
    using StencilTableG<FD>::_sizes;
    using StencilTableG<FD>::_indices;
    using StencilTableG<FD>::_weights;
};


// Update values by applying cached stencil weights to new control values
template <class FD>
template <class T>
void
StencilTableG<FD>::update(T const *controlValues, T *values,
    std::vector<FD> const &valueWeights, Index start, Index end) const {

    int const * sizes = &_sizes.at(0);
    Index const * indices = &_indices.at(0);
    FD const * weights = &valueWeights.at(0);

    if (start>0) {
        assert(start<(Index)_offsets.size());
        sizes += start;
        indices += _offsets[start];
        weights += _offsets[start];
        values += start;
    }

    if (end<start || end<0) {
        end = GetNumStencils();
    }

    int nstencils = end - std::max(0, start);
    for (int i=0; i<nstencils; ++i, ++sizes) {

        // Zero out the result accumulators
        values[i].Clear();

        // For each element in the array, add the coef's contribution
        for (int j=0; j<*sizes; ++j, ++indices, ++weights) {
            values[i].AddWithWeight( controlValues[*indices], *weights );
        }
    }
}

template<class FD>
inline void
StencilTableG<FD>::generateOffsets() {
    Index offset=0;
    int noffsets = (int)_sizes.size();
    _offsets.resize(noffsets);
    for (int i=0; i<(int)_sizes.size(); ++i ) {
        _offsets[i]=offset;
        offset+=_sizes[i];
    }
}

template<class FD>
inline void
StencilTableG<FD>::resize(int nstencils, int nelems) {
    _sizes.resize(nstencils);
    _indices.resize(nelems);
    _weights.resize(nelems);
}

template<class FD>
inline void
StencilTableG<FD>::reserve(int nstencils, int nelems) {
    _sizes.reserve(nstencils);
    _indices.reserve(nelems);
    _weights.reserve(nelems);
}

template<class FD>
inline void
StencilTableG<FD>::shrinkToFit() {
    std::vector<int>(_sizes).swap(_sizes);
    std::vector<Index>(_indices).swap(_indices);
    std::vector<FD>(_weights).swap(_weights);
}

template<class FD>
inline void
StencilTableG<FD>::finalize() {
    shrinkToFit();
    generateOffsets();
}

// Returns a Stencil at index i in the table
template<class FD>
inline StencilG<FD>
StencilTableG<FD>::GetStencil(Index i) const {
    assert((! _offsets.empty()) && i<(int)_offsets.size());

    Index ofs = _offsets[i];

    return StencilG<FD>( const_cast<int*>(&_sizes[i]),
                    const_cast<Index *>(&_indices[ofs]),
                    const_cast<FD *>(&_weights[ofs]) );
}

template<class FD>
inline StencilG<FD>
StencilTableG<FD>::operator[] (Index index) const {
    return GetStencil(index);
}

template<class FD>
inline void
LimitStencilTableG<FD>::resize(int nstencils, int nelems) {
    StencilTableG<FD>::resize(nstencils, nelems);
    _duWeights.resize(nelems);
    _dvWeights.resize(nelems);
}

// Returns a LimitStencil at index i in the table
template<class FD>
inline LimitStencilG<FD>
LimitStencilTableG<FD>::GetLimitStencil(Index i) const {
    assert((! StencilTableG<FD>::GetOffsets().empty()) && i<(int)StencilTableG<FD>::GetOffsets().size());

    Index ofs = StencilTableG<FD>::GetOffsets()[i];

    if (!_duWeights.empty() && !_dvWeights.empty() &&
        !_duuWeights.empty() && !_duvWeights.empty() && !_dvvWeights.empty()) {
        return LimitStencilG<FD>( const_cast<int *>(&StencilTableG<FD>::GetSizes()[i]),
                             const_cast<Index *>(&StencilTableG<FD>::GetControlIndices()[ofs]),
                             const_cast<FD *>(&StencilTableG<FD>::GetWeights()[ofs]),
                             const_cast<FD *>(&GetDuWeights()[ofs]),
                             const_cast<FD *>(&GetDvWeights()[ofs]),
                             const_cast<FD *>(&GetDuuWeights()[ofs]),
                             const_cast<FD *>(&GetDuvWeights()[ofs]),
                             const_cast<FD *>(&GetDvvWeights()[ofs]) );
    } else if (!_duWeights.empty() && !_dvWeights.empty()) {
        return LimitStencilG<FD>( const_cast<int *>(&StencilTableG<FD>::GetSizes()[i]),
                             const_cast<Index *>(&StencilTableG<FD>::GetControlIndices()[ofs]),
                             const_cast<FD *>(&StencilTableG<FD>::GetWeights()[ofs]),
                             const_cast<FD *>(&GetDuWeights()[ofs]),
                             const_cast<FD *>(&GetDvWeights()[ofs]) );
    } else {
        return LimitStencilG<FD>( const_cast<int *>(&StencilTableG<FD>::GetSizes()[i]),
                             const_cast<Index *>(&StencilTableG<FD>::GetControlIndices()[ofs]),
                             const_cast<FD *>(&StencilTableG<FD>::GetWeights()[ofs]) );
    }
}

template<class FD>
inline LimitStencilG<FD>
LimitStencilTableG<FD>::operator[] (Index index) const {
    return GetLimitStencil(index);
}

typedef StencilG<float> Stencil;
typedef LimitStencilG<float> LimitStencil;
typedef StencilTableG<float> StencilTable;
typedef LimitStencilTableG<float> LimitStencilTable;

typedef StencilG<double> StencilDbl;
typedef LimitStencilG<double> LimitStencilDbl;
typedef StencilTableG<double> StencilTableDbl;
typedef LimitStencilTableG<double> LimitStencilTableDbl;


} // end namespace Far

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif // OPENSUBDIV3_FAR_STENCILTABLE_H
