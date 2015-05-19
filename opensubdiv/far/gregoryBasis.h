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

#pragma once
#ifndef OPENSUBDIV3_FAR_GREGORY_BASIS_H
#define OPENSUBDIV3_FAR_GREGORY_BASIS_H

#include "../far/protoStencil.h"
#include "../vtr/level.h"
#include <cstring>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Far {

class TopologyRefiner;

/// \brief Container for gregory basis stencils
///
/// XXXtakahito: Currently these classes are being used by EndPatch factories.
///              These classes will likely go away once we get limit masks
///              from SchemeWorker.
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

    static const int MAX_VALENCE = (30*2);
    static const int MAX_ELEMS = (16 + MAX_VALENCE);

    // limit valence of 30 because we use a pre-computed closed-form 'ef' table
    // XXXtakahito: revisit here to determine appropriate size
    //              or remove fixed size limit and use Sdc mask
    static int getNumMaxElements(int maxValence) {
        return (16 + maxValence);
    }

    //
    // Basis point
    //
    // Implements arithmetic operators to manipulate the influence of the
    // 1-ring control vertices supporting the patch basis
    //
    class Point {
    public:

        Point() : _size(0) { }

        Point(Index idx, float weight = 1.0f) {
            _size = 1;
            _indices[0] = idx;
            _weights[0] = weight;
        }

        Point(Point const & other) {
            *this = other;
        }

        int GetSize() const {
            return _size;
        }

        Index const * GetIndices() const {
            return _indices;
        }

        float const * GetWeights() const {
            return _weights;
        }

        Point & operator = (Point const & other) {
            _size = other._size;
            memcpy(_indices, other._indices, other._size*sizeof(Index));
            memcpy(_weights, other._weights, other._size*sizeof(float));
            return *this;
        }

        Point & operator += (Point const & other) {
            for (int i=0; i<other._size; ++i) {
                Index idx = findIndex(other._indices[i]);
                _weights[idx] += other._weights[i];
            }
            return *this;
        }

        Point & operator -= (Point const & other) {
            for (int i=0; i<other._size; ++i) {
                Index idx = findIndex(other._indices[i]);
                _weights[idx] -= other._weights[i];
            }
            return *this;
        }

        Point & operator *= (float f) {
            for (int i=0; i<_size; ++i) {
                _weights[i] *= f;
            }
            return *this;
        }

        Point & operator /= (float f) {
            return (*this)*=(1.0f/f);
        }

        friend Point operator * (Point const & src, float f) {
            Point p( src ); return p*=f;
        }

        friend Point operator / (Point const & src, float f) {
            Point p( src ); return p*= (1.0f/f);
        }

        Point operator + (Point const & other) {
            Point p(*this); return p+=other;
        }

        Point operator - (Point const & other) {
            Point p(*this); return p-=other;
        }

        void OffsetIndices(Index offset) {
            for (int i=0; i<_size; ++i) {
                _indices[i] += offset;
            }
        }

        void Copy(int ** size, Index ** indices, float ** weights) const {
            memcpy(*indices, _indices, _size*sizeof(Index));
            memcpy(*weights, _weights, _size*sizeof(float));
            **size = _size;
            *indices += _size;
            *weights += _size;
            ++(*size);
        }

    private:

        int findIndex(Index idx) {
            for (int i=0; i<_size; ++i) {
                if (_indices[i]==idx) {
                    return i;
                }
            }
            _indices[_size]=idx;
            _weights[_size]=0.0f;
            ++_size;
            return _size-1;
        }

        int _size;
        // XXXX this would really be better with VLA where we only allocate
        // space based on the max vertex valence in the mesh, not the
        // absolute maximum supported by the closed-form tangents table.
        Index _indices[MAX_ELEMS];
        float _weights[MAX_ELEMS];
    };

    //
    // ProtoBasis
    //
    // Given a Vtr::Level and a face index, gathers all the influences of the 1-ring
    // that supports the 20 CVs of a Gregory patch basis.
    //
    struct ProtoBasis {

        ProtoBasis(Vtr::Level const & level, Index faceIndex, int fvarChannel=-1);

        int GetNumElements() const;

        void Copy(int * sizes, Index * indices, float * weights) const;

        // Control Vertices based on :
        // "Approximating Subdivision Surfaces with Gregory Patches for Hardware Tessellation"
        // Loop, Schaefer, Ni, Castafio (ACM ToG Siggraph Asia 2009)
        //
        //  P3         e3-      e2+         P2
        //     O--------O--------O--------O
        //     |        |        |        |
        //     |        |        |        |
        //     |        | f3-    | f2+    |
        //     |        O        O        |
        // e3+ O------O            O------O e2-
        //     |     f3+          f2-     |
        //     |                          |
        //     |                          |
        //     |      f0-         f1+     |
        // e0- O------O            O------O e1+
        //     |        O        O        |
        //     |        | f0+    | f1-    |
        //     |        |        |        |
        //     |        |        |        |
        //     O--------O--------O--------O
        //  P0         e0+      e1-         P1
        //

        Point P[4], Ep[4], Em[4], Fp[4], Fm[4];

        // for varying interpolation
        Point V[4];
    };

    typedef std::vector<GregoryBasis::Point> PointsVector;

    static StencilTables *CreateStencilTables(PointsVector const &stencils);

private:

    friend class EndCapGregoryBasisPatchFactory;

    int _sizes[20],
        _offsets[20];

    std::vector<Index> _indices;
    std::vector<float> _weights;
};

} // end namespace Far

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv

#endif /* OPENSUBDIV3_FAR_GREGORY_BASIS_H */
