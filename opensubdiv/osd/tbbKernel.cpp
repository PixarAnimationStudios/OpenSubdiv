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

#include "../osd/cpuKernel.h"
#include "../osd/tbbKernel.h"
#include "../osd/types.h"
#include "../osd/bufferDescriptor.h"
#include "../far/patchBasis.h"

#include <cassert>
#include <cstdlib>
#include <tbb/parallel_for.h>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

#define grain_size  200

template <class T> T *
elementAtIndex(T * src, int index, BufferDescriptor const &desc) {

    return src + index * desc.stride;
}

static inline void
clear(float *dst, BufferDescriptor const &desc) {

    assert(dst);
    memset(dst, 0, desc.length*sizeof(float));
}

static inline void
addWithWeight(float *dst, const float *src, int srcIndex, float weight,
              BufferDescriptor const &desc) {

    assert(src and dst);
    src = elementAtIndex(src, srcIndex, desc);
    for (int k = 0; k < desc.length; ++k) {
        dst[k] += src[k] * weight;
    }
}

static inline void
copy(float *dst, int dstIndex, const float *src,
     BufferDescriptor const &desc) {

    assert(src and dst);

    dst = elementAtIndex(dst, dstIndex, desc);
    memcpy(dst, src, desc.length*sizeof(float));
}


class TBBStencilKernel {

    BufferDescriptor _srcDesc;
    BufferDescriptor _dstDesc;
    float const * _vertexSrc;
    float * _vertexDst;

    int const * _sizes;
    int const * _offsets,
              * _indices;
    float const * _weights;


public:
    TBBStencilKernel(float const *src, BufferDescriptor srcDesc,
                     float *dst,       BufferDescriptor dstDesc,
                     int const * sizes, int const * offsets,
                     int const * indices, float const * weights) :
         _srcDesc(srcDesc),
         _dstDesc(dstDesc),
         _vertexSrc(src),
         _vertexDst(dst),
         _sizes(sizes),
         _offsets(offsets),
         _indices(indices),
         _weights(weights) { }

    TBBStencilKernel(TBBStencilKernel const & other) {
        _srcDesc    = other._srcDesc;
        _dstDesc    = other._dstDesc;
        _sizes      = other._sizes;
        _offsets    = other._offsets;
        _indices    = other._indices;
        _weights    = other._weights;
        _vertexSrc  = other._vertexSrc;
        _vertexDst  = other._vertexDst;
    }

    void operator() (tbb::blocked_range<int> const &r) const {
#define USE_SIMD
#ifdef USE_SIMD
        if (_srcDesc.length==4 and _srcDesc.stride==4 and _dstDesc.stride==4) {

            // SIMD fast path for aligned primvar data (4 floats)
            int offset = _offsets[r.begin()];
            ComputeStencilKernel<4>(_vertexSrc, _vertexDst,
                _sizes, _indices+offset, _weights+offset, r.begin(), r.end());

        } else if (_srcDesc.length==8 and _srcDesc.stride==4 and _dstDesc.stride==4) {

            // SIMD fast path for aligned primvar data (8 floats)
            int offset = _offsets[r.begin()];
            ComputeStencilKernel<8>(_vertexSrc, _vertexDst,
                _sizes, _indices+offset, _weights+offset, r.begin(), r.end());

        } else {
#else
        {
#endif
            int const * sizes = _sizes;
            int const * indices = _indices;
            float const * weights = _weights;

            if (r.begin()>0) {
                sizes += r.begin();
                indices += _offsets[r.begin()];
                weights += _offsets[r.begin()];
            }

            // Slow path for non-aligned data
            float * result = (float*)alloca(_srcDesc.length * sizeof(float));

            for (int i=r.begin(); i<r.end(); ++i, ++sizes) {

                clear(result, _dstDesc);

                for (int j=0; j<*sizes; ++j) {
                    addWithWeight(result, _vertexSrc, *indices++, *weights++, _srcDesc);
                }

                copy(_vertexDst, i, result, _dstDesc);
            }
        }
    }
};

void
TbbEvalStencils(float const * src, BufferDescriptor const &srcDesc,
                float * dst,       BufferDescriptor const &dstDesc,
                int const * sizes,
                int const * offsets,
                int const * indices,
                float const * weights,
                int start, int end) {

    src += srcDesc.offset;
    dst += dstDesc.offset;

    TBBStencilKernel kernel(src, srcDesc, dst, dstDesc,
                            sizes, offsets, indices, weights);

    tbb::blocked_range<int> range(start, end, grain_size);

    tbb::parallel_for(range, kernel);
}

void
TbbEvalStencils(float const * src, BufferDescriptor const &srcDesc,
                float * dst,       BufferDescriptor const &dstDesc,
                float * du,        BufferDescriptor const &duDesc,
                float * dv,        BufferDescriptor const &dvDesc,
                int const * sizes,
                int const * offsets,
                int const * indices,
                float const * weights,
                float const * duWeights,
                float const * dvWeights,
                int start, int end) {

    if (src) src += srcDesc.offset;
    if (dst) dst += dstDesc.offset;
    if (du)  du  += duDesc.offset;
    if (dv)  dv  += dvDesc.offset;

    // PERFORMANCE: need to combine 3 launches together
    if (dst) {
        TBBStencilKernel kernel(src, srcDesc, dst, dstDesc,
                                sizes, offsets, indices, weights);
        tbb::blocked_range<int> range(start, end, grain_size);
        tbb::parallel_for(range, kernel);
    }

    if (du) {
        TBBStencilKernel kernel(src, srcDesc, du, duDesc,
                                sizes, offsets, indices, duWeights);
        tbb::blocked_range<int> range(start, end, grain_size);
        tbb::parallel_for(range, kernel);
    }

    if (dv) {
        TBBStencilKernel kernel(src, srcDesc, dv, dvDesc,
                                sizes, offsets, indices, dvWeights);
        tbb::blocked_range<int> range(start, end, grain_size);
        tbb::parallel_for(range, kernel);
    }
}

// ---------------------------------------------------------------------------

template <typename T>
struct BufferAdapter {
    BufferAdapter(T *p, int length, int stride) :
        _p(p), _length(length), _stride(stride) { }
    void Clear() {
        for (int i = 0; i < _length; ++i) _p[i] = 0;
    }
    void AddWithWeight(T const *src, float w) {
        if (_p) {
            for (int i = 0; i < _length; ++i) {
                _p[i] += src[i] * w;
            }
        }
    }
    const T *operator[] (int index) const {
        return _p + _stride * index;
    }
    BufferAdapter<T> & operator ++() {
        if (_p) {
            _p += _stride;
        }
        return *this;
    }

    T *_p;
    int _length;
    int _stride;
};

class TbbEvalPatchesKernel {
    BufferDescriptor _srcDesc;
    BufferDescriptor _dstDesc;
    BufferDescriptor _dstDuDesc;
    BufferDescriptor _dstDvDesc;
    float const * _src;
    float * _dst;
    float * _dstDu;
    float * _dstDv;
    int _numPatchCoords;
    const PatchCoord *_patchCoords;
    const PatchArray *_patchArrayBuffer;
    const int        *_patchIndexBuffer;
    const PatchParam *_patchParamBuffer;

public:
    TbbEvalPatchesKernel(float const *src, BufferDescriptor srcDesc,
                         float *dst,       BufferDescriptor dstDesc,
                         float *dstDu,     BufferDescriptor dstDuDesc,
                         float *dstDv,     BufferDescriptor dstDvDesc,
                         int numPatchCoords,
                         const PatchCoord *patchCoords,
                         const PatchArray *patchArrayBuffer,
                         const int *patchIndexBuffer,
                         const PatchParam *patchParamBuffer) :
        _srcDesc(srcDesc), _dstDesc(dstDesc),
        _dstDuDesc(dstDuDesc), _dstDvDesc(dstDvDesc),
        _src(src), _dst(dst), _dstDu(dstDu), _dstDv(dstDv),
        _numPatchCoords(numPatchCoords),
        _patchCoords(patchCoords),
        _patchArrayBuffer(patchArrayBuffer),
        _patchIndexBuffer(patchIndexBuffer),
        _patchParamBuffer(patchParamBuffer) {
    }

    void operator() (tbb::blocked_range<int> const &r) const {
        if (_dstDu == NULL && _dstDv == NULL) {
            compute(r);
        } else {
            computeWithDerivative(r);
        }
    }

    void compute(tbb::blocked_range<int> const &r) const {
        float wP[20], wDs[20], wDt[20];
        BufferAdapter<const float> srcT(_src + _srcDesc.offset,
                                        _srcDesc.length,
                                        _srcDesc.stride);
        BufferAdapter<float> dstT(_dst + _dstDesc.offset
                                       + r.begin() * _dstDesc.stride,
                                  _dstDesc.length,
                                  _dstDesc.stride);

        BufferAdapter<float> dstDuT(_dstDu,
                                    _dstDuDesc.length,
                                    _dstDuDesc.stride);
        BufferAdapter<float> dstDvT(_dstDv,
                                    _dstDvDesc.length,
                                    _dstDvDesc.stride);

        for (int i = r.begin(); i < r.end(); ++i) {
            PatchCoord const &coord = _patchCoords[i];
            PatchArray const &array = _patchArrayBuffer[coord.handle.arrayIndex];

            int patchType = array.GetPatchType();
            Far::PatchParam const & param =
                _patchParamBuffer[coord.handle.patchIndex];

            int numControlVertices = 0;
            if (patchType == Far::PatchDescriptor::REGULAR) {
                Far::internal::GetBSplineWeights(param,
                                                 coord.s, coord.t, wP, wDs, wDt);
                numControlVertices = 16;
            } else if (patchType == Far::PatchDescriptor::GREGORY_BASIS) {
                Far::internal::GetGregoryWeights(param,
                                                 coord.s, coord.t, wP, wDs, wDt);
                numControlVertices = 20;
            } else if (patchType == Far::PatchDescriptor::QUADS) {
                Far::internal::GetBilinearWeights(param,
                                                  coord.s, coord.t, wP, wDs, wDt);
                numControlVertices = 4;
            } else {
                assert(0);
            }

            const int *cvs =
                &_patchIndexBuffer[array.indexBase + coord.handle.vertIndex];

            dstT.Clear();
            for (int j = 0; j < numControlVertices; ++j) {
                dstT.AddWithWeight(srcT[cvs[j]], wP[j]);
            }
            ++dstT;
        }
    }

    void computeWithDerivative(tbb::blocked_range<int> const &r) const {
        float wP[20], wDs[20], wDt[20];
        BufferAdapter<const float> srcT(_src + _srcDesc.offset,
                                        _srcDesc.length,
                                        _srcDesc.stride);
        BufferAdapter<float> dstT(_dst + _dstDesc.offset
                                       + r.begin() * _dstDesc.stride,
                                  _dstDesc.length,
                                  _dstDesc.stride);
        BufferAdapter<float> dstDuT(_dstDu + _dstDuDesc.offset
                                       + r.begin() * _dstDuDesc.stride,
                                  _dstDuDesc.length,
                                  _dstDuDesc.stride);
        BufferAdapter<float> dstDvT(_dstDv + _dstDvDesc.offset
                                       + r.begin() * _dstDvDesc.stride,
                                  _dstDvDesc.length,
                                  _dstDvDesc.stride);

        for (int i = r.begin(); i < r.end(); ++i) {
            PatchCoord const &coord = _patchCoords[i];
            PatchArray const &array = _patchArrayBuffer[coord.handle.arrayIndex];

            int patchType = array.GetPatchType();
            Far::PatchParam const & param =
                _patchParamBuffer[coord.handle.patchIndex];

            int numControlVertices = 0;
            if (patchType == Far::PatchDescriptor::REGULAR) {
                Far::internal::GetBSplineWeights(param,
                                                 coord.s, coord.t, wP, wDs, wDt);
                numControlVertices = 16;
            } else if (patchType == Far::PatchDescriptor::GREGORY_BASIS) {
                Far::internal::GetGregoryWeights(param,
                                                 coord.s, coord.t, wP, wDs, wDt);
                numControlVertices = 20;
            } else if (patchType == Far::PatchDescriptor::QUADS) {
                Far::internal::GetBilinearWeights(param,
                                                  coord.s, coord.t, wP, wDs, wDt);
                numControlVertices = 4;
            } else {
                assert(0);
            }

            const int *cvs =
                &_patchIndexBuffer[array.indexBase + coord.handle.vertIndex];

            dstT.Clear();
            dstDuT.Clear();
            dstDvT.Clear();
            for (int j = 0; j < numControlVertices; ++j) {
                dstT.AddWithWeight(srcT[cvs[j]], wP[j]);
                dstDuT.AddWithWeight(srcT[cvs[j]], wDs[j]);
                dstDvT.AddWithWeight(srcT[cvs[j]], wDt[j]);
            }
            ++dstT;
            ++dstDuT;
            ++dstDvT;
        }
    }
};


void
TbbEvalPatches(float const *src, BufferDescriptor const &srcDesc,
               float *dst,       BufferDescriptor const &dstDesc,
               float *dstDu,     BufferDescriptor const &dstDuDesc,
               float *dstDv,     BufferDescriptor const &dstDvDesc,
               int numPatchCoords,
               const PatchCoord *patchCoords,
               const PatchArray *patchArrayBuffer,
               const int *patchIndexBuffer,
               const PatchParam *patchParamBuffer) {

    TbbEvalPatchesKernel kernel(src, srcDesc, dst, dstDesc,
                                dstDu, dstDuDesc, dstDv, dstDvDesc,
                                numPatchCoords, patchCoords,
                                patchArrayBuffer,
                                patchIndexBuffer,
                                patchParamBuffer);

    tbb::blocked_range<int> range(0, numPatchCoords, grain_size);
    tbb::parallel_for(range, kernel);

}

}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
