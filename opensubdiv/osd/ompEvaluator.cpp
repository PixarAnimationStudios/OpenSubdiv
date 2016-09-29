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

#include "../osd/ompEvaluator.h"
#include "../osd/ompKernel.h"
#include "../far/patchBasis.h"
#include <omp.h>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

/* static */
bool
OmpEvaluator::EvalStencils(
    const float *src, BufferDescriptor const &srcDesc,
    float *dst,       BufferDescriptor const &dstDesc,
    const int * sizes,
    const int * offsets,
    const int * indices,
    const float * weights,
    int start, int end) {

    if (end <= start) return true;
    if (srcDesc.length != dstDesc.length) return false;

    // XXX: we can probably expand cpuKernel.cpp to here.
    OmpEvalStencils(src, srcDesc, dst, dstDesc,
                    sizes, offsets, indices, weights, start, end);

    return true;
}

/* static */
bool
OmpEvaluator::EvalStencils(
    const float *src, BufferDescriptor const &srcDesc,
    float *dst,       BufferDescriptor const &dstDesc,
    float *du,        BufferDescriptor const &duDesc,
    float *dv,        BufferDescriptor const &dvDesc,
    const int * sizes,
    const int * offsets,
    const int * indices,
    const float * weights,
    const float * duWeights,
    const float * dvWeights,
    int start, int end) {

    if (end <= start) return true;
    if (srcDesc.length != dstDesc.length) return false;
    if (srcDesc.length != duDesc.length) return false;
    if (srcDesc.length != dvDesc.length) return false;

    OmpEvalStencils(src, srcDesc,
                    dst, dstDesc,
                    du,  duDesc,
                    dv,  dvDesc,
                    sizes, offsets, indices,
                    weights, duWeights, dvWeights,
                    start, end);

    return true;
}

template <typename T>
struct BufferAdapter {
    BufferAdapter(T *p, int length, int stride) :
        _p(p), _length(length), _stride(stride) { }
    void Clear() {
        for (int i = 0; i < _length; ++i) _p[i] = 0;
    }
    void AddWithWeight(T const *src, float w) {
        if (_p) {
            // TODO: derivatives.
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

/* static */
bool
OmpEvaluator::EvalPatches(
    const float *src, BufferDescriptor const &srcDesc,
    float *dst,       BufferDescriptor const &dstDesc,
    int numPatchCoords,
    const PatchCoord *patchCoords,
    const PatchArray *patchArrays,
    const int *patchIndexBuffer,
    const PatchParam *patchParamBuffer){

    src += srcDesc.offset;
    if (dst) dst += dstDesc.offset;
    else return false;
    BufferAdapter<const float> srcT(src, srcDesc.length, srcDesc.stride);

#pragma omp parallel for
    for (int i = 0; i < numPatchCoords; ++i) {
        BufferAdapter<float> dstT(dst + dstDesc.stride*i, dstDesc.length, dstDesc.stride);

        float wP[20], wDs[20], wDt[20];
        PatchCoord const &coord = patchCoords[i];
        PatchArray const &array = patchArrays[coord.handle.arrayIndex];

        int patchType = array.GetPatchType();
        // XXX: patchIndex is absolute. not sure it's consistent.
        //      (should be offsetted by array.primitiveIdBase?)
        //    patchParamBuffer[array.primitiveIdBase + coord.handle.patchIndex]
        Far::PatchParam const & param =
            patchParamBuffer[coord.handle.patchIndex];

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
            continue;
        }
        const int *cvs =
            &patchIndexBuffer[array.indexBase + coord.handle.vertIndex];

        dstT.Clear();
        for (int j = 0; j < numControlVertices; ++j) {
            dstT.AddWithWeight(srcT[cvs[j]], wP[j]);
        }
    }
    return true;
}

/* static */
bool
OmpEvaluator::EvalPatches(
    const float *src, BufferDescriptor const &srcDesc,
    float *dst,       BufferDescriptor const &dstDesc,
    float *du,        BufferDescriptor const &duDesc,
    float *dv,        BufferDescriptor const &dvDesc,
    int numPatchCoords,
    PatchCoord const *patchCoords,
    PatchArray const *patchArrays,
    const int *patchIndexBuffer,
    PatchParam const *patchParamBuffer) {

    src += srcDesc.offset;
    if (dst) dst += dstDesc.offset;
    if (du)  du += duDesc.offset;
    if (dv)  dv += dvDesc.offset;

    BufferAdapter<const float> srcT(src, srcDesc.length, srcDesc.stride);

#pragma omp parallel for
    for (int i = 0; i < numPatchCoords; ++i) {
        float wP[20], wDs[20], wDt[20];
        BufferAdapter<float> dstT(dst + dstDesc.stride*i, dstDesc.length, dstDesc.stride);
        BufferAdapter<float> duT(du   + duDesc.stride*i, duDesc.length, duDesc.stride);
        BufferAdapter<float> dvT(dv   + dvDesc.stride*i, dvDesc.length, dvDesc.stride);

        PatchCoord const &coord = patchCoords[i];
        PatchArray const &array = patchArrays[coord.handle.arrayIndex];

        int patchType = array.GetPatchType();
        Far::PatchParam const & param =
            patchParamBuffer[coord.handle.patchIndex];

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
            continue;
        }
        const int *cvs =
            &patchIndexBuffer[array.indexBase + coord.handle.vertIndex];

        dstT.Clear();
        duT.Clear();
        dvT.Clear();
        for (int j = 0; j < numControlVertices; ++j) {
            dstT.AddWithWeight(srcT[cvs[j]], wP[j]);
            duT.AddWithWeight(srcT[cvs[j]], wDs[j]);
            dvT.AddWithWeight(srcT[cvs[j]], wDt[j]);
        }
        ++dstT;
        ++duT;
        ++dvT;
    }
    return true;
}

/* static */
void
OmpEvaluator::Synchronize(void * /*deviceContext*/) {
    // we use "omp parallel for" and it synchronizes by itself
}

/* static */
void
OmpEvaluator::SetNumThreads(int numThreads) {
    omp_set_num_threads(numThreads);
}

}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
