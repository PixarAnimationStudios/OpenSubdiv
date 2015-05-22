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

#include "../osd/cpuEvaluator.h"
#include "../osd/cpuKernel.h"

#include <cstdlib>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

/* static */
bool
CpuEvaluator::EvalStencils(const float *src,
                           VertexBufferDescriptor const &srcDesc,
                           float *dst,
                           VertexBufferDescriptor const &dstDesc,
                           const int * sizes,
                           const int * offsets,
                           const int * indices,
                           const float * weights,
                           int start, int end) {
    if (end <= start) return true;
    if (srcDesc.length != dstDesc.length) return false;

    // XXX: we can probably expand cpuKernel.cpp to here.
    CpuEvalStencils(src, srcDesc, dst, dstDesc,
                    sizes, offsets, indices, weights, start, end);

    return true;
}

/* static */
bool
CpuEvaluator::EvalStencils(const float *src,
                           VertexBufferDescriptor const &srcDesc,
                           float *dst,
                           VertexBufferDescriptor const &dstDesc,
                           float *dstDs,
                           VertexBufferDescriptor const &dstDsDesc,
                           float *dstDt,
                           VertexBufferDescriptor const &dstDtDesc,
                           const int * sizes,
                           const int * offsets,
                           const int * indices,
                           const float * weights,
                           const float * duWeights,
                           const float * dvWeights,
                           int start, int end) {
    if (end <= start) return true;
    if (srcDesc.length != dstDesc.length) return false;
    if (srcDesc.length != dstDsDesc.length) return false;
    if (srcDesc.length != dstDtDesc.length) return false;

    CpuEvalStencils(src, srcDesc,
                    dst, dstDesc,
                    dstDs, dstDsDesc,
                    dstDt, dstDtDesc,
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
CpuEvaluator::EvalPatches(const float *src,
                          VertexBufferDescriptor const &srcDesc,
                          float *dst,
                          VertexBufferDescriptor const &dstDesc,
                          int numPatchCoords,
                          PatchCoord const *patchCoords,
                          Far::PatchTables const *patchTable) {
    src += srcDesc.offset;
    if (dst) dst += dstDesc.offset;

    BufferAdapter<const float> srcT(src, srcDesc.length, srcDesc.stride);
    BufferAdapter<float>       dstT(dst, dstDesc.length, dstDesc.stride);

    float wP[20], wDs[20], wDt[20];

    for (int i = 0; i < numPatchCoords; ++i) {
        PatchCoord const &coords = patchCoords[i];

        patchTable->EvaluateBasis(coords.handle, coords.s, coords.t, wP, wDs, wDt);

        Far::ConstIndexArray cvs = patchTable->GetPatchVertices(coords.handle);

        dstT.Clear();
        for (int j = 0; j < cvs.size(); ++j) {
            dstT.AddWithWeight(srcT[cvs[j]], wP[j]);
        }
        ++dstT;
    }
    return true;
}

/* static */
bool
CpuEvaluator::EvalPatches(const float *src,
                          VertexBufferDescriptor const &srcDesc,
                          float *dst,
                          VertexBufferDescriptor const &dstDesc,
                          float *dstDs,
                          VertexBufferDescriptor const &dstDsDesc,
                          float *dstDt,
                          VertexBufferDescriptor const &dstDtDesc,
                          int numPatchCoords,
                          PatchCoord const *patchCoords,
                          Far::PatchTables const *patchTable) {
    src += srcDesc.offset;
    if (dst) dst += dstDesc.offset;
    if (dstDs) dstDs += dstDsDesc.offset;
    if (dstDt) dstDt += dstDtDesc.offset;

    BufferAdapter<const float> srcT(src, srcDesc.length, srcDesc.stride);
    BufferAdapter<float> dstT(dst, dstDesc.length, dstDesc.stride);
    BufferAdapter<float> dstDsT(dstDs, dstDsDesc.length, dstDsDesc.stride);
    BufferAdapter<float> dstDtT(dstDt, dstDtDesc.length, dstDtDesc.stride);

    float wP[20], wDs[20], wDt[20];

    for (int i = 0; i < numPatchCoords; ++i) {
        PatchCoord const &coords = patchCoords[i];

        patchTable->EvaluateBasis(coords.handle, coords.s, coords.t, wP, wDs, wDt);

        Far::ConstIndexArray cvs = patchTable->GetPatchVertices(coords.handle);

        dstT.Clear();
        dstDsT.Clear();
        dstDtT.Clear();
        for (int j = 0; j < cvs.size(); ++j) {
            dstT.AddWithWeight(srcT[cvs[j]], wP[j]);
            dstDsT.AddWithWeight(srcT[cvs[j]], wDs[j]);
            dstDtT.AddWithWeight(srcT[cvs[j]], wDt[j]);
        }
        ++dstT;
        ++dstDsT;
        ++dstDtT;
    }
    return true;
}


}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
