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
                           float *dstDu,
                           VertexBufferDescriptor const &dstDuDesc,
                           float *dstDv,
                           VertexBufferDescriptor const &dstDvDesc,
                           const int * sizes,
                           const int * offsets,
                           const int * indices,
                           const float * weights,
                           const float * duWeights,
                           const float * dvWeights,
                           int start, int end) {
    if (end <= start) return true;
    if (srcDesc.length != dstDesc.length) return false;
    if (srcDesc.length != dstDuDesc.length) return false;
    if (srcDesc.length != dstDvDesc.length) return false;

    CpuEvalStencils(src, srcDesc,
                    dst, dstDesc,
                    dstDu, dstDuDesc,
                    dstDv, dstDvDesc,
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
    void AddWithWeight(T const *src, float w, float wu, float wv) {
        (void)wu;
        (void)wv;
        // TODO: derivatives.
        for (int i = 0; i < _length; ++i) {
            _p[i] += src[i] * w;
        }
    }
    const T *operator[] (int index) const {
        return _p + _stride * index;
    }
    BufferAdapter<T> & operator ++() {
        _p += _stride;
        return *this;
    }

    T *_p;
    int _length;
    int _stride;
};

/* static */
int
CpuEvaluator::EvalPatches(const float *src,
                          VertexBufferDescriptor const &srcDesc,
                          float *dst,
                          VertexBufferDescriptor const &dstDesc,
                          PatchCoordArray const &patchCoords,
                          Far::PatchTables const *patchTable) {
    src += srcDesc.offset;
    dst += dstDesc.offset;
    int count = 0;

    // XXX: this implementaion is temporary.
    BufferAdapter<const float> srcT(src, srcDesc.length, srcDesc.stride);
    BufferAdapter<float>       dstT(dst, dstDesc.length, dstDesc.stride);

    for (size_t i = 0; i < patchCoords.size(); ++i) {
        PatchCoord const &coords = patchCoords[i];

        patchTable->Evaluate(coords.handle, coords.s, coords.t,
                             srcT, dstT);
        ++count;
        ++dstT;
    }
    return count;
}


}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
