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

#include "../osd/cudaEvaluator.h"

#include <cuda_runtime.h>
#include <vector>

#include "../far/stencilTable.h"
#include "../osd/types.h"

extern "C" {
    void CudaEvalStencils(const float *src,
                          float *dst,
                          int length,
                          int srcStride,
                          int dstStride,
                          const int * sizes,
                          const int * offsets,
                          const int * indices,
                          const float * weights,
                          int start,
                          int end);

    void CudaEvalPatches(
        const float *src, float *dst,
        int length, int srcStride, int dstStride,
        int numPatchCoords,
        const void *patchCoords,
        const void *patchArrays,
        const int *patchIndices,
        const void *patchParams);

    void CudaEvalPatchesWithDerivatives(
        const float *src, float *dst, float *du, float *dv,
        int length,
        int srcStride, int dstStride, int dvStride, int duStride,
        int numPatchCoords,
        const void *patchCoords,
        const void *patchArrays,
        const int *patchIndices,
        const void *patchParams);
}

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

template <class T> void *
createCudaBuffer(std::vector<T> const & src) {
    void * devicePtr = 0;

    size_t size = src.size()*sizeof(T);

    cudaError_t err = cudaMalloc(&devicePtr, size);
    if (err != cudaSuccess) {
        return devicePtr;
    }

    err = cudaMemcpy(devicePtr, &src.at(0), size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(devicePtr);
        return 0;
    }
    return devicePtr;
}

// ----------------------------------------------------------------------------

CudaStencilTable::CudaStencilTable(Far::StencilTable const *stencilTable) {
    _numStencils = stencilTable->GetNumStencils();
    if (_numStencils > 0) {
        _sizes   = createCudaBuffer(stencilTable->GetSizes());
        _offsets = createCudaBuffer(stencilTable->GetOffsets());
        _indices = createCudaBuffer(stencilTable->GetControlIndices());
        _weights = createCudaBuffer(stencilTable->GetWeights());
        _duWeights = _dvWeights = NULL;
    } else {
        _sizes = _offsets = _indices = _weights = NULL;
        _duWeights = _dvWeights = NULL;
    }
}

CudaStencilTable::CudaStencilTable(Far::LimitStencilTable const *limitStencilTable) {
    _numStencils = limitStencilTable->GetNumStencils();
    if (_numStencils > 0) {
        _sizes   = createCudaBuffer(limitStencilTable->GetSizes());
        _offsets = createCudaBuffer(limitStencilTable->GetOffsets());
        _indices = createCudaBuffer(limitStencilTable->GetControlIndices());
        _weights = createCudaBuffer(limitStencilTable->GetWeights());
        _duWeights = createCudaBuffer(limitStencilTable->GetDuWeights());
        _dvWeights = createCudaBuffer(limitStencilTable->GetDvWeights());
    } else {
        _sizes = _offsets = _indices = _weights = NULL;
        _duWeights = _dvWeights = NULL;
    }
}

CudaStencilTable::~CudaStencilTable() {
    if (_sizes)   cudaFree(_sizes);
    if (_offsets) cudaFree(_offsets);
    if (_indices) cudaFree(_indices);
    if (_weights) cudaFree(_weights);
    if (_duWeights) cudaFree(_duWeights);
    if (_dvWeights) cudaFree(_dvWeights);
}

// ---------------------------------------------------------------------------

/* static */
bool
CudaEvaluator::EvalStencils(const float *src, BufferDescriptor const &srcDesc,
                            float *dst,       BufferDescriptor const &dstDesc,
                            const int * sizes,
                            const int * offsets,
                            const int * indices,
                            const float * weights,
                            int start,
                            int end) {
    if (dst == NULL) return false;

    CudaEvalStencils(src + srcDesc.offset,
                     dst + dstDesc.offset,
                     srcDesc.length,
                     srcDesc.stride,
                     dstDesc.stride,
                     sizes, offsets, indices, weights,
                     start, end);
    return true;
}

/* static */
bool
CudaEvaluator::EvalStencils(const float *src, BufferDescriptor const &srcDesc,
                            float *dst,       BufferDescriptor const &dstDesc,
                            float *du,        BufferDescriptor const &duDesc,
                            float *dv,        BufferDescriptor const &dvDesc,
                            const int * sizes,
                            const int * offsets,
                            const int * indices,
                            const float * weights,
                            const float * duWeights,
                            const float * dvWeights,
                            int start,
                            int end) {
    // PERFORMANCE: need to combine 3 launches together
    if (dst) {
        CudaEvalStencils(src + srcDesc.offset,
                         dst + dstDesc.offset,
                         srcDesc.length,
                         srcDesc.stride,
                         dstDesc.stride,
                         sizes, offsets, indices, weights,
                         start, end);
    }
    if (du) {
        CudaEvalStencils(src + srcDesc.offset,
                         du +  duDesc.offset,
                         srcDesc.length,
                         srcDesc.stride,
                         duDesc.stride,
                         sizes, offsets, indices, duWeights,
                         start, end);
    }
    if (dv) {
        CudaEvalStencils(src + srcDesc.offset,
                         dv  + dvDesc.offset,
                         srcDesc.length,
                         srcDesc.stride,
                         dvDesc.stride,
                         sizes, offsets, indices, dvWeights,
                         start, end);
    }
    return true;
}

/* static */
bool
CudaEvaluator::EvalPatches(const float *src,
                           BufferDescriptor const &srcDesc,
                           float *dst,
                           BufferDescriptor const &dstDesc,
                           int numPatchCoords,
                           const PatchCoord *patchCoords,
                           const PatchArray *patchArrays,
                           const int *patchIndices,
                           const PatchParam *patchParams) {
    if (src) src += srcDesc.offset;
    if (dst) dst += dstDesc.offset;

    CudaEvalPatches(src, dst,
                    srcDesc.length, srcDesc.stride, dstDesc.stride,
                    numPatchCoords, patchCoords, patchArrays, patchIndices, patchParams);

    return true;
}

/* static */
bool
CudaEvaluator::EvalPatches(
    const float *src, BufferDescriptor const &srcDesc,
    float *dst,       BufferDescriptor const &dstDesc,
    float *du,        BufferDescriptor const &duDesc,
    float *dv,        BufferDescriptor const &dvDesc,
    int numPatchCoords,
    const PatchCoord *patchCoords,
    const PatchArray *patchArrays,
    const int *patchIndices,
    const PatchParam *patchParams) {

    if (src) src += srcDesc.offset;
    if (dst) dst += dstDesc.offset;
    if (du)  du  += duDesc.offset;
    if (dv)  dv  += dvDesc.offset;

    CudaEvalPatchesWithDerivatives(
        src, dst, du, dv,
        srcDesc.length, srcDesc.stride,
        dstDesc.stride, duDesc.stride, dvDesc.stride,
        numPatchCoords, patchCoords, patchArrays, patchIndices, patchParams);
    return true;
}



/* static */
void
CudaEvaluator::Synchronize(void * /*deviceContext*/) {
    cudaThreadSynchronize();
}

}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
