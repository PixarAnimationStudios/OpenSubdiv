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
    } else {
        _sizes = _offsets = _indices = _weights = NULL;
    }
}

CudaStencilTable::~CudaStencilTable() {
    if (_sizes)   cudaFree(_sizes);
    if (_offsets) cudaFree(_offsets);
    if (_indices) cudaFree(_indices);
    if (_weights) cudaFree(_weights);
}

// ---------------------------------------------------------------------------

/* static */
bool
CudaEvaluator::EvalStencils(const float *src,
                            VertexBufferDescriptor const &srcDesc,
                            float *dst,
                            VertexBufferDescriptor const &dstDesc,
                            const int * sizes,
                            const int * offsets,
                            const int * indices,
                            const float * weights,
                            int start,
                            int end) {
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
void
CudaEvaluator::Synchronize(void * /*deviceContext*/) {
    cudaThreadSynchronize();
}

}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
