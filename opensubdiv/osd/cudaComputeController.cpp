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

#include "../osd/cudaComputeController.h"

#include <cuda_runtime.h>
#include <string.h>

extern "C" {

    void CudaComputeStencils(float const *src, float * dst,
                                int length, int stride,
                                unsigned char const * sizes,
                                int const * offsets,
                                int const * indices,
                                float const * weights,
                                int start, int end);

}

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

void
CudaComputeController::ApplyStencilTableKernel(
    Far::KernelBatch const &batch, ComputeContext const *context) const {

    assert(context);

    if (context->HasVertexStencilTables()) {

        int length = _currentBindState.vertexDesc.length,
            stride = _currentBindState.vertexDesc.stride;

        float const * src = _currentBindState.GetVertexBufferAtOffset();

        float * dst = const_cast<float *>(src) +
            context->GetNumControlVertices() * stride;

        CudaComputeStencils(src, dst, length, stride,
                               (unsigned char const *)context->GetVertexStencilTablesSizes(),
                               (int const *)context->GetVertexStencilTablesOffsets(),
                               (int const *)context->GetVertexStencilTablesIndices(),
                               (float const *)context->GetVertexStencilTablesWeights(),
                               batch.start,
                               batch.end);
    }

    if (context->HasVaryingStencilTables()) {

        int length = _currentBindState.varyingDesc.length,
            stride = _currentBindState.varyingDesc.stride;

        float const * src = _currentBindState.GetVaryingBufferAtOffset();

        float * dst = const_cast<float *>(src) +
            context->GetNumControlVertices() * stride;

        CudaComputeStencils(src, dst, length, stride,
                               (unsigned char const *)context->GetVaryingStencilTablesSizes(),
                               (int const *)context->GetVaryingStencilTablesOffsets(),
                               (int const *)context->GetVaryingStencilTablesIndices(),
                               (float const *)context->GetVaryingStencilTablesWeights(),
                               batch.start,
                               batch.end);
    }
}

CudaComputeController::CudaComputeController() {
}

CudaComputeController::~CudaComputeController() {
}

void
CudaComputeController::Synchronize() {

    cudaThreadSynchronize();
}

} // end namespace Osd

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv
