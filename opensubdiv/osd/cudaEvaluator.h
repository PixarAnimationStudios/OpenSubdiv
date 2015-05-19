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

#pragma once
#ifndef OPENSUBDIV3_OSD_CUDA_EVALUATOR_H
#define OPENSUBDIV3_OSD_CUDA_EVALUATOR_H

#include "../version.h"

#include <vector>
#include "../osd/vertexDescriptor.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Far {
    class StencilTables;
}

namespace Osd {

/// \brief CUDA stencil tables
///
/// This class is a cuda buffer representation of Far::StencilTables.
///
/// CudaComputeKernel consumes this table to apply stencils
///
///
class CudaStencilTables {
public:
    static CudaStencilTables *Create(Far::StencilTables const *stencilTables,
                                     void *deviceContext = NULL) {
        (void)deviceContext;  // unused
        return new CudaStencilTables(stencilTables);
    }

    explicit CudaStencilTables(Far::StencilTables const *stencilTables);
    ~CudaStencilTables();

    // interfaces needed for CudaCompute
    void *GetSizesBuffer() const { return _sizes; }
    void *GetOffsetsBuffer() const { return _offsets; }
    void *GetIndicesBuffer() const { return _indices; }
    void *GetWeightsBuffer() const { return _weights; }
    int GetNumStencils() const { return _numStencils; }

private:
    void * _sizes,
         * _offsets,
         * _indices,
         * _weights;
    int _numStencils;
};

// ---------------------------------------------------------------------------

class CudaEvaluator {
public:
    /// \brief Generic static compute function. This function has a same
    ///        signature as other device kernels have so that it can be called
    ///        transparently from OsdMesh template interface.
    ///
    /// @param srcBuffer      Input primvar buffer.
    ///                       must have BindCudaBuffer() method returning a
    ///                       const float pointer for read
    ///
    /// @param srcDesc        vertex buffer descriptor for the input buffer
    ///
    /// @param dstBuffer      Output primvar buffer
    ///                       must have BindCudaBuffer() method returning a
    ///                       float pointer for write
    ///
    /// @param dstDesc        vertex buffer descriptor for the output buffer
    ///
    /// @param stencilTables  stencil table to be applied. The table must have
    ///                       Cuda memory interfaces.
    ///
    /// @param instance       not used in the CudaEvaluator
    ///
    /// @param deviceContext  not used in the CudaEvaluator
    ///
    template <typename VERTEX_BUFFER, typename STENCIL_TABLE>
    static bool EvalStencils(VERTEX_BUFFER *srcVertexBuffer,
                             VertexBufferDescriptor const &srcDesc,
                             VERTEX_BUFFER *dstVertexBuffer,
                             VertexBufferDescriptor const &dstDesc,
                             STENCIL_TABLE const *stencilTable,
                             const void *instance = NULL,
                             void * deviceContext = NULL) {

        (void)instance;  // unused
        (void)deviceContext;  // unused
        return EvalStencils(srcVertexBuffer->BindCudaBuffer(),
                            srcDesc,
                            dstVertexBuffer->BindCudaBuffer(),
                            dstDesc,
                            (int const *)stencilTable->GetSizesBuffer(),
                            (int const *)stencilTable->GetOffsetsBuffer(),
                            (int const *)stencilTable->GetIndicesBuffer(),
                            (float const *)stencilTable->GetWeightsBuffer(),
                            /*start = */ 0,
                            /*end   = */ stencilTable->GetNumStencils());
    }

    static bool EvalStencils(const float *src,
                             VertexBufferDescriptor const &srcDesc,
                             float *dst,
                             VertexBufferDescriptor const &dstDesc,
                             const int * sizes,
                             const int * offsets,
                             const int * indices,
                             const float * weights,
                             int start,
                             int end);

    static void Synchronize(void *deviceContext = NULL);
};


}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv


#endif  // OPENSUBDIV3_OSD_CUDA_EVALUATOR_H
