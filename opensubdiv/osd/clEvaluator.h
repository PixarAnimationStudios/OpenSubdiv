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

#ifndef OPENSUBDIV_OPENSUBDIV3_OSD_CL_EVALUATOR_H
#define OPENSUBDIV_OPENSUBDIV3_OSD_CL_EVALUATOR_H

#include "../version.h"

#include "../osd/opencl.h"
#include "../osd/vertexDescriptor.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Far {
    class StencilTables;
}

namespace Osd {

/// \brief OpenCL stencil tables
///
/// This class is an OpenCL buffer representation of Far::StencilTables.
///
/// CLCompute consumes this table to apply stencils
///
///
class CLStencilTables {
public:
    template <typename DEVICE_CONTEXT>
    static CLStencilTables *Create(Far::StencilTables const *stencilTables,
                                   DEVICE_CONTEXT context) {
        return new CLStencilTables(stencilTables, context->GetContext());
    }

    CLStencilTables(Far::StencilTables const *stencilTables,
                    cl_context clContext);
    ~CLStencilTables();

    // interfaces needed for CLComputeKernel
    cl_mem GetSizesBuffer()     const { return _sizes; }
    cl_mem GetOffsetsBuffer()   const { return _offsets; }
    cl_mem GetIndicesBuffer()   const { return _indices; }
    cl_mem GetWeightsBuffer()   const { return _weights; }
    int GetNumStencils()        const { return _numStencils; }

private:
    cl_mem _sizes;
    cl_mem _offsets;
    cl_mem _indices;
    cl_mem _weights;
    int _numStencils;
};

// ---------------------------------------------------------------------------

/// \brief OpenCL stencil kernel
///
///
class CLEvaluator {
public:
    typedef bool Instantiatable;
    /// Constructor.
    CLEvaluator(cl_context context, cl_command_queue queue);

    /// Desctructor.
    ~CLEvaluator();

    /// Generic creator template.
    template <typename DEVICE_CONTEXT>
    static CLEvaluator *Create(VertexBufferDescriptor const &srcDesc,
                               VertexBufferDescriptor const &dstDesc,
                               DEVICE_CONTEXT deviceContext) {
        return Create(srcDesc, dstDesc,
                      deviceContext->GetContext(),
                      deviceContext->GetCommandQueue());
    }

    static CLEvaluator * Create(VertexBufferDescriptor const &srcDesc,
                                VertexBufferDescriptor const &dstDesc,
                                cl_context clContext,
                                cl_command_queue clCommandQueue) {
        CLEvaluator *kernel = new CLEvaluator(clContext, clCommandQueue);
        if (kernel->Compile(srcDesc, dstDesc)) return kernel;
        delete kernel;
        return NULL;
    }

    /// \brief Generic static compute function. This function has a same
    ///        signature as other device kernels have so that it can be called
    ///        transparently from OsdMesh template interface.
    ///
    /// @param srcBuffer      Input primvar buffer.
    ///                       must have BindCLBuffer() method returning a
    ///                       const float pointer for read
    ///
    /// @param srcDesc        vertex buffer descriptor for the input buffer
    ///
    /// @param dstBuffer      Output primvar buffer
    ///                       must have BindCLBuffer() method returning a
    ///                       float pointer for write
    ///
    /// @param dstDesc        vertex buffer descriptor for the output buffer
    ///
    /// @param stencilTables  stencil table to be applied. The table must have
    ///                       OpenCL memory interfaces.
    ///
    /// @param instance       cached compiled instance. Clients are supposed to
    ///                       pre-compile an instance of this class and provide
    ///                       to this function. If it's null the kernel still
    ///                       compute by instantiating on-demand kernel although
    ///                       it may cause a performance problem.
    ///
    /// @param deviceContext  client providing context class which supports
    ///                         cL_context GetContext()
    ///                         cl_command_queue GetCommandQueue()
    ///                       methods.
    ///
    template <typename VERTEX_BUFFER, typename STENCIL_TABLE,
              typename DEVICE_CONTEXT>
    static bool EvalStencils(VERTEX_BUFFER *srcVertexBuffer,
                             VertexBufferDescriptor const &srcDesc,
                             VERTEX_BUFFER *dstVertexBuffer,
                             VertexBufferDescriptor const &dstDesc,
                             STENCIL_TABLE const *stencilTable,
                             CLEvaluator const *instance,
                             DEVICE_CONTEXT deviceContext) {
        if (instance) {
            return instance->EvalStencils(srcVertexBuffer, srcDesc,
                                          dstVertexBuffer, dstDesc,
                                          stencilTable);
        } else {
            // Create an instance on demand (slow)
            instance = Create(srcDesc, dstDesc, deviceContext);
            if (instance) {
                bool r = instance->EvalStencils(srcVertexBuffer, srcDesc,
                                                dstVertexBuffer, dstDesc,
                                                stencilTable);
                delete instance;
                return r;
            }
            return false;
        }
    }

    /// Generic compute function.
    /// Dispatch the CL compute kernel asynchronously.
    /// Returns false if the kernel hasn't been compiled yet.
    template <typename VERTEX_BUFFER, typename STENCIL_TABLE>
    bool EvalStencils(VERTEX_BUFFER *srcVertexBuffer,
                      VertexBufferDescriptor const &srcDesc,
                      VERTEX_BUFFER *dstVertexBuffer,
                      VertexBufferDescriptor const &dstDesc,
                      STENCIL_TABLE const *stencilTable) const {
        return EvalStencils(srcVertexBuffer->BindCLBuffer(_clCommandQueue),
                            srcDesc,
                            dstVertexBuffer->BindCLBuffer(_clCommandQueue),
                            dstDesc,
                            stencilTable->GetSizesBuffer(),
                            stencilTable->GetOffsetsBuffer(),
                            stencilTable->GetIndicesBuffer(),
                            stencilTable->GetWeightsBuffer(),
                            0,
                            stencilTable->GetNumStencils());
    }

    /// Dispatch the CL compute kernel asynchronously.
    /// returns false if the kernel hasn't been compiled yet.
    bool EvalStencils(cl_mem src,
                      VertexBufferDescriptor const &srcDesc,
                      cl_mem dst,
                      VertexBufferDescriptor const &dstDesc,
                      cl_mem sizes,
                      cl_mem offsets,
                      cl_mem indices,
                      cl_mem weights,
                      int start,
                      int end) const;

    /// Configure OpenCL kernel.
    /// Returns false if it fails to compile the kernel.
    bool Compile(VertexBufferDescriptor const &srcDesc,
                 VertexBufferDescriptor const &dstDesc);

    /// Wait the OpenCL kernels finish.
    template <typename DEVICE_CONTEXT>
    static void Synchronize(DEVICE_CONTEXT deviceContext) {
        Synchronize(deviceContext->GetCommandQueue());
    }

    static void Synchronize(cl_command_queue queue);

private:
    cl_context _clContext;
    cl_command_queue _clCommandQueue;
    cl_program _program;
    cl_kernel _stencilsKernel;
};


}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv


#endif  // OPENSUBDIV_OPENSUBDIV3_OSD_CL_EVALUATOR_H
