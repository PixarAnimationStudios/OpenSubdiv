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

#ifndef OSD_CL_COMPUTE_CONTROLLER_H
#define OSD_CL_COMPUTE_CONTROLLER_H

#include "../version.h"

#include "../far/kernelBatchDispatcher.h"
#include "../osd/clComputeContext.h"
#include "../osd/vertexDescriptor.h"
#include "../osd/opencl.h"

#include <vector>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

class CLKernelBundle;

/// \brief Compute controller for launching OpenCL Compute subdivision kernels.
///
/// CLComputeController is a compute controller class to launch
/// OpenCL subdivision kernels. It requires CLVertexBufferInterface
/// as arguments of Refine function.
///
/// Controller entities execute requests from Context instances that they share
/// common interfaces with. Controllers are attached to discrete compute devices
/// and share the devices resources with Context entities.
///
class CLComputeController {
public:
    typedef CLComputeContext ComputeContext;

    /// Constructor.
    ///
    /// @param clContext a valid instanciated OpenCL context
    ///
    /// @param queue a valid non-zero OpenCL command queue
    ///
    CLComputeController(cl_context clContext, cl_command_queue queue);

    /// Destructor.
    ~CLComputeController();

    /// Execute subdivision kernels and apply to given vertex buffers.
    ///
    /// @param  context       The CLContext to apply refinement operations to
    ///
    /// @param  batches       Vector of batches of vertices organized by operative
    ///                       kernel
    ///
    /// @param  vertexBuffer  Vertex-interpolated data buffer
    ///
    /// @param  vertexDesc    The descriptor of vertex elements to be refined.
    ///                       if it's null, all primvars in the vertex buffer
    ///                       will be refined.
    ///
    /// @param  varyingBuffer Vertex-interpolated data buffer
    ///
    /// @param  varyingDesc   The descriptor of varying elements to be refined.
    ///                       if it's null, all primvars in the vertex buffer
    ///                       will be refined.
    ///
    template<class VERTEX_BUFFER, class VARYING_BUFFER>
        void Compute( CLComputeContext const * context,
                      Far::KernelBatchVector const & batches,
                      VERTEX_BUFFER  * vertexBuffer,
                      VARYING_BUFFER * varyingBuffer,
                      VertexBufferDescriptor const * vertexDesc=NULL,
                      VertexBufferDescriptor const * varyingDesc=NULL ){

        if (batches.empty()) return;

        bind(vertexBuffer, varyingBuffer, vertexDesc, varyingDesc);

        Far::KernelBatchDispatcher::Apply(this, context, batches, /*maxlevel*/ -1);

        unbind();
    }

    /// Execute subdivision kernels and apply to given vertex buffers.
    ///
    /// @param  context       The CLContext to apply refinement operations to
    ///
    /// @param  batches       Vector of batches of vertices organized by operative
    ///                       kernel
    ///
    /// @param  vertexBuffer  Vertex-interpolated data buffer
    ///
    template<class VERTEX_BUFFER>
        void Compute(CLComputeContext const * context,
                     Far::KernelBatchVector const & batches,
                     VERTEX_BUFFER *vertexBuffer) {

        Compute<VERTEX_BUFFER>(context, batches, vertexBuffer, (VERTEX_BUFFER*)0);
    }

    /// Waits until all running subdivision kernels finish.
    void Synchronize();

    /// Returns CL context
    cl_context GetContext() const { return _clContext; }

    /// Returns CL command queue
    cl_command_queue GetCommandQueue() const { return _clQueue; }

protected:

    friend class Far::KernelBatchDispatcher;

    void ApplyStencilTableKernel(Far::KernelBatch const &batch,
        ComputeContext const *context);

    template<class VERTEX_BUFFER, class VARYING_BUFFER>
        void bind( VERTEX_BUFFER * vertexBuffer,
                   VARYING_BUFFER * varyingBuffer,
                   VertexBufferDescriptor const * vertexDesc,
                   VertexBufferDescriptor const * varyingDesc ) {

        // if the vertex buffer descriptor is specified, use it.
        // otherwise, assumes the data is tightly packed in the vertex buffer.
        if (vertexDesc) {
            _currentBindState.vertexDesc = *vertexDesc;
        } else {
            int numElements = vertexBuffer ? vertexBuffer->GetNumElements() : 0;
            _currentBindState.vertexDesc =
                VertexBufferDescriptor(0, numElements, numElements);
        }

        if (varyingDesc) {
            _currentBindState.varyingDesc = *varyingDesc;
        } else {
            int numElements = varyingBuffer ? varyingBuffer->GetNumElements() : 0;
            _currentBindState.varyingDesc =
                VertexBufferDescriptor(0, numElements, numElements);
        }

        _currentBindState.vertexBuffer = vertexBuffer ?
            vertexBuffer->BindCLBuffer(_clQueue) : 0;
        _currentBindState.varyingBuffer = varyingBuffer ?
            varyingBuffer->BindCLBuffer(_clQueue) : 0;
    }

    void unbind() {
        _currentBindState.Reset();
    }


private:

    class KernelBundle;

    // Bind state is a transitional state during refinement.
    // It doesn't take an ownership of the vertex buffers.
    struct BindState {

        BindState() : vertexBuffer(0), varyingBuffer(0) { }

        void Reset() {
            vertexBuffer = varyingBuffer = NULL;
            vertexDesc.Reset();
            varyingDesc.Reset();
        }

        cl_mem vertexBuffer,
               varyingBuffer;

        VertexBufferDescriptor vertexDesc,
                                  varyingDesc;
    };

    BindState _currentBindState;

    KernelBundle const * getKernel(VertexBufferDescriptor const &desc);

    typedef std::vector<KernelBundle *> KernelRegistry;

    KernelRegistry _kernelRegistry;

    cl_context _clContext;
    cl_command_queue _clQueue;
};

}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OSD_COMPUTE_CONTROLLER_H
