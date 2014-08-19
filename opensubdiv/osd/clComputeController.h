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

#include "../far/dispatcher.h"
#include "../osd/clComputeContext.h"
#include "../osd/vertexDescriptor.h"
#include "../osd/opencl.h"

#include <vector>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

class OsdCLKernelBundle;

/// \brief Compute controller for launching OpenCL subdivision kernels.
///
/// OsdCLComputeController is a compute controller class to launch
/// OpenCL subdivision kernels. It requires OsdCLVertexBufferInterface
/// as arguments of Refine function.
///
/// Controller entities execute requests from Context instances that they share
/// common interfaces with. Controllers are attached to discrete compute devices
/// and share the devices resources with Context entities.
///
class OsdCLComputeController {
public:
    typedef OsdCLComputeContext ComputeContext;

    /// Constructor.
    ///
    /// @param clContext a valid instanciated OpenCL context
    ///
    /// @param queue a valid non-zero OpenCL command queue
    ///
    OsdCLComputeController(cl_context clContext, cl_command_queue queue);

    /// Destructor.
    ~OsdCLComputeController();

    /// Launch subdivision kernels and apply to given vertex buffers.
    ///
    /// @param  context       the OsdCpuContext to apply refinement operations to
    ///
    /// @param  batches       vector of batches of vertices organized by operative 
    ///                       kernel
    ///
    /// @param  vertexBuffer  vertex-interpolated data buffer
    ///
    /// @param  varyingBuffer varying-interpolated data buffer
    ///
    /// @param  vertexDesc    the descriptor of vertex elements to be refined.
    ///                       if it's null, all primvars in the vertex buffer
    ///                       will be refined.
    ///
    /// @param  varyingDesc   the descriptor of varying elements to be refined.
    ///                       if it's null, all primvars in the varying buffer
    ///                       will be refined.
    ///
    template<class VERTEX_BUFFER, class VARYING_BUFFER>
    void Refine(ComputeContext const *context,
                FarKernelBatchVector const &batches,
                VERTEX_BUFFER *vertexBuffer,
                VARYING_BUFFER *varyingBuffer,
                OsdVertexBufferDescriptor const *vertexDesc=NULL,
                OsdVertexBufferDescriptor const *varyingDesc=NULL) {

        if (batches.empty()) return;

        bind(vertexBuffer, varyingBuffer, vertexDesc, varyingDesc);

        FarDispatcher::Refine(this, context, batches, /*maxlevel*/-1);

        unbind();
    }

    /// Launch subdivision kernels and apply to given vertex buffers.
    ///
    /// @param  context       the OsdCpuContext to apply refinement operations to
    ///
    /// @param  batches       vector of batches of vertices organized by operative 
    ///                       kernel
    ///
    /// @param  vertexBuffer  vertex-interpolated data buffer
    ///
    template<class VERTEX_BUFFER>
    void Refine(ComputeContext const *context,
                FarKernelBatchVector const &batches,
                VERTEX_BUFFER *vertexBuffer) {
        Refine(context, batches, vertexBuffer, (VERTEX_BUFFER*)NULL);
    }

    /// Waits until all running subdivision kernels finish.
    void Synchronize();

    /// Returns CL context
    cl_context GetContext() const { return _clContext; }

    /// Returns CL command queue
    cl_command_queue GetCommandQueue() const { return _clQueue; }

protected:
    friend class FarDispatcher;

    void ApplyBilinearFaceVerticesKernel(FarKernelBatch const &batch, ComputeContext const *context) const;

    void ApplyBilinearEdgeVerticesKernel(FarKernelBatch const &batch, ComputeContext const *context) const;

    void ApplyBilinearVertexVerticesKernel(FarKernelBatch const &batch, ComputeContext const *context) const;


    void ApplyCatmarkFaceVerticesKernel(FarKernelBatch const &batch, ComputeContext const *context) const;

    void ApplyCatmarkQuadFaceVerticesKernel(FarKernelBatch const &batch, ComputeContext const *context) const;

    void ApplyCatmarkTriQuadFaceVerticesKernel(FarKernelBatch const &batch, ComputeContext const *context) const;

    void ApplyCatmarkEdgeVerticesKernel(FarKernelBatch const &batch, ComputeContext const *context) const;

    void ApplyCatmarkRestrictedEdgeVerticesKernel(FarKernelBatch const &batch, ComputeContext const *context) const;

    void ApplyCatmarkVertexVerticesKernelB(FarKernelBatch const &batch, ComputeContext const *context) const;

    void ApplyCatmarkVertexVerticesKernelA1(FarKernelBatch const &batch, ComputeContext const *context) const;

    void ApplyCatmarkVertexVerticesKernelA2(FarKernelBatch const &batch, ComputeContext const *context) const;

    void ApplyCatmarkRestrictedVertexVerticesKernelB1(FarKernelBatch const &batch, ComputeContext const *context) const;

    void ApplyCatmarkRestrictedVertexVerticesKernelB2(FarKernelBatch const &batch, ComputeContext const *context) const;

    void ApplyCatmarkRestrictedVertexVerticesKernelA(FarKernelBatch const &batch, ComputeContext const *context) const;


    void ApplyLoopEdgeVerticesKernel(FarKernelBatch const &batch, ComputeContext const *context) const;

    void ApplyLoopVertexVerticesKernelB(FarKernelBatch const &batch, ComputeContext const *context) const;

    void ApplyLoopVertexVerticesKernelA1(FarKernelBatch const &batch, ComputeContext const *context) const;

    void ApplyLoopVertexVerticesKernelA2(FarKernelBatch const &batch, ComputeContext const *context) const;


    void ApplyVertexEdits(FarKernelBatch const &batch, ComputeContext const *context) const;


    OsdCLKernelBundle * getKernelBundle(
        OsdVertexBufferDescriptor const &vertexDesc,
        OsdVertexBufferDescriptor const &varyingDesc);

    template<class VERTEX_BUFFER, class VARYING_BUFFER>
    void bind(VERTEX_BUFFER *vertex, VARYING_BUFFER *varying,
              OsdVertexBufferDescriptor const *vertexDesc,
              OsdVertexBufferDescriptor const *varyingDesc) {

        // if the vertex buffer descriptor is specified, use it.
        // otherwise, assumes the data is tightly packed in the vertex buffer.
        if (vertexDesc) {
            _currentBindState.vertexDesc = *vertexDesc;
        } else {
            int numElements = vertex ? vertex->GetNumElements() : 0;
            _currentBindState.vertexDesc = OsdVertexBufferDescriptor(
                0, numElements, numElements);
        }
        if (varyingDesc) {
            _currentBindState.varyingDesc = *varyingDesc;
        } else {
            int numElements = varying ? varying->GetNumElements() : 0;
            _currentBindState.varyingDesc = OsdVertexBufferDescriptor(
                0, numElements, numElements);
        }

        _currentBindState.vertexBuffer = vertex ? vertex->BindCLBuffer(_clQueue) : 0;
        _currentBindState.varyingBuffer = varying ? varying->BindCLBuffer(_clQueue) : 0;
        _currentBindState.kernelBundle = getKernelBundle(_currentBindState.vertexDesc,
                                                         _currentBindState.varyingDesc);
    }

    void unbind() {
        _currentBindState.Reset();
    }

private:
    struct BindState {
        BindState() : vertexBuffer(NULL), varyingBuffer(NULL), kernelBundle(NULL) {}
        void Reset() {
            vertexBuffer = varyingBuffer = NULL;
            vertexDesc.Reset();
            varyingDesc.Reset();
            kernelBundle = NULL;
        }
        cl_mem vertexBuffer;
        cl_mem varyingBuffer;
        OsdVertexBufferDescriptor vertexDesc;
        OsdVertexBufferDescriptor varyingDesc;
        OsdCLKernelBundle *kernelBundle;
    };

    BindState _currentBindState;

    cl_context _clContext;
    cl_command_queue _clQueue;
    std::vector<OsdCLKernelBundle *> _kernelRegistry;
};

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OSD_COMPUTE_CONTROLLER_H
