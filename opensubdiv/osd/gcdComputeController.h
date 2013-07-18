//
//     Copyright 2013 Pixar
//
//     Licensed under the Apache License, Version 2.0 (the "License");
//     you may not use this file except in compliance with the License
//     and the following modification to it: Section 6 Trademarks.
//     deleted and replaced with:
//
//     6. Trademarks. This License does not grant permission to use the
//     trade names, trademarks, service marks, or product names of the
//     Licensor and its affiliates, except as required for reproducing
//     the content of the NOTICE file.
//
//     You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
//     Unless required by applicable law or agreed to in writing,
//     software distributed under the License is distributed on an
//     "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
//     either express or implied.  See the License for the specific
//     language governing permissions and limitations under the
//     License.
//
#ifndef OSD_GCD_COMPUTE_CONTROLLER_H
#define OSD_GCD_COMPUTE_CONTROLLER_H

#include "../version.h"

#include "../far/dispatcher.h"
#include "../osd/cpuComputeContext.h"

#include <dispatch/dispatch.h>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

/// \brief Compute controller for launching Grand Central Dispatch subdivision 
/// kernels.
///
/// OsdGcdComputeController is a compute controller class to launch Grand Central
/// Dispatch threaded subdivision kernels. It requires OsdCpuVertexBufferInterface
/// as arguments of Refine function.
///
/// Controller entities execute requests from Context instances that they share
/// common interfaces with. Controllers are attached to discrete compute devices
/// and share the devices resources with Context entities.
///
class OsdGcdComputeController {
public:
    typedef OsdCpuComputeContext ComputeContext;

    /// Constructor.
    explicit OsdGcdComputeController();

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
    template<class VERTEX_BUFFER, class VARYING_BUFFER>
    void Refine(OsdCpuComputeContext *context,
                FarKernelBatchVector const & batches,
                VERTEX_BUFFER *vertexBuffer,
                VARYING_BUFFER *varyingBuffer) {

        if (batches.empty()) return;

        context->Bind(vertexBuffer, varyingBuffer);
        FarDispatcher::Refine(this,
                              batches,
                              -1,
                              context);
        context->Unbind();
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
    void Refine(OsdCpuComputeContext *context,
                FarKernelBatchVector const & batches,
                VERTEX_BUFFER *vertexBuffer) {
        Refine(context, batches, vertexBuffer, (VERTEX_BUFFER*)0);
    }

    /// Waits until all running subdivision kernels finish.
    void Synchronize();

protected:
    friend class FarDispatcher;
    void ApplyBilinearFaceVerticesKernel(FarKernelBatch const &batch, void * clientdata) const;

    void ApplyBilinearEdgeVerticesKernel(FarKernelBatch const &batch, void * clientdata) const;

    void ApplyBilinearVertexVerticesKernel(FarKernelBatch const &batch, void * clientdata) const;


    void ApplyCatmarkFaceVerticesKernel(FarKernelBatch const &batch, void * clientdata) const;

    void ApplyCatmarkEdgeVerticesKernel(FarKernelBatch const &batch, void * clientdata) const;

    void ApplyCatmarkVertexVerticesKernelB(FarKernelBatch const &batch, void * clientdata) const;

    void ApplyCatmarkVertexVerticesKernelA1(FarKernelBatch const &batch, void * clientdata) const;

    void ApplyCatmarkVertexVerticesKernelA2(FarKernelBatch const &batch, void * clientdata) const;


    void ApplyLoopEdgeVerticesKernel(FarKernelBatch const &batch, void * clientdata) const;

    void ApplyLoopVertexVerticesKernelB(FarKernelBatch const &batch, void * clientdata) const;

    void ApplyLoopVertexVerticesKernelA1(FarKernelBatch const &batch, void * clientdata) const;

    void ApplyLoopVertexVerticesKernelA2(FarKernelBatch const &batch, void * clientdata) const;


    void ApplyVertexEdits(FarKernelBatch const &batch, void * clientdata) const;

private:
    dispatch_queue_t _gcd_queue;
};

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OSD_GCD_COMPUTE_CONTROLLER_H
