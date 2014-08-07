// Copyright 2014 Google Inc. All rights reserved.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef OSD_NEON_COMPUTE_CONTROLLER_H
#define OSD_NEON_COMPUTE_CONTROLLER_H

#include "../version.h"

#include "../far/dispatcher.h"
#include "../osd/cpuComputeController.h"
#include "../osd/neonComputeContext.h"
#include "../osd/vertexDescriptor.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

class OsdNeonComputeController : public OsdCpuComputeController
{
public:
    typedef OsdNeonComputeContext ComputeContext;

    /// Constructor.
    OsdNeonComputeController();

    /// Destructor.
    ~OsdNeonComputeController();

    /// Launch subdivision kernels and apply to given vertex buffers.
    ///
    /// @param  context       the OsdNeonContext to apply refinement operations to
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
    template <class VERTEX_BUFFER, class VARYING_BUFFER>
    void Refine(OsdNeonComputeContext const *context,
                FarKernelBatchVector const &batches,
                VERTEX_BUFFER *vertexBuffer,
                VARYING_BUFFER *varyingBuffer,
                OsdVertexBufferDescriptor const *vertexDesc = NULL,
                OsdVertexBufferDescriptor const *varyingDesc = NULL)
    {
        if (batches.empty())
            return;

        bind(vertexBuffer, varyingBuffer, vertexDesc, varyingDesc);

        FarDispatcher::Refine(this, context, batches, /* maxlevel */ -1);

        unbind();
    }

    /// Launch subdivision kernels and apply to given vertex buffers.
    ///
    /// @param  context       the OsdNeonContext to apply refinement operations to
    ///
    /// @param  batches       vector of batches of vertices organized by operative 
    ///                       kernel
    ///
    /// @param  vertexBuffer  vertex-interpolated data buffer
    ///
    template <class VERTEX_BUFFER>
    void Refine(OsdNeonComputeContext const *context,
                FarKernelBatchVector const &batches,
                VERTEX_BUFFER *vertexBuffer)
    {
        Refine(context, batches, vertexBuffer, (VERTEX_BUFFER*)NULL);
    }

    /// Waits until all running subdivision kernels finish.
    void Synchronize();

protected:
    friend class FarDispatcher;

    void ApplyCatmarkQuadFaceVerticesKernel(FarKernelBatch const &batch, ComputeContext const *context) const;

    void ApplyCatmarkTriQuadFaceVerticesKernel(FarKernelBatch const &batch, ComputeContext const *context) const;

    void ApplyCatmarkRestrictedEdgeVerticesKernel(FarKernelBatch const &batch, ComputeContext const *context) const;

    void ApplyCatmarkRestrictedVertexVerticesKernelB1(FarKernelBatch const &batch, ComputeContext const *context) const;

    void ApplyCatmarkRestrictedVertexVerticesKernelB2(FarKernelBatch const &batch, ComputeContext const *context) const;

    void ApplyCatmarkRestrictedVertexVerticesKernelA(FarKernelBatch const &batch, ComputeContext const *context) const;

private:
    bool neonKernelsSupported() const
    {
        return getVertexDesc().offset == 0 && getVertexDesc().length == 6 &&
            getVertexDesc().stride == 8 && (getVaryingBuffer() == NULL ||
            getVaryingDesc().length == 0);
    }
};

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif
