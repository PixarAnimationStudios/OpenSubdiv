//
//     Copyright (C) Pixar. All rights reserved.
//
//     This license governs use of the accompanying software. If you
//     use the software, you accept this license. If you do not accept
//     the license, do not use the software.
//
//     1. Definitions
//     The terms "reproduce," "reproduction," "derivative works," and
//     "distribution" have the same meaning here as under U.S.
//     copyright law.  A "contribution" is the original software, or
//     any additions or changes to the software.
//     A "contributor" is any person or entity that distributes its
//     contribution under this license.
//     "Licensed patents" are a contributor's patent claims that read
//     directly on its contribution.
//
//     2. Grant of Rights
//     (A) Copyright Grant- Subject to the terms of this license,
//     including the license conditions and limitations in section 3,
//     each contributor grants you a non-exclusive, worldwide,
//     royalty-free copyright license to reproduce its contribution,
//     prepare derivative works of its contribution, and distribute
//     its contribution or any derivative works that you create.
//     (B) Patent Grant- Subject to the terms of this license,
//     including the license conditions and limitations in section 3,
//     each contributor grants you a non-exclusive, worldwide,
//     royalty-free license under its licensed patents to make, have
//     made, use, sell, offer for sale, import, and/or otherwise
//     dispose of its contribution in the software or derivative works
//     of the contribution in the software.
//
//     3. Conditions and Limitations
//     (A) No Trademark License- This license does not grant you
//     rights to use any contributor's name, logo, or trademarks.
//     (B) If you bring a patent claim against any contributor over
//     patents that you claim are infringed by the software, your
//     patent license from such contributor to the software ends
//     automatically.
//     (C) If you distribute any portion of the software, you must
//     retain all copyright, patent, trademark, and attribution
//     notices that are present in the software.
//     (D) If you distribute any portion of the software in source
//     code form, you may do so only under this license by including a
//     complete copy of this license with your distribution. If you
//     distribute any portion of the software in compiled or object
//     code form, you may only do so under a license that complies
//     with this license.
//     (E) The software is licensed "as-is." You bear the risk of
//     using it. The contributors give no express warranties,
//     guarantees or conditions. You may have additional consumer
//     rights under your local laws which this license cannot change.
//     To the extent permitted under your local laws, the contributors
//     exclude the implied warranties of merchantability, fitness for
//     a particular purpose and non-infringement.
//
#ifndef OSD_CL_COMPUTE_CONTROLLER_H
#define OSD_CL_COMPUTE_CONTROLLER_H

#include "../version.h"

#include "../far/dispatcher.h"
#include "../osd/clComputeContext.h"

#if defined(__APPLE__)
    #include <OpenCL/opencl.h>
#else
    #include <CL/opencl.h>
#endif

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
    template<class VERTEX_BUFFER, class VARYING_BUFFER>
    void Refine(OsdCLComputeContext *context,
                FarKernelBatchVector const &batches,
                VERTEX_BUFFER *vertexBuffer,
                VARYING_BUFFER *varyingBuffer) {

        int numVertexElements = vertexBuffer ? vertexBuffer->GetNumElements() : 0;
        int numVaryingElements = varyingBuffer ? varyingBuffer->GetNumElements() : 0;

        context->SetKernelBundle(getKernelBundle(numVertexElements, numVaryingElements));

        context->Bind(vertexBuffer, varyingBuffer, _clQueue);
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
    void Refine(OsdCLComputeContext *context,
                FarKernelBatchVector const &batches,
                VERTEX_BUFFER *vertexBuffer) {
        Refine(context, batches, vertexBuffer, (VERTEX_BUFFER*)NULL);
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


    OsdCLKernelBundle * getKernelBundle(int numVertexElements,
                                        int numVaryingElements);

private:
    cl_context _clContext;
    cl_command_queue _clQueue;
    std::vector<OsdCLKernelBundle *> _kernelRegistry;
};

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OSD_COMPUTE_CONTROLLER_H
