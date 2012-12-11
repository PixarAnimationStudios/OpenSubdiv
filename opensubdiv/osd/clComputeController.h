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

#include "../osd/clComputeContext.h"
#include "../osd/clDispatcher.h"

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
/// OsdCLComputeController is a compute controller class to launch
/// OpenCL subdivision kernels. It requires OsdCLVertexBufferInterface
/// as arguments of Refine function.
class OsdCLComputeController {
public:
    typedef OsdCLComputeContext ComputeContext;

    /// Constructor.
    OsdCLComputeController(cl_context clContext, cl_command_queue queue);

    /// Destructor.
    ~OsdCLComputeController();

    /// Launch subdivision kernels and apply to given vertex buffers.
    /// vertexBuffer will be interpolated with vertex interpolation and
    /// varyingBuffer will be interpolated with varying interpolation.
    /// vertexBuffer and varyingBuffer should implement
    /// OsdCLVertexBufferInterface.
    template<class VERTEX_BUFFER, class VARYING_BUFFER>
    void Refine(OsdCLComputeContext *context,
                VERTEX_BUFFER *vertexBuffer,
                VARYING_BUFFER *varyingBuffer) {

        int numVertexElements = vertexBuffer ? vertexBuffer->GetNumElements() : 0;
        int numVaryingElements = varyingBuffer ? varyingBuffer->GetNumElements() : 0;

        context->SetKernelBundle(getKernelBundle(numVertexElements, numVaryingElements));

        context->Bind(vertexBuffer, varyingBuffer, _clQueue);
        OsdCLKernelDispatcher::GetInstance()->Refine(context->GetFarMesh(), context);
        context->Unbind();
    }

    template<class VERTEX_BUFFER>
    void Refine(OsdCLComputeContext *context, VERTEX_BUFFER *vertexBuffer) {
        Refine(context, vertexBuffer, (VERTEX_BUFFER*)NULL);
    }

    /// Waits until all running subdivision kernels finish.
    void Synchronize();

private:
    OsdCLKernelBundle * getKernelBundle(int numVertexElements,
                                        int numVaryingElements);
    cl_context _clContext;
    cl_command_queue _clQueue;
    std::vector<OsdCLKernelBundle *> _kernelRegistry;
};

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OSD_COMPUTE_CONTROLLER_H
