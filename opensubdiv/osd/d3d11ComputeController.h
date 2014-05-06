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

#ifndef OSD_D3D11_COMPUTE_CONTROLLER_H
#define OSD_D3D11_COMPUTE_CONTROLLER_H

#include "../version.h"

#include "../far/dispatcher.h"
#include "../osd/d3d11ComputeContext.h"

#include <vector>

struct ID3D11DeviceContext;
struct ID3D11Query;

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

class OsdD3D11ComputeKernelBundle;

/// \brief Compute controller for launching D3D11Compute transform feedback
/// subdivision kernels.
///
/// OsdD3D11ComputeController is a compute controller class to launch
/// D3D11Compute transfrom feedback subdivision kernels. It requires
/// OsdD3D11VertexBufferInterface as arguments of Refine function.
///
/// Controller entities execute requests from Context instances that they share
/// common interfaces with. Controllers are attached to discrete compute devices
/// and share the devices resources with Context entities.
///
class OsdD3D11ComputeController {
public:
    typedef OsdD3D11ComputeContext ComputeContext;

    /// Constructor.
    ///
    /// @param deviceContext  a valid instanciated D3D11 device context
    ///
    OsdD3D11ComputeController(ID3D11DeviceContext *deviceContext);

    /// Destructor.
    ~OsdD3D11ComputeController();

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
    void Refine(OsdD3D11ComputeContext const *context,
                FarKernelBatchVector const &batches,
                VERTEX_BUFFER *vertexBuffer,
                VARYING_BUFFER *varyingBuffer) {

        if (batches.empty()) return;

        bind(vertexBuffer, varyingBuffer);
        context->BindShaderStorageBuffers(_deviceContext);

        FarDispatcher::Refine(this,
                              context,
                              batches,
                              -1);

        context->UnbindShaderStorageBuffers(_deviceContext);
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
    void Refine(OsdD3D11ComputeContext const *context,
                FarKernelBatchVector const &batches,
                VERTEX_BUFFER *vertexBuffer) {
        Refine(context, batches, vertexBuffer, (VERTEX_BUFFER*)NULL);
    }

    /// Waits until all running subdivision kernels finish.
    void Synchronize();

protected:
    friend class FarDispatcher;
    void ApplyBilinearFaceVerticesKernel(FarKernelBatch const &batch, ComputeContext const *context) const;

    void ApplyBilinearEdgeVerticesKernel(FarKernelBatch const &batch, ComputeContext const *context) const;

    void ApplyBilinearVertexVerticesKernel(FarKernelBatch const &batch, ComputeContext const *context) const;


    void ApplyCatmarkFaceVerticesKernel(FarKernelBatch const &batch, ComputeContext const *context) const;

    void ApplyCatmarkEdgeVerticesKernel(FarKernelBatch const &batch, ComputeContext const *context) const;

    void ApplyCatmarkVertexVerticesKernelB(FarKernelBatch const &batch, ComputeContext const *context) const;

    void ApplyCatmarkVertexVerticesKernelA1(FarKernelBatch const &batch, ComputeContext const *context) const;

    void ApplyCatmarkVertexVerticesKernelA2(FarKernelBatch const &batch, ComputeContext const *context) const;


    void ApplyLoopEdgeVerticesKernel(FarKernelBatch const &batch, ComputeContext const *context) const;

    void ApplyLoopVertexVerticesKernelB(FarKernelBatch const &batch, ComputeContext const *context) const;

    void ApplyLoopVertexVerticesKernelA1(FarKernelBatch const &batch, ComputeContext const *context) const;

    void ApplyLoopVertexVerticesKernelA2(FarKernelBatch const &batch, ComputeContext const *context) const;


    void ApplyVertexEdits(FarKernelBatch const &batch, ComputeContext const *context) const;

    OsdD3D11ComputeKernelBundle * getKernels(int numVertexElements,
                                             int numVaryingElements);

    void bindShaderResources();

    void unbindShaderResources();

    template<class VERTEX_BUFFER, class VARYING_BUFFER>
    void bind(VERTEX_BUFFER *vertex, VARYING_BUFFER *varying) {

        _currentVertexBufferUAV = vertex ? vertex->BindD3D11UAV(_deviceContext) : 0;
        _currentVaryingBufferUAV = varying ? varying->BindD3D11UAV(_deviceContext) : 0;

        _vdesc.numVertexElements = vertex ? vertex->GetNumElements() : 0;
        _vdesc.numVaryingElements = varying ? varying->GetNumElements() : 0;

        _currentKernelBundle = getKernels(_vdesc.numVertexElements,
                                          _vdesc.numVaryingElements);

        bindShaderResources();
    }

    void unbind() {
        _currentVertexBufferUAV = 0;
        _currentVaryingBufferUAV = 0;
        _currentKernelBundle = 0;

        unbindShaderResources();
    }

private:
    ID3D11DeviceContext *_deviceContext;
    ID3D11Query *_query;
    std::vector<OsdD3D11ComputeKernelBundle *> _kernelRegistry;

    OsdVertexDescriptor _vdesc;

    ID3D11UnorderedAccessView * _currentVertexBufferUAV,
                              * _currentVaryingBufferUAV;

    OsdD3D11ComputeKernelBundle * _currentKernelBundle;

};

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OSD_D3D11_COMPUTE_CONTROLLER_H
