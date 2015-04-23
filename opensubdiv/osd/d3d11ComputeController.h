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

#include "../far/kernelBatchDispatcher.h"
#include "../osd/d3d11ComputeContext.h"
#include "../osd/vertexDescriptor.h"

#include <vector>

struct ID3D11DeviceContext;
struct ID3D11Query;
struct ID3D11UnorderedAccessView;

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

/// \brief Compute controller for launching D3D11 Compute subdivision kernels.
///
/// D3D11ComputeController is a compute controller class to launch
/// D3D11Compute transfrom feedback subdivision kernels. It requires
/// GLVertexBufferInterface as arguments of Refine function.
///
/// Controller entities execute requests from Context instances that they share
/// common interfaces with. Controllers are attached to discrete compute devices
/// and share the devices resources with Context entities.
///
class D3D11ComputeController {
public:
    typedef D3D11ComputeContext ComputeContext;

    /// Constructor.
    ///
    /// @param deviceContext  a valid instanciated D3D11 device context
    ///
    D3D11ComputeController(ID3D11DeviceContext *deviceContext);

    /// Destructor.
    ~D3D11ComputeController();

    /// Execute subdivision kernels and apply to given vertex buffers.
    ///
    /// @param  context       The D3D11Context to apply refinement operations to
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
        void Compute( D3D11ComputeContext const * context,
                      Far::KernelBatchVector const & batches,
                      VERTEX_BUFFER  * vertexBuffer,
                      VARYING_BUFFER * varyingBuffer,
                      VertexBufferDescriptor const * vertexDesc=NULL,
                      VertexBufferDescriptor const * varyingDesc=NULL ){

        if (batches.empty()) return;

        if (vertexBuffer) {
            bind(vertexBuffer, vertexDesc);

            context->BindVertexStencilTables(_deviceContext);

            Far::KernelBatchDispatcher::Apply(this, context, batches, /*maxlevel*/ -1);
        }

        if (varyingBuffer) {
            bind(varyingBuffer, varyingDesc);

            context->BindVaryingStencilTables(_deviceContext);

            Far::KernelBatchDispatcher::Apply(this, context, batches, /*maxlevel*/ -1);
        }

        context->UnbindStencilTables(_deviceContext);

        unbind();
    }

    /// Execute subdivision kernels and apply to given vertex buffers.
    ///
    /// @param  context       The D3D11Context to apply refinement operations to
    ///
    /// @param  batches       Vector of batches of vertices organized by operative
    ///                       kernel
    ///
    /// @param  vertexBuffer  Vertex-interpolated data buffer
    ///
    template<class VERTEX_BUFFER>
        void Compute(D3D11ComputeContext const * context,
                     Far::KernelBatchVector const & batches,
                     VERTEX_BUFFER *vertexBuffer) {

        Compute<VERTEX_BUFFER>(context, batches, vertexBuffer, (VERTEX_BUFFER*)0);
    }

    /// Waits until all running subdivision kernels finish.
    void Synchronize();

protected:

    friend class Far::KernelBatchDispatcher;

    void ApplyStencilTableKernel(Far::KernelBatch const &batch,
        ComputeContext const *context) const;

    template<class BUFFER>
        void bind( BUFFER * buffer,
                   VertexBufferDescriptor const * desc ) {

        assert(buffer);

        // if the vertex buffer descriptor is specified, use it
        // otherwise, assumes the data is tightly packed in the vertex buffer.
        if (desc) {
            _currentBindState.desc = *desc;
        } else {
            int numElements = buffer ? buffer->GetNumElements() : 0;
            _currentBindState.desc =
                VertexBufferDescriptor(0, numElements, numElements);
        }

        _currentBindState.buffer = buffer->BindD3D11UAV(_deviceContext);

        _currentBindState.kernelBundle = getKernel(_currentBindState.desc);

        bindBuffer();
    }


    // Unbinds any previously bound vertex and varying data buffers.
    void unbind() {
        _currentBindState.Reset();
        unbindBuffer();
    }

    // binds the primvar data buffer
    void bindBuffer();

    // unbinds the primvar data buffer
    void unbindBuffer();


private:

    ID3D11DeviceContext *_deviceContext;
    ID3D11Query *_query;

    class KernelBundle;

    // Bind state is a transitional state during refinement.
    // It doesn't take an ownership of the vertex buffers.
    struct BindState {

        BindState() : buffer(0), kernelBundle(0) { }

        void Reset() {
            buffer = 0;
            desc.Reset();
            kernelBundle = 0;
        }

        ID3D11UnorderedAccessView * buffer;

        VertexBufferDescriptor desc;

        KernelBundle const * kernelBundle;
    };

    BindState _currentBindState;

    typedef std::vector<KernelBundle *> KernelRegistry;

    KernelBundle const * getKernel(VertexBufferDescriptor const &desc);

    KernelRegistry _kernelRegistry;
};

}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OSD_D3D11_COMPUTE_CONTROLLER_H
