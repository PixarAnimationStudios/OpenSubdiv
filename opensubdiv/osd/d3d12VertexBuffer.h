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

#ifndef OPENSUBDIV3_OSD_D3D12_VERTEX_BUFFER_H
#define OPENSUBDIV3_OSD_D3D12_VERTEX_BUFFER_H

#include "../version.h"
#include "d3d12commandqueuecontext.h"
#include "d3d12deferredDeletionUniquePtr.h"

struct ID3D11Buffer;
struct ID3D11ImmediateContext;

struct ID3D12Resource;
struct ID3D12CommandQueue;
struct ID3D12Device;

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

///
/// \brief Concrete vertex buffer class for DirectX subvision and DirectX drawing.
///
/// D3D12VertexBuffer implements D3D12VertexBufferInterface. An instance
/// of this buffer class can be passed to D3D12ComputeEvaluator.
///
class D3D12VertexBuffer {
public:
    /// Creator. Returns NULL if error.
    static D3D12VertexBuffer * Create(int numElements, int numVertices,
                                      D3D12CommandQueueContext* D3D12CommandQueueContext);

    /// Destructor.
    virtual ~D3D12VertexBuffer();

    /// This method is meant to be used in client code in order to provide coarse
    /// vertices data to Osd.
    void UpdateData(const float *src, int startVertex, int numVertices,
                    D3D12CommandQueueContext* D3D12CommandQueueContext);

    /// Returns how many elements defined in this vertex buffer.
    int GetNumElements() const;

    /// Returns how many vertices allocated in this vertex buffer.
    int GetNumVertices() const;

    /// Returns the D3D11 buffer object.
    CPUDescriptorHandle BindD3D12Buffer(D3D12CommandQueueContext* D3D12CommandQueueContext);


    /// Returns the D3D11 buffer object (for Osd::Mesh interface)
    ID3D11Buffer *BindVBO(D3D12CommandQueueContext *D3D12CommandQueueContext);

    /// Returns the D3D12 UAV
    CPUDescriptorHandle BindD3D12UAV(D3D12CommandQueueContext* D3D12CommandQueueContext);

protected:
    /// Constructor.
    D3D12VertexBuffer(int numElements, int numVertices);

    // Allocates D3D11 buffer
    bool allocate(D3D12CommandQueueContext* D3D12CommandQueueContext);

private:
    int _numElements;
    int _numVertices;
    int _dataSize;
    DeferredDeletionUniquePtr<ID3D12Resource> _buffer;
    DeferredDeletionUniquePtr<ID3D12Resource> _readbackBuffer;
    DeferredDeletionUniquePtr<ID3D12Resource> _uploadBuffer;
    CComPtr<ID3D11Buffer> _d3d11Buffer;  

    CPUDescriptorHandle _uav;
};

}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OPENSUBDIV3_OSD_D3D12_VERTEX_BUFFER_H
