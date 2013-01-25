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

#include "../osd/cudaD3D11VertexBuffer.h"
#include "../osd/error.h"

#include <D3D11.h>
#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>
#include <cassert>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

OsdCudaD3D11VertexBuffer::OsdCudaD3D11VertexBuffer(int numElements,
                                                   int numVertices,
                                                   ID3D11Device *device) 
    : _numElements(numElements), _numVertices(numVertices),
      _d3d11Buffer(NULL), _cudaBuffer(NULL), _cudaResource(NULL) {
}

OsdCudaD3D11VertexBuffer::~OsdCudaD3D11VertexBuffer() {

    unmap();
    cudaGraphicsUnregisterResource(_cudaResource);
    _d3d11Buffer->Release();
}

OsdCudaD3D11VertexBuffer *
OsdCudaD3D11VertexBuffer::Create(int numElements, int numVertices,
                                 ID3D11Device *device) {
    OsdCudaD3D11VertexBuffer *instance =
        new OsdCudaD3D11VertexBuffer(numElements, numVertices, device);
    if (instance->allocate(device)) return instance;
    delete instance;
    return NULL;
}

void
OsdCudaD3D11VertexBuffer::UpdateData(const float *src, int numVertices,
                                     void *param) {

    map();
    cudaMemcpy(_cudaBuffer, src, _numElements * numVertices * sizeof(float),
               cudaMemcpyHostToDevice);
}

int
OsdCudaD3D11VertexBuffer::GetNumElements() const {

    return _numElements;
}

int
OsdCudaD3D11VertexBuffer::GetNumVertices() const {

    return _numVertices;
}

float *
OsdCudaD3D11VertexBuffer::BindCudaBuffer() {

    map();
    return (float*)_cudaBuffer;
}

ID3D11Buffer *
OsdCudaD3D11VertexBuffer::BindD3D11Buffer(ID3D11DeviceContext *deviceContext) {

    unmap();
    return _d3d11Buffer;
}

bool
OsdCudaD3D11VertexBuffer::allocate(ID3D11Device *device) {

    D3D11_BUFFER_DESC hBufferDesc;
    hBufferDesc.ByteWidth           = _numElements * _numVertices * sizeof(float);
    hBufferDesc.Usage               = D3D11_USAGE_DYNAMIC;
    hBufferDesc.BindFlags           = D3D11_BIND_VERTEX_BUFFER | D3D11_BIND_SHADER_RESOURCE;
    hBufferDesc.CPUAccessFlags      = D3D11_CPU_ACCESS_WRITE;
    hBufferDesc.MiscFlags           = 0;
    hBufferDesc.StructureByteStride = sizeof(float);

    HRESULT hr;
    hr = device->CreateBuffer(&hBufferDesc, NULL, &_d3d11Buffer);
    if(FAILED(hr)) {
        OsdError(OSD_D3D11_VERTEX_BUFFER_CREATE_ERROR,
                 "Fail in CreateBuffer\n");
        return false;
    }
    
    // register d3d11buffer as cuda resource
    cudaError_t err = cudaGraphicsD3D11RegisterResource(
        &_cudaResource, _d3d11Buffer, cudaGraphicsRegisterFlagsNone);

    if (err != cudaSuccess) return false;
    return true;
}

void
OsdCudaD3D11VertexBuffer::map() {

    if (_cudaBuffer) return;
    size_t num_bytes;
    void *ptr;
    
    cudaGraphicsMapResources(1, &_cudaResource, 0);
    cudaGraphicsResourceGetMappedPointer(&ptr, &num_bytes, _cudaResource);
    _cudaBuffer = ptr;
}

void
OsdCudaD3D11VertexBuffer::unmap() {
    
    if (_cudaBuffer == NULL) return;
    cudaGraphicsUnmapResources(1, &_cudaResource, 0);
    _cudaBuffer = NULL;
}


} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv

