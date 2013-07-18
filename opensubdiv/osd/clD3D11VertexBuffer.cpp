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

#include "../osd/clD3D11VertexBuffer.h"
#include "../osd/error.h"

#include <D3D11.h>
#include <CL/cl_d3d11.h>
#include <cassert>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

OsdCLD3D11VertexBuffer::OsdCLD3D11VertexBuffer(int numElements, int numVertices,
                                               cl_context clContext, ID3D11Device *device) 
    : _numElements(numElements), _numVertices(numVertices),
      _d3d11Buffer(NULL), _clMemory(NULL), _clQueue(NULL), _clMapped(false) {
;
}

OsdCLD3D11VertexBuffer::~OsdCLD3D11VertexBuffer() {

    unmap();
    clReleaseMemObject(_clMemory);
    _d3d11Buffer->Release();
}

OsdCLD3D11VertexBuffer *
OsdCLD3D11VertexBuffer::Create(int numElements, int numVertices,
                               cl_context clContext, ID3D11Device *device) {
    OsdCLD3D11VertexBuffer *instance =
        new OsdCLD3D11VertexBuffer(numElements, numVertices, clContext, device);
    if (instance->allocate(clContext, device)) return instance;
    delete instance;
    return NULL;
}

void
OsdCLD3D11VertexBuffer::UpdateData(const float *src, int startVertex, int numVertices, cl_command_queue queue) {

    size_t size = numVertices * _numElements * sizeof(float);
    size_t offset = startVertex * _numElements * sizeof(float);

    map(queue);
    clEnqueueWriteBuffer(queue, _clMemory, true, offset, size, src, 0, NULL, NULL);
}

cl_mem
OsdCLD3D11VertexBuffer::BindCLBuffer(cl_command_queue queue) {

    map(queue);
    return _clMemory;
}

ID3D11Buffer *
OsdCLD3D11VertexBuffer::BindD3D11Buffer(ID3D11DeviceContext *deviceContext) {

    unmap();
    return _d3d11Buffer;
}

bool
OsdCLD3D11VertexBuffer::allocate(cl_context clContext, ID3D11Device *device) {

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
        OsdError(OSD_D3D11_VERTEX_BUFFER_CREATE_ERROR, "Fail in CreateBuffer\n");
        return false;
    }

    // register d3d11buffer as cl memory
    cl_int err;
    _clMemory = clCreateFromD3D11BufferKHR(clContext, CL_MEM_READ_WRITE, _d3d11Buffer, &err);

    if (err != CL_SUCCESS) return false;
    return true;
}

void
OsdCLD3D11VertexBuffer::map(cl_command_queue queue) {

    if (_clMapped) return;
    _clQueue = queue;
    clEnqueueAcquireD3D11ObjectsKHR(queue, 1, &_clMemory, 0, 0, 0);
    _clMapped = true;
}

void
OsdCLD3D11VertexBuffer::unmap() {
    
    if (not _clMapped) return;
    clEnqueueReleaseD3D11ObjectsKHR(queue, 1, &_clMemory, 0, 0, 0);
    _clMapped = false;
}


}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv

