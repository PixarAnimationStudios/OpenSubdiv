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
OsdCLD3D11VertexBuffer::UpdateData(const float *src, int numVertices, void *param) {

    cl_command_queue queue = *(cl_command_queue*)param;

    size_t size = _numVertices * _numElements * sizeof(float);

    map(queue);
    clEnqueueWriteBuffer(queue, _clMemory, true, 0, size, src, 0, NULL, NULL);
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

