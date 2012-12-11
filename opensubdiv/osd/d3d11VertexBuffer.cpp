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

#include "../osd/d3d11VertexBuffer.h"
#include "../osd/error.h"

#include <D3D11.h>
#include <cassert>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

OsdD3D11VertexBuffer::OsdD3D11VertexBuffer(int numElements, int numVertices,
                                           ID3D11Device *device)
    : _numElements(numElements), _numVertices(numVertices), _d3d11Buffer(0) {
}

OsdD3D11VertexBuffer::~OsdD3D11VertexBuffer() {

    if (_d3d11Buffer) _d3d11Buffer->Release();
}

OsdD3D11VertexBuffer*
OsdD3D11VertexBuffer::Create(int numElements, int numVertices,
                             ID3D11Device *device) {
    OsdD3D11VertexBuffer *instance =
        new OsdD3D11VertexBuffer(numElements, numVertices, device);
    if (instance->allocate(device)) return instance;
    delete instance;
    return NULL;
}

void
OsdD3D11VertexBuffer::UpdateData(const float *src, int numVertices,
                                 void *param) {

    ID3D11DeviceContext * pd3dDeviceContext =
        static_cast<ID3D11DeviceContext*>(param);
    assert(pd3dDeviceContext);

    D3D11_MAPPED_SUBRESOURCE resource;
    HRESULT hr = pd3dDeviceContext->Map(_d3d11Buffer, 0,
                                        D3D11_MAP_WRITE_DISCARD, 0, &resource);

    if (FAILED(hr)) {
        OsdError(OSD_D3D11_BUFFER_MAP_ERROR, "Fail to map buffer\n");
        return;
    }

    int size = GetNumElements() * numVertices * sizeof(float);

    memcpy(resource.pData, src, size);

    pd3dDeviceContext->Unmap(_d3d11Buffer, 0);
}

int
OsdD3D11VertexBuffer::GetNumElements() const {

    return _numElements;
}

int
OsdD3D11VertexBuffer::GetNumVertices() const {

    return _numVertices;
}

ID3D11Buffer *
OsdD3D11VertexBuffer::BindD3D11Buffer(ID3D11DeviceContext *deviceContext) {

    return _d3d11Buffer;
}

bool
OsdD3D11VertexBuffer::allocate(ID3D11Device *device) {

    D3D11_BUFFER_DESC hBufferDesc;
    hBufferDesc.ByteWidth = _numElements * _numVertices * sizeof(float);
    hBufferDesc.Usage = D3D11_USAGE_DYNAMIC;
    hBufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER | D3D11_BIND_SHADER_RESOURCE;
    hBufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    hBufferDesc.MiscFlags = 0;
    hBufferDesc.StructureByteStride = sizeof(float);

    HRESULT hr;
    hr = device->CreateBuffer(&hBufferDesc, NULL, &_d3d11Buffer);
    if (FAILED(hr)) {
        OsdError(OSD_D3D11_VERTEX_BUFFER_CREATE_ERROR,
                 "Fail in CreateBuffer\n");
        return false;
    }
    return true;
}

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv

