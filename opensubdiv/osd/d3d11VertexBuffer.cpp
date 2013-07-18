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

#include "../osd/d3d11VertexBuffer.h"
#include "../osd/error.h"

#include <D3D11.h>
#include <cassert>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

#define SAFE_RELEASE(p) { if(p) { (p)->Release(); (p)=NULL; } }

OsdD3D11VertexBuffer::OsdD3D11VertexBuffer(int numElements, int numVertices,
                                           ID3D11Device *device)
    : _numElements(numElements), _numVertices(numVertices), _buffer(0), _uploadBuffer(0), _uav(0) {
}

OsdD3D11VertexBuffer::~OsdD3D11VertexBuffer() {

    SAFE_RELEASE(_buffer);
    SAFE_RELEASE(_uploadBuffer);
    SAFE_RELEASE(_uav);
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
OsdD3D11VertexBuffer::UpdateData(const float *src, int startVertex, int numVertices,
                                 void *param) {

    ID3D11DeviceContext * pd3dDeviceContext =
        static_cast<ID3D11DeviceContext*>(param);
    assert(pd3dDeviceContext);

    D3D11_MAPPED_SUBRESOURCE resource;
    HRESULT hr = pd3dDeviceContext->Map(_uploadBuffer, 0,
                                        D3D11_MAP_WRITE_DISCARD, 0, &resource);

    if (FAILED(hr)) {
        OsdError(OSD_D3D11_BUFFER_MAP_ERROR, "Failed to map buffer\n");
        return;
    }

    int size = GetNumElements() * numVertices * sizeof(float);

    memcpy((float*)resource.pData + startVertex * _numElements, src, size);

    pd3dDeviceContext->Unmap(_uploadBuffer, 0);

    D3D11_BOX srcBox = { 0, 0, 0, size, 1, 1 };
    pd3dDeviceContext->CopySubresourceRegion(_buffer, 0, 0, 0, 0,
                                             _uploadBuffer, 0, &srcBox);
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

    return _buffer;
}

ID3D11UnorderedAccessView *
OsdD3D11VertexBuffer::BindD3D11UAV(ID3D11DeviceContext *deviceContext) {

    return _uav;
}

bool
OsdD3D11VertexBuffer::allocate(ID3D11Device *device) {

    D3D11_BUFFER_DESC hBufferDesc;
    hBufferDesc.ByteWidth = _numElements * _numVertices * sizeof(float);
    hBufferDesc.Usage = D3D11_USAGE_DEFAULT;
    hBufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER | D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
    hBufferDesc.CPUAccessFlags = 0;
    hBufferDesc.MiscFlags = 0;
    hBufferDesc.StructureByteStride = sizeof(float);

    HRESULT hr = device->CreateBuffer(&hBufferDesc, NULL, &_buffer);
    if (FAILED(hr)) {
        OsdError(OSD_D3D11_VERTEX_BUFFER_CREATE_ERROR,
                 "Failed to create vertex buffer\n");
        return false;
    }

    hBufferDesc.ByteWidth = _numElements * _numVertices * sizeof(float);
    hBufferDesc.Usage = D3D11_USAGE_DYNAMIC;
    hBufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER | D3D11_BIND_SHADER_RESOURCE;
    hBufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    hBufferDesc.MiscFlags = 0;
    hBufferDesc.StructureByteStride = sizeof(float);

    hr = device->CreateBuffer(&hBufferDesc, NULL, &_uploadBuffer);
    if (FAILED(hr)) {
        OsdError(OSD_D3D11_VERTEX_BUFFER_CREATE_ERROR,
                 "Failed to create upload vertex buffer\n");
        return false;
    }

    D3D11_UNORDERED_ACCESS_VIEW_DESC uavd;
    ZeroMemory(&uavd, sizeof(uavd));
    uavd.Format = DXGI_FORMAT_R32_FLOAT;
    uavd.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
    uavd.Buffer.FirstElement = 0;
    uavd.Buffer.NumElements = _numElements * _numVertices;
    hr = device->CreateUnorderedAccessView(_buffer, &uavd, &_uav);
    if (FAILED(hr)) {
        OsdError(OSD_D3D11_VERTEX_BUFFER_CREATE_ERROR,
                 "Failed to create unordered access resource view\n");
        return false;
    }
    return true;
}

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv

