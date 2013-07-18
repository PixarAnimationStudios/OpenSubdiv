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

#include "../osd/d3d11PtexTexture.h"
#include "../osd/ptexTextureLoader.h"
#include "../osd/error.h"

#include <Ptexture.h>
#include <D3D11.h>
#include <cassert>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

OsdD3D11PtexTexture::OsdD3D11PtexTexture()
    : _width(0), _height(0), _depth(0), _pages(0), _layout(0), _texels(0) {
}

OsdD3D11PtexTexture::~OsdD3D11PtexTexture() {

    // delete pages lookup ---------------------------------
    if (_pages) _pages->Release();

    // delete layout lookup --------------------------------
    if (_layout) _layout->Release();

    // delete textures lookup ------------------------------
    if (_texels) _texels->Release();
}

static ID3D11Buffer *
genTextureBuffer(ID3D11DeviceContext *deviceContext, int size, void const * data) {

    D3D11_BUFFER_DESC hBufferDesc;
    hBufferDesc.ByteWidth = size;
    hBufferDesc.Usage = D3D11_USAGE_DYNAMIC;
    hBufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER | D3D11_BIND_SHADER_RESOURCE;
    hBufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    hBufferDesc.MiscFlags = 0;
    hBufferDesc.StructureByteStride = sizeof(float);

    HRESULT hr;
    ID3D11Buffer *buffer;
    ID3D11Device *device;
    deviceContext->GetDevice(&device);
    hr = device->CreateBuffer(&hBufferDesc, NULL, &buffer);
    if (FAILED(hr)) {
        OsdError(OSD_D3D11_VERTEX_BUFFER_CREATE_ERROR,
                 "Fail in CreateBuffer\n");
        return 0;
    }

    D3D11_MAPPED_SUBRESOURCE resource;
    hr = deviceContext->Map(buffer, 0,
                            D3D11_MAP_WRITE_DISCARD, 0, &resource);
    if (FAILED(hr)) {
        OsdError(OSD_D3D11_VERTEX_BUFFER_CREATE_ERROR,
                 "Fail in Map buffer\n");
        buffer->Release();
        return 0;
    }
    memcpy(resource.pData, data, size);
    deviceContext->Unmap(buffer, 0);

    return buffer;
}

OsdD3D11PtexTexture *
OsdD3D11PtexTexture::Create(ID3D11DeviceContext *deviceContext,
                         PtexTexture * reader,
                         unsigned long int targetMemory,
                         int gutterWidth,
                         int pageMargin) {

    OsdD3D11PtexTexture * result = NULL;

    // Read the ptex data and pack the texels
    OsdPtexTextureLoader ldr(reader, gutterWidth, pageMargin);

    unsigned long int nativeSize = ldr.GetNativeUncompressedSize(),
           targetSize = targetMemory;

    if (targetSize != 0 && targetSize != nativeSize)
        ldr.OptimizeResolution(targetSize);

    int maxnumpages = D3D10_REQ_TEXTURE2D_ARRAY_AXIS_DIMENSION;
    ldr.OptimizePacking(maxnumpages);

    if (!ldr.GenerateBuffers())
        return result;

    // Setup GPU memory
    unsigned long int nfaces = ldr.GetNumBlocks();

    ID3D11Buffer *pages = genTextureBuffer(deviceContext,
                                           nfaces * sizeof(int),
                                           ldr.GetIndexBuffer());

    ID3D11Buffer *layout = genTextureBuffer(deviceContext,
                                            nfaces * 4 * sizeof(float),
                                            ldr.GetLayoutBuffer());

    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    int bpp = 0;
    int numChannels = reader->numChannels();
    switch (reader->dataType()) {
    case Ptex::dt_uint16:
        switch (numChannels) {
        case 1: format = DXGI_FORMAT_R16_UINT; break;
        case 2: format = DXGI_FORMAT_R16G16_UINT; break;
        case 3: assert(false); break;
        case 4: format = DXGI_FORMAT_R16G16B16A16_UINT; break;
        }
        bpp = numChannels * 2;
        break;
    case Ptex::dt_float:
        switch (numChannels) {
        case 1: format = DXGI_FORMAT_R32_FLOAT; break;
        case 2: format = DXGI_FORMAT_R32G32_FLOAT; break;
        case 3: format = DXGI_FORMAT_R32G32B32_FLOAT; break;
        case 4: format = DXGI_FORMAT_R32G32B32A32_FLOAT; break;
        }
        bpp = numChannels * 4;
        break;
    case Ptex::dt_half:
        switch (numChannels) {
        case 1: format = DXGI_FORMAT_R16_FLOAT; break;
        case 2: format = DXGI_FORMAT_R16G16_FLOAT; break;
        case 3:assert(false); break;
        case 4: format = DXGI_FORMAT_R16G16B16A16_FLOAT; break;
        }
        bpp = numChannels * 2;
        break;
    default:
        switch (numChannels) {
        case 1: format = DXGI_FORMAT_R8_UINT; break;
        case 2: format = DXGI_FORMAT_R8G8_UINT; break;
        case 3: assert(false); break;
        case 4: format = DXGI_FORMAT_R8G8B8A8_UINT; break;
        }
        bpp = numChannels;
        break;
    }

    // actual texels texture array
    D3D11_TEXTURE2D_DESC desc;
    desc.Width = ldr.GetPageSize();
    desc.Height = ldr.GetPageSize();
    desc.MipLevels = 1;
    desc.ArraySize = ldr.GetNumPages();
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    desc.CPUAccessFlags = 0;
    desc.MiscFlags = 0;

    D3D11_SUBRESOURCE_DATA initData;
    initData.pSysMem = ldr.GetTexelBuffer();
    initData.SysMemPitch = ldr.GetPageSize() * bpp;
    initData.SysMemSlicePitch = ldr.GetPageSize() * ldr.GetPageSize() * bpp;

    ID3D11Device *device;
    ID3D11Texture2D *texels;
    deviceContext->GetDevice(&device);
    HRESULT hr = device->CreateTexture2D(&desc, &initData, &texels);

    ldr.ClearBuffers();

    // Return the Osd PtexTexture object
    result = new OsdD3D11PtexTexture;

    result->_width = ldr.GetPageSize();
    result->_height = ldr.GetPageSize();
    result->_depth = ldr.GetNumPages();

    result->_format = format;

    result->_pages = pages;
    result->_layout = layout;
    result->_texels = texels;

    return result;
}

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
