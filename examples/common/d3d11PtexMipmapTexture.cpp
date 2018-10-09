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

#include "d3d11PtexMipmapTexture.h"
#include "ptexMipmapTextureLoader.h"
#include <opensubdiv/far/error.h>  // XXX: to be replaced

#include <D3D11.h>
#include <cassert>

D3D11PtexMipmapTexture::D3D11PtexMipmapTexture()
    : _width(0), _height(0), _depth(0),
      _layout(0), _texels(0),
      _layoutSRV(0), _texelsSRV(0),
      _memoryUsage(0)
{
}

D3D11PtexMipmapTexture::~D3D11PtexMipmapTexture()
{
    if (_layout) _layout->Release();
    if (_layoutSRV) _layoutSRV->Release();

    if (_texels) _texels->Release();
    if (_texelsSRV) _texelsSRV->Release();
}

/*static*/
const char *
D3D11PtexMipmapTexture::GetShaderSource()
{
    static const char *ptexShaderSource =
#include "hlslPtexCommon.gen.h"
        ;
    return ptexShaderSource;
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
        OpenSubdiv::Far::Error(OpenSubdiv::Far::FAR_RUNTIME_ERROR,
                 "Fail in CreateBuffer\n");
        return 0;
    }

    D3D11_MAPPED_SUBRESOURCE resource;
    hr = deviceContext->Map(buffer, 0,
                            D3D11_MAP_WRITE_DISCARD, 0, &resource);
    if (FAILED(hr)) {
        OpenSubdiv::Far::Error(OpenSubdiv::Far::FAR_RUNTIME_ERROR,
                 "Fail in Map buffer\n");
        buffer->Release();
        return 0;
    }
    memcpy(resource.pData, data, size);
    deviceContext->Unmap(buffer, 0);

    return buffer;
}

D3D11PtexMipmapTexture *
D3D11PtexMipmapTexture::Create(ID3D11DeviceContext *deviceContext,
                                  PtexTexture * reader,
                                  int maxLevels,
                                  size_t targetMemory) {

    D3D11PtexMipmapTexture * result = NULL;

    int maxNumPages = D3D10_REQ_TEXTURE2D_ARRAY_AXIS_DIMENSION;

    // Read the ptex data and pack the texels
    bool padAlpha = reader->numChannels()==3 && reader->dataType()!=Ptex::dt_float;

    PtexMipmapTextureLoader loader(reader,
                                   maxNumPages,
                                   maxLevels,
                                   targetMemory,
                                   true, // seamlessMipmap
                                   padAlpha);

    int numChannels = reader->numChannels() + padAlpha,
        numFaces = loader.GetNumFaces();

    ID3D11Buffer *layout = genTextureBuffer(deviceContext,
                                            numFaces * 6 * sizeof(short),
                                            loader.GetLayoutBuffer());
    if (!layout) return NULL;

    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    int bpp = 0;
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
                case 3: assert(false); break;
                case 4: format = DXGI_FORMAT_R16G16B16A16_FLOAT; break;
            }
            bpp = numChannels * 2;
            break;
        default:
            switch (numChannels) {
                case 1: format = DXGI_FORMAT_R8_UNORM; break;
                case 2: format = DXGI_FORMAT_R8G8_UNORM; break;
                case 3: assert(false); break;
                case 4: format = DXGI_FORMAT_R8G8B8A8_UNORM; break;
            }
            bpp = numChannels;
            break;
    }

    // actual texels texture array
    D3D11_TEXTURE2D_DESC desc;
    desc.Width = loader.GetPageWidth();
    desc.Height = loader.GetPageHeight();
    desc.MipLevels = 1;
    desc.ArraySize = loader.GetNumPages();
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    desc.CPUAccessFlags = 0;
    desc.MiscFlags = 0;

    D3D11_SUBRESOURCE_DATA *initData = new D3D11_SUBRESOURCE_DATA[desc.ArraySize];
    int pageStride = loader.GetPageWidth()*loader.GetPageHeight() * bpp;
    for (unsigned int i = 0; i < desc.ArraySize; ++i) {
        initData[i].pSysMem = loader.GetTexelBuffer() + i * pageStride;
        initData[i].SysMemPitch = loader.GetPageWidth() * bpp;
        initData[i].SysMemSlicePitch = pageStride;
    }

    ID3D11Device *device;
    ID3D11Texture2D *texels;
    deviceContext->GetDevice(&device);
    HRESULT hr = device->CreateTexture2D(&desc, initData, &texels);
    if (FAILED(hr)) return NULL;

    delete[] initData;

    // create SRV
    ID3D11ShaderResourceView *layoutSRV;
    D3D11_SHADER_RESOURCE_VIEW_DESC srvd;
    ZeroMemory(&srvd, sizeof(srvd));
    srvd.Format = DXGI_FORMAT_R16_UINT;
    srvd.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
    srvd.Buffer.FirstElement = 0;
    srvd.Buffer.NumElements = numFaces * 6;
    hr = device->CreateShaderResourceView(layout, &srvd, &layoutSRV);
    if (FAILED(hr)) return NULL;

    ID3D11ShaderResourceView *texelsSRV;
    ZeroMemory(&srvd, sizeof(srvd));
    srvd.Format = format;
    srvd.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DARRAY;
    srvd.Texture2DArray.MipLevels = 1;
    srvd.Texture2DArray.ArraySize = desc.ArraySize;
    hr = device->CreateShaderResourceView(texels, &srvd, &texelsSRV);
    if (FAILED(hr)) return NULL;


    result = new D3D11PtexMipmapTexture;

    result->_width = loader.GetPageWidth();
    result->_height = loader.GetPageHeight();
    result->_depth = loader.GetNumPages();

    result->_format = format;

    result->_layout = layout;
    result->_texels = texels;

    result->_layoutSRV = layoutSRV;
    result->_texelsSRV = texelsSRV;
    result->_memoryUsage = loader.GetMemoryUsage();

    return result;
}
