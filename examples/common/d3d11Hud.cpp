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

#include <D3D11.h>
#include <string.h>
#include <stdio.h>
#include <cassert>
#include "d3d11Hud.h"
#include "d3d11Utils.h"
#include "font_image.h"
#include "../common/simple_math.h"

#define SAFE_RELEASE(p) { if(p) { (p)->Release(); (p)=NULL; } }

static const char *s_VS =
    "cbuffer cbPerFrame : register( b0 )\n"
    "{\n"
    "    matrix g_mViewProjection;\n"
    "};\n"
    "struct vertexIn { float2 pos : POSITION0; float3 color : COLOR0; float2 uv : TEXCOORD0; };\n"
    "struct vertexOut { float4 pos : SV_POSITION; float4 color : COLOR0; float2 uv : TEXCOORD0; };\n"
    "vertexOut vs_main(vertexIn IN) {\n"
    "  vertexOut vout;\n"
    "  vout.pos = mul(float4(IN.pos.x, IN.pos.y, 0, 1), g_mViewProjection);\n"
    "  vout.color = float4(IN.color, 1);\n"
    "  vout.uv = IN.uv;\n"
    " return vout;\n"
    "}";

static const char *s_PS =
    "struct pixelIn { float4 pos : SV_POSITION; float4 color : COLOR0; float2 uv : TEXCOORD0; };\n"
    "Texture2D tx : register(t0); \n"
    "SamplerState sm : register(s0); \n"
    "float4 ps_main(pixelIn IN) : SV_Target {\n"
    "  float4 c = tx.Sample(sm, IN.uv);\n"
    "  if( c.a == 0.0 ) \n"
    "   discard;\n"
    "  return IN.color * c;\n"
    "}";

struct CB_HUD_PROJECTION
{
    float mViewProjection[16];
};

D3D11hud::D3D11hud(ID3D11DeviceContext *deviceContext)
    : _deviceContext(deviceContext),
      _vbo(0), _staticVbo(0), _fontTexture(0), _inputLayout(0),
      _shaderResourceView(0), _samplerState(0), _vertexShader(0),
      _pixelShader(0), _rasterizerState(0)
{
}

D3D11hud::~D3D11hud()
{
    SAFE_RELEASE(_vbo);
    SAFE_RELEASE(_staticVbo);
    SAFE_RELEASE(_fontTexture);
    SAFE_RELEASE(_inputLayout);
    SAFE_RELEASE(_shaderResourceView);
    SAFE_RELEASE(_samplerState);
    SAFE_RELEASE(_vertexShader);
    SAFE_RELEASE(_pixelShader);
    SAFE_RELEASE(_rasterizerState);
}

void
D3D11hud::Init(int width, int height, int frameBufferWidth, int frameBufferHeight)
{
    Hud::Init(width, height, frameBufferWidth, frameBufferHeight);

    ID3D11Device *device = NULL;
    _deviceContext->GetDevice(&device);

    // define font texture
    D3D11_TEXTURE2D_DESC texDesc;
    texDesc.Width = FONT_TEXTURE_WIDTH;
    texDesc.Height = FONT_TEXTURE_HEIGHT;
    texDesc.MipLevels = 1;
    texDesc.ArraySize = 1;
    texDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    texDesc.SampleDesc.Count = 1;
    texDesc.SampleDesc.Quality = 0;
    texDesc.Usage = D3D11_USAGE_DEFAULT;
    texDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    texDesc.CPUAccessFlags = 0;
    texDesc.MiscFlags = 0;

    D3D11_SUBRESOURCE_DATA subData;
    subData.pSysMem = font_image;
    subData.SysMemPitch = FONT_TEXTURE_WIDTH*4;
    subData.SysMemSlicePitch = FONT_TEXTURE_WIDTH*FONT_TEXTURE_HEIGHT*4;

    HRESULT hr = device->CreateTexture2D(&texDesc, &subData, &_fontTexture);
    assert(_fontTexture);

    // shader resource view
    D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc;
    ZeroMemory(&srvDesc, sizeof(srvDesc));
    srvDesc.Format = texDesc.Format;
    srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Texture2D.MostDetailedMip = 0;
    srvDesc.Texture2D.MipLevels = texDesc.MipLevels;
    device->CreateShaderResourceView(_fontTexture, &srvDesc, &_shaderResourceView);
    assert(_shaderResourceView);

    D3D11_SAMPLER_DESC samplerDesc;
    ZeroMemory(&samplerDesc, sizeof(samplerDesc));
    samplerDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    samplerDesc.AddressU = samplerDesc.AddressV = samplerDesc.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
    samplerDesc.MaxAnisotropy = 1;
    samplerDesc.ComparisonFunc = D3D11_COMPARISON_ALWAYS;
    samplerDesc.MaxLOD = D3D11_FLOAT32_MAX;
    device->CreateSamplerState(&samplerDesc, &_samplerState);
    assert(_samplerState);

    ID3DBlob* pVSBlob;
    ID3DBlob* pPSBlob;
    pVSBlob = D3D11Utils::CompileShader(s_VS, "vs_main", "vs_4_0");
    pPSBlob = D3D11Utils::CompileShader(s_PS, "ps_main", "ps_4_0");
    assert(pVSBlob);
    assert(pPSBlob);

    D3D11_INPUT_ELEMENT_DESC inputElementDesc[] = {
        { "POSITION", 0, DXGI_FORMAT_R32G32_FLOAT,    0,               0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "COLOR",    0, DXGI_FORMAT_R32G32B32_FLOAT, 0, sizeof(float)*2, D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT,    0, sizeof(float)*5, D3D11_INPUT_PER_VERTEX_DATA, 0 }
    };
    device->CreateInputLayout(inputElementDesc, ARRAYSIZE(inputElementDesc),
                              pVSBlob->GetBufferPointer(),
                              pVSBlob->GetBufferSize(),
                              &_inputLayout);
    assert(_inputLayout);

    device->CreateVertexShader(pVSBlob->GetBufferPointer(),
                               pVSBlob->GetBufferSize(),
                               NULL, &_vertexShader);
    assert(_vertexShader);

    device->CreatePixelShader(pPSBlob->GetBufferPointer(),
                              pPSBlob->GetBufferSize(),
                              NULL, &_pixelShader);
    assert(_pixelShader);

    D3D11_RASTERIZER_DESC rasDesc;
    rasDesc.FillMode = D3D11_FILL_SOLID;
    rasDesc.CullMode = D3D11_CULL_NONE;
    rasDesc.FrontCounterClockwise = FALSE;
    rasDesc.DepthBias = 0;
    rasDesc.DepthBiasClamp = 0;
    rasDesc.SlopeScaledDepthBias = 0.0f;
    rasDesc.DepthClipEnable = FALSE;
    rasDesc.ScissorEnable = FALSE;
    rasDesc.MultisampleEnable = FALSE;
    rasDesc.AntialiasedLineEnable = FALSE;
    device->CreateRasterizerState(&rasDesc, &_rasterizerState);
    assert(_rasterizerState);

    // constant buffer
    D3D11_BUFFER_DESC cbDesc;
    cbDesc.Usage = D3D11_USAGE_DYNAMIC;
    cbDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    cbDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    cbDesc.MiscFlags = 0;
    cbDesc.ByteWidth = sizeof(CB_HUD_PROJECTION);

    device->CreateBuffer(&cbDesc, NULL, &_constantBuffer);
    assert(_constantBuffer);
}

void
D3D11hud::Rebuild(int width, int height, int framebufferWidth, int framebufferHeight)
{
    Hud::Rebuild(width, height, framebufferWidth, framebufferHeight);

    SAFE_RELEASE(_staticVbo);

    int size = (int)getStaticVboSource().size();
    if (size) {
        D3D11_BUFFER_DESC bufferDesc;
        bufferDesc.ByteWidth = size * sizeof(float);
        bufferDesc.Usage = D3D11_USAGE_DEFAULT;
        bufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
        bufferDesc.CPUAccessFlags = 0;
        bufferDesc.MiscFlags = 0;
        bufferDesc.StructureByteStride = 4*sizeof(float);

        D3D11_SUBRESOURCE_DATA subData;
        subData.pSysMem = &getStaticVboSource()[0];
        subData.SysMemPitch = 0;
        subData.SysMemSlicePitch = 0;

        ID3D11Device *device = NULL;
        _deviceContext->GetDevice(&device);
        HRESULT hr = device->CreateBuffer(&bufferDesc, &subData, &_staticVbo);
        assert(_staticVbo);
        _staticVboCount = size / 7;
    }
}

bool
D3D11hud::Flush()
{
    if (!Hud::Flush())
        return false;

    // update dynamic text
    D3D11_BUFFER_DESC bufferDesc;
    bufferDesc.ByteWidth = (int)getVboSource().size() * sizeof(float);
    bufferDesc.Usage = D3D11_USAGE_DEFAULT;
    bufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    bufferDesc.CPUAccessFlags = 0;
    bufferDesc.MiscFlags = 0;
    bufferDesc.StructureByteStride = 4*sizeof(float);

    D3D11_SUBRESOURCE_DATA subData;
    subData.pSysMem = &getVboSource()[0];
    subData.SysMemPitch = 0;
    subData.SysMemSlicePitch = 0;

    SAFE_RELEASE(_vbo);

    ID3D11Device *device = NULL;
    _deviceContext->GetDevice(&device);
    HRESULT hr = device->CreateBuffer(&bufferDesc, &subData, &_vbo);
    assert(_vbo);
    int numVertices = (int)getVboSource().size()/7;  /* (x, y, r, g, b, u, v) = 7*/

    // reserved space of the vector remains for the next frame.
    getVboSource().clear();

    D3D11_MAPPED_SUBRESOURCE MappedResource;
    _deviceContext->Map(_constantBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &MappedResource);
    CB_HUD_PROJECTION * pData = (CB_HUD_PROJECTION *)MappedResource.pData;

    ortho(pData->mViewProjection, 0, 0, (float)GetWidth(), (float)GetHeight());
    transpose(pData->mViewProjection);

    _deviceContext->Unmap( _constantBuffer, 0 );

    // setup graphics pipeline
    _deviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    _deviceContext->IASetInputLayout(_inputLayout);
    _deviceContext->VSSetShader(_vertexShader, NULL, 0);
    _deviceContext->HSSetShader(NULL, NULL, 0);
    _deviceContext->DSSetShader(NULL, NULL, 0);
    _deviceContext->GSSetShader(NULL, NULL, 0);
    _deviceContext->PSSetShader(_pixelShader, NULL, 0);
    _deviceContext->PSSetShaderResources(0, 1, &_shaderResourceView);
    _deviceContext->PSSetSamplers(0, 1, &_samplerState);
    _deviceContext->RSSetState(_rasterizerState);
    _deviceContext->VSSetConstantBuffers(0, 1, &_constantBuffer);

    UINT strides = 7*sizeof(float);
    UINT offsets = 0;
    _deviceContext->IASetVertexBuffers(0, 1, &_vbo, &strides, &offsets);
    _deviceContext->Draw(numVertices, 0);
    _deviceContext->IASetVertexBuffers(0, 1, &_staticVbo, &strides, &offsets);
    _deviceContext->Draw(_staticVboCount, 0);

    return true;
}
