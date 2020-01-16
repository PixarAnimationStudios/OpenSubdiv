//
//   Copyright 2013 Nvidia
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


#include "./sky.h"

#include "../common/d3d11Utils.h"

#include <cassert>
#include <vector>

static const char *g_skyShaderSource =
#include "skyshader.gen.h"
;

#define SAFE_RELEASE(p) { if(p) { (p)->Release(); (p)=NULL; } }

// shader constants
__declspec(align(16)) struct CB_CONSTANTS {
        float ModelViewMatrix[16];
};


Sky::Sky(ID3D11Device * device, ID3D11Texture2D * environmentMap) :
    numIndices(0), 
    vertexShader(0),
    pixelShader(0),
    shaderConstants(0),
    texture(environmentMap), // we do not own this - we do not release it !
    textureSRV(0),
    textureSS(0),
    inputLayout(0),
    rasterizerState(0),
    depthStencilState(0),
    sphere(0),
    sphereIndices(0) { 

    initialize(device);
}

Sky::~Sky() {
    SAFE_RELEASE(vertexShader);
    SAFE_RELEASE(pixelShader);
    SAFE_RELEASE(shaderConstants);
    SAFE_RELEASE(inputLayout);
    SAFE_RELEASE(rasterizerState);
    SAFE_RELEASE(depthStencilState);
    SAFE_RELEASE(textureSS);
    SAFE_RELEASE(textureSRV);
    SAFE_RELEASE(sphere);
    SAFE_RELEASE(sphereIndices);
}

void
Sky::initialize(ID3D11Device * device) {

    // compile shaders
    ID3DBlob * pVSBlob = D3D11Utils::CompileShader(g_skyShaderSource, "vs_main", "vs_5_0"),
             * pPSBlob = D3D11Utils::CompileShader(g_skyShaderSource, "ps_main", "ps_5_0");
    assert(pVSBlob && pPSBlob);

    device->CreateVertexShader(pVSBlob->GetBufferPointer(),
        pVSBlob->GetBufferSize(), NULL, &vertexShader);
    assert(vertexShader);

    device->CreatePixelShader(pPSBlob->GetBufferPointer(),
        pPSBlob->GetBufferSize(), NULL, &pixelShader);
    assert(pixelShader);

    // VBO layout
    D3D11_INPUT_ELEMENT_DESC inputElementDesc[] = {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, sizeof(float)*3, D3D11_INPUT_PER_VERTEX_DATA, 0 },
    };
    device->CreateInputLayout(inputElementDesc, ARRAYSIZE(inputElementDesc),
        pVSBlob->GetBufferPointer(), pVSBlob->GetBufferSize(), &inputLayout);
    assert(inputLayout);

    // shader constants
    D3D11_BUFFER_DESC cbDesc;
    ZeroMemory(&cbDesc, sizeof(cbDesc));
    cbDesc.Usage = D3D11_USAGE_DYNAMIC;
    cbDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    cbDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    cbDesc.MiscFlags = 0;
    cbDesc.ByteWidth = sizeof(CB_CONSTANTS);
    device->CreateBuffer(&cbDesc, NULL, &shaderConstants);
    assert(shaderConstants);

    // texture SRV
    assert(texture);
    D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc;
    ZeroMemory(&srvDesc, sizeof(srvDesc));
    srvDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Texture2D.MostDetailedMip = 0;
    srvDesc.Texture2D.MipLevels = 1;
    device->CreateShaderResourceView(texture, &srvDesc, &textureSRV);
    assert(textureSRV);

    // texture sampler
    D3D11_SAMPLER_DESC samplerDesc;
    ZeroMemory(&samplerDesc, sizeof(samplerDesc));
    samplerDesc.Filter = D3D11_FILTER_MIN_MAG_LINEAR_MIP_POINT;
    samplerDesc.AddressU = samplerDesc.AddressV = samplerDesc.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
    samplerDesc.MaxAnisotropy = 0;
    samplerDesc.ComparisonFunc = D3D11_COMPARISON_NEVER;
    samplerDesc.MinLOD = 0;
    samplerDesc.MaxLOD = D3D11_FLOAT32_MAX;
    samplerDesc.BorderColor[0] = samplerDesc.BorderColor[1] = samplerDesc.BorderColor[2] = samplerDesc.BorderColor[3] = 0.0f;
    device->CreateSamplerState(&samplerDesc, &textureSS);

    // depth stencil state
    D3D11_DEPTH_STENCIL_DESC depthStencilDesc;
    ZeroMemory(&depthStencilDesc, sizeof(depthStencilDesc));
    depthStencilDesc.DepthEnable = true;
    depthStencilDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO;
    depthStencilDesc.DepthFunc = D3D11_COMPARISON_LESS_EQUAL;
    depthStencilDesc.StencilEnable = false;
    device->CreateDepthStencilState(&depthStencilDesc, &depthStencilState);

    // rasterizer state
    D3D11_RASTERIZER_DESC rasDesc;
    rasDesc.FillMode = D3D11_FILL_SOLID;
    rasDesc.CullMode = D3D11_CULL_NONE;
    rasDesc.FrontCounterClockwise = FALSE;
    rasDesc.DepthBias = 0;
    rasDesc.DepthBiasClamp = 0;
    rasDesc.DepthClipEnable = FALSE;
    rasDesc.SlopeScaledDepthBias = 0.0f;
    rasDesc.ScissorEnable = FALSE;
    rasDesc.MultisampleEnable = FALSE;
    rasDesc.AntialiasedLineEnable = FALSE;
    device->CreateRasterizerState(&rasDesc, &rasterizerState);
    assert(rasterizerState);

    const int U_DIV = 20,
              V_DIV = 20;

    std::vector<float> vbo;
    std::vector<int> indices;
    for (int u = 0; u <= U_DIV; ++u) {
        for (int v = 0; v < V_DIV; ++v) {
            float s = float(2*M_PI*float(u)/U_DIV);
            float t = float(M_PI*float(v)/(V_DIV-1));
            vbo.push_back(-sinf(t)*sinf(s));
            vbo.push_back(cosf(t));
            vbo.push_back(-sinf(t)*cosf(s));
            vbo.push_back(u/float(U_DIV));
            vbo.push_back(v/float(V_DIV));

            if (v > 0 && u > 0) {
                indices.push_back((u-1)*V_DIV+v-1);
                indices.push_back(u*V_DIV+v-1);
                indices.push_back((u-1)*V_DIV+v);
                indices.push_back((u-1)*V_DIV+v);
                indices.push_back(u*V_DIV+v-1);
                indices.push_back(u*V_DIV+v);
            }
        }
    }

    D3D11_BUFFER_DESC bufferDesc;
    D3D11_SUBRESOURCE_DATA subData;

    // topology indices
    ZeroMemory(&bufferDesc, sizeof(bufferDesc));
    bufferDesc.ByteWidth = (int)indices.size() * sizeof(int);
    bufferDesc.Usage = D3D11_USAGE_DEFAULT;
    bufferDesc.BindFlags = D3D11_BIND_INDEX_BUFFER;
    bufferDesc.CPUAccessFlags = 0;
    bufferDesc.MiscFlags = 0;
    bufferDesc.StructureByteStride = sizeof(int);

    ZeroMemory(&subData, sizeof(subData));
    subData.pSysMem = &indices[0];
    subData.SysMemPitch = 0;
    subData.SysMemSlicePitch = 0;
    device->CreateBuffer(&bufferDesc, &subData, &sphereIndices);
    assert(sphereIndices);

    // VBO
    ZeroMemory(&bufferDesc, sizeof(bufferDesc));
    bufferDesc.ByteWidth = (int)vbo.size() * sizeof(float);
    bufferDesc.Usage = D3D11_USAGE_DEFAULT;
    bufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    bufferDesc.CPUAccessFlags = 0;
    bufferDesc.MiscFlags = 0;

    ZeroMemory(&subData, sizeof(subData));
    subData.pSysMem = &vbo[0];
    device->CreateBuffer(&bufferDesc, &subData, &sphere);
    assert(sphere);

    numIndices = (int)indices.size();
}

void
Sky::Draw(ID3D11DeviceContext * deviceContext, float const mvp[16]) {

    if (vertexShader==0 || pixelShader==0 || shaderConstants==0) return;

    if (texture==0 || textureSRV==0 || textureSS==0) return;

    if (sphere==0 || sphereIndices==0) return;

    // update shader constants
    D3D11_MAPPED_SUBRESOURCE MappedResource;
    deviceContext->Map(shaderConstants, 0, D3D11_MAP_WRITE_DISCARD, 0, &MappedResource);
    CB_CONSTANTS* pData = (CB_CONSTANTS*)MappedResource.pData;

    memcpy(pData->ModelViewMatrix, mvp, 16*sizeof(float));

    deviceContext->Unmap(shaderConstants, 0);

    // draw
    deviceContext->RSSetState(rasterizerState);
    deviceContext->OMSetDepthStencilState(depthStencilState, 1);

    deviceContext->VSSetShader(vertexShader, NULL, 0);
    deviceContext->VSSetConstantBuffers(0, 1, &shaderConstants);

    deviceContext->PSSetShader(pixelShader, NULL, 0);
    deviceContext->PSSetShaderResources(0, 1, &textureSRV);
    deviceContext->PSSetSamplers(0, 1, &textureSS);

    UINT hStrides = 5*sizeof(float);
    UINT hOffsets = 0;
    deviceContext->IASetVertexBuffers(0, 1, &sphere, &hStrides, &hOffsets);
    deviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    deviceContext->IASetInputLayout(inputLayout);

    deviceContext->IASetIndexBuffer(sphereIndices, DXGI_FORMAT_R32_UINT, 0);

    deviceContext->DrawIndexed(numIndices, 0, 0);
}
