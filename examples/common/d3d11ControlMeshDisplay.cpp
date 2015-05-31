//
//   Copyright 2015 Pixar
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

#include "d3d11ControlMeshDisplay.h"
#include "../common/d3d11Utils.h"

#include <vector>

#define SAFE_RELEASE(p) { if(p) { (p)->Release(); (p)=NULL; } }

static const char *s_VS =
"cbuffer cbPerFrame : register( b0 ) { matrix g_mViewProjection; };        \n"
"struct vertexIn { float3 pos : POSITION0; float sharpness : COLOR0; };    \n"
"struct vertexOut { float4 pos : SV_POSITION; float sharpness : COLOR0; }; \n"
"vertexOut vs_main(vertexIn IN) {                                          \n"
"  vertexOut vout;                                                         \n"
"  vout.pos = mul(g_mViewProjection, float4(IN.pos, 1));                   \n"
"  vout.sharpness = IN.sharpness;                                          \n"
"  return vout;                                                            \n"
"}                                                                         \n";

static const char *s_PS =
"struct pixelIn { float4 pos : SV_POSITION; float sharpness : COLOR0; };   \n"
"Buffer<float> edgeSharpness : register(t0);                               \n"
"float4 sharpnessToColor(float s) {                                        \n"
"  //  0.0       2.0       4.0                                             \n"
"  // green --- yellow --- red                                             \n"
"  return float4(min(1, s * 0.5),                                          \n"
"                min(1, 2 - s * 0.5),                                      \n"
"                0, 1);                                                    \n"
"}                                                                         \n"
"float4 ps_main(pixelIn IN, uint primitiveID : SV_PrimitiveID)             \n"
"  : SV_Target {                                                           \n"
"  float sharpness = edgeSharpness[primitiveID];                           \n"
"  return sharpnessToColor(sharpness);                                     \n"
"}                                                                         \n";

namespace {
    struct CB_CONTROL_MESH_DISPLAY
    {
        float mViewProjection[16];
    };
};

D3D11ControlMeshDisplay::D3D11ControlMeshDisplay(
    ID3D11DeviceContext *deviceContext) :
    _displayEdges(true), _displayVertices(false),
    _deviceContext(deviceContext), _inputLayout(0), _vertexShader(0),
    _pixelShader(0), _rasterizerState(0), _constantBuffer(0),
    _edgeSharpnessSRV(0), _edgeSharpness(0), _edgeIndices(0),
    _numEdges(0), _numPoints(0) {
}

D3D11ControlMeshDisplay::~D3D11ControlMeshDisplay() {
    SAFE_RELEASE(_inputLayout);
    SAFE_RELEASE(_vertexShader);
    SAFE_RELEASE(_pixelShader);
    SAFE_RELEASE(_rasterizerState);
    SAFE_RELEASE(_constantBuffer);
    SAFE_RELEASE(_edgeSharpness);
    SAFE_RELEASE(_edgeSharpnessSRV);
    SAFE_RELEASE(_edgeIndices);
}

bool
D3D11ControlMeshDisplay::createProgram() {
    ID3D11Device *device = NULL;
    _deviceContext->GetDevice(&device);

    ID3DBlob* pVSBlob;
    ID3DBlob* pPSBlob;
    pVSBlob = D3D11Utils::CompileShader(s_VS, "vs_main", "vs_4_0");
    pPSBlob = D3D11Utils::CompileShader(s_PS, "ps_main", "ps_4_0");
    assert(pVSBlob);
    assert(pPSBlob);

    D3D11_INPUT_ELEMENT_DESC inputElementDesc[] = {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0,               0,
          D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "COLOR",    0, DXGI_FORMAT_R32_FLOAT,       0, sizeof(float)*3,
          D3D11_INPUT_PER_VERTEX_DATA, 0 },
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
    cbDesc.ByteWidth = sizeof(CB_CONTROL_MESH_DISPLAY);

    device->CreateBuffer(&cbDesc, NULL, &_constantBuffer);
    assert(_constantBuffer);
    return true;
}


void
D3D11ControlMeshDisplay::Draw(ID3D11Buffer *buffer, int stride,
                              const float *modelViewProjectionMatrix) {

    if (_displayEdges == false && _displayVertices == false) return;

    if (_vertexShader == NULL) createProgram();
    if (_vertexShader == NULL) return;

    UINT hStrides = stride*sizeof(float);
    UINT hOffsets = 0;
    _deviceContext->IASetVertexBuffers(0, 1, &buffer, &hStrides, &hOffsets);

    D3D11_MAPPED_SUBRESOURCE MappedResource;
    _deviceContext->Map(_constantBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &MappedResource);
    CB_CONTROL_MESH_DISPLAY * pData = (CB_CONTROL_MESH_DISPLAY *)MappedResource.pData;
    memcpy(pData->mViewProjection, modelViewProjectionMatrix, sizeof(float) * 16);
    _deviceContext->Unmap(_constantBuffer, 0);

    _deviceContext->IASetInputLayout(_inputLayout);
    _deviceContext->VSSetShader(_vertexShader, NULL, 0);
    _deviceContext->HSSetShader(NULL, NULL, 0);
    _deviceContext->DSSetShader(NULL, NULL, 0);
    _deviceContext->GSSetShader(NULL, NULL, 0);
    _deviceContext->PSSetShaderResources(0, 1, &_edgeSharpnessSRV);
    _deviceContext->PSSetShader(_pixelShader, NULL, 0);
    _deviceContext->RSSetState(_rasterizerState);
    _deviceContext->VSSetConstantBuffers(0, 1, &_constantBuffer);

    if (_displayEdges) {
        _deviceContext->IASetIndexBuffer(_edgeIndices, DXGI_FORMAT_R32_UINT, 0);
        _deviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_LINELIST);
        _deviceContext->DrawIndexed(_numEdges * 2, 0, 0);
    }
    if (_displayVertices) {
        // TODO: need geometry shader to draw bigger points
        _deviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_POINTLIST);
        _deviceContext->Draw(_numPoints, 0);
    }

}

void
D3D11ControlMeshDisplay::SetTopology(
    OpenSubdiv::Far::TopologyLevel const &level) {
    int nEdges = level.GetNumEdges();
    int nVerts = level.GetNumVertices();

    std::vector<int> edgeIndices;
    std::vector<float> edgeSharpnesses;
    std::vector<float> vertSharpnesses;

    edgeIndices.reserve(nEdges * 2);
    edgeSharpnesses.reserve(nEdges);
    vertSharpnesses.reserve(nVerts);

    for (int i = 0; i < nEdges; ++i) {
        OpenSubdiv::Far::ConstIndexArray verts = level.GetEdgeVertices(i);
        edgeIndices.push_back(verts[0]);
        edgeIndices.push_back(verts[1]);
        edgeSharpnesses.push_back(level.GetEdgeSharpness(i));
    }

    for (int i = 0; i < nVerts; ++i) {
        vertSharpnesses.push_back(level.GetVertexSharpness(i));
    }

    _numEdges = nEdges;
    _numPoints = nVerts;

    ID3D11Device *device = NULL;
    _deviceContext->GetDevice(&device);

    SAFE_RELEASE(_edgeIndices);
    // create edge index buffer
    D3D11_BUFFER_DESC bufferDesc;
    ZeroMemory(&bufferDesc, sizeof(bufferDesc));
    bufferDesc.ByteWidth = (int)edgeIndices.size() * sizeof(int);
    bufferDesc.Usage = D3D11_USAGE_DEFAULT;
    bufferDesc.BindFlags = D3D11_BIND_INDEX_BUFFER;
    bufferDesc.CPUAccessFlags = 0;
    bufferDesc.MiscFlags = 0;
    bufferDesc.StructureByteStride = sizeof(int);

    D3D11_SUBRESOURCE_DATA subData;
    ZeroMemory(&subData, sizeof(subData));
    subData.pSysMem = &edgeIndices[0];
    subData.SysMemPitch = 0;
    subData.SysMemSlicePitch = 0;

    HRESULT hr = device->CreateBuffer(&bufferDesc, &subData, &_edgeIndices);
    assert(_edgeIndices);

    // edge sharpness
    SAFE_RELEASE(_edgeSharpness);
    SAFE_RELEASE(_edgeSharpnessSRV);
    ZeroMemory(&bufferDesc, sizeof(bufferDesc));
    bufferDesc.ByteWidth = (int)edgeSharpnesses.size() * sizeof(float);
    bufferDesc.Usage = D3D11_USAGE_DEFAULT;
    bufferDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    bufferDesc.CPUAccessFlags = 0;
    bufferDesc.MiscFlags = 0;
    bufferDesc.StructureByteStride = sizeof(float);

    ZeroMemory(&subData, sizeof(subData));
    subData.pSysMem = &edgeSharpnesses[0];
    subData.SysMemPitch = 0;
    subData.SysMemSlicePitch = 0;

    hr = device->CreateBuffer(&bufferDesc, &subData, &_edgeSharpness);
    assert(_edgeSharpness);

    D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc;
    ZeroMemory(&srvDesc, sizeof(srvDesc));
    srvDesc.Format = DXGI_FORMAT_R32_FLOAT;
    srvDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
    srvDesc.Buffer.NumElements = _numEdges;
    hr = device->CreateShaderResourceView(_edgeSharpness, &srvDesc, &_edgeSharpnessSRV);
    assert(_edgeSharpnessSRV);
}

