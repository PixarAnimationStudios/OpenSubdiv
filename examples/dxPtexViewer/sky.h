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

#include <D3D11.h>
#include <D3Dcompiler.h>

//
// Draws an environment sphere centered on the camera w/ a texture
//
class Sky {

public:

    // Constructor (Sky does not own the texture asset)
    Sky(ID3D11Device * device, ID3D11Texture2D * environmentMap);

    ~Sky();

    void Draw(ID3D11DeviceContext * deviceContext, float const mvp[16]);

private:

    void initialize(ID3D11Device * device);

private:

    int numIndices;

    ID3D11VertexShader * vertexShader;
    ID3D11PixelShader * pixelShader;
    ID3D11Buffer * shaderConstants;

    ID3D11InputLayout * inputLayout;
    ID3D11RasterizerState * rasterizerState;
    ID3D11DepthStencilState * depthStencilState;

    ID3D11Texture2D * texture;
    ID3D11ShaderResourceView * textureSRV;
    ID3D11SamplerState * textureSS;

    ID3D11Buffer * sphere;
    ID3D11Buffer * sphereIndices;
};

