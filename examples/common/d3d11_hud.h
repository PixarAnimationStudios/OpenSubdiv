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
#ifndef D3D11_HUD_H
#define D3D11_HUD_H

#include <D3D11.h>
#include "hud.h"

class D3D11hud : public Hud
{
public:
    D3D11hud(ID3D11DeviceContext *deviceContext);
    ~D3D11hud();

    virtual void Init(int width, int height);

    virtual void Rebuild(int width, int height);

    virtual bool Flush();

private:
    ID3D11DeviceContext *_deviceContext;
    ID3D11Buffer *_vbo;
    ID3D11Buffer *_staticVbo;
    ID3D11Texture2D *_fontTexture;
    ID3D11InputLayout *_inputLayout;
    ID3D11ShaderResourceView *_shaderResourceView;
    ID3D11SamplerState *_samplerState;
    ID3D11VertexShader *_vertexShader;
    ID3D11PixelShader *_pixelShader;
    ID3D11RasterizerState *_rasterizerState;
    ID3D11Buffer* _constantBuffer;
    int _staticVboCount;
};

#endif // D3D11_HUD_H
