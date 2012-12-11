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
