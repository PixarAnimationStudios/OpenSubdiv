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

#ifndef OPENSUBDIV_EXAMPLES_D3D11_PTEX_MIPMAP_TEXTURE_H
#define OPENSUBDIV_EXAMPLES_D3D11_PTEX_MIPMAP_TEXTURE_H

#include <opensubdiv/osd/nonCopyable.h>

#include <Ptexture.h>

struct ID3D11Buffer;
struct ID3D11Texture2D;
struct ID3D11DeviceContext;
struct ID3D11ShaderResourceView;

class D3D11PtexMipmapTexture : OpenSubdiv::Osd::NonCopyable<D3D11PtexMipmapTexture> {
public:
    static D3D11PtexMipmapTexture * Create(ID3D11DeviceContext *deviceContext,
                                              PtexTexture * reader,
                                              int maxLevels=10,
                                              size_t targetMemory=0);

    /// Returns GLSL shader snippet to fetch ptex
    static const char *GetShaderSource();

    /// Returns the texture buffer containing the layout of the ptex faces
    /// in the texels texture array.
    ID3D11Buffer *GetLayoutTextureBuffer() const { return _layout; }

    ID3D11ShaderResourceView **GetLayoutSRV() { return &_layoutSRV; }

    /// Returns the texels texture array.
    ID3D11Texture2D *GetTexelsTexture() const { return _texels; }

    ID3D11ShaderResourceView **GetTexelsSRV() { return &_texelsSRV; }

    /// Returns the amount of allocated memory (in byte)
    size_t GetMemoryUsage() const { return _memoryUsage; }

    ~D3D11PtexMipmapTexture();

private:
    D3D11PtexMipmapTexture();

    int _width,   // width / height / depth of the 3D texel buffer
        _height,
        _depth;

    int _format;  // texel color format

    ID3D11Buffer *_layout;     // per-face lookup table
    ID3D11Texture2D *_texels;  // texel data
    ID3D11ShaderResourceView *_layoutSRV;
    ID3D11ShaderResourceView *_texelsSRV;

    size_t _memoryUsage;  // total amount of memory used (estimate)
};

#endif  // OPENSUBDIV_EXAMPLES_D3D11_PTEX_TEXTURE_H
