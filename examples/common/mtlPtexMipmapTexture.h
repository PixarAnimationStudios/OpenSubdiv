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

#ifndef OPENSUBDIV_EXAMPLES_MTL_PTEX_MIPMAP_TEXTURE_H
#define OPENSUBDIV_EXAMPLES_MTL_PTEX_MIPMAP_TEXTURE_H

#include <opensubdiv/osd/mtlCommon.h>
#include <opensubdiv/osd/nonCopyable.h>
#include <Ptexture.h>


@protocol MTLBuffer;
@protocol MTLTexture;
@class MTLTextureDescriptor;

class MTLPtexMipmapTexture : OpenSubdiv::Osd::NonCopyable<MTLPtexMipmapTexture> {
public:
    static MTLPtexMipmapTexture * Create(OpenSubdiv::Osd::MTLContext * deviceContext,
                                        PtexTexture * reader,
                                        int maxLevels = 10);
    
    
    static const char* GetShaderSource();
    
    id<MTLBuffer> GetLayoutBuffer() const { return _layout; }
    id<MTLTexture> GetTexelsTexture() const { return _texels; }
    
private:
    MTLPtexMipmapTexture();
    
    id<MTLBuffer> _layout;
    id<MTLTexture> _texels;

    MTLTextureDescriptor* _textureDescriptor;
};

#endif  // OPENSUBDIV_EXAMPLES_MTL_PTEX_TEXTURE_H
