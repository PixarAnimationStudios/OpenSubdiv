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

#include "mtlPtexMipmapTexture.h"
#include "ptexMipmapTextureLoader.h"
#include <opensubdiv/far/error.h>
#include <Metal/Metal.h>
#include <cassert>

#include "mtlUtils.h"

using namespace OpenSubdiv;

MTLPtexMipmapTexture::MTLPtexMipmapTexture() :
    _layout(nil), _texels(nil), _textureDescriptor(nil) {
}

const char* MTLPtexMipmapTexture::GetShaderSource() {
    static const char* shaderSource =
#include "mtlPtexCommon.gen.h"
    ;
    return shaderSource;
}

MTLPtexMipmapTexture * MTLPtexMipmapTexture::Create(Osd::MTLContext *deviceContext, PtexTexture *reader, int maxLevels) {
    const auto maxNumPages = 2048;

    size_t targetMemory = 0;

    // Read the ptex data and pack the texels
    bool padAlpha = reader->numChannels()==3;

    PtexMipmapTextureLoader loader(reader,
                                   maxNumPages,
                                   maxLevels,
                                   targetMemory,
                                   true/*seemlessMipmap*/,
                                   padAlpha);
    const auto numFaces = loader.GetNumFaces();
    
    const auto layoutBuffer = [deviceContext->device newBufferWithBytes:loader.GetLayoutBuffer() length:numFaces * 6 * sizeof(short) options:Osd::MTLDefaultStorageMode];
    
    const auto textureDescriptor = [MTLTextureDescriptor new];
    int bpp = 0;
    textureDescriptor.pixelFormat = [&]() {
        const auto numChannels = reader->numChannels() + padAlpha;
        switch(reader->dataType()) {
            case Ptex::dt_uint16:
                bpp = sizeof(short) * numChannels;
                switch(numChannels) {
                    case 1: return MTLPixelFormatR16Unorm;
                    case 2: return MTLPixelFormatRG16Unorm;
                    case 3: assert("RGB16Uint not supported"); break;
                    case 4: return MTLPixelFormatRGBA16Unorm;
                }
            case Ptex::dt_float:
                bpp = sizeof(float) * numChannels;
                switch(numChannels) {
                    case 1: return MTLPixelFormatR32Float;
                    case 2: return MTLPixelFormatRG32Float;
                    case 3: assert("RGB32Float not supported"); break;
                    case 4: return MTLPixelFormatRGBA32Float;
                }
            case Ptex::dt_half:
                bpp = sizeof(short) * numChannels;
                switch(numChannels) {
                    case 1: return MTLPixelFormatR16Float;
                    case 2: return MTLPixelFormatRG16Float;
                    case 3: assert("RGB16Float not supported"); break;
                    case 4: return MTLPixelFormatRGBA16Float;
                }
            default:
                bpp = numChannels;
                switch(numChannels) {
                    case 1: return MTLPixelFormatR8Unorm;
                    case 2: return MTLPixelFormatRG8Unorm;
                    case 3: assert("RGB8Unorm not supported"); break;
                    case 4: return MTLPixelFormatRGBA8Unorm;
                }
        }
        return MTLPixelFormatInvalid;
    }();
    textureDescriptor.width = loader.GetPageWidth();
    textureDescriptor.height = loader.GetPageHeight();
    textureDescriptor.arrayLength = loader.GetNumPages();
    textureDescriptor.mipmapLevelCount = 1;
    textureDescriptor.usage = MTLTextureUsageShaderRead;
    textureDescriptor.textureType = MTLTextureType2DArray;
    
    const auto texelsTexture = [deviceContext->device newTextureWithDescriptor:textureDescriptor];
    if(!texelsTexture)
        return nullptr;
    
    auto result = new MTLPtexMipmapTexture();
    result->_textureDescriptor = textureDescriptor;
    result->_layout = layoutBuffer;
    result->_texels = texelsTexture;

    MTLRegion replaceRegion = MTLRegionMake2D(0, 0, loader.GetPageWidth(), loader.GetPageHeight());
    const auto bytesPerRow = loader.GetPageWidth() * bpp;
    const auto bytesPerImage = bytesPerRow * loader.GetPageHeight();
    for(auto i = 0; i < loader.GetNumPages(); i++) {
        [texelsTexture replaceRegion:replaceRegion mipmapLevel:0 slice:i
            withBytes:loader.GetTexelBuffer() + i * bytesPerImage bytesPerRow:bytesPerRow bytesPerImage:bytesPerImage];
    }



    return result;
}
