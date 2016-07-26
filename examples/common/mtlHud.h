//
//   Copyright 2016 Pixar
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

#ifndef OPENSUBDIV_EXAMPLES_MTL_HUD_H
#define OPENSUBDIV_EXAMPLES_MTL_HUD_H

@protocol MTLTexture;
@protocol MTLBuffer;
@protocol MTLRenderCommandEncoder;
@protocol MTLRenderPipelineState;
@protocol MTLDevice;
@class MTLRenderPipelineDescriptor;
@class MTLDepthStencilDescriptor;

#include "hud.h"
#include "mtlUtils.h"

class MTLhud : public Hud {

public:
    MTLhud();
    ~MTLhud();

    virtual void Init(id<MTLDevice> device, MTLRenderPipelineDescriptor* parentPipelineDescriptor, MTLDepthStencilDescriptor* depthStencilStateDescriptor,
    				  int width, int height, int framebufferWidth, int framebufferHeight);

    virtual void Rebuild(int width, int height,
                         int framebufferWidth, int framebufferHeight);

    virtual bool Flush(id<MTLRenderCommandEncoder> encoder);

    id<MTLTexture> GetFontTexture() const {
        return _fontTexture;
    }

    void FillBackground(id<MTLRenderCommandEncoder> encoder);
    
    float UIScale = 1.0f;

private:
	id<MTLDevice> _device;
    id<MTLTexture> _fontTexture;
	OpenSubdiv::OPENSUBDIV_VERSION::Osd::MTLRingBuffer<float, 1> _staticBuffer;
	OpenSubdiv::OPENSUBDIV_VERSION::Osd::MTLRingBuffer<float, 3> _dynamicBuffer;

    id<MTLRenderPipelineState> _fgPipelineState, _bgPipelineState;
    id<MTLDepthStencilState> _depthStencilState;
};


#endif //OPENSUBDIV_EXAMPLES_MTL_HUD_H
