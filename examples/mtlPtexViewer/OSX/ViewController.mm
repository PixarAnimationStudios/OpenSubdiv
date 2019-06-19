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

#import "ViewController.h"


enum {
    kHUD_RB_COLOR,
    kHUD_RB_SCHEME,
    kHUD_RB_LEVEL,
    kHUD_RB_DISPLACEMENT,
    kHUD_RB_NORMAL
};

enum {
    kHUD_SL_MIPMAPBIAS,
    kHUD_SL_DISPLACEMENT,
};

enum {
    kHUD_CB_DISPLAY_OCCLUSION,
    kHUD_CB_DISPLAY_SPECULAR,
    kHUD_CB_ANIMATE_VERTICES,
    kHUD_CB_VIEW_LOD,
    kHUD_CB_FRACTIONAL_SPACING,
    kHUD_CB_PATCH_CULL,
    kHUD_CB_FREEZE,
    kHUD_CB_ADAPTIVE,
    kHUD_CB_SEAMLESS_MIPMAP,
};

@implementation OSDView {
    bool _mouseDown;
    NSPoint _lastMouse;
}

-(void)mouseDown:(NSEvent *)event {
    if(event.buttonNumber == 0) {
        _lastMouse = [event locationInWindow];
        _mouseDown = !hud.MouseClick(_lastMouse.x, (self.bounds.size.height - _lastMouse.y));
    }
    [super mouseDown:event];
}

-(void)mouseUp:(NSEvent *)event {
    if(event.buttonNumber == 0) {
        _mouseDown = false;
        _lastMouse = [event locationInWindow];
        hud.MouseRelease();
    }
    [super mouseUp:event];
}

-(void)mouseDragged:(NSEvent *)event {
    auto mouse = [NSEvent mouseLocation];
    hud.MouseMotion(mouse.x, mouse.y);
    
    if(_mouseDown) {
        CGPoint delta;
        delta.x = mouse.x - _lastMouse.x;
        delta.y = mouse.y - _lastMouse.y;
        _lastMouse = mouse;
        
        _controller.osdRenderer.camera->rotationX += delta.x / 2.0;
        _controller.osdRenderer.camera->rotationY -= delta.y / 2.0;
    }
    [super mouseDragged:event];
}

-(void)keyDown:(NSEvent *)event {
    const auto key = [event.charactersIgnoringModifiers characterAtIndex:0];
    if (hud.KeyDown(key)) {
        return;
    } else if(key == '=') {
        _controller.osdRenderer.tessellationLevel = std::min(_controller.osdRenderer.tessellationLevel + 1, 16);
    } else if (key == '-') {
        _controller.osdRenderer.tessellationLevel = std::max(_controller.osdRenderer.tessellationLevel - 1, 0);
    } else if (key == 'f') {
        [_controller.osdRenderer fitFrame];
    } else {
        [super keyDown:event];
    }
}

-(void)scrollWheel:(NSEvent *)event {
    _controller.osdRenderer.camera->dollyDistance += event.deltaY / 100.0;
}

-(BOOL)acceptsFirstResponder {
    return true;
}

@end




#define FRAME_HISTORY 30
@implementation ViewController {
    id<MTLDevice> _device;
    id<MTLCommandQueue> _commandQueue;
    dispatch_semaphore_t _frameSemaphore;
    OSDRenderer* _osdRenderer;
    
    unsigned _currentFrame;
    double _frameBeginTimestamp[FRAME_HISTORY];
    
    NSMagnificationGestureRecognizer* _magnificationGesture;
}

-(void)viewDidLoad {
    
    _device = MTLCreateSystemDefaultDevice();
    _commandQueue = [_device newCommandQueue];
    
    _osdRenderer = [[OSDRenderer alloc] initWithDelegate:self];
    
    _frameSemaphore = dispatch_semaphore_create(3);
    
    self.view.device = _device;
    self.view.delegate = self;
    self.view.depthStencilPixelFormat = MTLPixelFormatDepth32Float;
    self.view.clearColor = MTLClearColorMake(0.4245, 0.4167, 0.4245, 1);
    self.view.controller = self;
    self.view.sampleCount = 2;
    
    
    _osdRenderer.camera->aspectRatio = self.view.bounds.size.width / self.view.bounds.size.height;
    
    _currentFrame = 0;
    
    auto renderPipelineDescriptor = [MTLRenderPipelineDescriptor new];
    renderPipelineDescriptor.colorAttachments[0].pixelFormat = self.view.colorPixelFormat;
    renderPipelineDescriptor.depthAttachmentPixelFormat = self.view.depthStencilPixelFormat;
    renderPipelineDescriptor.sampleCount = self.view.sampleCount;
    
    auto depthStencilDescriptor = [MTLDepthStencilDescriptor new];
    depthStencilDescriptor.depthWriteEnabled = false;
    depthStencilDescriptor.depthCompareFunction = MTLCompareFunctionAlways;

    auto& hud = self.view->hud;
    hud.Init(_device, renderPipelineDescriptor, depthStencilDescriptor, self.view.bounds.size.width, self.view.bounds.size.height, self.view.drawableSize.width, self.view.drawableSize.height);

    
    auto callbackCheckbox = [=](bool checked, int ID) {
        switch(ID) {
            case kHUD_CB_FREEZE:
                self.osdRenderer.freeze = checked;
                break;
            case kHUD_CB_ADAPTIVE:
                self.osdRenderer.useAdaptive = checked;
                break;
            case kHUD_CB_VIEW_LOD:
                self.osdRenderer.useScreenspaceTessellation = checked;
                break;
            case kHUD_CB_PATCH_CULL:
                self.osdRenderer.usePatchClipCulling = checked;
                break;
            case kHUD_CB_ANIMATE_VERTICES:
                self.osdRenderer.animateVertices = checked;
                break;
            case kHUD_CB_DISPLAY_SPECULAR:
                self.osdRenderer.displaySpecular = checked;
                break;
            case kHUD_CB_DISPLAY_OCCLUSION:
                self.osdRenderer.displayOcclusion = checked;
                break;
            case kHUD_CB_FRACTIONAL_SPACING:
                self.osdRenderer.useFractionalTessellation = checked;
                break;
            case kHUD_CB_SEAMLESS_MIPMAP:
                self.osdRenderer.useSeamlessMipmap = checked;
                break;
            default:
                assert("Unknown checkbox ID" && 0);
        }
    };
    
    auto callbackScheme = [=](int scheme) {
        return;
    };
    
    auto callbackLevel = [=](int level) {
        self.osdRenderer.refinementLevel = level;
    };
    
    auto callbackKernel = [=](int kernelType) {
        switch((KernelType)kernelType) {
            case kCPU:
            case kMetal:
                self.osdRenderer.kernelType = (KernelType)(kernelType);
                break;
            default:
                assert("Unknown kernelType" && 0);
        }
    };
    
    auto callbackDisplayStyle = [=](int displayStyle) {
        switch((DisplayStyle)displayStyle) {
            case kDisplayStyleWire:
            case kDisplayStyleShaded:
            case kDisplayStyleWireOnShaded:
                self.osdRenderer.displayStyle = (DisplayStyle)displayStyle;
                break;
            default:
                assert("Unknown displayStyle" && 0);
        }
    };
    
    auto callbackColor = [=](int colorMode) {
        switch((ColorMode)colorMode) {
            case kColorModeNone:
            case kColorModeNormal:
            case kColorModePatchType:
            case kColorModePatchCoord:
            case kColorModePtexNearest:
            case kColorModePtexBilinear:
            case kColorModePtexHWBilinear:
            case kColorModePtexBiQuadratic:
                self.osdRenderer.colorMode = (ColorMode)colorMode;
                break;
            default:
                assert("Unknown colorMode" && 0);
        }
    };
    
    auto callbackDisplacement = [=](int displacementMode) {
        switch((DisplacementMode)displacementMode) {
            case kDisplacementModeNone:
            case kDisplacementModeBilinear:
            case kDisplacementModeHWBilinear:
            case kDisplacementModeBiQuadratic:
                self.osdRenderer.displacementMode = (DisplacementMode)displacementMode;
                break;
            default:
                assert("Unknown displacementMode" && 0);
        }
    };
    
    auto callbackNormal = [=](int normalMode) {
        switch((NormalMode)normalMode) {
            case kNormalModeBiQuadraticWG:
            case kNormalModeBiQuadratic:
            case kNormalModeScreenspace:
            case kNormalModeHWScreenspace:
            case kNormalModeSurface:
                self.osdRenderer.normalMode = (NormalMode)normalMode;
                break;
            default:
                assert("Unknown normalMode" && 0);
        }
    };
    
    auto callbackSlider = [=](float sliderValue, int sliderID) {
        switch(sliderID) {
            case kHUD_SL_DISPLACEMENT:
                self.osdRenderer.displacementScale = sliderValue;
                break;
            case kHUD_SL_MIPMAPBIAS:
                self.osdRenderer.mipmapBias = sliderValue;
                break;
            default:
                assert("Unknown slider ID" && 0);
        }
    };
    
    if (_osdRenderer.ptexOcclusionFilename != NULL) {
        hud.AddCheckBox("Ambient Occlusion (A)", _osdRenderer.ptexOcclusionFilename,
                           -200, 570, callbackCheckbox, kHUD_CB_DISPLAY_OCCLUSION, 'a');
    }
    if (_osdRenderer.ptexSpecularFilename != NULL)
        hud.AddCheckBox("Specular (S)", _osdRenderer.ptexSpecularFilename,
                           -200, 590, callbackCheckbox, kHUD_CB_DISPLAY_SPECULAR, 's');
    
//    if (_osdRenderer.ptexColorFilename || g_diffuseEnvironmentMap) {
//        hud.AddCheckBox("IBL (I)", g_ibl,
//                           -200, 610, callbackCheckbox, HUD_CB_IBL, 'i');
//    }
    
    hud.AddCheckBox("Animate vertices (M)", _osdRenderer.animateVertices,
                       10, 30, callbackCheckbox, kHUD_CB_ANIMATE_VERTICES, 'm');
    hud.AddCheckBox("Screen space LOD (V)",  _osdRenderer.useScreenspaceTessellation,
                       10, 50, callbackCheckbox, kHUD_CB_VIEW_LOD, 'v');
    hud.AddCheckBox("Fractional spacing (T)",  _osdRenderer.useFractionalTessellation,
                       10, 70, callbackCheckbox, kHUD_CB_FRACTIONAL_SPACING, 't');
    hud.AddCheckBox("Frustum Patch Culling (B)",  _osdRenderer.usePatchClipCulling,
                       10, 90, callbackCheckbox, kHUD_CB_PATCH_CULL, 'b');
    hud.AddCheckBox("Freeze (spc)", _osdRenderer.freeze,
                       10, 110, callbackCheckbox, kHUD_CB_FREEZE, ' ');
    //    hud.AddCheckBox("Bloom (Y)", g_bloom,
    //                       10, 130, callbackCheckbox, HUD_CB_BLOOM, 'y');
    
    hud.AddRadioButton(kHUD_RB_SCHEME, "CATMARK", true, 10, 190, callbackScheme, 0);
    
//    Ptex without Adaptive is not supported
//    hud.AddCheckBox("Adaptive (`)", _osdRenderer.useAdaptive,
//                       10, 300, callbackCheckbox, kHUD_CB_ADAPTIVE, '`');
    
    for (int i = 1; i < 8; ++i) {
        char level[16];
        sprintf(level, "Lv. %d", i);
        hud.AddRadioButton(kHUD_RB_LEVEL, level, _osdRenderer.refinementLevel == i,
                              10, 320+i*20, callbackLevel, i, '0'+i);
    }
    
    int compute_pulldown = hud.AddPullDown("Compute (K)", 475, 10, 300, callbackKernel, 'k');
    hud.AddPullDownButton(compute_pulldown, "CPU", kCPU);
    hud.AddPullDownButton(compute_pulldown, "Metal", kMetal);
    
    int shading_pulldown = hud.AddPullDown("Shading (W)", 250, 10, 250, callbackDisplayStyle, 'w');
    hud.AddPullDownButton(shading_pulldown, "Wire", kDisplayStyleWire, _osdRenderer.displayStyle == kDisplayStyleWire);
    hud.AddPullDownButton(shading_pulldown, "Shaded", kDisplayStyleShaded, _osdRenderer.displayStyle == kDisplayStyleShaded);
    hud.AddPullDownButton(shading_pulldown, "Wire+Shaded", kDisplayStyleWireOnShaded, _osdRenderer.displayStyle ==kDisplayStyleWireOnShaded);
    
    hud.AddLabel("Color (C)", -200, 10);
    hud.AddRadioButton(kHUD_RB_COLOR, "None", (_osdRenderer.colorMode == kColorModeNone),
                          -200, 30, callbackColor, kColorModeNone, 'c');
    hud.AddRadioButton(kHUD_RB_COLOR, "Ptex Nearest", (_osdRenderer.colorMode == kColorModePtexNearest),
                          -200, 50, callbackColor, kColorModePtexNearest, 'c');
    hud.AddRadioButton(kHUD_RB_COLOR, "Ptex HW bilinear", (_osdRenderer.colorMode == kColorModePtexHWBilinear),
                         -200, 70, callbackColor, kColorModePtexHWBilinear, 'c');
    hud.AddRadioButton(kHUD_RB_COLOR, "Ptex bilinear", (_osdRenderer.colorMode == kColorModePtexBilinear),
                          -200, 90, callbackColor, kColorModePtexBilinear, 'c');
    hud.AddRadioButton(kHUD_RB_COLOR, "Ptex biquadratic", (_osdRenderer.colorMode == kColorModePtexBiQuadratic),
                          -200, 110, callbackColor, kColorModePtexBiQuadratic, 'c');
    hud.AddRadioButton(kHUD_RB_COLOR, "Patch type", (_osdRenderer.colorMode == kColorModePatchType),
                          -200, 130, callbackColor, kColorModePatchType, 'c');
    hud.AddRadioButton(kHUD_RB_COLOR, "Patch coord", (_osdRenderer.colorMode == kColorModePatchCoord),
                          -200, 150, callbackColor, kColorModePatchCoord, 'c');
    hud.AddRadioButton(kHUD_RB_COLOR, "Normal", (_osdRenderer.colorMode == kColorModeNormal),
                          -200, 170, callbackColor, kColorModeNormal, 'c');
    
    if (_osdRenderer.ptexDisplacementFilename) {
        hud.AddLabel("Displacement (D)", -200, 200);
        hud.AddRadioButton(kHUD_RB_DISPLACEMENT, "None",
                              (_osdRenderer.displacementMode  == kDisplacementModeNone),
                              -200, 220, callbackDisplacement, kDisplacementModeNone, 'd');
        hud.AddRadioButton(kHUD_RB_DISPLACEMENT, "HW bilinear",
                             (_osdRenderer.displacementMode  == kDisplacementModeHWBilinear),
                             -200, 240, callbackDisplacement, kDisplacementModeHWBilinear, 'd');
        hud.AddRadioButton(kHUD_RB_DISPLACEMENT, "Bilinear",
                              (_osdRenderer.displacementMode  == kDisplacementModeBilinear),
                              -200, 260, callbackDisplacement, kDisplacementModeBilinear, 'd');
        hud.AddRadioButton(kHUD_RB_DISPLACEMENT, "Biquadratic",
                              (_osdRenderer.displacementMode  == kDisplacementModeBiQuadratic),
                              -200, 280, callbackDisplacement, kDisplacementModeBiQuadratic, 'd');
        
        {
            int y = 310;
            hud.AddLabel("Normal (N)", -200, y); y += 20;
            hud.AddRadioButton(kHUD_RB_NORMAL, "Surface",
                               (_osdRenderer.normalMode == kNormalModeSurface),
                               -200, y, callbackNormal, kNormalModeSurface, 'n'); y += 20;
//           hud.AddRadioButton(kHUD_RB_NORMAL, "Facet", //We can't really do NORMAL_FACET
//                              (_osdRenderer.normalMode == NORMAL_FACET),
//                              -200, y, callbackNormal, NORMAL_FACET, 'n'); y += 20;
            hud.AddRadioButton(kHUD_RB_NORMAL, "HW Screen space",
                               (_osdRenderer.normalMode == kNormalModeHWScreenspace),
                               -200, y, callbackNormal, kNormalModeHWScreenspace, 'n'); y += 20;
            hud.AddRadioButton(kHUD_RB_NORMAL, "Screen space",
                               (_osdRenderer.normalMode == kNormalModeScreenspace),
                               -200, y, callbackNormal, kNormalModeScreenspace, 'n'); y += 20;
            hud.AddRadioButton(kHUD_RB_NORMAL, "Biquadratic",
                               (_osdRenderer.normalMode == kNormalModeBiQuadratic),
                               -200, y, callbackNormal, kNormalModeBiQuadratic, 'n'); y += 20;
            hud.AddRadioButton(kHUD_RB_NORMAL, "Biquadratic WG",
                               (_osdRenderer.normalMode == kNormalModeBiQuadraticWG),
                               -200, y, callbackNormal, kNormalModeBiQuadraticWG, 'n'); y += 20;
        }
    }
    
    hud.AddSlider("Mipmap Bias", 0, 5, 0,
                     -200, 450, 20, false, callbackSlider, kHUD_SL_MIPMAPBIAS);
    hud.AddSlider("Displacement", 0, 5, 1,
                     -200, 490, 20, false, callbackSlider, kHUD_SL_DISPLACEMENT);
    hud.AddCheckBox("Seamless Mipmap", _osdRenderer.useSeamlessMipmap,
                       -200, 530, callbackCheckbox, kHUD_CB_SEAMLESS_MIPMAP, 'j');
    
    hud.Rebuild(self.view.bounds.size.width, self.view.bounds.size.height, self.view.drawableSize.width, self.view.drawableSize.height);
}

-(void)drawInMTKView:(MTKView *)view {
    dispatch_semaphore_wait(_frameSemaphore, DISPATCH_TIME_FOREVER);
    
    auto commandBuffer = [_commandQueue commandBuffer];
    
    double avg = 0;
    for(int i = 0; i < FRAME_HISTORY; i++)
        avg += _frameBeginTimestamp[i];
    avg /= FRAME_HISTORY;
    
    auto renderEncoder = [_osdRenderer drawFrame:commandBuffer];
    
    auto& hud = self.view->hud;
    if(hud.IsVisible()) {
        hud.DrawString(10, -120, "Tess level : %d", _osdRenderer.tessellationLevel);
        hud.DrawString(10, -20, "FPS = %3.1f", 1.0 / avg);
        
        //Disable Culling & Force Fill mode when drawing the UI
        [renderEncoder setTriangleFillMode:MTLTriangleFillModeFill];
        [renderEncoder setCullMode:MTLCullModeNone];
        self.view->hud.Flush(renderEncoder);
    };
    
    [renderEncoder endEncoding];
    
    __weak auto blockSemaphore = _frameSemaphore;
    unsigned frameId = _currentFrame % FRAME_HISTORY;
    auto frameBeginTime = CACurrentMediaTime();
    [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> _Nonnull c) {
        dispatch_semaphore_signal(blockSemaphore);
        _frameBeginTimestamp[frameId] = CACurrentMediaTime() - frameBeginTime;
    }];
    
    [commandBuffer presentDrawable:view.currentDrawable];
    [commandBuffer commit];
    
    _currentFrame++;
}

-(void)mtkView:(MTKView *)view drawableSizeWillChange:(CGSize)size {
    _osdRenderer.camera->aspectRatio = size.width / size.height;
}

-(void)setupDepthStencilState:(MTLDepthStencilDescriptor *)descriptor for:(OSDRenderer *)renderer {
    
}

-(void)setupRenderPipelineState:(MTLRenderPipelineDescriptor *)descriptor for:(OSDRenderer *)renderer {
    descriptor.depthAttachmentPixelFormat = self.view.depthStencilPixelFormat;
    descriptor.colorAttachments[0].pixelFormat = self.view.colorPixelFormat;
    descriptor.sampleCount = self.view.sampleCount;
}

-(id<MTLCommandQueue>)commandQueueFor:(OSDRenderer *)renderer {
    return _commandQueue;
}

-(id<MTLDevice>)deviceFor:(OSDRenderer *)renderer {
    return _device;
}

-(MTLRenderPassDescriptor *)renderPassDescriptorFor:(OSDRenderer *)renderer {
    return self.view.currentRenderPassDescriptor;
}
@end
