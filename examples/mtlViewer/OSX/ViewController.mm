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


#import "ViewController.h"
#import <opensubdiv/far/patchDescriptor.h>

using namespace OpenSubdiv::OPENSUBDIV_VERSION;

enum {
    kHUD_CB_DISPLAY_CONTROL_MESH_EDGES,
    kHUD_CB_DISPLAY_CONTROL_MESH_VERTS,
    kHUD_CB_ANIMATE_VERTICES,
    kHUD_CB_DISPLAY_PATCH_COLOR, //Unused
    kHUD_CB_DISPLAY_PATCH_CVs, //Unused
    kHUD_CB_VIEW_LOD,
    kHUD_CB_FRACTIONAL_SPACING,
    kHUD_CB_PATCH_CULL,
    kHUD_CB_BACK_CULL,
    kHUD_CB_PATCH_INDEX_BUFFER,
    kHUD_CB_FREEZE,
    kHUD_CB_SMOOTH_CORNER_PATCH,
    kHUD_CB_SINGLE_CREASE_PATCH,
    kHUD_CB_INFINITE_SHARP_PATCH,
    kHUD_CB_ADAPTIVE,
    kHUD_CB_DISPLAY_PATCH_COUNTS
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
    
    int _showPatchCounts;
}

-(void)viewDidLoad {
    
    _device = MTLCreateSystemDefaultDevice();
    _commandQueue = [_device newCommandQueue];
    
    _osdRenderer = [[OSDRenderer alloc] initWithDelegate:self];
    _osdRenderer.displayControlMeshEdges = false;
    
    _frameSemaphore = dispatch_semaphore_create(3);
    
    self.view.device = _device;
    self.view.delegate = self;
    self.view.depthStencilPixelFormat = MTLPixelFormatDepth32Float;
    self.view.clearColor = MTLClearColorMake(0.4245, 0.4167, 0.4245, 1);
    self.view.controller = self;
    self.view.sampleCount = 1;
    
    self.view->hud.UIScale = 1.0;
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
    
    auto callbackCheckbox = [=](bool value, int ID) {
        switch(ID) {
            case kHUD_CB_FREEZE:
                self.osdRenderer.freeze = value;
                break;
            case kHUD_CB_VIEW_LOD:
                self.osdRenderer.useScreenspaceTessellation = value;
                break;
            case kHUD_CB_PATCH_CULL:
                self.osdRenderer.usePatchClipCulling = value;

                break;
            case kHUD_CB_ANIMATE_VERTICES:
                self.osdRenderer.animateVertices = value;
                break;
            case kHUD_CB_FRACTIONAL_SPACING:
                self.osdRenderer.useFractionalTessellation = value;
                break;
            case kHUD_CB_SMOOTH_CORNER_PATCH:
                self.osdRenderer.useSmoothCornerPatch = value;
                break;
            case kHUD_CB_SINGLE_CREASE_PATCH:
                self.osdRenderer.useSingleCreasePatch = value;
                break;
            case kHUD_CB_DISPLAY_CONTROL_MESH_EDGES:
                self.osdRenderer.displayControlMeshEdges = value;
                break;
            case kHUD_CB_DISPLAY_CONTROL_MESH_VERTS:
                self.osdRenderer.displayControlMeshVertices = value;
                break;
            case kHUD_CB_BACK_CULL:
                self.osdRenderer.usePatchBackfaceCulling = value;
                self.osdRenderer.usePrimitiveBackfaceCulling = value;
                break;
            case kHUD_CB_PATCH_INDEX_BUFFER:
                self.osdRenderer.usePatchIndexBuffer = value;
                break;
            case kHUD_CB_ADAPTIVE:
                self.osdRenderer.useAdaptive = value;
                break;
            case kHUD_CB_INFINITE_SHARP_PATCH:
                self.osdRenderer.useInfinitelySharpPatch = value;
                break;
            case kHUD_CB_DISPLAY_PATCH_COUNTS:
                _showPatchCounts = value;
                break;
            default:
                assert("Unknown checkbox ID" && 0);
        }
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
    
    auto callbackFVarLinearInterp = [=](int fVarLinearInterp) {
        switch((FVarLinearInterp)fVarLinearInterp) {
            case kFVarLinearNone:
            case kFVarLinearCornersOnly:
            case kFVarLinearCornersPlus1:
            case kFVarLinearCornersPlus2:
            case kFVarLinearBoundaries:
            case kFVarLinearAll:
                self.osdRenderer.fVarLinearInterp = (FVarLinearInterp)fVarLinearInterp;
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

    auto callbackShadingMode = [=](int shadingMode) {
        switch((ShadingMode)shadingMode) {
            case kShadingMaterial:
            case kShadingFaceVaryingColor:
            case kShadingPatchType:
            case kShadingPatchDepth:
            case kShadingPatchCoord:
            case kShadingNormal:
                self.osdRenderer.shadingMode = (ShadingMode)shadingMode;
                break;
            default:
                assert("Unknown shadingMode" && 0);
        }
    };
    
    auto callbackEndCap = [=](int endCap) {
        switch((EndCap)endCap) {
            case kEndCapBilinearBasis:
            case kEndCapBSplineBasis:
            case kEndCapGregoryBasis:
            case kEndCapLegacyGregory:
                self.osdRenderer.endCapMode = (EndCap)endCap;
                break;
            default:
                assert("Unknown endCap" && 0);
        }
    };
    
    auto callbackLevel = [=](int level) {
        self.osdRenderer.refinementLevel = level;
    };

    auto callbackModel = [=](int modelIndex) {
        assert(modelIndex >= 0);
        assert((NSUInteger)modelIndex < self.osdRenderer.loadedModels.count);
        
        self.osdRenderer.currentModel = self.osdRenderer.loadedModels[modelIndex];
    };
    
    int y = 10;
    hud.AddCheckBox("Control edges (H)",
                      _osdRenderer.displayControlMeshEdges,
                      10, y, callbackCheckbox,
                      kHUD_CB_DISPLAY_CONTROL_MESH_EDGES, 'h');
    y += 20;
    hud.AddCheckBox("Control vertices (J)",
                      _osdRenderer.displayControlMeshVertices,
                      10, y, callbackCheckbox,
                      kHUD_CB_DISPLAY_CONTROL_MESH_VERTS, 'j');
    y += 20;
    hud.AddCheckBox("Animate vertices (M)", _osdRenderer.animateVertices,
                      10, y, callbackCheckbox, kHUD_CB_ANIMATE_VERTICES, 'm');
    y += 20;
    hud.AddCheckBox("Screen space LOD (V)",  _osdRenderer.useScreenspaceTessellation,
                      10, y, callbackCheckbox, kHUD_CB_VIEW_LOD, 'v');
    y += 20;
    hud.AddCheckBox("Fractional spacing (T)",  _osdRenderer.useFractionalTessellation,
                      10, y, callbackCheckbox, kHUD_CB_FRACTIONAL_SPACING, 't');
    y += 20;
    hud.AddCheckBox("Frustum Patch Culling (P)",  _osdRenderer.usePatchClipCulling,
                      10, y, callbackCheckbox, kHUD_CB_PATCH_CULL, 'p');
    y += 20;
    hud.AddCheckBox("Backface Culling (B)", _osdRenderer.usePatchBackfaceCulling,
                    10, y, callbackCheckbox, kHUD_CB_BACK_CULL, 'b');
    
    y += 20;
    hud.AddCheckBox("Patch Index Buffer (D)", _osdRenderer.usePatchIndexBuffer,
                    10, y, callbackCheckbox, kHUD_CB_PATCH_INDEX_BUFFER, 'd');
    
    y += 20;
    hud.AddCheckBox("Freeze (spc)", _osdRenderer.freeze,
                    10, y, callbackCheckbox, kHUD_CB_FREEZE, ' ');
    y += 20;
    
    int displaystyle_pulldown = hud.AddPullDown("DisplayStyle (W)", 200, 10, 250,
                                                callbackDisplayStyle, 'w');
    hud.AddPullDownButton(displaystyle_pulldown, "Wire", kDisplayStyleWire,
                            _osdRenderer.displayStyle == kDisplayStyleWire);
    hud.AddPullDownButton(displaystyle_pulldown, "Shaded", kDisplayStyleShaded,
                            _osdRenderer.displayStyle == kDisplayStyleShaded);
    hud.AddPullDownButton(displaystyle_pulldown, "Wire+Shaded", kDisplayStyleWireOnShaded,
                            _osdRenderer.displayStyle == kDisplayStyleWireOnShaded);
    
    int shading_pulldown = hud.AddPullDown("Shading (C)", 200, 70, 250,
                                           callbackShadingMode, 'c');
    
    hud.AddPullDownButton(shading_pulldown, "Material",
                            kShadingMaterial,
                            _osdRenderer.shadingMode == kShadingMaterial);
    hud.AddPullDownButton(shading_pulldown, "FaceVarying Color",
                            kShadingFaceVaryingColor,
                            _osdRenderer.shadingMode == kShadingFaceVaryingColor);
    hud.AddPullDownButton(shading_pulldown, "Patch Type",
                            kShadingPatchType,
                            _osdRenderer.shadingMode == kShadingPatchType);
    hud.AddPullDownButton(shading_pulldown, "Patch Depth",
                            kShadingPatchDepth,
                            _osdRenderer.shadingMode == kShadingPatchDepth);
    hud.AddPullDownButton(shading_pulldown, "Patch Coord",
                            kShadingPatchCoord,
                            _osdRenderer.shadingMode == kShadingPatchCoord);
    hud.AddPullDownButton(shading_pulldown, "Normal",
                            kShadingNormal,
                            _osdRenderer.shadingMode == kShadingNormal);

    int compute_pulldown = hud.AddPullDown("Compute (K)", 475, 10, 175, callbackKernel, 'k');
    hud.AddPullDownButton(compute_pulldown, "CPU", kCPU, _osdRenderer.kernelType == kCPU);
    hud.AddPullDownButton(compute_pulldown, "Metal", kMetal, _osdRenderer.kernelType == kMetal);

    int fVarLinearInterp_pulldown = hud.AddPullDown("FVar Linear Interpolation (L)",
                          650, 10, 300, callbackFVarLinearInterp, 'l');
    hud.AddPullDownButton(fVarLinearInterp_pulldown, "None (edge only)",
                          kFVarLinearNone,
                          _osdRenderer.fVarLinearInterp == kFVarLinearNone);
    hud.AddPullDownButton(fVarLinearInterp_pulldown, "Corners Only",
                          kFVarLinearCornersOnly,
                          _osdRenderer.fVarLinearInterp == kFVarLinearCornersOnly);
    hud.AddPullDownButton(fVarLinearInterp_pulldown, "Corners 1 (edge corner)",
                          kFVarLinearCornersPlus1,
                          _osdRenderer.fVarLinearInterp == kFVarLinearCornersPlus1);
    hud.AddPullDownButton(fVarLinearInterp_pulldown, "Corners 2 (edge corner prop)",
                          kFVarLinearCornersPlus2,
                          _osdRenderer.fVarLinearInterp == kFVarLinearCornersPlus2);
    hud.AddPullDownButton(fVarLinearInterp_pulldown, "Boundaries (always sharp)",
                          kFVarLinearBoundaries,
                          _osdRenderer.fVarLinearInterp == kFVarLinearBoundaries);
    hud.AddPullDownButton(fVarLinearInterp_pulldown, "All (bilinear)",
                          kFVarLinearAll,
                          _osdRenderer.fVarLinearInterp == kFVarLinearAll);

    {
        hud.AddCheckBox("Adaptive (`)", _osdRenderer.useAdaptive,
                           10, 190, callbackCheckbox, kHUD_CB_ADAPTIVE, '`');
        hud.AddCheckBox("Smooth Corner Patch (O)", _osdRenderer.useSmoothCornerPatch,
                        10, 210, callbackCheckbox, kHUD_CB_SMOOTH_CORNER_PATCH, 'o');
        hud.AddCheckBox("Single Crease Patch (S)", _osdRenderer.useSingleCreasePatch,
                        10, 230, callbackCheckbox, kHUD_CB_SINGLE_CREASE_PATCH, 's');
        hud.AddCheckBox("Inf Sharp Patch (I)", _osdRenderer.useInfinitelySharpPatch,
                           10, 250, callbackCheckbox, kHUD_CB_INFINITE_SHARP_PATCH, 'i');

        int endcap_pulldown = hud.AddPullDown(
                                              "End cap (E)", 10, 270, 200, callbackEndCap, 'e');
        hud.AddPullDownButton(endcap_pulldown,"Linear",
                                kEndCapBilinearBasis,
                                _osdRenderer.endCapMode == kEndCapBilinearBasis);
        hud.AddPullDownButton(endcap_pulldown, "Regular",
                                kEndCapBSplineBasis,
                                _osdRenderer.endCapMode == kEndCapBSplineBasis);
        hud.AddPullDownButton(endcap_pulldown, "Gregory",
                                kEndCapGregoryBasis,
                                _osdRenderer.endCapMode == kEndCapGregoryBasis);
        if (_osdRenderer.legacyGregoryEnabled) {
            hud.AddPullDownButton(endcap_pulldown, "LegacyGregory",
                                    kEndCapLegacyGregory,
                                    _osdRenderer.endCapMode == kEndCapLegacyGregory);
        }
    }
    
    for (int i = 1; i < 11; ++i) {
        char level[16];
        sprintf(level, "Lv. %d", i);
        hud.AddRadioButton(3, level, i==_osdRenderer.refinementLevel, 10, 310+i*20, callbackLevel, i, '0'+(i%10));
    }

    int shapes_pulldown = hud.AddPullDown("Shape (N)", -300, 10, 300, callbackModel, 'n');
    for (int i = 0; i < (int)_osdRenderer.loadedModels.count; ++i) {
        hud.AddPullDownButton(shapes_pulldown, _osdRenderer.loadedModels[i].UTF8String,i);
    }

    hud.AddCheckBox("Show patch counts (,)", _showPatchCounts, -420, -20, callbackCheckbox, kHUD_CB_DISPLAY_PATCH_COUNTS, ',');

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
        if(_showPatchCounts) {
            int x = -420;
            int y = -180;
            hud.DrawString(x, y, "Quads            : %d",
                              _osdRenderer.patchCounts[Far::PatchDescriptor::QUADS]); y += 20;
            hud.DrawString(x, y, "Triangles        : %d",
                              _osdRenderer.patchCounts[Far::PatchDescriptor::TRIANGLES]); y += 20;
            hud.DrawString(x, y, "Regular          : %d",
                              _osdRenderer.patchCounts[Far::PatchDescriptor::REGULAR]); y+= 20;
            hud.DrawString(x, y, "Loop             : %d",
                              _osdRenderer.patchCounts[Far::PatchDescriptor::LOOP]); y+= 20;
            if (_osdRenderer.legacyGregoryEnabled) {
                hud.DrawString(x, y, "Gregory          : %d",
                                  _osdRenderer.patchCounts[Far::PatchDescriptor::GREGORY]); y+= 20;
                hud.DrawString(x, y, "Boundary Gregory : %d",
                                  _osdRenderer.patchCounts[Far::PatchDescriptor::GREGORY_BOUNDARY]); y+= 20;
            }
            hud.DrawString(x, y, "Gregory Basis    : %d",
                              _osdRenderer.patchCounts[Far::PatchDescriptor::GREGORY_BASIS]); y+= 20;
            hud.DrawString(x, y, "Gregory Triangle : %d",
                              _osdRenderer.patchCounts[Far::PatchDescriptor::GREGORY_TRIANGLE]); y+= 20;
        }
        
        hud.DrawString(10, -120, "Tess level : %d", _osdRenderer.tessellationLevel);
        hud.DrawString(10, -20, "FPS = %3.1f", 1.0 / avg);
        
        
        //Disable Culling & Force Fill mode when drawing the UI
        [renderEncoder setTriangleFillMode:MTLTriangleFillModeFill];
        [renderEncoder setCullMode:MTLCullModeNone];
        self.view->hud.Flush(renderEncoder);
    }
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
    self.view->hud.Rebuild(self.view.bounds.size.width, self.view.bounds.size.height, size.width, size.height);
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
