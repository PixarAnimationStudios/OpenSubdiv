#import "ViewController.h"

#define FRAME_HISTORY 30
@interface ViewController ()
{
    dispatch_semaphore_t _frameSemaphore;
    CGPoint _startTouch;
    bool _isTouching;
    OSDRenderer* _osdRenderer;
    MTKView* _view;
    id<MTLDevice> _device;
    id<MTLCommandQueue> _commandQueue;
    
    double _frameTimes[FRAME_HISTORY];
    uint64_t _currentFrame;
    
    UIPanGestureRecognizer *_zoomGesture;
    

}
@end

@implementation ViewController

-(void)mtkView:(MTKView *)view drawableSizeWillChange:(CGSize)size {
    _osdRenderer.camera->aspectRatio = size.width / size.height;
}

-(void)drawInMTKView:(MTKView *)view {
    dispatch_semaphore_wait(_frameSemaphore, DISPATCH_TIME_FOREVER);
    auto commandBuffer = [_commandQueue commandBuffer];
    [[_osdRenderer drawFrame:commandBuffer] endEncoding];
    
    __weak auto blockSemaphore = _frameSemaphore;
    unsigned frameIndex = _currentFrame % FRAME_HISTORY;
    const auto beginTime = CACurrentMediaTime();
    [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> _Nonnull c) {
        dispatch_semaphore_signal(blockSemaphore);
        _frameTimes[frameIndex] = CACurrentMediaTime() - beginTime;
    }];
    [commandBuffer presentDrawable:_view.currentDrawable];
    [commandBuffer commit];
    _currentFrame++;
    
    double frameAverage = 0;
    for(auto& x : _frameTimes)
        frameAverage += x;
    
    frameAverage /= 30.0;
    
    _frameTimeLabel.text = [NSString stringWithFormat:@"%0.2f ms", frameAverage * 1000.0];
}

-(UIInterfaceOrientationMask)supportedInterfaceOrientations {
    return UIInterfaceOrientationMaskLandscape;
}


- (void)viewDidLoad {
    [super viewDidLoad];
    
    _view = (MTKView*)self.view;
    _frameSemaphore = dispatch_semaphore_create(3);
    
    _device = MTLCreateSystemDefaultDevice();
    _commandQueue = [_device newCommandQueue];
    
    _view.device = _device;
    _view.depthStencilPixelFormat = MTLPixelFormatDepth32Float;
    _view.sampleCount = 2;
    _view.frame = CGRectMake(0, 0, 1920, 1080);
    _view.contentScaleFactor = 1;
    _view.clearColor = MTLClearColorMake(0.4245, 0.4167, 0.4245, 1);
    
    _osdRenderer = [[OSDRenderer alloc] initWithDelegate:self];
    
    _view.delegate = self;
    
    _zoomGesture = [[UIPanGestureRecognizer alloc] initWithTarget:self action:@selector(_zoomView)];
    _zoomGesture.minimumNumberOfTouches = 2;
    _zoomGesture.maximumNumberOfTouches = 2;
    _zoomGesture.cancelsTouchesInView = true;
    
    [_view addGestureRecognizer:_zoomGesture];
    
    
    [self _applyOptions];
}

-(void)_applyOptions {
    _osdRenderer.useSmoothCornerPatch = _smoothCornerSwitch.isOn;
    _osdRenderer.useSingleCreasePatch = _singleCreaseSwitch.isOn;
    _osdRenderer.useInfinitelySharpPatch = _infinitelySharpSwitch.isOn;
    _osdRenderer.usePatchBackfaceCulling = _backpatchCullingSwitch.isOn;
    _osdRenderer.usePrimitiveBackfaceCulling = _backfaceCullingSwitch.isOn;
    _osdRenderer.useScreenspaceTessellation = _screenspaceTessellationSwitch.isOn;
    _osdRenderer.displayStyle = _wireframeSwitch.isOn ? kDisplayStyleWireOnShaded : kDisplayStyleShaded;
    _osdRenderer.displayControlMeshVertices = _controlMeshSwitch.isOn;
    _osdRenderer.displayControlMeshEdges = _controlMeshSwitch.isOn;
    _osdRenderer.usePatchClipCulling = _patchClipCullingSwitch.isOn;
    _osdRenderer.useFractionalTessellation = _osdRenderer.useFractionalTessellation;
    _osdRenderer.useAdaptive = true;
    _osdRenderer.freeze = true;
    _osdRenderer.animateVertices = false;
    _osdRenderer.kernelType = kMetal;
    _osdRenderer.refinementLevel = _refinementStepper.value;
    _osdRenderer.tessellationLevel = _tessellationStepper.value;
    _osdRenderer.endCapMode = (EndCap)(_endcapSegmentedControl.selectedSegmentIndex);
    
    _tesLvLlabel.text = [NSString stringWithFormat:@"Tes Lvl. %d", (int)_osdRenderer.tessellationLevel];
    [_tesLvLlabel sizeToFit];
    
    _refLvlLabel.text = [NSString stringWithFormat:@"Ref Lvl. %d", _osdRenderer.refinementLevel];
    [_refLvlLabel sizeToFit];
}

-(void)_zoomView {
    static float lastY = 0;
    if(_zoomGesture.state == UIGestureRecognizerStateBegan) {
        lastY = [_zoomGesture translationInView:_view].y;
    } else if(_zoomGesture.state == UIGestureRecognizerStateChanged) {
        const auto currentY = [_zoomGesture translationInView:_view].y;
        const auto deltaY = (currentY - lastY) / 100.0;
        lastY = currentY;
        _osdRenderer.camera->dollyDistance += deltaY;
    }
}

-(void)touchesBegan:(NSSet<UITouch *> *)touches withEvent:(UIEvent *)event {
    _isTouching = true;
    _startTouch = [touches.anyObject locationInView:self.view];
    [super touchesBegan:touches withEvent:event];
}

-(void)touchesEnded:(NSSet<UITouch *> *)touches withEvent:(UIEvent *)event {
    _isTouching = false;
    [super touchesEnded:touches withEvent:event];
}

-(void)touchesMoved:(NSSet<UITouch *> *)touches withEvent:(UIEvent *)event {
    if(_isTouching)
    {
        for(UITouch* touch in touches)
        {
            CGPoint location = [touch locationInView:self.view];
            _startTouch = [touch previousLocationInView:self.view];
            
            double deltaX = location.x - _startTouch.x;
            double deltaY = location.y - _startTouch.y;
            
            _osdRenderer.camera->rotationX += deltaX / 5.0;
            _osdRenderer.camera->rotationY += deltaY / 5.0;
        }
    }
    [super touchesMoved:touches withEvent:event];
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

-(id<MTLDevice>)deviceFor:(OSDRenderer *)renderer {
    return _device;
}
-(id<MTLCommandQueue>)commandQueueFor:(OSDRenderer *)renderer {
    return _commandQueue;
}
-(MTLRenderPassDescriptor *)renderPassDescriptorFor:(OSDRenderer *)renderer {
    return _view.currentRenderPassDescriptor;
}
-(void)setupDepthStencilState:(MTLDepthStencilDescriptor *)descriptor for:(OSDRenderer *)renderer {
    
}
-(void)setupRenderPipelineState:(MTLRenderPipelineDescriptor *)descriptor for:(OSDRenderer *)renderer {
    descriptor.colorAttachments[0].pixelFormat = _view.colorPixelFormat;
    descriptor.depthAttachmentPixelFormat = _view.depthStencilPixelFormat;
    descriptor.sampleCount = _view.sampleCount;
}
- (IBAction)stepperChanged:(UIStepper *)sender {
    if(sender == _tessellationStepper) {
        _osdRenderer.tessellationLevel = sender.value;
        _tesLvLlabel.text = [NSString stringWithFormat:@"Tes Lvl. %d", (int)_osdRenderer.tessellationLevel];
        [_tesLvLlabel sizeToFit];
        
    } else if (sender == _refinementStepper) {
        _osdRenderer.refinementLevel = sender.value;
        _refLvlLabel.text = [NSString stringWithFormat:@"Ref Lvl. %d", _osdRenderer.refinementLevel];
        [_refLvlLabel sizeToFit];
    }
}

- (IBAction)switchChanged:(UISwitch *)sender {
    if(sender == _wireframeSwitch) {
        _osdRenderer.displayStyle = _wireframeSwitch.isOn ? kDisplayStyleWireOnShaded : kDisplayStyleShaded;
    } else if(sender == _backpatchCullingSwitch) {
        _osdRenderer.usePatchBackfaceCulling = sender.isOn;
    } else if(sender == _backfaceCullingSwitch) {
        _osdRenderer.usePrimitiveBackfaceCulling = sender.isOn;
    } else if(sender == _patchClipCullingSwitch) {
        _osdRenderer.usePatchClipCulling = sender.isOn;
    } else if(sender == _smoothCornerSwitch) {
        _osdRenderer.useSmoothCornerPatch = sender.isOn;
    } else if(sender == _singleCreaseSwitch) {
        _osdRenderer.useSingleCreasePatch = sender.isOn;
    } else if(sender == _infinitelySharpSwitch) {
        _osdRenderer.useInfinitelySharpPatch = sender.isOn;
    } else if(sender == _controlMeshSwitch) {
        _osdRenderer.displayControlMeshEdges = sender.isOn;
        _osdRenderer.displayControlMeshVertices = sender.isOn;
    } else if(sender == _screenspaceTessellationSwitch) {
        _osdRenderer.useScreenspaceTessellation = sender.isOn;
        _osdRenderer.useFractionalTessellation = _osdRenderer.useFractionalTessellation;
    }
}

- (IBAction)endcapChanged:(UISegmentedControl *)sender {
    _osdRenderer.endCapMode = (EndCap)(sender.selectedSegmentIndex);
}

-(NSInteger)numberOfComponentsInPickerView:(UIPickerView *)pickerView {
    return 1;
}

-(NSInteger)pickerView:(UIPickerView *)pickerView numberOfRowsInComponent:(NSInteger)component {
    if(pickerView == _modelPickerView) {
        return _osdRenderer.loadedModels.count;
    } else if(pickerView == _shadingModePickerView) {
        return 4;
    }
    return 0;
}

-(NSString *)pickerView:(UIPickerView *)pickerView titleForRow:(NSInteger)row forComponent:(NSInteger)component {
    if(pickerView == _modelPickerView) {
        return _osdRenderer.loadedModels[row];
    } else if(pickerView == _shadingModePickerView) {
        switch((ShadingMode)row) {
            case kShadingMaterial: return @"Material";
            case kShadingFaceVaryingColor: return @"FaceVarying Color";
            case kShadingPatchType: return @"Patch Type";
            case kShadingPatchDepth: return @"Patch Depth";
            case kShadingPatchCoord: return @"Patch Coord";
            case kShadingNormal: return @"Normal";
        }
    }
    return @"";
}

-(void)pickerView:(UIPickerView *)pickerView didSelectRow:(NSInteger)row inComponent:(NSInteger)component {
    if(pickerView == _modelPickerView) {
        _osdRenderer.currentModel = _osdRenderer.loadedModels[row];
    } else if(pickerView == _shadingModePickerView) {
        _osdRenderer.shadingMode = (ShadingMode)row;
    }
}
@end
