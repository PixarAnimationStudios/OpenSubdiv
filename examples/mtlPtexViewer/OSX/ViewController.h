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

#import <AppKit/AppKit.h>
#import <MetalKit/MetalKit.h>
#import "../mtlPtexViewer.h"
#import "../../common/mtlHud.h"

@class ViewController;

@interface OSDView : MTKView {
    @public
    MTLhud hud;
};

@property (nonatomic) ViewController* controller;
@end

@interface ViewController : NSViewController<MTKViewDelegate, OSDRendererDelegate>
@property (weak) IBOutlet OSDView *view;
@property (nonatomic) OSDRenderer* osdRenderer;

- (IBAction)checkboxChanged:(NSButton *)sender;
- (IBAction)popupChanged:(NSPopUpButton *)sender;
- (IBAction)sliderChanged:(NSSlider *)sender;

@property (weak) IBOutlet NSTextField *frameTimeLabel;
@property (weak) IBOutlet NSButton *wireframeCheckbox;
@property (weak) IBOutlet NSButton *singleCreaseCheckbox;
@property (weak) IBOutlet NSButton *patchIndexCheckbox;
@property (weak) IBOutlet NSButton *patchClipCullingCheckbox;
@property (weak) IBOutlet NSButton *backfaceCullingCheckbox;
@property (weak) IBOutlet NSButton *backpatchCullingCheckbox;
@property (weak) IBOutlet NSButton *screenspaceTessellationCheckbox;
@property (weak) IBOutlet NSPopUpButton *modelPopup;
@property (weak) IBOutlet NSPopUpButton *refinementLevelPopup;
@property (weak) IBOutlet NSPopUpButton *tessellationLevelPopup;
@property (weak) IBOutlet NSPopUpButton *displacementModePopup;
@property (weak) IBOutlet NSPopUpButton *normalModePopup;
@property (weak) IBOutlet NSPopUpButton *colorModePopup;
@property (weak) IBOutlet NSSlider *displacementScaleSlider;
@property (weak) IBOutlet NSSlider *mipmapBiasSlider;


@end
