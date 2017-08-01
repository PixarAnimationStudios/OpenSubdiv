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

#import <UIKit/UIKit.h>
#import <MetalKit/MetalKit.h>
#import "../mtlPtexViewer.h"

@interface ViewController : UIViewController<
                                MTKViewDelegate,
                                UIPickerViewDelegate,
                                UIPickerViewDataSource,
                                OSDRendererDelegate
                            >

@property (weak, nonatomic) IBOutlet UILabel *refLvlLabel;
@property (weak, nonatomic) IBOutlet UILabel *tesLvLlabel;
@property (weak, nonatomic) IBOutlet UILabel *frameTimeLabel;
@property (weak, nonatomic) IBOutlet UIPickerView *colorModePickerView;
@property (weak, nonatomic) IBOutlet UIPickerView *normalModePickerView;
@property (weak, nonatomic) IBOutlet UIPickerView *displacementModePickerView;
@property (weak, nonatomic) IBOutlet UIStepper *tessellationStepper;
@property (weak, nonatomic) IBOutlet UIStepper *refinementStepper;
@property (weak, nonatomic) IBOutlet UISwitch *wireframeSwitch;
@property (weak, nonatomic) IBOutlet UISwitch *backpatchCullingSwitch;
@property (weak, nonatomic) IBOutlet UISwitch *backfaceCullingSwitch;
@property (weak, nonatomic) IBOutlet UISwitch *patchClipCullingSwitch;
@property (weak, nonatomic) IBOutlet UISwitch *singleCreaseSwitch;
@property (weak, nonatomic) IBOutlet UISwitch *screenspaceTessellationSwitch;
@property (weak, nonatomic) IBOutlet UISlider *displacementSlider;
@property (weak, nonatomic) IBOutlet UISlider *mipmapBiasSlider;
- (IBAction)stepperChanged:(UIStepper *)sender;
- (IBAction)switchChanged:(UISwitch *)sender;
- (IBAction)sliderChanged:(UISlider*)sender;

@end

