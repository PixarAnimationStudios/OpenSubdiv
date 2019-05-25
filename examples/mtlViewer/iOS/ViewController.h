#import <UIKit/UIKit.h>
#import <MetalKit/MetalKit.h>
#import "../mtlViewer.h"

@interface ViewController : UIViewController<
                                MTKViewDelegate,
                                UIPickerViewDelegate,
                                UIPickerViewDataSource,
                                OSDRendererDelegate
                            >

@property (weak, nonatomic) IBOutlet UILabel *refLvlLabel;
@property (weak, nonatomic) IBOutlet UILabel *tesLvLlabel;
@property (weak, nonatomic) IBOutlet UILabel *frameTimeLabel;
@property (weak, nonatomic) IBOutlet UIPickerView *modelPickerView;
@property (weak, nonatomic) IBOutlet UIPickerView *shadingModePickerView;
@property (weak, nonatomic) IBOutlet UIStepper *tessellationStepper;
@property (weak, nonatomic) IBOutlet UIStepper *refinementStepper;
@property (weak, nonatomic) IBOutlet UISwitch *wireframeSwitch;
@property (weak, nonatomic) IBOutlet UISwitch *backpatchCullingSwitch;
@property (weak, nonatomic) IBOutlet UISwitch *backfaceCullingSwitch;
@property (weak, nonatomic) IBOutlet UISwitch *patchClipCullingSwitch;
@property (weak, nonatomic) IBOutlet UISwitch *smoothCornerSwitch;
@property (weak, nonatomic) IBOutlet UISwitch *singleCreaseSwitch;
@property (weak, nonatomic) IBOutlet UISwitch *infinitelySharpSwitch;
@property (weak, nonatomic) IBOutlet UISwitch *controlMeshSwitch;
@property (weak, nonatomic) IBOutlet UISwitch *screenspaceTessellationSwitch;
@property (weak, nonatomic) IBOutlet UISegmentedControl *endcapSegmentedControl;
- (IBAction)stepperChanged:(UIStepper *)sender;
- (IBAction)switchChanged:(UISwitch *)sender;
- (IBAction)endcapChanged:(UISegmentedControl *)sender;


@end

