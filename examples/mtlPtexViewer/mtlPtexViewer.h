#pragma once

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

typedef enum {
    kCPU = 0,
    kMetal,
} KernelType;

typedef enum {
    kDisplacementModeHWBilinear = 0,
    kDisplacementModeBilinear,
    kDisplacementModeBiQuadratic,
    kDisplacementModeNone
} DisplacementMode;

typedef enum {
    kNormalModeHWScreenspace = 0,
    kNormalModeScreenspace,
    kNormalModeBiQuadratic,
    kNormalModeBiQuadraticWG,
    kNormalModeSurface
} NormalMode;

typedef enum {
    kDisplayStyleWire = 0,
    kDisplayStyleShaded,
    kDisplayStyleWireOnShaded,
} DisplayStyle;

typedef enum {
    kColorModePtexNearest = 0,
    kColorModePtexBilinear,
    kColorModePtexHWBilinear,
    kColorModePtexBiQuadratic,
    kColorModePatchType,
    kColorModePatchCoord,
    kColorModeNormal,
    kColorModeNone
} ColorMode;

typedef struct {
    float rotationX;
    float rotationY;
    float dollyDistance;
    float aspectRatio;
} Camera;

@class OSDRenderer;

@protocol OSDRendererDelegate <NSObject>
-(id<MTLDevice>)deviceFor:(OSDRenderer*)renderer;
-(id<MTLCommandQueue>)commandQueueFor:(OSDRenderer*)renderer;
-(MTLRenderPassDescriptor*)renderPassDescriptorFor:(OSDRenderer*)renderer;
-(void)setupDepthStencilState:(MTLDepthStencilDescriptor*)descriptor for:(OSDRenderer*)renderer;
-(void)setupRenderPipelineState:(MTLRenderPipelineDescriptor*)descriptor for:(OSDRenderer*)renderer;
@end

@interface OSDRenderer : NSObject

-(instancetype)initWithDelegate:(id<OSDRendererDelegate>)delegate;

-(id<MTLRenderCommandEncoder>)drawFrame:(id<MTLCommandBuffer>)commandBuffer;

-(void)fitFrame;

@property (readonly, nonatomic) id<OSDRendererDelegate> delegate;

@property (nonatomic) unsigned refinementLevel;
@property (nonatomic) int tessellationLevel;

@property (readonly, nonatomic) Camera* camera;


@property (nonatomic) bool useSeamlessMipmap;
@property (nonatomic) bool useFractionalTessellation;
@property (nonatomic) bool useScreenspaceTessellation;
@property (nonatomic) bool usePatchIndexBuffer;
@property (nonatomic) bool usePatchBackfaceCulling;
@property (nonatomic) bool usePatchClipCulling;
@property (nonatomic) bool useSmoothCornerPatch;
@property (nonatomic) bool useSingleCreasePatch;
@property (nonatomic) bool useInfinitelySharpPatch;
@property (nonatomic) bool useStageIn;
@property (nonatomic) bool usePrimitiveBackfaceCulling;
@property (nonatomic) bool useAdaptive;
@property (nonatomic) bool freeze;
@property (nonatomic) bool animateVertices;
@property (nonatomic) bool displayControlMeshEdges;
@property (nonatomic) bool displayControlMeshVertices;
@property (nonatomic) bool displaySpecular;
@property (nonatomic) bool displayOcclusion;
@property (nonatomic) bool yup;
@property (nonatomic) float mipmapBias;
@property (nonatomic) float displacementScale;

@property (nonatomic) NSString* ptexColorFilename;
@property (nonatomic) NSString* ptexDisplacementFilename;
@property (nonatomic) NSString* ptexOcclusionFilename;
@property (nonatomic) NSString* ptexSpecularFilename;

@property (nonatomic) ColorMode colorMode;
@property (nonatomic) NormalMode normalMode;
@property (nonatomic) DisplacementMode displacementMode;
@property (nonatomic) DisplayStyle displayStyle;
@property (nonatomic) KernelType kernelType;
@end
