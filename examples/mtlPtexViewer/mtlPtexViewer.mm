
#import "mtlPtexViewer.h"

#import <simd/simd.h>
#import <algorithm>
#import <cfloat>
#import <fstream>
#import <iostream>
#import <iterator>
#import <memory>
#import <sstream>
#import <string>
#import <vector>

#import <opensubdiv/far/error.h>
#import <opensubdiv/osd/mesh.h>
#import <opensubdiv/osd/cpuVertexBuffer.h>
#import <opensubdiv/osd/cpuEvaluator.h>
#import <opensubdiv/osd/cpuPatchTable.h>
#import <opensubdiv/osd/mtlVertexBuffer.h>
#import <opensubdiv/osd/mtlMesh.h>
#import <opensubdiv/osd/mtlPatchTable.h>
#import <opensubdiv/osd/mtlComputeEvaluator.h>
#import <opensubdiv/osd/mtlPatchShaderSource.h>

#import "../../regression/common/far_utils.h"
#import "../../regression/common/arg_utils.h"
#import "../common/mtlUtils.h"
#import "../common/mtlControlMeshDisplay.h"
#import "../common/mtlPtexMipmapTexture.h"
#import "../common/simple_math.h"

#define VERTEX_BUFFER_INDEX 0
#define PATCH_INDICES_BUFFER_INDEX 1
#define CONTROL_INDICES_BUFFER_INDEX 2
#define OSD_PERPATCHVERTEX_BUFFER_INDEX 3
#define OSD_PERPATCHVERTEXBEZIER_BUFFER_INDEX OSD_PERPATCHVERTEX_BUFFER_INDEX
#define OSD_PERPATCHVERTEXGREGORY_BUFFER_INDEX OSD_PERPATCHVERTEX_BUFFER_INDEX
#define OSD_PATCHPARAM_BUFFER_INDEX 4
#define OSD_VALENCE_BUFFER_INDEX 6
#define OSD_QUADOFFSET_BUFFER_INDEX 7
#define OSD_PERPATCHTESSFACTORS_BUFFER_INDEX 8
#define PATCH_TESSFACTORS_INDEX 10
#define QUAD_TESSFACTORS_INDEX PATCH_TESSFACTORS_INDEX
#define TRIANGLE_TESSFACTORS_INDEX PATCH_TESSFACTORS_INDEX
#define OSD_PATCH_INDEX_BUFFER_INDEX 13
#define OSD_DRAWINDIRECT_BUFFER_INDEX 14
#define OSD_KERNELLIMIT_BUFFER_INDEX 15

#define DISPLACEMENT_TEXTURE_INDEX 0
#define IMAGE_TEXTURE_INDEX 1
#define OCCLUSION_TEXTURE_INDEX 2
#define SPECULAR_TEXTURE_INDEX 3

#define DISPLACEMENT_BUFFER_INDEX 15
#define IMAGE_BUFFER_INDEX 16
#define OCCLUSION_BUFFER_INDEX 17
#define SPECULAR_BUFFER_INDEX 18
#define CONFIG_BUFFER_INDEX 19

#define FRAME_CONST_BUFFER_INDEX 11
#define INDICES_BUFFER_INDEX 2

using namespace OpenSubdiv;

template <> Far::StencilTable const *
Osd::convertToCompatibleStencilTable<
        Far::StencilTable,
        Far::StencilTable,
        Osd::MTLContext>(Far::StencilTable const *table,
                         Osd::MTLContext* /*context*/) {
    // no need for conversion
    // XXX: We don't want to even copy.
    if (not table) return NULL;
    return new Far::StencilTable(*table);
}

struct alignas(256) DisplacementConfig {
    float displacementScale;
    float mipmapBias;
};

using CPUMeshType = Osd::Mesh<
    Osd::CPUMTLVertexBuffer,
    Far::StencilTable,
    Osd::CpuEvaluator,
    Osd::MTLPatchTable,
    Osd::MTLContext>;

using MTLMeshType = Osd::Mesh<
    Osd::CPUMTLVertexBuffer,
    Osd::MTLStencilTable,
    Osd::MTLComputeEvaluator,
    Osd::MTLPatchTable,
    Osd::MTLContext>;

using MTLMeshInterface = Osd::MTLMeshInterface;

struct alignas(256) PerFrameConstants
{
    float ModelViewMatrix[16];
    float ProjectionMatrix[16];
    float ModelViewProjectionMatrix[16];
    float ModelViewInverseMatrix[16];
    float TessLevel;
    DisplacementConfig displacementConfig;
};

struct alignas(16) Light {
    simd::float4 position;
    simd::float4 ambient;
    simd::float4 diffuse;
    simd::float4 specular;
};

static const char* shaderSource =
#include "mtlPtexViewer.gen.h"
;

using Osd::MTLRingBuffer;

#define FRAME_LAG 3
template<typename DataType>
using PerFrameBuffer = MTLRingBuffer<DataType, FRAME_LAG>;

#define DISPATCHSLOTS 11 // XXXdyu-mtl

@implementation OSDRenderer {

    MTLRingBuffer<Light, 1> _lightsBuffer;

    PerFrameBuffer<PerFrameConstants> _frameConstantsBuffer;
    PerFrameBuffer<uint8_t> _tessFactorsBuffer;
    PerFrameBuffer<uint8_t> _perPatchVertexBuffer;
    PerFrameBuffer<uint8_t> _perPatchTessFactorsBuffer;
    PerFrameBuffer<MTLDrawPatchIndirectArguments> _drawIndirectCommandsBuffer;

    PerFrameBuffer<unsigned> _patchIndexBuffers[DISPATCHSLOTS];

    unsigned _tessFactorsOffsets[DISPATCHSLOTS];
    unsigned _perPatchVertexOffsets[DISPATCHSLOTS];
    unsigned _perPatchTessFactorsOffsets[DISPATCHSLOTS];

    unsigned _threadgroupSizes[DISPATCHSLOTS];
    id<MTLComputePipelineState> _computePipelines[DISPATCHSLOTS];
    id<MTLRenderPipelineState> _renderPipelines[DISPATCHSLOTS];

    id<MTLDepthStencilState> _readWriteDepthStencilState;
    id<MTLDepthStencilState> _readOnlyDepthStencilState;

    Camera _cameraData;
    Osd::MTLContext _context;

    std::unique_ptr<MTLPtexMipmapTexture> _colorPtexture;
    std::unique_ptr<MTLPtexMipmapTexture> _displacementPtexture;
    std::unique_ptr<MTLPtexMipmapTexture> _occlusionPtexture;
    std::unique_ptr<MTLPtexMipmapTexture> _specularPtexture;

    int _numVertexElements;
    int _numVertices;
    int _frameCount;
    int _animationFrames;
    std::vector<float> _vertexData, _animatedVertices;

    std::unique_ptr<MTLMeshInterface> _mesh;
    std::unique_ptr<MTLControlMeshDisplay> _controlMesh;
    std::unique_ptr<Shape> _shape;

    bool _needsRebuild;
    NSString* _osdShaderSource;
    simd::float3 _meshCenter;
    float _meshSize;
}

-(Camera*)camera {
    return &_cameraData;
}

struct PipelineConfig {
    Far::PatchDescriptor::Type patchType;
    bool useTessellation;
    bool useTriangleTessellation;
    bool useSingleCreasePatch;
    bool useLegacyBuffers;
    bool drawIndexed;
    int numControlPointsPerPatchRefined;
    int numControlPointsPerPatchToDraw;
    int numControlPointsPerThreadRefined;
    int numControlPointsPerThreadToDraw;
    int numThreadsPerPatch;
};

-(PipelineConfig) _lookupPipelineConfig:(Far::PatchDescriptor::Type) patchType
                                         useSingleCreasePatch:(bool) useSingleCreasePatch {
    PipelineConfig config;

    config.patchType = patchType;
    config.useTessellation = false;
    config.useTriangleTessellation = false;
    config.useSingleCreasePatch = false;
    config.useLegacyBuffers = false;
    config.drawIndexed = false;
    switch(config.patchType)
    {
        case Far::PatchDescriptor::QUADS:
            config.numControlPointsPerPatchRefined = 4;
            config.numControlPointsPerPatchToDraw = 4;
            config.numControlPointsPerThreadRefined = 4;
            config.numControlPointsPerThreadToDraw = 4;
            config.numThreadsPerPatch = 1;
        break;
        case Far::PatchDescriptor::TRIANGLES:
            config.numControlPointsPerPatchRefined = 3;
            config.numControlPointsPerPatchToDraw = 3;
            config.numControlPointsPerThreadRefined = 3;
            config.numControlPointsPerThreadToDraw = 3;
            config.numThreadsPerPatch = 1;
        break;
        case Far::PatchDescriptor::LOOP:
            config.useTessellation = true;
            config.useTriangleTessellation = true;
            config.numControlPointsPerPatchRefined = 12;
            config.numControlPointsPerPatchToDraw = 15;
            config.numControlPointsPerThreadRefined = 3;
            config.numControlPointsPerThreadToDraw = 4;
            config.numThreadsPerPatch = 4;
        break;
        case Far::PatchDescriptor::REGULAR:
            config.useTessellation = true;
            config.useSingleCreasePatch = useSingleCreasePatch;
            config.numControlPointsPerPatchRefined = 16;
            config.numControlPointsPerPatchToDraw = 16;
            config.numControlPointsPerThreadRefined = 4;
            config.numControlPointsPerThreadToDraw = 4;
            config.numThreadsPerPatch = 4;
        break;
        case Far::PatchDescriptor::GREGORY:
            config.useTessellation = true;
            config.useLegacyBuffers = true;
            config.numControlPointsPerPatchRefined = 4;
            config.numControlPointsPerPatchToDraw = 4;
            config.numControlPointsPerThreadRefined = 1;
            config.numControlPointsPerThreadToDraw = 5;
            config.numThreadsPerPatch = 4;
        break;
        case Far::PatchDescriptor::GREGORY_BOUNDARY:
            config.useTessellation = true;
            config.useLegacyBuffers = true;
            config.numControlPointsPerPatchRefined = 4;
            config.numControlPointsPerPatchToDraw = 4;
            config.numControlPointsPerThreadRefined = 1;
            config.numControlPointsPerThreadToDraw = 5;
            config.numThreadsPerPatch = 4;
        break;
        case Far::PatchDescriptor::GREGORY_BASIS:
            config.useTessellation = true;
            config.drawIndexed = true;
            config.numControlPointsPerPatchRefined = 20;
            config.numControlPointsPerPatchToDraw = 20;
            config.numControlPointsPerThreadRefined = 5;
            config.numControlPointsPerThreadToDraw = 5;
            config.numThreadsPerPatch = 4;
        break;
        case Far::PatchDescriptor::GREGORY_TRIANGLE:
            config.useTessellation = true;
            config.useTriangleTessellation = true;
            config.drawIndexed = true;
            config.numControlPointsPerPatchRefined = 18;
            config.numControlPointsPerPatchToDraw = 18;
            config.numControlPointsPerThreadRefined = 5;
            config.numControlPointsPerThreadToDraw = 5;
            config.numThreadsPerPatch = 4;
        break;
        default:
            assert("Unsupported patch type" && 0); break;
    }
    return config;
}

-(void)_processArgs {

    NSEnumerator *argsArray =
            [[[NSProcessInfo processInfo] arguments] objectEnumerator];

    std::vector<char *> argsVector;
    for (id arg in argsArray) {
        argsVector.push_back((char *)[arg UTF8String]);
    }

    ArgOptions args;

    args.Parse(argsVector.size(), argsVector.data());

    self.yup = args.GetYUp();
    self.useAdaptive = args.GetAdaptive();
    self.refinementLevel = args.GetLevel();

    const char * colorFilename = getenv("COLOR_FILENAME");
    const char * displacementFilename = getenv("DISPLACEMENT_FILENAME");

    const std::vector<const char *> &argvRem = args.GetRemainingArgs();
    for (size_t i = 0; i < argvRem.size(); ++i) {
        if (!colorFilename) {
            colorFilename = argvRem[i];
        } else if (!displacementFilename) {
            displacementFilename = argvRem[i];
        }
    }

    if (colorFilename) {
        _ptexColorFilename =
                [NSString stringWithUTF8String:colorFilename];
    }

    if (displacementFilename) {
        _ptexDisplacementFilename =
                [NSString stringWithUTF8String:displacementFilename];
    }
}

-(instancetype)initWithDelegate:(id<OSDRendererDelegate>)delegate {
    self = [super init];
    if (self) {
        self.useSmoothCornerPatch = true;
        self.useSingleCreasePatch = true;
        self.useInfinitelySharpPatch = true;
        self.useStageIn = !TARGET_OS_EMBEDDED;
        self.useSeamlessMipmap = true;
        self.useScreenspaceTessellation = true;
        self.useFractionalTessellation = false;
        self.usePatchClipCulling = false;
        self.usePatchIndexBuffer = false;
        self.usePatchBackfaceCulling = false;
        self.usePrimitiveBackfaceCulling = false;
        self.useAdaptive = true;
        self.yup = false;
        self.kernelType = kMetal;
        self.refinementLevel = 2;
        self.tessellationLevel = 1;
        self.normalMode = kNormalModeSurface;
        self.colorMode = kColorModePtexBilinear;
        self.displacementMode = kDisplacementModeNone;
        self.displayStyle = kDisplayStyleShaded;
        self.mipmapBias = 0.0;
        self.displacementScale = 1.0;

        [self _processArgs];

        if (_ptexDisplacementFilename) {
            self.displacementMode = kDisplacementModeBilinear;
            self.normalMode = kNormalModeBiQuadratic;
        }

        _frameCount = 0;
        _animationFrames = 0;
        _delegate = delegate;
        _context.device = [delegate deviceFor:self];
        _context.commandQueue = [delegate commandQueueFor:self];

        _osdShaderSource = @(shaderSource);

        _needsRebuild = true;

        [self _initializeBuffers];
        [self _initializeCamera];
        [self _initializeLights];
    }
    return self;
}

-(id<MTLRenderCommandEncoder>)drawFrame:(id<MTLCommandBuffer>)commandBuffer {
    if(_needsRebuild) {
        [self _rebuildState];
    }

    if(!_freeze) {
        if(_animateVertices) {
            _animatedVertices.resize(_vertexData.size());
            auto p = _vertexData.data();
            auto n = _animatedVertices.data();

            float r = sin(_animationFrames*0.01f) * _animateVertices;
            for (int i = 0; i < _numVertices; ++i) {
                float ct = cos(p[2] * r);
                float st = sin(p[2] * r);
                n[0] = p[0]*ct + p[1]*st;
                n[1] = -p[0]*st + p[1]*ct;
                n[2] = p[2];

                p += _numVertexElements;
                n += _numVertexElements;
            }

            _mesh->UpdateVertexBuffer(_animatedVertices.data(), 0, _numVertices);
            _animationFrames++;
        }
        _mesh->Refine();
        _mesh->Synchronize();
    }

    [self _updateState];

    if (_useAdaptive) {
        auto computeEncoder = [commandBuffer computeCommandEncoder];
        [self _computeTessFactors:computeEncoder];
        [computeEncoder endEncoding];
    }

    auto renderEncoder = [commandBuffer renderCommandEncoderWithDescriptor:[_delegate renderPassDescriptorFor: self]];

    if (_usePrimitiveBackfaceCulling) {
        [renderEncoder setCullMode:MTLCullModeBack];
    } else {
        [renderEncoder setCullMode:MTLCullModeNone];
    }

    [self _renderMesh:renderEncoder];

    _lightsBuffer.next();

    _frameConstantsBuffer.next();
    _tessFactorsBuffer.next();
    _perPatchVertexBuffer.next();
    _perPatchTessFactorsBuffer.next();
    for (int i=0; i<DISPATCHSLOTS; ++i) {
        _patchIndexBuffers[i].next();
    }
    _drawIndirectCommandsBuffer.next();

    _frameCount++;

    return renderEncoder;
}

-(void)fitFrame {
    _cameraData.dollyDistance = _meshSize;
}

-(void)_renderMesh:(id<MTLRenderCommandEncoder>)renderCommandEncoder {

    auto patchVertexBuffer = _mesh->BindVertexBuffer();
    auto patchIndexBuffer = _mesh->GetPatchTable()->GetPatchIndexBuffer();

    [renderCommandEncoder setVertexBuffer:patchVertexBuffer offset:0 atIndex:VERTEX_BUFFER_INDEX];
    [renderCommandEncoder setVertexBuffer:patchIndexBuffer offset:0 atIndex:INDICES_BUFFER_INDEX];

    [renderCommandEncoder setVertexBuffer:_frameConstantsBuffer offset:0 atIndex:FRAME_CONST_BUFFER_INDEX];
    [renderCommandEncoder setFragmentBuffer:_lightsBuffer offset:0 atIndex:0];

    [renderCommandEncoder setFragmentBuffer:_frameConstantsBuffer offset:offsetof(PerFrameConstants, displacementConfig) atIndex:1];
    [renderCommandEncoder setVertexBuffer:_frameConstantsBuffer offset:offsetof(PerFrameConstants, displacementConfig) atIndex:CONFIG_BUFFER_INDEX];

    if(_useAdaptive)
    {
        [renderCommandEncoder setVertexBuffer:_perPatchTessFactorsBuffer offset:0 atIndex:OSD_PERPATCHTESSFACTORS_BUFFER_INDEX];
        [renderCommandEncoder setVertexBuffer:_perPatchVertexBuffer offset:0 atIndex:OSD_PERPATCHVERTEX_BUFFER_INDEX];
        [renderCommandEncoder setVertexBuffer:_mesh->GetPatchTable()->GetPatchParamBuffer() offset:0 atIndex:OSD_PATCHPARAM_BUFFER_INDEX];
    }

    [renderCommandEncoder setFragmentTexture:_colorPtexture->GetTexelsTexture() atIndex:IMAGE_TEXTURE_INDEX];
    [renderCommandEncoder setFragmentBuffer:_colorPtexture->GetLayoutBuffer() offset:0 atIndex:IMAGE_BUFFER_INDEX];
    if(_displacementPtexture)
    {
        [renderCommandEncoder setFragmentTexture:_displacementPtexture->GetTexelsTexture() atIndex:DISPLACEMENT_TEXTURE_INDEX];
        [renderCommandEncoder setFragmentBuffer:_displacementPtexture->GetLayoutBuffer() offset:0 atIndex:DISPLACEMENT_BUFFER_INDEX];

        [renderCommandEncoder setVertexTexture:_displacementPtexture->GetTexelsTexture() atIndex:DISPLACEMENT_TEXTURE_INDEX];
        [renderCommandEncoder setVertexBuffer:_displacementPtexture->GetLayoutBuffer() offset:0 atIndex:DISPLACEMENT_BUFFER_INDEX];
    }

    if(_displayStyle == kDisplayStyleWire)
        [renderCommandEncoder setTriangleFillMode:MTLTriangleFillModeLines];
    else
        [renderCommandEncoder setTriangleFillMode:MTLTriangleFillModeFill];

    for (auto& patch : _mesh->GetPatchTable()->GetPatchArrays())
    {
        auto patchType = patch.desc.GetType();
        PipelineConfig pipelineConfig = [self _lookupPipelineConfig:patchType useSingleCreasePatch:_useSingleCreasePatch];

        [renderCommandEncoder setVertexBufferOffset:patch.indexBase * sizeof(unsigned) atIndex:INDICES_BUFFER_INDEX];

        simd::float4 shade{.0f,0.0f,0.0f,1.0f};
        [renderCommandEncoder setFragmentBytes:&shade length:sizeof(shade) atIndex:2];
        [renderCommandEncoder setDepthBias:0 slopeScale:1.0 clamp:0];
        [renderCommandEncoder setDepthStencilState:_readWriteDepthStencilState];

        [renderCommandEncoder setRenderPipelineState:_renderPipelines[patchType]];

        [renderCommandEncoder setFrontFacingWinding:MTLWindingCounterClockwise];

        if (pipelineConfig.useTessellation) {
            [renderCommandEncoder setVertexBufferOffset:patch.primitiveIdBase * sizeof(int) * 3 atIndex:OSD_PATCHPARAM_BUFFER_INDEX];

            [renderCommandEncoder setTessellationFactorBuffer:_tessFactorsBuffer offset:_tessFactorsOffsets[patchType] instanceStride:0];
            [renderCommandEncoder setVertexBufferOffset:_perPatchTessFactorsOffsets[patchType] atIndex:OSD_PERPATCHTESSFACTORS_BUFFER_INDEX];

            if (!pipelineConfig.drawIndexed) {
                [renderCommandEncoder setVertexBufferOffset:_perPatchVertexOffsets[patchType] atIndex:OSD_PERPATCHVERTEX_BUFFER_INDEX];
            }

            if (_usePatchIndexBuffer) {
                if (pipelineConfig.drawIndexed) {
                    [renderCommandEncoder drawIndexedPatches:pipelineConfig.numControlPointsPerPatchToDraw
                                      patchStart:0 patchCount:patch.GetNumPatches()
                                      patchIndexBuffer:_patchIndexBuffers[patchType] patchIndexBufferOffset:0
                                      controlPointIndexBuffer:patchIndexBuffer controlPointIndexBufferOffset:patch.indexBase * sizeof(unsigned)
                                      instanceCount:1 baseInstance:0];
                } else {
                    [renderCommandEncoder drawPatches:pipelineConfig.numControlPointsPerPatchToDraw
                        patchIndexBuffer:_patchIndexBuffers[patchType] patchIndexBufferOffset:0
                          indirectBuffer:_drawIndirectCommandsBuffer indirectBufferOffset: sizeof(MTLDrawPatchIndirectArguments) * patchType];
                }
            } else {
                if (pipelineConfig.drawIndexed) {
                    [renderCommandEncoder drawIndexedPatches:pipelineConfig.numControlPointsPerPatchToDraw
                                      patchStart:0 patchCount:patch.GetNumPatches()
                                      patchIndexBuffer:nil patchIndexBufferOffset:0
                                      controlPointIndexBuffer:patchIndexBuffer controlPointIndexBufferOffset:patch.indexBase * sizeof(unsigned)
                                      instanceCount:1 baseInstance:0];
                } else {
                    [renderCommandEncoder drawPatches:pipelineConfig.numControlPointsPerPatchToDraw
                        patchStart:0 patchCount:patch.GetNumPatches()
                        patchIndexBuffer:nil patchIndexBufferOffset:0 instanceCount:1 baseInstance:0];
                }
            }

            if(_displayStyle == kDisplayStyleWireOnShaded)
            {
                simd::float4 shade = {1,1,1,1};
                [renderCommandEncoder setFragmentBytes:&shade length:sizeof(shade) atIndex:2];
                [renderCommandEncoder setTriangleFillMode:MTLTriangleFillModeLines];
                [renderCommandEncoder setDepthBias:-5 slopeScale:-1.0 clamp:-100.0];

                if (_usePatchIndexBuffer) {
                    if (pipelineConfig.drawIndexed) {
                        [renderCommandEncoder drawIndexedPatches:pipelineConfig.numControlPointsPerPatchToDraw
                                         patchStart:0 patchCount:patch.GetNumPatches()
                                         patchIndexBuffer:_patchIndexBuffers[patchType] patchIndexBufferOffset:0
                                         controlPointIndexBuffer:patchIndexBuffer controlPointIndexBufferOffset:patch.indexBase * sizeof(unsigned)
                                         instanceCount:1 baseInstance:0];
                    } else {
                        [renderCommandEncoder drawPatches:pipelineConfig.numControlPointsPerPatchToDraw
                                         patchIndexBuffer:_patchIndexBuffers[patchType] patchIndexBufferOffset:0
                                           indirectBuffer:_drawIndirectCommandsBuffer indirectBufferOffset: sizeof(MTLDrawPatchIndirectArguments) * patchType];
                    }
                } else {
                    if (pipelineConfig.drawIndexed) {
                        [renderCommandEncoder drawIndexedPatches:pipelineConfig.numControlPointsPerPatchToDraw
                                         patchStart:0 patchCount:patch.GetNumPatches()
                                         patchIndexBuffer:nil patchIndexBufferOffset:0
                                         controlPointIndexBuffer:patchIndexBuffer controlPointIndexBufferOffset:patch.indexBase * sizeof(unsigned)
                                         instanceCount:1 baseInstance:0];
                    } else {
                        [renderCommandEncoder drawPatches:pipelineConfig.numControlPointsPerPatchToDraw
                                         patchStart:0 patchCount:patch.GetNumPatches()
                                         patchIndexBuffer:nil patchIndexBufferOffset:0 instanceCount:1 baseInstance:0];
                    }
                }

                [renderCommandEncoder setTriangleFillMode:MTLTriangleFillModeFill];
            }
        } else {
            if (patchType == Far::PatchDescriptor::QUADS) {
                [renderCommandEncoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:patch.GetNumPatches() * 6];
                if(_displayStyle == kDisplayStyleWireOnShaded)
                {
                    simd::float4 shade = {1,1,1,1};
                    [renderCommandEncoder setFragmentBytes:&shade length:sizeof(shade) atIndex:2];
                    [renderCommandEncoder setTriangleFillMode:MTLTriangleFillModeLines];
                    [renderCommandEncoder setDepthBias:-5 slopeScale:-1.0 clamp:-100.0];
                    [renderCommandEncoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:patch.GetNumPatches() * 6];
                    [renderCommandEncoder setTriangleFillMode:MTLTriangleFillModeFill];
                }
            } else if (patchType == Far::PatchDescriptor::TRIANGLES) {
                [renderCommandEncoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:patch.GetNumPatches() * 3];
                if(_displayStyle == kDisplayStyleWireOnShaded)
                {
                    simd::float4 shade = {1,1,1,1};
                    [renderCommandEncoder setFragmentBytes:&shade length:sizeof(shade) atIndex:2];
                    [renderCommandEncoder setTriangleFillMode:MTLTriangleFillModeLines];
                    [renderCommandEncoder setDepthBias:-5 slopeScale:-1.0 clamp:-100.0];
                    [renderCommandEncoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:patch.GetNumPatches() * 3];
                    [renderCommandEncoder setTriangleFillMode:MTLTriangleFillModeFill];
                }
            }
        }
    }
}

-(void)_computeTessFactors:(id<MTLComputeCommandEncoder>)computeCommandEncoder {

    [computeCommandEncoder setBuffer:_mesh->BindVertexBuffer() offset:0 atIndex:VERTEX_BUFFER_INDEX];
    [computeCommandEncoder setBuffer:_mesh->GetPatchTable()->GetPatchIndexBuffer() offset:0 atIndex:CONTROL_INDICES_BUFFER_INDEX];
    [computeCommandEncoder setBuffer:_mesh->GetPatchTable()->GetPatchParamBuffer() offset:0 atIndex:OSD_PATCHPARAM_BUFFER_INDEX];
    [computeCommandEncoder setBuffer:_frameConstantsBuffer offset:0 atIndex:FRAME_CONST_BUFFER_INDEX];

    for (auto& patch : _mesh->GetPatchTable()->GetPatchArrays())
    {
        auto patchType = patch.desc.GetType();
        PipelineConfig pipelineConfig = [self _lookupPipelineConfig:patchType useSingleCreasePatch:_useSingleCreasePatch];

        // Don't compute tess factors when not using tessellation
        if (!pipelineConfig.useTessellation) {
            continue;
        }

        [computeCommandEncoder setComputePipelineState:_computePipelines[patchType]];

        [computeCommandEncoder setBufferOffset:patch.primitiveIdBase * sizeof(int) * 3 atIndex:OSD_PATCHPARAM_BUFFER_INDEX];
        [computeCommandEncoder setBufferOffset:patch.indexBase * sizeof(unsigned) atIndex:CONTROL_INDICES_BUFFER_INDEX];

        if (pipelineConfig.useTessellation) {
            [computeCommandEncoder setBuffer:_tessFactorsBuffer offset:_tessFactorsOffsets[patchType] atIndex:PATCH_TESSFACTORS_INDEX];
            [computeCommandEncoder setBuffer:_perPatchTessFactorsBuffer offset:_perPatchTessFactorsOffsets[patchType] atIndex:OSD_PERPATCHTESSFACTORS_BUFFER_INDEX];
            [computeCommandEncoder setBuffer:_perPatchVertexBuffer offset:_perPatchVertexOffsets[patchType] atIndex:OSD_PERPATCHVERTEX_BUFFER_INDEX];
        }
        if (_usePatchIndexBuffer) {
            [computeCommandEncoder setBuffer:_patchIndexBuffers[patchType] offset:0 atIndex:OSD_PATCH_INDEX_BUFFER_INDEX];
            [computeCommandEncoder setBuffer:_drawIndirectCommandsBuffer offset:sizeof(MTLDrawPatchIndirectArguments) * patchType atIndex:OSD_DRAWINDIRECT_BUFFER_INDEX];
        }

        int numTotalControlPoints = patch.GetNumPatches() * pipelineConfig.numControlPointsPerPatchRefined;
        int numTotalControlPointThreads = std::max<int>(1, numTotalControlPoints / pipelineConfig.numControlPointsPerThreadRefined);
        int numThreadsPerThreadgroup = _threadgroupSizes[patchType];
        int numTotalThreadgroups = std::max<int>(1, (numTotalControlPointThreads+numThreadsPerThreadgroup-1) / numThreadsPerThreadgroup);

        unsigned kernelExecutionLimit = patch.GetNumPatches() * pipelineConfig.numControlPointsPerPatchToDraw;
        [computeCommandEncoder setBytes:&kernelExecutionLimit length:sizeof(kernelExecutionLimit) atIndex:OSD_KERNELLIMIT_BUFFER_INDEX];

        [computeCommandEncoder dispatchThreadgroups:MTLSizeMake(numTotalThreadgroups,1, 1)
                              threadsPerThreadgroup:MTLSizeMake(numThreadsPerThreadgroup, 1, 1)];
    }
}

-(void)_rebuildState {
    [self _rebuildTextures];
    [self _rebuildModel];
    [self _rebuildBuffers];
    [self _rebuildPipelines];
    _needsRebuild = false;
}

-(void)_rebuildTextures {
    _colorPtexture.reset();
    _displacementPtexture.reset();
    _occlusionPtexture.reset();
    _specularPtexture.reset();

    _colorPtexture = [self _createPtex:_ptexColorFilename];
    if(_ptexDisplacementFilename) {
        _displacementPtexture = [self _createPtex:_ptexDisplacementFilename];
    }
}

-(std::unique_ptr<MTLPtexMipmapTexture>)_createPtex:(NSString*) filename {

    Ptex::String ptexError;
    printf("Loading ptex : %s\n", filename.UTF8String);
#if TARGET_OS_EMBEDDED
    const auto path = [[NSBundle mainBundle] pathForResource:filename ofType:nil];
#else
    const auto path = filename;
#endif

#define USE_PTEX_CACHE 1
#define PTEX_CACHE_SIZE (512*1024*1024)

#if USE_PTEX_CACHE
    PtexCache *cache = PtexCache::create(1, PTEX_CACHE_SIZE);
    PtexTexture *ptex = cache->get(path.UTF8String, ptexError);
#else
    PtexTexture *ptex = PtexTexture::open(path.UTF8String, ptexError, true);
#endif

    if (ptex == NULL) {
        printf("Error in reading %s\n", filename.UTF8String);
        exit(1);
    }

    std::unique_ptr<MTLPtexMipmapTexture> osdPtex(MTLPtexMipmapTexture::Create(&_context, ptex));

    ptex->release();

#if USE_PTEX_CACHE
    cache->release();
#endif

    return osdPtex;
}

-(std::unique_ptr<Shape>)_shapeFromPtex:(Ptex::PtexTexture*) tex {
    const auto meta = tex->getMetaData();

    if (meta->numKeys() < 3) {
        return NULL;
    }

    float const * vp;
    int const *vi, *vc;
    int nvp, nvi, nvc;

    meta->getValue("PtexFaceVertCounts", vc, nvc);
    if (nvc == 0) {
        return NULL;
    }
    meta->getValue("PtexVertPositions", vp, nvp);
    if (nvp == 0) {
        return NULL;
    }
    meta->getValue("PtexFaceVertIndices", vi, nvi);
    if (nvi == 0) {
        return NULL;
    }

    std::unique_ptr<Shape> shape(new Shape);

    shape->scheme = kCatmark;

    shape->verts.resize(nvp);
    for (int i=0; i<nvp; ++i) {
        shape->verts[i] = vp[i];
    }

    shape->nvertsPerFace.resize(nvc);
    for (int i=0; i<nvc; ++i) {
        shape->nvertsPerFace[i] = vc[i];
    }

    shape->faceverts.resize(nvi);
    for (int i=0; i<nvi; ++i) {
        shape->faceverts[i] = vi[i];
    }

    // compute model bounding
    float min[3] = {vp[0], vp[1], vp[2]};
    float max[3] = {vp[0], vp[1], vp[2]};
    for (int i = 0; i < nvp/3; ++i) {
        for (int j = 0; j < 3; ++j) {
            float v = vp[i*3+j];
            min[j] = std::min(min[j], v);
            max[j] = std::max(max[j], v);
        }
    }

    _meshSize = 0.0f;
    for (int j = 0; j < 3; ++j) {
        _meshCenter[j] = (min[j] + max[j]) * 0.5f;
        _meshSize += (max[j]-min[j])*(max[j]-min[j]);
    }
    _meshSize = sqrt(_meshSize);

    return shape;
}

-(void)_rebuildModel {

    Ptex::String ptexError;
#if TARGET_OS_EMBEDDED
    const auto ptexColor = PtexTexture::open([[NSBundle mainBundle] pathForResource:_ptexColorFilename ofType:nil].UTF8String, ptexError);
#else
    const auto ptexColor = PtexTexture::open(_ptexColorFilename.UTF8String, ptexError);
#endif
    _shape = [self _shapeFromPtex:ptexColor];

    // create Far mesh (topology)
    Sdc::SchemeType sdctype = GetSdcType(*_shape);
    Sdc::Options sdcoptions = GetSdcOptions(*_shape);

    std::unique_ptr<Far::TopologyRefiner> refiner;
    refiner.reset(Far::TopologyRefinerFactory<Shape>::Create(*_shape, Far::TopologyRefinerFactory<Shape>::Options(sdctype, sdcoptions)));

    // save coarse topology (used for coarse mesh drawing)
    Far::TopologyLevel const & refBaseLevel = refiner->GetLevel(0);

    Osd::MeshBitset bits;
    bits.set(Osd::MeshAdaptive,             _useAdaptive);
    bits.set(Osd::MeshUseSmoothCornerPatch, _useSmoothCornerPatch);
    bits.set(Osd::MeshUseSingleCreasePatch, _useSingleCreasePatch);
    bits.set(Osd::MeshUseInfSharpPatch,     _useInfinitelySharpPatch);
    bits.set(Osd::MeshEndCapBSplineBasis,   false);
    bits.set(Osd::MeshEndCapGregoryBasis,   true);

    int level = _refinementLevel;
    _numVertexElements = 3;
    int numVaryingElements = 0;

    if(_kernelType == kCPU)
    {
        _mesh.reset(new CPUMeshType(refiner.release(),
                                    _numVertexElements,
                                    numVaryingElements,
                                    level, bits, nullptr, &_context));
    }
    else
    {
        _mesh.reset(new MTLMeshType(refiner.release(),
                                    _numVertexElements,
                                    numVaryingElements,
                                    level, bits, nullptr, &_context));
    }

    MTLRenderPipelineDescriptor* desc = [MTLRenderPipelineDescriptor new];
    [_delegate setupRenderPipelineState:desc for:self];

    const auto vertexDescriptor = desc.vertexDescriptor;
    vertexDescriptor.layouts[0].stride = sizeof(float) * _numVertexElements;
    vertexDescriptor.layouts[0].stepFunction = MTLVertexStepFunctionPerVertex;
    vertexDescriptor.layouts[0].stepRate = 1;
    vertexDescriptor.attributes[0].format = MTLVertexFormatFloat3;
    vertexDescriptor.attributes[0].offset = 0;
    vertexDescriptor.attributes[0].bufferIndex = 0;

    _numVertices = refBaseLevel.GetNumVertices();

    _vertexData.resize(refBaseLevel.GetNumVertices() * _numVertexElements);

    for(int i = 0; i < refBaseLevel.GetNumVertices(); ++i)
    {
        _vertexData[i * _numVertexElements + 0] = _shape->verts[i * 3 + 0];
        _vertexData[i * _numVertexElements + 1] = _shape->verts[i * 3 + 1];
        _vertexData[i * _numVertexElements + 2] = _shape->verts[i * 3 + 2];
    }

    _mesh->UpdateVertexBuffer(_vertexData.data(), 0, refBaseLevel.GetNumVertices());
    _mesh->Refine();
    _mesh->Synchronize();
}

-(void)_updateState {
    [self _updateCamera];
    auto pData = _frameConstantsBuffer.data();

    pData->TessLevel = static_cast<float>(1 << _tessellationLevel);

    pData->displacementConfig.mipmapBias = _mipmapBias;
    pData->displacementConfig.displacementScale = _displacementScale;

    if (_useAdaptive && _usePatchIndexBuffer)
    {
        for (auto& patch : _mesh->GetPatchTable()->GetPatchArrays())
        {
            auto patchType = patch.desc.GetType();

            MTLDrawPatchIndirectArguments* drawCommand = _drawIndirectCommandsBuffer.data();
            drawCommand[patchType].baseInstance = 0;
            drawCommand[patchType].instanceCount = 1;
            drawCommand[patchType].patchCount = 0;
            drawCommand[patchType].patchStart = 0;
        }

        _drawIndirectCommandsBuffer.markModified();
    }

    _frameConstantsBuffer.markModified();
}

-(void)_rebuildBuffers {
    auto totalPatches = 0;
    auto totalPerPatchVertexSize = 0;
    auto totalPerPatchTessFactorsSize = 0;
    auto totalTessFactorsSize = 0;

    if (_usePatchIndexBuffer)
    {
        _drawIndirectCommandsBuffer.alloc(_context.device, DISPATCHSLOTS, @"draw patch indirect commands");
    }

    if (_useAdaptive)
    {
        for (auto& patch : _mesh->GetPatchTable()->GetPatchArrays())
        {
            auto patchType = patch.desc.GetType();
            PipelineConfig pipelineConfig = [self _lookupPipelineConfig:patchType useSingleCreasePatch:_useSingleCreasePatch];

            if (pipelineConfig.useTessellation) {
                float elementFloats = 3;
                if (patchType == Far::PatchDescriptor::GREGORY || patchType == Far::PatchDescriptor::GREGORY_BOUNDARY) { // XXXdyu-mtl
                    elementFloats *= 5;
                }
                if (pipelineConfig.useSingleCreasePatch) {
                    elementFloats += 6;
                }
                if (_usePatchIndexBuffer)
                {
                    _patchIndexBuffers[patchType].alloc(_context.device, patch.GetNumPatches(), @"patch indices", MTLResourceStorageModePrivate);
                }
                _perPatchTessFactorsOffsets[patchType] = totalPerPatchTessFactorsSize;
                _perPatchVertexOffsets[patchType] = totalPerPatchVertexSize;
                _tessFactorsOffsets[patchType] = totalTessFactorsSize;
                totalPerPatchTessFactorsSize += 2 * 4 * sizeof(float) * patch.GetNumPatches();
                totalPerPatchVertexSize += elementFloats * sizeof(float) * patch.GetNumPatches() * pipelineConfig.numControlPointsPerPatchToDraw;
                totalTessFactorsSize += patch.GetNumPatches() * (pipelineConfig.useTriangleTessellation
                                                                 ? sizeof(MTLTriangleTessellationFactorsHalf)
                                                                 : sizeof(MTLQuadTessellationFactorsHalf));
            }

            totalPatches += patch.GetNumPatches();
        }

        _tessFactorsBuffer.alloc(_context.device, totalTessFactorsSize, @"tessellation factors buffer", MTLResourceStorageModePrivate);
        _perPatchVertexBuffer.alloc(_context.device, totalPerPatchVertexSize, @"per patch data", MTLResourceStorageModePrivate);
        _perPatchTessFactorsBuffer.alloc(_context.device, totalPerPatchTessFactorsSize, @"per patch tess factors", MTLResourceStorageModePrivate);
    }
}

-(void)_rebuildPipelines {
    for (int i = 0; i < DISPATCHSLOTS; ++i) {
        _computePipelines[i] = nil;
        _renderPipelines[i] = nil;
    }

    Osd::MTLPatchShaderSource shaderSource;

    for (auto& patch : _mesh->GetPatchTable()->GetPatchArrays())
    {
        auto patchType = patch.desc.GetType();
        PipelineConfig pipelineConfig = [self _lookupPipelineConfig:patchType useSingleCreasePatch:_useSingleCreasePatch];

        auto compileOptions = [[MTLCompileOptions alloc] init];
        compileOptions.fastMathEnabled = YES;

        auto preprocessor = [[NSMutableDictionary alloc] init];

        std::stringstream shaderBuilder;
#define DEFINE(x,y) preprocessor[@(#x)] = @(y)

#if TARGET_OS_EMBEDDED
        shaderBuilder << "#define OSD_UV_CORRECTION if(t > 0.5){ ti += 0.01f; } else { ti += 0.01f; }\n";
#endif

        //Need to define the input vertex struct so that it's available everywhere.
        {
            shaderBuilder << R"(
                                #include <metal_stdlib>
                                using namespace metal;

                                struct OsdInputVertexType {
                                    metal::packed_float3 position;
                                };
            )";
        }

        shaderBuilder << shaderSource.GetHullShaderSource(patchType);
        shaderBuilder << MTLPtexMipmapTexture::GetShaderSource();
        shaderBuilder << _osdShaderSource.UTF8String;

        const auto str = shaderBuilder.str();

        DEFINE(CONFIG_BUFFER_INDEX,CONFIG_BUFFER_INDEX);
        DEFINE(VERTEX_BUFFER_INDEX,VERTEX_BUFFER_INDEX);
        DEFINE(PATCH_INDICES_BUFFER_INDEX,PATCH_INDICES_BUFFER_INDEX);
        DEFINE(CONTROL_INDICES_BUFFER_INDEX,CONTROL_INDICES_BUFFER_INDEX);
        DEFINE(OSD_PATCHPARAM_BUFFER_INDEX,OSD_PATCHPARAM_BUFFER_INDEX);
        DEFINE(OSD_PERPATCHVERTEX_BUFFER_INDEX,OSD_PERPATCHVERTEX_BUFFER_INDEX);
        DEFINE(OSD_PERPATCHVERTEXBEZIER_BUFFER_INDEX,OSD_PERPATCHVERTEXBEZIER_BUFFER_INDEX);
        DEFINE(OSD_PERPATCHVERTEXGREGORY_BUFFER_INDEX,OSD_PERPATCHVERTEXGREGORY_BUFFER_INDEX);
        DEFINE(OSD_PERPATCHTESSFACTORS_BUFFER_INDEX,OSD_PERPATCHTESSFACTORS_BUFFER_INDEX);
        DEFINE(OSD_VALENCE_BUFFER_INDEX,OSD_VALENCE_BUFFER_INDEX);
        DEFINE(OSD_QUADOFFSET_BUFFER_INDEX,OSD_QUADOFFSET_BUFFER_INDEX);
        DEFINE(FRAME_CONST_BUFFER_INDEX,FRAME_CONST_BUFFER_INDEX);
        DEFINE(INDICES_BUFFER_INDEX,INDICES_BUFFER_INDEX);
        DEFINE(PATCH_TESSFACTORS_INDEX,PATCH_TESSFACTORS_INDEX);
        DEFINE(QUAD_TESSFACTORS_INDEX,QUAD_TESSFACTORS_INDEX);
        DEFINE(TRIANGLE_TESSFACTORS_INDEX,TRIANGLE_TESSFACTORS_INDEX);
        DEFINE(OSD_PATCH_INDEX_BUFFER_INDEX,OSD_PATCH_INDEX_BUFFER_INDEX);
        DEFINE(OSD_DRAWINDIRECT_BUFFER_INDEX,OSD_DRAWINDIRECT_BUFFER_INDEX);
        DEFINE(DISPLACEMENT_TEXTURE_INDEX,DISPLACEMENT_TEXTURE_INDEX);
        DEFINE(DISPLACEMENT_BUFFER_INDEX,DISPLACEMENT_BUFFER_INDEX);
        DEFINE(IMAGE_TEXTURE_INDEX,IMAGE_TEXTURE_INDEX);
        DEFINE(IMAGE_BUFFER_INDEX,IMAGE_BUFFER_INDEX);
        DEFINE(OCCLUSION_TEXTURE_INDEX,OCCLUSION_TEXTURE_INDEX);
        DEFINE(OCCLUSION_BUFFER_INDEX,OCCLUSION_BUFFER_INDEX);
        DEFINE(SPECULAR_TEXTURE_INDEX,SPECULAR_TEXTURE_INDEX);
        DEFINE(SPECULAR_BUFFER_INDEX,SPECULAR_BUFFER_INDEX);
        DEFINE(OSD_KERNELLIMIT_BUFFER_INDEX,OSD_KERNELLIMIT_BUFFER_INDEX);

        DEFINE(COLOR_NORMAL, _colorMode == kColorModeNormal);
        DEFINE(COLOR_PATCHTYPE, _colorMode == kColorModePatchType);
        DEFINE(COLOR_PATCHCOORD, _colorMode == kColorModePatchCoord);
        DEFINE(COLOR_PTEX_BIQUADRATIC, _colorMode == kColorModePtexBiQuadratic);
        DEFINE(COLOR_PTEX_BILINEAR, _colorMode == kColorModePtexBilinear);
        DEFINE(COLOR_PTEX_NEAREST, _colorMode == kColorModePtexNearest);
        DEFINE(COLOR_PTEX_HW_BILINEAR, _colorMode == kColorModePtexHWBilinear);

        DEFINE(NORMAL_SCREENSPACE, _normalMode == kNormalModeScreenspace);
        DEFINE(NORMAL_HW_SCREENSPACE, _normalMode == kNormalModeHWScreenspace);
        DEFINE(NORMAL_BIQUADRATIC_WG, _normalMode == kNormalModeBiQuadraticWG);
        DEFINE(NORMAL_BIQUADRATIC, _normalMode == kNormalModeBiQuadratic);

        DEFINE(DISPLACEMENT_BILINEAR, _displacementMode == kDisplacementModeBilinear);
        DEFINE(DISPLACEMENT_HW_BILINEAR, _displacementMode == kDisplacementModeHWBilinear);
        DEFINE(DISPLACEMENT_BIQUADRATIC, _displacementMode == kDisplacementModeBiQuadratic);

        DEFINE(OSD_COMPUTE_NORMAL_DERIVATIVES, _normalMode == kNormalModeBiQuadraticWG);

        if (patchType == Far::PatchDescriptor::QUADS) {
            DEFINE(OSD_PATCH_QUADS, 1);
        } else if (patchType == Far::PatchDescriptor::TRIANGLES) {
            DEFINE(OSD_PATCH_TRIANGLES, 1);
        }

        DEFINE(CONTROL_POINTS_PER_PATCH, pipelineConfig.numControlPointsPerPatchRefined);
        DEFINE(VERTEX_CONTROL_POINTS_PER_PATCH, pipelineConfig.numControlPointsPerPatchToDraw);
        DEFINE(CONTROL_POINTS_PER_THREAD, pipelineConfig.numControlPointsPerThreadRefined);
        DEFINE(VERTEX_CONTROL_POINTS_PER_THREAD, pipelineConfig.numControlPointsPerThreadToDraw);
        DEFINE(THREADS_PER_PATCH, pipelineConfig.numThreadsPerPatch);

        DEFINE(OSD_PATCH_ENABLE_SINGLE_CREASE, pipelineConfig.useSingleCreasePatch);

        auto partitionMode = _useScreenspaceTessellation && _useFractionalTessellation
                                ? MTLTessellationPartitionModeFractionalOdd
                                : MTLTessellationPartitionModeInteger;
        if (partitionMode == MTLTessellationPartitionModeFractionalOdd) {
            DEFINE(OSD_FRACTIONAL_ODD_SPACING, 1);
        } else if (partitionMode == MTLTessellationPartitionModeFractionalEven) {
            DEFINE(OSD_FRACTIONAL_EVEN_SPACING, 1);
        }

#if TARGET_OS_EMBEDDED
        DEFINE(OSD_MAX_TESS_LEVEL, 16);
#else
        DEFINE(OSD_MAX_TESS_LEVEL, 64);
#endif
        DEFINE(USE_STAGE_IN, _useStageIn);
        DEFINE(USE_PTVS_FACTORS, !_useScreenspaceTessellation);
        DEFINE(USE_PTVS_SHARPNESS, 1);

        DEFINE(OSD_MAX_VALENCE, _mesh->GetMaxValence());
        DEFINE(OSD_NUM_ELEMENTS, _numVertexElements);
        DEFINE(OSD_ENABLE_BACKPATCH_CULL, _usePatchClipCulling);
        DEFINE(OSD_USE_PATCH_INDEX_BUFFER, _usePatchIndexBuffer);
        DEFINE(SEAMLESS_MIPMAP, _useSeamlessMipmap);
        DEFINE(USE_PTEX_OCCLUSION, _occlusionPtexture != nullptr && _displayOcclusion);
        DEFINE(USE_PTEX_SPECULAR, _specularPtexture != nullptr && _displaySpecular);
        DEFINE(OSD_ENABLE_SCREENSPACE_TESSELLATION, _useScreenspaceTessellation);
        DEFINE(OSD_ENABLE_PATCH_CULL, _usePatchClipCulling && _useAdaptive);

        auto & threadsPerThreadgroup = _threadgroupSizes[patchType];
        threadsPerThreadgroup = 32; //Initial guess of 32

        DEFINE(THREADS_PER_THREADGROUP, threadsPerThreadgroup);

        compileOptions.preprocessorMacros = preprocessor;

        NSError* err = nil;
        auto librarySource = [NSString stringWithUTF8String:str.data()];
        auto library = [_context.device newLibraryWithSource:librarySource options:compileOptions error:&err];
        if(!library && err) {
            NSLog(@"%s", [err localizedDescription].UTF8String);
        }
        assert(library);

        auto vertexFunction = [library newFunctionWithName:@"vertex_main"];
        auto fragmentFunction = [library newFunctionWithName:@"fragment_main"];
        if (vertexFunction && fragmentFunction)
        {

            MTLRenderPipelineDescriptor* pipelineDesc = [[MTLRenderPipelineDescriptor alloc] init];
            pipelineDesc.tessellationFactorFormat = MTLTessellationFactorFormatHalf;
            pipelineDesc.tessellationPartitionMode = partitionMode;
            pipelineDesc.tessellationFactorScaleEnabled = false;
            pipelineDesc.tessellationFactorStepFunction = MTLTessellationFactorStepFunctionPerPatch;

            if (pipelineConfig.drawIndexed && _useStageIn) {
                pipelineDesc.tessellationControlPointIndexType = MTLTessellationControlPointIndexTypeUInt32;
            }

            [_delegate setupRenderPipelineState:pipelineDesc for:self];

            pipelineDesc.fragmentFunction = fragmentFunction;
            pipelineDesc.vertexFunction = vertexFunction;

            if(_useStageIn)
            {
                auto vertexDesc = pipelineDesc.vertexDescriptor;
                [vertexDesc reset];

                if (_useAdaptive)
                {
                    vertexDesc.layouts[OSD_PATCHPARAM_BUFFER_INDEX].stepFunction = MTLVertexStepFunctionPerPatch;
                    vertexDesc.layouts[OSD_PATCHPARAM_BUFFER_INDEX].stepRate = 1;
                    vertexDesc.layouts[OSD_PATCHPARAM_BUFFER_INDEX].stride = sizeof(int) * 3;

                    // PatchInput :: int3 patchParam [[attribute(10)]];
                    vertexDesc.attributes[10].bufferIndex = OSD_PATCHPARAM_BUFFER_INDEX;
                    vertexDesc.attributes[10].format = MTLVertexFormatInt3;
                    vertexDesc.attributes[10].offset = 0;
                }

                switch(patchType)
                {
                    case Far::PatchDescriptor::LOOP:
                    case Far::PatchDescriptor::REGULAR:
                    case Far::PatchDescriptor::GREGORY_BASIS:
                    case Far::PatchDescriptor::GREGORY_TRIANGLE:
                        if (pipelineConfig.drawIndexed) {
                            vertexDesc.layouts[VERTEX_BUFFER_INDEX].stepFunction = MTLVertexStepFunctionPerPatchControlPoint;
                            vertexDesc.layouts[VERTEX_BUFFER_INDEX].stepRate = 1;
                            vertexDesc.layouts[VERTEX_BUFFER_INDEX].stride = sizeof(float) * 3;

                            // ControlPoint :: float3 position [[attribute(0)]];
                            vertexDesc.attributes[0].bufferIndex = VERTEX_BUFFER_INDEX;
                            vertexDesc.attributes[0].format = MTLVertexFormatFloat3;
                            vertexDesc.attributes[0].offset = 0;
                        } else {
                            vertexDesc.layouts[OSD_PERPATCHVERTEX_BUFFER_INDEX].stepFunction = MTLVertexStepFunctionPerPatchControlPoint;
                            vertexDesc.layouts[OSD_PERPATCHVERTEX_BUFFER_INDEX].stepRate = 1;
                            vertexDesc.layouts[OSD_PERPATCHVERTEX_BUFFER_INDEX].stride = sizeof(float) * 3;

                            // ControlPoint :: float3 P [[attribute(0)]];
                            // OsdPerPatchVertexBezier :: packed_float3 P
                            vertexDesc.attributes[0].bufferIndex = OSD_PERPATCHVERTEX_BUFFER_INDEX;
                            vertexDesc.attributes[0].format = MTLVertexFormatFloat3;
                            vertexDesc.attributes[0].offset = 0;
                        }

                        if (pipelineConfig.useSingleCreasePatch)
                        {
                            vertexDesc.layouts[OSD_PERPATCHVERTEX_BUFFER_INDEX].stride += sizeof(float) * 3 * 2;

                            // ControlPoint :: float3 P1 [[attribute(1)]];
                            // OsdPerPatchVertexBezier :: packed_float3 P1
                            vertexDesc.attributes[1].bufferIndex = OSD_PERPATCHVERTEX_BUFFER_INDEX;
                            vertexDesc.attributes[1].format = MTLVertexFormatFloat3;
                            vertexDesc.attributes[1].offset = sizeof(float) * 3;

                            // ControlPoint :: float3 P2 [[attribute(2)]];
                            // OsdPerPatchVertexBezier :: packed_float3 P2
                            vertexDesc.attributes[2].bufferIndex = OSD_PERPATCHVERTEX_BUFFER_INDEX;
                            vertexDesc.attributes[2].format = MTLVertexFormatFloat3;
                            vertexDesc.attributes[2].offset = sizeof(float) * 6;

                            // USE_PTVS_SHARPNESS is true and so OsdPerPatchVertexBezier :: float2 vSegments is not used
                        }

                        if(_useScreenspaceTessellation)
                        {
                            vertexDesc.layouts[OSD_PERPATCHTESSFACTORS_BUFFER_INDEX].stepFunction = MTLVertexStepFunctionPerPatch;
                            vertexDesc.layouts[OSD_PERPATCHTESSFACTORS_BUFFER_INDEX].stepRate = 1;
                            vertexDesc.layouts[OSD_PERPATCHTESSFACTORS_BUFFER_INDEX].stride = sizeof(float) * 4 * 2;

                            // PatchInput :: float4 tessOuterLo [[attribute(5)]];
                            // OsdPerPatchTessFactors :: float4 tessOuterLo;
                            vertexDesc.attributes[5].bufferIndex = OSD_PERPATCHTESSFACTORS_BUFFER_INDEX;
                            vertexDesc.attributes[5].format = MTLVertexFormatFloat4;
                            vertexDesc.attributes[5].offset = 0;

                            // PatchInput :: float4 tessOuterHi [[attribute(6)]];
                            // OsdPerPatchTessFactors :: float4 tessOuterHi;
                            vertexDesc.attributes[6].bufferIndex = OSD_PERPATCHTESSFACTORS_BUFFER_INDEX;
                            vertexDesc.attributes[6].format = MTLVertexFormatFloat4;
                            vertexDesc.attributes[6].offset = sizeof(float) * 4;
                        }
                    break;
                    case Far::PatchDescriptor::GREGORY:
                    case Far::PatchDescriptor::GREGORY_BOUNDARY:

                        vertexDesc.layouts[OSD_PERPATCHVERTEX_BUFFER_INDEX].stepFunction = MTLVertexStepFunctionPerPatchControlPoint;
                        vertexDesc.layouts[OSD_PERPATCHVERTEX_BUFFER_INDEX].stepRate = 1;
                        vertexDesc.layouts[OSD_PERPATCHVERTEX_BUFFER_INDEX].stride = sizeof(float) * 3 * 5;

                        // ControlPoint :: float3 P [[attribute(0)]];
                        // ControlPoint :: float3 Ep [[attribute(1)]];
                        // ControlPoint :: float3 Em [[attribute(2)]];
                        // ControlPoint :: float3 Fp [[attribute(3)]];
                        // ControlPoint :: float3 Fm [[attribute(4)]];
                        for (int i = 0; i < 5; ++i)
                        {
                            vertexDesc.attributes[i].bufferIndex = OSD_PERPATCHVERTEX_BUFFER_INDEX;
                            vertexDesc.attributes[i].format = MTLVertexFormatFloat3;
                            vertexDesc.attributes[i].offset = i * sizeof(float) * 3;
                        }

                        if(_useScreenspaceTessellation)
                        {
                            vertexDesc.layouts[OSD_PERPATCHTESSFACTORS_BUFFER_INDEX].stepFunction = MTLVertexStepFunctionPerPatch;
                            vertexDesc.layouts[OSD_PERPATCHTESSFACTORS_BUFFER_INDEX].stepRate = 1;
                            vertexDesc.layouts[OSD_PERPATCHTESSFACTORS_BUFFER_INDEX].stride = sizeof(float) * 4 * 2;

                            // PatchInput :: float4 tessOuterLo [[attribute(5)]];
                            // OsdPerPatchTessFactors :: float4 tessOuterLo;
                            vertexDesc.attributes[5].bufferIndex = OSD_PERPATCHTESSFACTORS_BUFFER_INDEX;
                            vertexDesc.attributes[5].format = MTLVertexFormatFloat4;
                            vertexDesc.attributes[5].offset = 0;

                            // PatchInput :: float4 tessOuterHi [[attribute(6)]];
                            // OsdPerPatchTessFactors :: float4 tessOuterHi;
                            vertexDesc.attributes[6].bufferIndex = OSD_PERPATCHTESSFACTORS_BUFFER_INDEX;
                            vertexDesc.attributes[6].format = MTLVertexFormatFloat4;
                            vertexDesc.attributes[6].offset = sizeof(float) * 4;
                        }
                    break;
                    case Far::PatchDescriptor::QUADS:
                        //Quads cannot use stage in, due to the need for re-indexing.
                        pipelineDesc.vertexDescriptor = nil;
                    case Far::PatchDescriptor::TRIANGLES:
                        [vertexDesc reset];
                        break;
                }

            }

            _renderPipelines[patchType] = [_context.device newRenderPipelineStateWithDescriptor:pipelineDesc error:&err];
            if (!_renderPipelines[patchType] && err)
            {
                NSLog(@"%s", [[err localizedDescription] UTF8String]);
            }
        }

        auto computeFunction = [library newFunctionWithName:@"compute_main"];
        if(computeFunction)
        {
            MTLComputePipelineDescriptor* computeDesc = [[MTLComputePipelineDescriptor alloc] init];
#if MTL_TARGET_IPHONE
            computeDesc.threadGroupSizeIsMultipleOfThreadExecutionWidth = true;
#else
            computeDesc.threadGroupSizeIsMultipleOfThreadExecutionWidth = false;
#endif
            computeDesc.computeFunction = computeFunction;

            NSError* err;

            _computePipelines[patchType] = [_context.device newComputePipelineStateWithDescriptor:computeDesc options:MTLPipelineOptionNone reflection:nil error:&err];
            if (err && _computePipelines[patchType] == nil)
            {
                NSLog(@"first compute compile: %s", [[err description] UTF8String]);
            }

            if (_computePipelines[patchType].threadExecutionWidth != threadsPerThreadgroup)
            {
                DEFINE(THREADS_PER_THREADGROUP, _computePipelines[patchType].threadExecutionWidth);

                compileOptions.preprocessorMacros = preprocessor;

                library = [_context.device newLibraryWithSource:librarySource options:compileOptions error:nil];
                assert(library);

                computeDesc.threadGroupSizeIsMultipleOfThreadExecutionWidth = true;
                computeDesc.computeFunction = [library newFunctionWithName:@"compute_main"];

                threadsPerThreadgroup = _computePipelines[patchType].threadExecutionWidth;
                _computePipelines[patchType] = [_context.device newComputePipelineStateWithDescriptor:computeDesc options:MTLPipelineOptionNone reflection:nil error:&err];
                if (err && _computePipelines[patchType] == nil)
                {
                    NSLog(@"second compute compile: %s", [[err description] UTF8String]);
                }

                if (_computePipelines[patchType].threadExecutionWidth != threadsPerThreadgroup)
                {
                    DEFINE(THREADS_PER_THREADGROUP, threadsPerThreadgroup);
                    DEFINE(NEEDS_BARRIER, 1);

                    compileOptions.preprocessorMacros = preprocessor;

                    library = [_context.device newLibraryWithSource:librarySource options:compileOptions error:nil];
                    assert(library);

                    computeDesc.threadGroupSizeIsMultipleOfThreadExecutionWidth = false;
                    computeDesc.computeFunction = [library newFunctionWithName:@"compute_main"];

                    threadsPerThreadgroup = _computePipelines[patchType].threadExecutionWidth;
                    _computePipelines[patchType] = [_context.device newComputePipelineStateWithDescriptor:computeDesc options:MTLPipelineOptionNone reflection:nil error:&err];
                    if (err && _computePipelines[patchType] == nil)
                    {
                        NSLog(@"third compute compile: %s", [[err description] UTF8String]);
                    }
                }
            }
        }
    }

    MTLDepthStencilDescriptor* depthStencilDesc = [[MTLDepthStencilDescriptor alloc] init];
    depthStencilDesc.depthCompareFunction = MTLCompareFunctionLess;

    [_delegate setupDepthStencilState:depthStencilDesc for:self];

    depthStencilDesc.depthWriteEnabled = YES;
    _readWriteDepthStencilState = [_context.device newDepthStencilStateWithDescriptor:depthStencilDesc];

    depthStencilDesc.depthWriteEnabled = NO;
    _readOnlyDepthStencilState = [_context.device newDepthStencilStateWithDescriptor:depthStencilDesc];
}

-(void)_updateCamera {
    auto pData = _frameConstantsBuffer.data();

    identity(pData->ModelViewMatrix);
    translate(pData->ModelViewMatrix, 0, 0, -_cameraData.dollyDistance);
    rotate(pData->ModelViewMatrix, _cameraData.rotationY, 1, 0, 0);
    rotate(pData->ModelViewMatrix, _cameraData.rotationX, 0, 1, 0);
    if (!_yup) {
        rotate(pData->ModelViewMatrix, -90, 1, 0, 0);
    }
    translate(pData->ModelViewMatrix, -_meshCenter[0], -_meshCenter[1], -_meshCenter[2]);
    inverseMatrix(pData->ModelViewInverseMatrix, pData->ModelViewMatrix);

    identity(pData->ProjectionMatrix);
    perspective(pData->ProjectionMatrix, 45.0, _cameraData.aspectRatio, _meshSize*0.001f, _meshSize+_cameraData.dollyDistance);
    multMatrix(pData->ModelViewProjectionMatrix, pData->ModelViewMatrix, pData->ProjectionMatrix);
}

-(void)_initializeBuffers {
    _frameConstantsBuffer.alloc(_context.device, 1, @"frame constants");
    _lightsBuffer.alloc(_context.device, 2, @"lights");
}

-(void)_initializeCamera {
    _cameraData.rotationY = 0;
    _cameraData.rotationX = 0;
    _cameraData.dollyDistance = 5;
    _cameraData.aspectRatio = 1;
}

-(void)_initializeLights {
    _lightsBuffer[0] = {
        simd::normalize(simd::float4{ 0.5,  0.2f, 1.0f, 0.0f }),
        { 0.1f, 0.1f, 0.1f, 1.0f },
        { 0.7f, 0.7f, 0.7f, 1.0f },
        { 0.8f, 0.8f, 0.8f, 1.0f },
    };

    _lightsBuffer[1] = {
        simd::normalize(simd::float4{ -0.8f, 0.4f, -1.0f, 0.0f }),
        {  0.0f, 0.0f,  0.0f, 1.0f },
        {  0.5f, 0.5f,  0.5f, 1.0f },
        {  0.8f, 0.8f,  0.8f, 1.0f }
    };

    _lightsBuffer.markModified();
}

//Setters for triggering _needsRebuild on property change

-(void)setKernelType:(KernelType)kernelType {
    _needsRebuild |= kernelType != _kernelType;
    _kernelType = kernelType;
}

-(void)setRefinementLevel:(unsigned int)refinementLevel {
    _needsRebuild |= refinementLevel != _refinementLevel;
    _refinementLevel = refinementLevel;
}

-(void)setUseSeamlessMipmap:(bool)useSeamlessMipmap {
    _needsRebuild |= useSeamlessMipmap != _useSeamlessMipmap;
    _useSeamlessMipmap = useSeamlessMipmap;
}

-(void)setUseSmoothCornerPatch:(bool)useSmoothCornerPatch {
    _needsRebuild |= useSmoothCornerPatch != _useSmoothCornerPatch;
    _useSmoothCornerPatch = useSmoothCornerPatch;
}

-(void)setUseSingleCreasePatch:(bool)useSingleCreasePatch {
    _needsRebuild |= useSingleCreasePatch != _useSingleCreasePatch;
    _useSingleCreasePatch = useSingleCreasePatch;
}

-(void)setUsePatchClipCulling:(bool)usePatchClipCulling {
    _needsRebuild |= usePatchClipCulling != _usePatchClipCulling;
    _usePatchClipCulling = usePatchClipCulling;
}

-(void)setUsePatchIndexBuffer:(bool)usePatchIndexBuffer {
    _needsRebuild |= usePatchIndexBuffer != _usePatchIndexBuffer;
    _usePatchIndexBuffer = usePatchIndexBuffer;
}

-(void)setUsePatchBackfaceCulling:(bool)usePatchBackfaceCulling {
    _needsRebuild |= usePatchBackfaceCulling != _usePatchBackfaceCulling;
    _usePatchBackfaceCulling = usePatchBackfaceCulling;
}

-(void)setUseScreenspaceTessellation:(bool)useScreenspaceTessellation {
    _needsRebuild |= useScreenspaceTessellation != _useScreenspaceTessellation;
    _useScreenspaceTessellation = useScreenspaceTessellation;
}

-(void)setColorMode:(ColorMode)colorMode {
    _needsRebuild |= colorMode != _colorMode;
    _colorMode = colorMode;
}

-(void)setNormalMode:(NormalMode)normalMode {
    _needsRebuild |= normalMode != _normalMode;
    _normalMode = normalMode;
}

-(void)setDisplacementMode:(DisplacementMode)displacementMode {
    _needsRebuild |= displacementMode != _displacementMode;
    _displacementMode = displacementMode;
}

-(void)setUseAdaptive:(bool)useAdaptive {
    _needsRebuild |= useAdaptive != _useAdaptive;
    _useAdaptive = useAdaptive;
}

-(void)setUseInfinitelySharpPatch:(bool)useInfinitelySharpPatch {
    _needsRebuild |= useInfinitelySharpPatch != _useInfinitelySharpPatch;
    _useInfinitelySharpPatch = useInfinitelySharpPatch;
}

-(void)setUseFractionalTessellation:(bool)useFractionalTessellation {
    _needsRebuild |= useFractionalTessellation != _useFractionalTessellation;
    _useFractionalTessellation = useFractionalTessellation;
}

-(void)setDisplayOcclusion:(bool)displayOcclusion {
    _needsRebuild |= displayOcclusion != _displayOcclusion;
    _displayOcclusion = displayOcclusion;
}

-(void)setDisplaySpecular:(bool)displaySpecular {
    _needsRebuild |= displaySpecular != _displaySpecular;
    _displaySpecular = displaySpecular;
}

@end
