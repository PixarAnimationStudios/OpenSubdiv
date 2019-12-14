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

#import "mtlViewer.h"

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
#import <opensubdiv/osd/mtlLegacyGregoryPatchTable.h>
#import <opensubdiv/osd/mtlVertexBuffer.h>
#import <opensubdiv/osd/mtlMesh.h>
#import <opensubdiv/osd/mtlPatchTable.h>
#import <opensubdiv/osd/mtlComputeEvaluator.h>
#import <opensubdiv/osd/mtlPatchShaderSource.h>

#import "../../regression/common/far_utils.h"
#import "../../regression/common/arg_utils.h"
#import "../common/mtlUtils.h"
#import "../common/mtlControlMeshDisplay.h"
#import "../common/simple_math.h"
#import "../common/viewerArgsUtils.h"
#import "init_shapes.h"

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

#define OSD_FVAR_DATA_BUFFER_INDEX 16
#define OSD_FVAR_INDICES_BUFFER_INDEX 17
#define OSD_FVAR_PATCHPARAM_BUFFER_INDEX 18
#define OSD_FVAR_PATCH_ARRAYS_BUFFER_INDEX 19

#define FRAME_CONST_BUFFER_INDEX 11
#define INDICES_BUFFER_INDEX 2

#define FVAR_SINGLE_BUFFER 1

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

using CPUMeshType = Osd::Mesh<
    Osd::CPUMTLVertexBuffer,
    Far::StencilTable,
    Osd::CpuEvaluator,
    Osd::MTLPatchTable,
    Osd::MTLContext>;

using mtlMeshType = Osd::Mesh<
    Osd::CPUMTLVertexBuffer,
    Osd::MTLStencilTable,
    Osd::MTLComputeEvaluator,
    Osd::MTLPatchTable,
    Osd::MTLContext>;

using MTLMeshInterface = Osd::MTLMeshInterface;

struct alignas(16) PerFrameConstants
{
    float ModelViewMatrix[16];
    float ProjectionMatrix[16];
    float ModelViewProjectionMatrix[16];
    float ModelViewInverseMatrix[16];
    float TessLevel;
};

struct alignas(16) Light {
    simd::float4 position;
    simd::float4 ambient;
    simd::float4 diffuse;
    simd::float4 specular;
};

static const char* shaderSource =
#include "mtlViewer.gen.h"
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
    id<MTLRenderPipelineState> _controlLineRenderPipelines[DISPATCHSLOTS];

    id<MTLRenderPipelineState> _renderControlEdgesPipeline;
    id<MTLDepthStencilState> _readWriteDepthStencilState;
    id<MTLDepthStencilState> _readOnlyDepthStencilState;

    id<MTLBuffer> _faceVaryingDataBuffer;
    id<MTLBuffer> _faceVaryingIndicesBuffer;
    id<MTLBuffer> _faceVaryingPatchParamBuffer;


    Camera _cameraData;
    Osd::MTLContext _context;


    int _numVertexElements;
    int _numVaryingElements;
    int _numFaceVaryingElements;
    int _numVertices;
    int _frameCount;
    int _animationFrames;
    std::vector<float> _vertexData, _animatedVertices;
    std::unique_ptr<MTLMeshInterface> _mesh;
    std::unique_ptr<MTLControlMeshDisplay> _controlMesh;
    std::unique_ptr<Osd::MTLLegacyGregoryPatchTable> _legacyGregoryPatchTable;
    bool _legacyGregoryEnabled;
    std::unique_ptr<Shape> _shape;

    bool _needsRebuild;
    NSString* _osdShaderSource;
    simd::float3 _meshCenter;
    float _meshSize;
    NSMutableArray<NSString*>* _loadedModels;
    int _patchCounts[DISPATCHSLOTS];
}

-(Camera*)camera {
    return &_cameraData;
}

-(int *)patchCounts {
    return _patchCounts;
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

    // Parse remaining args
    const std::vector<const char *> &rargs = args.GetRemainingArgs();
    for (size_t i = 0; i < rargs.size(); ++i) {

        if (!strcmp(rargs[i], "-lg")) {
            self.legacyGregoryEnabled = true;
        } else {
            args.PrintUnrecognizedArgWarning(rargs[i]);
        }
    }

    self.yup = args.GetYUp();
    self.useAdaptive = args.GetAdaptive();
    self.refinementLevel = args.GetLevel();

    ViewerArgsUtils::PopulateShapes(args, &g_defaultShapes);
}

-(instancetype)initWithDelegate:(id<OSDRendererDelegate>)delegate {
    self = [super init];
    if (self) {
        self.useSmoothCornerPatch = true;
        self.useSingleCreasePatch = true;
        self.useInfinitelySharpPatch = true;
        self.useStageIn = true;
        self.endCapMode = kEndCapGregoryBasis;
        self.useScreenspaceTessellation = false;
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
        self.shadingMode = kShadingPatchType;
        self.displayStyle = kDisplayStyleWireOnShaded;
        self.legacyGregoryEnabled = false;

        [self _processArgs];

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
        [self _initializeModels];
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

            int numElements = _numVertexElements + _numVaryingElements;

            float r = sin(_animationFrames*0.01f) * _animateVertices;
            for (int i = 0; i < _numVertices; ++i) {
                float ct = cos(p[2] * r);
                float st = sin(p[2] * r);
                n[0] = p[0]*ct + p[1]*st;
                n[1] = -p[0]*st + p[1]*ct;
                n[2] = p[2];

                for (int j = 0; j < _numVaryingElements; ++j) {
                    n[3 + j] = p[3 + j];
                }

                p += numElements;
                n += numElements;
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

    if (_numFaceVaryingElements > 0) {
#if FVAR_SINGLE_BUFFER
        int faceVaryingDataBufferOffset = _useAdaptive ? 0 : _shape->uvs.size() * sizeof(float);
        [renderCommandEncoder setVertexBuffer:_faceVaryingDataBuffer offset:faceVaryingDataBufferOffset atIndex:OSD_FVAR_DATA_BUFFER_INDEX];
#else
        [renderCommandEncoder setVertexBuffer:_faceVaryingDataBuffer offset:0 atIndex:OSD_FVAR_DATA_BUFFER_INDEX];
#endif
        [renderCommandEncoder setVertexBuffer:_faceVaryingIndicesBuffer offset:0 atIndex:OSD_FVAR_INDICES_BUFFER_INDEX];

        auto fvarPatchArrays = _mesh->GetPatchTable()->GetFVarPatchArrays();
        [renderCommandEncoder setVertexBytes:fvarPatchArrays.data()
                                      length:fvarPatchArrays.size() *
                                             sizeof(fvarPatchArrays[0])
                                     atIndex:OSD_FVAR_PATCH_ARRAYS_BUFFER_INDEX];
    }

    if (_useAdaptive)
    {
        [renderCommandEncoder setVertexBuffer:_perPatchTessFactorsBuffer offset:0 atIndex:OSD_PERPATCHTESSFACTORS_BUFFER_INDEX];
        [renderCommandEncoder setVertexBuffer:_perPatchVertexBuffer offset:0 atIndex:OSD_PERPATCHVERTEX_BUFFER_INDEX];
        [renderCommandEncoder setVertexBuffer:_mesh->GetPatchTable()->GetPatchParamBuffer() offset:0 atIndex:OSD_PATCHPARAM_BUFFER_INDEX];
        if (_numFaceVaryingElements > 0) {
            [renderCommandEncoder setVertexBuffer:_faceVaryingPatchParamBuffer
                                           offset:0
                                          atIndex:OSD_FVAR_PATCHPARAM_BUFFER_INDEX];
        }
    }

    if(_displayStyle == kDisplayStyleWire)
        [renderCommandEncoder setTriangleFillMode:MTLTriangleFillModeLines];
    else
        [renderCommandEncoder setTriangleFillMode:MTLTriangleFillModeFill];

    std::fill_n(_patchCounts, DISPATCHSLOTS, 0);

    for (auto& patch : _mesh->GetPatchTable()->GetPatchArrays())
    {
        auto patchType = patch.desc.GetType();
        PipelineConfig pipelineConfig = [self _lookupPipelineConfig:patchType useSingleCreasePatch:_useSingleCreasePatch];

        _patchCounts[patchType] = patch.GetNumPatches();

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
            if(pipelineConfig.useLegacyBuffers) {
                [renderCommandEncoder setVertexBuffer:_legacyGregoryPatchTable->GetQuadOffsetsBuffer() offset:_legacyGregoryPatchTable->GetQuadOffsetsBase(patchType) * sizeof(int)  atIndex:OSD_QUADOFFSET_BUFFER_INDEX];
                [renderCommandEncoder setVertexBuffer:_legacyGregoryPatchTable->GetVertexValenceBuffer() offset:0 atIndex:OSD_VALENCE_BUFFER_INDEX];
            }
            if(_numFaceVaryingElements) {;
                auto pfvarav = _mesh->GetPatchTable()->GetFVarPatchArrays();
                auto& fvarPatch = pfvarav[0]; // XXXdyu-mtl
                assert(sizeof(Osd::PatchParam) == sizeof(int) * 3);

                [renderCommandEncoder setVertexBufferOffset:(fvarPatch.primitiveIdBase+patch.primitiveIdBase) * sizeof(int) * 3 atIndex:OSD_FVAR_PATCHPARAM_BUFFER_INDEX];
                [renderCommandEncoder setVertexBufferOffset:(fvarPatch.indexBase+(patch.primitiveIdBase*fvarPatch.desc.GetNumControlVertices())) * sizeof(unsigned) atIndex:OSD_FVAR_INDICES_BUFFER_INDEX];
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

        if(_displayControlMeshEdges)
        {
            if(_displayControlMeshEdges && _controlLineRenderPipelines[patchType])
            {
                [renderCommandEncoder setRenderPipelineState:_controlLineRenderPipelines[patchType]];

                unsigned primPerPatch = 0;
                switch(patchType)
                {
                    case Far::PatchDescriptor::REGULAR:
                        primPerPatch = 48;
                        break;
                    case Far::PatchDescriptor::GREGORY:
                    case Far::PatchDescriptor::GREGORY_BOUNDARY:
                    case Far::PatchDescriptor::GREGORY_BASIS:
                        primPerPatch = 56;
                        break;
                }

                [renderCommandEncoder drawPrimitives:MTLPrimitiveTypeLine vertexStart:0 vertexCount:patch.GetNumPatches() * primPerPatch];
            }
        }
    }

    if(_displayControlMeshEdges)
    {
        [renderCommandEncoder setDepthStencilState:_readOnlyDepthStencilState];
        _controlMesh->Draw(renderCommandEncoder, _mesh->BindVertexBuffer(), _frameConstantsBuffer->ModelViewProjectionMatrix);
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
        if (pipelineConfig.useLegacyBuffers) {
            [computeCommandEncoder setBuffer:_legacyGregoryPatchTable->GetQuadOffsetsBuffer() offset:_legacyGregoryPatchTable->GetQuadOffsetsBase(patchType) * sizeof(int)  atIndex:OSD_QUADOFFSET_BUFFER_INDEX];
            [computeCommandEncoder setBuffer:_legacyGregoryPatchTable->GetVertexValenceBuffer() offset:0 atIndex:OSD_VALENCE_BUFFER_INDEX];
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
    [self _rebuildModel];
    [self _rebuildBuffers];
    [self _rebuildPipelines];
    _needsRebuild = false;
}

-(void)_rebuildModel {

    auto shapeDesc = &g_defaultShapes[[_loadedModels indexOfObject:_currentModel]];
    _shape.reset(Shape::parseObj(shapeDesc->data.c_str(), shapeDesc->scheme));

    // create Far mesh (topology)
    Sdc::SchemeType sdctype = GetSdcType(*_shape);
    Sdc::Options sdcoptions = GetSdcOptions(*_shape);

    sdcoptions.SetFVarLinearInterpolation((Sdc::Options::FVarLinearInterpolation)_fVarLinearInterp);

    std::unique_ptr<Far::TopologyRefiner> refiner;
    refiner.reset(Far::TopologyRefinerFactory<Shape>::Create(*_shape, Far::TopologyRefinerFactory<Shape>::Options(sdctype, sdcoptions)));

    // save coarse topology (used for coarse mesh drawing)
    Far::TopologyLevel const & refBaseLevel = refiner->GetLevel(0);
    _numVertices = refBaseLevel.GetNumVertices();

    Osd::MeshBitset bits;
    bits.set(Osd::MeshAdaptive,             _useAdaptive);
    bits.set(Osd::MeshUseSmoothCornerPatch, _useSmoothCornerPatch);
    bits.set(Osd::MeshUseSingleCreasePatch, _useSingleCreasePatch);
    bits.set(Osd::MeshUseInfSharpPatch,     _useInfinitelySharpPatch);
    bits.set(Osd::MeshEndCapBilinearBasis,  _endCapMode == kEndCapBilinearBasis);
    bits.set(Osd::MeshEndCapBSplineBasis,   _endCapMode == kEndCapBSplineBasis);
    bits.set(Osd::MeshEndCapGregoryBasis,   _endCapMode == kEndCapGregoryBasis);
    bits.set(Osd::MeshEndCapLegacyGregory,  _endCapMode == kEndCapLegacyGregory);

    int level = _refinementLevel;
    _numVertexElements = 3;

    _numVaryingElements = 0;
    bits.set(Osd::MeshInterleaveVarying, _numVaryingElements > 0);

    _numFaceVaryingElements = (_shadingMode == kShadingFaceVaryingColor && _shape->HasUV()) ? 2 : 0;
    if (_numFaceVaryingElements > 0) {;
        bits.set(Osd::MeshFVarData, _numFaceVaryingElements > 0);
        bits.set(Osd::MeshFVarAdaptive, _useAdaptive);
    }

    int numElements = _numVertexElements + _numVaryingElements;

    if(_kernelType == kCPU)
    {
        _mesh.reset(new CPUMeshType(refiner.get(),
                                    _numVertexElements,
                                    _numVaryingElements,
                                    level, bits, nullptr, &_context));
    }
    else
    {
        _mesh.reset(new mtlMeshType(refiner.get(),
                                    _numVertexElements,
                                    _numVaryingElements,
                                    level, bits, nullptr, &_context));
    }

    MTLRenderPipelineDescriptor* desc = [MTLRenderPipelineDescriptor new];
    [_delegate setupRenderPipelineState:desc for:self];

    const auto vertexDescriptor = desc.vertexDescriptor;
    vertexDescriptor.layouts[0].stride = sizeof(float) * numElements;
    vertexDescriptor.layouts[0].stepFunction = MTLVertexStepFunctionPerVertex;
    vertexDescriptor.layouts[0].stepRate = 1;
    vertexDescriptor.attributes[0].format = MTLVertexFormatFloat3;
    vertexDescriptor.attributes[0].offset = 0;
    vertexDescriptor.attributes[0].bufferIndex = 0;

    _controlMesh.reset(new MTLControlMeshDisplay(_context.device, desc));
    _controlMesh->SetTopology(refBaseLevel);
    _controlMesh->SetEdgesDisplay(true);
    _controlMesh->SetVerticesDisplay(false);

    _legacyGregoryPatchTable.reset();
    if(_endCapMode == kEndCapLegacyGregory)
    {
        _legacyGregoryPatchTable.reset(Osd::MTLLegacyGregoryPatchTable::Create(_mesh->GetFarPatchTable(),
                                                                          &_context));
    }

    _vertexData.resize(refBaseLevel.GetNumVertices() * numElements);

    for(int i = 0; i < refBaseLevel.GetNumVertices(); ++i)
    {
        _vertexData[i * numElements + 0] = _shape->verts[i * 3 + 0];
        _vertexData[i * numElements + 1] = _shape->verts[i * 3 + 1];
        _vertexData[i * numElements + 2] = _shape->verts[i * 3 + 2];
    }

    // compute model bounding
    float min[3] = { FLT_MAX, FLT_MAX, FLT_MAX};
    float max[3] = {-FLT_MAX,-FLT_MAX,-FLT_MAX};
    for (int i = 0; i < refBaseLevel.GetNumVertices(); ++i) {
        for (int j = 0; j < 3; ++j) {
            float v = _vertexData[i*numElements+j];
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

    _mesh->UpdateVertexBuffer(_vertexData.data(), 0, refBaseLevel.GetNumVertices());
    _mesh->Refine();
    _mesh->Synchronize();

    if(_numFaceVaryingElements > 0)
    {
        Far::StencilTableFactory::Options stencilTableFactoryOptions;
        stencilTableFactoryOptions.interpolationMode = Far::StencilTableFactory::INTERPOLATE_FACE_VARYING;
        stencilTableFactoryOptions.generateOffsets = true;
        stencilTableFactoryOptions.generateControlVerts = false;
        stencilTableFactoryOptions.generateIntermediateLevels = _useAdaptive;
        stencilTableFactoryOptions.factorizeIntermediateLevels = true;
        stencilTableFactoryOptions.maxLevel = level;
        stencilTableFactoryOptions.fvarChannel = 0;

        Far::PatchTable const *farPatchTable = _mesh->GetFarPatchTable();
        Far::StencilTable const *stencilTable = Far::StencilTableFactory::Create(*refiner, stencilTableFactoryOptions);
        Far::StencilTable const *stencilTableWithLocalPoints =
                Far::StencilTableFactory::AppendLocalPointStencilTableFaceVarying(
                        *refiner, stencilTable, farPatchTable->GetLocalPointFaceVaryingStencilTable(), 0);

        if(stencilTableWithLocalPoints) {
            delete stencilTable;
            stencilTable = stencilTableWithLocalPoints;
        }

        Osd::MTLStencilTable mtlStencilTable = Osd::MTLStencilTable(stencilTable, &_context);

        if (_numFaceVaryingElements > 0) {
            uint32_t fvarWidth             = _numFaceVaryingElements;
            uint32_t coarseFVarValuesCount = _shape->uvs.size() / fvarWidth;
            uint32_t finalFVarValuesCount  = stencilTable->GetNumStencils();

#if FVAR_SINGLE_BUFFER
            Osd::CPUMTLVertexBuffer *fvarDataBuffer = Osd::CPUMTLVertexBuffer::Create(fvarWidth, coarseFVarValuesCount + finalFVarValuesCount, &_context);
            fvarDataBuffer->UpdateData(_shape->uvs.data(), 0, coarseFVarValuesCount, &_context);

            _faceVaryingDataBuffer = fvarDataBuffer->BindMTLBuffer(&_context);
            _faceVaryingDataBuffer.label = @"OSD FVar data";

            Osd::BufferDescriptor srcDesc(0, fvarWidth, fvarWidth);
            Osd::BufferDescriptor dstDesc(coarseFVarValuesCount * fvarWidth, fvarWidth, fvarWidth);

            Osd::MTLComputeEvaluator::EvalStencils(fvarDataBuffer, srcDesc,
                                                   fvarDataBuffer, dstDesc,
                                                   &mtlStencilTable,
                                                   nullptr,
                                                   &_context);

            delete fvarDataBuffer;
#else
            Osd::CPUMTLVertexBuffer *coarseFVarDataBuffer = Osd::CPUMTLVertexBuffer::Create(fvarWidth, coarseFVarValuesCount, &_context);
            coarseFVarDataBuffer->UpdateData(_shape->uvs.data(), 0, coarseFVarValuesCount, &_context);

            id<MTLBuffer> mtlCoarseFVarDataBuffer = coarseFVarDataBuffer->BindMTLBuffer(&_context);
            mtlCoarseFVarDataBuffer.label = @"OSD FVar coarse data";

            Osd::CPUMTLVertexBuffer *refinedFVarDataBuffer = Osd::CPUMTLVertexBuffer::Create(fvarWidth, finalFVarValuesCount, &_context);
            _faceVaryingDataBuffer = refinedFVarDataBuffer->BindMTLBuffer(&_context);
            _faceVaryingDataBuffer.label = @"OSD FVar data";

            Osd::BufferDescriptor coarseBufferDescriptor(0, fvarWidth, fvarWidth);
            Osd::BufferDescriptor refinedBufferDescriptor(0, fvarWidth, fvarWidth);

            Osd::MTLComputeEvaluator::EvalStencils(coarseFVarDataBuffer, coarseBufferDescriptor,
                                                   refinedFVarDataBuffer, refinedBufferDescriptor,
                                                   &mtlStencilTable,
                                                   nullptr,
                                                   &_context);

            delete refinedFVarDataBuffer;
            delete coarseFVarDataBuffer;
#endif

            Osd::MTLPatchTable const *patchTable = _mesh->GetPatchTable();

            _faceVaryingIndicesBuffer = patchTable->GetFVarPatchIndexBuffer(0);
            _faceVaryingIndicesBuffer.label = @"OSD FVar indices";

            _faceVaryingPatchParamBuffer = patchTable->GetFVarPatchParamBuffer(0);
            _faceVaryingPatchParamBuffer.label = @"OSD FVar patch params";
        }

        delete stencilTable;
    }

    refiner.release();
}

-(void)_updateState {
    [self _updateCamera];
    auto pData = _frameConstantsBuffer.data();

    pData->TessLevel = static_cast<float>(1 << _tessellationLevel);

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
        _renderControlEdgesPipeline = nil;
    }

    Osd::MTLPatchShaderSource shaderSource;

    for (auto& patch : _mesh->GetPatchTable()->GetPatchArrays())
    {
        auto patchType = patch.desc.GetType();
        PipelineConfig pipelineConfig = [self _lookupPipelineConfig:patchType useSingleCreasePatch:_useSingleCreasePatch];

        auto compileOptions = [[MTLCompileOptions alloc] init];
        auto preprocessor = [[NSMutableDictionary alloc] init];

        std::stringstream shaderBuilder;
#define DEFINE(x,y) preprocessor[@(#x)] = @(y)

#if TARGET_OS_EMBEDDED
        shaderBuilder << "#define OSD_UV_CORRECTION if (t > 0.5){ ti += 0.01f; } else { ti += 0.01f; }\n";
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
        if (_numFaceVaryingElements > 0) {
            shaderBuilder << shaderSource.GetPatchBasisShaderSource();
        }
        shaderBuilder << _osdShaderSource.UTF8String;

        const auto str = shaderBuilder.str();

        int numElements = _numVertexElements + _numVaryingElements;

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
        DEFINE(OSD_KERNELLIMIT_BUFFER_INDEX,OSD_KERNELLIMIT_BUFFER_INDEX);

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
        DEFINE(OSD_NUM_ELEMENTS, numElements);
        DEFINE(OSD_ENABLE_BACKPATCH_CULL, _usePatchBackfaceCulling);
        DEFINE(SHADING_TYPE, _shadingMode);
        DEFINE(OSD_USE_PATCH_INDEX_BUFFER, _usePatchIndexBuffer);
        DEFINE(OSD_ENABLE_SCREENSPACE_TESSELLATION, _useScreenspaceTessellation);
        DEFINE(OSD_ENABLE_PATCH_CULL, _usePatchClipCulling && _useAdaptive);
        DEFINE(OSD_FVAR_DATA_BUFFER_INDEX, OSD_FVAR_DATA_BUFFER_INDEX);
        DEFINE(OSD_FVAR_INDICES_BUFFER_INDEX, OSD_FVAR_INDICES_BUFFER_INDEX);
        DEFINE(OSD_FVAR_PATCHPARAM_BUFFER_INDEX, OSD_FVAR_PATCHPARAM_BUFFER_INDEX);
        DEFINE(OSD_FVAR_PATCH_ARRAYS_BUFFER_INDEX, OSD_FVAR_PATCH_ARRAYS_BUFFER_INDEX);

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

            {
                pipelineDesc.fragmentFunction = [library newFunctionWithName:@"fragment_solidcolor"];
                pipelineDesc.vertexFunction = [library newFunctionWithName:@"vertex_lines"];

                if(pipelineDesc.vertexFunction)
                    _controlLineRenderPipelines[patchType] = [_context.device newRenderPipelineStateWithDescriptor:pipelineDesc error:&err];
                else
                    _controlLineRenderPipelines[patchType] = nil;
            }

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
    perspective(pData->ProjectionMatrix, 45.0, _cameraData.aspectRatio, 0.01f, 500.0);
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

-(void)_initializeModels {
    initShapes();
    _loadedModels = [[NSMutableArray alloc] initWithCapacity:g_defaultShapes.size()];
    int i = 0;
    for(auto& shape : g_defaultShapes)
    {
        _loadedModels[i++] = [NSString stringWithUTF8String:shape.name.c_str()];
    }
    _currentModel = _loadedModels[0];
}


//Setters for triggering _needsRebuild on property change

-(void)setEndCapMode:(EndCap)endCapMode {
    _needsRebuild |= endCapMode != _endCapMode;
    _endCapMode = endCapMode;
}

-(void)setUseStageIn:(bool)useStageIn {
    _needsRebuild |= useStageIn != _useStageIn;
    _useStageIn = useStageIn;
}

-(void)setShadingMode:(ShadingMode)shadingMode {
    _needsRebuild |= shadingMode != _shadingMode;
    _shadingMode = shadingMode;
}

-(void)setKernelType:(KernelType)kernelType {
    _needsRebuild |= kernelType != _kernelType;
    _kernelType = kernelType;
}

-(void)setFVarLinearInterp:(FVarLinearInterp)fVarLinearInterp {
    _needsRebuild |= (fVarLinearInterp != _fVarLinearInterp);
    _fVarLinearInterp = fVarLinearInterp;
}

-(void)setCurrentModel:(NSString *)currentModel {
    _needsRebuild |= ![currentModel isEqualToString:_currentModel];
    _currentModel = currentModel;
}

-(void)setRefinementLevel:(unsigned int)refinementLevel {
    _needsRebuild |= refinementLevel != _refinementLevel;
    _refinementLevel = refinementLevel;
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

@end
