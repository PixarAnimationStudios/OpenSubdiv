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

#include <D3D11.h>
#include <D3Dcompiler.h>

#include <opensubdiv/far/error.h>

#include <opensubdiv/osd/cpuD3D11VertexBuffer.h>
#include <opensubdiv/osd/cpuEvaluator.h>

#ifdef OPENSUBDIV_HAS_OPENMP
    #include <opensubdiv/osd/ompEvaluator.h>
#endif

#ifdef OPENSUBDIV_HAS_TBB
    #include <opensubdiv/osd/tbbEvaluator.h>
#endif

#ifdef OPENSUBDIV_HAS_OPENCL_DX_INTEROP
    #include <opensubdiv/osd/clD3D11VertexBuffer.h>
    #include <opensubdiv/osd/clEvaluator.h>
    #include "../common/clDeviceContext.h"
    CLD3D11DeviceContext g_clDeviceContext;
#endif

#ifdef OPENSUBDIV_HAS_CUDA
    #include <opensubdiv/osd/cudaD3D11VertexBuffer.h>
    #include <opensubdiv/osd/cudaEvaluator.h>
    #include "../common/cudaDeviceContext.h"
    CudaDeviceContext g_cudaDeviceContext;
#endif

#include <opensubdiv/osd/d3d11VertexBuffer.h>
#include <opensubdiv/osd/d3d11ComputeEvaluator.h>

#include <opensubdiv/osd/d3d11Mesh.h>
#include <opensubdiv/osd/d3d11LegacyGregoryPatchTable.h>
OpenSubdiv::Osd::D3D11MeshInterface *g_mesh = NULL;
OpenSubdiv::Osd::D3D11LegacyGregoryPatchTable *g_legacyGregoryPatchTable = NULL;
bool g_legacyGregoryEnabled = false;

#include "../../regression/common/far_utils.h"
#include "../../regression/common/arg_utils.h"
#include "../common/stopwatch.h"
#include "../common/simple_math.h"
#include "../common/d3d11ControlMeshDisplay.h"
#include "../common/d3d11Hud.h"
#include "../common/d3d11Utils.h"
#include "../common/d3d11ShaderCache.h"
#include "../common/viewerArgsUtils.h"

#include <opensubdiv/osd/hlslPatchShaderSource.h>
static const char *shaderSource =
#include "shader.gen.h"
;

#include <algorithm>
#include <cfloat>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <sstream>
#include <vector>

#define SAFE_RELEASE(p) { if(p) { (p)->Release(); (p)=NULL; } }

enum KernelType { kCPU           = 0,
                  kOPENMP        = 1,
                  kTBB           = 2,
                  kCUDA          = 3,
                  kCL            = 4,
                  kDirectCompute = 5 };

enum DisplayStyle { kDisplayStyleWire = 0,
                    kDisplayStyleShaded,
                    kDisplayStyleWireOnShaded };

enum ShadingMode { kShadingMaterial,
                   kShadingPatchType,
                   kShadingPatchDepth,
                   kShadingPatchCoord,
                   kShadingNormal };

enum EndCap      { kEndCapBilinearBasis = 0,
                   kEndCapBSplineBasis,
                   kEndCapGregoryBasis,
                   kEndCapLegacyGregory };

enum HudCheckBox { kHUD_CB_DISPLAY_CONTROL_MESH_EDGES,
                   kHUD_CB_DISPLAY_CONTROL_MESH_VERTS,
                   kHUD_CB_ANIMATE_VERTICES,
                   kHUD_CB_DISPLAY_PATCH_COLOR,
                   kHUD_CB_DISPLAY_PATCH_CVs,
                   kHUD_CB_VIEW_LOD,
                   kHUD_CB_FRACTIONAL_SPACING,
                   kHUD_CB_PATCH_CULL,
                   kHUD_CB_FREEZE,
                   kHUD_CB_DISPLAY_PATCH_COUNTS,
                   kHUD_CB_ADAPTIVE,
                   kHUD_CB_SMOOTH_CORNER_PATCH,
                   kHUD_CB_SINGLE_CREASE_PATCH,
                   kHUD_CB_INF_SHARP_PATCH };

int g_currentShape = 0;

int   g_frame = 0,
      g_repeatCount = 0;

// GUI variables
int   g_freeze = 0,
      g_shadingMode = kShadingPatchType,
      g_displayStyle = kDisplayStyleWireOnShaded,
      g_adaptive = 1,
      g_endCap = kEndCapGregoryBasis,
      g_smoothCornerPatch = 1,
      g_singleCreasePatch = 1,
      g_infSharpPatch = 1,
      g_drawNormals = 0,
      g_mbutton[3] = {0, 0, 0};

int   g_screenSpaceTess = 0,
      g_fractionalSpacing = 0,
      g_patchCull = 0,
      g_displayPatchCounts = 0;

float g_rotate[2] = {0, 0},
      g_prev_x = 0,
      g_prev_y = 0,
      g_dolly = 5,
      g_pan[2] = {0, 0},
      g_center[3] = {0, 0, 0},
      g_size = 0;

bool  g_yup = false;

int   g_width = 1024,
      g_height = 1024;

D3D11hud *g_hud = NULL;
D3D11ControlMeshDisplay *g_controlMeshDisplay = NULL;
float g_modelViewProjectionMatrix[16];

// performance
float g_cpuTime = 0;
float g_gpuTime = 0;
Stopwatch g_fpsTimer;

// geometry
std::vector<float> g_orgPositions,
                   g_positions;

Scheme             g_scheme;

int g_level = 2;
int g_tessLevel = 1;
int g_tessLevelMin = 1;
int g_kernel = kCPU;
float g_moveScale = 0.0f;

ID3D11Device * g_pd3dDevice = NULL;
ID3D11DeviceContext * g_pd3dDeviceContext = NULL;
IDXGISwapChain * g_pSwapChain = NULL;
ID3D11RenderTargetView * g_pSwapChainRTV = NULL;

ID3D11RasterizerState* g_pRasterizerState = NULL;
ID3D11InputLayout* g_pInputLayout = NULL;
ID3D11DepthStencilState* g_pDepthStencilState = NULL;
ID3D11Texture2D * g_pDepthStencilBuffer = NULL;
ID3D11Buffer* g_pcbPerFrame = NULL;
ID3D11Buffer* g_pcbTessellation = NULL;
ID3D11Buffer* g_pcbLighting = NULL;
ID3D11Buffer* g_pcbMaterial = NULL;
ID3D11DepthStencilView* g_pDepthStencilView = NULL;

bool g_bDone;

//------------------------------------------------------------------------------

#include "init_shapes.h"

//------------------------------------------------------------------------------
static void
updateGeom() {

    int nverts = (int)g_orgPositions.size() / 3;

    std::vector<float> vertex;
    vertex.reserve(nverts*6);

    const float *p = &g_orgPositions[0];

    float r = sin(g_frame*0.001f) * g_moveScale;
    for (int i = 0; i < nverts; ++i) {
        float move = 0.05f*cosf(p[0]*20+g_frame*0.01f);
        float ct = cos(p[2] * r);
        float st = sin(p[2] * r);
        g_positions[i*3+0] = p[0]*ct + p[1]*st;
        g_positions[i*3+1] = -p[0]*st + p[1]*ct;
        g_positions[i*3+2] = p[2];

        p += 3;
    }

    p = &g_positions[0];
    for (int i = 0; i < nverts; ++i) {
        vertex.push_back(p[0]);
        vertex.push_back(p[1]);
        vertex.push_back(p[2]);
        vertex.push_back(0.0f); // normal
        vertex.push_back(0.0f);
        vertex.push_back(0.0f);
        p += 3;
    }

    g_mesh->UpdateVertexBuffer(&vertex[0], 0, nverts);

    Stopwatch s;
    s.Start();

    g_mesh->Refine();

    s.Stop();
    g_cpuTime = float(s.GetElapsed() * 1000.0f);
    s.Start();

    g_mesh->Synchronize();

    s.Stop();
    g_gpuTime = float(s.GetElapsed() * 1000.0f);
}

//------------------------------------------------------------------------------
static const char *
getKernelName(int kernel) {

    if (kernel == kCPU)
        return "CPU";
    else if (kernel == kOPENMP)
        return "OpenMP";
    else if (kernel == kTBB)
        return "TBB";
    else if (kernel == kCUDA)
        return "Cuda";
    else if (kernel == kCL)
        return "OpenCL";
    else if (kernel == kDirectCompute)
        return "DirectCompute";
    return "Unknown";
}

//------------------------------------------------------------------------------
static void
createOsdMesh(ShapeDesc const & shapeDesc, int level, int kernel, Scheme scheme=kCatmark) {

    using namespace OpenSubdiv;
    typedef Far::ConstIndexArray IndexArray;

    Shape * shape = Shape::parseObj(shapeDesc);

    // create Far mesh (topology)
    Sdc::SchemeType sdctype = GetSdcType(*shape);
    Sdc::Options sdcoptions = GetSdcOptions(*shape);

    Far::TopologyRefiner * refiner =
        Far::TopologyRefinerFactory<Shape>::Create(*shape,
            Far::TopologyRefinerFactory<Shape>::Options(sdctype, sdcoptions));

    // save coarse topology (used for coarse mesh drawing)
    Far::TopologyLevel const & refBaseLevel = refiner->GetLevel(0);
    g_controlMeshDisplay->SetTopology(refBaseLevel);

    int nverts = refBaseLevel.GetNumVertices();

    g_orgPositions=shape->verts;

    g_positions.resize(g_orgPositions.size(),0.0f);

    delete g_mesh;
    g_mesh = NULL;

    g_scheme = scheme;

    Osd::MeshBitset bits;
    bits.set(Osd::MeshAdaptive, g_adaptive != 0);
    bits.set(Osd::MeshUseSmoothCornerPatch, g_smoothCornerPatch != 0);
    bits.set(Osd::MeshUseSingleCreasePatch, g_singleCreasePatch != 0);
    bits.set(Osd::MeshUseInfSharpPatch,     g_infSharpPatch != 0);
    bits.set(Osd::MeshEndCapBilinearBasis,  g_endCap == kEndCapBilinearBasis);
    bits.set(Osd::MeshEndCapBSplineBasis,   g_endCap == kEndCapBSplineBasis);
    bits.set(Osd::MeshEndCapGregoryBasis,   g_endCap == kEndCapGregoryBasis);
    bits.set(Osd::MeshEndCapLegacyGregory,  g_endCap == kEndCapLegacyGregory);

    int numVertexElements = 6;
    int numVaryingElements = 0;

    if (g_kernel == kCPU) {
        g_mesh = new Osd::Mesh<Osd::CpuD3D11VertexBuffer,
                               Far::StencilTable,
                               Osd::CpuEvaluator,
                               Osd::D3D11PatchTable,
                               ID3D11DeviceContext>(
                                   refiner,
                                   numVertexElements,
                                   numVaryingElements,
                                   level, bits, NULL, g_pd3dDeviceContext);

#ifdef OPENSUBDIV_HAS_OPENMP
    } else if (kernel == kOPENMP) {
        g_mesh = new Osd::Mesh<Osd::CpuD3D11VertexBuffer,
                               Far::StencilTable,
                               Osd::OmpEvaluator,
                               Osd::D3D11PatchTable,
                               ID3D11DeviceContext>(
                                   refiner,
                                   numVertexElements,
                                   numVaryingElements,
                                   level, bits, NULL, g_pd3dDeviceContext);
#endif
#ifdef OPENSUBDIV_HAS_TBB
    } else if (kernel == kTBB) {
        g_mesh = new Osd::Mesh<Osd::CpuD3D11VertexBuffer,
                               Far::StencilTable,
                               Osd::TbbEvaluator,
                               Osd::D3D11PatchTable,
                               ID3D11DeviceContext>(
                                   refiner,
                                   numVertexElements,
                                   numVaryingElements,
                                   level, bits, NULL, g_pd3dDeviceContext);
#endif
#ifdef OPENSUBDIV_HAS_OPENCL_DX_INTEROP
    } else if(kernel == kCL) {
        static Osd::EvaluatorCacheT<Osd::CLEvaluator> clEvaluatorCache;
        g_mesh = new Osd::Mesh<Osd::CLD3D11VertexBuffer,
                               Osd::CLStencilTable,
                               Osd::CLEvaluator,
                               Osd::D3D11PatchTable,
                               CLD3D11DeviceContext>(
                                   refiner,
                                   numVertexElements,
                                   numVaryingElements,
                                   level, bits,
                                   &clEvaluatorCache,
                                   &g_clDeviceContext);
#endif
#ifdef OPENSUBDIV_HAS_CUDA
    } else if (g_kernel == kCUDA) {
        g_mesh = new Osd::Mesh<Osd::CudaD3D11VertexBuffer,
                               Osd::CudaStencilTable,
                               Osd::CudaEvaluator,
                               Osd::D3D11PatchTable,
                               ID3D11DeviceContext>(
                                   refiner,
                                   numVertexElements,
                                   numVaryingElements,
                                   level, bits, NULL, g_pd3dDeviceContext);
#endif
    } else if (g_kernel == kDirectCompute) {
        static Osd::EvaluatorCacheT<Osd::D3D11ComputeEvaluator> d3d11ComputeEvaluatorCache;
        g_mesh = new Osd::Mesh<Osd::D3D11VertexBuffer,
                               Osd::D3D11StencilTable,
                               Osd::D3D11ComputeEvaluator,
                               Osd::D3D11PatchTable,
                               ID3D11DeviceContext>(
                                   refiner,
                                   numVertexElements,
                                   numVaryingElements,
                                   level, bits,
                                   &d3d11ComputeEvaluatorCache,
                                   g_pd3dDeviceContext);
    } else {
        printf("Unsupported kernel %s\n", getKernelName(kernel));
    }

    // legacy gregory
    delete g_legacyGregoryPatchTable;
    g_legacyGregoryPatchTable = NULL;
    if (g_endCap == kEndCapLegacyGregory) {
        g_legacyGregoryPatchTable =
            Osd::D3D11LegacyGregoryPatchTable::Create(
                g_mesh->GetFarPatchTable(), g_pd3dDeviceContext);
    }

    // compute model bounding
    float min[3] = { FLT_MAX,  FLT_MAX,  FLT_MAX};
    float max[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
    for (size_t i=0; i <g_orgPositions.size()/3; ++i) {
        for(int j=0; j<3; ++j) {
            float v = g_orgPositions[i*3+j];
            min[j] = std::min(min[j], v);
            max[j] = std::max(max[j], v);
        }
    }
    for (int j=0; j<3; ++j) {
        g_center[j] = (min[j] + max[j]) * 0.5f;
        g_size += (max[j]-min[j])*(max[j]-min[j]);
    }
    g_size = sqrtf(g_size);

    g_tessLevelMin = 1;

    g_tessLevel = std::max(g_tessLevel,g_tessLevelMin);

    updateGeom();
}

//------------------------------------------------------------------------------
static void
fitFrame() {

    g_pan[0] = g_pan[1] = 0;
    g_dolly = g_size;
}

//------------------------------------------------------------------------------
union Effect {
    Effect(int displayStyle_, int shadingMode_, int screenSpaceTess_,
           int fractionalSpacing_, int patchCull_, int singleCreasePatch_)
        : value(0) {
        displayStyle = displayStyle_;
        shadingMode = shadingMode_;
        screenSpaceTess = screenSpaceTess_;
        fractionalSpacing = fractionalSpacing_;
        patchCull = patchCull_;
        singleCreasePatch = singleCreasePatch_;
    }

    struct {
        unsigned int displayStyle:2;
        unsigned int shadingMode:4;
        unsigned int screenSpaceTess:1;
        unsigned int fractionalSpacing:1;
        unsigned int patchCull:1;
        unsigned int singleCreasePatch:1;
    };
    int value;

    bool operator < (const Effect &e) const {
        return value < e.value;
    }
};

struct EffectDesc {
    EffectDesc(OpenSubdiv::Far::PatchDescriptor desc,
               Effect effect) : desc(desc), effect(effect),
                                maxValence(0), numElements(0) { }

    OpenSubdiv::Far::PatchDescriptor desc;
    Effect effect;
    int maxValence;
    int numElements;

    bool operator < (const EffectDesc &e) const {
        return
            (desc < e.desc || ((desc == e.desc &&
            (maxValence < e.maxValence || ((maxValence == e.maxValence) &&
            (numElements < e.numElements || ((numElements == e.numElements) &&
            (effect < e.effect))))))));
    }
};

static Effect
GetEffect()
{
    return Effect(g_displayStyle,
                  g_shadingMode,
                  g_screenSpaceTess,
                  g_fractionalSpacing,
                  g_patchCull,
                  g_singleCreasePatch);
}

// ---------------------------------------------------------------------------

class ShaderCache : public D3D11ShaderCache<EffectDesc> {
public:
    virtual D3D11DrawConfig *CreateDrawConfig(EffectDesc const &effectDesc) {
        using namespace OpenSubdiv;

        D3D11DrawConfig *config = new D3D11DrawConfig();

        Far::PatchDescriptor::Type type = effectDesc.desc.GetType();

        // common defines
        std::stringstream ss;

        if (type == Far::PatchDescriptor::QUADS) {
            ss << "#define PRIM_QUAD\n";
        } else {
            ss << "#define PRIM_TRI\n";
        }

        // OSD tessellation controls
        if (effectDesc.effect.screenSpaceTess) {
            ss << "#define OSD_ENABLE_SCREENSPACE_TESSELLATION\n";
        }
        if (effectDesc.effect.fractionalSpacing) {
            ss << "#define OSD_FRACTIONAL_ODD_SPACING\n";
        }
        if (effectDesc.effect.patchCull) {
            ss << "#define OSD_ENABLE_PATCH_CULL\n";
        }
        if (effectDesc.effect.singleCreasePatch &&
            type == Far::PatchDescriptor::REGULAR) {
            ss << "#define OSD_PATCH_ENABLE_SINGLE_CREASE\n";
        }
        // for legacy gregory
        ss << "#define OSD_MAX_VALENCE " << effectDesc.maxValence << "\n";
        ss << "#define OSD_NUM_ELEMENTS " << effectDesc.numElements << "\n";

        // display styles
        std::string gs_entry =
            (type == Far::PatchDescriptor::QUADS ? "gs_quad" : "gs_triangle");
        if (effectDesc.desc.IsAdaptive()) gs_entry += "_smooth";

        switch (effectDesc.effect.displayStyle) {
        case kDisplayStyleWire:
            ss << "#define GEOMETRY_OUT_WIRE\n";
            gs_entry = gs_entry + "_wire";
            break;
        case kDisplayStyleWireOnShaded:
            ss << "#define GEOMETRY_OUT_LINE\n";
            gs_entry = gs_entry + "_wire";
            break;
        case kDisplayStyleShaded:
            ss << "#define GEOMETRY_OUT_FILL\n";
            break;
        }

        // shading mode
        switch(effectDesc.effect.shadingMode) {
        case kShadingMaterial:
            ss << "#define SHADING_MATERIAL\n";
            break;
        case kShadingPatchDepth:
            ss << "#define SHADING_PATCH_DEPTH\n";
            break;
        case kShadingPatchType:
            ss << "#define SHADING_PATCH_TYPE\n";
            break;
        case kShadingPatchCoord:
            ss << "#define SHADING_PATCH_COORD\n";
            break;
        case kShadingNormal:
            ss << "#define SHADING_NORMAL\n";
            break;
        }

        // need for patch color-coding : we need these defines in the fragment shader
        if (type == Far::PatchDescriptor::GREGORY) {
            ss << "#define OSD_PATCH_GREGORY\n";
        } else if (type == Far::PatchDescriptor::GREGORY_BOUNDARY) {
            ss << "#define OSD_PATCH_GREGORY_BOUNDARY\n";
        } else if (type == Far::PatchDescriptor::GREGORY_BASIS) {
            ss << "#define OSD_PATCH_GREGORY_BASIS\n";
        } else if (type == Far::PatchDescriptor::LOOP) {
            ss << "#define OSD_PATCH_LOOP\n";
        } else if (type == Far::PatchDescriptor::GREGORY_TRIANGLE) {
            ss << "#define OSD_PATCH_GREGORY_TRIANGLE\n";
        }

        // include osd PatchCommon
        ss << "#define OSD_PATCH_BASIS_HLSL\n";
        ss << Osd::HLSLPatchShaderSource::GetPatchBasisShaderSource();
        ss << Osd::HLSLPatchShaderSource::GetCommonShaderSource();
        std::string common = ss.str();
        ss.str("");

        // input layout
        const D3D11_INPUT_ELEMENT_DESC hInElementDesc[] = {
            { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
            { "NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 4*3, D3D11_INPUT_PER_VERTEX_DATA, 0 }
        };

        // vertex shader
        ss << common
           << shaderSource
           << Osd::HLSLPatchShaderSource::GetVertexShaderSource(type);
        if (effectDesc.desc.IsAdaptive()) {
            config->CompileVertexShader("vs_5_0", "vs_main_patches", ss.str(),
                                        &g_pInputLayout,
                                        hInElementDesc,
                                        ARRAYSIZE(hInElementDesc),
                                        g_pd3dDevice);
        } else {
            config->CompileVertexShader("vs_5_0", "vs_main",
                                        ss.str(),
                                        &g_pInputLayout,
                                        hInElementDesc,
                                        ARRAYSIZE(hInElementDesc),
                                        g_pd3dDevice);
        }
        ss.str("");


        if (effectDesc.desc.IsAdaptive()) {
            // hull shader
            ss << common
               << shaderSource
               << Osd::HLSLPatchShaderSource::GetHullShaderSource(type);
            config->CompileHullShader("hs_5_0", "hs_main_patches", ss.str(),
                                      g_pd3dDevice);
            ss.str("");

            // domain shader
            ss << common
               << shaderSource
               << Osd::HLSLPatchShaderSource::GetDomainShaderSource(type);
            config->CompileDomainShader("ds_5_0", "ds_main_patches", ss.str(),
                                        g_pd3dDevice);
            ss.str("");
        }

        // geometry shader
        ss << common
           << shaderSource;
        config->CompileGeometryShader("gs_5_0", gs_entry,
                                      ss.str(),
                                      g_pd3dDevice);
        ss.str("");

        // pixel shader
        ss << common
           << shaderSource;
        config->CompilePixelShader("ps_5_0", "ps_main", ss.str(),
                                   g_pd3dDevice);
        ss.str("");

        return config;
    };
};

ShaderCache g_shaderCache;

//------------------------------------------------------------------------------
static void
bindProgram(Effect effect, OpenSubdiv::Osd::PatchArray const & patch) {

    EffectDesc effectDesc(patch.GetDescriptor(), effect);
    typedef OpenSubdiv::Far::PatchDescriptor Descriptor;

    // only legacy gregory needs maxValence and numElements
    // neither legacy gregory nor gregory basis need single crease
    if (patch.GetDescriptor().GetType() == Descriptor::GREGORY ||
        patch.GetDescriptor().GetType() == Descriptor::GREGORY_BOUNDARY) {
        int maxValence = g_mesh->GetMaxValence();
        int numElements = 6;
        effectDesc.maxValence = maxValence;
        effectDesc.numElements = numElements;
        // note: singleCreasePatch needs to be left defined for the patchParam
        // datatype consistency.
    }

    D3D11DrawConfig *config = g_shaderCache.GetDrawConfig(effectDesc);

    assert(g_pInputLayout);

    // Update transform state
    {
        __declspec(align(16))
        struct CB_PER_FRAME_CONSTANTS
        {
            float ModelViewMatrix[16];
            float ProjectionMatrix[16];
            float ModelViewProjectionMatrix[16];
            float ModelViewInverseMatrix[16];
        };

        if (! g_pcbPerFrame) {
            D3D11_BUFFER_DESC cbDesc;
            ZeroMemory(&cbDesc, sizeof(cbDesc));
            cbDesc.Usage = D3D11_USAGE_DYNAMIC;
            cbDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
            cbDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
            cbDesc.MiscFlags = 0;
            cbDesc.ByteWidth = sizeof(CB_PER_FRAME_CONSTANTS);
            g_pd3dDevice->CreateBuffer(&cbDesc, NULL, &g_pcbPerFrame);
        }
        assert(g_pcbPerFrame);

        D3D11_MAPPED_SUBRESOURCE MappedResource;
        g_pd3dDeviceContext->Map(g_pcbPerFrame, 0, D3D11_MAP_WRITE_DISCARD, 0, &MappedResource);
        CB_PER_FRAME_CONSTANTS* pData = ( CB_PER_FRAME_CONSTANTS* )MappedResource.pData;

        float aspect = (g_height > 0) ? (float)g_width / g_height : 1.0f;
        identity(pData->ModelViewMatrix);
        translate(pData->ModelViewMatrix, -g_pan[0], -g_pan[1], -g_dolly);
        rotate(pData->ModelViewMatrix, g_rotate[1], 1, 0, 0);
        rotate(pData->ModelViewMatrix, g_rotate[0], 0, 1, 0);
        translate(pData->ModelViewMatrix, -g_center[0], -g_center[2], g_center[1]); // z-up model
        if (!g_yup) {
            rotate(pData->ModelViewMatrix, -90, 1, 0, 0); // z-up model
        }
        inverseMatrix(pData->ModelViewInverseMatrix, pData->ModelViewMatrix);

        identity(pData->ProjectionMatrix);
        perspective(pData->ProjectionMatrix, 45.0, aspect, 0.01f, 500.0);
        multMatrix(pData->ModelViewProjectionMatrix, pData->ModelViewMatrix, pData->ProjectionMatrix);

        memcpy(g_modelViewProjectionMatrix, pData->ModelViewProjectionMatrix, sizeof(float) * 16);

        g_pd3dDeviceContext->Unmap( g_pcbPerFrame, 0 );
    }

    // Update tessellation state
    {
        __declspec(align(16))
        struct Tessellation {
            float TessLevel;
            int GregoryQuadOffsetBase;
            int PrimitiveIdBase;
        };

        if (! g_pcbTessellation) {
            D3D11_BUFFER_DESC cbDesc;
            ZeroMemory(&cbDesc, sizeof(cbDesc));
            cbDesc.Usage = D3D11_USAGE_DYNAMIC;
            cbDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
            cbDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
            cbDesc.MiscFlags = 0;
            cbDesc.ByteWidth = sizeof(Tessellation);
            g_pd3dDevice->CreateBuffer(&cbDesc, NULL, &g_pcbTessellation);
        }
        assert(g_pcbTessellation);

        D3D11_MAPPED_SUBRESOURCE MappedResource;
        g_pd3dDeviceContext->Map(g_pcbTessellation, 0, D3D11_MAP_WRITE_DISCARD, 0, &MappedResource);
        Tessellation * pData = ( Tessellation* )MappedResource.pData;

        pData->TessLevel = static_cast<float>(1 << g_tessLevel);
        pData->GregoryQuadOffsetBase = g_legacyGregoryPatchTable ?
            g_legacyGregoryPatchTable->GetQuadOffsetsBase(patch.GetDescriptor().GetType()) : 0;
        pData->PrimitiveIdBase = patch.GetPrimitiveIdBase();

        g_pd3dDeviceContext->Unmap( g_pcbTessellation, 0 );
    }

    // Update material state
    {
        __declspec(align(16))
        struct Material {
            float color[4];
        };

        if (! g_pcbMaterial) {
            D3D11_BUFFER_DESC cbDesc;
            ZeroMemory(&cbDesc, sizeof(cbDesc));
            cbDesc.Usage = D3D11_USAGE_DYNAMIC;
            cbDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
            cbDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
            cbDesc.MiscFlags = 0;
            cbDesc.ByteWidth = sizeof(Material);
            g_pd3dDevice->CreateBuffer(&cbDesc, NULL, &g_pcbMaterial);
        }
        assert(g_pcbMaterial);

        D3D11_MAPPED_SUBRESOURCE MappedResource;
        g_pd3dDeviceContext->Map(g_pcbMaterial, 0, D3D11_MAP_WRITE_DISCARD, 0, &MappedResource);
        Material * pData = ( Material* )MappedResource.pData;

        static float const uniformColor[4] = {0.13f, 0.13f, 0.61f, 1.0f};
        memcpy(pData->color, uniformColor, 4*sizeof(float));

        g_pd3dDeviceContext->Unmap( g_pcbMaterial, 0 );
    }

    g_pd3dDeviceContext->IASetInputLayout(g_pInputLayout);

    g_pd3dDeviceContext->VSSetShader(config->GetVertexShader(), NULL, 0);
    g_pd3dDeviceContext->VSSetConstantBuffers(0, 1, &g_pcbPerFrame);

    g_pd3dDeviceContext->HSSetShader(config->GetHullShader(), NULL, 0);
    g_pd3dDeviceContext->HSSetConstantBuffers(0, 1, &g_pcbPerFrame);
    g_pd3dDeviceContext->HSSetConstantBuffers(1, 1, &g_pcbTessellation);

    g_pd3dDeviceContext->DSSetShader(config->GetDomainShader(), NULL, 0);
    g_pd3dDeviceContext->DSSetConstantBuffers(0, 1, &g_pcbPerFrame);

    g_pd3dDeviceContext->GSSetShader(config->GetGeometryShader(), NULL, 0);
    g_pd3dDeviceContext->GSSetConstantBuffers(0, 1, &g_pcbPerFrame);

    g_pd3dDeviceContext->PSSetShader(config->GetPixelShader(), NULL, 0);
    g_pd3dDeviceContext->PSSetConstantBuffers(0, 1, &g_pcbPerFrame);
    g_pd3dDeviceContext->PSSetConstantBuffers(2, 1, &g_pcbLighting);
    g_pd3dDeviceContext->PSSetConstantBuffers(3, 1, &g_pcbMaterial);

    ID3D11ShaderResourceView *srv = g_mesh->GetPatchTable()->GetPatchParamSRV();
    if (srv) {
        g_pd3dDeviceContext->HSSetShaderResources(0, 1, &srv);  // t0
        g_pd3dDeviceContext->DSSetShaderResources(0, 1, &srv);
        g_pd3dDeviceContext->PSSetShaderResources(0, 1, &srv);
    }

    if (g_legacyGregoryPatchTable) {
        ID3D11ShaderResourceView *vertexSRV =
            g_legacyGregoryPatchTable->GetVertexSRV();
        ID3D11ShaderResourceView *vertexValenceSRV =
            g_legacyGregoryPatchTable->GetVertexValenceSRV();
        ID3D11ShaderResourceView *quadOffsetsSRV =
            g_legacyGregoryPatchTable->GetQuadOffsetsSRV();
        g_pd3dDeviceContext->VSSetShaderResources(2, 1, &vertexSRV);       // t2
        g_pd3dDeviceContext->VSSetShaderResources(3, 1, &vertexValenceSRV);// t3
        g_pd3dDeviceContext->HSSetShaderResources(4, 1, &quadOffsetsSRV);  // t4
    }
}

//------------------------------------------------------------------------------
static void
display() {

    float color[4] = {0.006f, 0.006f, 0.006f, 1.0f};
    g_pd3dDeviceContext->ClearRenderTargetView(g_pSwapChainRTV, color);

    // Clear the depth buffer.
    g_pd3dDeviceContext->ClearDepthStencilView(g_pDepthStencilView, D3D11_CLEAR_DEPTH, 1.0f, 0);

    g_pd3dDeviceContext->OMSetDepthStencilState(g_pDepthStencilState, 1);
    g_pd3dDeviceContext->RSSetState(g_pRasterizerState);

    ID3D11Buffer *buffer = g_mesh->BindVertexBuffer();
    assert(buffer);

    // vertex texture update for legacy gregory drawing
    if (g_legacyGregoryPatchTable) {
        g_legacyGregoryPatchTable->UpdateVertexBuffer(buffer,
                                                      g_mesh->GetNumVertices(),
                                                      6,
                                                      g_pd3dDeviceContext);
    }

    UINT hStrides = 6*sizeof(float);
    UINT hOffsets = 0;
    g_pd3dDeviceContext->IASetVertexBuffers(0, 1, &buffer, &hStrides, &hOffsets);

    OpenSubdiv::Osd::PatchArrayVector const & patches =
        g_mesh->GetPatchTable()->GetPatchArrays();

    g_pd3dDeviceContext->IASetIndexBuffer(
        g_mesh->GetPatchTable()->GetPatchIndexBuffer(), DXGI_FORMAT_R32_UINT, 0);

    // patch drawing
    int patchCount[13]; // [Type] (see far/patchTable.h)
    int numTotalPatches = 0;
    int numDrawCalls = 0;
    memset(patchCount, 0, sizeof(patchCount));

    for (int i=0; i<(int)patches.size(); ++i) {
        OpenSubdiv::Osd::PatchArray const & patch = patches[i];

        OpenSubdiv::Far::PatchDescriptor desc = patch.GetDescriptor();
        OpenSubdiv::Far::PatchDescriptor::Type patchType = desc.GetType();

        patchCount[patchType] += patch.GetNumPatches();
        numTotalPatches += patch.GetNumPatches();

        D3D11_PRIMITIVE_TOPOLOGY topology;

        switch (patchType) {
        case OpenSubdiv::Far::PatchDescriptor::TRIANGLES:
            topology = D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
            break;
        case OpenSubdiv::Far::PatchDescriptor::QUADS:
            topology = D3D11_PRIMITIVE_TOPOLOGY_LINELIST_ADJ;
            break;
        default:
            switch (desc.GetNumControlVertices()) {
            case 4:
                topology = D3D11_PRIMITIVE_TOPOLOGY_4_CONTROL_POINT_PATCHLIST;
                break;
            case 9:
                topology = D3D11_PRIMITIVE_TOPOLOGY_9_CONTROL_POINT_PATCHLIST;
                break;
            case 12:
                topology = D3D11_PRIMITIVE_TOPOLOGY_12_CONTROL_POINT_PATCHLIST;
                break;
            case 15:
                topology = D3D11_PRIMITIVE_TOPOLOGY_15_CONTROL_POINT_PATCHLIST;
                break;
            case 16:
                topology = D3D11_PRIMITIVE_TOPOLOGY_16_CONTROL_POINT_PATCHLIST;
                break;
            case 18:
                topology = D3D11_PRIMITIVE_TOPOLOGY_18_CONTROL_POINT_PATCHLIST;
                break;
            case 20:
                topology = D3D11_PRIMITIVE_TOPOLOGY_20_CONTROL_POINT_PATCHLIST;
                break;
            default:
                assert(false);
                break;
            }
            break;
        }

        bindProgram(GetEffect(), patch);

        g_pd3dDeviceContext->IASetPrimitiveTopology(topology);

        g_pd3dDeviceContext->DrawIndexed(
            patch.GetNumPatches() * desc.GetNumControlVertices(),
            patch.GetIndexBase(), 0);
    }

    // draw the control mesh
    g_controlMeshDisplay->Draw(buffer, 6, g_modelViewProjectionMatrix);

    g_fpsTimer.Stop();
    float elapsed = (float)g_fpsTimer.GetElapsed();
    g_fpsTimer.Start();

    if (g_hud->IsVisible()) {

        typedef OpenSubdiv::Far::PatchDescriptor Descriptor;

        double fps = 1.0/g_fpsTimer.GetElapsed();

        if (g_displayPatchCounts) {
            int x = -420;
            int y = g_legacyGregoryEnabled ? -180 : -140;
            g_hud->DrawString(x, y, "Quads            : %d",
                             patchCount[Descriptor::QUADS]); y += 20;
            g_hud->DrawString(x, y, "Triangles        : %d",
                             patchCount[Descriptor::TRIANGLES]); y += 20;
            g_hud->DrawString(x, y, "Regular          : %d",
                             patchCount[Descriptor::REGULAR]); y+= 20;
            g_hud->DrawString(x, y, "Loop             : %d",
                             patchCount[Descriptor::LOOP]); y+= 20;
            if (g_legacyGregoryEnabled) {
                g_hud->DrawString(x, y, "Gregory          : %d",
                                 patchCount[Descriptor::GREGORY]); y+= 20;
                g_hud->DrawString(x, y, "Boundary Gregory : %d",
                                 patchCount[Descriptor::GREGORY_BOUNDARY]); y+= 20;
            }
            g_hud->DrawString(x, y, "Gregory Basis    : %d",
                             patchCount[Descriptor::GREGORY_BASIS]); y+= 20;
            g_hud->DrawString(x, y, "Gregory Triangle : %d",
                             patchCount[Descriptor::GREGORY_TRIANGLE]); y+= 20;
        }

        g_hud->DrawString(10, -120, "Tess level : %d", g_tessLevel);
        g_hud->DrawString(10, -100, "Control Vertices = %d", g_mesh->GetNumVertices());
        g_hud->DrawString(10, -80, "Scheme = %s", g_scheme==kBilinear ? "BILINEAR" : (g_scheme == kLoop ? "LOOP" : "CATMARK"));
        g_hud->DrawString(10, -60, "GPU TIME = %.3f ms", g_gpuTime);
        g_hud->DrawString(10, -40, "CPU TIME = %.3f ms", g_cpuTime);
        g_hud->DrawString(10, -20, "FPS = %3.1f", fps);

        g_hud->Flush();
    }

    g_pSwapChain->Present(0, 0);
}

//------------------------------------------------------------------------------
static void
motion(int x, int y) {

    if (g_mbutton[0] && !g_mbutton[1] && !g_mbutton[2]) {
        // orbit
        g_rotate[0] += x - g_prev_x;
        g_rotate[1] += y - g_prev_y;
    } else if (!g_mbutton[0] && g_mbutton[1] && !g_mbutton[2]) {
        // pan
        g_pan[0] -= g_dolly*(x - g_prev_x)/g_width;
        g_pan[1] += g_dolly*(y - g_prev_y)/g_height;
    } else if ((g_mbutton[0] && g_mbutton[1] && !g_mbutton[2]) ||
               (!g_mbutton[0] && !g_mbutton[1] && g_mbutton[2])) {
        // dolly
        g_dolly -= g_dolly*0.01f*(x - g_prev_x);
        if(g_dolly <= 0.01) g_dolly = 0.01f;
    }

    g_prev_x = float(x);
    g_prev_y = float(y);
}

//------------------------------------------------------------------------------
static void
mouse(int button, int state, int x, int y) {

    if (button == 0 && state == 1 && g_hud->MouseClick(x, y)) return;

    if (button < 3) {
        g_prev_x = float(x);
        g_prev_y = float(y);
        g_mbutton[button] = state;
    }
}

//-----------------------------------------------------------------------------
static void
quit() {

    g_bDone = true;

    if (g_mesh)
        delete g_mesh;

    if (g_hud)
        delete g_hud;

    if (g_controlMeshDisplay)
        delete g_controlMeshDisplay;

    SAFE_RELEASE(g_pRasterizerState);
    SAFE_RELEASE(g_pInputLayout);
    SAFE_RELEASE(g_pDepthStencilState);
    SAFE_RELEASE(g_pcbPerFrame);
    SAFE_RELEASE(g_pcbTessellation);
    SAFE_RELEASE(g_pcbLighting);
    SAFE_RELEASE(g_pcbMaterial);
    SAFE_RELEASE(g_pDepthStencilView);

    SAFE_RELEASE(g_pSwapChainRTV);
    SAFE_RELEASE(g_pSwapChain);
    SAFE_RELEASE(g_pd3dDeviceContext);
    SAFE_RELEASE(g_pd3dDevice);

    PostQuitMessage(0);
    exit(0);
}

//------------------------------------------------------------------------------
static void
keyboard(char key) {

    if (g_hud->KeyDown((int)key)) return;

    switch (key) {
        case 'Q': quit();
        case 'F': fitFrame(); break;
        case '+':
        case '=': g_tessLevel++; break;
        case '-': g_tessLevel = std::max(g_tessLevelMin, g_tessLevel-1); break;
        case 0x1b: g_hud->SetVisible(!g_hud->IsVisible()); break;
    }
}

//------------------------------------------------------------------------------
static void
rebuildOsdMesh() {
    createOsdMesh( g_defaultShapes[ g_currentShape ], g_level, g_kernel, g_defaultShapes[ g_currentShape ].scheme );
}

static void
callbackDisplayStyle(int b) {
    g_displayStyle = b;
}

static void
callbackEndCap(int endCap) {
    g_endCap = endCap;
    rebuildOsdMesh();
}

static void
callbackKernel(int k) {

    g_kernel = k;

#ifdef OPENSUBDIV_HAS_OPENCL_DX_INTEROP
    if (g_kernel == kCL && (!g_clDeviceContext.IsInitialized())) {
        if (g_clDeviceContext.Initialize(g_pd3dDeviceContext) == false) {
            printf("Error in initializing OpenCL\n");
            exit(1);
        }
    }
#endif
#ifdef OPENSUBDIV_HAS_CUDA
    if (g_kernel == kCUDA && (!g_cudaDeviceContext.IsInitialized())) {
        if (g_cudaDeviceContext.Initialize(g_pd3dDevice) == false) {
            printf("Error in initializing Cuda\n");
            exit(1);
        }
    }
#endif

    rebuildOsdMesh();
}

static void
callbackLevel(int l) {
    g_level = l;
    rebuildOsdMesh();
}

static void
callbackModel(int m) {

    if (m < 0) {
        m = 0;
    }

    if (m >= (int)g_defaultShapes.size()) {
        m = (int)g_defaultShapes.size() - 1;
    }

    g_currentShape = m;
    rebuildOsdMesh();
}

static void
callbackDisplayNormal(bool checked, int n) {
    g_drawNormals = checked;
}

static void
callbackShadingMode(int b) {
    g_shadingMode = b;
}


static void
callbackCheckBox(bool checked, int button) {
    switch (button) {
    case kHUD_CB_DISPLAY_CONTROL_MESH_EDGES:
        g_controlMeshDisplay->SetEdgesDisplay(checked);
        break;
    case kHUD_CB_DISPLAY_CONTROL_MESH_VERTS:
        g_controlMeshDisplay->SetVerticesDisplay(checked);
        break;
    case kHUD_CB_ANIMATE_VERTICES:
        g_moveScale = checked;
        break;
    case kHUD_CB_VIEW_LOD:
        g_screenSpaceTess = checked;
        break;
    case kHUD_CB_FRACTIONAL_SPACING:
        g_fractionalSpacing = checked;
        break;
    case kHUD_CB_PATCH_CULL:
        g_patchCull = checked;
        break;
    case kHUD_CB_FREEZE:
        g_freeze = checked;
        break;
    case kHUD_CB_DISPLAY_PATCH_COUNTS:
        g_displayPatchCounts = checked;
        break;
    case kHUD_CB_ADAPTIVE:
        g_adaptive = checked;
        rebuildOsdMesh();
        break;
    case kHUD_CB_SMOOTH_CORNER_PATCH:
        g_smoothCornerPatch = checked;
        rebuildOsdMesh();
        break;
    case kHUD_CB_SINGLE_CREASE_PATCH:
        g_singleCreasePatch = checked;
        rebuildOsdMesh();
        break;
    case kHUD_CB_INF_SHARP_PATCH:
        g_infSharpPatch = checked;
        rebuildOsdMesh();
        break;
    }
}


static void
initHUD() {

    g_hud = new D3D11hud(g_pd3dDeviceContext);
    g_hud->Init(g_width, g_height);


    int compute_pulldown = g_hud->AddPullDown("Compute (K)", 475, 10, 300, callbackKernel, 'K');
    g_hud->AddPullDownButton(compute_pulldown, "CPU", kCPU);
#ifdef OPENSUBDIV_HAS_OPENMP
    g_hud->AddPullDownButton(compute_pulldown, "OpenMP", kOPENMP);
#endif
#ifdef OPENSUBDIV_HAS_TBB
    g_hud->AddPullDownButton(compute_pulldown, "TBB", kTBB);
#endif
#ifdef OPENSUBDIV_HAS_CUDA
    g_hud->AddPullDownButton(compute_pulldown, "CUDA", kCUDA);
#endif
#ifdef OPENSUBDIV_HAS_OPENCL_DX_INTEROP
    if (CLDeviceContext::HAS_CL_VERSION_1_1()) {
        g_hud->AddPullDownButton(compute_pulldown, "OpenCL", kCL);
    }
#endif
    g_hud->AddPullDownButton(compute_pulldown, "HLSL Compute", kDirectCompute);

    int displaystyle_pulldown = g_hud->AddPullDown("DisplayStyle (W)", 200, 10, 250,
                                                   callbackDisplayStyle, 'W');
    g_hud->AddPullDownButton(displaystyle_pulldown, "Wire", kDisplayStyleWire,
                            g_displayStyle == kDisplayStyleWire);
    g_hud->AddPullDownButton(displaystyle_pulldown, "Shaded", kDisplayStyleShaded,
                            g_displayStyle == kDisplayStyleShaded);
    g_hud->AddPullDownButton(displaystyle_pulldown, "Wire+Shaded", kDisplayStyleWireOnShaded,
                            g_displayStyle == kDisplayStyleWireOnShaded);

    int shading_pulldown = g_hud->AddPullDown("Shading (C)", 200, 70, 250,
                                             callbackShadingMode, 'C');
    g_hud->AddPullDownButton(shading_pulldown, "Material",
                            kShadingMaterial,
                            g_shadingMode == kShadingMaterial);
    g_hud->AddPullDownButton(shading_pulldown, "Patch Type",
                            kShadingPatchType,
                            g_shadingMode == kShadingPatchType);
    g_hud->AddPullDownButton(shading_pulldown, "Patch Depth",
                            kShadingPatchDepth,
                            g_shadingMode == kShadingPatchCoord);
    g_hud->AddPullDownButton(shading_pulldown, "Patch Coord",
                            kShadingPatchCoord,
                            g_shadingMode == kShadingPatchCoord);
    g_hud->AddPullDownButton(shading_pulldown, "Normal",
                            kShadingNormal,
                            g_shadingMode == kShadingNormal);

    int y = 10;
    g_hud->AddCheckBox("Control edges (H)",
                       g_controlMeshDisplay->GetEdgesDisplay(),
                       10, y, callbackCheckBox,
                       kHUD_CB_DISPLAY_CONTROL_MESH_EDGES, 'H');
    y += 20;
    g_hud->AddCheckBox("Control vertices (J)",
                       g_controlMeshDisplay->GetVerticesDisplay(),
                       10, y, callbackCheckBox,
                       kHUD_CB_DISPLAY_CONTROL_MESH_VERTS, 'J');
    y += 20;
    g_hud->AddCheckBox("Animate vertices (M)", g_moveScale != 0,
                       10, y, callbackCheckBox, kHUD_CB_ANIMATE_VERTICES, 'M');
    y += 20;
    g_hud->AddCheckBox("Screen space LOD (V)", g_screenSpaceTess != 0,
                       10, y, callbackCheckBox, kHUD_CB_VIEW_LOD, 'V');
    y += 20;
    g_hud->AddCheckBox("Fractional spacing (T)", g_fractionalSpacing != 0,
                       10, y, callbackCheckBox, kHUD_CB_FRACTIONAL_SPACING, 'T');
    y += 20;
    g_hud->AddCheckBox("Frustum Patch Culling (B)", g_patchCull != 0,
                       10, y, callbackCheckBox, kHUD_CB_PATCH_CULL, 'B');
    y += 20;
    g_hud->AddCheckBox("Freeze (spc)", g_freeze != 0,
                       10, y, callbackCheckBox, kHUD_CB_FREEZE, ' ');
    y += 20;

    g_hud->AddCheckBox("Adaptive (`)", true,
                       10, 190, callbackCheckBox, kHUD_CB_ADAPTIVE, '`');
    g_hud->AddCheckBox("Smooth Corner Patch (O)", g_smoothCornerPatch!=0,
                       10, 210, callbackCheckBox, kHUD_CB_SMOOTH_CORNER_PATCH, 'O');
    g_hud->AddCheckBox("Single Crease Patch (S)", g_singleCreasePatch!=0,
                       10, 230, callbackCheckBox, kHUD_CB_SINGLE_CREASE_PATCH, 'S');
    g_hud->AddCheckBox("Inf Sharp Patch (I)", g_infSharpPatch!=0,
                       10, 250, callbackCheckBox, kHUD_CB_INF_SHARP_PATCH, 'I');

    int endcap_pulldown = g_hud->AddPullDown(
        "End cap (E)", 10, 270, 200, callbackEndCap, 'E');
    g_hud->AddPullDownButton(endcap_pulldown,"Linear",
                             kEndCapBilinearBasis,
                             g_endCap == kEndCapBilinearBasis);
    g_hud->AddPullDownButton(endcap_pulldown, "Regular",
                             kEndCapBSplineBasis,
                             g_endCap == kEndCapBSplineBasis);
    g_hud->AddPullDownButton(endcap_pulldown, "Gregory",
                             kEndCapGregoryBasis,
                             g_endCap == kEndCapGregoryBasis);
    if (g_legacyGregoryEnabled) {
        g_hud->AddPullDownButton(endcap_pulldown, "LegacyGregory",
                                 kEndCapLegacyGregory,
                                 g_endCap == kEndCapLegacyGregory);
    }

    for (int i = 1; i < 11; ++i) {
        char level[16];
        sprintf(level, "Lv. %d", i);
        g_hud->AddRadioButton(3, level, i==2, 10, 290+i*20, callbackLevel, i, '0'+(i%10));
    }

    int shapes_pulldown = g_hud->AddPullDown("Shape (N)", -300, 10, 300, callbackModel, 'N');
    for (int i = 0; i < (int)g_defaultShapes.size(); ++i) {
        g_hud->AddPullDownButton(shapes_pulldown, g_defaultShapes[i].name.c_str(),i);
    }

    g_hud->AddCheckBox("Show patch counts", g_displayPatchCounts!=0, -420, -20, callbackCheckBox, kHUD_CB_DISPLAY_PATCH_COUNTS);

    callbackModel(g_currentShape);
}

//------------------------------------------------------------------------------
static bool
initD3D11(HWND hWnd) {

    D3D_DRIVER_TYPE driverTypes[] = {
        D3D_DRIVER_TYPE_HARDWARE,
        D3D_DRIVER_TYPE_WARP,
        D3D_DRIVER_TYPE_REFERENCE,
    };

    UINT numDriverTypes = ARRAYSIZE(driverTypes);

    DXGI_SWAP_CHAIN_DESC hDXGISwapChainDesc;
    hDXGISwapChainDesc.BufferDesc.Width = g_width;
    hDXGISwapChainDesc.BufferDesc.Height = g_height;
    hDXGISwapChainDesc.BufferDesc.RefreshRate.Numerator  = 0;
    hDXGISwapChainDesc.BufferDesc.RefreshRate.Denominator = 1;
    hDXGISwapChainDesc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
    hDXGISwapChainDesc.BufferDesc.ScanlineOrdering = DXGI_MODE_SCANLINE_ORDER_UNSPECIFIED;
    hDXGISwapChainDesc.BufferDesc.Scaling = DXGI_MODE_SCALING_UNSPECIFIED;
    hDXGISwapChainDesc.SampleDesc.Count = 1;
    hDXGISwapChainDesc.SampleDesc.Quality = 0;
    hDXGISwapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    hDXGISwapChainDesc.BufferCount = 1;
    hDXGISwapChainDesc.OutputWindow = hWnd;
    hDXGISwapChainDesc.Windowed = TRUE;
    hDXGISwapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
    hDXGISwapChainDesc.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;

    // create device and swap chain
    HRESULT hr;
    D3D_DRIVER_TYPE hDriverType = D3D_DRIVER_TYPE_NULL;
    D3D_FEATURE_LEVEL hFeatureLevel = D3D_FEATURE_LEVEL_11_0;
    for(UINT driverTypeIndex=0; driverTypeIndex < numDriverTypes; driverTypeIndex++){
        hDriverType = driverTypes[driverTypeIndex];
        hr = D3D11CreateDeviceAndSwapChain(NULL,
                                           hDriverType, NULL, 0, NULL, 0,
                                           D3D11_SDK_VERSION, &hDXGISwapChainDesc,
                                           &g_pSwapChain, &g_pd3dDevice,
                                           &hFeatureLevel, &g_pd3dDeviceContext);
        if(SUCCEEDED(hr)){
            break;
        }
    }

    if(FAILED(hr)){
        MessageBoxW(hWnd, L"D3D11CreateDeviceAndSwapChain", L"Err", MB_ICONSTOP);
        return false;
    }

    // create rasterizer
    D3D11_RASTERIZER_DESC rasterDesc;
    ZeroMemory(&rasterDesc, sizeof(rasterDesc));
    rasterDesc.AntialiasedLineEnable = false;
    rasterDesc.CullMode = D3D11_CULL_BACK;
    rasterDesc.DepthBias = 0;
    rasterDesc.DepthBiasClamp = 0.0f;
    rasterDesc.DepthClipEnable = true;
    rasterDesc.FillMode = D3D11_FILL_SOLID;
    rasterDesc.FrontCounterClockwise = true;
    rasterDesc.MultisampleEnable = false;
    rasterDesc.ScissorEnable = false;
    rasterDesc.SlopeScaledDepthBias = 0.0f;

    g_pd3dDevice->CreateRasterizerState(&rasterDesc, &g_pRasterizerState);
    assert(g_pRasterizerState);

    {   // update the lighting constant buffer
        __declspec(align(16))
        struct Lighting {
            struct Light {
                float position[4];
                float ambient[4];
                float diffuse[4];
                float specular[4];
            } lightSource[2];
        } lightingData = {
            {{ { 0.5, 0.2f, 1.0f, 0.0f },
               { 0.1f, 0.1f, 0.1f, 1.0f },
               { 0.7f, 0.7f, 0.7f, 1.0f },
               { 0.8f, 0.8f, 0.8f, 1.0f } },

             { { -0.8f, 0.4f, -1.0f, 0.0f },
               { 0.0f, 0.0f, 0.0f, 1.0f },
               { 0.5f, 0.5f, 0.5f, 1.0f },
               { 0.8f, 0.8f, 0.8f, 1.0f } }},
        };
        D3D11_BUFFER_DESC cbDesc;
        ZeroMemory(&cbDesc, sizeof(cbDesc));
        cbDesc.Usage = D3D11_USAGE_DYNAMIC;
        cbDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        cbDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        cbDesc.MiscFlags = 0;
        cbDesc.ByteWidth = sizeof(lightingData);
        D3D11_SUBRESOURCE_DATA initData;
        initData.pSysMem = &lightingData;
        g_pd3dDevice->CreateBuffer(&cbDesc, &initData, &g_pcbLighting);
        assert(g_pcbLighting);
    }

    // create depth stencil state
    D3D11_DEPTH_STENCIL_DESC depthStencilDesc;
    ZeroMemory(&depthStencilDesc, sizeof(depthStencilDesc));
    depthStencilDesc.DepthEnable = true;
    depthStencilDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
    depthStencilDesc.DepthFunc = D3D11_COMPARISON_LESS_EQUAL;
    depthStencilDesc.StencilEnable = false;

    g_pd3dDevice->CreateDepthStencilState(&depthStencilDesc, &g_pDepthStencilState);
    assert(g_pDepthStencilState);

    // initialize control mesh display
    g_controlMeshDisplay = new D3D11ControlMeshDisplay(g_pd3dDeviceContext);

    return true;
}

static bool
updateRenderTarget(HWND hWnd) {

    RECT rc;
    GetClientRect(hWnd, &rc);
    UINT width = rc.right - rc.left;
    UINT height = rc.bottom - rc.top;

    if (g_pSwapChainRTV && (g_width == width) && (g_height == height)) {
        return true;
    }
    g_width = width;
    g_height = height;

    g_hud->Rebuild(g_width, g_height);

    SAFE_RELEASE(g_pSwapChainRTV);

    g_pSwapChain->ResizeBuffers(0, g_width, g_height, DXGI_FORMAT_UNKNOWN, 0);

    // get backbuffer of swap chain
    ID3D11Texture2D* hpBackBuffer = NULL;
    if(FAILED(g_pSwapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), (void**)&hpBackBuffer))){
        MessageBoxW(hWnd, L"SwpChain GetBuffer", L"Err", MB_ICONSTOP);
        return false;
    }

    // create render target from the back buffer
    if(FAILED(g_pd3dDevice->CreateRenderTargetView(hpBackBuffer, NULL, &g_pSwapChainRTV))){
        MessageBoxW(hWnd, L"CreateRenderTargetView", L"Err", MB_ICONSTOP);
        return false;
    }
    SAFE_RELEASE(hpBackBuffer);

    // create depth buffer
    D3D11_TEXTURE2D_DESC depthBufferDesc;
    ZeroMemory(&depthBufferDesc, sizeof(depthBufferDesc));
    depthBufferDesc.Width = g_width;
    depthBufferDesc.Height = g_height;
    depthBufferDesc.MipLevels = 1;
    depthBufferDesc.ArraySize = 1;
    depthBufferDesc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
    depthBufferDesc.SampleDesc.Count = 1;
    depthBufferDesc.SampleDesc.Quality = 0;
    depthBufferDesc.Usage = D3D11_USAGE_DEFAULT;
    depthBufferDesc.BindFlags = D3D11_BIND_DEPTH_STENCIL;
    depthBufferDesc.CPUAccessFlags = 0;
    depthBufferDesc.MiscFlags = 0;

    g_pd3dDevice->CreateTexture2D(&depthBufferDesc, NULL, &g_pDepthStencilBuffer);
    assert(g_pDepthStencilBuffer);

    D3D11_DEPTH_STENCIL_VIEW_DESC depthStencilViewDesc;
    ZeroMemory(&depthStencilViewDesc, sizeof(depthStencilViewDesc));
    depthStencilViewDesc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
    depthStencilViewDesc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
    depthStencilViewDesc.Texture2D.MipSlice = 0;

    g_pd3dDevice->CreateDepthStencilView(g_pDepthStencilBuffer, &depthStencilViewDesc, &g_pDepthStencilView);
    assert(g_pDepthStencilView);

    // set device context to the render target
    g_pd3dDeviceContext->OMSetRenderTargets(1, &g_pSwapChainRTV, g_pDepthStencilView);

    // init viewport
    D3D11_VIEWPORT vp;
    vp.TopLeftX = 0;
    vp.TopLeftY = 0;
    vp.Width = (float)g_width;
    vp.Height = (float)g_height;
    vp.MinDepth = 0.0f;
    vp.MaxDepth = 1.0f;
    g_pd3dDeviceContext->RSSetViewports(1, &vp);

    return true;
}

//------------------------------------------------------------------------------
static void
callbackError(OpenSubdiv::Far::ErrorType err, const char *message) {

    std::ostringstream s;
    s << "Error: " << err << "\n";
    s << message;
    OutputDebugString(s.str().c_str());
}

//------------------------------------------------------------------------------
static LRESULT WINAPI
msgProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {

    switch(msg)
    {
        case WM_KEYDOWN:
            keyboard(MapVirtualKey(UINT(wParam), MAPVK_VK_TO_CHAR));
            break;
        case WM_DESTROY:
            quit();
            return 0;
        case WM_MOUSEMOVE:
            motion(LOWORD(lParam), HIWORD(lParam));
            return 0;
        case WM_LBUTTONDOWN:
            mouse(0, 1, LOWORD(lParam), HIWORD(lParam));
            return 0;
        case WM_LBUTTONUP:
            mouse(0, 0, LOWORD(lParam), HIWORD(lParam));
            return 0;
        case WM_MBUTTONDOWN:
            mouse(1, 1, LOWORD(lParam), HIWORD(lParam));
            return 0;
        case WM_MBUTTONUP:
            mouse(1, 0, LOWORD(lParam), HIWORD(lParam));
            return 0;
        case WM_RBUTTONDOWN:
            mouse(2, 1, LOWORD(lParam), HIWORD(lParam));
            return 0;
        case WM_RBUTTONUP:
            mouse(2, 0, LOWORD(lParam), HIWORD(lParam));
            return 0;
        case WM_PAINT:
            ValidateRect(hWnd, NULL);
            return 0;
    }
    return DefWindowProc(hWnd, msg, wParam, lParam);
}

int WINAPI
WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPTSTR lpCmdLine, 
        int nCmdShow) 
{
    // register window class
    TCHAR szWindowClass[] = "OPENSUBDIV_EXAMPLE";
    WNDCLASS wcex;
    wcex.style          = CS_HREDRAW | CS_VREDRAW;
    wcex.lpfnWndProc    = msgProc;
    wcex.cbClsExtra     = 0;
    wcex.cbWndExtra     = 0;
    wcex.hInstance      = hInstance;
    wcex.hIcon          = NULL;
    wcex.hCursor        = LoadCursor(NULL, IDC_ARROW);
    wcex.hbrBackground  = (HBRUSH)(COLOR_WINDOW+1);
    wcex.lpszMenuName   = NULL;
    wcex.lpszClassName  = szWindowClass;
    RegisterClass(&wcex);

    // create window
    RECT rect = { 0, 0, g_width, g_height };
    AdjustWindowRect(&rect, WS_OVERLAPPEDWINDOW, FALSE);

    static const char windowTitle[] = 
        "OpenSubdiv dxViewer " OPENSUBDIV_VERSION_STRING;

    HWND hWnd = CreateWindow(szWindowClass,
                        windowTitle,
                        WS_OVERLAPPEDWINDOW | WS_VISIBLE,
                        CW_USEDEFAULT,
                        CW_USEDEFAULT,
                        rect.right - rect.left,
                        rect.bottom - rect.top,
                        NULL,
                        NULL,
                        hInstance,
                        NULL);

    // Parse the command line arguments
    ArgOptions args;
    args.Parse(__argc, __argv);
    const std::vector<const char *> &rargs = args.GetRemainingArgs();
    for (size_t i = 0; i < rargs.size(); ++i) {
        if (!strcmp(rargs[i], "-lg")) {
            g_legacyGregoryEnabled = true;
        } else {
            args.PrintUnrecognizedArgWarning(rargs[i]);
        }
    }

    g_yup = args.GetYUp();
    g_adaptive = args.GetAdaptive();
    g_level = args.GetLevel();
    g_repeatCount = args.GetRepeatCount();

    ViewerArgsUtils::PopulateShapes(args, &g_defaultShapes);

    initShapes();

    OpenSubdiv::Far::SetErrorCallback(callbackError);

    initD3D11(hWnd);

    initHUD();

    // main loop
    while (g_bDone == false) {
        MSG msg;
        ZeroMemory(&msg, sizeof(msg));
        while (msg.message != WM_QUIT) {
            while (PeekMessage(&msg, NULL, 0U, 0U, PM_REMOVE)) {
                if (msg.message == WM_QUIT) goto end;
                TranslateMessage(&msg);
                DispatchMessage(&msg);
            }
            if (! g_freeze)
                g_frame++;

            updateGeom();
            updateRenderTarget(hWnd);
            display();
        }
    }
    end:

    quit();
}

//------------------------------------------------------------------------------
