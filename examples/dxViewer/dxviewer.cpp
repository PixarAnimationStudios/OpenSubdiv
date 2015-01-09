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

#include <osd/vertex.h>
#include <osd/d3d11DrawContext.h>
#include <osd/d3d11DrawRegistry.h>
#include <far/error.h>

#include <osd/cpuD3D11VertexBuffer.h>
#include <osd/cpuComputeContext.h>
#include <osd/cpuComputeController.h>
OpenSubdiv::Osd::CpuComputeController * g_cpuComputeController = NULL;

#ifdef OPENSUBDIV_HAS_OPENMP
    #include <osd/ompComputeController.h>
    OpenSubdiv::Osd::OmpComputeController * g_ompComputeController = NULL;
#endif

#ifdef OPENSUBDIV_HAS_TBB
    #include <osd/tbbComputeController.h>
    OpenSubdiv::Osd::TbbComputeController *g_tbbComputeController = NULL;
#endif

#undef OPENSUBDIV_HAS_OPENCL    // XXX: dyu OpenCL D3D11 interop needs work...
#ifdef OPENSUBDIV_HAS_OPENCL
    #include <osd/clD3D11VertexBuffer.h>
    #include <osd/clComputeContext.h>
    #include <osd/clComputeController.h>

    #include "../common/clInit.h"

    cl_context g_clContext;
    cl_command_queue g_clQueue;
    OpenSubdiv::Osd::CLComputeController * g_clComputeController = NULL;
#endif

#ifdef OPENSUBDIV_HAS_CUDA
    #include <osd/cudaD3D11VertexBuffer.h>
    #include <osd/cudaComputeContext.h>
    #include <osd/cudaComputeController.h>

    #include <cuda_runtime_api.h>
    #include <cuda_d3d11_interop.h>

    bool g_cudaInitialized = false;
    OpenSubdiv::Osd::CudaComputeController * g_cudaComputeController = NULL;
#endif

#include <osd/d3d11VertexBuffer.h>
#include <osd/d3d11ComputeContext.h>
#include <osd/d3d11ComputeController.h>
OpenSubdiv::Osd::D3D11ComputeController * g_d3d11ComputeController = NULL;

#include <osd/d3d11Mesh.h>
OpenSubdiv::Osd::D3D11MeshInterface *g_mesh;

#include <common/vtr_utils.h>
#include "../common/stopwatch.h"
#include "../common/simple_math.h"
#include "../common/d3d11_hud.h"
#include "../common/patchColors.h"

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

enum DisplayStyle { kQuadWire = 0,
                    kQuadFill = 1,
                    kQuadLine = 2,
                    kTriWire = 3,
                    kTriFill = 4,
                    kTriLine = 5,
                    kPoint = 6 };

enum HudCheckBox { kHUD_CB_DISPLAY_CAGE_EDGES,
                   kHUD_CB_DISPLAY_CAGE_VERTS,
                   kHUD_CB_ANIMATE_VERTICES,
                   kHUD_CB_DISPLAY_PATCH_COLOR,
                   kHUD_CB_DISPLAY_PATCH_CVs,
                   kHUD_CB_VIEW_LOD,
                   kHUD_CB_FRACTIONAL_SPACING,
                   kHUD_CB_PATCH_CULL,
                   kHUD_CB_FREEZE };

int g_currentShape = 0;

int   g_frame = 0,
      g_repeatCount = 0;

// GUI variables
int   g_freeze = 0,
      g_wire = 2,
      g_adaptive = 1,
      g_drawCageEdges = 1,
      g_drawCageVertices = 0,
      g_drawPatchCVs = 0,
      g_drawNormals = 0,
      g_mbutton[3] = {0, 0, 0};

int   g_displayPatchColor = 1,
      g_screenSpaceTess = 0,
      g_fractionalSpacing = 0,
      g_patchCull = 0;

float g_rotate[2] = {0, 0},
      g_prev_x = 0,
      g_prev_y = 0,
      g_dolly = 5,
      g_pan[2] = {0, 0},
      g_center[3] = {0, 0, 0},
      g_size = 0;

int   g_width = 1024,
      g_height = 1024;

D3D11hud *g_hud = NULL;

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

std::vector<int> g_coarseEdges;
std::vector<float> g_coarseEdgeSharpness;
std::vector<float> g_coarseVertexSharpness;

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

    typedef OpenSubdiv::Far::ConstIndexArray IndexArray;

    Shape * shape = Shape::parseObj(shapeDesc.data.c_str(), shapeDesc.scheme);

    // create Vtr mesh (topology)
    OpenSubdiv::Sdc::SchemeType sdctype = GetSdcType(*shape);
    OpenSubdiv::Sdc::Options sdcoptions = GetSdcOptions(*shape);

    OpenSubdiv::Far::TopologyRefiner * refiner =
        OpenSubdiv::Far::TopologyRefinerFactory<Shape>::Create(*shape,
            OpenSubdiv::Far::TopologyRefinerFactory<Shape>::Options(sdctype, sdcoptions));

    // save coarse topology (used for coarse mesh drawing)
    int nedges = refiner->GetNumEdges(0),
        nverts = refiner->GetNumVertices(0);

    g_coarseEdges.resize(nedges*2);
    g_coarseEdgeSharpness.resize(nedges);
    g_coarseVertexSharpness.resize(nverts);

    for(int i=0; i<nedges; ++i) {
        IndexArray verts = refiner->GetEdgeVertices(0, i);
        g_coarseEdges[i*2  ]=verts[0];
        g_coarseEdges[i*2+1]=verts[1];
        g_coarseEdgeSharpness[i]=refiner->GetEdgeSharpness(0, i);
    }

    for(int i=0; i<nverts; ++i) {
        g_coarseVertexSharpness[i]=refiner->GetVertexSharpness(0, i);
    }

    g_orgPositions=shape->verts;

    g_positions.resize(g_orgPositions.size(),0.0f);

    delete g_mesh;
    g_mesh = NULL;

    g_scheme = scheme;

    // Adaptive refinement currently supported only for catmull-clark scheme
    bool doAdaptive = (g_adaptive!=0 and g_scheme==kCatmark);

    OpenSubdiv::Osd::MeshBitset bits;
    bits.set(OpenSubdiv::Osd::MeshAdaptive, doAdaptive);

    int numVertexElements = 6;
    int numVaryingElements = 0;

    if (g_kernel == kCPU) {
        if (not g_cpuComputeController) {
            g_cpuComputeController = new OpenSubdiv::Osd::CpuComputeController();
        }
        g_mesh = new OpenSubdiv::Osd::Mesh<OpenSubdiv::Osd::CpuD3D11VertexBuffer,
                                         OpenSubdiv::Osd::CpuComputeController,
                                         OpenSubdiv::Osd::D3D11DrawContext>(
                                                g_cpuComputeController,
                                                refiner,
                                                numVertexElements,
                                                numVaryingElements,
                                                level, bits, g_pd3dDeviceContext);
#ifdef OPENSUBDIV_HAS_OPENMP
    } else if (kernel == kOPENMP) {
        if (not g_ompComputeController) {
            g_ompComputeController = new OpenSubdiv::Osd::OmpComputeController();
        }
        g_mesh = new OpenSubdiv::Osd::Mesh<OpenSubdiv::Osd::CpuD3D11VertexBuffer,
                                         OpenSubdiv::Osd::OmpComputeController,
                                         OpenSubdiv::Osd::D3D11DrawContext>(
                                                g_ompComputeController,
                                                refiner,
                                                numVertexElements,
                                                numVaryingElements,
                                                level, bits, g_pd3dDeviceContext);
#endif
#ifdef OPENSUBDIV_HAS_TBB
    } else if (kernel == kTBB) {
        if (not g_tbbComputeController) {
            g_tbbComputeController = new OpenSubdiv::Osd::TbbComputeController();
        }
        g_mesh = new OpenSubdiv::Osd::Mesh<OpenSubdiv::Osd::CpuD3D11VertexBuffer,
                                         OpenSubdiv::Osd::TbbComputeController,
                                         OpenSubdiv::Osd::D3D11DrawContext>(
                                                g_tbbComputeController,
                                                refiner,
                                                numVertexElements,
                                                numVaryingElements,
                                                level, bits, g_pd3dDeviceContext);
#endif
#ifdef OPENSUBDIV_HAS_OPENCL
    } else if(kernel == kCL) {
        if (not g_clComputeController) {
            g_clComputeController = new OpenSubdiv::Osd::CLComputeController(g_clContext, g_clQueue);
        }
        g_mesh = new OpenSubdiv::Osd::Mesh<OpenSubdiv::Osd::CLD3D11VertexBuffer,
                                         OpenSubdiv::Osd::CLComputeController,
                                         OpenSubdiv::Osd::D3D11DrawContext>(
                                                g_clComputeController,
                                                refiner,
                                                numVertexElements,
                                                numVaryingElements,
                                                level, bits, g_clContext, g_clQueue, g_pd3dDeviceContext);
#endif
#ifdef OPENSUBDIV_HAS_CUDA
    } else if (g_kernel == kCUDA) {
        if (not g_cudaComputeController) {
            g_cudaComputeController = new OpenSubdiv::Osd::CudaComputeController();
        }
        g_mesh = new OpenSubdiv::Osd::Mesh<OpenSubdiv::Osd::CudaD3D11VertexBuffer,
                                         OpenSubdiv::Osd::CudaComputeController,
                                         OpenSubdiv::Osd::D3D11DrawContext>(
                                                g_cudaComputeController,
                                                refiner,
                                                numVertexElements,
                                                numVaryingElements,
                                                level, bits, g_pd3dDeviceContext);
#endif
    } else if (g_kernel == kDirectCompute) {
        if (not g_d3d11ComputeController) {
            g_d3d11ComputeController = new OpenSubdiv::Osd::D3D11ComputeController(g_pd3dDeviceContext);
        }
        g_mesh = new OpenSubdiv::Osd::Mesh<OpenSubdiv::Osd::D3D11VertexBuffer,
                                         OpenSubdiv::Osd::D3D11ComputeController,
                                         OpenSubdiv::Osd::D3D11DrawContext>(
                                                g_d3d11ComputeController,
                                                refiner,
                                                numVertexElements,
                                                numVaryingElements,
                                                level, bits, g_pd3dDeviceContext);
    } else {
        printf("Unsupported kernel %s\n", getKernelName(kernel));
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
    Effect(int displayStyle_, int screenSpaceTess_, int fractionalSpacing_, int patchCull_) : value(0) {
        displayStyle = displayStyle_;
        screenSpaceTess = screenSpaceTess_;
        fractionalSpacing = fractionalSpacing_;
        patchCull = patchCull_;
    }

    struct {
        unsigned int displayStyle:3;
        unsigned int screenSpaceTess:1;
        unsigned int fractionalSpacing:1;
        unsigned int patchCull:1;
    };
    int value;

    bool operator < (const Effect &e) const {
        return value < e.value;
    }
};


typedef std::pair<OpenSubdiv::Osd::DrawContext::PatchDescriptor, Effect> EffectDesc;

class EffectDrawRegistry : public OpenSubdiv::Osd::D3D11DrawRegistry<EffectDesc> {

protected:
    virtual ConfigType *
    _CreateDrawConfig(DescType const & desc,
                      SourceConfigType const * sconfig,
                      ID3D11Device * pd3dDevice,
                      ID3D11InputLayout ** ppInputLayout,
                      D3D11_INPUT_ELEMENT_DESC const * pInputElementDescs,
                      int numInputElements);

    virtual SourceConfigType *
    _CreateDrawSourceConfig(DescType const & desc, ID3D11Device * pd3dDevice);
};

EffectDrawRegistry::SourceConfigType *
EffectDrawRegistry::_CreateDrawSourceConfig(
        DescType const & desc, ID3D11Device * pd3dDevice) {

    Effect effect = desc.second;

    SourceConfigType * sconfig =
        BaseRegistry::_CreateDrawSourceConfig(desc.first, pd3dDevice);

    sconfig->commonShader.AddDefine("OSD_ENABLE_PATCH_CULL");
    sconfig->commonShader.AddDefine("OSD_ENABLE_SCREENSPACE_TESSELLATION");

    bool smoothNormals = false;
    if (desc.first.GetType() == OpenSubdiv::Far::PatchDescriptor::QUADS ||
        desc.first.GetType() == OpenSubdiv::Far::PatchDescriptor::TRIANGLES) {
        sconfig->vertexShader.source = shaderSource;
        sconfig->vertexShader.target = "vs_5_0";
        sconfig->vertexShader.entry = "vs_main";
    } else if (desc.first.GetType() == OpenSubdiv::Far::PatchDescriptor::TRIANGLES) {
        if (effect.displayStyle == kQuadWire) effect.displayStyle = kTriWire;
        if (effect.displayStyle == kQuadFill) effect.displayStyle = kTriFill;
        if (effect.displayStyle == kQuadLine) effect.displayStyle = kTriLine;
        smoothNormals = true;
    } else {
        // adaptive
        if (effect.displayStyle == kQuadWire) effect.displayStyle = kTriWire;
        if (effect.displayStyle == kQuadFill) effect.displayStyle = kTriFill;
        if (effect.displayStyle == kQuadLine) effect.displayStyle = kTriLine;
        smoothNormals = true;
        sconfig->vertexShader.source = shaderSource + sconfig->vertexShader.source;
        sconfig->hullShader.source = shaderSource + sconfig->hullShader.source;
        sconfig->domainShader.source = shaderSource + sconfig->domainShader.source;
    }
    assert(sconfig);

    sconfig->geometryShader.source = shaderSource;
    sconfig->geometryShader.target = "gs_5_0";

    sconfig->pixelShader.source = shaderSource;
    sconfig->pixelShader.target = "ps_5_0";

    if (effect.screenSpaceTess) {
        sconfig->commonShader.AddDefine("OSD_ENABLE_SCREENSPACE_TESSELLATION");
    }
    if (effect.fractionalSpacing) {
        sconfig->commonShader.AddDefine("OSD_FRACTIONAL_ODD_SPACING");
    }
    if (effect.patchCull) {
        sconfig->commonShader.AddDefine("OSD_ENABLE_PATCH_CULL");
    }


    switch (effect.displayStyle) {
        case kQuadWire:
            sconfig->geometryShader.entry = "gs_quad_wire";
            sconfig->geometryShader.AddDefine("PRIM_QUAD");
            sconfig->geometryShader.AddDefine("GEOMETRY_OUT_WIRE");
            sconfig->pixelShader.entry = "ps_main";
            sconfig->pixelShader.AddDefine("PRIM_QUAD");
            sconfig->pixelShader.AddDefine("GEOMETRY_OUT_WIRE");
            break;
        case kQuadFill:
            sconfig->geometryShader.entry = "gs_quad";
            sconfig->geometryShader.AddDefine("PRIM_QUAD");
            sconfig->geometryShader.AddDefine("GEOMETRY_OUT_FILL");
            sconfig->pixelShader.entry = "ps_main";
            sconfig->pixelShader.AddDefine("PRIM_QUAD");
            sconfig->pixelShader.AddDefine("GEOMETRY_OUT_FILL");
            break;
        case kQuadLine:
            sconfig->geometryShader.entry = "gs_quad_wire";
            sconfig->geometryShader.AddDefine("PRIM_QUAD");
            sconfig->geometryShader.AddDefine("GEOMETRY_OUT_LINE");
            sconfig->pixelShader.entry = "ps_main";
            sconfig->pixelShader.AddDefine("PRIM_QUAD");
            sconfig->pixelShader.AddDefine("GEOMETRY_OUT_LINE");
            break;
        case kTriWire:
            sconfig->geometryShader.entry =
                smoothNormals ? "gs_triangle_smooth_wire" : "gs_triangle_wire";
            sconfig->geometryShader.AddDefine("PRIM_TRI");
            sconfig->geometryShader.AddDefine("GEOMETRY_OUT_WIRE");
            sconfig->pixelShader.entry = "ps_main";
            sconfig->pixelShader.AddDefine("PRIM_TRI");
            sconfig->pixelShader.AddDefine("GEOMETRY_OUT_WIRE");
            break;
        case kTriFill:
            sconfig->geometryShader.entry =
                smoothNormals ? "gs_triangle_smooth" : "gs_triangle";
            sconfig->geometryShader.AddDefine("PRIM_TRI");
            sconfig->geometryShader.AddDefine("GEOMETRY_OUT_FILL");
            sconfig->pixelShader.entry = "ps_main";
            sconfig->pixelShader.AddDefine("PRIM_TRI");
            sconfig->pixelShader.AddDefine("GEOMETRY_OUT_FILL");
            break;
        case kTriLine:
            sconfig->geometryShader.entry =
                smoothNormals ? "gs_triangle_smooth_wire" : "gs_triangle_wire";
            sconfig->geometryShader.AddDefine("PRIM_TRI");
            sconfig->geometryShader.AddDefine("GEOMETRY_OUT_LINE");
            sconfig->pixelShader.entry = "ps_main";
            sconfig->pixelShader.AddDefine("PRIM_TRI");
            sconfig->pixelShader.AddDefine("GEOMETRY_OUT_LINE");
            break;
        case kPoint:
            sconfig->geometryShader.entry = "gs_point";
            sconfig->pixelShader.entry = "ps_main_point";
            break;
    }

    return sconfig;
}

EffectDrawRegistry::ConfigType *
EffectDrawRegistry::_CreateDrawConfig(
        DescType const & desc,
        SourceConfigType const * sconfig,
        ID3D11Device * pd3dDevice,
        ID3D11InputLayout ** ppInputLayout,
        D3D11_INPUT_ELEMENT_DESC const * pInputElementDescs,
        int numInputElements) {

    ConfigType * config = BaseRegistry::_CreateDrawConfig(desc.first, sconfig,
        pd3dDevice, ppInputLayout, pInputElementDescs, numInputElements);
    assert(config);

    return config;
}

EffectDrawRegistry effectRegistry;

static Effect
GetEffect() {

   DisplayStyle style;

    if (g_scheme == kLoop) {
        style = (g_wire == 0 ? kTriWire : (g_wire == 1 ? kTriFill : kTriLine));
    } else {
        style = (g_wire == 0 ? style=kQuadWire : (g_wire == 1 ? kQuadFill : kQuadLine));
    }
    return Effect(style, g_screenSpaceTess, g_fractionalSpacing, g_patchCull);
}

//------------------------------------------------------------------------------
static void
bindProgram(Effect effect, OpenSubdiv::Osd::DrawContext::PatchArray const & patch) {

    EffectDesc effectDesc(patch.GetDescriptor(), effect);

    // input layout
    const D3D11_INPUT_ELEMENT_DESC hInElementDesc[] = {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 4*3, D3D11_INPUT_PER_VERTEX_DATA, 0 }
    };

    EffectDrawRegistry::ConfigType *
        config = effectRegistry.GetDrawConfig(
                effectDesc, g_pd3dDevice,
                &g_pInputLayout, hInElementDesc, ARRAYSIZE(hInElementDesc));

    assert(g_pInputLayout);

    // Update transform state
    {
        __declspec(align(16))
        struct CB_PER_FRAME_CONSTANTS
        {
            float ModelViewMatrix[16];
            float ProjectionMatrix[16];
            float ModelViewProjectionMatrix[16];
        };

        if (not g_pcbPerFrame) {
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
        rotate(pData->ModelViewMatrix, -90, 1, 0, 0); // z-up model

        identity(pData->ProjectionMatrix);
        perspective(pData->ProjectionMatrix, 45.0, aspect, 0.01f, 500.0);
        multMatrix(pData->ModelViewProjectionMatrix, pData->ModelViewMatrix, pData->ProjectionMatrix);

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

        if (not g_pcbTessellation) {
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
        pData->GregoryQuadOffsetBase = patch.GetQuadOffsetIndex();
        pData->PrimitiveIdBase = patch.GetPatchIndex();

        g_pd3dDeviceContext->Unmap( g_pcbTessellation, 0 );
    }

    // Update material state
    {
        __declspec(align(16))
        struct Material {
            float color[4];
        };

        if (not g_pcbMaterial) {
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

        float const * patchColor;
        if (g_displayPatchColor and g_mesh->GetDrawContext()->IsAdaptive()) {
            patchColor = getAdaptivePatchColor( patch.GetDescriptor() );
        } else {
            static float const uniformColor[4] = {0.13f, 0.13f, 0.61f, 1.0f};
            patchColor = uniformColor;
        }
        memcpy(pData->color, patchColor, 4*sizeof(float));

        g_pd3dDeviceContext->Unmap( g_pcbMaterial, 0 );
    }

    g_pd3dDeviceContext->IASetInputLayout(g_pInputLayout);

    g_pd3dDeviceContext->VSSetShader(config->vertexShader, NULL, 0);
    g_pd3dDeviceContext->VSSetConstantBuffers(0, 1, &g_pcbPerFrame);

    g_pd3dDeviceContext->HSSetShader(config->hullShader, NULL, 0);
    g_pd3dDeviceContext->HSSetConstantBuffers(0, 1, &g_pcbPerFrame);
    g_pd3dDeviceContext->HSSetConstantBuffers(1, 1, &g_pcbTessellation);

    g_pd3dDeviceContext->DSSetShader(config->domainShader, NULL, 0);
    g_pd3dDeviceContext->DSSetConstantBuffers(0, 1, &g_pcbPerFrame);

    g_pd3dDeviceContext->GSSetShader(config->geometryShader, NULL, 0);
    g_pd3dDeviceContext->GSSetConstantBuffers(0, 1, &g_pcbPerFrame);

    g_pd3dDeviceContext->PSSetShader(config->pixelShader, NULL, 0);
    g_pd3dDeviceContext->PSSetConstantBuffers(0, 1, &g_pcbPerFrame);
    g_pd3dDeviceContext->PSSetConstantBuffers(2, 1, &g_pcbLighting);
    g_pd3dDeviceContext->PSSetConstantBuffers(3, 1, &g_pcbMaterial);

    if (g_mesh->GetDrawContext()->vertexBufferSRV) {
        g_pd3dDeviceContext->VSSetShaderResources(0, 1, &g_mesh->GetDrawContext()->vertexBufferSRV);
    }
    if (g_mesh->GetDrawContext()->vertexValenceBufferSRV) {
        g_pd3dDeviceContext->VSSetShaderResources(1, 1, &g_mesh->GetDrawContext()->vertexValenceBufferSRV);
    }
    if (g_mesh->GetDrawContext()->quadOffsetBufferSRV) {
        g_pd3dDeviceContext->HSSetShaderResources(2, 1, &g_mesh->GetDrawContext()->quadOffsetBufferSRV);
    }
    if (g_mesh->GetDrawContext()->ptexCoordinateBufferSRV) {
        g_pd3dDeviceContext->HSSetShaderResources(3, 1, &g_mesh->GetDrawContext()->ptexCoordinateBufferSRV);
        g_pd3dDeviceContext->DSSetShaderResources(3, 1, &g_mesh->GetDrawContext()->ptexCoordinateBufferSRV);
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

    UINT hStrides = 6*sizeof(float);
    UINT hOffsets = 0;
    g_pd3dDeviceContext->IASetVertexBuffers(0, 1, &buffer, &hStrides, &hOffsets);

    OpenSubdiv::Osd::DrawContext::PatchArrayVector const & patches = g_mesh->GetDrawContext()->GetPatchArrays();

    g_pd3dDeviceContext->IASetIndexBuffer(g_mesh->GetDrawContext()->patchIndexBuffer, DXGI_FORMAT_R32_UINT, 0);

    // cv drawing
#if 0

    if (g_drawPatchCVs) {

        bindProgram(kPoint, OpenSubdiv::Osd::DrawContext::PatchArray());

        g_pd3dDeviceContext->IASetPrimitiveTopology(
                                D3D11_PRIMITIVE_TOPOLOGY_POINTLIST);

        for (int i=0; i<(int)patches.size(); ++i) {
            OpenSubdiv::Osd::DrawContext::PatchArray const & patch = patches[i];

            g_pd3dDeviceContext->DrawIndexed(patch.GetNumIndices(),
                                             patch.GetVertIndex(), 0);
        }
    }
#endif

    // patch drawing
    for (int i=0; i<(int)patches.size(); ++i) {
        OpenSubdiv::Osd::DrawContext::PatchArray const & patch = patches[i];

        D3D11_PRIMITIVE_TOPOLOGY topology;

        if (g_mesh->GetDrawContext()->IsAdaptive()) {

            OpenSubdiv::Osd::DrawContext::PatchDescriptor desc = patch.GetDescriptor();

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
            case 16:
                topology = D3D11_PRIMITIVE_TOPOLOGY_16_CONTROL_POINT_PATCHLIST;
                break;
            default:
                assert(false);
                break;
            }
        } else {

            if (g_scheme == kLoop) {
                topology = D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
            } else {
                topology = D3D11_PRIMITIVE_TOPOLOGY_LINELIST_ADJ;
            }
        }

        bindProgram(GetEffect(), patch);

        g_pd3dDeviceContext->IASetPrimitiveTopology(topology);

        g_pd3dDeviceContext->DrawIndexed(patch.GetNumIndices(), patch.GetVertIndex(), 0);
    }

    if (g_hud->IsVisible()) {
        g_fpsTimer.Stop();
        double fps = 1.0/g_fpsTimer.GetElapsed();
        g_fpsTimer.Start();

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
    } else if ((g_mbutton[0] && g_mbutton[1] && !g_mbutton[2]) or
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

    delete g_cpuComputeController;

#ifdef OPENSUBDIV_HAS_OPENMP
    delete g_ompComputeController;
#endif

#ifdef OPENSUBDIV_HAS_TBB
    delete g_tbbComputeController;
#endif

#ifdef OPENSUBDIV_HAS_OPENCL
    delete g_clComputeController;
    uninitCL(g_clContext, g_clQueue);
#endif

#ifdef OPENSUBDIV_HAS_CUDA
    delete g_cudaComputeController;
    cudaDeviceReset();
#endif

    delete g_d3d11ComputeController;

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
callbackWireframe(int b) {
    g_wire = b;
}

static void
callbackKernel(int k) {

    g_kernel = k;

#ifdef OPENSUBDIV_HAS_OPENCL
    if (g_kernel == kCL and g_clContext == NULL) {
        if (initCL(&g_clContext, &g_clQueue) == false) {
            printf("Error in initializing OpenCL\n");
            exit(1);
        }
    }
#endif
#ifdef OPENSUBDIV_HAS_CUDA
    if (g_kernel == kCUDA and g_cudaInitialized == false) {
        g_cudaInitialized = true;
        cudaD3D11SetDirect3DDevice( g_pd3dDevice );
    }
#endif

    createOsdMesh(g_defaultShapes[g_currentShape], g_level, g_kernel, g_defaultShapes[ g_currentShape ].scheme);
}

static void
callbackLevel(int l) {
    g_level = l;
    createOsdMesh(g_defaultShapes[g_currentShape], g_level, g_kernel, g_defaultShapes[ g_currentShape ].scheme);
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

    createOsdMesh(g_defaultShapes[g_currentShape], g_level, g_kernel, g_defaultShapes[ g_currentShape ].scheme);
}

static void
callbackDisplayNormal(bool checked, int n) {
    g_drawNormals = checked;
}

static void
callbackAnimate(bool checked, int m) {
    g_moveScale = checked;
}

static void
callbackFreeze(bool checked, int f) {
    g_freeze = checked;
}

static void
callbackAdaptive(bool checked, int a) {
    g_adaptive = checked;
    createOsdMesh(g_defaultShapes[g_currentShape], g_level, g_kernel, g_defaultShapes[ g_currentShape ].scheme);
}

static void
callbackCheckBox(bool checked, int button) {
    switch (button) {
    case kHUD_CB_DISPLAY_CAGE_EDGES:
        g_drawCageEdges = checked;
        break;
    case kHUD_CB_DISPLAY_CAGE_VERTS:
        g_drawCageVertices = checked;
        break;
    case kHUD_CB_ANIMATE_VERTICES:
        g_moveScale = checked;
        break;
    case kHUD_CB_DISPLAY_PATCH_COLOR:
        g_displayPatchColor = checked;
        break;
    case kHUD_CB_DISPLAY_PATCH_CVs:
        g_drawPatchCVs = checked;
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
#ifdef OPENSUBDIV_HAS_OPENCL
    if (HAS_CL_VERSION_1_1()) {
        g_hud->AddPullDownButton(compute_pulldown, "OpenCL", kCL);
    }
#endif
    g_hud->AddPullDownButton(compute_pulldown, "HLSL Compute", kDirectCompute);

    int shading_pulldown = g_hud->AddPullDown("Shading (W)", 200, 10, 250, callbackWireframe, 'W');
    g_hud->AddPullDownButton(shading_pulldown, "Wire",        0, g_wire==0);
    g_hud->AddPullDownButton(shading_pulldown, "Shaded",      1, g_wire==1);
    g_hud->AddPullDownButton(shading_pulldown, "Wire+Shaded", 2, g_wire==2);

//    g_hud->AddCheckBox("Cage Edges (H)",         true,  10, 10, callbackDisplayCageEdges, 0, 'H');
//    g_hud->AddCheckBox("Cage Verts (J)",         false, 10, 30, callbackDisplayCageVertices, 0, 'J');
//    g_hud->AddCheckBox("Show normal vector (E)", false, 10, 10, callbackDisplayNormal, 0, 'E');

    g_hud->AddCheckBox("Patch CVs (L)",             false,                    10, 10,  callbackCheckBox, kHUD_CB_DISPLAY_PATCH_CVs, 'L');
    g_hud->AddCheckBox("Patch Color (P)",           true,                     10, 30,  callbackCheckBox, kHUD_CB_DISPLAY_PATCH_COLOR, 'P');
    g_hud->AddCheckBox("Animate vertices (M)",      g_moveScale != 0,         10, 50,  callbackCheckBox, kHUD_CB_ANIMATE_VERTICES, 'M');
    g_hud->AddCheckBox("Freeze (spc)",              false,                    10, 70,  callbackCheckBox, kHUD_CB_FREEZE, ' ');
    g_hud->AddCheckBox("Screen space LOD (V)",      g_screenSpaceTess != 0,   10, 110,  callbackCheckBox, kHUD_CB_VIEW_LOD, 'V');
    g_hud->AddCheckBox("Fractional spacing (T)",    g_fractionalSpacing != 0, 10, 130, callbackCheckBox, kHUD_CB_FRACTIONAL_SPACING, 'T');
    g_hud->AddCheckBox("Frustum Patch Culling (B)", g_patchCull != 0,         10, 150, callbackCheckBox, kHUD_CB_PATCH_CULL, 'B');

    g_hud->AddCheckBox("Adaptive (`)", true, 10, 190, callbackAdaptive, 0, '`');

    for (int i = 1; i < 11; ++i) {
        char level[16];
        sprintf(level, "Lv. %d", i);
        g_hud->AddRadioButton(3, level, i==2, 10, 210+i*20, callbackLevel, i, '0'+(i%10));
    }

    int shapes_pulldown = g_hud->AddPullDown("Shape (N)", -300, 10, 300, callbackModel, 'n');
    for (int i = 0; i < (int)g_defaultShapes.size(); ++i) {
        g_hud->AddPullDownButton(shapes_pulldown, g_defaultShapes[i].name.c_str(),i);
    }

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
    rasterDesc.CullMode = D3D11_CULL_NONE; // XXX
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
            0.5, 0.2f, 1.0f, 0.0f,
            0.1f, 0.1f, 0.1f, 1.0f,
            0.7f, 0.7f, 0.7f, 1.0f,
            0.8f, 0.8f, 0.8f, 1.0f,

            -0.8f, 0.4f, -1.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f,
            0.5f, 0.5f, 0.5f, 1.0f,
            0.8f, 0.8f, 0.8f, 1.0f,
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

static std::vector<std::string>
tokenize(std::string const & src) {

    std::vector<std::string> result;

    std::stringstream input(src);
    std::copy(std::istream_iterator<std::string>(input),
              std::istream_iterator<std::string>(),
              std::back_inserter< std::vector<std::string> >(result));

    return result;
}

int WINAPI
WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPTSTR lpCmdLine, int nCmdShow) {

    // register window class
    TCHAR szWindowClass[] = "OPENSUBDIV_EXAMPLE";
    WNDCLASS wcex;
    wcex.style          = CS_HREDRAW | CS_VREDRAW;
    wcex.lpfnWndProc    = msgProc;
    wcex.cbClsExtra     = 0;
    wcex.cbWndExtra     = 0;
    wcex.hInstance      = hInstance;
    wcex.hIcon          = NULL;
    wcex.hCursor        = NULL;
    wcex.hbrBackground  = (HBRUSH)(COLOR_WINDOW+1);
    wcex.lpszMenuName   = NULL;
    wcex.lpszClassName  = szWindowClass;
    RegisterClass(&wcex);

    // crete window
    RECT rect = { 0, 0, g_width, g_height };
    AdjustWindowRect(&rect, WS_OVERLAPPEDWINDOW, FALSE);

    static const char windowTitle[] = "OpenSubdiv dxViewer " OPENSUBDIV_VERSION_STRING;

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

    std::vector<std::string> args = tokenize(lpCmdLine);
    for (int i=0; i<args.size(); ++i) {
        std::ifstream ifs(args[i]);
        if (ifs) {
            std::stringstream ss;
            ss << ifs.rdbuf();
            ifs.close();
            std::string str = ss.str();
            g_defaultShapes.push_back(ShapeDesc(__argv[1], str.c_str(), kCatmark));
        }
    }

    std::string str;
    for (int i = 1; i < __argc; ++i) {
        if (!strcmp(__argv[i], "-d"))
            g_level = atoi(__argv[++i]);
        else if (!strcmp(__argv[i], "-c"))
            g_repeatCount = atoi(__argv[++i]);
        else {
            std::ifstream ifs(__argv[1]);
            if (ifs) {
                std::stringstream ss;
                ss << ifs.rdbuf();
                ifs.close();
                str = ss.str();
                g_defaultShapes.push_back(ShapeDesc(__argv[1], str.c_str(), kCatmark));
            }
        }
    }

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
            if (not g_freeze)
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
