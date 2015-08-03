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

#include <far/error.h>

#include <osd/cpuD3D11VertexBuffer.h>
#include <osd/cpuEvaluator.h>

#ifdef OPENSUBDIV_HAS_OPENMP
    #include <osd/ompEvaluator.h>
#endif

#ifdef OPENSUBDIV_HAS_TBB
    #include <osd/tbbEvaluator.h>
#endif

#ifdef OPENSUBDIV_HAS_OPENCL
    #include <osd/clD3D11VertexBuffer.h>
    #include <osd/clEvaluator.h>
    #include "../common/clDeviceContext.h"
    CLD3D11DeviceContext g_clDeviceContext;
#endif

#ifdef OPENSUBDIV_HAS_CUDA
    #include <osd/cudaD3D11VertexBuffer.h>
    #include <osd/cudaEvaluator.h>
    #include "../common/cudaDeviceContext.h"
    CudaDeviceContext g_cudaDeviceContext;
#endif

#include <osd/d3d11VertexBuffer.h>
#include <osd/d3d11ComputeEvaluator.h>

#include <osd/d3d11Mesh.h>
OpenSubdiv::Osd::D3D11MeshInterface *g_mesh;

#include "Ptexture.h"
#include "PtexUtils.h"

#include "../../regression/common/far_utils.h"
#include "../common/stopwatch.h"
#include "../common/simple_math.h"
#include "../common/d3d11Hud.h"
#include "../common/d3d11PtexMipmapTexture.h"
#include "../common/d3d11ShaderCache.h"

#include <osd/hlslPatchShaderSource.h>
static const char *g_shaderSource =
#include "shader.gen.h"
;

#include <algorithm>
#include <cfloat>
#include <fstream>
#include <string>
#include <iostream>
#include <iterator>
#include <sstream>
#include <vector>

#define SAFE_RELEASE(p) { if(p) { (p)->Release(); (p)=NULL; } }

enum KernelType { kCPU = 0,
                  kOPENMP = 1,
                  kTBB = 2,
                  kCUDA = 3,
                  kCL = 4,
                  kDirectCompute = 5 };

enum HudCheckBox { HUD_CB_ADAPTIVE,
                   HUD_CB_DISPLAY_OCCLUSION,
                   HUD_CB_DISPLAY_NORMALMAP,
                   HUD_CB_DISPLAY_SPECULAR,
                   HUD_CB_ANIMATE_VERTICES,
                   HUD_CB_VIEW_LOD,
                   HUD_CB_FRACTIONAL_SPACING,
                   HUD_CB_PATCH_CULL,
                   HUD_CB_IBL,
                   HUD_CB_BLOOM,
                   HUD_CB_SEAMLESS_MIPMAP,
                   HUD_CB_FREEZE };

enum HudRadioGroup { HUD_RB_KERNEL,
                     HUD_RB_LEVEL,
                     HUD_RB_SCHEME,
                     HUD_RB_WIRE,
                     HUD_RB_COLOR,
                     HUD_RB_DISPLACEMENT,
                     HUD_RB_NORMAL };

enum DisplayType { DISPLAY_WIRE,
                   DISPLAY_SHADED,
                   DISPLAY_WIRE_ON_SHADED };

enum ColorType { COLOR_NONE,
                 COLOR_PTEX_NEAREST,
                 COLOR_PTEX_HW_BILINEAR,
                 COLOR_PTEX_BILINEAR,
                 COLOR_PTEX_BIQUADRATIC,
                 COLOR_PATCHTYPE,
                 COLOR_PATCHCOORD,
                 COLOR_NORMAL };

enum DisplacementType { DISPLACEMENT_NONE,
                        DISPLACEMENT_HW_BILINEAR,
                        DISPLACEMENT_BILINEAR,
                        DISPLACEMENT_BIQUADRATIC };

enum NormalType { NORMAL_SURFACE,
                  NORMAL_FACET,
                  NORMAL_HW_SCREENSPACE,
                  NORMAL_SCREENSPACE,
                  NORMAL_BIQUADRATIC,
                  NORMAL_BIQUADRATIC_WG };

//-----------------------------------------------------------------------------
int   g_frame = 0,
      g_repeatCount = 0;

// GUI variables
int   g_fullscreen = 0,
      g_wire = DISPLAY_SHADED,
      g_drawNormals = 0,
      g_mbutton[3] = {0, 0, 0},
      g_level = 2,
      g_tessLevel = 2,
      g_kernel = kCPU,
      g_scheme = 0,
      g_running = 1,
      g_maxMipmapLevels = 10,
      g_color = COLOR_PTEX_BILINEAR,
      g_displacement = DISPLACEMENT_NONE,
      g_normal = NORMAL_SURFACE;


float g_moveScale = 0.0f,
      g_displacementScale = 1.0f,
      g_mipmapBias = 0.0;

bool  g_adaptive = true,
      g_yup = false,
      g_patchCull = true,
      g_screenSpaceTess = true,
      g_fractionalSpacing = true,
      g_ibl = false,
      g_bloom = false,
      g_freeze = false;

// ptex switch
bool  g_occlusion = false,
      g_specular = false;

bool g_seamless = true;

// camera
float g_rotate[2] = {0, 0},
      g_prev_x = 0,
      g_prev_y = 0,
      g_dolly = 5,
      g_pan[2] = {0, 0},
      g_center[3] = {0, 0, 0},
      g_size = 0;


// viewport
int   g_width = 1024,
      g_height = 1024;

D3D11hud *g_hud = NULL;

// performance
float g_cpuTime = 0;
float g_gpuTime = 0;
#define NUM_FPS_TIME_SAMPLES 6
float g_fpsTimeSamples[NUM_FPS_TIME_SAMPLES] = {0, 0, 0, 0, 0, 0};
int   g_currentFpsTimeSample = 0;
Stopwatch g_fpsTimer;
float g_animTime = 0;

// geometry
std::vector<float> g_positions,
                   g_normals;

D3D11PtexMipmapTexture * g_osdPTexImage = 0;
D3D11PtexMipmapTexture * g_osdPTexDisplacement = 0;
D3D11PtexMipmapTexture * g_osdPTexOcclusion = 0;
D3D11PtexMipmapTexture * g_osdPTexSpecular = 0;
const char * g_ptexColorFilename;

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
ID3D11Buffer* g_pcbConfig = NULL;
ID3D11DepthStencilView* g_pDepthStencilView = NULL;

bool g_bDone = false;

//------------------------------------------------------------------------------
static void
calcNormals(OpenSubdiv::Far::TopologyRefiner * refiner,
    std::vector<float> const & pos, std::vector<float> & result ) {

    typedef OpenSubdiv::Far::ConstIndexArray IndexArray;

    // calc normal vectors
    OpenSubdiv::Far::TopologyLevel const & refBaseLevel = refiner->GetLevel(0);

    int nverts = refBaseLevel.GetNumVertices(),
        nfaces = refBaseLevel.GetNumFaces();

    for (int face = 0; face < nfaces; ++face) {

        IndexArray fverts = refBaseLevel.GetFaceVertices(face);

        float const * p0 = &pos[fverts[0]*3],
                    * p1 = &pos[fverts[1]*3],
                    * p2 = &pos[fverts[2]*3];

        float n[3];
        cross(n, p0, p1, p2);

        for (int vert = 0; vert < fverts.size(); ++vert) {
            int idx = fverts[vert] * 3;
            result[idx  ] += n[0];
            result[idx+1] += n[1];
            result[idx+2] += n[2];
        }
    }
    for (int i = 0; i < nverts; ++i)
        normalize(&result[i*3]);
}

//------------------------------------------------------------------------------
static void
updateGeom() {

    int nverts = (int)g_positions.size() / 3;

    std::vector<float> vertex;
    vertex.reserve(nverts*6);

    const float *p = &g_positions[0];
    const float *n = &g_normals[0];

    for (int i = 0; i < nverts; ++i) {
        float move = g_size*0.005f*cosf(p[0]*100/g_size+g_frame*0.01f);
        vertex.push_back(p[0]);
        vertex.push_back(p[1]+g_moveScale*move);
        vertex.push_back(p[2]);
        p += 3;
//        if (g_adaptive == false)
        {
            vertex.push_back(n[0]);
            vertex.push_back(n[1]);
            vertex.push_back(n[2]);
            n += 3;
        }
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

//-------------------------------------------------------------------------------
static void
fitFrame() {
    g_pan[0] = g_pan[1] = 0;
    g_dolly = g_size;
}

//-------------------------------------------------------------------------------
Shape *
createPTexGeo(PtexTexture * r) {

    PtexMetaData* meta = r->getMetaData();

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

    Shape * shape = new Shape;

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

    for (int j = 0; j < 3; ++j) {
        g_center[j] = (min[j] + max[j]) * 0.5f;
        g_size += (max[j]-min[j])*(max[j]-min[j]);
    }
    g_size = sqrtf(g_size);

    return shape;
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

union Effect {
    struct {
        unsigned int wire:2;
        unsigned int color:3;
        unsigned int displacement:2;
        unsigned int normal:3;
        int occlusion:1;
        int specular:1;
        int patchCull:1;
        int screenSpaceTess:1;
        int fractionalSpacing:1;
        int ibl:1;
        int seamless:1;
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
        return desc < e.desc || (desc == e.desc &&
              (maxValence < e.maxValence || ((maxValence == e.maxValence) &&
              (effect < e.effect))));
    }
};

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

       // add ptex functions
        ss << D3D11PtexMipmapTexture::GetShaderSource();

        // -------------------------------------------------------------
        // display styles
        // -------------------------------------------------------------

        // mipmap
        if (effectDesc.effect.seamless) {
            ss << "#define SEAMLESS_MIPMAP\n";
        }

        //  wire
        if (effectDesc.effect.wire == 0) {
            ss << "#define GEOMETRY_OUT_WIRE\n";
        } else if (effectDesc.effect.wire == 1) {
            ss << "#define GEOMETRY_OUT_FILL\n";
        } else if (effectDesc.effect.wire == 2) {
            ss << "#define GEOMETRY_OUT_LINE\n";
        }

        //  color
        switch(effectDesc.effect.color) {
        case COLOR_NONE:
            break;
        case COLOR_PTEX_NEAREST:
            ss << "#define COLOR_PTEX_NEAREST\n";
            break;
        case COLOR_PTEX_HW_BILINEAR:
            ss << "#define COLOR_PTEX_HW_BILINEAR\n";
            break;
        case COLOR_PTEX_BILINEAR:
            ss << "#define COLOR_PTEX_BILINEAR\n";
            break;
        case COLOR_PTEX_BIQUADRATIC:
            ss << "#define COLOR_PTEX_BIQUADRATIC\n";
            break;
        case COLOR_PATCHTYPE:
            ss << "#define COLOR_PATCHTYPE\n";
            break;
        case COLOR_PATCHCOORD:
            ss << "#define COLOR_PATCHCOORD\n";
            break;
        case COLOR_NORMAL:
            ss << "#define COLOR_NORMAL\n";
            break;
        }

        // displacement
        switch (effectDesc.effect.displacement) {
        case DISPLACEMENT_NONE:
            break;
        case DISPLACEMENT_HW_BILINEAR:
            ss << "#define DISPLACEMENT_HW_BILINEAR\n";
            break;
        case DISPLACEMENT_BILINEAR:
            ss << "#define DISPLACEMENT_BILINEAR\n";
            break;
        case DISPLACEMENT_BIQUADRATIC:
            ss << "#define DISPLACEMENT_BIQUADRATIC\n";
            break;
        }

        // normal
        switch (effectDesc.effect.normal) {
        case NORMAL_FACET:
            ss << "#define NORMAL_FACET\n";
            break;
        case NORMAL_HW_SCREENSPACE:
            ss << "#define NORMAL_HW_SCREENSPACE\n";
            break;
        case NORMAL_SCREENSPACE:
            ss << "#define NORMAL_SCREENSPACE\n";
            break;
        case NORMAL_BIQUADRATIC:
            ss << "#define NORMAL_BIQUADRATIC\n";
            break;
        case NORMAL_BIQUADRATIC_WG:
            ss << "#define OSD_COMPUTE_NORMAL_DERIVATIVES\n";
            ss << "#define NORMAL_BIQUADRATIC_WG\n";
            break;
        }

        // occlusion
        if (effectDesc.effect.occlusion)
            ss << "#define USE_PTEX_OCCLUSION\n";

        // specular
        if (effectDesc.effect.specular)
            ss << "#define USE_PTEX_SPECULAR\n";

        // IBL
        if (effectDesc.effect.ibl)
            ss << "#define USE_IBL\n";

        // need for patch color-coding : we need these defines in the fragment shader
        if (type == Far::PatchDescriptor::GREGORY) {
            ss << "#define OSD_PATCH_GREGORY\n";
        } else if (type == Far::PatchDescriptor::GREGORY_BOUNDARY) {
            ss << "#define OSD_PATCH_GREGORY_BOUNDARY\n";
        } else if (type == Far::PatchDescriptor::GREGORY_BASIS) {
            ss << "#define OSD_PATCH_GREGORY_BASIS\n";
        }

        // include osd PatchCommon
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
           << g_shaderSource
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
               << g_shaderSource
               << Osd::HLSLPatchShaderSource::GetHullShaderSource(type);
            config->CompileHullShader("hs_5_0", "hs_main_patches", ss.str(),
                                      g_pd3dDevice);
            ss.str("");

            // domain shader
            ss << common
               << g_shaderSource
               << Osd::HLSLPatchShaderSource::GetDomainShaderSource(type);
            config->CompileDomainShader("ds_5_0", "ds_main_patches", ss.str(),
                                        g_pd3dDevice);
            ss.str("");
        }

        // geometry shader
        ss << common
           << g_shaderSource;
        config->CompileGeometryShader("gs_5_0", "gs_main", ss.str(),
                                      g_pd3dDevice);
        ss.str("");

        // pixel shader
        ss << common
           << g_shaderSource;
        config->CompilePixelShader("ps_5_0", "ps_main", ss.str(),
                                   g_pd3dDevice);
        ss.str("");

        return config;
    };
};

ShaderCache g_shaderCache;

//------------------------------------------------------------------------------
D3D11PtexMipmapTexture *
createPtex(const char *filename) {

    Ptex::String ptexError;
    printf("Loading ptex : %s\n", filename);

#define USE_PTEX_CACHE
#define PTEX_CACHE_SIZE (512*1024*1024)

#ifdef USE_PTEX_CACHE
    PtexCache *cache = PtexCache::create(1, PTEX_CACHE_SIZE);
    PtexTexture *ptex = cache->get(filename, ptexError);
#else
    PtexTexture *ptex = PtexTexture::open(filename, ptexError, true);
#endif

    if (ptex == NULL) {
        printf("Error in reading %s\n", filename);
        exit(1);
    }
    D3D11PtexMipmapTexture *osdPtex = D3D11PtexMipmapTexture::Create(
        g_pd3dDeviceContext, ptex, g_maxMipmapLevels);

    ptex->release();

#ifdef USE_PTEX_CACHE
    cache->release();
#endif

    return osdPtex;
}

//------------------------------------------------------------------------------
void
createOsdMesh(int level, int kernel) {

    using namespace OpenSubdiv;
    Ptex::String ptexError;
    PtexTexture *ptexColor = PtexTexture::open(g_ptexColorFilename, ptexError, true);
    if (ptexColor == NULL) {
        printf("Error in reading %s\n", g_ptexColorFilename);
        exit(1);
    }

    // generate Hbr representation from ptex
    Shape * shape = createPTexGeo(ptexColor);
    if (not shape) {
        return;
    }

    g_positions=shape->verts;

    typedef OpenSubdiv::Far::ConstIndexArray IndexArray;

    // create Far mesh (topology)
    OpenSubdiv::Sdc::SchemeType sdctype = GetSdcType(*shape);
    OpenSubdiv::Sdc::Options sdcoptions = GetSdcOptions(*shape);

    OpenSubdiv::Far::TopologyRefiner * refiner =
        OpenSubdiv::Far::TopologyRefinerFactory<Shape>::Create(*shape,
            OpenSubdiv::Far::TopologyRefinerFactory<Shape>::Options(sdctype, sdcoptions));

    // save coarse topology (used for coarse mesh drawing)
    OpenSubdiv::Far::TopologyLevel const & refBaseLevel = refiner->GetLevel(0);

    // create cage edge index
    int nedges = refBaseLevel.GetNumEdges();
    std::vector<int> edgeIndices(nedges*2);
    for(int i=0; i<nedges; ++i) {
        IndexArray verts = refBaseLevel.GetEdgeVertices(i);
        edgeIndices[i*2  ]=verts[0];
        edgeIndices[i*2+1]=verts[1];
    }

    delete shape;

    g_normals.resize(g_positions.size(), 0.0f);
    calcNormals(refiner, g_positions, g_normals);

    delete g_mesh;
    g_mesh = NULL;

    // Adaptive refinement currently supported only for catmull-clark scheme
    bool doAdaptive = (g_adaptive != 0 and g_scheme == 0);

    OpenSubdiv::Osd::MeshBitset bits;
    bits.set(OpenSubdiv::Osd::MeshAdaptive, doAdaptive);
    bits.set(OpenSubdiv::Osd::MeshEndCapGregoryBasis, true);

    int numVertexElements = 6; //g_adaptive ? 3 : 6;
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
#ifdef OPENSUBDIV_HAS_OPENCL
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

    updateGeom();
}

//------------------------------------------------------------------------------
static void
bindProgram(Effect effect, OpenSubdiv::Osd::PatchArray const & patch) {

    EffectDesc effectDesc(patch.GetDescriptor(), effect);

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
        translate(pData->ModelViewMatrix, -g_center[0], -g_center[1], -g_center[2]);

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
        pData->PrimitiveIdBase = patch.GetPrimitiveIdBase();

        g_pd3dDeviceContext->Unmap( g_pcbTessellation, 0 );
    }

    // Update config state
    {
        __declspec(align(16))
            struct Config {
                float displacementScale;
                float mipmapBias;
            };

        if (! g_pcbConfig) {
            D3D11_BUFFER_DESC cbDesc;
            ZeroMemory(&cbDesc, sizeof(cbDesc));
            cbDesc.Usage = D3D11_USAGE_DYNAMIC;
            cbDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
            cbDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
            cbDesc.MiscFlags = 0;
            cbDesc.ByteWidth = sizeof(Config);
            g_pd3dDevice->CreateBuffer(&cbDesc, NULL, &g_pcbConfig);
        }
        assert(g_pcbConfig);

        D3D11_MAPPED_SUBRESOURCE MappedResource;
        g_pd3dDeviceContext->Map(g_pcbConfig, 0, D3D11_MAP_WRITE_DISCARD, 0, &MappedResource);
        Config * pData = ( Config* )MappedResource.pData;

        pData->displacementScale = g_displacementScale;
        pData->mipmapBias = g_mipmapBias;

        g_pd3dDeviceContext->Unmap( g_pcbConfig, 0 );
    }

    g_pd3dDeviceContext->IASetInputLayout(g_pInputLayout);

    g_pd3dDeviceContext->VSSetShader(config->GetVertexShader(), NULL, 0);
    g_pd3dDeviceContext->VSSetConstantBuffers(0, 1, &g_pcbPerFrame);
    g_pd3dDeviceContext->HSSetShader(config->GetHullShader(), NULL, 0);
    g_pd3dDeviceContext->HSSetConstantBuffers(0, 1, &g_pcbPerFrame);
    g_pd3dDeviceContext->HSSetConstantBuffers(1, 1, &g_pcbTessellation);
    g_pd3dDeviceContext->DSSetShader(config->GetDomainShader(), NULL, 0);
    g_pd3dDeviceContext->DSSetConstantBuffers(0, 1, &g_pcbPerFrame);
    g_pd3dDeviceContext->DSSetConstantBuffers(3, 1, &g_pcbConfig);
    g_pd3dDeviceContext->GSSetShader(config->GetGeometryShader(), NULL, 0);
    g_pd3dDeviceContext->GSSetConstantBuffers(0, 1, &g_pcbPerFrame);
    g_pd3dDeviceContext->PSSetShader(config->GetPixelShader(), NULL, 0);
    g_pd3dDeviceContext->PSSetConstantBuffers(0, 1, &g_pcbPerFrame);
    g_pd3dDeviceContext->PSSetConstantBuffers(2, 1, &g_pcbLighting);
    g_pd3dDeviceContext->PSSetConstantBuffers(3, 1, &g_pcbConfig);

    ID3D11ShaderResourceView *srv = g_mesh->GetPatchTable()->GetPatchParamSRV();
    if (srv) {
        g_pd3dDeviceContext->HSSetShaderResources(0, 1, &srv);
        g_pd3dDeviceContext->DSSetShaderResources(0, 1, &srv);
        g_pd3dDeviceContext->GSSetShaderResources(0, 1, &srv);
        g_pd3dDeviceContext->PSSetShaderResources(0, 1, &srv);
    }

    g_pd3dDeviceContext->PSSetShaderResources(4, 1, g_osdPTexImage->GetTexelsSRV());
    g_pd3dDeviceContext->PSSetShaderResources(5, 1, g_osdPTexImage->GetLayoutSRV());

    if (g_osdPTexDisplacement) {
        g_pd3dDeviceContext->DSSetShaderResources(6, 1, g_osdPTexDisplacement->GetTexelsSRV());
        g_pd3dDeviceContext->DSSetShaderResources(7, 1, g_osdPTexDisplacement->GetLayoutSRV());
        g_pd3dDeviceContext->PSSetShaderResources(6, 1, g_osdPTexDisplacement->GetTexelsSRV());
        g_pd3dDeviceContext->PSSetShaderResources(7, 1, g_osdPTexDisplacement->GetLayoutSRV());
    }

    if (g_osdPTexOcclusion) {
        g_pd3dDeviceContext->PSSetShaderResources(8, 1, g_osdPTexOcclusion->GetTexelsSRV());
        g_pd3dDeviceContext->PSSetShaderResources(9, 1, g_osdPTexOcclusion->GetLayoutSRV());
    }

    if (g_osdPTexSpecular) {
        g_pd3dDeviceContext->PSSetShaderResources(10, 1, g_osdPTexSpecular->GetTexelsSRV());
        g_pd3dDeviceContext->PSSetShaderResources(11, 1, g_osdPTexSpecular->GetLayoutSRV());
    }
}

//------------------------------------------------------------------------------
static void
drawModel() {

    ID3D11Buffer *buffer = g_mesh->BindVertexBuffer();
    assert(buffer);

    UINT hStrides = 6*sizeof(float);
    UINT hOffsets = 0;
    g_pd3dDeviceContext->IASetVertexBuffers(0, 1, &buffer, &hStrides, &hOffsets);

    OpenSubdiv::Osd::PatchArrayVector const & patches =
        g_mesh->GetPatchTable()->GetPatchArrays();

    g_pd3dDeviceContext->IASetIndexBuffer(
        g_mesh->GetPatchTable()->GetPatchIndexBuffer(),
        DXGI_FORMAT_R32_UINT, 0);

    // patch drawing
    for (int i = 0; i < (int)patches.size(); ++i) {
        OpenSubdiv::Osd::PatchArray const & patch = patches[i];
        OpenSubdiv::Far::PatchDescriptor desc = patch.GetDescriptor();
        OpenSubdiv::Far::PatchDescriptor::Type patchType = desc.GetType();

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
            case 16:
                topology = D3D11_PRIMITIVE_TOPOLOGY_16_CONTROL_POINT_PATCHLIST;
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

        Effect effect;
        effect.value = 0;

        effect.color = g_color;
        effect.displacement = g_displacement;

        effect.occlusion = g_occlusion;
        effect.normal = g_normal;
        effect.specular = g_specular;
        effect.patchCull = g_patchCull;
        effect.screenSpaceTess = g_screenSpaceTess;
        effect.fractionalSpacing = g_fractionalSpacing;
        effect.ibl = g_ibl;
        effect.wire = g_wire;
        effect.seamless = g_seamless;

        bindProgram(effect, patch);

        g_pd3dDeviceContext->IASetPrimitiveTopology(topology);

        g_pd3dDeviceContext->DrawIndexed(
            patch.GetNumPatches() * desc.GetNumControlVertices(),
            patch.GetIndexBase(), 0);

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

    drawModel();


    if (g_hud->IsVisible()) {
        g_fpsTimer.Stop();
        double fps = 1.0/g_fpsTimer.GetElapsed();
        g_fpsTimer.Start();

        g_hud->DrawString(10, -100, "# of Vertices = %d", g_mesh->GetNumVertices());
        g_hud->DrawString(10, -60, "GPU TIME = %.3f ms", g_gpuTime);
        g_hud->DrawString(10, -40, "CPU TIME = %.3f ms", g_cpuTime);
        g_hud->DrawString(10, -20, "FPS = %3.1f", fps);
    }

    g_hud->Flush();

    g_pSwapChain->Present(0, 0);
}

//------------------------------------------------------------------------------
static void
mouse(int button, int state, int x, int y) {

    if (state == 0)
        g_hud->MouseRelease();

    if (button == 0 && state == 1 && g_hud->MouseClick(x, y)) return;

    if (button < 3) {
        g_prev_x = float(x);
        g_prev_y = float(y);
        g_mbutton[button] = state;
    }
}

//------------------------------------------------------------------------------
static void
motion(int x, int y) {

    if (g_hud->MouseCapture()) {
        // check gui
        g_hud->MouseMotion(x, y);
    } else if (g_mbutton[0] && !g_mbutton[1] && !g_mbutton[2]) {
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

//-----------------------------------------------------------------------------
static void
quit() {

    g_bDone = true;

    if (g_osdPTexImage) delete g_osdPTexImage;
    if (g_osdPTexDisplacement) delete g_osdPTexDisplacement;
    if (g_osdPTexOcclusion) delete g_osdPTexOcclusion;
    if (g_osdPTexSpecular) delete g_osdPTexSpecular;

    if (g_mesh) delete g_mesh;
    if (g_hud) delete g_hud;

    SAFE_RELEASE(g_pRasterizerState);
    SAFE_RELEASE(g_pInputLayout);
    SAFE_RELEASE(g_pDepthStencilState);
    SAFE_RELEASE(g_pcbPerFrame);
    SAFE_RELEASE(g_pcbTessellation);
    SAFE_RELEASE(g_pcbLighting);
    SAFE_RELEASE(g_pcbConfig);
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

    if (g_hud->KeyDown((int)tolower(key))) return;

    switch (key) {
        case 'Q': quit();
        case 'F': fitFrame(); break;
        case '+':
        case '=': g_tessLevel++; break;
        case '-': g_tessLevel = std::max(1, g_tessLevel-1); break;
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
    if (g_kernel == kCL and (not g_clDeviceContext.IsInitialized())) {
        if (g_clDeviceContext.Initialize(g_pd3dDeviceContext) == false) {
            printf("Error in initializing OpenCL\n");
            exit(1);
        }
    }
#endif
#ifdef OPENSUBDIV_HAS_CUDA
    if (g_kernel == kCUDA and (not g_cudaDeviceContext.IsInitialized())) {
        if (g_cudaDeviceContext.Initialize(g_pd3dDevice) == false) {
            printf("Error in initializing Cuda\n");
            exit(1);
        }
    }
#endif

    createOsdMesh(g_level, g_kernel);
}

static void
callbackScheme(int s) {
    g_scheme = s;
    createOsdMesh(g_level, g_kernel);
}
static void
callbackLevel(int l) {
    g_level = l;
    createOsdMesh(g_level, g_kernel);
}
static void
callbackColor(int c) {
    g_color = c;
}
static void
callbackDisplacement(int d) {
    g_displacement = d;
}
static void
callbackNormal(int n) {
    g_normal = n;
}

static void
callbackCheckBox(bool checked, int button) {
    bool rebuild = false;

    switch (button) {
    case HUD_CB_ADAPTIVE:
        g_adaptive = checked;
        rebuild = true;
        break;
    case HUD_CB_DISPLAY_OCCLUSION:
        g_occlusion = checked;
        break;
    case HUD_CB_DISPLAY_SPECULAR:
        g_specular = checked;
        break;
    case HUD_CB_ANIMATE_VERTICES:
        g_moveScale = checked ? 1.0f : 0.0f;
        g_animTime = 0;
        break;
    case HUD_CB_VIEW_LOD:
        g_screenSpaceTess = checked;
        break;
    case HUD_CB_FRACTIONAL_SPACING:
        g_fractionalSpacing = checked;
        break;
    case HUD_CB_PATCH_CULL:
        g_patchCull = checked;
        break;
    case HUD_CB_IBL:
        g_ibl = checked;
        break;
    case HUD_CB_SEAMLESS_MIPMAP:
        g_seamless = checked;
        break;
    case HUD_CB_FREEZE:
        g_freeze = checked;
        break;
    }

    if (rebuild)
        createOsdMesh(g_level, g_kernel);
}

static void
callbackSlider(float value, int data) {
    switch (data) {
    case 0:
        g_mipmapBias = value;
        break;
    case 1:
        g_displacementScale = value;
        break;
    }
}

static void
initHUD() {
    g_hud = new D3D11hud(g_pd3dDeviceContext);
    g_hud->Init(g_width, g_height);

    g_hud->AddRadioButton(0, "CPU (K)", true, 10, 10, callbackKernel, kCPU, 'K');
#ifdef OPENSUBDIV_HAS_OPENMP
    g_hud->AddRadioButton(0, "OPENMP", false, 10, 30, callbackKernel, kOPENMP, 'K');
#endif
#ifdef OPENSUBDIV_HAS_TBB
    g_hud->AddRadioButton(0, "TBB", false, 10, 50, callbackKernel, kTBB, 'K');
#endif
#ifdef OPENSUBDIV_HAS_CUDA
    g_hud->AddRadioButton(0, "CUDA", false, 10, 70, callbackKernel, kCUDA, 'K');
#endif
#ifdef OPENSUBDIV_HAS_OPENCL
    if (CLDeviceContext::HAS_CL_VERSION_1_1()) {
        g_hud->AddRadioButton(0, "OPENCL", false, 10, 90, callbackKernel, kCL, 'K');
    }
#endif
    g_hud->AddRadioButton(0, "DirectCompute", false, 10, 110, callbackKernel, kDirectCompute, 'K');

    g_hud->AddCheckBox("Adaptive (`)", g_adaptive,
                       10, 150, callbackCheckBox, HUD_CB_ADAPTIVE, '`');

    g_hud->AddRadioButton(HUD_RB_SCHEME, "CATMARK", true, 10, 190, callbackScheme, 0, 's');
    g_hud->AddRadioButton(HUD_RB_SCHEME, "BILINEAR", false, 10, 210, callbackScheme, 1, 's');

    for (int i = 1; i < 8; ++i) {
        char level[16];
        sprintf(level, "Lv. %d", i);
        g_hud->AddRadioButton(HUD_RB_LEVEL, level, i == g_level,
                              10, 220+i*20, callbackLevel, i, '0'+i);
    }

    g_hud->AddRadioButton(HUD_RB_WIRE, "Wire (W)",       (g_wire == DISPLAY_WIRE),
                         100, 10, callbackWireframe, 0, 'w');
    g_hud->AddRadioButton(HUD_RB_WIRE, "Shaded",         (g_wire == DISPLAY_SHADED),
                         100, 30, callbackWireframe, 1, 'w');
    g_hud->AddRadioButton(HUD_RB_WIRE, "Wire on Shaded", (g_wire == DISPLAY_WIRE_ON_SHADED),
                         100, 50, callbackWireframe, 2, 'w');

    g_hud->AddLabel("Color (C)", -200, 10);
    g_hud->AddRadioButton(HUD_RB_COLOR, "None", (g_color == COLOR_NONE),
                         -200, 30, callbackColor, COLOR_NONE, 'c');
    g_hud->AddRadioButton(HUD_RB_COLOR, "Ptex Nearest", (g_color == COLOR_PTEX_NEAREST),
                         -200, 50, callbackColor, COLOR_PTEX_NEAREST, 'c');
    g_hud->AddRadioButton(HUD_RB_COLOR, "Ptex HW bilinear", (g_color == COLOR_PTEX_HW_BILINEAR),
                         -200, 70, callbackColor, COLOR_PTEX_HW_BILINEAR, 'c');
    g_hud->AddRadioButton(HUD_RB_COLOR, "Ptex bilinear", (g_color == COLOR_PTEX_BILINEAR),
                         -200, 90, callbackColor, COLOR_PTEX_BILINEAR, 'c');
    g_hud->AddRadioButton(HUD_RB_COLOR, "Ptex biquadratic", (g_color == COLOR_PTEX_BIQUADRATIC),
                         -200, 110, callbackColor, COLOR_PTEX_BIQUADRATIC, 'c');
    g_hud->AddRadioButton(HUD_RB_COLOR, "Patch type", (g_color == COLOR_PATCHTYPE),
                         -200, 130, callbackColor, COLOR_PATCHTYPE, 'c');
    g_hud->AddRadioButton(HUD_RB_COLOR, "Patch coord", (g_color == COLOR_PATCHCOORD),
                         -200, 150, callbackColor, COLOR_PATCHCOORD, 'c');
    g_hud->AddRadioButton(HUD_RB_COLOR, "Normal", (g_color == COLOR_NORMAL),
                         -200, 170, callbackColor, COLOR_NORMAL, 'c');

    if (g_osdPTexDisplacement != NULL) {
        g_hud->AddLabel("Displacement (D)", -200, 200);
        g_hud->AddRadioButton(HUD_RB_DISPLACEMENT, "None",
                             (g_displacement == DISPLACEMENT_NONE),
                             -200, 220, callbackDisplacement, DISPLACEMENT_NONE, 'd');
        g_hud->AddRadioButton(HUD_RB_DISPLACEMENT, "HW bilinear",
                             (g_displacement == DISPLACEMENT_HW_BILINEAR),
                             -200, 240, callbackDisplacement, DISPLACEMENT_HW_BILINEAR, 'd');
        g_hud->AddRadioButton(HUD_RB_DISPLACEMENT, "Bilinear",
                             (g_displacement == DISPLACEMENT_BILINEAR),
                             -200, 260, callbackDisplacement, DISPLACEMENT_BILINEAR, 'd');
        g_hud->AddRadioButton(HUD_RB_DISPLACEMENT, "Biquadratic",
                             (g_displacement == DISPLACEMENT_BIQUADRATIC),
                             -200, 280, callbackDisplacement, DISPLACEMENT_BIQUADRATIC, 'd');

        g_hud->AddLabel("Normal (N)", -200, 310);
        g_hud->AddRadioButton(HUD_RB_NORMAL, "Surface",
                             (g_normal == NORMAL_SURFACE),
                             -200, 330, callbackNormal, NORMAL_SURFACE, 'n');
        g_hud->AddRadioButton(HUD_RB_NORMAL, "Facet",
                             (g_normal == NORMAL_FACET),
                             -200, 350, callbackNormal, NORMAL_FACET, 'n');
        g_hud->AddRadioButton(HUD_RB_NORMAL, "HW Screen space",
                             (g_normal == NORMAL_HW_SCREENSPACE),
                             -200, 370, callbackNormal, NORMAL_HW_SCREENSPACE, 'n');
        g_hud->AddRadioButton(HUD_RB_NORMAL, "Screen space",
                             (g_normal == NORMAL_SCREENSPACE),
                             -200, 390, callbackNormal, NORMAL_SCREENSPACE, 'n');
        g_hud->AddRadioButton(HUD_RB_NORMAL, "Biquadratic",
                             (g_normal == NORMAL_BIQUADRATIC),
                             -200, 410, callbackNormal, NORMAL_BIQUADRATIC, 'n');
        g_hud->AddRadioButton(HUD_RB_NORMAL, "Biquadratic WG",
                             (g_normal == NORMAL_BIQUADRATIC_WG),
                             -200, 430, callbackNormal, NORMAL_BIQUADRATIC_WG, 'n');
    }

    g_hud->AddSlider("Mipmap Bias", 0, 5, 0,
                    -200, 450, 20, false, callbackSlider, 0);
    g_hud->AddSlider("Displacement", 0, 5, 1,
                    -200, 490, 20, false, callbackSlider, 1);
    g_hud->AddCheckBox("Seamless Mipmap", g_seamless,
                       -200, 530, callbackCheckBox, HUD_CB_SEAMLESS_MIPMAP, 'j');

    if (g_osdPTexOcclusion != NULL) {
        g_hud->AddCheckBox("Ambient Occlusion (A)", g_occlusion,
                          250, 10, callbackCheckBox, HUD_CB_DISPLAY_OCCLUSION, 'a');
    }
    if (g_osdPTexSpecular != NULL)
        g_hud->AddCheckBox("Specular (S)", g_specular,
                          250, 30, callbackCheckBox, HUD_CB_DISPLAY_SPECULAR, 's');

    g_hud->AddCheckBox("Animate vertices (M)", g_moveScale != 0.0,
                      450, 10, callbackCheckBox, HUD_CB_ANIMATE_VERTICES, 'm');
    g_hud->AddCheckBox("Screen space LOD (V)",  g_screenSpaceTess,
                      450, 30, callbackCheckBox, HUD_CB_VIEW_LOD, 'v');
    //g_hud->AddCheckBox("Fractional spacing (T)",  g_fractionalSpacing,
    //                  450, 50, callbackCheckBox, HUD_CB_FRACTIONAL_SPACING, 't');
    g_hud->AddCheckBox("Frustum Patch Culling (B)",  g_patchCull,
                      450, 70, callbackCheckBox, HUD_CB_PATCH_CULL, 'b');
    g_hud->AddCheckBox("Freeze (spc)", g_freeze,
                      450, 90, callbackCheckBox, HUD_CB_FREEZE, ' ');
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
        unsigned int deviceFlags = 0;
#ifndef NDEBUG		
                // XXX: this is problematic in some environments.
//		deviceFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif
        hr = D3D11CreateDeviceAndSwapChain(NULL,
                                           hDriverType, NULL, deviceFlags, NULL, 0,
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

#ifndef NDEBUG
    // set break points on directx errors
    ID3D11Debug *d3dDebug = nullptr;
    hr = g_pd3dDevice->QueryInterface(__uuidof(ID3D11Debug), (void**)&d3dDebug);
    if (SUCCEEDED(hr)) {
        ID3D11InfoQueue *d3dInfoQueue = nullptr;
        hr = d3dDebug->QueryInterface(__uuidof(ID3D11InfoQueue), (void**)&d3dInfoQueue);
        if (SUCCEEDED(hr)) {

            d3dInfoQueue->SetBreakOnSeverity(D3D11_MESSAGE_SEVERITY_CORRUPTION, true);
            d3dInfoQueue->SetBreakOnSeverity(D3D11_MESSAGE_SEVERITY_ERROR, true);
            d3dInfoQueue->SetBreakOnSeverity(D3D11_MESSAGE_SEVERITY_WARNING, true);

            D3D11_MESSAGE_ID denied[] = { D3D11_MESSAGE_ID_SETPRIVATEDATA_CHANGINGPARAMS };
            D3D11_INFO_QUEUE_FILTER filter;
            memset(&filter, 0, sizeof(filter));
            filter.DenyList.NumIDs = _countof(denied);
            filter.DenyList.pIDList = denied;
            d3dInfoQueue->AddStorageFilterEntries(&filter);

            d3dInfoQueue->Release();
        }
        d3dDebug->Release();
    }
#endif

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
    wcex.hCursor        = LoadCursor(NULL, IDC_ARROW);
    wcex.hbrBackground  = (HBRUSH)(COLOR_WINDOW+1);
    wcex.lpszMenuName   = NULL;
    wcex.lpszClassName  = szWindowClass;
    RegisterClass(&wcex);

    // crete window
    RECT rect = { 0, 0, g_width, g_height };
    AdjustWindowRect(&rect, WS_OVERLAPPEDWINDOW, FALSE);

    static const char windowTitle[] = "OpenSubdiv dxPtexViewer " OPENSUBDIV_VERSION_STRING;

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

    std::vector<std::string> argv = tokenize(lpCmdLine);
    std::vector<std::string> animobjs;
    const char *diffuseEnvironmentMap = NULL, *specularEnvironmentMap = NULL;
    const char *colorFilename = NULL, *displacementFilename = NULL,
        *occlusionFilename = NULL, *specularFilename = NULL;

    for (int i = 0; i < (int)argv.size(); ++i) {
        if (strstr(argv[i].c_str(), ".obj"))
            animobjs.push_back(argv[i]);
        else if (argv[i] == "-l")
            g_level = atoi(argv[++i].c_str());
        else if (argv[i] == "-c")
            g_repeatCount = atoi(argv[++i].c_str());
        else if (argv[i] == "-d")
            diffuseEnvironmentMap = argv[++i].c_str();
        else if (argv[i] == "-e")
            specularEnvironmentMap = argv[++i].c_str();
        else if (argv[i] == "-y")
            g_yup = true;
        else if (argv[i] == "-m")
            g_maxMipmapLevels = atoi(argv[++i].c_str());
        else if (argv[i] == "--disp")
            g_displacementScale = (float)atof(argv[++i].c_str());
        else if (colorFilename == NULL)
            colorFilename = argv[i].c_str();
        else if (displacementFilename == NULL) {
            displacementFilename = argv[i].c_str();
            g_displacement = DISPLACEMENT_BILINEAR;
            g_normal = NORMAL_BIQUADRATIC;
        } else if (occlusionFilename == NULL) {
            occlusionFilename = argv[i].c_str();
            g_occlusion = 1;
        } else if (specularFilename == NULL) {
            specularFilename = argv[i].c_str();
            g_specular = 1;
        }
    }

    OpenSubdiv::Far::SetErrorCallback(callbackError);

    g_ptexColorFilename = colorFilename;
    if (g_ptexColorFilename == NULL) {
        printf("Usage: \n");
        return 1;
    }

    initD3D11(hWnd);

    createOsdMesh(g_level, g_kernel);

    // load ptex files
    g_osdPTexImage = createPtex(colorFilename);
    if (displacementFilename)
        g_osdPTexDisplacement = createPtex(displacementFilename);
    if (occlusionFilename)
        g_osdPTexOcclusion = createPtex(occlusionFilename);
    if (specularFilename)
        g_osdPTexSpecular = createPtex(specularFilename);

    initHUD();

    fitFrame();

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
    return 0;
}

//------------------------------------------------------------------------------
