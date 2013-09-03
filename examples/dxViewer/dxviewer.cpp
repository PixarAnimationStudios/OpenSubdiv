//
//     Copyright 2013 Pixar
//
//     Licensed under the Apache License, Version 2.0 (the "License");
//     you may not use this file except in compliance with the License
//     and the following modification to it: Section 6 Trademarks.
//     deleted and replaced with:
//
//     6. Trademarks. This License does not grant permission to use the
//     trade names, trademarks, service marks, or product names of the
//     Licensor and its affiliates, except as required for reproducing
//     the content of the NOTICE file.
//
//     You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
//     Unless required by applicable law or agreed to in writing,
//     software distributed under the License is distributed on an
//     "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
//     either express or implied.  See the License for the specific
//     language governing permissions and limitations under the
//     License.
//
#include <D3D11.h>
#include <D3Dcompiler.h>

#include <osd/error.h>
#include <osd/vertex.h>
#include <osd/d3d11DrawContext.h>
#include <osd/d3d11DrawRegistry.h>

#include <osd/cpuD3D11VertexBuffer.h>
#include <osd/cpuComputeContext.h>
#include <osd/cpuComputeController.h>
OpenSubdiv::OsdCpuComputeController * g_cpuComputeController = NULL;

#ifdef OPENSUBDIV_HAS_OPENMP
    #include <osd/ompComputeController.h>
    OpenSubdiv::OsdOmpComputeController * g_ompComputeController = NULL;
#endif

#undef OPENSUBDIV_HAS_OPENCL    // XXX: dyu OpenCL D3D11 interop needs work...
#ifdef OPENSUBDIV_HAS_OPENCL
    #include <osd/clD3D11VertexBuffer.h>
    #include <osd/clComputeContext.h>
    #include <osd/clComputeController.h>

    #include "../common/clInit.h"

    cl_context g_clContext;
    cl_command_queue g_clQueue;
    OpenSubdiv::OsdCLComputeController * g_clComputeController = NULL;
#endif

#ifdef OPENSUBDIV_HAS_CUDA
    #include <osd/cudaD3D11VertexBuffer.h>
    #include <osd/cudaComputeContext.h>
    #include <osd/cudaComputeController.h>

    #include <cuda_runtime_api.h>
    #include <cuda_d3d11_interop.h>

    bool g_cudaInitialized = false;
    OpenSubdiv::OsdCudaComputeController * g_cudaComputeController = NULL;
#endif

#include <osd/d3d11VertexBuffer.h>
#include <osd/d3d11ComputeContext.h>
#include <osd/d3d11ComputeController.h>
OpenSubdiv::OsdD3D11ComputeController * g_d3d11ComputeController = NULL;

#include <osd/d3d11Mesh.h>
OpenSubdiv::OsdD3D11MeshInterface *g_mesh;

#include "../../regression/common/shape_utils.h"
#include "../common/stopwatch.h"
#include "../common/simple_math.h"
#include "../common/d3d11_hud.h"

static const char *shaderSource =
#include "shader.inc"
;

#include <algorithm>
#include <cfloat>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>

#define SAFE_RELEASE(p) { if(p) { (p)->Release(); (p)=NULL; } }

typedef OpenSubdiv::HbrMesh<OpenSubdiv::OsdVertex>     OsdHbrMesh;
typedef OpenSubdiv::HbrVertex<OpenSubdiv::OsdVertex>   OsdHbrVertex;
typedef OpenSubdiv::HbrFace<OpenSubdiv::OsdVertex>     OsdHbrFace;
typedef OpenSubdiv::HbrHalfedge<OpenSubdiv::OsdVertex> OsdHbrHalfedge;

enum KernelType { kCPU = 0,
                  kOPENMP = 1,
                  kCUDA = 2,
                  kCL = 3,
                  kDirectCompute = 4 };

struct SimpleShape {
    std::string  name;
    Scheme       scheme;
    std::string  data;

    SimpleShape() { }
    SimpleShape( std::string const & idata, char const * iname, Scheme ischeme )
        : name(iname), scheme(ischeme), data(idata) { }
};

std::vector<SimpleShape> g_defaultShapes;

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

int   g_displayPatchColor = 1;

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
                   g_positions,
                   g_normals;

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
ID3D11DepthStencilView* g_pDepthStencilView = NULL;

bool g_bDone;

//------------------------------------------------------------------------------
static void
initializeShapes( ) {

#include <shapes/catmark_cube_corner0.h>
    g_defaultShapes.push_back(SimpleShape(catmark_cube_corner0, "catmark_cube_corner0", kCatmark));

#include <shapes/catmark_cube_corner1.h>
    g_defaultShapes.push_back(SimpleShape(catmark_cube_corner1, "catmark_cube_corner1", kCatmark));

#include <shapes/catmark_cube_corner2.h>
    g_defaultShapes.push_back(SimpleShape(catmark_cube_corner2, "catmark_cube_corner2", kCatmark));

#include <shapes/catmark_cube_corner3.h>
    g_defaultShapes.push_back(SimpleShape(catmark_cube_corner3, "catmark_cube_corner3", kCatmark));

#include <shapes/catmark_cube_corner4.h>
    g_defaultShapes.push_back(SimpleShape(catmark_cube_corner4, "catmark_cube_corner4", kCatmark));

#include <shapes/catmark_cube_creases0.h>
    g_defaultShapes.push_back(SimpleShape(catmark_cube_creases0, "catmark_cube_creases0", kCatmark));

#include <shapes/catmark_cube_creases1.h>
    g_defaultShapes.push_back(SimpleShape(catmark_cube_creases1, "catmark_cube_creases1", kCatmark));

#include <shapes/catmark_cube.h>
    g_defaultShapes.push_back(SimpleShape(catmark_cube, "catmark_cube", kCatmark));

#include <shapes/catmark_dart_edgecorner.h>
    g_defaultShapes.push_back(SimpleShape(catmark_dart_edgecorner, "catmark_dart_edgecorner", kCatmark));

#include <shapes/catmark_dart_edgeonly.h>
    g_defaultShapes.push_back(SimpleShape(catmark_dart_edgeonly, "catmark_dart_edgeonly", kCatmark));

#include <shapes/catmark_edgecorner.h>
    g_defaultShapes.push_back(SimpleShape(catmark_edgecorner ,"catmark_edgecorner", kCatmark));

#include <shapes/catmark_edgeonly.h>
    g_defaultShapes.push_back(SimpleShape(catmark_edgeonly, "catmark_edgeonly", kCatmark));

#include <shapes/catmark_gregory_test1.h>
    g_defaultShapes.push_back(SimpleShape(catmark_gregory_test1, "catmark_gregory_test1", kCatmark));

#include <shapes/catmark_gregory_test2.h>
    g_defaultShapes.push_back(SimpleShape(catmark_gregory_test2, "catmark_gregory_test2", kCatmark));

#include <shapes/catmark_gregory_test3.h>
    g_defaultShapes.push_back(SimpleShape(catmark_gregory_test3, "catmark_gregory_test3", kCatmark));

#include <shapes/catmark_gregory_test4.h>
    g_defaultShapes.push_back(SimpleShape(catmark_gregory_test4, "catmark_gregory_test4", kCatmark));

#include <shapes/catmark_hole_test1.h>
    g_defaultShapes.push_back(SimpleShape(catmark_hole_test1, "catmark_hole_test1", kCatmark));

#include <shapes/catmark_hole_test2.h>
    g_defaultShapes.push_back(SimpleShape(catmark_hole_test2, "catmark_hole_test2", kCatmark));

#include <shapes/catmark_pyramid_creases0.h>
    g_defaultShapes.push_back(SimpleShape(catmark_pyramid_creases0, "catmark_pyramid_creases0", kCatmark));

#include <shapes/catmark_pyramid_creases1.h>
    g_defaultShapes.push_back(SimpleShape(catmark_pyramid_creases1, "catmark_pyramid_creases1", kCatmark));

#include <shapes/catmark_pyramid.h>
    g_defaultShapes.push_back(SimpleShape(catmark_pyramid, "catmark_pyramid", kCatmark));

#include <shapes/catmark_tent_creases0.h>
    g_defaultShapes.push_back(SimpleShape(catmark_tent_creases0, "catmark_tent_creases0", kCatmark));

#include <shapes/catmark_tent_creases1.h>
    g_defaultShapes.push_back(SimpleShape(catmark_tent_creases1, "catmark_tent_creases1", kCatmark));

#include <shapes/catmark_tent.h>
    g_defaultShapes.push_back(SimpleShape(catmark_tent, "catmark_tent", kCatmark));

#include <shapes/catmark_torus.h>
    g_defaultShapes.push_back(SimpleShape(catmark_torus, "catmark_torus", kCatmark));

#include <shapes/catmark_torus_creases0.h>
    g_defaultShapes.push_back(SimpleShape(catmark_torus_creases0, "catmark_torus_creases0", kCatmark));

#include <shapes/catmark_square_hedit0.h>
    g_defaultShapes.push_back(SimpleShape(catmark_square_hedit0, "catmark_square_hedit0", kCatmark));

#include <shapes/catmark_square_hedit1.h>
    g_defaultShapes.push_back(SimpleShape(catmark_square_hedit1, "catmark_square_hedit1", kCatmark));

#include <shapes/catmark_square_hedit2.h>
    g_defaultShapes.push_back(SimpleShape(catmark_square_hedit2, "catmark_square_hedit2", kCatmark));

#include <shapes/catmark_square_hedit3.h>
    g_defaultShapes.push_back(SimpleShape(catmark_square_hedit3, "catmark_square_hedit3", kCatmark));

#include <shapes/catmark_square_hedit4.h>
    g_defaultShapes.push_back(SimpleShape(catmark_square_hedit4, "catmark_square_hedit4", kCatmark));

#include <shapes/catmark_bishop.h>
    g_defaultShapes.push_back(SimpleShape(catmark_bishop, "catmark_bishop", kCatmark));

#include <shapes/catmark_car.h>
    g_defaultShapes.push_back(SimpleShape(catmark_car, "catmark_car", kCatmark));

#include <shapes/catmark_helmet.h>
    g_defaultShapes.push_back(SimpleShape(catmark_helmet, "catmark_helmet", kCatmark));

#include <shapes/catmark_pawn.h>
    g_defaultShapes.push_back(SimpleShape(catmark_pawn, "catmark_pawn", kCatmark));

#include <shapes/catmark_rook.h>
    g_defaultShapes.push_back(SimpleShape(catmark_rook, "catmark_rook", kCatmark));

#include <shapes/bilinear_cube.h>
    g_defaultShapes.push_back(SimpleShape(bilinear_cube, "bilinear_cube", kBilinear));


#include <shapes/loop_cube_creases0.h>
    g_defaultShapes.push_back(SimpleShape(loop_cube_creases0, "loop_cube_creases0", kLoop));

#include <shapes/loop_cube_creases1.h>
    g_defaultShapes.push_back(SimpleShape(loop_cube_creases1, "loop_cube_creases1", kLoop));

#include <shapes/loop_cube.h>
    g_defaultShapes.push_back(SimpleShape(loop_cube, "loop_cube", kLoop));

#include <shapes/loop_icosahedron.h>
    g_defaultShapes.push_back(SimpleShape(loop_icosahedron, "loop_icosahedron", kLoop));

#include <shapes/loop_saddle_edgecorner.h>
    g_defaultShapes.push_back(SimpleShape(loop_saddle_edgecorner, "loop_saddle_edgecorner", kLoop));

#include <shapes/loop_saddle_edgeonly.h>
    g_defaultShapes.push_back(SimpleShape(loop_saddle_edgeonly, "loop_saddle_edgeonly", kLoop));

#include <shapes/loop_triangle_edgecorner.h>
    g_defaultShapes.push_back(SimpleShape(loop_triangle_edgecorner, "loop_triangle_edgecorner", kLoop));

#include <shapes/loop_triangle_edgeonly.h>
    g_defaultShapes.push_back(SimpleShape(loop_triangle_edgeonly, "loop_triangle_edgeonly", kLoop));
}

//------------------------------------------------------------------------------
static void
calcNormals(OsdHbrMesh * mesh, std::vector<float> const & pos, std::vector<float> & result ) {

    // calc normal vectors
    int nverts = (int)pos.size()/3;

    int nfaces = mesh->GetNumCoarseFaces();
    for (int i = 0; i < nfaces; ++i) {

        OsdHbrFace * f = mesh->GetFace(i);

        float const * p0 = &pos[f->GetVertex(0)->GetID()*3],
                    * p1 = &pos[f->GetVertex(1)->GetID()*3],
                    * p2 = &pos[f->GetVertex(2)->GetID()*3];

        float n[3];
        cross( n, p0, p1, p2 );

        for (int j = 0; j < f->GetNumVertices(); j++) {
            int idx = f->GetVertex(j)->GetID() * 3;
            result[idx  ] += n[0];
            result[idx+1] += n[1];
            result[idx+2] += n[2];
        }
    }
    for (int i = 0; i < nverts; ++i)
        normalize( &result[i*3] );
}

//------------------------------------------------------------------------------
static void
updateGeom() {

    int nverts = (int)g_orgPositions.size() / 3;

    std::vector<float> vertex;
    vertex.reserve(nverts*6);

    const float *p = &g_orgPositions[0];
    const float *n = &g_normals[0];

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
        vertex.push_back(n[0]);
        vertex.push_back(n[1]);
        vertex.push_back(n[2]);
        
        p += 3;
        n += 3;
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
createOsdMesh( const std::string &shape, int level, int kernel, Scheme scheme=kCatmark ) {

    // generate Hbr representation from "obj" description
    OsdHbrMesh * hmesh = simpleHbr<OpenSubdiv::OsdVertex>(shape.c_str(), scheme, g_orgPositions);

    g_normals.resize(g_orgPositions.size(),0.0f);
    g_positions.resize(g_orgPositions.size(),0.0f);
    calcNormals( hmesh, g_orgPositions, g_normals );

    // save coarse topology (used for coarse mesh drawing)
    g_coarseEdges.clear();
    g_coarseEdgeSharpness.clear();
    g_coarseVertexSharpness.clear();
    int nf = hmesh->GetNumFaces();
    for(int i=0; i<nf; ++i) {
        OsdHbrFace *face = hmesh->GetFace(i);
        int nv = face->GetNumVertices();
        for(int j=0; j<nv; ++j) {
            g_coarseEdges.push_back(face->GetVertex(j)->GetID());
            g_coarseEdges.push_back(face->GetVertex((j+1)%nv)->GetID());
            g_coarseEdgeSharpness.push_back(face->GetEdge(j)->GetSharpness());
        }
    }
    int nv = hmesh->GetNumVertices();
    for(int i=0; i<nv; ++i) {
        g_coarseVertexSharpness.push_back(hmesh->GetVertex(i)->GetSharpness());
    }

    delete g_mesh;
    g_mesh = NULL;

    g_scheme = scheme;

    // Adaptive refinement currently supported only for catmull-clark scheme
    bool doAdaptive = (g_adaptive!=0 and g_scheme==kCatmark);

    OpenSubdiv::OsdMeshBitset bits;
    bits.set(OpenSubdiv::MeshAdaptive, doAdaptive);

    int numVertexElements = 6;
    int numVaryingElements = 0;

    if (g_kernel == kCPU) {
        if (not g_cpuComputeController) {
            g_cpuComputeController = new OpenSubdiv::OsdCpuComputeController();
        }
        g_mesh = new OpenSubdiv::OsdMesh<OpenSubdiv::OsdCpuD3D11VertexBuffer,
                                         OpenSubdiv::OsdCpuComputeController,
                                         OpenSubdiv::OsdD3D11DrawContext>(
                                                g_cpuComputeController,
                                                hmesh,
                                                numVertexElements,
                                                numVaryingElements,
                                                level, bits, g_pd3dDeviceContext);
#ifdef OPENSUBDIV_HAS_OPENMP
    } else if (kernel == kOPENMP) {
        if (not g_ompComputeController) {
            g_ompComputeController = new OpenSubdiv::OsdOmpComputeController();
        }
        g_mesh = new OpenSubdiv::OsdMesh<OpenSubdiv::OsdCpuD3D11VertexBuffer,
                                         OpenSubdiv::OsdOmpComputeController,
                                         OpenSubdiv::OsdD3D11DrawContext>(
                                                g_ompComputeController,
                                                hmesh,
                                                numVertexElements,
                                                numVaryingElements,
                                                level, bits, g_pd3dDeviceContext);
#endif
#ifdef OPENSUBDIV_HAS_OPENCL
    } else if(kernel == kCL) {
        if (not g_clComputeController) {
            g_clComputeController = new OpenSubdiv::OsdCLComputeController(g_clContext, g_clQueue);
        }
        g_mesh = new OpenSubdiv::OsdMesh<OpenSubdiv::OsdCLD3D11VertexBuffer,
                                         OpenSubdiv::OsdCLComputeController,
                                         OpenSubdiv::OsdD3D11DrawContext>(
                                                g_clComputeController,
                                                hmesh,
                                                numVertexElements,
                                                numVaryingElements,
                                                level, bits, g_pd3dDeviceContext);
#endif
#ifdef OPENSUBDIV_HAS_CUDA
    } else if (g_kernel == kCUDA) {
        if (not g_cudaComputeController) {
            g_cudaComputeController = new OpenSubdiv::OsdCudaComputeController();
        }
        g_mesh = new OpenSubdiv::OsdMesh<OpenSubdiv::OsdCudaD3D11VertexBuffer,
                                         OpenSubdiv::OsdCudaComputeController,
                                         OpenSubdiv::OsdD3D11DrawContext>(
                                                g_cudaComputeController,
                                                hmesh,
                                                numVertexElements,
                                                numVaryingElements,
                                                level, bits, g_pd3dDeviceContext);
#endif
    } else if (g_kernel == kDirectCompute) {
        if (not g_d3d11ComputeController) {
            g_d3d11ComputeController = new OpenSubdiv::OsdD3D11ComputeController(g_pd3dDeviceContext);
        }
        g_mesh = new OpenSubdiv::OsdMesh<OpenSubdiv::OsdD3D11VertexBuffer,
                                         OpenSubdiv::OsdD3D11ComputeController,
                                         OpenSubdiv::OsdD3D11DrawContext>(
                                                g_d3d11ComputeController,
                                                hmesh,
                                                numVertexElements,
                                                numVaryingElements,
                                                level, bits, g_pd3dDeviceContext);
    } else {
        printf("Unsupported kernel %s\n", getKernelName(kernel));
    }

    // Hbr mesh can be deleted
    delete hmesh;

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
enum Effect {
    kQuadWire = 0,
    kQuadFill = 1,
    kQuadLine = 2,
    kTriWire = 3,
    kTriFill = 4,
    kTriLine = 5,
    kPoint = 6,
};

typedef std::pair<OpenSubdiv::OsdDrawContext::PatchDescriptor, Effect> EffectDesc;

class EffectDrawRegistry : public OpenSubdiv::OsdD3D11DrawRegistry<EffectDesc> {

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
        DescType const & desc, ID3D11Device * pd3dDevice)
{
    Effect effect = desc.second;

    SourceConfigType * sconfig =
        BaseRegistry::_CreateDrawSourceConfig(desc.first, pd3dDevice);

    sconfig->commonShader.AddDefine("OSD_ENABLE_PATCH_CULL");
    sconfig->commonShader.AddDefine("OSD_ENABLE_SCREENSPACE_TESSELLATION");

    bool smoothNormals = false;
    if (desc.first.GetType() == OpenSubdiv::FarPatchTables::QUADS ||
        desc.first.GetType() == OpenSubdiv::FarPatchTables::TRIANGLES) {
        sconfig->vertexShader.source = shaderSource;
        sconfig->vertexShader.target = "vs_5_0";
        sconfig->vertexShader.entry = "vs_main";
    } else {
        if (effect == kQuadWire) effect = kTriWire;
        if (effect == kQuadFill) effect = kTriFill;
        if (effect == kQuadLine) effect = kTriLine;
        smoothNormals = true;
    }
    assert(sconfig);

    sconfig->geometryShader.source = shaderSource;
    sconfig->geometryShader.target = "gs_5_0";

    sconfig->pixelShader.source = shaderSource;
    sconfig->pixelShader.target = "ps_5_0";

    switch (effect) {
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
        int numInputElements)
{
    ConfigType * config = BaseRegistry::_CreateDrawConfig(desc.first, sconfig,
        pd3dDevice, ppInputLayout, pInputElementDescs, numInputElements);
    assert(config);

    return config;
}

EffectDrawRegistry effectRegistry;

static Effect
GetEffect()
{
    if (g_scheme == kLoop) {
        return (g_wire == 0 ? kTriWire : (g_wire == 1 ? kTriFill : kTriLine));
    } else {
        return (g_wire == 0 ? kQuadWire : (g_wire == 1 ? kQuadFill : kQuadLine));
    }
}

//------------------------------------------------------------------------------
static void
bindProgram(Effect effect, OpenSubdiv::OsdDrawContext::PatchArray const & patch)
{
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
        pData->GregoryQuadOffsetBase = patch.GetQuadOffsetIndex();
        pData->PrimitiveIdBase = patch.GetPatchIndex();

        g_pd3dDeviceContext->Unmap( g_pcbTessellation, 0 );
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
display()
{
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

    OpenSubdiv::OsdDrawContext::PatchArrayVector const & patches = g_mesh->GetDrawContext()->patchArrays;

    g_pd3dDeviceContext->IASetIndexBuffer(g_mesh->GetDrawContext()->patchIndexBuffer, DXGI_FORMAT_R32_UINT, 0);

    // cv drawing
#if 0
    if (g_drawPatchCVs) {

        bindProgram(kPoint, OpenSubdiv::OsdDrawContext::PatchArray());

        g_pd3dDeviceContext->IASetPrimitiveTopology(
                                D3D11_PRIMITIVE_TOPOLOGY_POINTLIST);

        for (int i=0; i<(int)patches.size(); ++i) {
            OpenSubdiv::OsdDrawContext::PatchArray const & patch = patches[i];

            g_pd3dDeviceContext->DrawIndexed(patch.GetNumIndices(),
                                             patch.GetVertIndex(), 0);
        }
    }
#endif

    // patch drawing
    for (int i=0; i<(int)patches.size(); ++i) {
        OpenSubdiv::OsdDrawContext::PatchArray const & patch = patches[i];

        D3D11_PRIMITIVE_TOPOLOGY topology;

        if (g_mesh->GetDrawContext()->IsAdaptive()) {

            switch (patch.GetDescriptor().GetNumControlVertices()) {
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

        g_pd3dDeviceContext->DrawIndexed(
                                    patch.GetNumIndices(), patch.GetVertIndex(), 0);
    }

    if (g_hud->IsVisible()) {
        g_fpsTimer.Stop();
        double fps = 1.0/g_fpsTimer.GetElapsed();
        g_fpsTimer.Start();

        g_hud->DrawString(10, -100, "# of Vertices = %d", g_mesh->GetNumVertices());
        g_hud->DrawString(10, -80, "SUBDIVISION = %s",
                          g_scheme==kBilinear ? "BILINEAR" : (g_scheme == kLoop ? "LOOP" : "CATMARK"));
        g_hud->DrawString(10, -60, "GPU TIME = %.3f ms", g_gpuTime);
        g_hud->DrawString(10, -40, "CPU TIME = %.3f ms", g_cpuTime);
        g_hud->DrawString(10, -20, "FPS = %3.1f", fps);
    }

    g_hud->Flush();

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
    SAFE_RELEASE(g_pDepthStencilView);

    SAFE_RELEASE(g_pSwapChainRTV);
    SAFE_RELEASE(g_pSwapChain);
    SAFE_RELEASE(g_pd3dDeviceContext);
    SAFE_RELEASE(g_pd3dDevice);
    
    delete g_cpuComputeController;

#ifdef OPENSUBDIV_HAS_OPENMP
    delete g_ompComputeController;
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
callbackWireframe(int b)
{
    g_wire = b;
}

static void
callbackKernel(int k)
{
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

    createOsdMesh( g_defaultShapes[ g_currentShape ].data, g_level, g_kernel, g_defaultShapes[ g_currentShape ].scheme );
}

static void
callbackLevel(int l)
{
    g_level = l;
    createOsdMesh( g_defaultShapes[g_currentShape].data, g_level, g_kernel, g_defaultShapes[ g_currentShape ].scheme );
}

static void
callbackModel(int m)
{
    if (m < 0)
        m = 0;

    if (m >= (int)g_defaultShapes.size())
        m = (int)g_defaultShapes.size() - 1;

    g_currentShape = m;

    createOsdMesh( g_defaultShapes[m].data, g_level, g_kernel, g_defaultShapes[ g_currentShape ].scheme );
}

static void
callbackDisplayNormal(bool checked, int n)
{
    g_drawNormals = checked;
}

static void
callbackAnimate(bool checked, int m)
{
    g_moveScale = checked;
}

static void
callbackFreeze(bool checked, int f)
{
    g_freeze = checked;
}

static void
callbackAdaptive(bool checked, int a)
{
    g_adaptive = checked;

    createOsdMesh( g_defaultShapes[g_currentShape].data, g_level, g_kernel, g_defaultShapes[ g_currentShape ].scheme );
}

static void
callbackDisplayCageEdges(bool checked, int d)
{
    g_drawCageEdges = checked;
}

static void
callbackDisplayCageVertices(bool checked, int d)
{
    g_drawCageVertices = checked;
}

static void
callbackDisplayPatchCVs(bool checked, int d)
{
    g_drawPatchCVs = checked;
}

static void
callbackDisplayPatchColor(bool checked, int p)
{
    g_displayPatchColor = checked;
}

static void
initHUD()
{
    g_hud = new D3D11hud(g_pd3dDeviceContext);
    g_hud->Init(g_width, g_height);

    g_hud->AddRadioButton(0, "CPU (K)", true, 10, 10, callbackKernel, kCPU, 'K');
#ifdef OPENSUBDIV_HAS_OPENMP
    g_hud->AddRadioButton(0, "OPENMP", false, 10, 30, callbackKernel, kOPENMP, 'K');
#endif
#ifdef OPENSUBDIV_HAS_CUDA
    g_hud->AddRadioButton(0, "CUDA",   false, 10, 50, callbackKernel, kCUDA, 'K');
#endif
#ifdef OPENSUBDIV_HAS_OPENCL
    g_hud->AddRadioButton(0, "OPENCL", false, 10, 70, callbackKernel, kCL, 'K');
#endif
    g_hud->AddRadioButton(0, "DirectCompute", false, 10, 90, callbackKernel, kDirectCompute, 'K');

    g_hud->AddRadioButton(1, "Wire (W)",    g_wire == 0,  200, 10, callbackWireframe, 0, 'W');
    g_hud->AddRadioButton(1, "Shaded",      g_wire == 1, 200, 30, callbackWireframe, 1, 'W');
    g_hud->AddRadioButton(1, "Wire+Shaded", g_wire == 2, 200, 50, callbackWireframe, 2, 'W');

//    g_hud->AddCheckBox("Cage Edges (H)",    true,  350, 10, callbackDisplayCageEdges, 0, 'H');
//    g_hud->AddCheckBox("Cage Verts (J)", false, 350, 30, callbackDisplayCageVertices, 0, 'J');
    g_hud->AddCheckBox("Patch CVs (L)", false, 350, 50, callbackDisplayPatchCVs, 0, 'L');
//    g_hud->AddCheckBox("Show normal vector (E)", false, 350, 10, callbackDisplayNormal, 0, 'E');
    g_hud->AddCheckBox("Animate vertices (M)", g_moveScale != 0, 350, 70, callbackAnimate, 0, 'M');
    g_hud->AddCheckBox("Patch Color (P)",   true, 350, 90, callbackDisplayPatchColor, 0, 'p');
    g_hud->AddCheckBox("Freeze (spc)", false, 350, 130, callbackFreeze, 0, ' ');

    g_hud->AddCheckBox("Adaptive (`)", true, 10, 150, callbackAdaptive, 0, '`');

    for (int i = 1; i < 11; ++i) {
        char level[16];
        sprintf(level, "Lv. %d", i);
        g_hud->AddRadioButton(3, level, i==2, 10, 170+i*20, callbackLevel, i, '0'+(i%10));
    }

    for(int i = 0; i < (int)g_defaultShapes.size(); ++i){
        g_hud->AddRadioButton(4, g_defaultShapes[i].name.c_str(), i==0, -220, 10+i*16, callbackModel, i, 'N');
    }

    callbackModel(g_currentShape);
}

//------------------------------------------------------------------------------
static bool
initD3D11(HWND hWnd)
{
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
updateRenderTarget(HWND hWnd)
{
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
callbackError(OpenSubdiv::OsdErrorType err, const char *message)
{
    std::ostringstream s;
    s << "OsdError: " << err << "\n";
    s << message;
    OutputDebugString(s.str().c_str());
}

//------------------------------------------------------------------------------
static LRESULT WINAPI
msgProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
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
tokenize(std::string const & src)
{
    std::vector<std::string> result;

    std::stringstream input(src);
    std::copy(std::istream_iterator<std::string>(input),
              std::istream_iterator<std::string>(),
              std::back_inserter< std::vector<std::string> >(result));

    return result;
}

int WINAPI
WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPTSTR lpCmdLine, int nCmdShow)
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
    wcex.hCursor        = NULL;
    wcex.hbrBackground  = (HBRUSH)(COLOR_WINDOW+1);
    wcex.lpszMenuName   = NULL;
    wcex.lpszClassName  = szWindowClass;
    RegisterClass(&wcex);

    // crete window
    RECT rect = { 0, 0, g_width, g_height };
    AdjustWindowRect(&rect, WS_OVERLAPPEDWINDOW, FALSE);

    HWND hWnd = CreateWindow(szWindowClass,
                        "OpenSubdiv DirectX Viewer",
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
            g_defaultShapes.push_back(SimpleShape(str.c_str(), args[i].c_str(), kCatmark));
        }
    }

    initializeShapes();

    OsdSetErrorCallback(callbackError);

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
