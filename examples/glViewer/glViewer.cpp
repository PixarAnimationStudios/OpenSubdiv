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

#include "glLoader.h"

#include <GLFW/glfw3.h>
GLFWwindow* g_window=0;
GLFWmonitor* g_primary=0;

#include <opensubdiv/far/error.h>

#include <opensubdiv/osd/cpuEvaluator.h>
#include <opensubdiv/osd/cpuGLVertexBuffer.h>

#ifdef OPENSUBDIV_HAS_OPENMP
    #include <opensubdiv/osd/ompEvaluator.h>
#endif

#ifdef OPENSUBDIV_HAS_TBB
    #include <opensubdiv/osd/tbbEvaluator.h>
#endif

#ifdef OPENSUBDIV_HAS_OPENCL
    #include <opensubdiv/osd/clGLVertexBuffer.h>
    #include <opensubdiv/osd/clEvaluator.h>
    #include "../common/clDeviceContext.h"

    CLDeviceContext g_clDeviceContext;
#endif

#ifdef OPENSUBDIV_HAS_CUDA
    #include <opensubdiv/osd/cudaGLVertexBuffer.h>
    #include <opensubdiv/osd/cudaEvaluator.h>
    #include "../common/cudaDeviceContext.h"

    CudaDeviceContext g_cudaDeviceContext;
#endif

#ifdef OPENSUBDIV_HAS_GLSL_TRANSFORM_FEEDBACK
    #include <opensubdiv/osd/glXFBEvaluator.h>
    #include <opensubdiv/osd/glVertexBuffer.h>
#endif

#ifdef OPENSUBDIV_HAS_GLSL_COMPUTE
    #include <opensubdiv/osd/glComputeEvaluator.h>
    #include <opensubdiv/osd/glVertexBuffer.h>
#endif

#include <opensubdiv/osd/glMesh.h>
#include <opensubdiv/osd/glLegacyGregoryPatchTable.h>
OpenSubdiv::Osd::GLMeshInterface *g_mesh = NULL;
OpenSubdiv::Osd::GLLegacyGregoryPatchTable *g_legacyGregoryPatchTable = NULL;
bool g_legacyGregoryEnabled = false;

#include "../../regression/common/far_utils.h"
#include "../../regression/common/arg_utils.h"
#include "../common/glHud.h"
#include "../common/glUtils.h"
#include "../common/glControlMeshDisplay.h"
#include "../common/glShaderCache.h"
#include "../common/glUtils.h"
#include "../common/objAnim.h"
#include "../common/simple_math.h"
#include "../common/stopwatch.h"
#include "../common/viewerArgsUtils.h"
#include <opensubdiv/osd/glslPatchShaderSource.h>

static const char *shaderSource(){
    static const char *res = NULL;
    if (!res) {
        static const char *gen =
#include "shader.gen.h"
            ;
        static const char *gen3 =
#include "shader_gl3.gen.h"
            ;
        if (GLUtils::SupportsAdaptiveTessellation()) {
            res = gen;
        } else {
            res = gen3;
        }
    }
    return res;
}

#include <cfloat>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>

enum KernelType { kCPU = 0,
                  kOPENMP = 1,
                  kTBB = 2,
                  kCUDA = 3,
                  kCL = 4,
                  kGLSL = 5,
                  kGLSLCompute = 6 };

enum DisplayStyle { kDisplayStyleWire,
                    kDisplayStyleShaded,
                    kDisplayStyleWireOnShaded };

enum ShadingMode { kShadingMaterial,
                   kShadingVaryingColor,
                   kShadingInterleavedVaryingColor,
                   kShadingFaceVaryingColor,
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

ObjAnim const * g_objAnim = 0;

int   g_frame = 0,
      g_repeatCount = 0;
float g_animTime = 0;

// GUI variables
int   g_freeze = 0,
      g_shadingMode = kShadingPatchType,
      g_displayStyle = kDisplayStyleWireOnShaded,
      g_adaptive = 1,
      g_endCap = kEndCapGregoryBasis,
      g_smoothCornerPatch = 1,
      g_singleCreasePatch = 1,
      g_infSharpPatch = 1,
      g_mbutton[3] = {0, 0, 0},
      g_running = 1;

int   g_screenSpaceTess = 0,
      g_fractionalSpacing = 0,
      g_patchCull = 0,
      g_displayPatchCounts = 0;

float g_rotate[2] = {0, 0},
      g_dolly = 5,
      g_pan[2] = {0, 0},
      g_center[3] = {0, 0, 0},
      g_size = 0;

bool  g_yup = false;

int   g_prev_x = 0,
      g_prev_y = 0;

int   g_width = 1024,
      g_height = 1024;

GLhud g_hud;
GLControlMeshDisplay g_controlMeshDisplay;

// performance
float g_cpuTime = 0;
float g_gpuTime = 0;
Stopwatch g_fpsTimer;

// geometry
std::vector<float> g_orgPositions;

int g_level = 2;
int g_tessLevel = 1;
int g_tessLevelMin = 1;
int g_kernel = kCPU;
float g_moveScale = 0.0f;

GLuint g_queries[2] = {0, 0};

GLuint g_transformUB = 0,
       g_transformBinding = 0,
       g_tessellationUB = 0,
       g_tessellationBinding = 1,
       g_lightingUB = 0,
       g_lightingBinding = 2,
       g_fvarArrayDataUB = 0,
       g_fvarArrayDataBinding = 3;

struct Transform {
    float ModelViewMatrix[16];
    float ProjectionMatrix[16];
    float ModelViewProjectionMatrix[16];
    float ModelViewInverseMatrix[16];
} g_transformData;

GLuint g_vao = 0;

// XXX:
// this struct meant to be used as a stopgap entity until we fully implement
// face-varying stuffs into patch table.
//
struct FVarData
{
    FVarData() :
        textureBuffer(0), textureParamBuffer(0) {
    }
    ~FVarData() {
        Release();
    }
    void Release() {
        if (textureBuffer)
            glDeleteTextures(1, &textureBuffer);
        textureBuffer = 0;
        if (textureParamBuffer)
            glDeleteTextures(1, &textureParamBuffer);
        textureParamBuffer = 0;
    }
    void Create(OpenSubdiv::Far::PatchTable const *patchTable,
                int fvarWidth, std::vector<float> const & fvarSrcData) {

        using namespace OpenSubdiv;

        Release();
        Far::ConstIndexArray indices = patchTable->GetFVarValues();

        // expand fvardata to per-patch array
        std::vector<float> data;
        data.reserve(indices.size() * fvarWidth);

        for (int fvert = 0; fvert < (int)indices.size(); ++fvert) {
            int index = indices[fvert] * fvarWidth;
            for (int i = 0; i < fvarWidth; ++i) {
                data.push_back(fvarSrcData[index++]);
            }
        }
        GLuint buffer;
        glGenBuffers(1, &buffer);
        glBindBuffer(GL_ARRAY_BUFFER, buffer);
        glBufferData(GL_ARRAY_BUFFER, data.size()*sizeof(float),
                     &data[0], GL_STATIC_DRAW);

        glGenTextures(1, &textureBuffer);
        glBindTexture(GL_TEXTURE_BUFFER, textureBuffer);
        glTexBuffer(GL_TEXTURE_BUFFER, GL_R32F, buffer);
        glBindTexture(GL_TEXTURE_BUFFER, 0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        glDeleteBuffers(1, &buffer);

        Far::ConstPatchParamArray fvarParam = patchTable->GetFVarPatchParams();
        glGenBuffers(1, &buffer);
        glBindBuffer(GL_ARRAY_BUFFER, buffer);
        glBufferData(GL_ARRAY_BUFFER, fvarParam.size()*sizeof(Far::PatchParam),
                     &fvarParam[0], GL_STATIC_DRAW);

        glGenTextures(1, &textureParamBuffer);
        glBindTexture(GL_TEXTURE_BUFFER, textureParamBuffer);
        glTexBuffer(GL_TEXTURE_BUFFER, GL_RG32I, buffer);
        glBindTexture(GL_TEXTURE_BUFFER, 0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        glDeleteBuffers(1, &buffer);
    }
    GLuint textureBuffer, textureParamBuffer;
} g_fvarData;

//------------------------------------------------------------------------------

#include "init_shapes.h"

//------------------------------------------------------------------------------
static void
updateGeom() {

    std::vector<float> vertex, varying;

    int nverts = 0;
    int stride = (g_shadingMode == kShadingInterleavedVaryingColor ? 7 : 3);

    if (g_objAnim && g_currentShape==0) {

        nverts = g_objAnim->GetShape()->GetNumVertices(),

        vertex.resize(nverts*stride);

        if (g_shadingMode == kShadingVaryingColor) {
            varying.resize(nverts*4);
        }

        g_objAnim->InterpolatePositions(g_animTime, &vertex[0], stride);

        if (g_shadingMode == kShadingVaryingColor ||
            g_shadingMode == kShadingInterleavedVaryingColor) {

            const float *p = &g_objAnim->GetShape()->verts[0];
            for (int i = 0; i < nverts; ++i) {
                if (g_shadingMode == kShadingInterleavedVaryingColor) {
                    int ofs = i * stride;
                    vertex[ofs + 0] = p[1];
                    vertex[ofs + 1] = p[2];
                    vertex[ofs + 2] = p[0];
                    vertex[ofs + 3] = 0.0f;
                    p += 3;
                }
                if (g_shadingMode == kShadingVaryingColor) {
                    varying.push_back(p[2]);
                    varying.push_back(p[1]);
                    varying.push_back(p[0]);
                    varying.push_back(1);
                    p += 3;
                }
            }
        }
    } else {

        nverts = (int)g_orgPositions.size() / 3;

        vertex.reserve(nverts*stride);

        if (g_shadingMode == kShadingVaryingColor) {
            varying.reserve(nverts*4);
        }

        const float *p = &g_orgPositions[0];
        float r = sin(g_frame*0.001f) * g_moveScale;
        for (int i = 0; i < nverts; ++i) {
            float ct = cos(p[2] * r);
            float st = sin(p[2] * r);
            vertex.push_back( p[0]*ct + p[1]*st);
            vertex.push_back(-p[0]*st + p[1]*ct);
            vertex.push_back( p[2]);
            if (g_shadingMode == kShadingInterleavedVaryingColor) {
                vertex.push_back(p[1]);
                vertex.push_back(p[2]);
                vertex.push_back(p[0]);
                vertex.push_back(1.0f);
            } else if (g_shadingMode == kShadingVaryingColor) {
                varying.push_back(p[2]);
                varying.push_back(p[1]);
                varying.push_back(p[0]);
                varying.push_back(1);
            }
            p += 3;
        }
    }

    g_mesh->UpdateVertexBuffer(&vertex[0], 0, nverts);

    if (g_shadingMode == kShadingVaryingColor)
        g_mesh->UpdateVaryingBuffer(&varying[0], 0, nverts);

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
    else if (kernel == kGLSL)
        return "GLSL TransformFeedback";
    else if (kernel == kGLSLCompute)
        return "GLSL Compute";
    else if (kernel == kCL)
        return "OpenCL";
    return "Unknown";
}

//------------------------------------------------------------------------------
static void
rebuildMesh() {
    using namespace OpenSubdiv;

    ShapeDesc const &shapeDesc = g_defaultShapes[g_currentShape];
    int level = g_level;
    int kernel = g_kernel;
    bool doAnim = g_objAnim && g_currentShape==0;

    Shape const * shape = 0;
    if (doAnim) {
        shape = g_objAnim->GetShape();
    } else {
        shape = Shape::parseObj(shapeDesc);
    }

    // create Far mesh (topology)
    Sdc::SchemeType sdctype = GetSdcType(*shape);
    Sdc::Options sdcoptions = GetSdcOptions(*shape);

    Far::TopologyRefiner * refiner =
        Far::TopologyRefinerFactory<Shape>::Create(*shape,
            Far::TopologyRefinerFactory<Shape>::Options(sdctype, sdcoptions));

    // save coarse topology (used for coarse mesh drawing)
    g_controlMeshDisplay.SetTopology(refiner->GetLevel(0));

    g_orgPositions = shape->verts;

    delete g_mesh;
    g_mesh = NULL;

    Osd::MeshBitset bits;
    bits.set(Osd::MeshAdaptive,             g_adaptive != 0);
    bits.set(Osd::MeshUseSmoothCornerPatch, g_smoothCornerPatch != 0);
    bits.set(Osd::MeshUseSingleCreasePatch, g_singleCreasePatch != 0);
    bits.set(Osd::MeshUseInfSharpPatch,     g_infSharpPatch != 0);
    bits.set(Osd::MeshInterleaveVarying,    g_shadingMode == kShadingInterleavedVaryingColor);
    bits.set(Osd::MeshFVarData,             g_shadingMode == kShadingFaceVaryingColor);
    bits.set(Osd::MeshEndCapBilinearBasis,  g_endCap == kEndCapBilinearBasis);
    bits.set(Osd::MeshEndCapBSplineBasis,   g_endCap == kEndCapBSplineBasis);
    bits.set(Osd::MeshEndCapGregoryBasis,   g_endCap == kEndCapGregoryBasis);
    bits.set(Osd::MeshEndCapLegacyGregory,  g_endCap == kEndCapLegacyGregory);

    int numVertexElements = 3;
    int numVaryingElements =
        (g_shadingMode == kShadingVaryingColor || bits.test(Osd::MeshInterleaveVarying)) ? 4 : 0;


    if (kernel == kCPU) {
        g_mesh = new Osd::Mesh<Osd::CpuGLVertexBuffer,
                               Far::StencilTable,
                               Osd::CpuEvaluator,
                               Osd::GLPatchTable>(
                                   refiner,
                                   numVertexElements,
                                   numVaryingElements,
                                   level, bits);
#ifdef OPENSUBDIV_HAS_OPENMP
    } else if (kernel == kOPENMP) {
        g_mesh = new Osd::Mesh<Osd::CpuGLVertexBuffer,
                               Far::StencilTable,
                               Osd::OmpEvaluator,
                               Osd::GLPatchTable>(
                                   refiner,
                                   numVertexElements,
                                   numVaryingElements,
                                   level, bits);
#endif
#ifdef OPENSUBDIV_HAS_TBB
    } else if (kernel == kTBB) {
        g_mesh = new Osd::Mesh<Osd::CpuGLVertexBuffer,
                               Far::StencilTable,
                               Osd::TbbEvaluator,
                               Osd::GLPatchTable>(
                                   refiner,
                                   numVertexElements,
                                   numVaryingElements,
                                   level, bits);
#endif
#ifdef OPENSUBDIV_HAS_OPENCL
    } else if(kernel == kCL) {
        // CLKernel
        static Osd::EvaluatorCacheT<Osd::CLEvaluator> clEvaluatorCache;
        g_mesh = new Osd::Mesh<Osd::CLGLVertexBuffer,
                               Osd::CLStencilTable,
                               Osd::CLEvaluator,
                               Osd::GLPatchTable,
                               CLDeviceContext>(
                                   refiner,
                                   numVertexElements,
                                   numVaryingElements,
                                   level, bits,
                                   &clEvaluatorCache,
                                   &g_clDeviceContext);
#endif
#ifdef OPENSUBDIV_HAS_CUDA
    } else if(kernel == kCUDA) {
        g_mesh = new Osd::Mesh<Osd::CudaGLVertexBuffer,
                               Osd::CudaStencilTable,
                               Osd::CudaEvaluator,
                               Osd::GLPatchTable>(
                                   refiner,
                                   numVertexElements,
                                   numVaryingElements,
                                   level, bits);
#endif
#ifdef OPENSUBDIV_HAS_GLSL_TRANSFORM_FEEDBACK
    } else if(kernel == kGLSL) {
        static Osd::EvaluatorCacheT<Osd::GLXFBEvaluator> glXFBEvaluatorCache;
        g_mesh = new Osd::Mesh<Osd::GLVertexBuffer,
                               Osd::GLStencilTableTBO,
                               Osd::GLXFBEvaluator,
                               Osd::GLPatchTable>(
                                   refiner,
                                   numVertexElements,
                                   numVaryingElements,
                                   level, bits,
                                   &glXFBEvaluatorCache);
#endif
#ifdef OPENSUBDIV_HAS_GLSL_COMPUTE
    } else if(kernel == kGLSLCompute) {
        static Osd::EvaluatorCacheT<Osd::GLComputeEvaluator> glComputeEvaluatorCache;
        g_mesh = new Osd::Mesh<Osd::GLVertexBuffer,
                               Osd::GLStencilTableSSBO,
                               Osd::GLComputeEvaluator,
                               Osd::GLPatchTable>(
                                   refiner,
                                   numVertexElements,
                                   numVaryingElements,
                                   level, bits,
                                   &glComputeEvaluatorCache);


#endif
    } else {
        printf("Unsupported kernel %s\n", getKernelName(kernel));
    }

    if (g_shadingMode == kShadingFaceVaryingColor && shape->HasUV()) {

        std::vector<float> fvarData;

        InterpolateFVarData(*refiner, *shape, fvarData);

        // set fvardata to texture buffer
        g_fvarData.Create(g_mesh->GetFarPatchTable(),
                          shape->GetFVarWidth(), fvarData);
    }

    // legacy gregory
    delete g_legacyGregoryPatchTable;
    g_legacyGregoryPatchTable = NULL;
    if (g_endCap == kEndCapLegacyGregory) {
        g_legacyGregoryPatchTable =
            Osd::GLLegacyGregoryPatchTable::Create(g_mesh->GetFarPatchTable());
    }

    if (! doAnim) {
        delete shape;
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

    // -------- VAO
    glBindVertexArray(g_vao);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_mesh->GetPatchTable()->GetPatchIndexBuffer());
    glBindBuffer(GL_ARRAY_BUFFER, g_mesh->BindVertexBuffer());

    glEnableVertexAttribArray(0);

    if (g_shadingMode == kShadingVaryingColor) {
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof (GLfloat) * 3, 0);
        glBindBuffer(GL_ARRAY_BUFFER, g_mesh->BindVaryingBuffer());
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof (GLfloat) * 4, 0);
    } else if (g_shadingMode == kShadingInterleavedVaryingColor) {
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof (GLfloat) * 7, 0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof (GLfloat) * 7, (void*)(sizeof (GLfloat) * 3));
    } else {
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof (GLfloat) * 3, 0);
        glDisableVertexAttribArray(1);
    }

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
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

// ---------------------------------------------------------------------------

class ShaderCache : public GLShaderCache<EffectDesc> {
public:
    virtual GLDrawConfig *CreateDrawConfig(EffectDesc const &effectDesc) {

        using namespace OpenSubdiv;

        // compile shader program

        GLDrawConfig *config = new GLDrawConfig(GLUtils::GetShaderVersionInclude().c_str());

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
        switch (effectDesc.effect.displayStyle) {
        case kDisplayStyleWire:
            ss << "#define GEOMETRY_OUT_WIRE\n";
            break;
        case kDisplayStyleWireOnShaded:
            ss << "#define GEOMETRY_OUT_LINE\n";
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
        case kShadingVaryingColor:
            ss << "#define SHADING_VARYING_COLOR\n";
            break;
        case kShadingInterleavedVaryingColor:
            ss << "#define SHADING_VARYING_COLOR\n";
            break;
        case kShadingFaceVaryingColor:
            ss << "#define OSD_FVAR_WIDTH 2\n";
            ss << "#define SHADING_FACEVARYING_COLOR\n";
            if (! effectDesc.desc.IsAdaptive()) {
                ss << "#define SHADING_FACEVARYING_UNIFORM_SUBDIVISION\n";
            }
            break;
        case kShadingPatchType:
            ss << "#define SHADING_PATCH_TYPE\n";
            break;
        case kShadingPatchDepth:
            ss << "#define SHADING_PATCH_DEPTH\n";
            break;
        case kShadingPatchCoord:
            ss << "#define SHADING_PATCH_COORD\n";
            break;
        case kShadingNormal:
            ss << "#define SHADING_NORMAL\n";
            break;
        }

        if (type != Far::PatchDescriptor::TRIANGLES &&
            type != Far::PatchDescriptor::QUADS) {
            ss << "#define SMOOTH_NORMALS\n";
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
        ss << "#define OSD_PATCH_BASIS_GLSL\n";
        ss << Osd::GLSLPatchShaderSource::GetPatchBasisShaderSource();
        ss << Osd::GLSLPatchShaderSource::GetCommonShaderSource();
        std::string common = ss.str();
        ss.str("");

        // vertex shader
        ss << common
            // enable local vertex shader
           << (effectDesc.desc.IsAdaptive() ? "" : "#define VERTEX_SHADER\n")
           << shaderSource()
           << Osd::GLSLPatchShaderSource::GetVertexShaderSource(type);
        config->CompileAndAttachShader(GL_VERTEX_SHADER, ss.str());
        ss.str("");

        if (effectDesc.desc.IsAdaptive()) {
            // tess control shader
            ss << common
                << shaderSource()
               << Osd::GLSLPatchShaderSource::GetTessControlShaderSource(type);
            config->CompileAndAttachShader(GL_TESS_CONTROL_SHADER, ss.str());
            ss.str("");

            // tess eval shader
            ss << common
                << shaderSource()
               << Osd::GLSLPatchShaderSource::GetTessEvalShaderSource(type);
            config->CompileAndAttachShader(GL_TESS_EVALUATION_SHADER, ss.str());
            ss.str("");
        }

        // geometry shader
        ss << common
           << "#define GEOMETRY_SHADER\n"
           << shaderSource();
        config->CompileAndAttachShader(GL_GEOMETRY_SHADER, ss.str());
        ss.str("");

        // fragment shader
        ss << common
           << "#define FRAGMENT_SHADER\n"
           << shaderSource();
        config->CompileAndAttachShader(GL_FRAGMENT_SHADER, ss.str());
        ss.str("");

        if (!config->Link()) {
            delete config;
            return NULL;
        }

        // assign uniform locations
        GLuint uboIndex;
        GLuint program = config->GetProgram();
        uboIndex = glGetUniformBlockIndex(program, "Transform");
        if (uboIndex != GL_INVALID_INDEX)
            glUniformBlockBinding(program, uboIndex, g_transformBinding);

        uboIndex = glGetUniformBlockIndex(program, "Tessellation");
        if (uboIndex != GL_INVALID_INDEX)
            glUniformBlockBinding(program, uboIndex, g_tessellationBinding);

        uboIndex = glGetUniformBlockIndex(program, "Lighting");
        if (uboIndex != GL_INVALID_INDEX)
            glUniformBlockBinding(program, uboIndex, g_lightingBinding);

        uboIndex = glGetUniformBlockIndex(program, "OsdFVarArrayData");
        if (uboIndex != GL_INVALID_INDEX)
            glUniformBlockBinding(program, uboIndex, g_fvarArrayDataBinding);

        // assign texture locations
        GLint loc;
        glUseProgram(program);
        if ((loc = glGetUniformLocation(program, "OsdPatchParamBuffer")) != -1) {
            glUniform1i(loc, 0); // GL_TEXTURE0
        }
        if ((loc = glGetUniformLocation(program, "OsdFVarDataBuffer")) != -1) {
            glUniform1i(loc, 1); // GL_TEXTURE1
        }
        if ((loc = glGetUniformLocation(program, "OsdFVarParamBuffer")) != -1) {
            glUniform1i(loc, 2); // GL_TEXTURE2
        }
        // for legacy gregory patches
        if ((loc = glGetUniformLocation(program, "OsdVertexBuffer")) != -1) {
            glUniform1i(loc, 3); // GL_TEXTURE3
        }
        if ((loc = glGetUniformLocation(program, "OsdValenceBuffer")) != -1) {
            glUniform1i(loc, 4); // GL_TEXTURE4
        }
        if ((loc = glGetUniformLocation(program, "OsdQuadOffsetBuffer")) != -1) {
            glUniform1i(loc, 5); // GL_TEXTURE5
        }
        glUseProgram(0);

        return config;
    }
};

ShaderCache g_shaderCache;

//------------------------------------------------------------------------------
static void
updateUniformBlocks() {

    using namespace OpenSubdiv;

    if (! g_transformUB) {
        glGenBuffers(1, &g_transformUB);
        glBindBuffer(GL_UNIFORM_BUFFER, g_transformUB);
        glBufferData(GL_UNIFORM_BUFFER,
                sizeof(g_transformData), NULL, GL_STATIC_DRAW);
    };
    glBindBuffer(GL_UNIFORM_BUFFER, g_transformUB);
    glBufferSubData(GL_UNIFORM_BUFFER,
                0, sizeof(g_transformData), &g_transformData);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    glBindBufferBase(GL_UNIFORM_BUFFER, g_transformBinding, g_transformUB);

    // Update and bind tessellation state
    struct Tessellation {
        float TessLevel;
    } tessellationData;

    tessellationData.TessLevel = static_cast<float>(1 << g_tessLevel);

    if (! g_tessellationUB) {
        glGenBuffers(1, &g_tessellationUB);
        glBindBuffer(GL_UNIFORM_BUFFER, g_tessellationUB);
        glBufferData(GL_UNIFORM_BUFFER,
                sizeof(tessellationData), NULL, GL_STATIC_DRAW);
    };
    glBindBuffer(GL_UNIFORM_BUFFER, g_tessellationUB);
    glBufferSubData(GL_UNIFORM_BUFFER,
                0, sizeof(tessellationData), &tessellationData);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    glBindBufferBase(GL_UNIFORM_BUFFER, g_tessellationBinding, g_tessellationUB);

    // Update and bind fvar patch array state
    if (g_mesh->GetPatchTable()->GetNumFVarChannels() > 0) {
        Osd::PatchArrayVector const &fvarPatchArrays =
            g_mesh->GetPatchTable()->GetFVarPatchArrays();

        // bind patch arrays UBO (std140 struct size padded to vec4 alignment)
        int patchArraySize =
            sizeof(GLint) * ((sizeof(Osd::PatchArray)/sizeof(GLint) + 3) & ~3);
        if (!g_fvarArrayDataUB) {
            glGenBuffers(1, &g_fvarArrayDataUB);
        }
        glBindBuffer(GL_UNIFORM_BUFFER, g_fvarArrayDataUB);
        glBufferData(GL_UNIFORM_BUFFER,
            fvarPatchArrays.size()*patchArraySize, NULL, GL_STATIC_DRAW);
        for (int i=0; i<(int)fvarPatchArrays.size(); ++i) {
            glBufferSubData(GL_UNIFORM_BUFFER,
                i*patchArraySize, sizeof(Osd::PatchArray), &fvarPatchArrays[i]);
        }

        glBindBufferBase(GL_UNIFORM_BUFFER,
                g_fvarArrayDataBinding, g_fvarArrayDataUB);
    }

    // Update and bind lighting state
    struct Lighting {
        struct Light {
            float position[4];
            float ambient[4];
            float diffuse[4];
            float specular[4];
        } lightSource[2];
    } lightingData = {
       {{  { 0.5,  0.2f, 1.0f, 0.0f },
           { 0.1f, 0.1f, 0.1f, 1.0f },
           { 0.7f, 0.7f, 0.7f, 1.0f },
           { 0.8f, 0.8f, 0.8f, 1.0f } },

         { { -0.8f, 0.4f, -1.0f, 0.0f },
           {  0.0f, 0.0f,  0.0f, 1.0f },
           {  0.5f, 0.5f,  0.5f, 1.0f },
           {  0.8f, 0.8f,  0.8f, 1.0f } }}
    };
    if (! g_lightingUB) {
        glGenBuffers(1, &g_lightingUB);
        glBindBuffer(GL_UNIFORM_BUFFER, g_lightingUB);
        glBufferData(GL_UNIFORM_BUFFER,
                sizeof(lightingData), NULL, GL_STATIC_DRAW);
    };
    glBindBuffer(GL_UNIFORM_BUFFER, g_lightingUB);
    glBufferSubData(GL_UNIFORM_BUFFER,
                0, sizeof(lightingData), &lightingData);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    glBindBufferBase(GL_UNIFORM_BUFFER, g_lightingBinding, g_lightingUB);
}

static void
bindTextures() {
    // bind patch textures
    if (g_mesh->GetPatchTable()->GetPatchParamTextureBuffer()) {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_BUFFER,
            g_mesh->GetPatchTable()->GetPatchParamTextureBuffer());
    }

    if (true) {
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_BUFFER,
                      g_fvarData.textureBuffer);
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_BUFFER,
                      g_fvarData.textureParamBuffer);
    }

    // legacy gregory
    if (g_legacyGregoryPatchTable) {
        glActiveTexture(GL_TEXTURE3);
        glBindTexture(GL_TEXTURE_BUFFER,
                      g_legacyGregoryPatchTable->GetVertexTextureBuffer());
        glActiveTexture(GL_TEXTURE4);
        glBindTexture(GL_TEXTURE_BUFFER,
                      g_legacyGregoryPatchTable->GetVertexValenceTextureBuffer());
        glActiveTexture(GL_TEXTURE5);
        glBindTexture(GL_TEXTURE_BUFFER,
                      g_legacyGregoryPatchTable->GetQuadOffsetsTextureBuffer());
    }
    glActiveTexture(GL_TEXTURE0);
}

static GLenum
bindProgram(Effect effect,
            OpenSubdiv::Osd::PatchArray const & patch) {
    EffectDesc effectDesc(patch.GetDescriptor(), effect);

    // only legacy gregory needs maxValence and numElements
    // neither legacy gregory nor gregory basis need single crease
    typedef OpenSubdiv::Far::PatchDescriptor Descriptor;
    if (patch.GetDescriptor().GetType() == Descriptor::GREGORY ||
        patch.GetDescriptor().GetType() == Descriptor::GREGORY_BOUNDARY) {
        int maxValence = g_mesh->GetMaxValence();
        int numElements = (g_shadingMode == kShadingInterleavedVaryingColor ? 7 : 3);
        effectDesc.maxValence = maxValence;
        effectDesc.numElements = numElements;
        effectDesc.effect.singleCreasePatch = 0;
    }
    if (patch.GetDescriptor().GetType() == Descriptor::GREGORY_BASIS) {
        effectDesc.effect.singleCreasePatch = 0;
    }

    // lookup shader cache (compile the shader if needed)
    GLDrawConfig *config = g_shaderCache.GetDrawConfig(effectDesc);
    if (!config) return 0;

    GLuint program = config->GetProgram();

    glUseProgram(program);

    // bind standalone uniforms
    GLint uniformPrimitiveIdBase =
        glGetUniformLocation(program, "PrimitiveIdBase");
    if (uniformPrimitiveIdBase >=0)
        glUniform1i(uniformPrimitiveIdBase, patch.GetPrimitiveIdBase());

    // legacy gregory
    if (g_endCap == kEndCapLegacyGregory) {
        GLint uniformGregoryQuadOffsetBase =
            glGetUniformLocation(program, "GregoryQuadOffsetBase");
        int quadOffsetBase =
            g_legacyGregoryPatchTable->GetQuadOffsetsBase(patch.GetDescriptor().GetType());
        if (uniformGregoryQuadOffsetBase >= 0)
            glUniform1i(uniformGregoryQuadOffsetBase, quadOffsetBase);
    }

    // update uniform
    GLint uniformDiffuseColor =
        glGetUniformLocation(program, "diffuseColor");
    if (uniformDiffuseColor >= 0)
        glUniform4f(uniformDiffuseColor, 0.4f, 0.4f, 0.8f, 1);

    // return primtype
    GLenum primType;
    switch(effectDesc.desc.GetType()) {
    case Descriptor::QUADS:
        primType = GL_LINES_ADJACENCY;
        break;
    case Descriptor::TRIANGLES:
        primType = GL_TRIANGLES;
        break;
    default:
#if defined(GL_ARB_tessellation_shader) || defined(GL_VERSION_4_0)
        primType = GL_PATCHES;
        glPatchParameteri(GL_PATCH_VERTICES, effectDesc.desc.GetNumControlVertices());
#else
        primType = GL_POINTS;
#endif
        break;
    }

    return primType;
}

//------------------------------------------------------------------------------
static void
display() {
    Stopwatch s;
    s.Start();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glViewport(0, 0, g_width, g_height);
    g_hud.FillBackground();

    // prepare view matrix
    double aspect = g_width/(double)g_height;
    identity(g_transformData.ModelViewMatrix);
    translate(g_transformData.ModelViewMatrix, -g_pan[0], -g_pan[1], -g_dolly);
    rotate(g_transformData.ModelViewMatrix, g_rotate[1], 1, 0, 0);
    rotate(g_transformData.ModelViewMatrix, g_rotate[0], 0, 1, 0);
    if (!g_yup) {
        rotate(g_transformData.ModelViewMatrix, -90, 1, 0, 0);
    }
    translate(g_transformData.ModelViewMatrix,
              -g_center[0], -g_center[1], -g_center[2]);
    perspective(g_transformData.ProjectionMatrix,
                45.0f, (float)aspect, 0.1f, 500.0f);
    multMatrix(g_transformData.ModelViewProjectionMatrix,
               g_transformData.ModelViewMatrix,
               g_transformData.ProjectionMatrix);
    inverseMatrix(g_transformData.ModelViewInverseMatrix,
                  g_transformData.ModelViewMatrix);

    // make sure that the vertex buffer is interoped back as a GL resource.
    GLuint vbo = g_mesh->BindVertexBuffer();

    // vertex texture update for legacy gregory drawing
    if (g_legacyGregoryPatchTable) {
        glActiveTexture(GL_TEXTURE1);
        g_legacyGregoryPatchTable->UpdateVertexBuffer(vbo);
    }

    if (g_shadingMode == kShadingVaryingColor)
        g_mesh->BindVaryingBuffer();

    // update transform and lighting uniform blocks
    updateUniformBlocks();

    // also bind patch related textures
    bindTextures();

    if (g_displayStyle == kDisplayStyleWire)
        glDisable(GL_CULL_FACE);

    glEnable(GL_DEPTH_TEST);

    glBindVertexArray(g_vao);

    OpenSubdiv::Osd::PatchArrayVector const & patches =
        g_mesh->GetPatchTable()->GetPatchArrays();

    // patch drawing
    int patchCount[13]; // [Type] (see far/patchTable.h)
    int numTotalPatches = 0;
    int numDrawCalls = 0;
    memset(patchCount, 0, sizeof(patchCount));

    // primitive counting
    glBeginQuery(GL_PRIMITIVES_GENERATED, g_queries[0]);
#if defined(GL_VERSION_3_3)
    glBeginQuery(GL_TIME_ELAPSED, g_queries[1]);
#endif

    // core draw-calls
    for (int i=0; i<(int)patches.size(); ++i) {
        OpenSubdiv::Osd::PatchArray const & patch = patches[i];

        OpenSubdiv::Far::PatchDescriptor desc = patch.GetDescriptor();
        OpenSubdiv::Far::PatchDescriptor::Type patchType = desc.GetType();

        patchCount[patchType] += patch.GetNumPatches();
        numTotalPatches += patch.GetNumPatches();

        GLenum primType = bindProgram(GetEffect(), patch);


        glDrawElements(primType,
                       patch.GetNumPatches() * desc.GetNumControlVertices(),
                       GL_UNSIGNED_INT,
                       (void *)(patch.GetIndexBase() * sizeof(unsigned int)));
        ++numDrawCalls;
    }

    s.Stop();
    float drawCpuTime = float(s.GetElapsed() * 1000.0f);

    glEndQuery(GL_PRIMITIVES_GENERATED);
#if defined(GL_VERSION_3_3)
    glEndQuery(GL_TIME_ELAPSED);
#endif

    glBindVertexArray(0);

    glUseProgram(0);

    if (g_displayStyle == kDisplayStyleWire)
        glEnable(GL_CULL_FACE);

    // draw the control mesh
    int stride = g_shadingMode == kShadingInterleavedVaryingColor ? 7 : 3;
    g_controlMeshDisplay.Draw(vbo, stride*sizeof(float),
                              g_transformData.ModelViewProjectionMatrix);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    GLuint numPrimsGenerated = 0;
    GLuint timeElapsed = 0;
    glGetQueryObjectuiv(g_queries[0], GL_QUERY_RESULT, &numPrimsGenerated);
#if defined(GL_VERSION_3_3)
    glGetQueryObjectuiv(g_queries[1], GL_QUERY_RESULT, &timeElapsed);
#endif

    float drawGpuTime = timeElapsed / 1000.0f / 1000.0f;

    g_fpsTimer.Stop();
    float elapsed = (float)g_fpsTimer.GetElapsed();
    if (! g_freeze) {
        g_animTime += elapsed;
    }
    g_fpsTimer.Start();

    if (g_hud.IsVisible()) {

        typedef OpenSubdiv::Far::PatchDescriptor Descriptor;

        double fps = 1.0/elapsed;

        if (g_displayPatchCounts) {
            int x = -420;
            int y = g_legacyGregoryEnabled ? -180 : -140;
            g_hud.DrawString(x, y, "Quads            : %d",
                             patchCount[Descriptor::QUADS]); y += 20;
            g_hud.DrawString(x, y, "Triangles        : %d",
                             patchCount[Descriptor::TRIANGLES]); y += 20;
            g_hud.DrawString(x, y, "Regular          : %d",
                             patchCount[Descriptor::REGULAR]); y+= 20;
            g_hud.DrawString(x, y, "Loop             : %d",
                             patchCount[Descriptor::LOOP]); y+= 20;
            if (g_legacyGregoryEnabled) {
                g_hud.DrawString(x, y, "Gregory          : %d",
                                 patchCount[Descriptor::GREGORY]); y+= 20;
                g_hud.DrawString(x, y, "Gregory Boundary : %d",
                                 patchCount[Descriptor::GREGORY_BOUNDARY]); y+= 20;
            }
            g_hud.DrawString(x, y, "Gregory Basis    : %d",
                             patchCount[Descriptor::GREGORY_BASIS]); y+= 20;
            g_hud.DrawString(x, y, "Gregory Triangle : %d",
                             patchCount[Descriptor::GREGORY_TRIANGLE]); y+= 20;
        }

        int y = -220;
        g_hud.DrawString(10, y, "Tess level : %d", g_tessLevel); y+= 20;
        g_hud.DrawString(10, y, "Patches    : %d", numTotalPatches); y+= 20;
        g_hud.DrawString(10, y, "Draw calls : %d", numDrawCalls); y+= 20;
        g_hud.DrawString(10, y, "Primitives : %d", numPrimsGenerated); y+= 20;
        g_hud.DrawString(10, y, "Vertices   : %d", g_mesh->GetNumVertices()); y+= 20;
        g_hud.DrawString(10, y, "GPU Kernel : %.3f ms", g_gpuTime); y+= 20;
        g_hud.DrawString(10, y, "CPU Kernel : %.3f ms", g_cpuTime); y+= 20;
        g_hud.DrawString(10, y, "GPU Draw   : %.3f ms", drawGpuTime); y+= 20;
        g_hud.DrawString(10, y, "CPU Draw   : %.3f ms", drawCpuTime); y+= 20;
        g_hud.DrawString(10, y, "FPS        : %3.1f", fps); y+= 20;

        g_hud.Flush();
    }

    glFinish();

    GLUtils::CheckGLErrors("display leave\n");
}

//------------------------------------------------------------------------------
static void
motion(GLFWwindow *, double dx, double dy) {

    int x=(int)dx, y=(int)dy;

    if (g_hud.MouseCapture()) {
        // check gui
        g_hud.MouseMotion(x, y);
    } else if (g_mbutton[0] && !g_mbutton[1] && !g_mbutton[2]) {
        // orbit
        g_rotate[0] += x - g_prev_x;
        g_rotate[1] += y - g_prev_y;
    } else if (!g_mbutton[0] && !g_mbutton[1] && g_mbutton[2]) {
        // pan
        g_pan[0] -= g_dolly*(x - g_prev_x)/g_width;
        g_pan[1] += g_dolly*(y - g_prev_y)/g_height;
    } else if ((g_mbutton[0] && !g_mbutton[1] && g_mbutton[2]) ||
               (!g_mbutton[0] && g_mbutton[1] && !g_mbutton[2])) {
        // dolly
        g_dolly -= g_dolly*0.01f*(x - g_prev_x);
        if(g_dolly <= 0.01) g_dolly = 0.01f;
    }

    g_prev_x = x;
    g_prev_y = y;
}

//------------------------------------------------------------------------------
static void
mouse(GLFWwindow *, int button, int state, int /* mods */) {

    if (state == GLFW_RELEASE)
        g_hud.MouseRelease();

    if (button == 0 && state == GLFW_PRESS && g_hud.MouseClick(g_prev_x, g_prev_y))
        return;

    if (button < 3) {
        g_mbutton[button] = (state == GLFW_PRESS);
    }
}

//------------------------------------------------------------------------------
static void
uninitGL() {

    glDeleteQueries(2, g_queries);
    glDeleteVertexArrays(1, &g_vao);

    if (g_mesh)
        delete g_mesh;

    if (g_legacyGregoryPatchTable)
        delete g_legacyGregoryPatchTable;

}

//------------------------------------------------------------------------------
static void
reshape(GLFWwindow *, int width, int height) {

    g_width = width;
    g_height = height;

    int windowWidth = g_width, windowHeight = g_height;

    // window size might not match framebuffer size on a high DPI display
    glfwGetWindowSize(g_window, &windowWidth, &windowHeight);

    g_hud.Rebuild(windowWidth, windowHeight, width, height);
}

//------------------------------------------------------------------------------
void windowClose(GLFWwindow*) {
    g_running = false;
}

//------------------------------------------------------------------------------
static void
toggleFullScreen() {
    // XXXX manuelk : to re-implement from glut
}

//------------------------------------------------------------------------------
static void
keyboard(GLFWwindow *, int key, int /* scancode */, int event, int /* mods */) {

    if (event == GLFW_RELEASE) return;
    if (g_hud.KeyDown(tolower(key))) return;

    switch (key) {
        case 'Q': g_running = 0; break;
        case 'F': fitFrame(); break;
        case GLFW_KEY_TAB: toggleFullScreen(); break;
        case '+':
        case '=':  g_tessLevel++; break;
        case '-':  g_tessLevel = std::max(g_tessLevelMin, g_tessLevel-1); break;
        case GLFW_KEY_ESCAPE: g_hud.SetVisible(!g_hud.IsVisible()); break;
        case 'X': GLUtils::WriteScreenshot(g_width, g_height); break;
    }
}

//------------------------------------------------------------------------------
static void
callbackDisplayStyle(int b) {
    g_displayStyle = b;
}

static void
callbackShadingMode(int b) {
    if (g_shadingMode == kShadingVaryingColor || b == kShadingVaryingColor ||
        g_shadingMode == kShadingInterleavedVaryingColor || b == kShadingInterleavedVaryingColor ||
        g_shadingMode == kShadingFaceVaryingColor || b == kShadingFaceVaryingColor) {
        // need to rebuild for varying reconstruct
        g_shadingMode = b;
        rebuildMesh();
        return;
    }
    g_shadingMode = b;
}

static void
callbackEndCap(int endCap) {
    g_endCap = endCap;
    rebuildMesh();
}

static void
callbackKernel(int k) {
    g_kernel = k;

#ifdef OPENSUBDIV_HAS_OPENCL
    if (g_kernel == kCL && (!g_clDeviceContext.IsInitialized())) {
        if (g_clDeviceContext.Initialize() == false) {
            printf("Error in initializing OpenCL\n");
            exit(1);
        }
    }
#endif
#ifdef OPENSUBDIV_HAS_CUDA
    if (g_kernel == kCUDA && (!g_cudaDeviceContext.IsInitialized())) {
        if (g_cudaDeviceContext.Initialize() == false) {
            printf("Error in initializing Cuda\n");
            exit(1);
        }
    }
#endif

    rebuildMesh();
}

static void
callbackLevel(int l) {
    g_level = l;
    rebuildMesh();
}

static void
callbackModel(int m) {
    if (m < 0)
        m = 0;

    if (m >= (int)g_defaultShapes.size())
        m = (int)g_defaultShapes.size() - 1;

    g_currentShape = m;
    rebuildMesh();
}

static void
callbackCheckBox(bool checked, int button) {

    if (GLUtils::SupportsAdaptiveTessellation()) {
        switch(button) {
        case kHUD_CB_ADAPTIVE:
            g_adaptive = checked;
            rebuildMesh();
            return;
        case kHUD_CB_SMOOTH_CORNER_PATCH:
            g_smoothCornerPatch = checked;
            rebuildMesh();
            return;
        case kHUD_CB_SINGLE_CREASE_PATCH:
            g_singleCreasePatch = checked;
            rebuildMesh();
            return;
        case kHUD_CB_INF_SHARP_PATCH:
            g_infSharpPatch = checked;
            rebuildMesh();
            return;
        default:
            break;
        }
    }

    switch (button) {
    case kHUD_CB_DISPLAY_CONTROL_MESH_EDGES:
        g_controlMeshDisplay.SetEdgesDisplay(checked);
        break;
    case kHUD_CB_DISPLAY_CONTROL_MESH_VERTS:
        g_controlMeshDisplay.SetVerticesDisplay(checked);
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
    }
}

static void
initHUD() {
    int windowWidth = g_width, windowHeight = g_height;
    int frameBufferWidth = g_width, frameBufferHeight = g_height;

    // window size might not match framebuffer size on a high DPI display
    glfwGetWindowSize(g_window, &windowWidth, &windowHeight);
    glfwGetFramebufferSize(g_window, &frameBufferWidth, &frameBufferHeight);

    g_hud.Init(windowWidth, windowHeight, frameBufferWidth, frameBufferHeight);

    int y = 10;
    g_hud.AddCheckBox("Control edges (H)",
                      g_controlMeshDisplay.GetEdgesDisplay(),
                      10, y, callbackCheckBox,
                      kHUD_CB_DISPLAY_CONTROL_MESH_EDGES, 'h');
    y += 20;
    g_hud.AddCheckBox("Control vertices (J)",
                      g_controlMeshDisplay.GetVerticesDisplay(),
                      10, y, callbackCheckBox,
                      kHUD_CB_DISPLAY_CONTROL_MESH_VERTS, 'j');
    y += 20;
    g_hud.AddCheckBox("Animate vertices (M)", g_moveScale != 0,
                      10, y, callbackCheckBox, kHUD_CB_ANIMATE_VERTICES, 'm');
    y += 20;
    g_hud.AddCheckBox("Screen space LOD (V)",  g_screenSpaceTess != 0,
                      10, y, callbackCheckBox, kHUD_CB_VIEW_LOD, 'v');
    y += 20;
    g_hud.AddCheckBox("Fractional spacing (T)",  g_fractionalSpacing != 0,
                      10, y, callbackCheckBox, kHUD_CB_FRACTIONAL_SPACING, 't');
    y += 20;
    g_hud.AddCheckBox("Frustum Patch Culling (B)",  g_patchCull != 0,
                      10, y, callbackCheckBox, kHUD_CB_PATCH_CULL, 'b');
    y += 20;
    g_hud.AddCheckBox("Freeze (spc)", g_freeze != 0,
                      10, y, callbackCheckBox, kHUD_CB_FREEZE, ' ');
    y += 20;

    int displaystyle_pulldown = g_hud.AddPullDown("DisplayStyle (W)", 200, 10, 250,
                                                  callbackDisplayStyle, 'w');
    g_hud.AddPullDownButton(displaystyle_pulldown, "Wire", kDisplayStyleWire,
                            g_displayStyle == kDisplayStyleWire);
    g_hud.AddPullDownButton(displaystyle_pulldown, "Shaded", kDisplayStyleShaded,
                            g_displayStyle == kDisplayStyleShaded);
    g_hud.AddPullDownButton(displaystyle_pulldown, "Wire+Shaded", kDisplayStyleWireOnShaded,
                            g_displayStyle == kDisplayStyleWireOnShaded);

    int shading_pulldown = g_hud.AddPullDown("Shading (C)", 200, 70, 250,
                                             callbackShadingMode, 'c');
    g_hud.AddPullDownButton(shading_pulldown, "Material",
                            kShadingMaterial,
                            g_shadingMode == kShadingMaterial);
    g_hud.AddPullDownButton(shading_pulldown, "Varying Color",
                            kShadingVaryingColor,
                            g_shadingMode == kShadingVaryingColor);
    g_hud.AddPullDownButton(shading_pulldown, "Varying Color (Interleaved)",
                            kShadingInterleavedVaryingColor,
                            g_shadingMode == kShadingInterleavedVaryingColor);
    g_hud.AddPullDownButton(shading_pulldown, "FaceVarying Color",
                            kShadingFaceVaryingColor,
                            g_shadingMode == kShadingFaceVaryingColor);
    g_hud.AddPullDownButton(shading_pulldown, "Patch Type",
                            kShadingPatchType,
                            g_shadingMode == kShadingPatchType);
    g_hud.AddPullDownButton(shading_pulldown, "Patch Depth",
                            kShadingPatchDepth,
                            g_shadingMode == kShadingPatchDepth);
    g_hud.AddPullDownButton(shading_pulldown, "Patch Coord",
                            kShadingPatchCoord,
                            g_shadingMode == kShadingPatchCoord);
    g_hud.AddPullDownButton(shading_pulldown, "Normal",
                            kShadingNormal,
                            g_shadingMode == kShadingNormal);

    int compute_pulldown = g_hud.AddPullDown("Compute (K)", 475, 10, 300, callbackKernel, 'k');
    g_hud.AddPullDownButton(compute_pulldown, "CPU", kCPU);
#ifdef OPENSUBDIV_HAS_OPENMP
    g_hud.AddPullDownButton(compute_pulldown, "OpenMP", kOPENMP);
#endif
#ifdef OPENSUBDIV_HAS_TBB
    g_hud.AddPullDownButton(compute_pulldown, "TBB", kTBB);
#endif
#ifdef OPENSUBDIV_HAS_CUDA
    g_hud.AddPullDownButton(compute_pulldown, "CUDA", kCUDA);
#endif
#ifdef OPENSUBDIV_HAS_OPENCL
    if (CLDeviceContext::HAS_CL_VERSION_1_1()) {
        g_hud.AddPullDownButton(compute_pulldown, "OpenCL", kCL);
    }
#endif
#ifdef OPENSUBDIV_HAS_GLSL_TRANSFORM_FEEDBACK
    g_hud.AddPullDownButton(compute_pulldown, "GLSL TransformFeedback", kGLSL);
#endif
#ifdef OPENSUBDIV_HAS_GLSL_COMPUTE
    if (GLUtils::GL_ARBComputeShaderOrGL_VERSION_4_3()) {
        g_hud.AddPullDownButton(compute_pulldown, "GLSL Compute", kGLSLCompute);
    }
#endif
    if (GLUtils::SupportsAdaptiveTessellation()) {
        g_hud.AddCheckBox("Adaptive (`)", g_adaptive!=0,
                          10, 190, callbackCheckBox, kHUD_CB_ADAPTIVE, '`');
        g_hud.AddCheckBox("Smooth Corner Patch (O)", g_smoothCornerPatch!=0,
                          10, 210, callbackCheckBox, kHUD_CB_SMOOTH_CORNER_PATCH, 'o');
        g_hud.AddCheckBox("Single Crease Patch (S)", g_singleCreasePatch!=0,
                          10, 230, callbackCheckBox, kHUD_CB_SINGLE_CREASE_PATCH, 's');
        g_hud.AddCheckBox("Inf Sharp Patch (I)", g_infSharpPatch!=0,
                          10, 250, callbackCheckBox, kHUD_CB_INF_SHARP_PATCH, 'i');

        int endcap_pulldown = g_hud.AddPullDown(
            "End cap (E)", 10, 270, 200, callbackEndCap, 'e');
        g_hud.AddPullDownButton(endcap_pulldown,"Linear",
                                kEndCapBilinearBasis,
                                g_endCap == kEndCapBilinearBasis);
        g_hud.AddPullDownButton(endcap_pulldown, "Regular",
                                kEndCapBSplineBasis,
                                g_endCap == kEndCapBSplineBasis);
        g_hud.AddPullDownButton(endcap_pulldown, "Gregory",
                                kEndCapGregoryBasis,
                                g_endCap == kEndCapGregoryBasis);
        if (g_legacyGregoryEnabled) {
            g_hud.AddPullDownButton(endcap_pulldown, "LegacyGregory",
                                    kEndCapLegacyGregory,
                                    g_endCap == kEndCapLegacyGregory);
        }
    }

    for (int i = 1; i < 11; ++i) {
        char level[16];
        sprintf(level, "Lv. %d", i);
        g_hud.AddRadioButton(3, level, i == g_level, 10, 310+i*20, callbackLevel, i, '0'+(i%10));
    }

    int shapes_pulldown = g_hud.AddPullDown("Shape (N)", -300, 10, 300, callbackModel, 'n');
    for (int i = 0; i < (int)g_defaultShapes.size(); ++i) {
        g_hud.AddPullDownButton(shapes_pulldown, g_defaultShapes[i].name.c_str(),i);
    }

    g_hud.AddCheckBox("Show patch counts", g_displayPatchCounts!=0, -420, -20, callbackCheckBox, kHUD_CB_DISPLAY_PATCH_COUNTS);

    g_hud.Rebuild(windowWidth, windowHeight, frameBufferWidth, frameBufferHeight);
}

//------------------------------------------------------------------------------
static void
initGL() {
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glCullFace(GL_BACK);
    glEnable(GL_CULL_FACE);

    glGenQueries(2, g_queries);

    glGenVertexArrays(1, &g_vao);
}

//------------------------------------------------------------------------------
static void
idle() {

    if (! g_freeze) {
        g_frame++;
        updateGeom();
    }

    if (g_repeatCount != 0 && g_frame >= g_repeatCount)
        g_running = 0;
}

//------------------------------------------------------------------------------
static void
callbackErrorOsd(OpenSubdiv::Far::ErrorType err, const char *message) {
    printf("OpenSubdiv Error: %d\n", err);
    printf("    %s\n", message);
}

//------------------------------------------------------------------------------
static void
callbackErrorGLFW(int error, const char* description) {
    fprintf(stderr, "GLFW Error (%d) : %s\n", error, description);
}
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

int main(int argc, char ** argv) {

    ArgOptions args;
        
    args.Parse(argc, argv);

    g_yup = args.GetYUp();
    g_adaptive = args.GetAdaptive();
    g_level = args.GetLevel();
    g_repeatCount = args.GetRepeatCount();

    // Parse remaining args
    const std::vector<const char *> &rargs = args.GetRemainingArgs();
    for (size_t i = 0; i < rargs.size(); ++i) {

        if (!strcmp(rargs[i], "-lg")) {
            g_legacyGregoryEnabled = true;
        } else {
            args.PrintUnrecognizedArgWarning(rargs[i]);
        }
    }

    ViewerArgsUtils::PopulateShapesOrAnimShapes(
        args, &g_defaultShapes, &g_objAnim);

    initShapes();

    g_fpsTimer.Start();

    OpenSubdiv::Far::SetErrorCallback(callbackErrorOsd);

    glfwSetErrorCallback(callbackErrorGLFW);
    if (! glfwInit()) {
        printf("Failed to initialize GLFW\n");
        return 1;
    }

    static const char windowTitle[] = "OpenSubdiv glViewer " OPENSUBDIV_VERSION_STRING;

    GLUtils::SetMinimumGLVersion(argc, argv);

    if (args.GetFullScreen()) {

        g_primary = glfwGetPrimaryMonitor();

        // apparently glfwGetPrimaryMonitor fails under linux : if no primary,
        // settle for the first one in the list
        if (! g_primary) {
            int count = 0;
            GLFWmonitor ** monitors = glfwGetMonitors(&count);

            if (count)
                g_primary = monitors[0];
        }

        if (g_primary) {
            GLFWvidmode const * vidmode = glfwGetVideoMode(g_primary);
            g_width = vidmode->width;
            g_height = vidmode->height;
        }
    }

    g_window = glfwCreateWindow(g_width, g_height, windowTitle,
        args.GetFullScreen() && g_primary ? g_primary : NULL, NULL);

    if (! g_window) {
        std::cerr << "Failed to create OpenGL context.\n";
        glfwTerminate();
        return 1;
    }

    glfwMakeContextCurrent(g_window);

    GLUtils::InitializeGL();
    GLUtils::PrintGLVersion();

    // accommodate high DPI displays (e.g. mac retina displays)
    glfwGetFramebufferSize(g_window, &g_width, &g_height);
    glfwSetFramebufferSizeCallback(g_window, reshape);

    glfwSetKeyCallback(g_window, keyboard);
    glfwSetCursorPosCallback(g_window, motion);
    glfwSetMouseButtonCallback(g_window, mouse);
    glfwSetWindowCloseCallback(g_window, windowClose);

    // activate feature adaptive tessellation if OSD supports it
    if (g_adaptive) {
        g_adaptive = GLUtils::SupportsAdaptiveTessellation();
    }

    initGL();

    glfwSwapInterval(0);

    initHUD();
    rebuildMesh();

    while (g_running) {
        idle();
        display();

        glfwPollEvents();
        glfwSwapBuffers(g_window);

        glFinish();
    }

    uninitGL();
    glfwTerminate();
}

//------------------------------------------------------------------------------
