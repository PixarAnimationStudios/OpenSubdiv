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
GLFWwindow* g_window = 0;
GLFWmonitor* g_primary = 0;

#include <vector>
#include <sstream>
#include <iostream>
#include <fstream>
#include <string>
#include <utility>
#include <algorithm>
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
    #include <opensubdiv/osd/clEvaluator.h>
    #include <opensubdiv/osd/clGLVertexBuffer.h>
    #include "../common/clDeviceContext.h"
    CLDeviceContext g_clDeviceContext;
#endif

#ifdef OPENSUBDIV_HAS_CUDA
    #include <opensubdiv/osd/cudaEvaluator.h>
    #include <opensubdiv/osd/cudaGLVertexBuffer.h>
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
OpenSubdiv::Osd::GLMeshInterface *g_mesh;

#include "Ptexture.h"
#include "PtexUtils.h"

#include "../../regression/common/far_utils.h"
#include "../../regression/common/arg_utils.h"
#include "../common/objAnim.h"
#include "../common/stopwatch.h"
#include "../common/simple_math.h"
#include "../common/glControlMeshDisplay.h"
#include "../common/glHud.h"
#include "../common/hdr_reader.h"
#include "../common/glPtexMipmapTexture.h"
#include "../common/glShaderCache.h"
#include "../common/glUtils.h"

#include <opensubdiv/osd/glslPatchShaderSource.h>
static const char *g_defaultShaderSource =
#if defined(GL_ARB_tessellation_shader) || defined(GL_VERSION_4_0)
    #include "shader.gen.h"
#else
    #include "shader_gl3.gen.h"
#endif
;
static const char *g_skyShaderSource =
#include "skyshader.gen.h"
;
static std::string g_shaderSource;
static const char *g_shaderFilename = NULL;

enum KernelType { kCPU = 0,
                  kOPENMP = 1,
                  kTBB = 2,
                  kCUDA = 3,
                  kCL = 4,
                  kGLSL = 5,
                  kGLSLCompute = 6 };

enum HudCheckBox { HUD_CB_ADAPTIVE,
                   HUD_CB_DISPLAY_OCCLUSION,
                   HUD_CB_DISPLAY_NORMALMAP,
                   HUD_CB_DISPLAY_SPECULAR,
                   HUD_CB_CONTROL_MESH_EDGES,
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

GLuint g_constantUB = 0,
       g_constantBinding = 0;

// ptex switch
bool  g_occlusion = false,
      g_specular = false;

bool g_seamless = true;

// camera
float g_rotate[2] = {0, 0},
      g_dolly = 5,
      g_pan[2] = {0, 0},
      g_center[3] = {0, 0, 0},
      g_size = 0;
float g_modelViewProjection[16];

int   g_prev_x = 0,
      g_prev_y = 0;

// viewport
int   g_width = 1024,
      g_height = 1024;

GLhud g_hud;
GLControlMeshDisplay g_controlMeshDisplay;

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

ObjAnim const * g_objAnim = 0;

GLuint g_queries[2] = {0, 0};
GLuint g_vao = 0;
GLuint g_skyVAO = 0;
GLuint g_edgeIndexBuffer = 0;

GLuint g_diffuseEnvironmentMap = 0;
GLuint g_specularEnvironmentMap = 0;

//------------------------------------------------------------------------------

struct Sky {
    int numIndices;
    GLuint vertexBuffer;
    GLuint elementBuffer;
    GLuint mvpMatrix;
    GLDrawConfig *drawConfig;

    Sky() : numIndices(0), vertexBuffer(0), elementBuffer(0), mvpMatrix(0),
            drawConfig(NULL) {}
    ~Sky() {
        delete drawConfig;
    }

    bool BuildProgram(const char *source) {
        if (drawConfig) delete drawConfig;

        drawConfig = new GLDrawConfig("#version 410\n");

        drawConfig->CompileAndAttachShader(GL_VERTEX_SHADER,
                                           "#define SKY_VERTEX_SHADER\n" +
                                           std::string(source));
        drawConfig->CompileAndAttachShader(GL_FRAGMENT_SHADER,
                                           "#define SKY_FRAGMENT_SHADER\n" +
                                           std::string(source));
        if (drawConfig->Link() == false) {
            delete drawConfig;
            drawConfig = NULL;
            return false;
        }
        return true;
    }

    int GetProgram() const {
        if (drawConfig) return drawConfig->GetProgram();
        return 0;
    }

} g_sky;

//------------------------------------------------------------------------------

GLPtexMipmapTexture * g_osdPTexImage = 0;
GLPtexMipmapTexture * g_osdPTexDisplacement = 0;
GLPtexMipmapTexture * g_osdPTexOcclusion = 0;
GLPtexMipmapTexture * g_osdPTexSpecular = 0;
const char * g_ptexColorFilename;
size_t g_ptexMemoryUsage = 0;


//------------------------------------------------------------------------------
static void
calcNormals(OpenSubdiv::Far::TopologyRefiner * refiner,
    std::vector<float> const & pos, std::vector<float> & result ) {

    typedef OpenSubdiv::Far::ConstIndexArray IndexArray;

    OpenSubdiv::Far::TopologyLevel const & refBaseLevel = refiner->GetLevel(0);

    // calc normal vectors
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
void
updateGeom() {

    int nverts = (int)g_positions.size() / 3;

    if (g_moveScale && g_adaptive && g_objAnim) {

        std::vector<float> vertex;
        vertex.resize(nverts*3);

        g_objAnim->InterpolatePositions(g_animTime, &vertex[0], 3);

        g_mesh->UpdateVertexBuffer(&vertex[0], 0, nverts);

    } else {
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
            if (g_adaptive == false) {
                vertex.push_back(n[0]);
                vertex.push_back(n[1]);
                vertex.push_back(n[2]);
                n += 3;
            }
        }

        g_mesh->UpdateVertexBuffer(&vertex[0], 0, nverts);
    }

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
void
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
    assert(r->meshType() == Ptex::mt_quad);

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

void
reshape(GLFWwindow *, int width, int height) {

    g_width = width;
    g_height = height;

    int windowWidth = g_width, windowHeight = g_height;

    // window size might not match framebuffer size on a high DPI display
    glfwGetWindowSize(g_window, &windowWidth, &windowHeight);

    g_hud.Rebuild(windowWidth, windowHeight, width, height);

    glBindTexture(GL_TEXTURE_2D, 0);

    GLUtils::CheckGLErrors("Reshape");
}

void reshape() {
    reshape(g_window, g_width, g_height);
}

void windowClose(GLFWwindow*) {
    g_running = false;
}

//------------------------------------------------------------------------------
const char *getKernelName(int kernel) {

         if (kernel == kCPU)
        return "CPU";
    else if (kernel == kOPENMP)
        return "OpenMP";
    else if (kernel == kCUDA)
        return "Cuda";
    else if (kernel == kGLSL)
        return "GLSL";
    else if (kernel == kCL)
        return "OpenCL";
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

//------------------------------------------------------------------------------
class ShaderCache : public GLShaderCache<EffectDesc> {
public:
    virtual GLDrawConfig *CreateDrawConfig(EffectDesc const &effectDesc) {

        using namespace OpenSubdiv;

        // compile shader program
#if defined(GL_ARB_tessellation_shader) || defined(GL_VERSION_4_0)
        const char *glslVersion = "#version 400\n";
#else
        const char *glslVersion = "#version 330\n";
#endif
        GLDrawConfig *config = new GLDrawConfig(glslVersion);

        Far::PatchDescriptor::Type type = effectDesc.desc.GetType();

        // common defines
        std::stringstream ss;

        if (type == Far::PatchDescriptor::QUADS) {
            ss << "#define PRIM_QUAD\n";
        } else if (type == Far::PatchDescriptor::LINES) {
            ss << "#define PRIM_LINE\n";
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

        // for legacy gregory
        ss << "#define OSD_MAX_VALENCE " << effectDesc.maxValence << "\n";
        ss << "#define OSD_NUM_ELEMENTS " << effectDesc.numElements << "\n";

        // add ptex functions
        ss << GLPtexMipmapTexture::GetShaderSource();

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
        } else if (type == Far::PatchDescriptor::LOOP) {
            ss << "#define OSD_PATCH_LOOP\n";
        } else if (type == Far::PatchDescriptor::GREGORY_TRIANGLE) {
            ss << "#define OSD_PATCH_GREGORY_TRIANGLE\n";
        }

        // include osd PatchCommon
        ss << Osd::GLSLPatchShaderSource::GetCommonShaderSource();
        std::string common = ss.str();
        ss.str("");

        // vertex shader
        ss << common
            // enable local vertex shader
           << (effectDesc.desc.IsAdaptive() ? "" : "#define VERTEX_SHADER\n")
           << g_shaderSource
           << Osd::GLSLPatchShaderSource::GetVertexShaderSource(type);
        config->CompileAndAttachShader(GL_VERTEX_SHADER, ss.str());
        ss.str("");

        if (effectDesc.desc.IsAdaptive()) {
            // tess control shader
            ss << common
               << g_shaderSource
               << Osd::GLSLPatchShaderSource::GetTessControlShaderSource(type);
            config->CompileAndAttachShader(GL_TESS_CONTROL_SHADER, ss.str());
            ss.str("");

            // tess eval shader
            ss << common
               << g_shaderSource
               << Osd::GLSLPatchShaderSource::GetTessEvalShaderSource(type);
            config->CompileAndAttachShader(GL_TESS_EVALUATION_SHADER, ss.str());
            ss.str("");
        }

        // geometry shader
        ss << common
           << "#define GEOMETRY_SHADER\n" // enable local geometry shader
           << g_shaderSource;
        config->CompileAndAttachShader(GL_GEOMETRY_SHADER, ss.str());
        ss.str("");

        // fragment shader
        ss << common
           << "#define FRAGMENT_SHADER\n" // enable local fragment shader
           << g_shaderSource;
        config->CompileAndAttachShader(GL_FRAGMENT_SHADER, ss.str());
        ss.str("");

        if (!config->Link()) {
            delete config;
            return NULL;
        }

        // assign uniform locations
        GLuint program = config->GetProgram();
        GLuint uboIndex = glGetUniformBlockIndex(program, "Constant");
        if (uboIndex != GL_INVALID_INDEX)
            glUniformBlockBinding(program, uboIndex, g_constantBinding);

        // assign texture locations
        GLint loc;
        // patch textures
        glUseProgram(program);
        if ((loc = glGetUniformLocation(program, "OsdPatchParamBuffer")) != -1) {
            glUniform1i(loc, 0); // GL_TEXTURE0
        }

        // environment textures
        if ((loc = glGetUniformLocation(program, "diffuseEnvironmentMap")) != -1) {
            glUniform1i(loc, 5);
        }
        if ((loc = glGetUniformLocation(program, "specularEnvironmentMap")) != -1) {
            glUniform1i(loc, 6);
        }

        // ptex textures
        if ((loc = glGetUniformLocation(program, "textureImage_Data")) != -1) {
            glUniform1i(loc, 7);
        }
        if ((loc = glGetUniformLocation(program, "textureImage_Packing")) != -1) {
            glUniform1i(loc, 8);
        }
        if ((loc = glGetUniformLocation(program, "textureDisplace_Data")) != -1) {
            glUniform1i(loc, 9);
        }
        if ((loc = glGetUniformLocation(program, "textureDisplace_Packing")) != -1) {
            glUniform1i(loc, 10);
        }
        if ((loc = glGetUniformLocation(program, "textureOcclusion_Data")) != -1) {
            glUniform1i(loc, 11);
        }
        if ((loc = glGetUniformLocation(program, "textureOcclusion_Packing")) != -1) {
            glUniform1i(loc, 12);
        }
        if ((loc = glGetUniformLocation(program, "textureSpecular_Data")) != -1) {
            glUniform1i(loc, 13);
        }
        if ((loc = glGetUniformLocation(program, "textureSpecular_Packing")) != -1) {
            glUniform1i(loc, 14);
        }

        glUseProgram(0);

        return config;
    }
};

ShaderCache g_shaderCache;

//------------------------------------------------------------------------------
GLPtexMipmapTexture *
createPtex(const char *filename, int memLimit) {

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
    if (ptex->meshType() == Ptex::mt_triangle) {
        printf("Error in %s:  triangular Ptex not yet supported\n", filename);
        exit(1);
    }

    size_t targetMemory = memLimit * 1024 * 1024; // MB

    GLPtexMipmapTexture *osdPtex = GLPtexMipmapTexture::Create(
        ptex, g_maxMipmapLevels, targetMemory);

    GLuint texture = osdPtex->GetTexelsTexture();
    glBindTexture(GL_TEXTURE_2D_ARRAY, texture);
    GLint w, h, d;
    glGetTexLevelParameteriv(GL_TEXTURE_2D_ARRAY, 0, GL_TEXTURE_WIDTH, &w);
    glGetTexLevelParameteriv(GL_TEXTURE_2D_ARRAY, 0, GL_TEXTURE_HEIGHT, &h);
    glGetTexLevelParameteriv(GL_TEXTURE_2D_ARRAY, 0, GL_TEXTURE_DEPTH, &d);
    printf("PageSize = %d x %d x %d\n", w, h, d);
    glBindTexture(GL_TEXTURE_2D_ARRAY, 0);

    ptex->release();

#ifdef USE_PTEX_CACHE
    cache->release();
#endif

    return osdPtex;
}

//------------------------------------------------------------------------------

void
createOsdMesh(int level, int kernel) {

    GLUtils::CheckGLErrors("createOsdMesh");

    Ptex::String ptexError;
    PtexTexture *ptexColor = PtexTexture::open(g_ptexColorFilename, ptexError, true);
    if (ptexColor == NULL) {
        printf("Error in reading %s\n", g_ptexColorFilename);
        exit(1);
    }
    if (ptexColor->meshType() == Ptex::mt_triangle) {
        printf("Error in %s:  triangular Ptex not yet supported\n", g_ptexColorFilename);
        exit(1);
    }

    // generate Shape representation from ptex
    Shape * shape = createPTexGeo(ptexColor);
    if (!shape) {
        return;
    }

    g_positions=shape->verts;

    // create Far mesh (topology)
    OpenSubdiv::Sdc::SchemeType sdctype = GetSdcType(*shape);
    OpenSubdiv::Sdc::Options sdcoptions = GetSdcOptions(*shape);

    OpenSubdiv::Far::TopologyRefiner * refiner =
        OpenSubdiv::Far::TopologyRefinerFactory<Shape>::Create(*shape,
            OpenSubdiv::Far::TopologyRefinerFactory<Shape>::Options(sdctype, sdcoptions));

    // save coarse topology (used for coarse mesh drawing)
    g_controlMeshDisplay.SetTopology(refiner->GetLevel(0));

    delete shape;

    g_normals.resize(g_positions.size(), 0.0f);
    calcNormals(refiner, g_positions, g_normals);

    delete g_mesh;
    g_mesh = NULL;

    OpenSubdiv::Osd::MeshBitset bits;
    bits.set(OpenSubdiv::Osd::MeshAdaptive, g_adaptive);
    bits.set(OpenSubdiv::Osd::MeshEndCapGregoryBasis, true);

    int numVertexElements = g_adaptive ? 3 : 6;
    int numVaryingElements = 0;

    if (kernel == kCPU) {
        g_mesh = new OpenSubdiv::Osd::Mesh<OpenSubdiv::Osd::CpuGLVertexBuffer,
                                           OpenSubdiv::Far::StencilTable,
                                           OpenSubdiv::Osd::CpuEvaluator,
                                           OpenSubdiv::Osd::GLPatchTable>(
                                                refiner,
                                                numVertexElements,
                                                numVaryingElements,
                                                level, bits);
#ifdef OPENSUBDIV_HAS_OPENMP
    } else if (kernel == kOPENMP) {
        g_mesh = new OpenSubdiv::Osd::Mesh<OpenSubdiv::Osd::CpuGLVertexBuffer,
                                           OpenSubdiv::Far::StencilTable,
                                           OpenSubdiv::Osd::OmpEvaluator,
                                           OpenSubdiv::Osd::GLPatchTable>(
                                                refiner,
                                                numVertexElements,
                                                numVaryingElements,
                                                level, bits);
#endif
#ifdef OPENSUBDIV_HAS_TBB
    } else if (kernel == kTBB) {
        g_mesh = new OpenSubdiv::Osd::Mesh<OpenSubdiv::Osd::CpuGLVertexBuffer,
                                           OpenSubdiv::Far::StencilTable,
                                           OpenSubdiv::Osd::TbbEvaluator,
                                           OpenSubdiv::Osd::GLPatchTable>(
                                                refiner,
                                                numVertexElements,
                                                numVaryingElements,
                                                level, bits);
#endif
#ifdef OPENSUBDIV_HAS_OPENCL
    } else if (kernel == kCL) {
        static OpenSubdiv::Osd::EvaluatorCacheT<OpenSubdiv::Osd::CLEvaluator> clEvaluatorCache;
        g_mesh = new OpenSubdiv::Osd::Mesh<OpenSubdiv::Osd::CLGLVertexBuffer,
                                           OpenSubdiv::Osd::CLStencilTable,
                                           OpenSubdiv::Osd::CLEvaluator,
                                           OpenSubdiv::Osd::GLPatchTable,
                                           CLDeviceContext>(
                                                refiner,
                                                numVertexElements,
                                                numVaryingElements,
                                                level, bits,
                                                &clEvaluatorCache,
                                                &g_clDeviceContext);
#endif
#ifdef OPENSUBDIV_HAS_CUDA
    } else if (kernel == kCUDA) {
        g_mesh = new OpenSubdiv::Osd::Mesh<OpenSubdiv::Osd::CudaGLVertexBuffer,
                                           OpenSubdiv::Osd::CudaStencilTable,
                                           OpenSubdiv::Osd::CudaEvaluator,
                                           OpenSubdiv::Osd::GLPatchTable>(
                                                refiner,
                                                numVertexElements,
                                                numVaryingElements,
                                                level, bits);
#endif
#ifdef OPENSUBDIV_HAS_GLSL_TRANSFORM_FEEDBACK
    } else if (kernel == kGLSL) {
        static OpenSubdiv::Osd::EvaluatorCacheT<OpenSubdiv::Osd::GLXFBEvaluator> glXFBEvaluatorCache;
        g_mesh = new OpenSubdiv::Osd::Mesh<OpenSubdiv::Osd::GLVertexBuffer,
                                           OpenSubdiv::Osd::GLStencilTableTBO,
                                           OpenSubdiv::Osd::GLXFBEvaluator,
                                           OpenSubdiv::Osd::GLPatchTable>(
                                               refiner,
                                               numVertexElements,
                                               numVaryingElements,
                                               level, bits,
                                               &glXFBEvaluatorCache);
#endif
#ifdef OPENSUBDIV_HAS_GLSL_COMPUTE
    } else if (kernel == kGLSLCompute) {
        static OpenSubdiv::Osd::EvaluatorCacheT<OpenSubdiv::Osd::GLComputeEvaluator> glComputeEvaluatorCache;
        g_mesh = new OpenSubdiv::Osd::Mesh<OpenSubdiv::Osd::GLVertexBuffer,
                                           OpenSubdiv::Osd::GLStencilTableSSBO,
                                           OpenSubdiv::Osd::GLComputeEvaluator,
                                           OpenSubdiv::Osd::GLPatchTable>(
                                                refiner,
                                                numVertexElements,
                                                numVaryingElements,
                                                level, bits,
                                               &glComputeEvaluatorCache);
#endif
    } else {
        printf("Unsupported kernel %s\n", getKernelName(kernel));
    }

    if (glGetError() != GL_NO_ERROR) {
        printf("GLERROR\n");
    }

    updateGeom();

    // ------ VAO
    glBindVertexArray(g_vao);

    glBindBuffer(GL_ARRAY_BUFFER, g_mesh->BindVertexBuffer());

    if (g_adaptive) {
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    } else {
        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(GLfloat) * 6, 0);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(GLfloat) * 6, (float*)12);
    }
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_mesh->GetPatchTable()->GetPatchIndexBuffer());

    glBindVertexArray(0);
}

//------------------------------------------------------------------------------

void
createSky() {

    const int U_DIV = 20;
    const int V_DIV = 20;

    std::vector<float> vbo;
    std::vector<int> indices;
    for (int u = 0; u <= U_DIV; ++u) {
        for (int v = 0; v < V_DIV; ++v) {
            float s = float(2*M_PI*float(u)/U_DIV);
            float t = float(M_PI*float(v)/(V_DIV-1));
            vbo.push_back(-sin(t)*sin(s));
            vbo.push_back(cos(t));
            vbo.push_back(-sin(t)*cos(s));
            vbo.push_back(u/float(U_DIV));
            vbo.push_back(v/float(V_DIV));

            if (v > 0 && u > 0) {
                indices.push_back((u-1)*V_DIV+v-1);
                indices.push_back(u*V_DIV+v-1);
                indices.push_back((u-1)*V_DIV+v);
                indices.push_back((u-1)*V_DIV+v);
                indices.push_back(u*V_DIV+v-1);
                indices.push_back(u*V_DIV+v);
            }
        }
    }

    glGenBuffers(1, &g_sky.vertexBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, g_sky.vertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float)*vbo.size(), &vbo[0], GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glGenBuffers(1, &g_sky.elementBuffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_sky.elementBuffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(int)*indices.size(), &indices[0], GL_STATIC_DRAW);

    g_sky.numIndices = (int)indices.size();

    g_sky.BuildProgram(g_skyShaderSource);

    GLint environmentMap = glGetUniformLocation(g_sky.GetProgram(), "environmentMap");
    glUseProgram(g_sky.GetProgram());
    if (g_specularEnvironmentMap)
        glUniform1i(environmentMap, 6);
    else
        glUniform1i(environmentMap, 5);
    glUseProgram(0);

    g_sky.mvpMatrix = glGetUniformLocation(g_sky.GetProgram(), "ModelViewProjectionMatrix");
}

//------------------------------------------------------------------------------

static void
updateConstantUniformBlock() {
    struct Constant {
        float ModelViewMatrix[16];
        float ProjectionMatrix[16];
        float ModelViewProjectionMatrix[16];
        float ModelViewInverseMatrix[16];
        struct Light {
            float position[4];
            float ambient[4];
            float diffuse[4];
            float specular[4];
        } lightSource[2];
        float TessLevel;
        float displacementScale;
        float mipmapBias;
    } constantData;

    // transforms
    double aspect = g_width/(double)g_height;
    identity(constantData.ModelViewMatrix);
    translate(constantData.ModelViewMatrix, -g_pan[0], -g_pan[1], -g_dolly);
    rotate(constantData.ModelViewMatrix, g_rotate[1], 1, 0, 0);
    rotate(constantData.ModelViewMatrix, g_rotate[0], 0, 1, 0);
    if (!g_yup) {
        rotate(constantData.ModelViewMatrix, -90, 1, 0, 0);
    }
    translate(constantData.ModelViewMatrix, -g_center[0], -g_center[1], -g_center[2]);
    perspective(constantData.ProjectionMatrix, 45.0f, (float)aspect, g_size*0.001f,
                g_size+g_dolly);
    multMatrix(constantData.ModelViewProjectionMatrix,
               constantData.ModelViewMatrix,
               constantData.ProjectionMatrix);
    inverseMatrix(constantData.ModelViewInverseMatrix,
                  constantData.ModelViewMatrix);
    // save mvp for the control mesh drawing
    memcpy(g_modelViewProjection, constantData.ModelViewProjectionMatrix,
           16*sizeof(float));

    // lights
    Constant::Light light0 = {  { 0.6f, 1.0f, 0.6f, 0.0f },
                                { 0.1f, 0.1f, 0.1f, 1.0f },
                                { 1.7f, 1.3f, 1.1f, 1.0f },
                                { 1.0f, 1.0f, 1.0f, 1.0f } };
    Constant::Light light1 = {  { -0.8f, 0.6f, -0.7f, 0.0f },
                                {  0.0f, 0.0f,  0.0f, 1.0f },
                                {  0.8f, 0.8f,  1.5f, 1.0f },
                                {  0.4f, 0.4f,  0.4f, 1.0f } };
    constantData.lightSource[0] = light0;
    constantData.lightSource[1] = light1;

    // other
    constantData.TessLevel = static_cast<float>(1 << g_tessLevel);
    constantData.displacementScale = g_displacementScale;
    constantData.mipmapBias = g_mipmapBias;

    // update GPU buffer
    if (g_constantUB == 0) {
        glGenBuffers(1, &g_constantUB);
        glBindBuffer(GL_UNIFORM_BUFFER, g_constantUB);
        glBufferData(GL_UNIFORM_BUFFER,
                     sizeof(constantData), NULL, GL_STATIC_DRAW);
    };
    glBindBuffer(GL_UNIFORM_BUFFER, g_constantUB);
    glBufferSubData(GL_UNIFORM_BUFFER,
                0, sizeof(constantData), &constantData);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    glBindBufferBase(GL_UNIFORM_BUFFER, g_constantBinding, g_constantUB);

}

static void
bindTextures() {
    if (g_mesh->GetPatchTable()->GetPatchParamTextureBuffer()) {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_BUFFER,
                      g_mesh->GetPatchTable()->GetPatchParamTextureBuffer());
    }

    // other textures
    if (g_ibl) {
        if (g_diffuseEnvironmentMap) {
            glActiveTexture(GL_TEXTURE5);
            glBindTexture(GL_TEXTURE_2D, g_diffuseEnvironmentMap);
        }
        if (g_specularEnvironmentMap) {
            glActiveTexture(GL_TEXTURE6);
            glBindTexture(GL_TEXTURE_2D, g_specularEnvironmentMap);
        }
        glActiveTexture(GL_TEXTURE0);
    }

    // color ptex
    glActiveTexture(GL_TEXTURE7);
    glBindTexture(GL_TEXTURE_2D_ARRAY, g_osdPTexImage->GetTexelsTexture());
    glActiveTexture(GL_TEXTURE8);
    glBindTexture(GL_TEXTURE_BUFFER, g_osdPTexImage->GetLayoutTextureBuffer());

    // displacement ptex
    if (g_displacement != DISPLACEMENT_NONE || g_normal) {
        glActiveTexture(GL_TEXTURE9);
        glBindTexture(GL_TEXTURE_2D_ARRAY, g_osdPTexDisplacement->GetTexelsTexture());
        glActiveTexture(GL_TEXTURE10);
        glBindTexture(GL_TEXTURE_BUFFER, g_osdPTexDisplacement->GetLayoutTextureBuffer());
    }

    // occlusion ptex
    if (g_occlusion) {
        glActiveTexture(GL_TEXTURE11);
        glBindTexture(GL_TEXTURE_2D_ARRAY, g_osdPTexOcclusion->GetTexelsTexture());
        glActiveTexture(GL_TEXTURE12);
        glBindTexture(GL_TEXTURE_BUFFER, g_osdPTexOcclusion->GetLayoutTextureBuffer());
    }

    // specular ptex
    if (g_specular) {
        glActiveTexture(GL_TEXTURE13);
        glBindTexture(GL_TEXTURE_2D_ARRAY, g_osdPTexSpecular->GetTexelsTexture());
        glActiveTexture(GL_TEXTURE14);
        glBindTexture(GL_TEXTURE_BUFFER, g_osdPTexSpecular->GetLayoutTextureBuffer());
    }

    glActiveTexture(GL_TEXTURE0);
}

//------------------------------------------------------------------------------
static GLenum
bindProgram(Effect effect,
            OpenSubdiv::Osd::PatchArray const & patch) {
    EffectDesc effectDesc(patch.GetDescriptor(), effect);

    GLDrawConfig *config = g_shaderCache.GetDrawConfig(effectDesc);
    if (!config) return 0;

    GLuint program = config->GetProgram();

    glUseProgram(program);

    // bind standalone uniforms
    GLint uniformPrimitiveIdBase =
        glGetUniformLocation(program, "PrimitiveIdBase");
    if (uniformPrimitiveIdBase >= 0)
        glUniform1i(uniformPrimitiveIdBase, patch.GetPrimitiveIdBase());

    GLenum primType;
    switch(effectDesc.desc.GetType()) {
    case OpenSubdiv::Far::PatchDescriptor::QUADS:
        primType = GL_LINES_ADJACENCY;
        break;
    case OpenSubdiv::Far::PatchDescriptor::TRIANGLES:
        primType = GL_TRIANGLES;
        break;
    default:
#if defined(GL_ARB_tessellation_shader) || defined(GL_VERSION_4_0)
        primType = GL_PATCHES;
        glPatchParameteri(GL_PATCH_VERTICES, effectDesc.desc.GetNumControlVertices());
#else
        primType = GL_POINTS;
#endif
    }

    return primType;
}

//------------------------------------------------------------------------------
void
drawModel() {
    g_mesh->BindVertexBuffer();

    // bind patch related textures and PtexTexture
    bindTextures();

    glBindVertexArray(g_vao);

    // patch drawing
    OpenSubdiv::Osd::PatchArrayVector const & patches =
        g_mesh->GetPatchTable()->GetPatchArrays();
    for (int i = 0; i < (int)patches.size(); ++i) {
        OpenSubdiv::Osd::PatchArray const & patch = patches[i];

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

        GLenum primType = bindProgram(effect, patch);

        glDrawElements(primType,
                       patch.GetNumPatches() * patch.GetDescriptor().GetNumControlVertices(),
                       GL_UNSIGNED_INT,
                       (void *)(patch.GetIndexBase() * sizeof(unsigned int)));
    }

    glBindVertexArray(0);
}

//------------------------------------------------------------------------------

void
drawSky() {
    glUseProgram(g_sky.GetProgram());

    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);

    float modelView[16], projection[16], mvp[16];
    double aspect = g_width/(double)g_height;

    identity(modelView);
    rotate(modelView, g_rotate[1], 1, 0, 0);
    rotate(modelView, g_rotate[0], 0, 1, 0);
    perspective(projection, 45.0f, (float)aspect, g_size*0.001f, g_size+g_dolly);
    multMatrix(mvp, modelView, projection);
    glUniformMatrix4fv(g_sky.mvpMatrix, 1, GL_FALSE, mvp);

    glBindVertexArray(g_skyVAO);

    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, g_sky.vertexBuffer);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(GLfloat) * 5, 0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(GLfloat) * 5,
                          (void*)(sizeof(GLfloat)*3));

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_sky.elementBuffer);
    glDrawElements(GL_TRIANGLES, g_sky.numIndices, GL_UNSIGNED_INT, 0);

    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);

    glBindVertexArray(0);

    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);

    GLUtils::CheckGLErrors("draw model");
}

//------------------------------------------------------------------------------

void
display() {

    Stopwatch s;
    s.Start();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glViewport(0, 0, g_width, g_height);
    g_hud.FillBackground();

    if (g_ibl) {
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        drawSky();
    }

    // update transform and light
    updateConstantUniformBlock();

    glEnable(GL_DEPTH_TEST);
    if (g_wire == DISPLAY_WIRE) {
        glDisable(GL_CULL_FACE);
    }

    // primitive counting
    glBeginQuery(GL_PRIMITIVES_GENERATED, g_queries[0]);
#if defined(GL_VERSION_3_3)
    glBeginQuery(GL_TIME_ELAPSED, g_queries[1]);
#endif

    drawModel();

    glEndQuery(GL_PRIMITIVES_GENERATED);
#if defined(GL_VERSION_3_3)
    glEndQuery(GL_TIME_ELAPSED);
#endif

    // draw the control mesh
    {
        GLuint vbo = g_mesh->BindVertexBuffer();
        int stride = g_adaptive ? 3 : 6;
        g_controlMeshDisplay.Draw(vbo, stride*sizeof(float),
                                  g_modelViewProjection);
    }

    if (g_wire == DISPLAY_WIRE) {
        glEnable(GL_CULL_FACE);
    }
    glDisable(GL_DEPTH_TEST);

    glUseProgram(0);

    s.Stop();
    float drawCpuTime = float(s.GetElapsed() * 1000.0f);

    GLuint numPrimsGenerated = 0;
    GLuint timeElapsed = 0;
    glGetQueryObjectuiv(g_queries[0], GL_QUERY_RESULT, &numPrimsGenerated);
#if defined(GL_VERSION_3_3)
    glGetQueryObjectuiv(g_queries[1], GL_QUERY_RESULT, &timeElapsed);
#endif
    float drawGpuTime = timeElapsed / 1000.0f / 1000.0f;

    g_fpsTimer.Stop();
    float elapsed = (float)g_fpsTimer.GetElapsed();
    if (!g_freeze)
        g_animTime += elapsed;
    g_fpsTimer.Start();

    if (g_hud.IsVisible()) {
        double fps = 1.0/elapsed;

        // Average fps over a defined number of time samples for
        // easier reading in the HUD
        g_fpsTimeSamples[g_currentFpsTimeSample++] = float(fps);
        if (g_currentFpsTimeSample >= NUM_FPS_TIME_SAMPLES)
            g_currentFpsTimeSample = 0;
        double averageFps = 0;
        for (int i = 0; i < NUM_FPS_TIME_SAMPLES; ++i) {
            averageFps += g_fpsTimeSamples[i]/(float)NUM_FPS_TIME_SAMPLES;
        }

        g_hud.DrawString(10, -220, "Ptex memory use : %.1f mb", g_ptexMemoryUsage/1024.0/1024.0);
        g_hud.DrawString(10, -180, "Tess level (+/-): %d", g_tessLevel);
        if (numPrimsGenerated > 1000000) {
            g_hud.DrawString(10, -160, "Primitives      : %3.1f million",
                             (float)numPrimsGenerated/1000000.0);
        } else if (numPrimsGenerated > 1000) {
            g_hud.DrawString(10, -160, "Primitives      : %3.1f thousand",
                             (float)numPrimsGenerated/1000.0);
        } else {
            g_hud.DrawString(10, -160, "Primitives      : %d", numPrimsGenerated);
        }
        g_hud.DrawString(10, -140, "Vertices        : %d", g_mesh->GetNumVertices());
        g_hud.DrawString(10, -120, "Scheme          : %s", g_scheme == 0 ? "CATMARK" : "BILINEAR");
        g_hud.DrawString(10, -100, "GPU Kernel      : %.3f ms", g_gpuTime);
        g_hud.DrawString(10, -80,  "CPU Kernel      : %.3f ms", g_cpuTime);
        g_hud.DrawString(10, -60,  "GPU Draw        : %.3f ms", drawGpuTime);
        g_hud.DrawString(10, -40,  "CPU Draw        : %.3f ms", drawCpuTime);
        g_hud.DrawString(10, -20,  "FPS             : %3.1f", averageFps);

        g_hud.Flush();
    }

    glFinish();

    GLUtils::CheckGLErrors("draw end");
}

//------------------------------------------------------------------------------
static void
mouse(GLFWwindow *, int button, int state, int /* mods */) {

    if (state == GLFW_RELEASE)
        g_hud.MouseRelease();

    if (button == 0 && state == GLFW_PRESS && g_hud.MouseClick(g_prev_x, g_prev_y))
        return;
    g_mbutton[button] = (state == GLFW_PRESS);
}

//------------------------------------------------------------------------------
static void
motion(GLFWwindow *, double dx, double dy) {
    int x = (int)dx, y = (int)dy;

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
        if (g_dolly <= 0.01) g_dolly = 0.01f;
    }

    g_prev_x = x;
    g_prev_y = y;
}

//------------------------------------------------------------------------------
void uninitGL() {
    if (g_osdPTexImage) delete g_osdPTexImage;
    if (g_osdPTexDisplacement) delete g_osdPTexDisplacement;
    if (g_osdPTexOcclusion) delete g_osdPTexOcclusion;
    if (g_osdPTexSpecular) delete g_osdPTexSpecular;

    glDeleteQueries(2, g_queries);
    glDeleteVertexArrays(1, &g_vao);
    glDeleteVertexArrays(1, &g_skyVAO);

    if (g_mesh)
        delete g_mesh;

    if (g_diffuseEnvironmentMap)
        glDeleteTextures(1, &g_diffuseEnvironmentMap);
    if (g_specularEnvironmentMap)
        glDeleteTextures(1, &g_specularEnvironmentMap);

    if (g_sky.vertexBuffer) glDeleteBuffers(1, &g_sky.vertexBuffer);
    if (g_sky.elementBuffer) glDeleteBuffers(1, &g_sky.elementBuffer);
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
    if (g_kernel == kCL && (!g_clDeviceContext.IsInitialized())) {
        // Initialize OpenCL
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
        if (GLUtils::SupportsAdaptiveTessellation()) {
            g_adaptive = checked;
            rebuild = true;
        }
        break;
    case HUD_CB_DISPLAY_OCCLUSION:
        g_occlusion = checked;
        break;
    case HUD_CB_DISPLAY_SPECULAR:
        g_specular = checked;
        break;
    case HUD_CB_CONTROL_MESH_EDGES:
        g_controlMeshDisplay.SetEdgesDisplay(checked);
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
    case HUD_CB_BLOOM:
        g_bloom = checked;
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
//-------------------------------------------------------------------------------
void
reloadShaderFile() {
    if (!g_shaderFilename) return;

    std::ifstream ifs(g_shaderFilename);
    if (!ifs) return;
    printf("Load shader %s\n", g_shaderFilename);

    std::stringstream ss;
    ss << ifs.rdbuf();
    ifs.close();

    g_shaderSource = ss.str();

    g_shaderCache.Reset();
}

//------------------------------------------------------------------------------
static void
toggleFullScreen() {
    // XXXX manuelk : to re-implement from glut
}

//------------------------------------------------------------------------------
void
keyboard(GLFWwindow *, int key, int /* scancode */, int event, int /* mods */) {

    if (event == GLFW_RELEASE) return;
    if (g_hud.KeyDown(tolower(key))) return;

    switch (key) {
        case 'Q': g_running = 0; break;
        case 'E': g_drawNormals = (g_drawNormals+1)%2; break;
        case 'F': fitFrame(); break;
        case GLFW_KEY_TAB: toggleFullScreen(); break;
        case 'R': reloadShaderFile(); createOsdMesh(g_level, g_kernel); break;
        case '+':
        case '=': g_tessLevel++; break;
        case '-': g_tessLevel = std::max(1, g_tessLevel-1); break;
        case GLFW_KEY_ESCAPE: g_hud.SetVisible(!g_hud.IsVisible()); break;
        case 'X': GLUtils::WriteScreenshot(g_width, g_height); break;
    }
}

//------------------------------------------------------------------------------
void
idle() {
    if (!g_freeze)
        g_frame++;

    updateGeom();

    if (g_repeatCount != 0 && g_frame >= g_repeatCount)
        g_running = 0;
}

//------------------------------------------------------------------------------
void
initGL() {
    glClearColor(0.1f, 0.1f, 0.1f, 0.0f);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glEnable(GL_CULL_FACE);

    glGenQueries(2, g_queries);
    glGenVertexArrays(1, &g_vao);
    glGenVertexArrays(1, &g_skyVAO);

    glBindTexture(GL_TEXTURE_2D, 0);
}

//------------------------------------------------------------------------------
void usage(const char *program) {
    printf("Usage: %s [options] <color.ptx> [<displacement.ptx>] [occlusion.ptx>] "
           "[specular.ptx] [pose.obj]...\n", program);
    printf("Options:  -l level                : subdivision level\n");
    printf("          -c count                : frame count until exit (for profiler)\n");
    printf("          -d <diffseEnvMap.hdr>   : diffuse environment map for IBL\n");
    printf("          -e <specularEnvMap.hdr> : specular environment map for IBL\n");
    printf("          -s <shaderfile.glsl>    : custom shader file\n");
    printf("          -yup                    : Y-up model\n");
    printf("          -m level                : max mipmap level (default=10)\n");
    printf("          -x <ptex limit MB>      : ptex target memory size\n");
    printf("          --disp <scale>          : Displacement scale\n");
}

//------------------------------------------------------------------------------
static void
callbackError(OpenSubdiv::Far::ErrorType err, const char *message) {
    printf("OpenSubdiv Error: %d\n", err);
    printf("    %s\n", message);
}

//------------------------------------------------------------------------------
static void
callbackErrorGLFW(int error, const char* description) {
    fprintf(stderr, "GLFW Error (%d) : %s\n", error, description);
}

//------------------------------------------------------------------------------
int main(int argc, char ** argv) {

    ArgOptions args;
    args.Parse(argc, argv);

    const std::vector<const char *> &animobjs = args.GetObjFiles();
    bool fullscreen = args.GetFullScreen();

    g_yup = args.GetYUp();
    g_adaptive = args.GetAdaptive();
    g_level = args.GetLevel();
    g_repeatCount = args.GetRepeatCount();

    //  Retrieve and parse remaining args:
    const std::vector<const char *> &argvRem = args.GetRemainingArgs();

    const char *diffuseEnvironmentMap = NULL, *specularEnvironmentMap = NULL;
    const char *colorFilename = NULL, *displacementFilename = NULL,
        *occlusionFilename = NULL, *specularFilename = NULL;
    int memLimit = 0, colorMem = 0, displacementMem = 0,
        occlusionMem = 0, specularMem = 0;

    for (size_t i = 0; i < argvRem.size(); ++i) {
        if (!strcmp(argvRem[i], "-d"))
            diffuseEnvironmentMap = argvRem[++i];
        else if (!strcmp(argvRem[i], "-e"))
            specularEnvironmentMap = argvRem[++i];
        else if (!strcmp(argvRem[i], "-s"))
            g_shaderFilename = argvRem[++i];
        else if (!strcmp(argvRem[i], "-m"))
            g_maxMipmapLevels = atoi(argvRem[++i]);
        else if (!strcmp(argvRem[i], "-x"))
            memLimit = atoi(argvRem[++i]);
        else if (!strcmp(argvRem[i], "--disp"))
            g_displacementScale = (float)atof(argvRem[++i]);
        else if (colorFilename == NULL) {
            colorFilename = argvRem[i];
            colorMem = memLimit;
        } else if (displacementFilename == NULL) {
            displacementFilename = argvRem[i];
            displacementMem = memLimit;
            g_displacement = DISPLACEMENT_BILINEAR;
            g_normal = NORMAL_BIQUADRATIC;
        } else if (occlusionFilename == NULL) {
            occlusionFilename = argvRem[i];
            occlusionMem = memLimit;
            g_occlusion = 1;
        } else if (specularFilename == NULL) {
            specularFilename = argvRem[i];
            specularMem = memLimit;
            g_specular = 1;
        }
    }

    OpenSubdiv::Far::SetErrorCallback(callbackError);

    g_shaderSource = g_defaultShaderSource;
    reloadShaderFile();

    g_ptexColorFilename = colorFilename;
    if (g_ptexColorFilename == NULL) {
        usage(argv[0]);
        return 1;
    }

    glfwSetErrorCallback(callbackErrorGLFW);
    if (!glfwInit()) {
        printf("Failed to initialize GLFW\n");
        return 1;
    }

    static const char windowTitle[] = "OpenSubdiv glPtexViewer" OPENSUBDIV_VERSION_STRING;

    GLUtils::SetMinimumGLVersion();

    if (fullscreen) {
        g_primary = glfwGetPrimaryMonitor();

        // apparently glfwGetPrimaryMonitor fails under linux : if no primary,
        // settle for the first one in the list
        if (!g_primary) {
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

    if (!(g_window=glfwCreateWindow(g_width, g_height, windowTitle,
                               fullscreen && g_primary ? g_primary : NULL, NULL))) {
        std::cerr << "Failed to create OpenGL context.\n";
        glfwTerminate();
        return 1;
    }

    glfwMakeContextCurrent(g_window);

    GLUtils::InitializeGL();
    GLUtils::PrintGLVersion();

    glfwSetKeyCallback(g_window, keyboard);
    glfwSetCursorPosCallback(g_window, motion);
    glfwSetMouseButtonCallback(g_window, mouse);

    initGL();

    // accommodate high DPI displays (e.g. mac retina displays)
    glfwGetFramebufferSize(g_window, &g_width, &g_height);
    glfwSetFramebufferSizeCallback(g_window, reshape);

    glfwSetWindowCloseCallback(g_window, windowClose);
    // as of GLFW 3.0.1 this callback is not implicit
    reshape();

    // activate feature adaptive tessellation if OSD supports it
    g_adaptive = g_adaptive && GLUtils::SupportsAdaptiveTessellation();

    int windowWidth = g_width, windowHeight = g_height;

    // window size might not match framebuffer size on a high DPI display
    glfwGetWindowSize(g_window, &windowWidth, &windowHeight);

    g_hud.Init(windowWidth, windowHeight, g_width, g_height);

    g_controlMeshDisplay.SetEdgesDisplay(false);

    if (occlusionFilename != NULL) {
        g_hud.AddCheckBox("Ambient Occlusion (A)", g_occlusion,
                          -200, 570, callbackCheckBox, HUD_CB_DISPLAY_OCCLUSION, 'a');
    }
    if (specularFilename != NULL)
        g_hud.AddCheckBox("Specular (S)", g_specular,
                          -200, 590, callbackCheckBox, HUD_CB_DISPLAY_SPECULAR, 's');

    if (diffuseEnvironmentMap || specularEnvironmentMap) {
        g_hud.AddCheckBox("IBL (I)", g_ibl,
                          -200, 610, callbackCheckBox, HUD_CB_IBL, 'i');
    }

    g_hud.AddCheckBox("Control edges (H)",
                      g_controlMeshDisplay.GetEdgesDisplay(),
                      10, 10, callbackCheckBox,
                      HUD_CB_CONTROL_MESH_EDGES, 'h');
    g_hud.AddCheckBox("Animate vertices (M)", g_moveScale != 0.0,
                      10, 30, callbackCheckBox, HUD_CB_ANIMATE_VERTICES, 'm');
    g_hud.AddCheckBox("Screen space LOD (V)",  g_screenSpaceTess,
                      10, 50, callbackCheckBox, HUD_CB_VIEW_LOD, 'v');
    g_hud.AddCheckBox("Fractional spacing (T)",  g_fractionalSpacing,
                      10, 70, callbackCheckBox, HUD_CB_FRACTIONAL_SPACING, 't');
    g_hud.AddCheckBox("Frustum Patch Culling (B)",  g_patchCull,
                      10, 90, callbackCheckBox, HUD_CB_PATCH_CULL, 'b');
    g_hud.AddCheckBox("Bloom (Y)", g_bloom,
                      10, 110, callbackCheckBox, HUD_CB_BLOOM, 'y');
    g_hud.AddCheckBox("Freeze (spc)", g_freeze,
                      10, 130, callbackCheckBox, HUD_CB_FREEZE, ' ');

    g_hud.AddRadioButton(HUD_RB_SCHEME, "CATMARK", true, 10, 190, callbackScheme, 0);
    g_hud.AddRadioButton(HUD_RB_SCHEME, "BILINEAR", false, 10, 210, callbackScheme, 1);

    if (GLUtils::SupportsAdaptiveTessellation())
        g_hud.AddCheckBox("Adaptive (`)", g_adaptive,
                          10, 300, callbackCheckBox, HUD_CB_ADAPTIVE, '`');

    for (int i = 1; i < 8; ++i) {
        char level[16];
        sprintf(level, "Lv. %d", i);
        g_hud.AddRadioButton(HUD_RB_LEVEL, level, i == g_level,
                             10, 320+i*20, callbackLevel, i, '0'+i);
    }

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

    int shading_pulldown = g_hud.AddPullDown("Shading (W)", 250, 10, 250, callbackWireframe, 'w');
    g_hud.AddPullDownButton(shading_pulldown, "Wire", DISPLAY_WIRE, g_wire==DISPLAY_WIRE);
    g_hud.AddPullDownButton(shading_pulldown, "Shaded", DISPLAY_SHADED, g_wire==DISPLAY_SHADED);
    g_hud.AddPullDownButton(shading_pulldown, "Wire+Shaded", DISPLAY_WIRE_ON_SHADED, g_wire==DISPLAY_WIRE_ON_SHADED);

    g_hud.AddLabel("Color (C)", -200, 10);
    g_hud.AddRadioButton(HUD_RB_COLOR, "None", (g_color == COLOR_NONE),
                         -200, 30, callbackColor, COLOR_NONE, 'c');
    g_hud.AddRadioButton(HUD_RB_COLOR, "Ptex Nearest", (g_color == COLOR_PTEX_NEAREST),
                         -200, 50, callbackColor, COLOR_PTEX_NEAREST, 'c');
    g_hud.AddRadioButton(HUD_RB_COLOR, "Ptex HW bilinear", (g_color == COLOR_PTEX_HW_BILINEAR),
                         -200, 70, callbackColor, COLOR_PTEX_HW_BILINEAR, 'c');
    g_hud.AddRadioButton(HUD_RB_COLOR, "Ptex bilinear", (g_color == COLOR_PTEX_BILINEAR),
                         -200, 90, callbackColor, COLOR_PTEX_BILINEAR, 'c');
    g_hud.AddRadioButton(HUD_RB_COLOR, "Ptex biquadratic", (g_color == COLOR_PTEX_BIQUADRATIC),
                         -200, 110, callbackColor, COLOR_PTEX_BIQUADRATIC, 'c');
    g_hud.AddRadioButton(HUD_RB_COLOR, "Patch type", (g_color == COLOR_PATCHTYPE),
                         -200, 130, callbackColor, COLOR_PATCHTYPE, 'c');
    g_hud.AddRadioButton(HUD_RB_COLOR, "Patch coord", (g_color == COLOR_PATCHCOORD),
                         -200, 150, callbackColor, COLOR_PATCHCOORD, 'c');
    g_hud.AddRadioButton(HUD_RB_COLOR, "Normal", (g_color == COLOR_NORMAL),
                         -200, 170, callbackColor, COLOR_NORMAL, 'c');

    if (displacementFilename != NULL) {
        g_hud.AddLabel("Displacement (D)", -200, 200);
        g_hud.AddRadioButton(HUD_RB_DISPLACEMENT, "None",
                             (g_displacement == DISPLACEMENT_NONE),
                             -200, 220, callbackDisplacement, DISPLACEMENT_NONE, 'd');
        g_hud.AddRadioButton(HUD_RB_DISPLACEMENT, "HW bilinear",
                             (g_displacement == DISPLACEMENT_HW_BILINEAR),
                             -200, 240, callbackDisplacement, DISPLACEMENT_HW_BILINEAR, 'd');
        g_hud.AddRadioButton(HUD_RB_DISPLACEMENT, "Bilinear",
                             (g_displacement == DISPLACEMENT_BILINEAR),
                             -200, 260, callbackDisplacement, DISPLACEMENT_BILINEAR, 'd');
        g_hud.AddRadioButton(HUD_RB_DISPLACEMENT, "Biquadratic",
                             (g_displacement == DISPLACEMENT_BIQUADRATIC),
                             -200, 280, callbackDisplacement, DISPLACEMENT_BIQUADRATIC, 'd');

        g_hud.AddLabel("Normal (N)", -200, 310);
        g_hud.AddRadioButton(HUD_RB_NORMAL, "Surface",
                             (g_normal == NORMAL_SURFACE),
                             -200, 330, callbackNormal, NORMAL_SURFACE, 'n');
        g_hud.AddRadioButton(HUD_RB_NORMAL, "Facet",
                             (g_normal == NORMAL_FACET),
                             -200, 350, callbackNormal, NORMAL_FACET, 'n');
        g_hud.AddRadioButton(HUD_RB_NORMAL, "HW Screen space",
                             (g_normal == NORMAL_HW_SCREENSPACE),
                             -200, 370, callbackNormal, NORMAL_HW_SCREENSPACE, 'n');
        g_hud.AddRadioButton(HUD_RB_NORMAL, "Screen space",
                             (g_normal == NORMAL_SCREENSPACE),
                             -200, 390, callbackNormal, NORMAL_SCREENSPACE, 'n');
        g_hud.AddRadioButton(HUD_RB_NORMAL, "Biquadratic",
                             (g_normal == NORMAL_BIQUADRATIC),
                             -200, 410, callbackNormal, NORMAL_BIQUADRATIC, 'n');
        g_hud.AddRadioButton(HUD_RB_NORMAL, "Biquadratic WG",
                             (g_normal == NORMAL_BIQUADRATIC_WG),
                             -200, 430, callbackNormal, NORMAL_BIQUADRATIC_WG, 'n');
    }

    g_hud.AddSlider("Mipmap Bias", 0, 5, 0,
                    -200, 450, 20, false, callbackSlider, 0);
    g_hud.AddSlider("Displacement", 0, 5, 1,
                    -200, 490, 20, false, callbackSlider, 1);
    g_hud.AddCheckBox("Seamless Mipmap", g_seamless,
                      -200, 530, callbackCheckBox, HUD_CB_SEAMLESS_MIPMAP, 'j');

    g_hud.Rebuild(windowWidth, windowHeight, g_width, g_height);

    // create mesh from ptex metadata
    createOsdMesh(g_level, g_kernel);

    // load ptex files
    if (colorFilename)
        g_osdPTexImage = createPtex(colorFilename, colorMem);
    if (displacementFilename)
        g_osdPTexDisplacement = createPtex(displacementFilename, displacementMem);
    if (occlusionFilename)
        g_osdPTexOcclusion = createPtex(occlusionFilename, occlusionMem);
    if (specularFilename)
        g_osdPTexSpecular = createPtex(specularFilename, specularMem);

    g_ptexMemoryUsage =
        (g_osdPTexImage ? g_osdPTexImage->GetMemoryUsage() : 0)
        + (g_osdPTexDisplacement ? g_osdPTexDisplacement->GetMemoryUsage() : 0)
        + (g_osdPTexOcclusion ? g_osdPTexOcclusion->GetMemoryUsage() : 0)
        + (g_osdPTexSpecular ? g_osdPTexSpecular->GetMemoryUsage() : 0);

    // load animation obj sequences (optional)
    if (!animobjs.empty()) {
        //  The Scheme passed here should ideally match the Ptex geometry (not the
        //  defaults from the command line), but only the vertex positions of the
        //  ObjAnim are used, so it is effectively ignored
        g_objAnim = ObjAnim::Create(animobjs, kCatmark);
        if (g_objAnim == 0) {
            printf("Error in reading animation Obj file sequence\n");
            goto error;
        }

        const Shape *animShape = g_objAnim->GetShape();
        if (animShape->verts.size() != g_positions.size()) {
            printf("Error in animation sequence, does not match ptex vertex count\n");
            goto error;
        }

        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    if (diffuseEnvironmentMap) {
        HdrInfo info;
        unsigned char * image = loadHdr(diffuseEnvironmentMap, &info, /*convertToFloat=*/true);
        if (image) {
            glGenTextures(1, &g_diffuseEnvironmentMap);
            glBindTexture(GL_TEXTURE_2D, g_diffuseEnvironmentMap);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, info.width, info.height,
                         0, GL_RGBA, GL_FLOAT, image);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glBindTexture(GL_TEXTURE_2D, 0);
            free(image);
        }
    }
    if (specularEnvironmentMap) {
        HdrInfo info;
        unsigned char * image = loadHdr(specularEnvironmentMap, &info, /*convertToFloat=*/true);
        if (image) {
            glGenTextures(1, &g_specularEnvironmentMap);
            glBindTexture(GL_TEXTURE_2D, g_specularEnvironmentMap);
            // glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP, GL_TRUE);  // deprecated
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, info.width, info.height,
                         0, GL_RGBA, GL_FLOAT, image);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glBindTexture(GL_TEXTURE_2D, 0);
            free(image);
        }
    }
    if (diffuseEnvironmentMap || specularEnvironmentMap) {
        createSky();
    }

    fitFrame();

    while (g_running) {
        idle();
        display();

        glfwPollEvents();
        glfwSwapBuffers(g_window);

        glFinish();
    }
  error:
    uninitGL();
    glfwTerminate();
}
