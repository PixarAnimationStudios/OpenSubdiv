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

#if defined(__APPLE__)
    #if defined(OSD_USES_GLEW)
        #include <GL/glew.h>
    #else
        #include <OpenGL/gl3.h>
    #endif
    #define GLFW_INCLUDE_GL3
    #define GLFW_NO_GLU
#else
    #include <stdlib.h>
    #include <GL/glew.h>
    #if defined(WIN32)
        #include <GL/wglew.h>
    #endif
#endif

#if defined(GLFW_VERSION_3)
    #include <GLFW/glfw3.h>
    GLFWwindow* g_window=0;
    GLFWmonitor* g_primary=0;
#else
    #include <GL/glfw.h>
#endif

#define HBR_ADAPTIVE

#include <hbr/mesh.h>
#include <hbr/bilinear.h>
#include <hbr/catmark.h>
#include <hbr/face.h>

#include <osd/error.h>
#include <osd/glPtexTexture.h>
#include <osd/glDrawContext.h>
#include <osd/glDrawRegistry.h>

#include <osd/cpuGLVertexBuffer.h>
#include <osd/cpuComputeContext.h>
#include <osd/cpuComputeController.h>
OpenSubdiv::OsdCpuComputeController * g_cpuComputeController = NULL;

#ifdef OPENSUBDIV_HAS_OPENMP
    #include <osd/ompComputeController.h>
    OpenSubdiv::OsdOmpComputeController * g_ompComputeController = NULL;
#endif

#ifdef OPENSUBDIV_HAS_OPENCL
    #include <osd/clGLVertexBuffer.h>
    #include <osd/clComputeContext.h>
    #include <osd/clComputeController.h>

    #include "../common/clInit.h"

    cl_context g_clContext;
    cl_command_queue g_clQueue;
    OpenSubdiv::OsdCLComputeController * g_clComputeController = NULL;
#endif

#ifdef OPENSUBDIV_HAS_CUDA
    #include <osd/cudaGLVertexBuffer.h>
    #include <osd/cudaComputeContext.h>
    #include <osd/cudaComputeController.h>

    #include <cuda_runtime_api.h>
    #include <cuda_gl_interop.h>

    #include "../common/cudaInit.h"

    bool g_cudaInitialized = false;
    OpenSubdiv::OsdCudaComputeController * g_cudaComputeController = NULL;
#endif

#ifdef OPENSUBDIV_HAS_GLSL_TRANSFORM_FEEDBACK
    #include <osd/glslTransformFeedbackComputeContext.h>
    #include <osd/glslTransformFeedbackComputeController.h>
    #include <osd/glVertexBuffer.h>
    OpenSubdiv::OsdGLSLTransformFeedbackComputeController * g_glslTransformFeedbackComputeController = NULL;
#endif

#ifdef OPENSUBDIV_HAS_GLSL_COMPUTE
    #include <osd/glslComputeContext.h>
    #include <osd/glslComputeController.h>
    #include <osd/glVertexBuffer.h>
    OpenSubdiv::OsdGLSLComputeController * g_glslComputeController = NULL;
#endif

#include <osd/glMesh.h>
OpenSubdiv::OsdGLMeshInterface *g_mesh;

#include "Ptexture.h"
#include "PtexUtils.h"

#include "../common/stopwatch.h"
#include "../common/simple_math.h"
#include "../common/gl_hud.h"
#include "../common/patchColors.h"
#include "../common/hdr_reader.h"
#include "../../regression/common/shape_utils.h"

#include <vector>
#include <sstream>
#include <fstream>

static const char *g_defaultShaderSource =
#if defined(GL_ARB_tessellation_shader) || defined(GL_VERSION_4_0)
    #include "shader.inc"
#else
    #include "shader_gl3.inc"
#endif
;
static const char *g_skyShaderSource =
#include "skyshader.inc"
;
static const char *g_imageShaderSource =
#include "imageshader.inc"
;
static std::string g_shaderSource;
static const char *g_shaderFilename = NULL;

typedef OpenSubdiv::HbrMesh<OpenSubdiv::OsdVertex>     OsdHbrMesh;
typedef OpenSubdiv::HbrVertex<OpenSubdiv::OsdVertex>   OsdHbrVertex;
typedef OpenSubdiv::HbrFace<OpenSubdiv::OsdVertex>     OsdHbrFace;
typedef OpenSubdiv::HbrHalfedge<OpenSubdiv::OsdVertex> OsdHbrHalfedge;

enum KernelType { kCPU = 0,
                  kOPENMP = 1,
                  kCUDA = 2,
                  kCL = 3,
                  kGLSL = 4,
                  kGLSLCompute = 5 };

enum HudCheckBox { HUD_CB_ADAPTIVE,
                   HUD_CB_DISPLAY_COLOR,
                   HUD_CB_DISPLAY_OCCLUSION,
                   HUD_CB_DISPLAY_DISPLACEMENT,
                   HUD_CB_DISPLAY_NORMALMAP,
                   HUD_CB_DISPLAY_SPECULAR,
                   HUD_CB_ANIMATE_VERTICES,
                   HUD_CB_DISPLAY_PATCH_COLOR,
                   HUD_CB_VIEW_LOD,
                   HUD_CB_FRACTIONAL_SPACING,
                   HUD_CB_PATCH_CULL,
                   HUD_CB_IBL,
                   HUD_CB_BLOOM,
                   HUD_CB_FREEZE };
    
//-----------------------------------------------------------------------------
int   g_frame = 0,
      g_repeatCount = 0;

// GUI variables
int   g_fullscreen=0,
      g_wire = 1,
      g_drawNormals = 0,
      g_mbutton[3] = {0, 0, 0},
      g_level = 2,
      g_tessLevel = 2,
      g_kernel = kCPU,
      g_scheme = 0,
      g_gutterWidth = 1,
      g_running = 1;

float g_moveScale = 0.0f,
      g_displacementScale = 1.0f,
      g_bumpScale = 1.0f;

bool  g_adaptive = false,
      g_yup = false,
      g_displayPatchColor = false,
      g_patchCull = true,
      g_screenSpaceTess = true,
      g_fractionalSpacing = false,
      g_ibl = false,
      g_bloom = false,
      g_freeze = false;

GLuint g_transformUB = 0,
       g_transformBinding = 0,
       g_tessellationUB = 0,
       g_tessellationBinding = 0,
       g_lightingUB = 0,
       g_lightingBinding = 0;

struct Transform {
    float ModelViewMatrix[16];
    float ProjectionMatrix[16];
    float ModelViewProjectionMatrix[16];
    float ModelViewInverseMatrix[16];
} transformData;

// ptex switch
bool  g_color = true,
      g_occlusion = false,
      g_displacement = false,
      g_normal = false,
      g_specular = false;

// camera
float g_rotate[2] = {0, 0},
      g_dolly = 5,
      g_pan[2] = {0, 0},
      g_center[3] = {0, 0, 0},
      g_size = 0;

int   g_prev_x = 0,
      g_prev_y = 0;

// viewport
int   g_width = 1024,
      g_height = 1024;

GLhud g_hud;

// performance
float g_cpuTime = 0;
float g_gpuTime = 0;
#define NUM_FPS_TIME_SAMPLES 6
float g_fpsTimeSamples[NUM_FPS_TIME_SAMPLES] = {0,0,0,0,0,0};
int   g_currentFpsTimeSample = 0;
Stopwatch g_fpsTimer;
float g_animTime = 0;

// geometry
std::vector<float> g_positions,
                   g_normals;

std::vector<std::vector<float> > g_animPositions;

GLuint g_primQuery = 0;
GLuint g_vao = 0;
GLuint g_skyVAO = 0;

GLuint g_diffuseEnvironmentMap = 0;
GLuint g_specularEnvironmentMap = 0;

struct Sky {
    int numIndices;
    GLuint vertexBuffer;
    GLuint elementBuffer;
    GLuint mvpMatrix;
    GLuint program;

    Sky() : numIndices(0), vertexBuffer(0), elementBuffer(0), mvpMatrix(0),
            program(0) {}
} g_sky;

struct ImageShader {
    GLuint blurProgram;
    GLuint hipassProgram;
    GLuint compositeProgram;

    GLuint frameBuffer;
    GLuint frameBufferTexture;
    GLuint frameBufferDepthTexture;

    GLuint smallFrameBuffer[2];
    GLuint smallFrameBufferTexture[2];

    GLuint smallWidth, smallHeight;

    GLuint vao;
    GLuint vbo;

    ImageShader() : blurProgram(0), hipassProgram(0), compositeProgram(0),
                    frameBuffer(0), frameBufferTexture(0), frameBufferDepthTexture(0) {
        smallFrameBuffer[0] = smallFrameBuffer[1] = 0;
        smallFrameBufferTexture[0] = smallFrameBufferTexture[1] = 0;
    }
} g_imageShader;

OpenSubdiv::OsdGLPtexTexture * g_osdPTexImage = 0;
OpenSubdiv::OsdGLPtexTexture * g_osdPTexDisplacement = 0;
OpenSubdiv::OsdGLPtexTexture * g_osdPTexOcclusion = 0;
OpenSubdiv::OsdGLPtexTexture * g_osdPTexSpecular = 0;
const char * g_ptexColorFilename;

static void
checkGLErrors(std::string const & where = "")
{
    GLuint err;
    while ((err = glGetError()) != GL_NO_ERROR) {
        std::cerr << "GL error: "
                  << (where.empty() ? "" : where + " ")
                  << err << "\n";
    }
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
void
updateGeom() {

    int nverts = (int)g_positions.size() / 3;

    if (g_moveScale and g_adaptive and not g_animPositions.empty()) {
        // baked animation only works with adaptive for now
        // (since non-adaptive requires normals)
        int nkey = (int)g_animPositions.size();
        const float fps = 24.0f;

        float p = fmodf(g_animTime * fps, (float)nkey);
        int key = (int)p;
        float b = p - key;

        std::vector<float> vertex;
        vertex.reserve(nverts*3);
        for (int i = 0; i < nverts*3; ++i) {
            float p0 = g_animPositions[key][i];
            float p1 = g_animPositions[(key+1)%nkey][i];
            vertex.push_back(p0*(1-b) + p1*b);
        }
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
template <class T>
OpenSubdiv::HbrMesh<T> * createPTexGeo(PtexTexture * r)
{
  PtexMetaData* meta = r->getMetaData();
  if(meta->numKeys()<3) return NULL;

  const float* vp;
  const int *vi, *vc;
  int nvp, nvi, nvc;

  meta->getValue("PtexFaceVertCounts", vc, nvc);
  if (nvc==0)
      return NULL;

  meta->getValue("PtexVertPositions", vp, nvp);
  if (nvp==0)
      return NULL;

  meta->getValue("PtexFaceVertIndices", vi, nvi);
  if (nvi==0)
      return NULL;

  static OpenSubdiv::HbrCatmarkSubdivision<T>  _catmark;
  static OpenSubdiv::HbrBilinearSubdivision<T>  _bilinear;
  OpenSubdiv::HbrMesh<T> * mesh;
  if(g_scheme == 0)
      mesh = new OpenSubdiv::HbrMesh<T>(&_catmark);
  else
      mesh = new OpenSubdiv::HbrMesh<T>(&_bilinear);

  g_positions.clear();
  g_positions.reserve(nvp);

  // compute model bounding
  float min[3] = {vp[0], vp[1], vp[2]};
  float max[3] = {vp[0], vp[1], vp[2]};
  for (int i=0; i<nvp/3; ++i) {
      for(int j=0; j<3; ++j) {
          float v = vp[i*3+j];
          g_positions.push_back(v);
          min[j] = std::min(min[j], v);
          max[j] = std::max(max[j], v);
      }
      mesh->NewVertex(i, T());
  }
  for (int j=0; j<3; ++j) {
      g_center[j] = (min[j] + max[j]) * 0.5f;
      g_size += (max[j]-min[j])*(max[j]-min[j]);
  }
  g_size = sqrtf(g_size);

  const int *fv = vi;
  for (int i=0, ptxidx=0; i<nvc; ++i) {
      int nv = vc[i];
      OpenSubdiv::HbrFace<T> * face = mesh->NewFace(nv, (int *)fv, 0);

      face->SetPtexIndex(ptxidx);
      if(nv != 4)
          ptxidx+=nv;
      else
          ptxidx++;

      fv += nv;
  }
  mesh->SetInterpolateBoundaryMethod( OpenSubdiv::HbrMesh<T>::k_InterpolateBoundaryEdgeOnly );
//  set creases here
//  applyTags<T>( mesh, sh );
  mesh->Finish();

  return mesh;
}


//------------------------------------------------------------------------------
void
#if GLFW_VERSION_MAJOR>=3
reshape(GLFWwindow *, int width, int height) {
#else
reshape(int width, int height) {
#endif

    g_width = width;
    g_height = height;

    g_hud.Rebuild(width, height);

    // resize framebuffers
    glBindTexture(GL_TEXTURE_2D, g_imageShader.frameBufferTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, 0);
    
    glBindTexture(GL_TEXTURE_2D, g_imageShader.frameBufferDepthTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, width, height, 0,
                 GL_DEPTH_COMPONENT, GL_FLOAT, 0);

    glBindFramebuffer(GL_FRAMEBUFFER, g_imageShader.frameBuffer);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, g_imageShader.frameBufferTexture, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                           GL_TEXTURE_2D, g_imageShader.frameBufferDepthTexture, 0);

    const int d = 4;
    g_imageShader.smallWidth = width/d;
    g_imageShader.smallHeight = height/d;
    for (int i = 0; i < 2; ++i) {
        glBindTexture(GL_TEXTURE_2D, g_imageShader.smallFrameBufferTexture[i]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width/d, height/d, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, 0);
        
        glBindFramebuffer(GL_FRAMEBUFFER, g_imageShader.smallFrameBuffer[i]);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D, g_imageShader.smallFrameBufferTexture[i], 0);
    }
        
    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if(status != GL_FRAMEBUFFER_COMPLETE)
        assert(false);

    glBindTexture(GL_TEXTURE_2D, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    checkGLErrors("Reshape");
}

void reshape() {
#if GLFW_VERSION_MAJOR>=3
    reshape(g_window, g_width, g_height);
#else
    reshape(g_width, g_height);
#endif
}

#if GLFW_VERSION_MAJOR>=3
void windowClose(GLFWwindow*) {
    g_running = false;
}
#else
int windowClose() {
    g_running = false;
    return GL_TRUE;
}
#endif

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
static GLuint compileShader(GLenum shaderType,
                            OpenSubdiv::OsdDrawShaderSource const & common,
                            OpenSubdiv::OsdDrawShaderSource const & source)
{
    const char *sources[4];
    std::stringstream definitions;
    for (int i=0; i<(int)common.defines.size(); ++i) {
        definitions << "#define "
                    << common.defines[i].first << " "
                    << common.defines[i].second << "\n";
    }
    for (int i=0; i<(int)source.defines.size(); ++i) {
        definitions << "#define "
                    << source.defines[i].first << " "
                    << source.defines[i].second << "\n";
    }
    std::string defString = definitions.str();

    sources[0] = source.version.c_str();
    sources[1] = defString.c_str();
    sources[2] = common.source.c_str();
    sources[3] = source.source.c_str();

    GLuint shader = glCreateShader(shaderType);
    glShaderSource(shader, 4, sources, NULL);
    glCompileShader(shader);

    GLint status;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if( status == GL_FALSE ) {
        GLchar emsg[40960];
        glGetShaderInfoLog(shader, sizeof(emsg), 0, emsg);
        fprintf(stderr, "Error compiling GLSL shader: %s\n", emsg );
        fprintf(stderr, "Defines: %s\n", defString.c_str());
        return 0;
    }

    return shader;
}

int bindPTexture(GLint program, OpenSubdiv::OsdGLPtexTexture *osdPTex,
                 GLuint data, GLuint packing, GLuint pages, int samplerUnit)
{
#if defined(GL_ARB_separate_shader_objects) || defined(GL_VERSION_4_1)
    glProgramUniform1i(program, data, samplerUnit + 0);
    glProgramUniform1i(program, packing, samplerUnit + 1);
    glProgramUniform1i(program, pages, samplerUnit + 2);
#else
    glUniform1i(data, samplerUnit + 0);
    glUniform1i(packing, samplerUnit + 1);
    glUniform1i(pages, samplerUnit + 2);
#endif

    glActiveTexture(GL_TEXTURE0 + samplerUnit + 0);
    glBindTexture(GL_TEXTURE_2D_ARRAY, osdPTex->GetTexelsTexture());

    glActiveTexture(GL_TEXTURE0 + samplerUnit + 1);
    glBindTexture(GL_TEXTURE_BUFFER, osdPTex->GetLayoutTextureBuffer());

    glActiveTexture(GL_TEXTURE0 + samplerUnit + 2);
    glBindTexture(GL_TEXTURE_BUFFER, osdPTex->GetPagesTextureBuffer());

    glActiveTexture(GL_TEXTURE0);

    return samplerUnit + 3;
}

//------------------------------------------------------------------------------

union Effect {

    struct {
        int color:1;
        int occlusion:1;
        int displacement:1;
        int normal:1;
        int specular:1;
        int patchCull:1;
        int screenSpaceTess:1;
        int fractionalSpacing:1;
        int ibl:1;
        unsigned int wire:2;
    };
    int value;

    bool operator < (const Effect &e) const {
        return value < e.value;
    }
};

typedef std::pair<OpenSubdiv::OsdDrawContext::PatchDescriptor, Effect> EffectDesc;

class EffectDrawRegistry : public OpenSubdiv::OsdGLDrawRegistry<EffectDesc> {

protected:
    virtual ConfigType *
    _CreateDrawConfig(DescType const & desc, SourceConfigType const * sconfig);

    virtual SourceConfigType *
    _CreateDrawSourceConfig(DescType const & desc);
};

EffectDrawRegistry::SourceConfigType *
EffectDrawRegistry::_CreateDrawSourceConfig(DescType const & desc)
{
    Effect effect = desc.second;

    SourceConfigType * sconfig =
        BaseRegistry::_CreateDrawSourceConfig(desc.first);
    sconfig->commonShader.AddDefine("USE_PTEX_COORD");

    if (effect.patchCull)
        sconfig->commonShader.AddDefine("OSD_ENABLE_PATCH_CULL");
    if (effect.screenSpaceTess)
        sconfig->commonShader.AddDefine("OSD_ENABLE_SCREENSPACE_TESSELLATION");
    if (effect.fractionalSpacing)
        sconfig->commonShader.AddDefine("OSD_FRACTIONAL_ODD_SPACING");

#if defined(GL_ARB_tessellation_shader) || defined(GL_VERSION_4_0)
    const char *glslVersion = "#version 400\n";
#else
    const char *glslVersion = "#version 330\n";
#endif

    bool quad = true;
    if (desc.first.GetType() == OpenSubdiv::FarPatchTables::QUADS) {
        sconfig->vertexShader.source = g_shaderSource;
        sconfig->vertexShader.version = glslVersion;
        sconfig->vertexShader.AddDefine("VERTEX_SHADER");
        if (effect.displacement) {
            sconfig->geometryShader.AddDefine("USE_PTEX_DISPLACEMENT");
            sconfig->geometryShader.AddDefine("FLAT_NORMALS");
        }
    } else {
        quad = false;
        sconfig->tessEvalShader.source = g_shaderSource + sconfig->tessEvalShader.source;
        sconfig->tessEvalShader.version = glslVersion;
        if (effect.displacement and (not effect.normal))
            sconfig->geometryShader.AddDefine("FLAT_NORMALS");
        if (effect.displacement)
            sconfig->tessEvalShader.AddDefine("USE_PTEX_DISPLACEMENT");
    }
    assert(sconfig);

    sconfig->geometryShader.source = g_shaderSource;
    sconfig->geometryShader.version = glslVersion;
    sconfig->geometryShader.AddDefine("GEOMETRY_SHADER");

    sconfig->fragmentShader.source = g_shaderSource;
    sconfig->fragmentShader.version = glslVersion;
    sconfig->fragmentShader.AddDefine("FRAGMENT_SHADER");
    if (effect.color)
        sconfig->fragmentShader.AddDefine("USE_PTEX_COLOR");
    if (effect.occlusion)
        sconfig->fragmentShader.AddDefine("USE_PTEX_OCCLUSION");
    if (effect.normal)
        sconfig->fragmentShader.AddDefine("USE_PTEX_NORMAL");
    if (effect.specular)
        sconfig->fragmentShader.AddDefine("USE_PTEX_SPECULAR");
    if (effect.ibl)
        sconfig->fragmentShader.AddDefine("USE_IBL");

    if (quad) {
        sconfig->geometryShader.AddDefine("PRIM_QUAD");
        sconfig->fragmentShader.AddDefine("PRIM_QUAD");
    } else {
        sconfig->geometryShader.AddDefine("PRIM_TRI");
        sconfig->fragmentShader.AddDefine("PRIM_TRI");
    }
    if (effect.wire == 0) {
        sconfig->geometryShader.AddDefine("GEOMETRY_OUT_WIRE");
        sconfig->fragmentShader.AddDefine("GEOMETRY_OUT_WIRE");
    } else if (effect.wire == 1) {
        sconfig->geometryShader.AddDefine("GEOMETRY_OUT_FILL");
        sconfig->fragmentShader.AddDefine("GEOMETRY_OUT_FILL");
    } else if (effect.wire == 2) {
        sconfig->geometryShader.AddDefine("GEOMETRY_OUT_LINE");
        sconfig->fragmentShader.AddDefine("GEOMETRY_OUT_LINE");
    } 

    return sconfig;
}

EffectDrawRegistry::ConfigType *
EffectDrawRegistry::_CreateDrawConfig(
        DescType const & desc,
        SourceConfigType const * sconfig)
{
    ConfigType * config = BaseRegistry::_CreateDrawConfig(desc.first, sconfig);
    assert(config);

    // XXXdyu can use layout(binding=) with GLSL 4.20 and beyond
    g_transformBinding = 0;
    glUniformBlockBinding(config->program,
        glGetUniformBlockIndex(config->program, "Transform"),
        g_transformBinding);

    g_tessellationBinding = 1;

#if defined(GL_ARB_tessellation_shader) || defined(GL_VERSION_4_0)
    glUniformBlockBinding(config->program,
        glGetUniformBlockIndex(config->program, "Tessellation"),
        g_tessellationBinding);
#endif

    g_lightingBinding = 2;
    glUniformBlockBinding(config->program,
        glGetUniformBlockIndex(config->program, "Lighting"),
        g_lightingBinding);

    GLint loc;
#if defined(GL_ARB_separate_shader_objects) || defined(GL_VERSION_4_1)
    if ((loc = glGetUniformLocation(config->program, "OsdVertexBuffer")) != -1) {
        glProgramUniform1i(config->program, loc, 0); // GL_TEXTURE0
    }
    if ((loc = glGetUniformLocation(config->program, "OsdValenceBuffer")) != -1) {
        glProgramUniform1i(config->program, loc, 1); // GL_TEXTURE1
    }
    if ((loc = glGetUniformLocation(config->program, "OsdQuadOffsetBuffer")) != -1) {
        glProgramUniform1i(config->program, loc, 2); // GL_TEXTURE2
    }
    if ((loc = glGetUniformLocation(config->program, "OsdPatchParamBuffer")) != -1) {
        glProgramUniform1i(config->program, loc, 3); // GL_TEXTURE3
    }
#else
    glUseProgram(config->program);
    if ((loc = glGetUniformLocation(config->program, "OsdVertexBuffer")) != -1) {
        glUniform1i(loc, 0); // GL_TEXTURE0
    }
    if ((loc = glGetUniformLocation(config->program, "OsdValenceBuffer")) != -1) {
        glUniform1i(loc, 1); // GL_TEXTURE1
    }
    if ((loc = glGetUniformLocation(config->program, "OsdQuadOffsetBuffer")) != -1) {
        glUniform1i(loc, 2); // GL_TEXTURE2
    }
    if ((loc = glGetUniformLocation(config->program, "OsdPatchParamBuffer")) != -1) {
        glUniform1i(loc, 3); // GL_TEXTURE3
    }
#endif

    return config;
}

EffectDrawRegistry effectRegistry;

EffectDrawRegistry::ConfigType *
getInstance(Effect effect, OpenSubdiv::OsdDrawContext::PatchDescriptor const & patchDesc) {

    EffectDesc desc(patchDesc, effect);

    EffectDrawRegistry::ConfigType * config =
        effectRegistry.GetDrawConfig(desc);
    assert(config);

    return config;
}

//------------------------------------------------------------------------------
OpenSubdiv::OsdGLPtexTexture *
createPtex(const char *filename) {
    Ptex::String ptexError;
    printf("Loading ptex : %s\n", filename);
    PtexTexture *ptex = PtexTexture::open(filename, ptexError, true);
    if (ptex == NULL) {
        printf("Error in reading %s\n", filename);
        exit(1);
    }
    OpenSubdiv::OsdGLPtexTexture *osdPtex = OpenSubdiv::OsdGLPtexTexture::Create(
        ptex, /*targetMemory =*/0, /*gutterWidth =*/g_gutterWidth, /*pageMargin = */g_gutterWidth*8);

    ptex->release();
    return osdPtex;
}

void
createOsdMesh(int level, int kernel) {

    checkGLErrors("createOsdMesh");

    Ptex::String ptexError;
    PtexTexture *ptexColor = PtexTexture::open(g_ptexColorFilename, ptexError, true);
    if (ptexColor == NULL) {
        printf("Error in reading %s\n", g_ptexColorFilename);
        exit(1);
    }

    // generate Hbr representation from ptex
    OsdHbrMesh * hmesh = createPTexGeo<OpenSubdiv::OsdVertex>(ptexColor);
    if(hmesh == NULL) return;

    g_normals.resize(g_positions.size(),0.0f);
    calcNormals( hmesh, g_positions, g_normals );

    delete g_mesh;
    g_mesh = NULL;

    // Adaptive refinement currently supported only for catmull-clark scheme
    bool doAdaptive = (g_adaptive!=0 and g_scheme==0);

    OpenSubdiv::OsdMeshBitset bits;
    bits.set(OpenSubdiv::MeshAdaptive, doAdaptive);
    bits.set(OpenSubdiv::MeshPtexData, true);

    int numVertexElements = g_adaptive ? 3 : 6;
    int numVaryingElements = 0;

    if (kernel == kCPU) {
        if (not g_cpuComputeController) {
            g_cpuComputeController = new OpenSubdiv::OsdCpuComputeController();
        }
        g_mesh = new OpenSubdiv::OsdMesh<OpenSubdiv::OsdCpuGLVertexBuffer,
                                         OpenSubdiv::OsdCpuComputeController,
                                         OpenSubdiv::OsdGLDrawContext>(
                                                g_cpuComputeController,
                                                hmesh,
                                                numVertexElements,
                                                numVaryingElements,
                                                level, bits);
#ifdef OPENSUBDIV_HAS_OPENMP
    } else if (kernel == kOPENMP) {
        if (not g_ompComputeController) {
            g_ompComputeController = new OpenSubdiv::OsdOmpComputeController();
        }
        g_mesh = new OpenSubdiv::OsdMesh<OpenSubdiv::OsdCpuGLVertexBuffer,
                                         OpenSubdiv::OsdOmpComputeController,
                                         OpenSubdiv::OsdGLDrawContext>(
                                                g_ompComputeController,
                                                hmesh,
                                                numVertexElements,
                                                numVaryingElements,
                                                level, bits);
#endif
#ifdef OPENSUBDIV_HAS_OPENCL
    } else if(kernel == kCL) {
        if (not g_clComputeController) {
            g_clComputeController = new OpenSubdiv::OsdCLComputeController(g_clContext, g_clQueue);
        }
        g_mesh = new OpenSubdiv::OsdMesh<OpenSubdiv::OsdCLGLVertexBuffer,
                                         OpenSubdiv::OsdCLComputeController,
                                         OpenSubdiv::OsdGLDrawContext>(
                                                g_clComputeController,
                                                hmesh,
                                                numVertexElements,
                                                numVaryingElements,
                                                level, bits, g_clContext, g_clQueue);
#endif
#ifdef OPENSUBDIV_HAS_CUDA
    } else if(kernel == kCUDA) {
        if (not g_cudaComputeController) {
            g_cudaComputeController = new OpenSubdiv::OsdCudaComputeController();
        }
        g_mesh = new OpenSubdiv::OsdMesh<OpenSubdiv::OsdCudaGLVertexBuffer,
                                         OpenSubdiv::OsdCudaComputeController,
                                         OpenSubdiv::OsdGLDrawContext>(
                                                g_cudaComputeController,
                                                hmesh,
                                                numVertexElements,
                                                numVaryingElements,
                                                level, bits);
#endif
#ifdef OPENSUBDIV_HAS_GLSL_TRANSFORM_FEEDBACK
    } else if(kernel == kGLSL) {
        if (not g_glslTransformFeedbackComputeController) {
            g_glslTransformFeedbackComputeController = new OpenSubdiv::OsdGLSLTransformFeedbackComputeController();
        }
        g_mesh = new OpenSubdiv::OsdMesh<OpenSubdiv::OsdGLVertexBuffer,
                                         OpenSubdiv::OsdGLSLTransformFeedbackComputeController,
                                         OpenSubdiv::OsdGLDrawContext>(
                                                g_glslTransformFeedbackComputeController,
                                                hmesh,
                                                numVertexElements,
                                                numVaryingElements,
                                                level, bits);
#endif
#ifdef OPENSUBDIV_HAS_GLSL_COMPUTE
    } else if(kernel == kGLSLCompute) {
        if (not g_glslComputeController) {
            g_glslComputeController = new OpenSubdiv::OsdGLSLComputeController();
        }
        g_mesh = new OpenSubdiv::OsdMesh<OpenSubdiv::OsdGLVertexBuffer,
                                         OpenSubdiv::OsdGLSLComputeController,
                                         OpenSubdiv::OsdGLDrawContext>(
                                                g_glslComputeController,
                                                hmesh,
                                                numVertexElements,
                                                numVaryingElements,
                                                level, bits);
#endif
    } else {
        printf("Unsupported kernel %s\n", getKernelName(kernel));
    }

    delete hmesh;
    
    if (glGetError() != GL_NO_ERROR){
        printf ("GLERROR\n");
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
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof (GLfloat) * 6, 0);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof (GLfloat) * 6, (float*)12);
    }
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_mesh->GetDrawContext()->GetPatchIndexBuffer());

    glBindVertexArray(0);
}

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

    g_sky.program = glCreateProgram();

    OpenSubdiv::OsdDrawShaderSource common, vertexShader, fragmentShader;
    vertexShader.source = g_skyShaderSource;
    vertexShader.version = "#version 410\n";
    vertexShader.AddDefine("SKY_VERTEX_SHADER");
    fragmentShader.source = g_skyShaderSource;
    fragmentShader.version = "#version 410\n";
    fragmentShader.AddDefine("SKY_FRAGMENT_SHADER");
    GLuint vs = compileShader(GL_VERTEX_SHADER, 
                              common, vertexShader);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER,
                              common, fragmentShader);

    glAttachShader(g_sky.program, vs);
    glAttachShader(g_sky.program, fs);
    glLinkProgram(g_sky.program);

    glDeleteShader(vs);
    glDeleteShader(fs);

    GLint environmentMap = glGetUniformLocation(g_sky.program, "environmentMap");
#if defined(GL_ARB_separate_shader_objects) || defined(GL_VERSION_4_1)
    if (g_specularEnvironmentMap)
        glProgramUniform1i(g_sky.program, environmentMap, 6);
    else
        glProgramUniform1i(g_sky.program, environmentMap, 5);
#else
    glUseProgram(g_sky.program);
    if (g_specularEnvironmentMap)
      glUniform1i(environmentMap, 6);
    else
      glUniform1i(environmentMap, 5);
#endif

    g_sky.mvpMatrix = glGetUniformLocation(g_sky.program, "ModelViewProjectionMatrix");
}

GLuint
compileImageShader(const char *define) {

    GLuint program = glCreateProgram();

    OpenSubdiv::OsdDrawShaderSource common, vertexShader, fragmentShader;
    vertexShader.source = g_imageShaderSource;
    vertexShader.version = "#version 410\n";
    vertexShader.AddDefine("IMAGE_VERTEX_SHADER");
    fragmentShader.source = g_imageShaderSource;
    fragmentShader.version = "#version 410\n";
    fragmentShader.AddDefine("IMAGE_FRAGMENT_SHADER");
    fragmentShader.AddDefine(define);

    GLuint vs = compileShader(GL_VERTEX_SHADER,
                              common, vertexShader);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER,
                              common, fragmentShader);

    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);

    glDeleteShader(vs);
    glDeleteShader(fs);

#if defined(GL_ARB_separate_shader_objects) || defined(GL_VERSION_4_1)
    GLint colorMap = glGetUniformLocation(program, "colorMap");
    if (colorMap != -1)
        glProgramUniform1i(program, colorMap, 0);  // GL_TEXTURE0
    GLint depthMap = glGetUniformLocation(program, "depthMap");
    if (depthMap != -1)
        glProgramUniform1i(program, depthMap, 1);  // GL_TEXTURE1
#else
    glUseProgram(program);
    GLint colorMap = glGetUniformLocation(program, "colorMap");
    if (colorMap != -1)
        glUniform1i(colorMap, 0);  // GL_TEXTURE0
    GLint depthMap = glGetUniformLocation(program, "depthMap");
    if (depthMap != -1)
        glUniform1i(depthMap, 1);  // GL_TEXTURE1
#endif

    return program;
}

void
createImageShader() {

    g_imageShader.blurProgram = compileImageShader("BLUR");
    g_imageShader.hipassProgram = compileImageShader("HIPASS");
    g_imageShader.compositeProgram = compileImageShader("COMPOSITE");

    glGenVertexArrays(1, &g_imageShader.vao);
    glBindVertexArray(g_imageShader.vao);
    glGenBuffers(1, &g_imageShader.vbo);
    float pos[] = { -1, -1, 1, -1, -1,  1, 1,  1 };
    glGenBuffers(1, &g_imageShader.vbo);
    glBindBuffer(GL_ARRAY_BUFFER, g_imageShader.vbo);
    glBufferData(GL_ARRAY_BUFFER,
                 sizeof(pos), pos, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, g_imageShader.vbo);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void
applyImageShader() {

    int w = g_imageShader.smallWidth, h = g_imageShader.smallHeight;
    const float hoffsets[10] = {
        -2.0f / w, 0,
        -1.0f / w, 0,
        0, 0,
        +1.0f / w, 0,
        +2.0f / w, 0,
    };
    const float voffsets[10] = {
        0, -2.0f / h,
        0, -1.0f / h,
        0, 0,
        0, +1.0f / h,
        0, +2.0f / h,
    };
    const float weights[5] = {
        1.0f / 16.0f,
        4.0f / 16.0f,
        6.0f / 16.0f,
        4.0f / 16.0f,
        1.0f / 16.0f,
    };

    checkGLErrors("image shader begin");
    glBindVertexArray(g_imageShader.vao);

    GLint uniformAlpha = glGetUniformLocation(g_imageShader.compositeProgram, "alpha");

    if (g_bloom) {
        // XXX: fix me
        GLint uniformOffsets = glGetUniformLocation(g_imageShader.blurProgram, "Offsets");
        GLint uniformWeights = glGetUniformLocation(g_imageShader.blurProgram, "Weights");

        // down sample
        glUseProgram(g_imageShader.hipassProgram);
        glViewport(0, 0, g_imageShader.smallWidth, g_imageShader.smallHeight);
        glBindFramebuffer(GL_FRAMEBUFFER, g_imageShader.smallFrameBuffer[0]);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, g_imageShader.frameBufferTexture);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        // horizontal blur pass
        glUseProgram(g_imageShader.blurProgram);
        glBindFramebuffer(GL_FRAMEBUFFER, g_imageShader.smallFrameBuffer[1]);
        glBindTexture(GL_TEXTURE_2D, g_imageShader.smallFrameBufferTexture[0]);
        glUniform2fv(uniformOffsets, 5, hoffsets);
        glUniform1fv(uniformWeights, 5, weights);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        // vertical blur pass
        glBindFramebuffer(GL_FRAMEBUFFER, g_imageShader.smallFrameBuffer[0]);
        glBindTexture(GL_TEXTURE_2D, g_imageShader.smallFrameBufferTexture[1]);
        glUniform2fv(uniformOffsets, 5, voffsets);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    }


    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glClear(GL_COLOR_BUFFER_BIT);
    glViewport(0, 0, g_width, g_height);

    // blit full-res
    glUseProgram(g_imageShader.compositeProgram);
    glUniform1f(uniformAlpha, 1);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, g_imageShader.frameBufferTexture);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    if (g_bloom) {
        glUseProgram(g_imageShader.compositeProgram);
        glUniform1f(uniformAlpha, 0.5);
        glBlendFunc(GL_ONE, GL_ONE);
        glEnable(GL_BLEND);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, g_imageShader.smallFrameBufferTexture[0]);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        glDisable(GL_BLEND);
    }

    glBindVertexArray(0);
    glUseProgram(0);

    checkGLErrors("image shader");
}

//------------------------------------------------------------------------------
static GLuint
bindProgram(Effect effect, OpenSubdiv::OsdDrawContext::PatchArray const & patch)
{
    OpenSubdiv::OsdDrawContext::PatchDescriptor const & desc = patch.GetDescriptor();

    EffectDrawRegistry::ConfigType *
        config = getInstance(effect, desc);

    GLuint program = config->program;

    glUseProgram(program);

    if (! g_transformUB) {
        glGenBuffers(1, &g_transformUB);
        glBindBuffer(GL_UNIFORM_BUFFER, g_transformUB);
        glBufferData(GL_UNIFORM_BUFFER,
                sizeof(transformData), NULL, GL_STATIC_DRAW);
    };
    glBindBuffer(GL_UNIFORM_BUFFER, g_transformUB);
    glBufferSubData(GL_UNIFORM_BUFFER,
                0, sizeof(transformData), &transformData);
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

    // Update and bind lighting state
    struct Lighting {
        struct Light {
            float position[4];
            float ambient[4];
            float diffuse[4];
            float specular[4];
        } lightSource[2];
    } lightingData = {
       {{  { 0.6f, 1.0f, 0.6f, 0.0f },
           { 0.1f, 0.1f, 0.1f, 1.0f },
           { 1.7f, 1.3f, 1.1f, 1.0f },
           { 1.0f, 1.0f, 1.0f, 1.0f } },
 
         { { -0.8f, 0.6f, -0.7f, 0.0f },
           {  0.0f, 0.0f,  0.0f, 1.0f },
           {  0.8f, 0.8f,  1.5f, 1.0f },
           {  0.4f, 0.4f,  0.4f, 1.0f } }}
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

    //-----------------
    int sampler = 7;

    // color ptex
    GLint texData = glGetUniformLocation(program, "textureImage_Data");
    GLint texPacking = glGetUniformLocation(program, "textureImage_Packing");
    GLint texPages = glGetUniformLocation(program, "textureImage_Pages");
    sampler = bindPTexture(program, g_osdPTexImage, texData, texPacking, texPages, sampler);

    // displacement ptex
    if (g_displacement || g_normal) {
        texData = glGetUniformLocation(program, "textureDisplace_Data");
        texPacking = glGetUniformLocation(program, "textureDisplace_Packing");
        texPages = glGetUniformLocation(program, "textureDisplace_Pages");
        sampler = bindPTexture(program, g_osdPTexDisplacement, texData, texPacking, texPages, sampler);
    }

    // occlusion ptex
    if (g_occlusion) {
        texData = glGetUniformLocation(program, "textureOcclusion_Data");
        texPacking = glGetUniformLocation(program, "textureOcclusion_Packing");
        texPages = glGetUniformLocation(program, "textureOcclusion_Pages");
        sampler = bindPTexture(program, g_osdPTexOcclusion, texData, texPacking, texPages, sampler);
    }

    // specular ptex
    if (g_specular) {
        texData = glGetUniformLocation(program, "textureSpecular_Data");
        texPacking = glGetUniformLocation(program, "textureSpecular_Packing");
        texPages = glGetUniformLocation(program, "textureSpecular_Pages");
        sampler = bindPTexture(program, g_osdPTexSpecular, texData, texPacking, texPages, sampler);
    }

    // other textures
    if (g_ibl) {
        if (g_diffuseEnvironmentMap) {
#if defined(GL_ARB_separate_shader_objects) || defined(GL_VERSION_4_1)
            glProgramUniform1i(program, glGetUniformLocation(program, "diffuseEnvironmentMap"), 5);
#else
            glUniform1i(glGetUniformLocation(program, "diffuseEnvironmentMap"), 5);
#endif
            glActiveTexture(GL_TEXTURE5);
            glBindTexture(GL_TEXTURE_2D, g_diffuseEnvironmentMap);
            sampler++;
        }
        if (g_specularEnvironmentMap) {
#if defined(GL_ARB_separate_shader_objects) || defined(GL_VERSION_4_1)
            glProgramUniform1i(program, glGetUniformLocation(program, "specularEnvironmentMap"), 6);
#else
            glUniform1i(glGetUniformLocation(program, "specularEnvironmentMap"), 6);
#endif
            glActiveTexture(GL_TEXTURE6);
            glBindTexture(GL_TEXTURE_2D, g_specularEnvironmentMap);
            sampler++;
        }
        glActiveTexture(GL_TEXTURE0);
    }

    return program;
}

//------------------------------------------------------------------------------
void
drawModel() {
#if defined(GL_ARB_tessellation_shader) || defined(GL_VERSION_4_0)
    GLuint bVertex = g_mesh->BindVertexBuffer();
#else
    g_mesh->BindVertexBuffer();
#endif

    OpenSubdiv::OsdDrawContext::PatchArrayVector const & patches = g_mesh->GetDrawContext()->patchArrays;
    glBindVertexArray(g_vao);

    // patch drawing
    for (int i=0; i<(int)patches.size(); ++i) {
        OpenSubdiv::OsdDrawContext::PatchArray const & patch = patches[i];

        OpenSubdiv::OsdDrawContext::PatchDescriptor desc = patch.GetDescriptor();
        OpenSubdiv::FarPatchTables::Type patchType = desc.GetType();

        GLenum primType;
        switch(patchType) {
        case OpenSubdiv::FarPatchTables::QUADS:
            primType = GL_LINES_ADJACENCY;
            break;
        case OpenSubdiv::FarPatchTables::TRIANGLES:
            primType = GL_TRIANGLES;
            break;
        default:
#if defined(GL_ARB_tessellation_shader) || defined(GL_VERSION_4_0)
            primType = GL_PATCHES;
            glPatchParameteri(GL_PATCH_VERTICES, desc.GetNumControlVertices());
#else
            primType = GL_POINTS;
#endif
        }

#if defined(GL_ARB_tessellation_shader) || defined(GL_VERSION_4_0)
        if (g_mesh->GetDrawContext()->GetVertexTextureBuffer()) {
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_BUFFER,
                g_mesh->GetDrawContext()->GetVertexTextureBuffer());
            glTexBuffer(GL_TEXTURE_BUFFER, GL_R32F, bVertex);
        }
        if (g_mesh->GetDrawContext()->GetVertexValenceTextureBuffer()) {
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_BUFFER,
                g_mesh->GetDrawContext()->GetVertexValenceTextureBuffer());
        }
        if (g_mesh->GetDrawContext()->GetQuadOffsetsTextureBuffer()) {
            glActiveTexture(GL_TEXTURE2);
            glBindTexture(GL_TEXTURE_BUFFER,
                g_mesh->GetDrawContext()->GetQuadOffsetsTextureBuffer());
        }
#endif
        if (g_mesh->GetDrawContext()->GetPatchParamTextureBuffer()) {
            glActiveTexture(GL_TEXTURE3);
            glBindTexture(GL_TEXTURE_BUFFER,
                g_mesh->GetDrawContext()->GetPatchParamTextureBuffer());
        }
        glActiveTexture(GL_TEXTURE0);

        Effect effect;
        effect.value = 0;
        effect.color = g_color;
        effect.occlusion = g_occlusion;
        effect.displacement = g_displacement;
        effect.normal = g_normal;
        effect.specular = g_specular;
        effect.patchCull = g_patchCull;
        effect.screenSpaceTess = g_screenSpaceTess;
        effect.fractionalSpacing = g_fractionalSpacing;
        effect.ibl = g_ibl;
        effect.wire = g_wire;

        GLuint program = bindProgram(effect, patch);

        GLint nonAdaptiveLevel = glGetUniformLocation(program, "nonAdaptiveLevel");
        if (nonAdaptiveLevel != -1) {
#if defined(GL_ARB_separate_shader_objects) || defined(GL_VERSION_4_1)
            glProgramUniform1i(program, nonAdaptiveLevel, g_level);
#else
            glUniform1i(nonAdaptiveLevel, g_level);
#endif
        }

        GLint displacementScale = glGetUniformLocation(program, "displacementScale");
        if (displacementScale != -1)
            glUniform1f(displacementScale, g_displacementScale);
        GLint bumpScale = glGetUniformLocation(program, "bumpScale");
        if (bumpScale != -1)
            glUniform1f(bumpScale, g_bumpScale);

#if defined(GL_ARB_tessellation_shader) || defined(GL_VERSION_4_0)
        GLuint overrideColorEnable = glGetUniformLocation(program, "overrideColorEnable");
        GLuint overrideColor = glGetUniformLocation(program, "overrideColor");

        float const * color = getAdaptivePatchColor( desc );
        glProgramUniform4f(program, overrideColor, color[0], color[1], color[2], color[3]);

        if (g_displayPatchColor or g_wire == 2) {
            glProgramUniform1i(program, overrideColorEnable, 1);
        } else {
            glProgramUniform1i(program, overrideColorEnable, 0);
        }
#endif

        if (g_wire == 0) {
            glDisable(GL_CULL_FACE);
        }

        GLuint uniformGregoryQuadOffsetBase =
	  glGetUniformLocation(program, "OsdGregoryQuadOffsetBase");
        GLuint uniformPrimitiveIdBase =
	  glGetUniformLocation(program, "OsdPrimitiveIdBase");

#if defined(GL_ARB_tessellation_shader) || defined(GL_VERSION_4_0)
        glProgramUniform1i(program, uniformGregoryQuadOffsetBase,
			   patch.GetQuadOffsetIndex());
        glProgramUniform1i(program, uniformPrimitiveIdBase,
			   patch.GetPatchIndex());
#else
        glUniform1i(uniformGregoryQuadOffsetBase,
		    patch.GetQuadOffsetIndex());
        glUniform1i(uniformPrimitiveIdBase,
		    patch.GetPatchIndex());
#endif

        glDrawElements(primType,
                       patch.GetNumIndices(), GL_UNSIGNED_INT,
                       (void *)(patch.GetVertIndex() * sizeof(unsigned int)));
        if (g_wire == 0) {
            glEnable(GL_CULL_FACE);
        }
    }
    glBindVertexArray(0);
}

void
drawSky() {
    
    glUseProgram(g_sky.program);

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
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof (GLfloat) * 5, 0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof (GLfloat) * 5, (void*)(sizeof(GLfloat)*3));

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_sky.elementBuffer);
    glDrawElements(GL_TRIANGLES, g_sky.numIndices, GL_UNSIGNED_INT, 0);

    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);

    glBindVertexArray(0);

    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
 }

void
display() {

    glBindFramebuffer(GL_FRAMEBUFFER, g_imageShader.frameBuffer);

    Stopwatch s;
    s.Start();

    glViewport(0, 0, g_width, g_height);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (g_ibl) {
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        drawSky();
    }

    // primitive counting
    glBeginQuery(GL_PRIMITIVES_GENERATED, g_primQuery);

    double aspect = g_width/(double)g_height;
    identity(transformData.ModelViewMatrix);
    translate(transformData.ModelViewMatrix, -g_pan[0], -g_pan[1], -g_dolly);
    rotate(transformData.ModelViewMatrix, g_rotate[1], 1, 0, 0);
    rotate(transformData.ModelViewMatrix, g_rotate[0], 0, 1, 0);
    if (g_yup)
        rotate(transformData.ModelViewMatrix, -90, 1, 0, 0);
    translate(transformData.ModelViewMatrix, -g_center[0], -g_center[1], -g_center[2]);
    perspective(transformData.ProjectionMatrix, 45.0f, (float)aspect, g_size*0.001f,
                g_size+g_dolly);

    multMatrix(transformData.ModelViewProjectionMatrix, transformData.ModelViewMatrix, transformData.ProjectionMatrix);
    inverseMatrix(transformData.ModelViewInverseMatrix, transformData.ModelViewMatrix);

    glEnable(GL_DEPTH_TEST);

    drawModel();

    glDisable(GL_DEPTH_TEST);

    glUseProgram(0);

    glEndQuery(GL_PRIMITIVES_GENERATED);

    applyImageShader();

    s.Stop();
    float drawCpuTime = float(s.GetElapsed() * 1000.0f);
    s.Start();
    glFinish();
    s.Stop();
    float drawGpuTime = float(s.GetElapsed() * 1000.0f);

    GLuint numPrimsGenerated = 0;
    glGetQueryObjectuiv(g_primQuery, GL_QUERY_RESULT, &numPrimsGenerated);

    g_fpsTimer.Stop();
    float elapsed = (float)g_fpsTimer.GetElapsed();
    if (not g_freeze)
        g_animTime += elapsed;
    g_fpsTimer.Start();

    if (g_hud.IsVisible()) {
        double fps = 1.0/elapsed;

        // Avereage fps over a defined number of time samples for
        // easier reading in the HUD
        g_fpsTimeSamples[g_currentFpsTimeSample++] = float(fps);
        if (g_currentFpsTimeSample >= NUM_FPS_TIME_SAMPLES)
            g_currentFpsTimeSample = 0;
        double averageFps = 0;
        for (int i=0; i< NUM_FPS_TIME_SAMPLES; ++i) {
            averageFps += g_fpsTimeSamples[i]/(float)NUM_FPS_TIME_SAMPLES;
        }

        g_hud.DrawString(10, -180, "Tess level (+/-): %d", g_tessLevel);
        if (numPrimsGenerated > 1000000) {
            g_hud.DrawString(10, -160, "Primitives      : %3.1f million", (float)numPrimsGenerated/1000000.0);
        } else if (numPrimsGenerated > 1000) {
            g_hud.DrawString(10, -160, "Primitives      : %3.1f thousand", (float)numPrimsGenerated/1000.0);
        } else {
            g_hud.DrawString(10, -160, "Primitives      : %d", numPrimsGenerated);
        }
        g_hud.DrawString(10, -140, "Vertices        : %d", g_mesh->GetNumVertices());
        g_hud.DrawString(10, -120, "Scheme          : %s", g_scheme == 0 ? "CATMARK" : "LOOP");
        g_hud.DrawString(10, -100, "GPU Kernel      : %.3f ms", g_gpuTime);
        g_hud.DrawString(10, -80,  "CPU Kernel      : %.3f ms", g_cpuTime);
        g_hud.DrawString(10, -60,  "GPU Draw        : %.3f ms", drawGpuTime);
        g_hud.DrawString(10, -40,  "CPU Draw        : %.3f ms", drawCpuTime);
        g_hud.DrawString(10, -20,  "FPS             : %3.1f", averageFps);
    
        g_hud.Flush();
    }

    glFinish();

    checkGLErrors("draw end");
}

//------------------------------------------------------------------------------
static void
#if GLFW_VERSION_MAJOR>=3
mouse(GLFWwindow *, int button, int state, int mods) {
#else
mouse(int button, int state) {
#endif

    if (button == 0 && state == GLFW_PRESS && g_hud.MouseClick(g_prev_x, g_prev_y))
        return;
    g_mbutton[button] = (state == GLFW_PRESS);
}

//------------------------------------------------------------------------------
static void 
#if GLFW_VERSION_MAJOR>=3
motion(GLFWwindow *, double dx, double dy) {
    int x=(int)dx, y=(int)dy;
#else
motion(int x, int y) {
#endif

    if (g_mbutton[0] && !g_mbutton[1] && !g_mbutton[2]) {
        // orbit
        g_rotate[0] += x - g_prev_x;
        g_rotate[1] += y - g_prev_y;
    } else if (!g_mbutton[0] && !g_mbutton[1] && g_mbutton[2]) {
        // pan
        g_pan[0] -= g_dolly*(x - g_prev_x)/g_width;
        g_pan[1] += g_dolly*(y - g_prev_y)/g_height;
    } else if ((g_mbutton[0] && !g_mbutton[1] && g_mbutton[2]) or
               (!g_mbutton[0] && g_mbutton[1] && !g_mbutton[2])) {
        // dolly
        g_dolly -= g_dolly*0.01f*(x - g_prev_x);
        if(g_dolly <= 0.01) g_dolly = 0.01f;
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

    glDeleteQueries(1, &g_primQuery);
    glDeleteVertexArrays(1, &g_vao);
    glDeleteVertexArrays(1, &g_skyVAO);

    if(g_mesh)
        delete g_mesh;

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

#ifdef OPENSUBDIV_HAS_GLSL_TRANSFORM_FEEDBACK
    delete g_glslTransformFeedbackComputeController;
#endif

#ifdef OPENSUBDIV_HAS_GLSL_COMPUTE
    delete g_glslComputeController;
#endif

    if (g_diffuseEnvironmentMap) glDeleteTextures(1, &g_diffuseEnvironmentMap);
    if (g_specularEnvironmentMap) glDeleteTextures(1, &g_specularEnvironmentMap);

    if (g_sky.program) glDeleteProgram(g_sky.program);
    if (g_sky.vertexBuffer) glDeleteBuffers(1, &g_sky.vertexBuffer);
    if (g_sky.elementBuffer) glDeleteBuffers(1, &g_sky.elementBuffer);

    glDeleteFramebuffers(1, &g_imageShader.frameBuffer);
    glDeleteTextures(1, &g_imageShader.frameBufferTexture);
    glDeleteTextures(1, &g_imageShader.frameBufferDepthTexture);

    glDeleteFramebuffers(2, g_imageShader.smallFrameBuffer);
    glDeleteTextures(2, g_imageShader.smallFrameBufferTexture);

    glDeleteProgram(g_imageShader.blurProgram);
    glDeleteProgram(g_imageShader.hipassProgram);
    glDeleteProgram(g_imageShader.compositeProgram);

    glDeleteVertexArrays(1, &g_imageShader.vao);
    glDeleteBuffers(1, &g_imageShader.vbo);
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
    createOsdMesh(g_level, g_kernel);
}
static void
callbackScheme(int s)
{
    g_scheme = s;
    createOsdMesh(g_level, g_kernel);
}
static void
callbackLevel(int l)
{
    g_level = l;
    createOsdMesh(g_level, g_kernel);
}
static void
callbackCheckBox(bool checked, int button)
{
    bool rebuild = false;

    switch(button) {
    case HUD_CB_ADAPTIVE:
        if (OpenSubdiv::OsdGLDrawContext::SupportsAdaptiveTessellation()) {
            g_adaptive = checked;
            rebuild = true;
        }
        break;
    case HUD_CB_DISPLAY_COLOR:
        g_color = checked;
        break;
    case HUD_CB_DISPLAY_OCCLUSION:
        g_occlusion = checked;
        break;
    case HUD_CB_DISPLAY_DISPLACEMENT:
        g_displacement = checked;
        break;
    case HUD_CB_DISPLAY_NORMALMAP:
        g_normal = checked;
        break;
    case HUD_CB_DISPLAY_SPECULAR:
        g_specular = checked;
        break;
    case HUD_CB_ANIMATE_VERTICES:
        g_moveScale = checked ? 1.0f : 0.0f;
        g_animTime = 0;
        break;
    case HUD_CB_DISPLAY_PATCH_COLOR:
        g_displayPatchColor = checked;
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
    case HUD_CB_FREEZE:
        g_freeze = checked;
        break;
    }

    if (rebuild)
        createOsdMesh(g_level, g_kernel);
}
//-------------------------------------------------------------------------------
void
reloadShaderFile() {
    if (not g_shaderFilename) return;

    std::ifstream ifs(g_shaderFilename);
    if (not ifs) return;
    printf("Load shader %s\n", g_shaderFilename);

    std::stringstream ss;
    ss << ifs.rdbuf();
    ifs.close();

    g_shaderSource = ss.str();

    effectRegistry.Reset();
}

//------------------------------------------------------------------------------
static void 
toggleFullScreen() {
    // XXXX manuelk : to re-implement from glut
}

//------------------------------------------------------------------------------
void
#if GLFW_VERSION_MAJOR>=3
keyboard(GLFWwindow *, int key, int scancode, int event, int mods) {
#else
#define GLFW_KEY_ESCAPE GLFW_KEY_ESC
keyboard(int key, int event) {
#endif

    if (event == GLFW_RELEASE) return;
    if (g_hud.KeyDown(tolower(key))) return;

    switch (key) {
        case 'Q': g_running = 0; break;
        case 'E': g_drawNormals = (g_drawNormals+1)%2; break;
        case 'F': fitFrame(); break;
        case GLFW_KEY_TAB: toggleFullScreen(); break;
        case 'G': g_gutterWidth = (g_gutterWidth+1)%8; createOsdMesh(g_level, g_kernel); break;
        case 'R': reloadShaderFile(); createOsdMesh(g_level, g_kernel); break;
        case '+':
        case '=': g_tessLevel++; break;
        case '-': g_tessLevel = std::max(1, g_tessLevel-1); break;
        case GLFW_KEY_ESCAPE: g_hud.SetVisible(!g_hud.IsVisible()); break;
    }
}

//------------------------------------------------------------------------------
void
idle() {

    if (not g_freeze)
        g_frame++;

    updateGeom();

    if(g_repeatCount != 0 && g_frame >= g_repeatCount)
        g_running = 0;
}

//------------------------------------------------------------------------------
void
initGL() {

    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glEnable(GL_CULL_FACE);

    glGenQueries(1, &g_primQuery);
    glGenVertexArrays(1, &g_vao);
    glGenVertexArrays(1, &g_skyVAO);

    glGenFramebuffers(1, &g_imageShader.frameBuffer);
    glGenTextures(1, &g_imageShader.frameBufferTexture);
    glGenTextures(1, &g_imageShader.frameBufferDepthTexture);

    glGenFramebuffers(2, g_imageShader.smallFrameBuffer);
    glGenTextures(2, g_imageShader.smallFrameBufferTexture);

    glBindTexture(GL_TEXTURE_2D, g_imageShader.frameBufferTexture);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glBindTexture(GL_TEXTURE_2D, g_imageShader.frameBufferDepthTexture);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    for (int i = 0; i < 2; ++i) {
        glBindTexture(GL_TEXTURE_2D, g_imageShader.smallFrameBufferTexture[i]);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }
    
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
    printf("          -y                      : Y-up model\n");
    printf("          --disp <scale>          : Displacment scale\n");
    printf("          --bump <scale>          : Bump normal scale\n");

}

//------------------------------------------------------------------------------
static void
callbackError(OpenSubdiv::OsdErrorType err, const char *message)
{
    printf("OsdError: %d\n", err);
    printf("%s", message);
}

//------------------------------------------------------------------------------
static void
setGLCoreProfile()
{
#if GLFW_VERSION_MAJOR>=3
    #define glfwOpenWindowHint glfwWindowHint
    #define GLFW_OPENGL_VERSION_MAJOR GLFW_CONTEXT_VERSION_MAJOR
    #define GLFW_OPENGL_VERSION_MINOR GLFW_CONTEXT_VERSION_MINOR
#endif

#if GLFW_VERSION_MAJOR>=2 and GLFW_VERSION_MINOR >=7
    glfwOpenWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#if not defined(__APPLE__)
    glfwOpenWindowHint(GLFW_OPENGL_VERSION_MAJOR, 4);
    glfwOpenWindowHint(GLFW_OPENGL_VERSION_MINOR, 2);
    glfwOpenWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#else
    glfwOpenWindowHint(GLFW_OPENGL_VERSION_MAJOR, 3);
    glfwOpenWindowHint(GLFW_OPENGL_VERSION_MINOR, 2);
#endif
#endif
    
}

//------------------------------------------------------------------------------
int main(int argc, char ** argv) {

    std::vector<std::string> animobjs;
    const char *diffuseEnvironmentMap = NULL, *specularEnvironmentMap = NULL;
    const char *colorFilename = NULL, *displacementFilename = NULL,
        *occlusionFilename = NULL, *specularFilename = NULL;
    bool fullscreen = false;

    for (int i = 1; i < argc; ++i) {
        if (strstr(argv[i], ".obj")) {
            animobjs.push_back(argv[i]);
        } else if (!strcmp(argv[i], "-l"))
            g_level = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-c"))
            g_repeatCount = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-d"))
            diffuseEnvironmentMap = argv[++i];
        else if (!strcmp(argv[i], "-e"))
            specularEnvironmentMap = argv[++i];
        else if (!strcmp(argv[i], "-s"))
            g_shaderFilename = argv[++i];
        else if (!strcmp(argv[i], "-f"))
            fullscreen = true;
        else if (!strcmp(argv[i], "-y"))
            g_yup = true;
        else if (!strcmp(argv[i], "--disp"))
            g_displacementScale = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--bump"))
            g_bumpScale = (float)atof(argv[++i]);
        else if (colorFilename == NULL)
            colorFilename = argv[i];
        else if (displacementFilename == NULL) {
            displacementFilename = argv[i];
            g_displacement = 1;
            g_normal = 1;
        } else if (occlusionFilename == NULL) {
            occlusionFilename = argv[i];
            g_occlusion = 1;
        } else if (specularFilename == NULL) {
            specularFilename = argv[i];
            g_specular = 1;
        }
    }

    OsdSetErrorCallback(callbackError);

    g_shaderSource = g_defaultShaderSource;
    reloadShaderFile();

    g_ptexColorFilename = colorFilename;
    if (g_ptexColorFilename == NULL) {
        usage(argv[0]);
        return 1;
    }

    if (not glfwInit()) {
        printf("Failed to initialize GLFW\n");
        return 1;
    }
    
    static const char windowTitle[] = "OpenSubdiv ptexViewer";
    
#define CORE_PROFILE
#ifdef CORE_PROFILE
    setGLCoreProfile();
#endif

#if GLFW_VERSION_MAJOR>=3

    if (fullscreen) {
    
        g_primary = glfwGetPrimaryMonitor();

        // apparently glfwGetPrimaryMonitor fails under linux : if no primary,
        // settle for the first one in the list    
        if (not g_primary) {
            int count=0;
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

    if (not (g_window=glfwCreateWindow(g_width, g_height, windowTitle, 
                                       fullscreen and g_primary ? g_primary : NULL, NULL))) {
        printf("Failed to open window.\n");
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(g_window);
    glfwSetKeyCallback(g_window, keyboard);
    glfwSetCursorPosCallback(g_window, motion);
    glfwSetMouseButtonCallback(g_window, mouse);
#else
    if (glfwOpenWindow(g_width, g_height, 8, 8, 8, 8, 24, 8,
                       fullscreen ? GLFW_FULLSCREEN : GLFW_WINDOW) == GL_FALSE) {
        printf("Failed to open window.\n");
        glfwTerminate();
        return 1;
    }
    glfwSetWindowTitle(windowTitle);
    glfwSetKeyCallback(keyboard);
    glfwSetMousePosCallback(motion);
    glfwSetMouseButtonCallback(mouse);
#endif

#if defined(OSD_USES_GLEW)
#ifdef CORE_PROFILE
    // this is the only way to initialize glew correctly under core profile context.
    glewExperimental = true;
#endif
    if (GLenum r = glewInit() != GLEW_OK) {
        printf("Failed to initialize glew. error = %d\n", r);
        exit(1);
    }
#ifdef CORE_PROFILE
    // clear GL errors which was generated during glewInit()
    glGetError();
#endif
#endif

    initGL();

#if GLFW_VERSION_MAJOR>=3
    glfwSetWindowSizeCallback(g_window, reshape);
    glfwSetWindowCloseCallback(g_window, windowClose);
    // as of GLFW 3.0.1 this callback is not implicit
    reshape();
#else
    glfwSetWindowSizeCallback(reshape);
    glfwSetWindowCloseCallback(windowClose);
#endif

    // activate feature adaptive tessellation if OSD supports it
    g_adaptive = OpenSubdiv::OsdGLDrawContext::SupportsAdaptiveTessellation();

#ifdef OPENSUBDIV_HAS_OPENCL
    // Initialize OpenCL
    if (initCL(&g_clContext, &g_clQueue) == false) {
        printf("Error in initializing OpenCL\n");
        exit(1);
    }
#endif

#if OPENSUBDIV_HAS_CUDA
    // Note: This function randomly crashes with linux 5.0-dev driver.
    // cudaGetDeviceProperties overrun stack..?
    cudaGLSetGLDevice( cutGetMaxGflopsDeviceId() );
#endif

    g_hud.Init(g_width, g_height);

    g_hud.AddRadioButton(0, "CPU (K)", true, 10, 10, callbackKernel, kCPU, 'k');
#ifdef OPENSUBDIV_HAS_OPENMP
    g_hud.AddRadioButton(0, "OPENMP", false, 10, 30, callbackKernel, kOPENMP, 'k');
#endif
#ifdef OPENSUBDIV_HAS_CUDA
    g_hud.AddRadioButton(0, "CUDA",   false, 10, 50, callbackKernel, kCUDA, 'k');
#endif
#ifdef OPENSUBDIV_HAS_OPENCL
    g_hud.AddRadioButton(0, "OPENCL", false, 10, 70, callbackKernel, kCL, 'k');
#endif
#ifdef OPENSUBDIV_HAS_GLSL_TRANSFORM_FEEDBACK
    g_hud.AddRadioButton(0, "GLSL Transform Feedback",   false, 10, 90, callbackKernel, kGLSL, 'k');
#endif
#ifdef OPENSUBDIV_HAS_GLSL_COMPUTE
    // Must also check at run time for OpenGL 4.3
    if (GLEW_VERSION_4_3) {
        g_hud.AddRadioButton(0, "GLSL Compute",   false, 10, 110, callbackKernel, kGLSLCompute, 'k');
    }
#endif

    g_hud.AddRadioButton(1, "Wire (W)",       (g_wire==0), 100, 10, callbackWireframe, 0, 'w');
    g_hud.AddRadioButton(1, "Shaded",         (g_wire==1), 100, 30, callbackWireframe, 1, 'w');
    g_hud.AddRadioButton(1, "Wire on Shaded", (g_wire==2), 100, 50, callbackWireframe, 2, 'w');

    g_hud.AddCheckBox("Color (C)", g_color, 250, 10, callbackCheckBox, HUD_CB_DISPLAY_COLOR, 'c');
    if (occlusionFilename != NULL) 
        g_hud.AddCheckBox("Ambient Occlusion (A)", g_occlusion,
                          250, 30, callbackCheckBox, HUD_CB_DISPLAY_OCCLUSION, 'a');
    if (displacementFilename != NULL) {
        g_hud.AddCheckBox("Displacement (D)", g_displacement,
                          250, 50, callbackCheckBox, HUD_CB_DISPLAY_DISPLACEMENT, 'd');
        g_hud.AddCheckBox("Normal (N)", g_normal,
                          250, 70, callbackCheckBox, HUD_CB_DISPLAY_NORMALMAP, 'n');
    }
    if (specularFilename != NULL)
        g_hud.AddCheckBox("Specular (S)", g_specular,
                          250, 90, callbackCheckBox, HUD_CB_DISPLAY_SPECULAR, 's');

    if (diffuseEnvironmentMap || specularEnvironmentMap) {
        g_hud.AddCheckBox("IBL (I)", g_ibl, 250, 110, callbackCheckBox, HUD_CB_IBL, 'i');
    }

    g_hud.AddCheckBox("Animate vertices (M)", g_moveScale != 0.0,
                      450, 10, callbackCheckBox, HUD_CB_ANIMATE_VERTICES, 'm');
    g_hud.AddCheckBox("Patch Color (P)",  g_displayPatchColor,
                      450, 30, callbackCheckBox, HUD_CB_DISPLAY_PATCH_COLOR, 'p');
    g_hud.AddCheckBox("Screen space LOD (V)",  g_screenSpaceTess,
                      450, 50, callbackCheckBox, HUD_CB_VIEW_LOD, 'v');
    g_hud.AddCheckBox("Fractional spacing (T)",  g_fractionalSpacing,
                      450, 70, callbackCheckBox, HUD_CB_FRACTIONAL_SPACING, 't');
    g_hud.AddCheckBox("Frustum Patch Culling (B)",  g_patchCull,
                      450, 90, callbackCheckBox, HUD_CB_PATCH_CULL, 'b');
    g_hud.AddCheckBox("Bloom (Y)", g_bloom,
                      450, 110, callbackCheckBox, HUD_CB_BLOOM, 'y');
    g_hud.AddCheckBox("Freeze (spc)", g_freeze,
                      450, 130, callbackCheckBox, HUD_CB_FREEZE, ' ');

    if (OpenSubdiv::OsdGLDrawContext::SupportsAdaptiveTessellation())
        g_hud.AddCheckBox("Adaptive (`)", g_adaptive, 10, 150, callbackCheckBox, HUD_CB_ADAPTIVE, '`');


    for (int i = 1; i < 8; ++i) {
        char level[16];
        sprintf(level, "Lv. %d", i);
        g_hud.AddRadioButton(3, level, i==2, 10, 150+i*20, callbackLevel, i, '0'+i);
    }

    g_hud.AddRadioButton(4, "CATMARK", true, -220, 10, callbackScheme, 0);
    g_hud.AddRadioButton(4, "BILINEAR", false, -220, 30, callbackScheme, 1);

    // create mesh from ptex metadata
    createOsdMesh(g_level, g_kernel);

    // load ptex files
    if (colorFilename)
        g_osdPTexImage = createPtex(colorFilename);
    if (displacementFilename)
        g_osdPTexDisplacement = createPtex(displacementFilename);
    if (occlusionFilename)
        g_osdPTexOcclusion = createPtex(occlusionFilename);
    if (specularFilename)
        g_osdPTexSpecular = createPtex(specularFilename);

    // load animation obj sequences (optional)
    if (not animobjs.empty()) {
        for (int i = 0; i < (int)animobjs.size(); ++i) {
            std::ifstream ifs(animobjs[i].c_str());
            if (ifs) {
                std::stringstream ss;
                ss << ifs.rdbuf();
                ifs.close();

                printf("Reading %s\r", animobjs[i].c_str());
                std::string str = ss.str();
                shape *shape = shape::parseShape(str.c_str());

                if (shape->verts.size() != g_positions.size()) {
                    printf("Error: vertex count doesn't match.\n");
                    goto error;
                }

                g_animPositions.push_back(shape->verts);
                delete shape;
            } else {
                printf("Error in reading %s\n", animobjs[i].c_str());
                goto error;
            }

        }
        printf("\n");
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
    createImageShader();

    fitFrame();

    while (g_running) {
        idle();
        display();

#if GLFW_VERSION_MAJOR>=3
        glfwPollEvents();
        glfwSwapBuffers(g_window);
#else
        glfwSwapBuffers();
#endif
        
        glFinish();
    }

error:
    uninitGL();
    glfwTerminate();
}
