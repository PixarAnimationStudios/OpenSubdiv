//
//     Copyright (C) Pixar. All rights reserved.
//
//     This license governs use of the accompanying software. If you
//     use the software, you accept this license. If you do not accept
//     the license, do not use the software.
//
//     1. Definitions
//     The terms "reproduce," "reproduction," "derivative works," and
//     "distribution" have the same meaning here as under U.S.
//     copyright law.  A "contribution" is the original software, or
//     any additions or changes to the software.
//     A "contributor" is any person or entity that distributes its
//     contribution under this license.
//     "Licensed patents" are a contributor's patent claims that read
//     directly on its contribution.
//
//     2. Grant of Rights
//     (A) Copyright Grant- Subject to the terms of this license,
//     including the license conditions and limitations in section 3,
//     each contributor grants you a non-exclusive, worldwide,
//     royalty-free copyright license to reproduce its contribution,
//     prepare derivative works of its contribution, and distribute
//     its contribution or any derivative works that you create.
//     (B) Patent Grant- Subject to the terms of this license,
//     including the license conditions and limitations in section 3,
//     each contributor grants you a non-exclusive, worldwide,
//     royalty-free license under its licensed patents to make, have
//     made, use, sell, offer for sale, import, and/or otherwise
//     dispose of its contribution in the software or derivative works
//     of the contribution in the software.
//
//     3. Conditions and Limitations
//     (A) No Trademark License- This license does not grant you
//     rights to use any contributor's name, logo, or trademarks.
//     (B) If you bring a patent claim against any contributor over
//     patents that you claim are infringed by the software, your
//     patent license from such contributor to the software ends
//     automatically.
//     (C) If you distribute any portion of the software, you must
//     retain all copyright, patent, trademark, and attribution
//     notices that are present in the software.
//     (D) If you distribute any portion of the software in source
//     code form, you may do so only under this license by including a
//     complete copy of this license with your distribution. If you
//     distribute any portion of the software in compiled or object
//     code form, you may only do so under a license that complies
//     with this license.
//     (E) The software is licensed "as-is." You bear the risk of
//     using it. The contributors give no express warranties,
//     guarantees or conditions. You may have additional consumer
//     rights under your local laws which this license cannot change.
//     To the extent permitted under your local laws, the contributors
//     exclude the implied warranties of merchantability, fitness for
//     a particular purpose and non-infringement.
//

#if defined(__APPLE__)
    #include <GLUT/glut.h>
#else
    #include <GL/glew.h>
    #include <GL/glut.h>
#endif

#include "../../regression/common/mutex.h"
//XXX
#define HBR_ADAPTIVE

#include <hbr/mesh.h>
#include <hbr/bilinear.h>
#include <hbr/catmark.h>
#include <hbr/face.h>

#include <osd/glPtexTexture.h>
#include <osd/glDrawContext.h>
#include <osd/glDrawRegistry.h>

#include <osd/cpuDispatcher.h>
#include <osd/cpuGLVertexBuffer.h>
#include <osd/cpuComputeContext.h>
#include <osd/cpuComputeController.h>

#ifdef OPENSUBDIV_HAS_OPENMP
    #include <osd/ompDispatcher.h>
    #include <osd/ompComputeController.h>
#endif

#ifdef OPENSUBDIV_HAS_OPENCL
    #include <osd/clDispatcher.h>
    #include <osd/clGLVertexBuffer.h>
    #include <osd/clComputeContext.h>
    #include <osd/clComputeController.h>

    #include "../common/clInit.h"

    cl_context g_clContext;
    cl_command_queue g_clQueue;
#endif

#ifdef OPENSUBDIV_HAS_CUDA
    #include <osd/cudaDispatcher.h>
    #include <osd/cudaGLVertexBuffer.h>
    #include <osd/cudaComputeContext.h>
    #include <osd/cudaComputeController.h>

    #include <cuda_runtime_api.h>
    #include <cuda_gl_interop.h>

    #include "../common/cudaInit.h"

    bool g_cudaInitialized = false;
#endif

#ifdef OPENSUBDIV_HAS_GLSL_TRANSFORM_FEEDBACK
    #include <osd/glslTransformFeedbackDispatcher.h>
    #include <osd/glslTransformFeedbackComputeContext.h>
    #include <osd/glslTransformFeedbackComputeController.h>
    #include <osd/glVertexBuffer.h>
#endif

#ifdef OPENSUBDIV_HAS_GLSL_COMPUTE
    #include <osd/glslDispatcher.h>
    #include <osd/glslComputeContext.h>
    #include <osd/glslComputeController.h>
    #include <osd/glVertexBuffer.h>
#endif

#include <osd/glMesh.h>
OpenSubdiv::OsdGLMeshInterface *g_mesh;

#include "Ptexture.h"
#include "PtexUtils.h"

#include "../common/stopwatch.h"
#include "../common/simple_math.h"
#include "../common/gl_hud.h"
#include "../common/hdr_reader.h"
#include "../../regression/common/shape_utils.h"

#include <vector>
#include <sstream>
#include <fstream>

static const char *g_defaultShaderSource =
#include "shader.inc"
;
static const char *g_skyShaderSource =
#include "skyshader.inc"
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
                   HUD_CB_PATCH_CULL,
                   HUD_CB_IBL };
    
//-----------------------------------------------------------------------------
int   g_frame = 0,
      g_repeatCount = 0;

// GLUT GUI variables
int   g_fullscreen=0,
      g_wire = 1,
      g_drawNormals = 0,
      g_mbutton[3] = {0, 0, 0},
      g_level = 2,
      g_tessLevel = 2,
      g_kernel = kCPU,
      g_scheme = 0,
      g_gutterWidth = 1;

float g_moveScale = 0.0f;
bool  g_adaptive = true,
      g_displayPatchColor = false,
      g_patchCull = true,
      g_screenSpaceTess = true,
      g_ibl = false;

GLuint g_transformUB = 0,
       g_transformBinding = 0,
       g_tessellationUB = 0,
       g_tessellationBinding = 0,
       g_lightingUB = 0,
       g_lightingBinding = 0;

// ptex switch
bool  g_color = true,
      g_occlusion = false,
      g_displacement = false,
      g_normal = false,
      g_specular = false;

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

GLhud g_hud;

// performance
float g_cpuTime = 0;
float g_gpuTime = 0;
#define NUM_FPS_TIME_SAMPLES 6
float g_fpsTimeSamples[NUM_FPS_TIME_SAMPLES] = {0,0,0,0,0,0};
int   g_currentFpsTimeSample = 0;
Stopwatch g_fpsTimer;

// geometry
std::vector<float> g_positions,
                   g_normals;

std::vector<std::vector<float> > g_animPositions;
std::vector<GLuint> g_animPositionBuffers;

GLuint g_primQuery = 0;

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

OpenSubdiv::OsdGLPtexTexture * g_osdPTexImage = 0;
OpenSubdiv::OsdGLPtexTexture * g_osdPTexDisplacement = 0;
OpenSubdiv::OsdGLPtexTexture * g_osdPTexOcclusion = 0;
OpenSubdiv::OsdGLPtexTexture * g_osdPTexSpecular = 0;
const char * g_ptexColorFilename;

/*static void
checkGLErrors(std::string const & where = "")
{
    GLuint err;
    while ((err = glGetError()) != GL_NO_ERROR) {
        std::cerr << "GL error: "
                  << (where.empty() ? "" : where + " ")
                  << err << "\n";
    }
}*/
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

    if (g_moveScale and g_adaptive and g_animPositionBuffers.size()) {
        // baked animation only works with adaptive for now
        // (since non-adaptive requires normals)
        int nkey = (int)g_animPositionBuffers.size();

#if 1
        glBindBuffer(GL_COPY_READ_BUFFER, g_animPositionBuffers[g_frame%nkey]);
        glBindBuffer(GL_COPY_WRITE_BUFFER, g_mesh->BindVertexBuffer());
        glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER,
                            0, 0, nverts * 3 * sizeof(float));
        glBindBuffer(GL_COPY_READ_BUFFER, 0);
        glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
        
#else
        g_mesh->UpdateVertexBuffer(&g_animPositions[g_frame%nkey][0], nverts);
#endif

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

        g_mesh->UpdateVertexBuffer(&vertex[0], nverts);
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
reshape(int width, int height) {

    g_width = width;
    g_height = height;

    g_hud.Rebuild(width, height);
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

    sources[0] = defString.c_str();
    sources[1] = source.version.c_str();
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
    glProgramUniform1i(program, data, samplerUnit + 0);
    glActiveTexture(GL_TEXTURE0 + samplerUnit + 0);
    glBindTexture(GL_TEXTURE_2D_ARRAY, osdPTex->GetTexelsTexture());

    glProgramUniform1i(program, packing, samplerUnit + 1);
    glActiveTexture(GL_TEXTURE0 + samplerUnit + 1);
    glBindTexture(GL_TEXTURE_BUFFER, osdPTex->GetLayoutTextureBuffer());

    glProgramUniform1i(program, pages, samplerUnit + 2);
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
        int ibl:1;
        unsigned int wire:2;
    };
    int value;

    bool operator < (const Effect &e) const {
        return value < e.value;
    }
};

typedef std::pair<OpenSubdiv::OsdPatchDescriptor, Effect> EffectDesc;

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

    bool quad = true;
    if (desc.first.type != OpenSubdiv::kNonPatch) {

        quad = false;
        sconfig->tessEvalShader.source = g_shaderSource + sconfig->tessEvalShader.source;
        sconfig->tessEvalShader.version = "#version 410\n";
        if (effect.displacement and (not effect.normal))
            sconfig->geometryShader.AddDefine("FLAT_NORMALS");
        if (effect.displacement)
            sconfig->tessEvalShader.AddDefine("USE_PTEX_DISPLACEMENT");
    } else {
        sconfig->vertexShader.source = g_shaderSource;
        sconfig->vertexShader.version = "#version 410\n";
        sconfig->vertexShader.AddDefine("VERTEX_SHADER");
        if (effect.displacement) {
            sconfig->geometryShader.AddDefine("USE_PTEX_DISPLACEMENT");
            sconfig->geometryShader.AddDefine("FLAT_NORMALS");
        }
    }
    assert(sconfig);

    sconfig->geometryShader.source = g_shaderSource;
    sconfig->geometryShader.version = "#version 410\n";
    sconfig->geometryShader.AddDefine("GEOMETRY_SHADER");

    sconfig->fragmentShader.source = g_shaderSource;
    sconfig->fragmentShader.version = "#version 410\n";
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
    glUniformBlockBinding(config->program,
        glGetUniformBlockIndex(config->program, "Tessellation"),
        g_tessellationBinding);

    g_lightingBinding = 2;
    glUniformBlockBinding(config->program,
        glGetUniformBlockIndex(config->program, "Lighting"),
        g_lightingBinding);

    GLint loc;
    if ((loc = glGetUniformLocation(config->program, "g_VertexBuffer")) != -1) {
        glProgramUniform1i(config->program, loc, 0); // GL_TEXTURE0
    }
    if ((loc = glGetUniformLocation(config->program, "g_ValenceBuffer")) != -1) {
        glProgramUniform1i(config->program, loc, 1); // GL_TEXTURE1
    }
    if ((loc = glGetUniformLocation(config->program, "g_QuadOffsetBuffer")) != -1) {
        glProgramUniform1i(config->program, loc, 2); // GL_TEXTURE2
    }
    if ((loc = glGetUniformLocation(config->program, "g_patchLevelBuffer")) != -1) {
        glProgramUniform1i(config->program, loc, 3); // GL_TEXTURE3
    }
    if ((loc = glGetUniformLocation(config->program, "g_ptexIndicesBuffer")) != -1) {
        glProgramUniform1i(config->program, loc, 4); // GL_TEXTURE4
    }

    return config;
}

EffectDrawRegistry effectRegistry;

EffectDrawRegistry::ConfigType *
getInstance(Effect effect, OpenSubdiv::OsdPatchDescriptor const & patchDesc) {

    EffectDesc desc;
    desc.first = patchDesc;
    desc.second = effect;

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

    if (kernel == kCPU) {
        g_mesh = new OpenSubdiv::OsdMesh<OpenSubdiv::OsdCpuGLVertexBuffer,
                                         OpenSubdiv::OsdCpuComputeController,
                                         OpenSubdiv::OsdGLDrawContext>(hmesh, numVertexElements, level, bits);
#ifdef OPENSUBDIV_HAS_OPENMP
    } else if (kernel == kOPENMP) {
        g_mesh = new OpenSubdiv::OsdMesh<OpenSubdiv::OsdCpuGLVertexBuffer,
                                         OpenSubdiv::OsdOmpComputeController,
                                         OpenSubdiv::OsdGLDrawContext>(hmesh, numVertexElements, level, bits);
#endif
#ifdef OPENSUBDIV_HAS_OPENCL
    } else if(kernel == kCL) {
        g_mesh = new OpenSubdiv::OsdMesh<OpenSubdiv::OsdCLGLVertexBuffer,
                                         OpenSubdiv::OsdCLComputeController,
                                         OpenSubdiv::OsdGLDrawContext>(hmesh, numVertexElements, level, bits, g_clContext, g_clQueue);
#endif
#ifdef OPENSUBDIV_HAS_CUDA
    } else if(kernel == kCUDA) {
        g_mesh = new OpenSubdiv::OsdMesh<OpenSubdiv::OsdCudaGLVertexBuffer,
                                         OpenSubdiv::OsdCudaComputeController,
                                         OpenSubdiv::OsdGLDrawContext>(hmesh, numVertexElements, level, bits);
#endif
#ifdef OPENSUBDIV_HAS_GLSL_TRANSFORM_FEEDBACK
    } else if(kernel == kGLSL) {
        g_mesh = new OpenSubdiv::OsdMesh<OpenSubdiv::OsdGLVertexBuffer,
                                         OpenSubdiv::OsdGLSLTransformFeedbackComputeController,
                                         OpenSubdiv::OsdGLDrawContext>(hmesh, numVertexElements, level, bits);
#endif
#ifdef OPENSUBDIV_HAS_GLSL_COMPUTE
    } else if(kernel == kGLSLCompute) {
        g_mesh = new OpenSubdiv::OsdMesh<OpenSubdiv::OsdGLVertexBuffer,
                                         OpenSubdiv::OsdGLSLComputeController,
                                         OpenSubdiv::OsdGLDrawContext>(hmesh, numVertexElements, level, bits);
#endif
    } else {
        printf("Unsupported kernel %s\n", getKernelName(kernel));
    }

    delete hmesh;
    
    if (glGetError() != GL_NO_ERROR){
        printf ("GLERROR\n");
    }

    updateGeom();
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
    if (g_specularEnvironmentMap)
        glProgramUniform1i(g_sky.program, environmentMap, 6);
    else
        glProgramUniform1i(g_sky.program, environmentMap, 5);

    g_sky.mvpMatrix = glGetUniformLocation(g_sky.program, "ModelViewProjectionMatrix");
}

//------------------------------------------------------------------------------
static GLuint
bindProgram(Effect effect, OpenSubdiv::OsdPatchArray const & patch)
{
    OpenSubdiv::OsdPatchDescriptor const & desc = patch.desc;

    EffectDrawRegistry::ConfigType *
        config = getInstance(effect, desc);

    GLuint program = config->program;

    glUseProgram(program);

    // Update and bind transform state
    struct Transform {
        float ModelViewMatrix[16];
        float ProjectionMatrix[16];
        float ModelViewProjectionMatrix[16];
        float ModelViewInverseMatrix[16];
    } transformData;
    glGetFloatv(GL_MODELVIEW_MATRIX, transformData.ModelViewMatrix);
    glGetFloatv(GL_PROJECTION_MATRIX, transformData.ProjectionMatrix);
    multMatrix(transformData.ModelViewProjectionMatrix,
               transformData.ModelViewMatrix,
               transformData.ProjectionMatrix);
    inverseMatrix(transformData.ModelViewInverseMatrix, transformData.ModelViewMatrix);

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
        int GregoryQuadOffsetBase;
        int LevelBase;
    } tessellationData;

    tessellationData.TessLevel = static_cast<float>(1 << g_tessLevel);
    tessellationData.GregoryQuadOffsetBase = patch.gregoryQuadOffsetBase;
    tessellationData.LevelBase = patch.levelBase;

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
            glProgramUniform1i(program, glGetUniformLocation(program, "diffuseEnvironmentMap"), 5);
            glActiveTexture(GL_TEXTURE5);
            glBindTexture(GL_TEXTURE_2D, g_diffuseEnvironmentMap);
            sampler++;
        }
        if (g_specularEnvironmentMap) {
            glProgramUniform1i(program, glGetUniformLocation(program, "specularEnvironmentMap"), 6);
            glActiveTexture(GL_TEXTURE6);
            glBindTexture(GL_TEXTURE_2D, g_specularEnvironmentMap);
            sampler++;
        }
        glActiveTexture(GL_TEXTURE0);
    }

    // checkGLErrors("bindProgram leave");

    return program;
}

//------------------------------------------------------------------------------
void
drawModel() {
    GLuint bVertex = g_mesh->BindVertexBuffer();
    glBindBuffer(GL_ARRAY_BUFFER, bVertex);

    if (g_adaptive) {
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    } else {
        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof (GLfloat) * 6, 0);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof (GLfloat) * 6, (float*)12);
    }

    OpenSubdiv::OsdPatchArrayVector const & patches = g_mesh->GetDrawContext()->patchArrays;
    GLenum primType = GL_LINES_ADJACENCY;

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_mesh->GetDrawContext()->patchIndexBuffer);

    // patch drawing
    for (int i=0; i<(int)patches.size(); ++i) {
        OpenSubdiv::OsdPatchArray const & patch = patches[i];

        OpenSubdiv::OsdPatchType patchType = patch.desc.type;
        int patchPattern = patch.desc.pattern;
        //int patchRotation = patch.desc.rotation;

        if (g_mesh->GetDrawContext()->IsAdaptive()) {

            primType = GL_PATCHES;
            glPatchParameteri(GL_PATCH_VERTICES, patch.patchSize);

            if (g_mesh->GetDrawContext()->vertexTextureBuffer) {
                glActiveTexture(GL_TEXTURE0);
                glBindTexture(GL_TEXTURE_BUFFER, 
                    g_mesh->GetDrawContext()->vertexTextureBuffer);
                glTexBuffer(GL_TEXTURE_BUFFER, GL_R32F, bVertex);
            }

            if (g_mesh->GetDrawContext()->vertexValenceTextureBuffer) {
                glActiveTexture(GL_TEXTURE1);
                glBindTexture(GL_TEXTURE_BUFFER, 
                    g_mesh->GetDrawContext()->vertexValenceTextureBuffer);
            }

            if (g_mesh->GetDrawContext()->quadOffsetTextureBuffer) {
                glActiveTexture(GL_TEXTURE2);
                glBindTexture(GL_TEXTURE_BUFFER, 
                    g_mesh->GetDrawContext()->quadOffsetTextureBuffer);
            }
            if (g_mesh->GetDrawContext()->patchLevelTextureBuffer) {
                glActiveTexture(GL_TEXTURE3);
                glBindTexture(GL_TEXTURE_BUFFER, 
                    g_mesh->GetDrawContext()->patchLevelTextureBuffer);
            }
            if (g_mesh->GetDrawContext()->ptexCoordinateTextureBuffer) {
                glActiveTexture(GL_TEXTURE4);
                glBindTexture(GL_TEXTURE_BUFFER,
                    g_mesh->GetDrawContext()->ptexCoordinateTextureBuffer);
            }
            glActiveTexture(GL_TEXTURE0);
        } else {
            if (g_mesh->GetDrawContext()->ptexCoordinateTextureBuffer) {
                glActiveTexture(GL_TEXTURE4);
                glBindTexture(GL_TEXTURE_BUFFER,
                    g_mesh->GetDrawContext()->ptexCoordinateTextureBuffer);
            }
            glActiveTexture(GL_TEXTURE0);
        }

        Effect effect;
        effect.value = 0;
        effect.color = g_color;
        effect.occlusion = g_occlusion;
        effect.displacement = g_displacement;
        effect.normal = g_normal;
        effect.specular = g_specular;
        effect.patchCull = g_patchCull;
        effect.screenSpaceTess = g_screenSpaceTess;
        effect.ibl = g_ibl;
        effect.wire = g_wire;

        GLuint program = bindProgram(effect, patch);

        GLint nonAdaptiveLevel = glGetUniformLocation(program, "nonAdaptiveLevel");
        if (nonAdaptiveLevel != -1) {
            glProgramUniform1i(program, nonAdaptiveLevel, g_level);
        }

        GLuint overrideColorEnable = glGetUniformLocation(program, "overrideColorEnable");
        GLuint overrideColor = glGetUniformLocation(program, "overrideColor");
        switch(patchType) {
        case OpenSubdiv::kRegular:
            glProgramUniform4f(program, overrideColor, 1.0f, 1.0f, 1.0f, 1);
            break;
        case OpenSubdiv::kBoundary:
            glProgramUniform4f(program, overrideColor, 0.8f, 0.0f, 0.0f, 1);
            break;
        case OpenSubdiv::kCorner:
            glProgramUniform4f(program, overrideColor, 0, 1.0, 0, 1);
            break;
        case OpenSubdiv::kGregory:
            glProgramUniform4f(program, overrideColor, 1.0f, 1.0f, 0.0f, 1);
            break;
        case OpenSubdiv::kBoundaryGregory:
            glProgramUniform4f(program, overrideColor, 1.0f, 0.5f, 0.0f, 1);
            break;
        case OpenSubdiv::kTransitionRegular:
            switch (patchPattern) {
            case 0:
                glProgramUniform4f(program, overrideColor, 0, 1.0f, 1.0f, 1);
                break;
            case 1:
                glProgramUniform4f(program, overrideColor, 0, 0.5f, 1.0f, 1);
                break;
            case 2:
                glProgramUniform4f(program, overrideColor, 0, 0.5f, 0.5f, 1);
                break;
            case 3:
                glProgramUniform4f(program, overrideColor, 0.5f, 0, 1.0f, 1);
                break;
            case 4:
                glProgramUniform4f(program, overrideColor, 1.0f, 0.5f, 1.0f, 1);
                break;
            }
            break;
        case OpenSubdiv::kTransitionBoundary:
            glProgramUniform4f(program, overrideColor, 0, 0, 0.5f, 1);
            break;
        case OpenSubdiv::kTransitionCorner:
            glProgramUniform4f(program, overrideColor, 0, 0, 0.5f, 1);
            break;
        default:
            glProgramUniform4f(program, overrideColor, 0.4f, 0.4f, 0.8f, 1);
            break;
        }

        if (g_displayPatchColor or g_wire == 2) {
            glProgramUniform1i(program, overrideColorEnable, 1);
        } else {
            glProgramUniform1i(program, overrideColorEnable, 0);
        }

        if (g_wire == 0) {
            glDisable(GL_CULL_FACE);
        }
        glDrawElements(primType,
                       patch.numIndices, GL_UNSIGNED_INT,
                       (void *)(patch.firstIndex * sizeof(unsigned int)));
        if (g_wire == 0) {
            glEnable(GL_CULL_FACE);
        }
    }
    if (g_adaptive) {
        glDisableVertexAttribArray(0);
    } else {
        glDisableVertexAttribArray(0);
        glDisableVertexAttribArray(1);
    }

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void
drawSky() {
    
    glUseProgram(g_sky.program);

    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glRotatef(g_rotate[1], 1, 0, 0);
    glRotatef(g_rotate[0], 0, 1, 0);
    glScalef(g_size, g_size, g_size);

    float modelView[16], projection[16], mvp[16];
    glGetFloatv(GL_MODELVIEW_MATRIX, modelView);
    glGetFloatv(GL_PROJECTION_MATRIX, projection);
    multMatrix(mvp, modelView, projection);
    glProgramUniformMatrix4fv(g_sky.program, g_sky.mvpMatrix, 1, GL_FALSE, mvp);

    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, g_sky.vertexBuffer);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof (GLfloat) * 5, 0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof (GLfloat) * 5, (void*)(sizeof(GLfloat)*3));

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_sky.elementBuffer);
    glDrawElements(GL_TRIANGLES, g_sky.numIndices, GL_UNSIGNED_INT, 0);

    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);

    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
 }

void
display() {

    Stopwatch s;
    s.Start();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // primitive counting
    glBeginQuery(GL_PRIMITIVES_GENERATED, g_primQuery);

    glViewport(0, 0, g_width, g_height);

    double aspect = g_width/(double)g_height;
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0, aspect, g_size*0.001f, g_size+g_dolly);

    if (g_ibl) {
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        drawSky();
    }

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(-g_pan[0], -g_pan[1], -g_dolly);
    glRotatef(g_rotate[1], 1, 0, 0);
    glRotatef(g_rotate[0], 0, 1, 0);
    glTranslatef(-g_center[0], -g_center[1], -g_center[2]);

    drawModel();

    glUseProgram(0);

    glEndQuery(GL_PRIMITIVES_GENERATED);

    s.Stop();
    float drawCpuTime = float(s.GetElapsed() * 1000.0f);
    s.Start();
    glFinish();
    s.Stop();
    float drawGpuTime = float(s.GetElapsed() * 1000.0f);

    GLuint numPrimsGenerated = 0;
    glGetQueryObjectuiv(g_primQuery, GL_QUERY_RESULT, &numPrimsGenerated);

    if (g_hud.IsVisible()) {
        g_fpsTimer.Stop();
        double fps = 1.0/g_fpsTimer.GetElapsed();
        g_fpsTimer.Start();

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
    }

    g_hud.Flush();

    glutSwapBuffers();
    glFinish();

    // checkGLErrors("draw end");
}

//------------------------------------------------------------------------------
void mouse(int button, int state, int x, int y) {

    if (button == 0 && state == 1 && g_hud.MouseClick(x, y)) return;

    g_prev_x = float(x);
    g_prev_y = float(y);
    g_mbutton[button] = !state;
}

//------------------------------------------------------------------------------
void motion(int x, int y) {

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
void quit() {

    if (g_osdPTexImage) delete g_osdPTexImage;
    if (g_osdPTexDisplacement) delete g_osdPTexDisplacement;
    if (g_osdPTexOcclusion) delete g_osdPTexOcclusion;
    if (g_osdPTexSpecular) delete g_osdPTexSpecular;

    glDeleteQueries(1, &g_primQuery);

    if(g_mesh)
        delete g_mesh;

#ifdef OPENSUBDIV_HAS_CUDA
    cudaDeviceReset();
#endif

#ifdef OPENSUBDIV_HAS_OPENCL
    uninitCL(g_clContext, g_clQueue);
#endif

    if (g_animPositionBuffers.size())
        glDeleteBuffers((int)g_animPositionBuffers.size(), &g_animPositionBuffers[0]);
    if (g_diffuseEnvironmentMap) glDeleteTextures(1, &g_diffuseEnvironmentMap);
    if (g_specularEnvironmentMap) glDeleteTextures(1, &g_specularEnvironmentMap);

    if (g_sky.program) glDeleteProgram(g_sky.program);
    if (g_sky.vertexBuffer) glDeleteBuffers(1, &g_sky.vertexBuffer);
    if (g_sky.elementBuffer) glDeleteBuffers(1, &g_sky.elementBuffer);

    exit(0);
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
        g_adaptive = checked;
        rebuild = true;
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
        break;
    case HUD_CB_DISPLAY_PATCH_COLOR:
        g_displayPatchColor = checked;
        break;
    case HUD_CB_VIEW_LOD:
        g_screenSpaceTess = checked;
        break;
    case HUD_CB_PATCH_CULL:
        g_patchCull = checked;
        break;
    case HUD_CB_IBL:
        g_ibl = checked;
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
static void toggleFullScreen() {

    static int x,y,w,h;
    
    g_fullscreen = !g_fullscreen;
    
    if (g_fullscreen) {
        x = glutGet((GLenum)GLUT_WINDOW_X);
        y = glutGet((GLenum)GLUT_WINDOW_Y);
        w = glutGet((GLenum)GLUT_WINDOW_WIDTH);
        h = glutGet((GLenum)GLUT_WINDOW_HEIGHT);
        
        glutFullScreen( );
        
        reshape( glutGet(GLUT_SCREEN_WIDTH),
                 glutGet(GLUT_SCREEN_HEIGHT) );
    } else {
        glutReshapeWindow(w, h);
        glutPositionWindow(x,y);
        reshape( w, h );
    }
}

//------------------------------------------------------------------------------
void
keyboard(unsigned char key, int x, int y) {

    if (g_hud.KeyDown(key)) return;

    switch (key) {
        case 'q': quit();
        case 'e': g_drawNormals = (g_drawNormals+1)%2; break;
        case 'f': fitFrame(); break;
        case '\t': toggleFullScreen(); break;
        case 'g': g_gutterWidth = (g_gutterWidth+1)%8; createOsdMesh(g_level, g_kernel); break;
        case 'r': reloadShaderFile(); createOsdMesh(g_level, g_kernel); break;
        case '+':
        case '=': g_tessLevel++; break;
        case '-': g_tessLevel = std::max(1, g_tessLevel-1); break;
    }
}

//------------------------------------------------------------------------------
void
idle() {

    g_frame++;
    updateGeom();
    glutPostRedisplay();

    if(g_repeatCount != 0 && g_frame >= g_repeatCount)
        quit();
}

//------------------------------------------------------------------------------
void
initGL() {

    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glEnable(GL_LIGHT0);
    glColor3f(1, 1, 1);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glEnable(GL_CULL_FACE);

    GLfloat color[4] = {1, 1, 1, 1};
    GLfloat position[4] = {5, 5, 10, 1};
    GLfloat ambient[4] = {0.9f, 0.9f, 0.9f, 1.0f};
    GLfloat diffuse[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    GLfloat shininess = 25.0;

    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, color);
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, color);
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, color);
    glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, &shininess);
    glLightfv(GL_LIGHT0, GL_POSITION, position);
    glLightfv(GL_LIGHT0, GL_AMBIENT, ambient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse);
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

}
//------------------------------------------------------------------------------
int main(int argc, char ** argv) {

    std::vector<std::string> animobjs;
    const char *diffuseEnvironmentMap = NULL, *specularEnvironmentMap = NULL;
    const char *colorFilename = NULL, *displacementFilename = NULL,
        *occlusionFilename = NULL, *specularFilename = NULL;

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

    g_shaderSource = g_defaultShaderSource;
    reloadShaderFile();

    g_ptexColorFilename = colorFilename;
    if (g_ptexColorFilename == NULL) {
        usage(argv[0]);
        return 1;
    }

    glutInit(&argc, argv);

    glutInitDisplayMode(GLUT_RGBA |GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(1024, 1024);
    glutCreateWindow("OpenSubdiv ptexViewer");

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

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutMouseFunc(mouse);
    glutKeyboardFunc(keyboard);
    glutMotionFunc(motion);
    glewInit();
    initGL();
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
    g_hud.AddCheckBox("Frustum Patch Culling (B)",  g_patchCull,
                      450, 70, callbackCheckBox, HUD_CB_PATCH_CULL, 'b');

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
        g_animPositionBuffers.resize(animobjs.size());
        glGenBuffers((int)animobjs.size(), &g_animPositionBuffers[0]);

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
                    quit();
                }

                g_animPositions.push_back(shape->verts);

                glBindBuffer(GL_ARRAY_BUFFER, g_animPositionBuffers[i]);
                glBufferData(GL_ARRAY_BUFFER, shape->verts.size()*sizeof(float), &shape->verts[0], GL_STATIC_DRAW);

                delete shape;
            } else {
                printf("Error in reading %s\n", animobjs[i].c_str());
                quit();
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
            glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP, GL_TRUE);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, info.width, info.height,
                         0, GL_RGBA, GL_FLOAT, image);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glBindTexture(GL_TEXTURE_2D, 0);
            free(image);
        }
    }
    if (diffuseEnvironmentMap || specularEnvironmentMap) {
        createSky();
    }

    fitFrame();

    glGenQueries(1, &g_primQuery);
    glutIdleFunc(idle);
    glutMainLoop();

    quit();
}
