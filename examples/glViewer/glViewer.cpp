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

#include <GLFW/glfw3.h>
GLFWwindow* g_window=0;
GLFWmonitor* g_primary=0;

#include <far/error.h>
#include <osd/vertex.h>
#include <osd/glDrawContext.h>
#include <osd/glDrawRegistry.h>

#include <osd/cpuGLVertexBuffer.h>
#include <osd/cpuComputeContext.h>
#include <osd/cpuComputeController.h>
OpenSubdiv::Osd::CpuComputeController *g_cpuComputeController = NULL;

#ifdef OPENSUBDIV_HAS_OPENMP
    #include <osd/ompComputeController.h>
    OpenSubdiv::Osd::OmpComputeController *g_ompComputeController = NULL;
#endif

#ifdef OPENSUBDIV_HAS_TBB
    #include <osd/tbbComputeController.h>
    OpenSubdiv::Osd::TbbComputeController *g_tbbComputeController = NULL;
#endif

#ifdef OPENSUBDIV_HAS_OPENCL
    #include <osd/clGLVertexBuffer.h>
    #include <osd/clComputeContext.h>
    #include <osd/clComputeController.h>

    #include "../common/clInit.h"

    cl_context g_clContext;
    cl_command_queue g_clQueue;
    OpenSubdiv::Osd::CLComputeController *g_clComputeController = NULL;
#endif

#ifdef OPENSUBDIV_HAS_CUDA
    #include <osd/cudaGLVertexBuffer.h>
    #include <osd/cudaComputeContext.h>
    #include <osd/cudaComputeController.h>

    #include <cuda_runtime_api.h>
    #include <cuda_gl_interop.h>

    #include "../common/cudaInit.h"

    bool g_cudaInitialized = false;
    OpenSubdiv::Osd::CudaComputeController *g_cudaComputeController = NULL;
#endif

#ifdef OPENSUBDIV_HAS_GLSL_TRANSFORM_FEEDBACK
    #include <osd/glslTransformFeedbackComputeContext.h>
    #include <osd/glslTransformFeedbackComputeController.h>
    #include <osd/glVertexBuffer.h>
    OpenSubdiv::Osd::GLSLTransformFeedbackComputeController *g_glslTransformFeedbackComputeController = NULL;
#endif

#ifdef OPENSUBDIV_HAS_GLSL_COMPUTE
    #include <osd/glslComputeContext.h>
    #include <osd/glslComputeController.h>
    #include <osd/glVertexBuffer.h>
    OpenSubdiv::Osd::GLSLComputeController *g_glslComputeController = NULL;
#endif

#include <osd/glMesh.h>
OpenSubdiv::Osd::GLMeshInterface *g_mesh;

#include <common/vtr_utils.h>
#include "../common/stopwatch.h"
#include "../common/simple_math.h"
#include "../common/gl_hud.h"
#include "../common/objAnim.h"
#include "../common/patchColors.h"

static const char *shaderSource =
#if defined(GL_ARB_tessellation_shader) || defined(GL_VERSION_4_0)
    #include "shader.gen.h"
#else
    #include "shader_gl3.gen.h"
#endif
;

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

enum DisplayStyle { kWire = 0,
                    kShaded,
                    kWireShaded,
                    kVaryingColor,
                    kInterleavedVaryingColor,
                    kFaceVaryingColor };

enum HudCheckBox { kHUD_CB_DISPLAY_CAGE_EDGES,
                   kHUD_CB_DISPLAY_CAGE_VERTS,
                   kHUD_CB_ANIMATE_VERTICES,
                   kHUD_CB_DISPLAY_PATCH_COLOR,
                   kHUD_CB_VIEW_LOD,
                   kHUD_CB_FRACTIONAL_SPACING,
                   kHUD_CB_PATCH_CULL,
                   kHUD_CB_FREEZE,
                   kHUD_CB_DISPLAY_PATCH_COUNTS };

int g_currentShape = 0;

ObjAnim const * g_objAnim = 0;

bool g_axis=true;

int   g_frame = 0,
      g_repeatCount = 0;
float g_animTime = 0;

// GUI variables
int   g_fullscreen = 0,
      g_freeze = 0,
      g_displayStyle = kWireShaded,
      g_adaptive = 1,
      g_singleCreasePatch = 1,
      g_drawCageEdges = 1,
      g_drawCageVertices = 0,
      g_mbutton[3] = {0, 0, 0},
      g_running = 1;

int   g_displayPatchColor = 1,
      g_screenSpaceTess = 1,
      g_fractionalSpacing = 1,
      g_patchCull = 0,
      g_displayPatchCounts = 0;

float g_rotate[2] = {0, 0},
      g_dolly = 5,
      g_pan[2] = {0, 0},
      g_center[3] = {0, 0, 0},
      g_size = 0;

int   g_prev_x = 0,
      g_prev_y = 0;

int   g_width = 1024,
      g_height = 1024;

GLhud g_hud;

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

GLuint g_queries[2] = {0, 0};

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
} g_transformData;

GLuint g_vao = 0;
GLuint g_cageEdgeVAO = 0,
       g_cageEdgeVBO = 0,
       g_cageVertexVAO = 0,
       g_cageVertexVBO = 0;

std::vector<int> g_coarseEdges;
std::vector<float> g_coarseEdgeSharpness;
std::vector<float> g_coarseVertexSharpness;

struct Program
{
    GLuint program;
    GLuint uniformModelViewProjectionMatrix;
    GLuint attrPosition;
    GLuint attrColor;
} g_defaultProgram;

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
static GLuint
compileShader(GLenum shaderType, const char *source)
{
    GLuint shader = glCreateShader(shaderType);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);
    checkGLErrors("compileShader");
    return shader;
}

static bool
linkDefaultProgram() {

#if defined(GL_ARB_tessellation_shader) || defined(GL_VERSION_4_0)
    #define GLSL_VERSION_DEFINE "#version 400\n"
#else
    #define GLSL_VERSION_DEFINE "#version 150\n"
#endif

    static const char *vsSrc =
        GLSL_VERSION_DEFINE
        "in vec3 position;\n"
        "in vec3 color;\n"
        "out vec4 fragColor;\n"
        "uniform mat4 ModelViewProjectionMatrix;\n"
        "void main() {\n"
        "  fragColor = vec4(color, 1);\n"
        "  gl_Position = ModelViewProjectionMatrix * "
        "                  vec4(position, 1);\n"
        "}\n";

    static const char *fsSrc =
        GLSL_VERSION_DEFINE
        "in vec4 fragColor;\n"
        "out vec4 color;\n"
        "void main() {\n"
        "  color = fragColor;\n"
        "}\n";

    GLuint program = glCreateProgram();
    GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vsSrc);
    GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, fsSrc);

    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);

    glLinkProgram(program);

    GLint status;
    glGetProgramiv(program, GL_LINK_STATUS, &status);
    if (status == GL_FALSE) {
        GLint infoLogLength;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &infoLogLength);
        char *infoLog = new char[infoLogLength];
        glGetProgramInfoLog(program, infoLogLength, NULL, infoLog);
        printf("%s\n", infoLog);
        delete[] infoLog;
        exit(1);
    }

    g_defaultProgram.program = program;
    g_defaultProgram.uniformModelViewProjectionMatrix =
        glGetUniformLocation(program, "ModelViewProjectionMatrix");
    g_defaultProgram.attrPosition = glGetAttribLocation(program, "position");
    g_defaultProgram.attrColor = glGetAttribLocation(program, "color");

    return true;
}

//------------------------------------------------------------------------------

#include "init_shapes.h"

//------------------------------------------------------------------------------
static void
updateGeom() {

    std::vector<float> vertex, varying;

    int nverts=0, stride=g_displayStyle == kInterleavedVaryingColor ? 7 : 3;

    if (g_objAnim and g_currentShape==0) {

        nverts = g_objAnim->GetShape()->GetNumVertices(),

        vertex.resize(nverts*stride);

        if (g_displayStyle == kVaryingColor) {
            varying.resize(nverts*4);
        }

        g_objAnim->InterpolatePositions(g_animTime, &vertex[0], stride);

        if (g_drawCageEdges or g_drawCageVertices) {
            g_positions.resize(nverts*3);
            for (int i=0; i<nverts; ++i) {
                int ofs = i * stride;
                g_positions[i*3+0] = vertex[ofs+0];
                g_positions[i*3+1] = vertex[ofs+1];
                g_positions[i*3+2] = vertex[ofs+2];
            }
        }

        if (g_displayStyle == kVaryingColor or
            g_displayStyle == kInterleavedVaryingColor) {

            const float *p = &g_objAnim->GetShape()->verts[0];
            for (int i = 0; i < nverts; ++i) {
                if (g_displayStyle == kInterleavedVaryingColor) {
                    int ofs = i * stride;
                    vertex[ofs + 0] = p[1];
                    vertex[ofs + 1] = p[2];
                    vertex[ofs + 2] = p[0];
                    vertex[ofs + 3] = 0.0f;
                    p += 3;
                }
                if (g_displayStyle == kVaryingColor) {
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

        if (g_displayStyle == kVaryingColor) {
            varying.reserve(nverts*4);
        }

        const float *p = &g_orgPositions[0];

        float r = sin(g_frame*0.001f) * g_moveScale;
        for (int i = 0; i < nverts; ++i) {
            //float move = 0.05f*cosf(p[0]*20+g_frame*0.01f);
            float ct = cos(p[2] * r);
            float st = sin(p[2] * r);
            g_positions[i*3+0] = p[0]*ct + p[1]*st;
            g_positions[i*3+1] = -p[0]*st + p[1]*ct;
            g_positions[i*3+2] = p[2];

            p += 3;
        }

        p = &g_orgPositions[0];
        const float *pp = &g_positions[0];
        for (int i = 0; i < nverts; ++i) {
            vertex.push_back(pp[0]);
            vertex.push_back(pp[1]);
            vertex.push_back(pp[2]);
            if (g_displayStyle == kInterleavedVaryingColor) {
                vertex.push_back(p[1]);
                vertex.push_back(p[2]);
                vertex.push_back(p[0]);
                vertex.push_back(1.0f);
                p += 3;
            }
            if (g_displayStyle == kVaryingColor) {
                varying.push_back(p[2]);
                varying.push_back(p[1]);
                varying.push_back(p[0]);
                varying.push_back(1);
                p += 3;
            }
            pp += 3;
        }
    }

    g_mesh->UpdateVertexBuffer(&vertex[0], 0, nverts);

    if (g_displayStyle == kVaryingColor)
        g_mesh->UpdateVaryingBuffer(&varying[0], 0, nverts);

    Stopwatch s;
    s.Start();

    if (g_displayStyle == kInterleavedVaryingColor) {
        OpenSubdiv::Osd::VertexBufferDescriptor vertexDesc(0, 3, 7);
        OpenSubdiv::Osd::VertexBufferDescriptor varyingDesc(3, 4, 7);
        g_mesh->Refine(&vertexDesc, &varyingDesc, true);
    } else {
        g_mesh->Refine();
    }

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
createOsdMesh(ShapeDesc const & shapeDesc, int level, int kernel, Scheme scheme=kCatmark) {

    typedef OpenSubdiv::Far::ConstIndexArray IndexArray;

    bool doAnim = g_objAnim and g_currentShape==0;

    Shape const * shape = 0;
    if (doAnim) {
        shape = g_objAnim->GetShape();
    } else {
        shape = Shape::parseObj(shapeDesc.data.c_str(), shapeDesc.scheme);
    }

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
    bool doAdaptive = (g_adaptive!=0 and g_scheme==kCatmark),
         interleaveVarying = g_displayStyle == kInterleavedVaryingColor,
         doSingleCreasePatch = (g_singleCreasePatch!=0 and g_scheme==kCatmark);

    OpenSubdiv::Osd::MeshBitset bits;
    bits.set(OpenSubdiv::Osd::MeshAdaptive, doAdaptive);
    bits.set(OpenSubdiv::Osd::MeshUseSingleCreasePatch, doSingleCreasePatch);
    bits.set(OpenSubdiv::Osd::MeshInterleaveVarying, interleaveVarying);
    bits.set(OpenSubdiv::Osd::MeshFVarData, g_displayStyle == kFaceVaryingColor);

    int numVertexElements = 3;
    int numVaryingElements =
        (g_displayStyle == kVaryingColor or interleaveVarying) ? 4 : 0;

    if (kernel == kCPU) {
        if (not g_cpuComputeController) {
            g_cpuComputeController = new OpenSubdiv::Osd::CpuComputeController();
        }
        g_mesh = new OpenSubdiv::Osd::Mesh<OpenSubdiv::Osd::CpuGLVertexBuffer,
                                         OpenSubdiv::Osd::CpuComputeController,
                                         OpenSubdiv::Osd::GLDrawContext>(
                                                g_cpuComputeController,
                                                refiner,
                                                numVertexElements,
                                                numVaryingElements,
                                                level, bits);
#ifdef OPENSUBDIV_HAS_OPENMP
    } else if (kernel == kOPENMP) {
        if (not g_ompComputeController) {
            g_ompComputeController = new OpenSubdiv::Osd::OmpComputeController();
        }
        g_mesh = new OpenSubdiv::Osd::Mesh<OpenSubdiv::Osd::CpuGLVertexBuffer,
                                         OpenSubdiv::Osd::OmpComputeController,
                                         OpenSubdiv::Osd::GLDrawContext>(
                                                g_ompComputeController,
                                                refiner,
                                                numVertexElements,
                                                numVaryingElements,
                                                level, bits);
#endif
#ifdef OPENSUBDIV_HAS_TBB
    } else if (kernel == kTBB) {
        if (not g_tbbComputeController) {
            g_tbbComputeController = new OpenSubdiv::Osd::TbbComputeController();
        }
        g_mesh = new OpenSubdiv::Osd::Mesh<OpenSubdiv::Osd::CpuGLVertexBuffer,
                                         OpenSubdiv::Osd::TbbComputeController,
                                         OpenSubdiv::Osd::GLDrawContext>(
                                                g_tbbComputeController,
                                                refiner,
                                                numVertexElements,
                                                numVaryingElements,
                                                level, bits);
#endif
#ifdef OPENSUBDIV_HAS_OPENCL
    } else if(kernel == kCL) {
        if (not g_clComputeController) {
            g_clComputeController = new OpenSubdiv::Osd::CLComputeController(g_clContext, g_clQueue);
        }
        g_mesh = new OpenSubdiv::Osd::Mesh<OpenSubdiv::Osd::CLGLVertexBuffer,
                                         OpenSubdiv::Osd::CLComputeController,
                                         OpenSubdiv::Osd::GLDrawContext>(
                                                g_clComputeController,
                                                refiner,
                                                numVertexElements,
                                                numVaryingElements,
                                                level, bits, g_clContext, g_clQueue);
#endif
#ifdef OPENSUBDIV_HAS_CUDA
    } else if(kernel == kCUDA) {
        if (not g_cudaComputeController) {
            g_cudaComputeController = new OpenSubdiv::Osd::CudaComputeController();
        }
        g_mesh = new OpenSubdiv::Osd::Mesh<OpenSubdiv::Osd::CudaGLVertexBuffer,
                                         OpenSubdiv::Osd::CudaComputeController,
                                         OpenSubdiv::Osd::GLDrawContext>(
                                         g_cudaComputeController,
                                         refiner,
                                         numVertexElements,
                                         numVaryingElements,
                                         level, bits);
#endif
#ifdef OPENSUBDIV_HAS_GLSL_TRANSFORM_FEEDBACK
    } else if(kernel == kGLSL) {
        if (not g_glslTransformFeedbackComputeController) {
            g_glslTransformFeedbackComputeController = new OpenSubdiv::Osd::GLSLTransformFeedbackComputeController();
        }
        g_mesh = new OpenSubdiv::Osd::Mesh<OpenSubdiv::Osd::GLVertexBuffer,
                                         OpenSubdiv::Osd::GLSLTransformFeedbackComputeController,
                                         OpenSubdiv::Osd::GLDrawContext>(
                                                g_glslTransformFeedbackComputeController,
                                                refiner,
                                                numVertexElements,
                                                numVaryingElements,
                                                level, bits);
#endif
#ifdef OPENSUBDIV_HAS_GLSL_COMPUTE
    } else if(kernel == kGLSLCompute) {
        if (not g_glslComputeController) {
            g_glslComputeController = new OpenSubdiv::Osd::GLSLComputeController();
        }
        g_mesh = new OpenSubdiv::Osd::Mesh<OpenSubdiv::Osd::GLVertexBuffer,
                                         OpenSubdiv::Osd::GLSLComputeController,
                                         OpenSubdiv::Osd::GLDrawContext>(
                                         g_glslComputeController,
                                         refiner,
                                         numVertexElements,
                                         numVaryingElements,
                                         level, bits);
#endif
    } else {
        printf("Unsupported kernel %s\n", getKernelName(kernel));
    }

    if (g_displayStyle == kFaceVaryingColor and shape->HasUV()) {

        std::vector<float> fvarData;

        InterpolateFVarData(*refiner, *shape, fvarData);

        g_mesh->SetFVarDataChannel(shape->GetFVarWidth(), fvarData);
    }

    if (not doAnim) {
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

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_mesh->GetDrawContext()->GetPatchIndexBuffer());
    glBindBuffer(GL_ARRAY_BUFFER, g_mesh->BindVertexBuffer());

    glEnableVertexAttribArray(0);


    if (g_displayStyle == kVaryingColor) {
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof (GLfloat) * 3, 0);
        glBindBuffer(GL_ARRAY_BUFFER, g_mesh->BindVaryingBuffer());
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof (GLfloat) * 4, 0);
    } else if (g_displayStyle == kInterleavedVaryingColor) {
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
static inline void
setSharpnessColor(float s, float *r, float *g, float *b) {
    //  0.0       2.0       4.0
    // green --- yellow --- red
    *r = std::min(1.0f, s * 0.5f);
    *g = std::min(1.0f, 2.0f - s*0.5f);
    *b = 0;
}

static void
drawCageEdges() {

    glUseProgram(g_defaultProgram.program);
    glUniformMatrix4fv(g_defaultProgram.uniformModelViewProjectionMatrix,
                       1, GL_FALSE, g_transformData.ModelViewProjectionMatrix);

    std::vector<float> vbo;
    vbo.reserve(g_coarseEdges.size() * 6);
    float r, g, b;
    for (int i = 0; i < (int)g_coarseEdges.size(); i+=2) {
        setSharpnessColor(g_coarseEdgeSharpness[i/2], &r, &g, &b);
        for (int j = 0; j < 2; ++j) {
            vbo.push_back(g_positions[g_coarseEdges[i+j]*3]);
            vbo.push_back(g_positions[g_coarseEdges[i+j]*3+1]);
            vbo.push_back(g_positions[g_coarseEdges[i+j]*3+2]);
            vbo.push_back(r);
            vbo.push_back(g);
            vbo.push_back(b);
        }
    }

    glBindVertexArray(g_cageEdgeVAO);

    glBindBuffer(GL_ARRAY_BUFFER, g_cageEdgeVBO);
    glBufferData(GL_ARRAY_BUFFER, (int)vbo.size() * sizeof(float), &vbo[0],
                 GL_STATIC_DRAW);

    glEnableVertexAttribArray(g_defaultProgram.attrPosition);
    glEnableVertexAttribArray(g_defaultProgram.attrColor);
    glVertexAttribPointer(g_defaultProgram.attrPosition,
                          3, GL_FLOAT, GL_FALSE, sizeof (GLfloat) * 6, 0);
    glVertexAttribPointer(g_defaultProgram.attrColor,
                          3, GL_FLOAT, GL_FALSE, sizeof (GLfloat) * 6, (void*)12);

    glDrawArrays(GL_LINES, 0, (int)g_coarseEdges.size());

    glBindVertexArray(0);
    glUseProgram(0);
}

static void
drawCageVertices() {

    glUseProgram(g_defaultProgram.program);
    glUniformMatrix4fv(g_defaultProgram.uniformModelViewProjectionMatrix,
                       1, GL_FALSE, g_transformData.ModelViewProjectionMatrix);

    int numPoints = (int)g_positions.size()/3;
    std::vector<float> vbo;
    vbo.reserve(numPoints*6);
    float r, g, b;
    for (int i = 0; i < numPoints; ++i) {
        setSharpnessColor(g_coarseVertexSharpness[i], &r, &g, &b);
        vbo.push_back(g_positions[i*3+0]);
        vbo.push_back(g_positions[i*3+1]);
        vbo.push_back(g_positions[i*3+2]);
        vbo.push_back(r);
        vbo.push_back(g);
        vbo.push_back(b);
    }

    glBindVertexArray(g_cageVertexVAO);

    glBindBuffer(GL_ARRAY_BUFFER, g_cageVertexVBO);
    glBufferData(GL_ARRAY_BUFFER, (int)vbo.size() * sizeof(float), &vbo[0],
                 GL_STATIC_DRAW);

    glEnableVertexAttribArray(g_defaultProgram.attrPosition);
    glEnableVertexAttribArray(g_defaultProgram.attrColor);
    glVertexAttribPointer(g_defaultProgram.attrPosition,
                          3, GL_FLOAT, GL_FALSE, sizeof (GLfloat) * 6, 0);
    glVertexAttribPointer(g_defaultProgram.attrColor,
                          3, GL_FLOAT, GL_FALSE, sizeof (GLfloat) * 6, (void*)12);

    glPointSize(10.0f);
    glDrawArrays(GL_POINTS, 0, numPoints);
    glPointSize(1.0f);

    glBindVertexArray(0);
    glUseProgram(0);
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

class EffectDrawRegistry : public OpenSubdiv::Osd::GLDrawRegistry<EffectDesc> {

protected:
    virtual ConfigType *
    _CreateDrawConfig(DescType const & desc, SourceConfigType const * sconfig);

    virtual SourceConfigType *
    _CreateDrawSourceConfig(DescType const & desc);
};

EffectDrawRegistry::SourceConfigType *
EffectDrawRegistry::_CreateDrawSourceConfig(DescType const & desc)
{
    typedef OpenSubdiv::Far::PatchDescriptor Descriptor;

    Effect effect = desc.second;

    SourceConfigType * sconfig =
        BaseRegistry::_CreateDrawSourceConfig(desc.first);

    assert(sconfig);

#if defined(GL_ARB_tessellation_shader) || defined(GL_VERSION_4_0)
    const char *glslVersion = "#version 400\n";
#else
    const char *glslVersion = "#version 330\n";
#endif

    if (desc.first.GetType() == Descriptor::QUADS or
        desc.first.GetType() == Descriptor::TRIANGLES) {
        sconfig->vertexShader.source = shaderSource;
        sconfig->vertexShader.version = glslVersion;
        sconfig->vertexShader.AddDefine("VERTEX_SHADER");
    } else {
        sconfig->geometryShader.AddDefine("SMOOTH_NORMALS");
    }

    sconfig->geometryShader.source = shaderSource;
    sconfig->geometryShader.version = glslVersion;
    sconfig->geometryShader.AddDefine("GEOMETRY_SHADER");

    sconfig->fragmentShader.source = shaderSource;
    sconfig->fragmentShader.version = glslVersion;
    sconfig->fragmentShader.AddDefine("FRAGMENT_SHADER");

    if (desc.first.GetType() == Descriptor::QUADS) {
        // uniform catmark, bilinear
        sconfig->geometryShader.AddDefine("PRIM_QUAD");
        sconfig->fragmentShader.AddDefine("PRIM_QUAD");
        sconfig->commonShader.AddDefine("UNIFORM_SUBDIVISION");
    } else if (desc.first.GetType() == Descriptor::TRIANGLES) {
        // uniform loop
        sconfig->geometryShader.AddDefine("PRIM_TRI");
        sconfig->fragmentShader.AddDefine("PRIM_TRI");
        sconfig->commonShader.AddDefine("LOOP");
        sconfig->commonShader.AddDefine("UNIFORM_SUBDIVISION");
    } else {
        // adaptive
        sconfig->vertexShader.source = shaderSource + sconfig->vertexShader.source;
        sconfig->tessControlShader.source = shaderSource + sconfig->tessControlShader.source;
        sconfig->tessEvalShader.source = shaderSource + sconfig->tessEvalShader.source;

        sconfig->geometryShader.AddDefine("PRIM_TRI");
        sconfig->fragmentShader.AddDefine("PRIM_TRI");
    }

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
    case kWire:
        sconfig->commonShader.AddDefine("GEOMETRY_OUT_WIRE");
        break;
    case kWireShaded:
        sconfig->commonShader.AddDefine("GEOMETRY_OUT_LINE");
        break;
    case kShaded:
        sconfig->commonShader.AddDefine("GEOMETRY_OUT_FILL");
        break;
    case kVaryingColor:
        sconfig->commonShader.AddDefine("VARYING_COLOR");
        sconfig->commonShader.AddDefine("GEOMETRY_OUT_FILL");
        break;
    case kInterleavedVaryingColor:
        sconfig->commonShader.AddDefine("VARYING_COLOR");
        sconfig->commonShader.AddDefine("GEOMETRY_OUT_FILL");
        break;
    case kFaceVaryingColor:
        sconfig->commonShader.AddDefine("OSD_FVAR_WIDTH", "2");
        sconfig->commonShader.AddDefine("FACEVARYING_COLOR");
        sconfig->commonShader.AddDefine("GEOMETRY_OUT_FILL");
        break;
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

    GLuint uboIndex;

    // XXXdyu can use layout(binding=) with GLSL 4.20 and beyond
    g_transformBinding = 0;
    uboIndex = glGetUniformBlockIndex(config->program, "Transform");
    if (uboIndex != GL_INVALID_INDEX)
        glUniformBlockBinding(config->program, uboIndex, g_transformBinding);

    g_tessellationBinding = 1;
    uboIndex = glGetUniformBlockIndex(config->program, "Tessellation");
    if (uboIndex != GL_INVALID_INDEX)
        glUniformBlockBinding(config->program, uboIndex, g_tessellationBinding);

    g_lightingBinding = 2;
    uboIndex = glGetUniformBlockIndex(config->program, "Lighting");
    if (uboIndex != GL_INVALID_INDEX)
        glUniformBlockBinding(config->program, uboIndex, g_lightingBinding);

    GLint loc;
#if not defined(GL_ARB_separate_shader_objects) || defined(GL_VERSION_4_1)
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
    if ((loc = glGetUniformLocation(config->program, "OsdFVarDataBuffer")) != -1) {
        glUniform1i(loc, 4); // GL_TEXTURE4
    }
#else
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
    if ((loc = glGetUniformLocation(config->program, "OsdFVarDataBuffer")) != -1) {
        glProgramUniform1i(config->program, loc, 4); // GL_TEXTURE4
    }
#endif

    return config;
}

EffectDrawRegistry effectRegistry;

static Effect
GetEffect()
{
    return Effect(g_displayStyle, g_screenSpaceTess, g_fractionalSpacing, g_patchCull);
}

//------------------------------------------------------------------------------
static GLuint
bindProgram(Effect effect, OpenSubdiv::Osd::DrawContext::PatchArray const & patch)
{
    EffectDesc effectDesc(patch.GetDescriptor(), effect);
    EffectDrawRegistry::ConfigType *
        config = effectRegistry.GetDrawConfig(effectDesc);

    GLuint program = config->program;

    glUseProgram(program);

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

    if (g_mesh->GetDrawContext()->GetVertexTextureBuffer()) {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_BUFFER,
            g_mesh->GetDrawContext()->GetVertexTextureBuffer());
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
    if (g_mesh->GetDrawContext()->GetPatchParamTextureBuffer()) {
        glActiveTexture(GL_TEXTURE3);
        glBindTexture(GL_TEXTURE_BUFFER,
            g_mesh->GetDrawContext()->GetPatchParamTextureBuffer());
    }
    if (g_mesh->GetDrawContext()->GetFvarDataTextureBuffer()) {
        glActiveTexture(GL_TEXTURE4);
        glBindTexture(GL_TEXTURE_BUFFER,
            g_mesh->GetDrawContext()->GetFvarDataTextureBuffer());
    }

    glActiveTexture(GL_TEXTURE0);

    return program;
}

//------------------------------------------------------------------------------
static void
display() {

    SSAOGLFrameBuffer * fb = (SSAOGLFrameBuffer *)g_hud.GetFrameBuffer();
    fb->Bind();

    Stopwatch s;
    s.Start();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glViewport(0, 0, g_width, g_height);

    // prepare view matrix
    double aspect = g_width/(double)g_height;
    identity(g_transformData.ModelViewMatrix);
    translate(g_transformData.ModelViewMatrix, -g_pan[0], -g_pan[1], -g_dolly);
    rotate(g_transformData.ModelViewMatrix, g_rotate[1], 1, 0, 0);
    rotate(g_transformData.ModelViewMatrix, g_rotate[0], 0, 1, 0);
    rotate(g_transformData.ModelViewMatrix, -90, 1, 0, 0);
    translate(g_transformData.ModelViewMatrix,
              -g_center[0], -g_center[1], -g_center[2]);
    perspective(g_transformData.ProjectionMatrix,
                45.0f, (float)aspect, fb->IsActive() ? 1.0f : 0.0001f, 500.0f);
    multMatrix(g_transformData.ModelViewProjectionMatrix,
               g_transformData.ModelViewMatrix,
               g_transformData.ProjectionMatrix);

    // make sure that the vertex buffer is interoped back as a GL resources.
    g_mesh->BindVertexBuffer();

    if (g_displayStyle == kVaryingColor)
        g_mesh->BindVaryingBuffer();

    glEnable(GL_DEPTH_TEST);

    glBindVertexArray(g_vao);

    OpenSubdiv::Osd::DrawContext::PatchArrayVector const & patches =
        g_mesh->GetDrawContext()->GetPatchArrays();

    // patch drawing
    int patchCount[12][6][4]; // [Type][Pattern][Rotation] (see far/patchTables.h)
    int numTotalPatches = 0;
    int numDrawCalls = 0;
    memset(patchCount, 0, sizeof(patchCount));

    // primitive counting
    glBeginQuery(GL_PRIMITIVES_GENERATED, g_queries[0]);
#if defined(GL_VERSION_3_3)
    glBeginQuery(GL_TIME_ELAPSED, g_queries[1]);
#endif

    for (int i=0; i<(int)patches.size(); ++i) {
        OpenSubdiv::Osd::DrawContext::PatchArray const & patch = patches[i];

        OpenSubdiv::Osd::DrawContext::PatchDescriptor desc = patch.GetDescriptor();
        OpenSubdiv::Far::PatchDescriptor::Type patchType = desc.GetType();
        int patchPattern = desc.GetPattern();
        int patchRotation = desc.GetRotation();
        int subPatch = desc.GetSubPatch();

        if (subPatch == 0) {
            patchCount[patchType][patchPattern][patchRotation] += patch.GetNumPatches();
        }
        numTotalPatches += patch.GetNumPatches();

        GLenum primType;

        switch(patchType) {
        case OpenSubdiv::Far::PatchDescriptor::QUADS:
            primType = GL_LINES_ADJACENCY;
            break;
        case OpenSubdiv::Far::PatchDescriptor::TRIANGLES:
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
        GLuint program = bindProgram(GetEffect(), patch);

        GLuint diffuseColor = glGetUniformLocation(program, "diffuseColor");

        if (g_displayPatchColor and primType == GL_PATCHES) {
            float const * color = getAdaptivePatchColor( desc );
            glProgramUniform4f(program, diffuseColor, color[0], color[1], color[2], color[3]);
        } else {
            glProgramUniform4f(program, diffuseColor, 0.4f, 0.4f, 0.8f, 1);
        }

        GLuint uniformGregoryQuadOffsetBase =
          glGetUniformLocation(program, "GregoryQuadOffsetBase");
        GLuint uniformPrimitiveIdBase =
          glGetUniformLocation(program, "PrimitiveIdBase");

        glProgramUniform1i(program, uniformGregoryQuadOffsetBase,
                           patch.GetQuadOffsetIndex());
        glProgramUniform1i(program, uniformPrimitiveIdBase,
                           patch.GetPatchIndex());
#else
        GLuint program = bindProgram(GetEffect(), patch);
        GLint uniformPrimitiveIdBase =
          glGetUniformLocation(program, "PrimitiveIdBase");
        if (uniformPrimitiveIdBase != -1)
            glUniform1i(uniformPrimitiveIdBase, patch.GetPatchIndex());
#endif

        if (g_displayStyle == kWire) {
            glDisable(GL_CULL_FACE);
        }

        glDrawElements(primType, patch.GetNumIndices(), GL_UNSIGNED_INT,
                       (void *)(patch.GetVertIndex() * sizeof(unsigned int)));
        ++numDrawCalls;
        if (g_displayStyle == kWire) {
            glEnable(GL_CULL_FACE);
        }
    }

    s.Stop();
    float drawCpuTime = float(s.GetElapsed() * 1000.0f);

    glEndQuery(GL_PRIMITIVES_GENERATED);
#if defined(GL_VERSION_3_3)
    glEndQuery(GL_TIME_ELAPSED);
#endif

    glBindVertexArray(0);

    glUseProgram(0);

    if (g_drawCageEdges)
        drawCageEdges();

    if (g_drawCageVertices)
        drawCageVertices();

    fb->ApplyImageShader();

    GLuint numPrimsGenerated = 0;
    GLuint timeElapsed = 0;
    glGetQueryObjectuiv(g_queries[0], GL_QUERY_RESULT, &numPrimsGenerated);
#if defined(GL_VERSION_3_3)
    glGetQueryObjectuiv(g_queries[1], GL_QUERY_RESULT, &timeElapsed);
#endif

    float drawGpuTime = timeElapsed / 1000.0f / 1000.0f;

    g_fpsTimer.Stop();
    float elapsed = (float)g_fpsTimer.GetElapsed();
    if (not g_freeze) {
        g_animTime += elapsed;
    }
    g_fpsTimer.Start();

    if (g_hud.IsVisible()) {

        typedef OpenSubdiv::Far::PatchDescriptor Descriptor;

        double fps = 1.0/elapsed;

        if (g_displayPatchCounts) {
            int x = -280;
            int y = -480;
            g_hud.DrawString(x, y, "NonPatch         : %d",
                             patchCount[Descriptor::QUADS][0][0]); y += 20;
            g_hud.DrawString(x, y, "Regular          : %d",
                             patchCount[Descriptor::REGULAR][0][0]); y+= 20;
            g_hud.DrawString(x, y, "Boundary         : %d",
                             patchCount[Descriptor::BOUNDARY][0][0]); y+= 20;
            g_hud.DrawString(x, y, "Corner           : %d",
                             patchCount[Descriptor::CORNER][0][0]); y+= 20;
            g_hud.DrawString(x, y, "Single Crease    : %d",
                             patchCount[Descriptor::SINGLE_CREASE][0][0]); y+= 20;
            g_hud.DrawString(x, y, "Gregory          : %d",
                             patchCount[Descriptor::GREGORY][0][0]); y+= 20;
            g_hud.DrawString(x, y, "Boundary Gregory : %d",
                             patchCount[Descriptor::GREGORY_BOUNDARY][0][0]); y+= 20;
            g_hud.DrawString(x, y, "Trans. Regular   : %d %d %d %d %d",
                             patchCount[Descriptor::REGULAR][Descriptor::PATTERN0][0],
                             patchCount[Descriptor::REGULAR][Descriptor::PATTERN1][0],
                             patchCount[Descriptor::REGULAR][Descriptor::PATTERN2][0],
                             patchCount[Descriptor::REGULAR][Descriptor::PATTERN3][0],
                             patchCount[Descriptor::REGULAR][Descriptor::PATTERN4][0]); y+= 20;
            for (int i=0; i < 5; i++) {
                g_hud.DrawString(x, y, "Trans. Boundary%d : %d %d %d %d", i,
                                 patchCount[Descriptor::BOUNDARY][i+1][0],
                                 patchCount[Descriptor::BOUNDARY][i+1][1],
                                 patchCount[Descriptor::BOUNDARY][i+1][2],
                                 patchCount[Descriptor::BOUNDARY][i+1][3]); y+= 20;
            }
            for (int i=0; i < 5; i++) {
                g_hud.DrawString(x, y, "Trans. Corner%d  : %d %d %d %d", i,
                                 patchCount[Descriptor::CORNER][i+1][0],
                                 patchCount[Descriptor::CORNER][i+1][1],
                                 patchCount[Descriptor::CORNER][i+1][2],
                                 patchCount[Descriptor::CORNER][i+1][3]); y+= 20;
            }
            for (int i=0; i < 5; i++) {
                g_hud.DrawString(x, y, "Trans. Single Crease%d : %d %d %d %d", i,
                                 patchCount[Descriptor::SINGLE_CREASE][i+1][0],
                                 patchCount[Descriptor::SINGLE_CREASE][i+1][1],
                                 patchCount[Descriptor::SINGLE_CREASE][i+1][2],
                                 patchCount[Descriptor::SINGLE_CREASE][i+1][3]); y+= 20;
            }
        }

        int y = -220;
        g_hud.DrawString(10, y, "Tess level : %d", g_tessLevel); y+= 20;
        g_hud.DrawString(10, y, "Patches    : %d", numTotalPatches); y+= 20;
        g_hud.DrawString(10, y, "Draw calls : %d", numDrawCalls); y+= 20;
        g_hud.DrawString(10, y, "Primitives : %d", numPrimsGenerated); y+= 20;
        g_hud.DrawString(10, y, "Vertices   : %d", g_mesh->GetNumVertices()); y+= 20;
        g_hud.DrawString(10, y, "Scheme     : %s", g_scheme==kBilinear ? "BILINEAR" : (g_scheme == kLoop ? "LOOP" : "CATMARK")); y+= 20;
        g_hud.DrawString(10, y, "GPU Kernel : %.3f ms", g_gpuTime); y+= 20;
        g_hud.DrawString(10, y,  "CPU Kernel : %.3f ms", g_cpuTime); y+= 20;
        g_hud.DrawString(10, y,  "GPU Draw   : %.3f ms", drawGpuTime); y+= 20;
        g_hud.DrawString(10, y,  "CPU Draw   : %.3f ms", drawCpuTime); y+= 20;
        g_hud.DrawString(10, y,  "FPS        : %3.1f", fps); y+= 20;

        g_hud.Flush();
    }

    glFinish();

    //checkGLErrors("display leave");
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

    glDeleteBuffers(1, &g_cageVertexVBO);
    glDeleteBuffers(1, &g_cageEdgeVBO);
    glDeleteVertexArrays(1, &g_vao);
    glDeleteVertexArrays(1, &g_cageVertexVAO);
    glDeleteVertexArrays(1, &g_cageEdgeVAO);

    if (g_mesh)
        delete g_mesh;

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
#ifdef OPENSUBDIV_HAS_GLSL_TRANSFORM_FEEDBACK
    delete g_glslTransformFeedbackComputeController;
#endif
#ifdef OPENSUBDIV_HAS_GLSL_COMPUTE
    delete g_glslComputeController;
#endif
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
        case 'X': g_hud.GetFrameBuffer()->Screenshot(); break;
    }
}

//------------------------------------------------------------------------------
static void
rebuildOsdMesh() {
    createOsdMesh( g_defaultShapes[ g_currentShape ], g_level, g_kernel, g_defaultShapes[ g_currentShape ].scheme );
}

static void
callbackDisplayStyle(int b) {
    if (g_displayStyle == kVaryingColor or b == kVaryingColor or
        g_displayStyle == kInterleavedVaryingColor or b == kInterleavedVaryingColor or
        g_displayStyle == kFaceVaryingColor or b == kFaceVaryingColor) {
        // need to rebuild for varying reconstruct
        g_displayStyle = b;
        rebuildOsdMesh();
        return;
    }
    g_displayStyle = b;
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
        cudaGLSetGLDevice( cutGetMaxGflopsDeviceId() );
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
    if (m < 0)
        m = 0;

    if (m >= (int)g_defaultShapes.size())
        m = (int)g_defaultShapes.size() - 1;

    g_currentShape = m;
    rebuildOsdMesh();
}

static void
callbackAdaptive(bool checked, int /* a */) {
    if (OpenSubdiv::Osd::GLDrawContext::SupportsAdaptiveTessellation()) {
        g_adaptive = checked;
        rebuildOsdMesh();
    }
}

static void
callbackSingleCreasePatch(bool checked, int /* a */) {
    if (OpenSubdiv::Osd::GLDrawContext::SupportsAdaptiveTessellation()) {
        g_singleCreasePatch = checked;
        rebuildOsdMesh();
    }
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

    g_hud.SetFrameBuffer(new SSAOGLFrameBuffer);

    g_hud.AddCheckBox("Cage Edges (H)", g_drawCageEdges != 0,
                      10, 10, callbackCheckBox, kHUD_CB_DISPLAY_CAGE_EDGES, 'h');
    g_hud.AddCheckBox("Cage Verts (J)", g_drawCageVertices != 0,
                      10, 30, callbackCheckBox, kHUD_CB_DISPLAY_CAGE_VERTS, 'j');
    g_hud.AddCheckBox("Animate vertices (M)", g_moveScale != 0,
                      10, 50, callbackCheckBox, kHUD_CB_ANIMATE_VERTICES, 'm');
    g_hud.AddCheckBox("Patch Color (P)", g_displayPatchColor != 0,
                      10, 70, callbackCheckBox, kHUD_CB_DISPLAY_PATCH_COLOR, 'p');
    g_hud.AddCheckBox("Screen space LOD (V)",  g_screenSpaceTess != 0,
                      10, 90, callbackCheckBox, kHUD_CB_VIEW_LOD, 'v');
    g_hud.AddCheckBox("Fractional spacing (T)",  g_fractionalSpacing != 0,
                      10, 110, callbackCheckBox, kHUD_CB_FRACTIONAL_SPACING, 't');
    g_hud.AddCheckBox("Frustum Patch Culling (B)",  g_patchCull != 0,
                      10, 130, callbackCheckBox, kHUD_CB_PATCH_CULL, 'b');
    g_hud.AddCheckBox("Freeze (spc)", g_freeze != 0,
                      10, 150, callbackCheckBox, kHUD_CB_FREEZE, ' ');

    int shading_pulldown = g_hud.AddPullDown("Shading (W)", 200, 10, 250, callbackDisplayStyle, 'w');
    g_hud.AddPullDownButton(shading_pulldown, "Wire", kWire, g_displayStyle==kWire);
    g_hud.AddPullDownButton(shading_pulldown, "Shaded", kShaded, g_displayStyle==kShaded);
    g_hud.AddPullDownButton(shading_pulldown, "Wire+Shaded", kWireShaded, g_displayStyle==kWireShaded);
    g_hud.AddPullDownButton(shading_pulldown, "Varying Color", kVaryingColor, g_displayStyle==kVaryingColor);
    g_hud.AddPullDownButton(shading_pulldown, "Varying Color (Interleaved)", kInterleavedVaryingColor, g_displayStyle==kInterleavedVaryingColor);
    g_hud.AddPullDownButton(shading_pulldown, "FaceVarying Color", kFaceVaryingColor, g_displayStyle==kFaceVaryingColor);

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
    if (HAS_CL_VERSION_1_1()) {
        g_hud.AddPullDownButton(compute_pulldown, "OpenCL", kCL);
    }
#endif
#ifdef OPENSUBDIV_HAS_GLSL_TRANSFORM_FEEDBACK
    g_hud.AddPullDownButton(compute_pulldown, "GLSL TransformFeedback", kGLSL);
#endif
#ifdef OPENSUBDIV_HAS_GLSL_COMPUTE
    // Must also check at run time for OpenGL 4.3
    if (GLEW_VERSION_4_3) {
        g_hud.AddPullDownButton(compute_pulldown, "GLSL Compute", kGLSLCompute);
    }
#endif
    if (OpenSubdiv::Osd::GLDrawContext::SupportsAdaptiveTessellation()) {
        g_hud.AddCheckBox("Adaptive (`)", g_adaptive!=0, 10, 190, callbackAdaptive, 0, '`');
        g_hud.AddCheckBox("Single Crease Patch (S)", g_singleCreasePatch!=0, 10, 210, callbackSingleCreasePatch, 0, 's');
    }

    for (int i = 1; i < 11; ++i) {
        char level[16];
        sprintf(level, "Lv. %d", i);
        g_hud.AddRadioButton(3, level, i==2, 10, 210+i*20, callbackLevel, i, '0'+(i%10));
    }

    int shapes_pulldown = g_hud.AddPullDown("Shape (N)", -300, 10, 300, callbackModel, 'n');
    for (int i = 0; i < (int)g_defaultShapes.size(); ++i) {
        g_hud.AddPullDownButton(shapes_pulldown, g_defaultShapes[i].name.c_str(),i);
    }

    g_hud.AddCheckBox("Show patch counts", g_displayPatchCounts!=0, -280, -20, callbackCheckBox, kHUD_CB_DISPLAY_PATCH_COUNTS);

    g_hud.Rebuild(windowWidth, windowHeight, frameBufferWidth, frameBufferHeight);
}

//------------------------------------------------------------------------------
static void
initGL() {
    glClearColor(0.1f, 0.1f, 0.1f, 0.0f);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glCullFace(GL_BACK);
    glEnable(GL_CULL_FACE);

    glGenQueries(2, g_queries);

    glGenVertexArrays(1, &g_vao);
    glGenVertexArrays(1, &g_cageVertexVAO);
    glGenVertexArrays(1, &g_cageEdgeVAO);
    glGenBuffers(1, &g_cageVertexVBO);
    glGenBuffers(1, &g_cageEdgeVBO);
}

//------------------------------------------------------------------------------
static void
idle() {

    if (not g_freeze) {
        g_frame++;
        updateGeom();
    }

    if (g_repeatCount != 0 and g_frame >= g_repeatCount)
        g_running = 0;
}

//------------------------------------------------------------------------------
static void
callbackErrorOsd(OpenSubdiv::Far::ErrorType err, const char *message) {
    printf("Error: %d\n", err);
    printf("%s", message);
}

//------------------------------------------------------------------------------
static void
callbackErrorGLFW(int error, const char* description) {
    fprintf(stderr, "GLFW Error (%d) : %s\n", error, description);
}
//------------------------------------------------------------------------------
static void
setGLCoreProfile() {
    #define glfwOpenWindowHint glfwWindowHint
    #define GLFW_OPENGL_VERSION_MAJOR GLFW_CONTEXT_VERSION_MAJOR
    #define GLFW_OPENGL_VERSION_MINOR GLFW_CONTEXT_VERSION_MINOR

    glfwOpenWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#if not defined(__APPLE__)
    glfwOpenWindowHint(GLFW_OPENGL_VERSION_MAJOR, 4);
#ifdef OPENSUBDIV_HAS_GLSL_COMPUTE
    glfwOpenWindowHint(GLFW_OPENGL_VERSION_MINOR, 3);
#else
    glfwOpenWindowHint(GLFW_OPENGL_VERSION_MINOR, 2);
#endif

#else
    glfwOpenWindowHint(GLFW_OPENGL_VERSION_MAJOR, 3);
    glfwOpenWindowHint(GLFW_OPENGL_VERSION_MINOR, 2);
#endif
    glfwOpenWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
}

//------------------------------------------------------------------------------
int main(int argc, char ** argv) {

    bool fullscreen = false;
    std::string str;
    std::vector<char const *> animobjs;

    for (int i = 1; i < argc; ++i) {
        if (strstr(argv[i], ".obj")) {
            animobjs.push_back(argv[i]);
        } else if (!strcmp(argv[i], "-axis")) {
            g_axis = false;
        } else if (!strcmp(argv[i], "-d")) {
            g_level = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-c")) {
            g_repeatCount = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-f")) {
            fullscreen = true;
        } else {
            std::ifstream ifs(argv[1]);
            if (ifs) {
                std::stringstream ss;
                ss << ifs.rdbuf();
                ifs.close();
                str = ss.str();
                g_defaultShapes.push_back(ShapeDesc(argv[1], str.c_str(), kCatmark));
            }
        }
    }

    if (not animobjs.empty()) {

        g_defaultShapes.push_back(ShapeDesc(animobjs[0], "", kCatmark));

        g_objAnim = ObjAnim::Create(animobjs, g_axis);
    }

    initShapes();

    g_fpsTimer.Start();

    OpenSubdiv::Far::SetErrorCallback(callbackErrorOsd);

    glfwSetErrorCallback(callbackErrorGLFW);
    if (not glfwInit()) {
        printf("Failed to initialize GLFW\n");
        return 1;
    }

    static const char windowTitle[] = "OpenSubdiv glViewer " OPENSUBDIV_VERSION_STRING;

#define CORE_PROFILE
#ifdef CORE_PROFILE
    setGLCoreProfile();
#endif

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

    // accommocate high DPI displays (e.g. mac retina displays)
    glfwGetFramebufferSize(g_window, &g_width, &g_height);
    glfwSetFramebufferSizeCallback(g_window, reshape);

    glfwSetKeyCallback(g_window, keyboard);
    glfwSetCursorPosCallback(g_window, motion);
    glfwSetMouseButtonCallback(g_window, mouse);
    glfwSetWindowCloseCallback(g_window, windowClose);

#if defined(OSD_USES_GLEW)
#ifdef CORE_PROFILE
    // this is the only way to initialize glew correctly under core profile context.
    glewExperimental = true;
#endif
    if (GLenum r = glewInit() != GLEW_OK) {
        printf("Failed to initialize glew. Error = %s\n", glewGetErrorString(r));
        exit(1);
    }
#ifdef CORE_PROFILE
    // clear GL errors which was generated during glewInit()
    glGetError();
#endif
#endif

    // activate feature adaptive tessellation if OSD supports it
    g_adaptive = OpenSubdiv::Osd::GLDrawContext::SupportsAdaptiveTessellation();

    initGL();
    linkDefaultProgram();

    glfwSwapInterval(0);

    initHUD();
    rebuildOsdMesh();

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
