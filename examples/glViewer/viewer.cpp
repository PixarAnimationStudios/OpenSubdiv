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

#include <osd/error.h>
#include <osd/vertex.h>
#include <osd/glDrawContext.h>
#include <osd/glDrawRegistry.h>

#include <osd/cpuGLVertexBuffer.h>
#include <osd/cpuComputeContext.h>
#include <osd/cpuComputeController.h>
OpenSubdiv::OsdCpuComputeController *g_cpuComputeController = NULL;

#ifdef OPENSUBDIV_HAS_OPENMP
    #include <osd/ompComputeController.h>
    OpenSubdiv::OsdOmpComputeController *g_ompComputeController = NULL;
#endif

#ifdef OPENSUBDIV_HAS_TBB
    #include <osd/tbbComputeController.h>
    OpenSubdiv::OsdTbbComputeController *g_tbbComputeController = NULL;
#endif

#ifdef OPENSUBDIV_HAS_GCD
    #include <osd/gcdComputeController.h>
    OpenSubdiv::OsdGcdComputeController *g_gcdComputeController = NULL;
#endif

#ifdef OPENSUBDIV_HAS_OPENCL
    #include <osd/clGLVertexBuffer.h>
    #include <osd/clComputeContext.h>
    #include <osd/clComputeController.h>

    #include "../common/clInit.h"

    cl_context g_clContext;
    cl_command_queue g_clQueue;
    OpenSubdiv::OsdCLComputeController *g_clComputeController = NULL;
#endif

#ifdef OPENSUBDIV_HAS_CUDA
    #include <osd/cudaGLVertexBuffer.h>
    #include <osd/cudaComputeContext.h>
    #include <osd/cudaComputeController.h>

    #include <cuda_runtime_api.h>
    #include <cuda_gl_interop.h>

    #include "../common/cudaInit.h"

    bool g_cudaInitialized = false;
    OpenSubdiv::OsdCudaComputeController *g_cudaComputeController = NULL;
#endif

#ifdef OPENSUBDIV_HAS_GLSL_TRANSFORM_FEEDBACK
    #include <osd/glslTransformFeedbackComputeContext.h>
    #include <osd/glslTransformFeedbackComputeController.h>
    #include <osd/glVertexBuffer.h>
    OpenSubdiv::OsdGLSLTransformFeedbackComputeController *g_glslTransformFeedbackComputeController = NULL;
#endif

#ifdef OPENSUBDIV_HAS_GLSL_COMPUTE
    #include <osd/glslComputeContext.h>
    #include <osd/glslComputeController.h>
    #include <osd/glVertexBuffer.h>
    OpenSubdiv::OsdGLSLComputeController *g_glslComputeController = NULL;
#endif

#include <osd/glMesh.h>
OpenSubdiv::OsdGLMeshInterface *g_mesh;

#include <common/shape_utils.h>
#include "../common/stopwatch.h"
#include "../common/simple_math.h"
#include "../common/gl_hud.h"
#include "../common/patchColors.h"

static const char *shaderSource =
#if defined(GL_ARB_tessellation_shader) || defined(GL_VERSION_4_0)
    #include "shader.inc"
#else
    #include "shader_gl3.inc"
#endif
;

#include <cfloat>
#include <vector>
#include <fstream>
#include <sstream>

typedef OpenSubdiv::HbrMesh<OpenSubdiv::OsdVertex>     OsdHbrMesh;
typedef OpenSubdiv::HbrVertex<OpenSubdiv::OsdVertex>   OsdHbrVertex;
typedef OpenSubdiv::HbrFace<OpenSubdiv::OsdVertex>     OsdHbrFace;
typedef OpenSubdiv::HbrHalfedge<OpenSubdiv::OsdVertex> OsdHbrHalfedge;

enum KernelType { kCPU = 0,
                  kOPENMP = 1,
                  kTBB = 2,
                  kGCD = 3,
                  kCUDA = 4,
                  kCL = 5,
                  kGLSL = 6,
                  kGLSLCompute = 7 };

enum DisplayStyle { kWire = 0,
                    kShaded,
                    kWireShaded,
                    kVaryingColor,
                    kFaceVaryingColor };

enum HudCheckBox { kHUD_CB_DISPLAY_CAGE_EDGES,
                   kHUD_CB_DISPLAY_CAGE_VERTS,
                   kHUD_CB_ANIMATE_VERTICES,
                   kHUD_CB_DISPLAY_PATCH_COLOR,
                   kHUD_CB_VIEW_LOD,
                   kHUD_CB_FRACTIONAL_SPACING,
                   kHUD_CB_PATCH_CULL,
                   kHUD_CB_FREEZE };

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
int   g_fullscreen = 0,
      g_freeze = 0,
      g_displayStyle = kWireShaded,
      g_adaptive = 0,
      g_drawCageEdges = 1,
      g_drawCageVertices = 0,
      g_mbutton[3] = {0, 0, 0},
      g_running = 1;

int   g_displayPatchColor = 1,
      g_screenSpaceTess = 0,
      g_fractionalSpacing = 0,
      g_patchCull = 0;

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
                   g_positions,
                   g_normals;

Scheme             g_scheme;

int g_level = 2;
int g_tessLevel = 1;
int g_tessLevelMin = 1;
int g_kernel = kCPU;
float g_moveScale = 0.0f;

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

GLuint g_primQuery = 0;

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
        /*
        std::cerr << "GL error: "
                  << (where.empty() ? "" : where + " ")
                  << err << "\n";
        */
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
linkDefaultProgram()
{
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

    std::vector<float> vertex, varying;
    vertex.reserve(nverts*3);
    if (g_displayStyle == kVaryingColor)
        varying.reserve(nverts*3);

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
        if (g_displayStyle == kVaryingColor) {
            varying.push_back(p[2]);
            varying.push_back(p[1]);
            varying.push_back(p[0]);
            p += 3;
        }
        pp += 3;
    }

    g_mesh->UpdateVertexBuffer(&vertex[0], 0, nverts);

    if (g_displayStyle == kVaryingColor)
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
    else if (kernel == kGCD)
        return "GCD";
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
createOsdMesh( const std::string &shape, int level, int kernel, Scheme scheme=kCatmark ) {

    checkGLErrors("create osd enter");
    // generate Hbr representation from "obj" description
    OsdHbrMesh * hmesh = simpleHbr<OpenSubdiv::OsdVertex>(shape.c_str(), scheme, g_orgPositions,
                                                          g_displayStyle == kFaceVaryingColor);

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
    bits.set(OpenSubdiv::MeshFVarData, 1);

    int numVertexElements = 3;
    int numVaryingElements = (g_displayStyle == kVaryingColor) ? 3 : 0;

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
#ifdef OPENSUBDIV_HAS_TBB
    } else if (kernel == kTBB) {
        if (not g_tbbComputeController) {
            g_tbbComputeController = new OpenSubdiv::OsdTbbComputeController();
        }
        g_mesh = new OpenSubdiv::OsdMesh<OpenSubdiv::OsdCpuGLVertexBuffer,
                                         OpenSubdiv::OsdTbbComputeController,
                                         OpenSubdiv::OsdGLDrawContext>(
                                                g_tbbComputeController,
                                                hmesh,
                                                numVertexElements,
                                                numVaryingElements,
                                                level, bits);
#endif
#ifdef OPENSUBDIV_HAS_GCD
    } else if (kernel == kGCD) {
        if (not g_gcdComputeController) {
            g_gcdComputeController = new OpenSubdiv::OsdGcdComputeController();
        }
        g_mesh = new OpenSubdiv::OsdMesh<OpenSubdiv::OsdCpuGLVertexBuffer,
                                         OpenSubdiv::OsdGcdComputeController,
                                         OpenSubdiv::OsdGLDrawContext>(
                                                g_gcdComputeController,
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

    // -------- VAO
    glBindVertexArray(g_vao);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_mesh->GetDrawContext()->GetPatchIndexBuffer());
    glBindBuffer(GL_ARRAY_BUFFER, g_mesh->BindVertexBuffer());

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof (GLfloat) * 3, 0);

    if (g_displayStyle == kVaryingColor) {
        glBindBuffer(GL_ARRAY_BUFFER, g_mesh->BindVaryingBuffer());
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof (GLfloat) * 3, 0);
    } else {
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
setSharpnessColor(float s, float *r, float *g, float *b)
{
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

    assert(sconfig);

#if defined(GL_ARB_tessellation_shader) || defined(GL_VERSION_4_0)
    const char *glslVersion = "#version 400\n";
#else
    const char *glslVersion = "#version 330\n";
#endif

    if (desc.first.GetType() == OpenSubdiv::FarPatchTables::QUADS or
        desc.first.GetType() == OpenSubdiv::FarPatchTables::TRIANGLES) {
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

    if (desc.first.GetType() == OpenSubdiv::FarPatchTables::QUADS) {
        // uniform catmark, bilinear
        sconfig->geometryShader.AddDefine("PRIM_QUAD");
        sconfig->fragmentShader.AddDefine("PRIM_QUAD");
        sconfig->commonShader.AddDefine("UNIFORM_SUBDIVISION");
    } else if (desc.first.GetType() == OpenSubdiv::FarPatchTables::TRIANGLES) {
        // uniform loop
        sconfig->geometryShader.AddDefine("PRIM_TRI");
        sconfig->fragmentShader.AddDefine("PRIM_TRI");
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
bindProgram(Effect effect, OpenSubdiv::OsdDrawContext::PatchArray const & patch)
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
                45.0f, (float)aspect, 0.01f, 500.0f);
    multMatrix(g_transformData.ModelViewProjectionMatrix,
               g_transformData.ModelViewMatrix,
               g_transformData.ProjectionMatrix);

    // make sure that the vertex buffer is interoped back as a GL resources.
    g_mesh->BindVertexBuffer();

    if (g_displayStyle == kVaryingColor)
        g_mesh->BindVaryingBuffer();

    glBindVertexArray(g_vao);

    OpenSubdiv::OsdDrawContext::PatchArrayVector const & patches = g_mesh->GetDrawContext()->patchArrays;

    // patch drawing
    int patchCount[11][6][4]; // [Type][Pattern][Rotation] (see far/patchTables.h)
    memset(patchCount, 0, sizeof(patchCount));

    // primitive counting
    glBeginQuery(GL_PRIMITIVES_GENERATED, g_primQuery);

    for (int i=0; i<(int)patches.size(); ++i) {
        OpenSubdiv::OsdDrawContext::PatchArray const & patch = patches[i];

        OpenSubdiv::OsdDrawContext::PatchDescriptor desc = patch.GetDescriptor();
        OpenSubdiv::FarPatchTables::Type patchType = desc.GetType();
        int patchPattern = desc.GetPattern();
        int patchRotation = desc.GetRotation();
        int subPatch = desc.GetSubPatch();

        if (subPatch == 0) {
            patchCount[patchType][patchPattern][patchRotation] += patch.GetNumPatches();
        }

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
        GLuint program = bindProgram(GetEffect(), patch);

        GLuint diffuseColor = glGetUniformLocation(program, "diffuseColor");

        if (g_displayPatchColor and primType == GL_PATCHES) {
            float const * color = getAdaptivePatchColor( desc );
            glProgramUniform4f(program, diffuseColor, color[0], color[1], color[2], color[3]);
        } else {
            glProgramUniform4f(program, diffuseColor, 0.4f, 0.4f, 0.8f, 1);
        }

        GLuint uniformGregoryQuadOffsetBase =
          glGetUniformLocation(program, "OsdGregoryQuadOffsetBase");
        GLuint uniformPrimitiveIdBase =
          glGetUniformLocation(program, "OsdPrimitiveIdBase");

        glProgramUniform1i(program, uniformGregoryQuadOffsetBase,
                           patch.GetQuadOffsetIndex());
        glProgramUniform1i(program, uniformPrimitiveIdBase,
                           patch.GetPatchIndex());
#else
        GLuint program = bindProgram(GetEffect(), patch);
        GLint uniformPrimitiveIdBase =
          glGetUniformLocation(program, "OsdPrimitiveIdBase");
        if (uniformPrimitiveIdBase != -1)
            glUniform1i(uniformPrimitiveIdBase, patch.GetPatchIndex());
#endif

        if (g_displayStyle == kWire) {
            glDisable(GL_CULL_FACE);
        }

        glDrawElements(primType, patch.GetNumIndices(), GL_UNSIGNED_INT,
                       (void *)(patch.GetVertIndex() * sizeof(unsigned int)));
        if (g_displayStyle == kWire) {
            glEnable(GL_CULL_FACE);
        }
    }

    glEndQuery(GL_PRIMITIVES_GENERATED);

    GLuint numPrimsGenerated = 0;
    glGetQueryObjectuiv(g_primQuery, GL_QUERY_RESULT, &numPrimsGenerated);

    glBindVertexArray(0);

    glUseProgram(0);

    if (g_drawCageEdges)
        drawCageEdges();

    if (g_drawCageVertices)
        drawCageVertices();

    s.Stop();
    float drawCpuTime = float(s.GetElapsed() * 1000.0f);
    s.Start();
    glFinish();
    s.Stop();
    float drawGpuTime = float(s.GetElapsed() * 1000.0f);

    if (g_hud.IsVisible()) {
        g_fpsTimer.Stop();
        double fps = 1.0/g_fpsTimer.GetElapsed();
        g_fpsTimer.Start();

        int x = -280;
        g_hud.DrawString(x, -360, "NonPatch         : %d",
                         patchCount[OpenSubdiv::FarPatchTables::QUADS][0][0]);
        g_hud.DrawString(x, -340, "Regular          : %d",
                         patchCount[OpenSubdiv::FarPatchTables::REGULAR][0][0]);
        g_hud.DrawString(x, -320, "Boundary         : %d",
                         patchCount[OpenSubdiv::FarPatchTables::BOUNDARY][0][0]);
        g_hud.DrawString(x, -300, "Corner           : %d",
                         patchCount[OpenSubdiv::FarPatchTables::CORNER][0][0]);
        g_hud.DrawString(x, -280, "Gregory          : %d",
                         patchCount[OpenSubdiv::FarPatchTables::GREGORY][0][0]);
        g_hud.DrawString(x, -260, "Boundary Gregory : %d",
                         patchCount[OpenSubdiv::FarPatchTables::GREGORY_BOUNDARY][0][0]);
        g_hud.DrawString(x, -240, "Trans. Regular   : %d %d %d %d %d",
                         patchCount[OpenSubdiv::FarPatchTables::REGULAR][OpenSubdiv::FarPatchTables::PATTERN0][0],
                         patchCount[OpenSubdiv::FarPatchTables::REGULAR][OpenSubdiv::FarPatchTables::PATTERN1][0],
                         patchCount[OpenSubdiv::FarPatchTables::REGULAR][OpenSubdiv::FarPatchTables::PATTERN2][0],
                         patchCount[OpenSubdiv::FarPatchTables::REGULAR][OpenSubdiv::FarPatchTables::PATTERN3][0],
                         patchCount[OpenSubdiv::FarPatchTables::REGULAR][OpenSubdiv::FarPatchTables::PATTERN4][0]);
        for (int i=0; i < 5; i++)
            g_hud.DrawString(x, -220+i*20, "Trans. Boundary%d : %d %d %d %d", i,
                             patchCount[OpenSubdiv::FarPatchTables::BOUNDARY][i+1][0],
                             patchCount[OpenSubdiv::FarPatchTables::BOUNDARY][i+1][1],
                             patchCount[OpenSubdiv::FarPatchTables::BOUNDARY][i+1][2],
                             patchCount[OpenSubdiv::FarPatchTables::BOUNDARY][i+1][3]);
        for (int i=0; i < 5; i++)
            g_hud.DrawString(x, -100+i*20, "Trans. Corner%d  : %d %d %d %d", i,
                             patchCount[OpenSubdiv::FarPatchTables::CORNER][i+1][0],
                             patchCount[OpenSubdiv::FarPatchTables::CORNER][i+1][1],
                             patchCount[OpenSubdiv::FarPatchTables::CORNER][i+1][2],
                             patchCount[OpenSubdiv::FarPatchTables::CORNER][i+1][3]);

        g_hud.DrawString(10, -180, "Tess level : %d", g_tessLevel);
        g_hud.DrawString(10, -160, "Primitives : %d", numPrimsGenerated);
        g_hud.DrawString(10, -140, "Vertices   : %d", g_mesh->GetNumVertices());
        g_hud.DrawString(10, -120, "Scheme     : %s", g_scheme==kBilinear ? "BILINEAR" : (g_scheme == kLoop ? "LOOP" : "CATMARK"));
        g_hud.DrawString(10, -100, "GPU Kernel : %.3f ms", g_gpuTime);
        g_hud.DrawString(10, -80,  "CPU Kernel : %.3f ms", g_cpuTime);
        g_hud.DrawString(10, -60,  "GPU Draw   : %.3f ms", drawGpuTime);
        g_hud.DrawString(10, -40,  "CPU Draw   : %.3f ms", drawCpuTime);
        g_hud.DrawString(10, -20,  "FPS        : %3.1f", fps);

        g_hud.Flush();
    }

    glFinish();

    checkGLErrors("display leave");
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
static void
#if GLFW_VERSION_MAJOR>=3
mouse(GLFWwindow *, int button, int state, int mods) {
#else
mouse(int button, int state) {
#endif

    if (button == 0 && state == GLFW_PRESS && g_hud.MouseClick(g_prev_x, g_prev_y))
        return;

    if (button < 3) {
        g_mbutton[button] = (state == GLFW_PRESS);
    }
}

//------------------------------------------------------------------------------
static void
uninitGL() {

    glDeleteQueries(1, &g_primQuery);

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

#ifdef OPENSUBDIV_HAS_GCD
    delete g_gcdComputeController;
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
#if GLFW_VERSION_MAJOR>=3
reshape(GLFWwindow *, int width, int height) {
#else
reshape(int width, int height) {
#endif

    g_width = width;
    g_height = height;

    g_hud.Rebuild(width, height);
}

//------------------------------------------------------------------------------
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
static void
toggleFullScreen() {
    // XXXX manuelk : to re-implement from glut
}

//------------------------------------------------------------------------------
static void
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
        case 'F': fitFrame(); break;
        case GLFW_KEY_TAB: toggleFullScreen(); break;
        case '+':
        case '=':  g_tessLevel++; break;
        case '-':  g_tessLevel = std::max(g_tessLevelMin, g_tessLevel-1); break;
        case GLFW_KEY_ESCAPE: g_hud.SetVisible(!g_hud.IsVisible()); break;
    }
}

//------------------------------------------------------------------------------
static void
rebuildOsdMesh()
{
    createOsdMesh( g_defaultShapes[ g_currentShape ].data, g_level, g_kernel, g_defaultShapes[ g_currentShape ].scheme );
}

static void
callbackDisplayStyle(int b)
{
    if (g_displayStyle == kVaryingColor or b == kVaryingColor or
        g_displayStyle == kFaceVaryingColor or b == kFaceVaryingColor) {
        // need to rebuild for varying reconstruct
        g_displayStyle = b;
        rebuildOsdMesh();
        return;
    }
    g_displayStyle = b;
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
        cudaGLSetGLDevice( cutGetMaxGflopsDeviceId() );
    }
#endif

    rebuildOsdMesh();
}

static void
callbackLevel(int l)
{
    g_level = l;
    rebuildOsdMesh();
}

static void
callbackModel(int m)
{
    if (m < 0)
        m = 0;

    if (m >= (int)g_defaultShapes.size())
        m = (int)g_defaultShapes.size() - 1;

    g_currentShape = m;
    rebuildOsdMesh();
}

static void
callbackAdaptive(bool checked, int a)
{
    if (OpenSubdiv::OsdGLDrawContext::SupportsAdaptiveTessellation()) {
        g_adaptive = checked;
        rebuildOsdMesh();
    }
}

static void
callbackCheckBox(bool checked, int button)
{
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
    }
}

static void
initHUD()
{
    g_hud.Init(g_width, g_height);

    g_hud.AddRadioButton(0, "CPU (K)", true, 10, 10, callbackKernel, kCPU, 'k');
#ifdef OPENSUBDIV_HAS_OPENMP
    g_hud.AddRadioButton(0, "OPENMP", false, 10, 30, callbackKernel, kOPENMP, 'k');
#endif
#ifdef OPENSUBDIV_HAS_TBB
    g_hud.AddRadioButton(0, "TBB", false, 10, 50, callbackKernel, kTBB, 'k');
#endif
#ifdef OPENSUBDIV_HAS_GCD
    g_hud.AddRadioButton(0, "GCD", false, 10, 70, callbackKernel, kGCD, 'k');
#endif
#ifdef OPENSUBDIV_HAS_CUDA
    g_hud.AddRadioButton(0, "CUDA",   false, 10, 90, callbackKernel, kCUDA, 'k');
#endif
#ifdef OPENSUBDIV_HAS_OPENCL
    g_hud.AddRadioButton(0, "OPENCL", false, 10, 110, callbackKernel, kCL, 'k');
#endif
#ifdef OPENSUBDIV_HAS_GLSL_TRANSFORM_FEEDBACK
    g_hud.AddRadioButton(0, "GLSL TransformFeedback",   false, 10, 130, callbackKernel, kGLSL, 'k');
#endif
#ifdef OPENSUBDIV_HAS_GLSL_COMPUTE
    // Must also check at run time for OpenGL 4.3
    if (GLEW_VERSION_4_3) {
        g_hud.AddRadioButton(0, "GLSL Compute",   false, 10, 150, callbackKernel, kGLSLCompute, 'k');
    }
#endif

    g_hud.AddRadioButton(1, "Wire (W)",    g_displayStyle == kWire,  200, 10, callbackDisplayStyle, 0, 'w');
    g_hud.AddRadioButton(1, "Shaded",      g_displayStyle == kShaded, 200, 30, callbackDisplayStyle, 1, 'w');
    g_hud.AddRadioButton(1, "Wire+Shaded", g_displayStyle == kWireShaded, 200, 50, callbackDisplayStyle, 2, 'w');
    g_hud.AddRadioButton(1, "Varying color", g_displayStyle == kVaryingColor, 200, 70, callbackDisplayStyle, 3, 'w');
    g_hud.AddRadioButton(1, "FaceVarying color", g_displayStyle == kFaceVaryingColor, 200, 90, callbackDisplayStyle, 4, 'w');

    g_hud.AddCheckBox("Cage Edges (H)", g_drawCageEdges != 0,
                      350, 10, callbackCheckBox, kHUD_CB_DISPLAY_CAGE_EDGES, 'h');
    g_hud.AddCheckBox("Cage Verts (J)", g_drawCageVertices != 0,
                      350, 30, callbackCheckBox, kHUD_CB_DISPLAY_CAGE_VERTS, 'j');
    g_hud.AddCheckBox("Animate vertices (M)", g_moveScale != 0,
                      350, 50, callbackCheckBox, kHUD_CB_ANIMATE_VERTICES, 'm');
    g_hud.AddCheckBox("Patch Color (P)", g_displayPatchColor != 0,
                      350, 70, callbackCheckBox, kHUD_CB_DISPLAY_PATCH_COLOR, 'p');
    g_hud.AddCheckBox("Screen space LOD (V)",  g_screenSpaceTess != 0,
                      350, 90, callbackCheckBox, kHUD_CB_VIEW_LOD, 'v');
    g_hud.AddCheckBox("Fractional spacing (T)",  g_fractionalSpacing != 0,
                      350, 110, callbackCheckBox, kHUD_CB_FRACTIONAL_SPACING, 't');
    g_hud.AddCheckBox("Frustum Patch Culling (B)",  g_patchCull != 0,
                      350, 130, callbackCheckBox, kHUD_CB_PATCH_CULL, 'b');
    g_hud.AddCheckBox("Freeze (spc)", g_freeze != 0,
                      350, 150, callbackCheckBox, kHUD_CB_FREEZE, ' ');

    if (OpenSubdiv::OsdGLDrawContext::SupportsAdaptiveTessellation())
        g_hud.AddCheckBox("Adaptive (`)", g_adaptive!=0, 10, 150, callbackAdaptive, 0, '`');

    for (int i = 1; i < 11; ++i) {
        char level[16];
        sprintf(level, "Lv. %d", i);
        g_hud.AddRadioButton(3, level, i==2, 10, 170+i*20, callbackLevel, i, '0'+(i%10));
    }

    for (int i = 0; i < (int)g_defaultShapes.size(); ++i) {
        g_hud.AddRadioButton(4, g_defaultShapes[i].name.c_str(), i==g_currentShape, -220, 10+i*16, callbackModel, i, 'n');
    }
}

//------------------------------------------------------------------------------
static void
initGL()
{
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glCullFace(GL_BACK);
    glEnable(GL_CULL_FACE);

    glGenQueries(1, &g_primQuery);

    glGenVertexArrays(1, &g_vao);
    glGenVertexArrays(1, &g_cageVertexVAO);
    glGenVertexArrays(1, &g_cageEdgeVAO);
    glGenBuffers(1, &g_cageVertexVBO);
    glGenBuffers(1, &g_cageEdgeVBO);
}

//------------------------------------------------------------------------------
static void
idle() {

    if (not g_freeze)
        g_frame++;

    updateGeom();

    if (g_repeatCount != 0 and g_frame >= g_repeatCount)
        g_running = 0;
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

    glfwOpenWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#if not defined(__APPLE__)
    glfwOpenWindowHint(GLFW_OPENGL_VERSION_MAJOR, 4);
    glfwOpenWindowHint(GLFW_OPENGL_VERSION_MINOR, 2);
    glfwOpenWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#else
    glfwOpenWindowHint(GLFW_OPENGL_VERSION_MAJOR, 3);
    glfwOpenWindowHint(GLFW_OPENGL_VERSION_MINOR, 2);
#endif

}

//------------------------------------------------------------------------------
int main(int argc, char ** argv)
{
    bool fullscreen = false;
    std::string str;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-d"))
            g_level = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-c"))
            g_repeatCount = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-f"))
            fullscreen = true;
        else {
            std::ifstream ifs(argv[1]);
            if (ifs) {
                std::stringstream ss;
                ss << ifs.rdbuf();
                ifs.close();
                str = ss.str();
                g_defaultShapes.push_back(SimpleShape(str.c_str(), argv[1], kCatmark));
            }
        }
    }
    initializeShapes();
    OsdSetErrorCallback(callbackError);

    if (not glfwInit()) {
        printf("Failed to initialize GLFW\n");
        return 1;
    }

    static const char windowTitle[] = "OpenSubdiv glViewer";

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
    glfwSetWindowSizeCallback(g_window, reshape);
    glfwSetWindowCloseCallback(g_window, windowClose);
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
    glfwSetWindowSizeCallback(reshape);
    glfwSetWindowCloseCallback(windowClose);
#endif


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
    g_adaptive = OpenSubdiv::OsdGLDrawContext::SupportsAdaptiveTessellation();

    initGL();
    linkDefaultProgram();

    glfwSwapInterval(0);

    initHUD();
    rebuildOsdMesh();

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

    uninitGL();
    glfwTerminate();
}

//------------------------------------------------------------------------------
