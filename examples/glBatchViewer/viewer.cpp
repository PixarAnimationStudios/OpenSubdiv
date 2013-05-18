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
    #include <OpenGL/gl3.h>
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
    #include <GL/glfw3.h>
    GLFWwindow* g_window=0;
    GLFWmonitor* g_primary=0;
#else
    #include <GL/glfw.h>
#endif

#include <osd/error.h>
#include <osd/vertex.h>
#include <osd/glDrawContext.h>
#include <osd/glDrawRegistry.h>
#include <osd/sortedDrawContext.h>

#include <osd/cpuGLVertexBuffer.h>
#include <osd/cpuComputeContext.h>
#include <osd/cpuComputeController.h>
OpenSubdiv::OsdCpuComputeController *g_cpuComputeController = NULL;

#ifdef OPENSUBDIV_HAS_OPENMP
    #include <osd/ompComputeController.h>
#endif

#ifdef OPENSUBDIV_HAS_GCD
    #include <osd/gcdComputeController.h>
#endif

#ifdef OPENSUBDIV_HAS_OPENCL
    #include <osd/clGLVertexBuffer.h>
    #include <osd/clComputeContext.h>
    #include <osd/clComputeController.h>

    #include "../common/clInit.h"

    cl_context g_clContext;
    cl_command_queue g_clQueue;
#endif

#ifdef OPENSUBDIV_HAS_CUDA
    #include <osd/cudaGLVertexBuffer.h>
    #include <osd/cudaComputeContext.h>
    #include <osd/cudaComputeController.h>

    #include <cuda_runtime_api.h>
    #include <cuda_gl_interop.h>

    #include "../common/cudaInit.h"

    bool g_cudaInitialized = false;
#endif

#ifdef OPENSUBDIV_HAS_GLSL_TRANSFORM_FEEDBACK
    #include <osd/glslTransformFeedbackComputeContext.h>
    #include <osd/glslTransformFeedbackComputeController.h>
    #include <osd/glVertexBuffer.h>
#endif

#ifdef OPENSUBDIV_HAS_GLSL_COMPUTE
    #include <osd/glslComputeContext.h>
    #include <osd/glslComputeController.h>
    #include <osd/glVertexBuffer.h>
#endif

#include <far/meshFactory.h>
#include <far/multiMeshFactory.h>

OpenSubdiv::FarMesh<OpenSubdiv::OsdVertex> *g_farMesh = NULL;

template <class CONTROLLER> class Controller
{
public:
    static CONTROLLER *GetInstance() {
        static CONTROLLER *instance = NULL;
        if (not instance) instance = new CONTROLLER();
        return instance;
    }
};

template<typename V> void updateData(V *vertexBuffer, const float *src, int numVertices, int startVertex) {
    vertexBuffer->UpdateData(src, numVertices, startVertex);
}

#ifdef OPENSUBDIV_HAS_OPENCL
template<> class Controller<OpenSubdiv::OsdCLComputeController>
{
public:
    static OpenSubdiv::OsdCLComputeController *GetInstance() {
        static OpenSubdiv::OsdCLComputeController *instance = NULL;
        if (not instance) instance = new OpenSubdiv::OsdCLComputeController(g_clContext, g_clQueue);
        return instance;
    }
};

template<> void updateData(OpenSubdiv::OsdCLGLVertexBuffer *vertexBuffer, const float *src, int numVertices, int startVertex) {
    vertexBuffer->UpdateData(src, numVertices, startVertex, g_clQueue);
}
#endif

class MeshBase
{
public:
    virtual ~MeshBase() {}
    virtual void UpdateData(const float *src, int numVertices, int startVertex)=0;
    virtual void Refine()=0;
    virtual void Synchronize()=0;
    virtual GLuint BindVBO()=0;
    virtual size_t GetVBOMemoryUsed() const=0;
    virtual OpenSubdiv::OsdGLDrawContext *GetDrawContext()=0;
    virtual OpenSubdiv::OsdSortedDrawContext *GetSortedDrawContext()=0;
};

template <class VERTEX_BUFFER,
          class COMPUTE_CONTEXT,
          class COMPUTE_CONTROLLER> class Mesh : public MeshBase
{
public:
    static Mesh *Create(OpenSubdiv::FarMesh<OpenSubdiv::OsdVertex> *farmesh) {
        COMPUTE_CONTEXT *computeContext = COMPUTE_CONTEXT::Create(farmesh);
        VERTEX_BUFFER *computeVertexBuffer = VERTEX_BUFFER::Create(3, farmesh->GetNumVertices());
        OpenSubdiv::OsdGLDrawContext *drawContext = OpenSubdiv::OsdGLDrawContext::Create(farmesh, computeVertexBuffer);
        OpenSubdiv::OsdSortedDrawContext *sortedDrawContext = farmesh->GetPatchTables() ?
            new OpenSubdiv::OsdSortedDrawContext(farmesh->GetPatchTables()->GetPatchCounts(), drawContext->patchArrays) : NULL;
        return new Mesh(computeContext,
                        computeVertexBuffer,
                        drawContext,
                        sortedDrawContext,
                        farmesh->GetKernelBatches());
    }

    static Mesh *Create(OpenSubdiv::FarMesh<OpenSubdiv::OsdVertex> *farmesh, cl_context clContext) {
        COMPUTE_CONTEXT *computeContext = COMPUTE_CONTEXT::Create(farmesh, clContext);
        VERTEX_BUFFER *computeVertexBuffer = VERTEX_BUFFER::Create(3, farmesh->GetNumVertices(), clContext);
        OpenSubdiv::OsdGLDrawContext *drawContext = OpenSubdiv::OsdGLDrawContext::Create(farmesh, computeVertexBuffer);
        OpenSubdiv::OsdSortedDrawContext *sortedDrawContext = farmesh->GetPatchTables() ?
            new OpenSubdiv::OsdSortedDrawContext(farmesh->GetPatchTables()->GetPatchCounts(), drawContext->patchArrays) : NULL;

        return new Mesh(computeContext,
                        computeVertexBuffer,
                        drawContext,
                        sortedDrawContext,
                        farmesh->GetKernelBatches());
    }

    virtual ~Mesh() {
        delete _computeContext;
        delete _computeVertexBuffer;
        delete _drawContext;
    }

    virtual OpenSubdiv::OsdGLDrawContext *GetDrawContext() {
        return _drawContext;
    }

    virtual OpenSubdiv::OsdSortedDrawContext *GetSortedDrawContext() {
        return _sortedDrawContext;
    }

    virtual void UpdateData(const float *src, int startVertex, int numVertices) {
        updateData(_computeVertexBuffer, src, startVertex, numVertices) ;
    }

    virtual void Refine() {
        Controller<COMPUTE_CONTROLLER>::GetInstance()->Refine(_computeContext,
                                                              _batches,
                                                              _computeVertexBuffer);
    }

    virtual void Synchronize() {
        Controller<COMPUTE_CONTROLLER>::GetInstance()->Synchronize();
    }

    virtual GLuint BindVBO() {
        return _computeVertexBuffer->BindVBO();
    }

    virtual size_t GetVBOMemoryUsed() const {
        size_t size = _computeVertexBuffer->GetNumElements() * _computeVertexBuffer->GetNumVertices() * sizeof(float);
        return size;
    }

protected:
    Mesh(COMPUTE_CONTEXT *computeContext,
         VERTEX_BUFFER *computeVertexBuffer,
         OpenSubdiv::OsdGLDrawContext *drawContext,
         OpenSubdiv::OsdSortedDrawContext *sortedDrawContext,
         OpenSubdiv::FarKernelBatchVector const &batches) :

        _computeVertexBuffer(computeVertexBuffer),
        _computeContext(computeContext),
        _drawContext(drawContext),
        _sortedDrawContext(sortedDrawContext),
        _batches(batches) {}

private:
    VERTEX_BUFFER	*_computeVertexBuffer;
    COMPUTE_CONTEXT	*_computeContext;
    OpenSubdiv::OsdGLDrawContext *_drawContext;
    OpenSubdiv::OsdSortedDrawContext *_sortedDrawContext;
    OpenSubdiv::FarKernelBatchVector _batches;
};

MeshBase *g_mesh;

#include <common/shape_utils.h>
#include "../common/stopwatch.h"
#include "../common/simple_math.h"
#include "../common/gl_hud.h"

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
                  kGCD = 2,
                  kCUDA = 3,
                  kCL = 4,
                  kGLSL = 5,
                  kGLSLCompute = 6 };

enum HudCheckBox { HUD_CB_DRAW_AT_ONCE,
                   HUD_CB_ADAPTIVE,
                   HUD_CB_ANIMATE_VERTICES,
                   HUD_CB_DISPLAY_PATCH_COLOR,
                   HUD_CB_VIEW_LOD,
                   HUD_CB_FREEZE };

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
      g_wire = 2,
      g_adaptive = 1,
      g_drawAtOnce = 0,
      g_mbutton[3] = {0, 0, 0}, 
      g_running = 1;

int   g_displayPatchColor = 1;

float g_rotate[2] = {0, 0},
      g_dolly = 5,
      g_pan[2] = {0, 0},
      g_center[3] = {0, 0, 0},
      g_size = 1;

int   g_prev_x = 0,
      g_prev_y = 0;

int   g_width = 1024,
      g_height = 1024;

GLhud g_hud;

// performance
float g_cpuTime = 0;
float g_gpuTime = 0;
Stopwatch g_fpsTimer;
double g_totalTime = 0;

// geometry
std::vector<std::vector<float> > g_positions;
std::vector<int>   g_vertexOffsets;

struct BBox {
    BBox(const float *min, const float *max) {
        for (int i = 0; i < 3; ++i) {
            _p[0][i] = min[i];
            _p[1][i] = max[i];
        }
    }
    bool OutOfFrustum(const float *modelViewProjection) {
        int clip_or = 0;
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                for (int k = 0; k < 2; ++k) {
                    float v[4];
                    v[0] = _p[i][0];
                    v[1] = _p[j][1];
                    v[2] = _p[k][2];
                    v[3] = 1.0f;
                    apply(v, modelViewProjection);
                    int clip = 0;
                    clip = (v[0] < v[3]);
                    clip = (v[1] < v[3]) | (clip << 1);
                    clip = (v[0] > -v[3]) | (clip << 1);
                    clip = (v[1] > -v[3]) | (clip << 1);
                    clip_or |= clip;
                }
            }
        }
        return clip_or != 0xf;
    }
    float _p[2][3];
};
std::vector<BBox> g_bboxes;
struct Matrix {
    float value[16];
};
std::vector<Matrix> g_transforms;

Scheme             g_scheme;

int g_level = 2;
int g_tessLevel = 1;
int g_tessLevelMin = 0;
int g_screenSpaceTess = 1;
int g_kernel = kCPU;
float g_moveScale = 0.0f;
#define MAX_MODELS 600
int g_modelCount = 4;

GLuint g_transformUB = 0,
       g_transformBinding = 0,
       g_tessellationUB = 0,
       g_tessellationBinding = 0,
       g_lightingUB = 0,
       g_lightingBinding = 0;

// XXX: temp. should be moved to glDrawContext
std::map<GLuint, GLuint> g_gregoryQuadOffsetBaseMap;
std::map<GLuint, GLuint> g_levelBaseMap;

struct Transform {
    float ModelViewMatrix[16];
    float ProjectionMatrix[16];
    float ModelViewProjectionMatrix[16];
} g_transformData;

GLuint g_primQuery = 0;

GLuint g_vao = 0;

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



#ifndef WIN32 // exceeds max string literal (65535 chars)
#include <shapes/catmark_bishop.h>
    g_defaultShapes.push_back(SimpleShape(catmark_bishop, "catmark_bishop", kCatmark));
#endif

#ifndef WIN32 // exceeds max string literal (65535 chars)
#include <shapes/catmark_car.h>
    g_defaultShapes.push_back(SimpleShape(catmark_car, "catmark_car", kCatmark));
#endif

#include <shapes/catmark_helmet.h>
    g_defaultShapes.push_back(SimpleShape(catmark_helmet, "catmark_helmet", kCatmark));

#include <shapes/catmark_pawn.h>
    g_defaultShapes.push_back(SimpleShape(catmark_pawn, "catmark_pawn", kCatmark));

#ifndef WIN32 // exceeds max string literal (65535 chars)
#include <shapes/catmark_rook.h>
    g_defaultShapes.push_back(SimpleShape(catmark_rook, "catmark_rook", kCatmark));
#endif


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

static void
updateGeom() {

    float r = (float)sin(g_totalTime) * g_moveScale;
    for (int j = 0; j < g_modelCount * g_modelCount; ++j) {
        int nverts = (int)g_positions[j].size()/3;

        std::vector<float> vertex;
        vertex.resize(nverts * 3);
        float * d = &vertex[0];

        const float *p = &g_positions[j][0];
        for (int i = 0; i < nverts; ++i) {
            float ct = cos(p[2] * r);
            float st = sin(p[2] * r);
            float v[4];
            v[0] = p[0]*ct + p[1]*st;
            v[1] = -p[0]*st + p[1]*ct;
            v[2] = p[2];
            v[3] = 1;
            apply(v, g_transforms[j].value);
            *d++ = v[0];
            *d++ = v[1];
            *d++ = v[2];
        
            p += 3;
        }
        
        g_mesh->UpdateData(&vertex[0], g_vertexOffsets[j], nverts);
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

//------------------------------------------------------------------------------
static const char *
getKernelName(int kernel) {

         if (kernel == kCPU)
        return "CPU";
    else if (kernel == kOPENMP)
        return "OpenMP";
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
static OpenSubdiv::FarMesh<OpenSubdiv::OsdVertex> *
createFarMesh( const char * shape, int level, bool adaptive, Scheme scheme=kCatmark ) {

    checkGLErrors("create osd enter");
    // generate Hbr representation from "obj" description
    std::vector<float> positions;
    OsdHbrMesh * hmesh = simpleHbr<OpenSubdiv::OsdVertex>(shape, scheme, positions);

    size_t nModel = g_bboxes.size();
    float x = nModel%g_modelCount - g_modelCount*0.5f;
    float y = nModel/g_modelCount - g_modelCount*0.5f;
    // align origins
    Matrix matrix;
    identity(matrix.value);
    translate(matrix.value, 3*x, 3*y, 0);
    g_transforms.push_back(matrix);
    g_positions.push_back(positions);

    OpenSubdiv::FarMeshFactory<OpenSubdiv::OsdVertex> meshFactory(hmesh, level, adaptive);
    OpenSubdiv::FarMesh<OpenSubdiv::OsdVertex> *farMesh = meshFactory.Create();

    // Hbr mesh can be deleted
    delete hmesh;

    // compute model bounding (vertex animation isn't taken into account)
    float min[4] = { FLT_MAX,  FLT_MAX,  FLT_MAX, 1};
    float max[4] = {-FLT_MAX, -FLT_MAX, -FLT_MAX, 1};
    for (size_t i=0; i <positions.size()/3; ++i) {
        float v[4] = {positions[i*3], positions[i*3+1], positions[i*3+2], 1 };
        for(int j=0; j<3; ++j) {
            min[j] = std::min(min[j], v[j]);
            max[j] = std::max(max[j], v[j]);
        }
    }
    g_bboxes.push_back(BBox(min, max));

    return farMesh;
}

static void
rebuildOsd()
{
    delete g_mesh;
    g_mesh = NULL;

    if (g_kernel == kCPU) {
        g_mesh = Mesh<OpenSubdiv::OsdCpuGLVertexBuffer,
            OpenSubdiv::OsdCpuComputeContext,
            OpenSubdiv::OsdCpuComputeController>::Create(g_farMesh);
#ifdef OPENSUBDIV_HAS_OPENMP
    } else if (g_kernel == kOPENMP) {
        g_mesh = Mesh<OpenSubdiv::OsdCpuGLVertexBuffer,
            OpenSubdiv::OsdCpuComputeContext,
            OpenSubdiv::OsdOmpComputeController>::Create(g_farMesh);
#endif
/*
#ifdef OPENSUBDIV_HAS_GCD
    } else if (kernel == kGCD) {
        g_mesh = new OpenSubdiv::OsdMesh<OpenSubdiv::OsdCpuGLVertexBuffer,
                                         OpenSubdiv::OsdGcdComputeController,
                                         OpenSubdiv::OsdGLDrawContext>(hmesh, 6, level, bits);
#endif
*/
#ifdef OPENSUBDIV_HAS_OPENCL
    } else if(g_kernel == kCL) {
        g_mesh = Mesh<OpenSubdiv::OsdCLGLVertexBuffer,
            OpenSubdiv::OsdCLComputeContext,
            OpenSubdiv::OsdCLComputeController>::Create(g_farMesh, g_clContext);
#endif
#ifdef OPENSUBDIV_HAS_CUDA
    } else if(g_kernel == kCUDA) {
        g_mesh = Mesh<OpenSubdiv::OsdCudaGLVertexBuffer,
            OpenSubdiv::OsdCudaComputeContext,
            OpenSubdiv::OsdCudaComputeController>::Create(g_farMesh);
#endif
#ifdef OPENSUBDIV_HAS_GLSL_TRANSFORM_FEEDBACK
    } else if(g_kernel == kGLSL) {
        g_mesh = Mesh<OpenSubdiv::OsdCpuGLVertexBuffer,
            OpenSubdiv::OsdGLSLTransformFeedbackComputeContext,
            OpenSubdiv::OsdGLSLTransformFeedbackComputeController>::Create(g_farMesh);
#endif
#ifdef OPENSUBDIV_HAS_GLSL_COMPUTE
    } else if(g_kernel == kGLSLCompute) {
        g_mesh = Mesh<OpenSubdiv::OsdCpuGLVertexBuffer,
            OpenSubdiv::OsdGLSLComputeContext,
            OpenSubdiv::OsdGLSLComputeController>::Create(g_farMesh);
#endif
    } else {
        printf("Unsupported kernel %s\n", getKernelName(g_kernel));
    }

    g_tessLevelMin = 0;

    g_tessLevel = std::max(g_tessLevel,g_tessLevelMin);

    updateGeom();

    // -------- VAO 
    glBindVertexArray(g_vao);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_mesh->GetDrawContext()->patchIndexBuffer);

    glBindBuffer(GL_ARRAY_BUFFER, g_mesh->BindVBO());

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof (GLfloat) * 3, 0);

    glBindVertexArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

static void
rebuildFar()
{
    delete g_farMesh;
    g_farMesh = NULL;

    g_positions.clear();
    g_bboxes.clear();
    g_transforms.clear();

    // align scheme and adaptive to the first shape
    int shape = g_currentShape;
    Scheme scheme = g_defaultShapes[ shape ].scheme;
    bool adaptive = scheme==kCatmark ? g_adaptive!=0 : false;

    g_vertexOffsets.clear();
    int vertexOffset = 0;

    // prepare meshes
    std::vector<OpenSubdiv::FarMesh<OpenSubdiv::OsdVertex> const *> farMeshes;
    for (int i = 0; i < g_modelCount*g_modelCount; ++i) {
        g_vertexOffsets.push_back(vertexOffset);

        OpenSubdiv::FarMesh<OpenSubdiv::OsdVertex> *farMesh = createFarMesh(
            g_defaultShapes[ shape ].data.c_str(), g_level, adaptive, scheme);
        farMeshes.push_back(farMesh);

        vertexOffset += farMesh->GetNumVertices();
        shape++;
        if (shape >= (int)g_defaultShapes.size()) shape = 0;
    }
    
    // create multimesh
    OpenSubdiv::FarMultiMeshFactory<OpenSubdiv::OsdVertex> multiMeshFactory;
    g_farMesh = multiMeshFactory.Create(farMeshes);
    g_scheme = scheme;

    for (size_t i = 0; i < farMeshes.size(); ++i) {
        delete farMeshes[i];
    }
    rebuildOsd();
}

//------------------------------------------------------------------------------
static void
fitFrame() {

    g_pan[0] = g_pan[1] = 0;
    g_dolly = g_size;
}

//------------------------------------------------------------------------------
union Effect {

    enum {
        kQuad, kTri
    };
    enum {
        kWire, kFill, kLine
    };
    struct {
        unsigned int prim:1;
        unsigned int wire:2;
        unsigned int screenSpaceTess:1;
    };
    int value;

    bool operator < (const Effect &e) const {
        return value < e.value;
    }
};

typedef std::pair<OpenSubdiv::OsdPatchDescriptor,Effect> EffectDesc;

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

    if (effect.screenSpaceTess) {
        sconfig->commonShader.AddDefine("OSD_ENABLE_PATCH_CULL");
        sconfig->commonShader.AddDefine("OSD_ENABLE_SCREENSPACE_TESSELLATION");
    }

#if defined(GL_ARB_tessellation_shader) || defined(GL_VERSION_4_0)
    const char *glslVersion = "#version 400\n";
#else
    const char *glslVersion = "#version 330\n";
#endif

    if (desc.first.type != OpenSubdiv::kNonPatch) {

        if (effect.prim == Effect::kQuad) effect.prim = Effect::kTri;
        sconfig->geometryShader.AddDefine("SMOOTH_NORMALS");

    } else {
        sconfig->vertexShader.source = shaderSource;
        sconfig->vertexShader.version = glslVersion;
        sconfig->vertexShader.AddDefine("VERTEX_SHADER");
    }
    assert(sconfig);

    sconfig->geometryShader.source = shaderSource;
    sconfig->geometryShader.version = glslVersion;
    sconfig->geometryShader.AddDefine("GEOMETRY_SHADER");

    sconfig->fragmentShader.source = shaderSource;
    sconfig->fragmentShader.version = glslVersion;
    sconfig->fragmentShader.AddDefine("FRAGMENT_SHADER");

    if (effect.prim == Effect::kQuad) {
        sconfig->geometryShader.AddDefine("PRIM_QUAD");
        sconfig->fragmentShader.AddDefine("PRIM_QUAD");
    } else {
        sconfig->geometryShader.AddDefine("PRIM_TRI");
        sconfig->fragmentShader.AddDefine("PRIM_TRI");
    }

    if (effect.wire == Effect::kWire) {
        sconfig->geometryShader.AddDefine("GEOMETRY_OUT_WIRE");
        sconfig->fragmentShader.AddDefine("GEOMETRY_OUT_WIRE");
    } else if (effect.wire == Effect::kFill) {
        sconfig->geometryShader.AddDefine("GEOMETRY_OUT_FILL");
        sconfig->fragmentShader.AddDefine("GEOMETRY_OUT_FILL");
    } else if (effect.wire == Effect::kLine) {
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

    g_lightingBinding = 3;
    uboIndex = glGetUniformBlockIndex(config->program, "Lighting");
    if (uboIndex != GL_INVALID_INDEX)
        glUniformBlockBinding(config->program, uboIndex, g_lightingBinding);

    g_gregoryQuadOffsetBaseMap[config->program] = glGetUniformLocation(config->program, "GregoryQuadOffsetBase");
    g_levelBaseMap[config->program] = glGetUniformLocation(config->program, "LevelBase");

    GLint loc;
#if not (defined(GL_ARB_separate_shader_objects) || defined(GL_VERSION_4_1))
    glUseProgram(config->program);
    if ((loc = glGetUniformLocation(config->program, "g_VertexBuffer")) != -1) {
        glUniform1i(loc, 0); // GL_TEXTURE0
    }
    if ((loc = glGetUniformLocation(config->program, "g_ValenceBuffer")) != -1) {
        glUniform1i(loc, 1); // GL_TEXTURE1
    }
    if ((loc = glGetUniformLocation(config->program, "g_QuadOffsetBuffer")) != -1) {
        glUniform1i(loc, 2); // GL_TEXTURE2
    }
    if ((loc = glGetUniformLocation(config->program, "g_patchLevelBuffer")) != -1) {
        glUniform1i(loc, 3); // GL_TEXTURE3
    }
#else
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
#endif

    return config;
}

EffectDrawRegistry effectRegistry;

static Effect
GetEffect()
{
    Effect effect;
    if (g_scheme == kLoop) {
        effect.prim = Effect::kTri;
    } else {
        effect.prim = Effect::kQuad;
    }
    if (g_wire == 0){
        effect.wire = Effect::kWire;
    } else if (g_wire == 1) {
        effect.wire = Effect::kFill;
    } else {
        effect.wire = Effect::kLine;
    }
    effect.screenSpaceTess = g_screenSpaceTess;
    return effect;
}

//------------------------------------------------------------------------------
static GLuint
bindProgram(Effect effect, OpenSubdiv::OsdPatchArray const & patch)
{
    EffectDesc effectDesc(patch.desc, effect);
    EffectDrawRegistry::ConfigType *
        config = effectRegistry.GetDrawConfig(effectDesc);

    GLuint program = config->program;

    glUseProgram(program);

    return program;
}

void applyPatchColor(GLuint program, int patchType, int patchPattern)
{
#if (defined(GL_ARB_separate_shader_objects) || defined(GL_VERSION_4_1))
    GLuint diffuseColor = glGetUniformLocation(program, "diffuseColor");
    if (g_displayPatchColor) {
        switch(patchType) {
        case OpenSubdiv::kRegular:
            glProgramUniform4f(program, diffuseColor, 1.0f, 1.0f, 1.0f, 1);
            break;
        case OpenSubdiv::kBoundary:
            glProgramUniform4f(program, diffuseColor, 0.8f, 0.0f, 0.0f, 1);
            break;
        case OpenSubdiv::kCorner:
            glProgramUniform4f(program, diffuseColor, 0, 1.0, 0, 1);
            break;
        case OpenSubdiv::kGregory:
            glProgramUniform4f(program, diffuseColor, 1.0f, 1.0f, 0.0f, 1);
            break;
        case OpenSubdiv::kBoundaryGregory:
            glProgramUniform4f(program, diffuseColor, 1.0f, 0.5f, 0.0f, 1);
            break;
        case OpenSubdiv::kTransitionRegular:
            switch (patchPattern) {
            case 0:
                glProgramUniform4f(program, diffuseColor, 0, 1.0f, 1.0f, 1);
                break;
            case 1:
                glProgramUniform4f(program, diffuseColor, 0, 0.5f, 1.0f, 1);
                break;
            case 2:
                glProgramUniform4f(program, diffuseColor, 0, 0.5f, 0.5f, 1);
                break;
            case 3:
                glProgramUniform4f(program, diffuseColor, 0.5f, 0, 1.0f, 1);
                break;
            case 4:
                glProgramUniform4f(program, diffuseColor, 1.0f, 0.5f, 1.0f, 1);
                break;
            }
            break;
        case OpenSubdiv::kTransitionBoundary: {
            float p = patchPattern * 0.2f;
            glProgramUniform4f(program, diffuseColor, 0.0f, p, 0.75f, 1);
        } break;
        case OpenSubdiv::kTransitionCorner:
            glProgramUniform4f(program, diffuseColor, 0.25f, 0.25f, 0.25f, 1);
            break;
        default:
            glProgramUniform4f(program, diffuseColor, 0.4f, 0.4f, 0.8f, 1);
            break;
        }
    } else {
        glProgramUniform4f(program, diffuseColor, 0.4f, 0.4f, 0.8f, 1);
    }
#endif
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
    g_mesh->BindVBO();

    glBindVertexArray(g_vao);

    OpenSubdiv::OsdPatchArrayVector const & patches = g_mesh->GetDrawContext()->patchArrays;

    // prim visibility
    // XXX: currently OsdSortedDrawContext is not available for uniform quads
    if (g_adaptive) {
        for (int i = 0; i < g_modelCount*g_modelCount; ++i) {
            // frustum culling
            float mat[16];
            multMatrix(mat, g_transforms[i].value, g_transformData.ModelViewProjectionMatrix);
            bool visible = not g_bboxes[i].OutOfFrustum(mat);
            g_mesh->GetSortedDrawContext()->SetPrimFidelity(i, visible ?
                                                            OpenSubdiv::OsdSortedDrawContext::kHigh :
                                                            OpenSubdiv::OsdSortedDrawContext::kInvisible);
        }
    }

    // set transform
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

    // prepare textures
    if (g_mesh->GetDrawContext()->vertexTextureBuffer) {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_BUFFER,
            g_mesh->GetDrawContext()->vertexTextureBuffer);
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
    glActiveTexture(GL_TEXTURE0);


    // primitive counting
    glBeginQuery(GL_PRIMITIVES_GENERATED, g_primQuery);
    
    for (int i=0; i<(int)patches.size(); ++i) {
        OpenSubdiv::OsdPatchArray const & patch = patches[i];

#if defined(GL_ARB_tessellation_shader) || defined(GL_VERSION_4_0)
        OpenSubdiv::OsdPatchType patchType = patch.desc.type;
        int patchPattern = patch.desc.pattern;
#endif

        GLenum primType;
        if (g_mesh->GetDrawContext()->IsAdaptive()) {
#if defined(GL_ARB_tessellation_shader) || defined(GL_VERSION_4_0)
            primType = GL_PATCHES;
            glPatchParameteri(GL_PATCH_VERTICES, patch.desc.GetPatchSize());
#endif
        } else {
            if (g_scheme == kLoop) {
                primType = GL_TRIANGLES;
            } else {
                primType = GL_LINES_ADJACENCY; // GL_QUADS is deprecated
            }
        }

        GLuint program = bindProgram(GetEffect(), patch);
#if defined(GL_ARB_tessellation_shader) || defined(GL_VERSION_4_0)
        applyPatchColor(program, patchType, patchPattern);
#endif

        if (g_wire == 0) {
            glDisable(GL_CULL_FACE);
        }
        GLuint uniformGregoryQuadOffset = g_gregoryQuadOffsetBaseMap[program];
        GLuint uniformLevelBase = g_levelBaseMap[program];
        if (g_drawAtOnce or (not g_adaptive)) {
            glUniform1i(uniformGregoryQuadOffset, patch.gregoryQuadOffsetBase);
            glUniform1i(uniformLevelBase, patch.levelBase);

            glDrawElements(primType,
                           patch.numIndices, GL_UNSIGNED_INT,
                           (void *)(patch.firstIndex * sizeof(unsigned int)));
        } else {
            OpenSubdiv::OsdPatchDrawRangeVector const & drawRanges = g_mesh->GetSortedDrawContext()->GetPatchDrawRanges(patch.desc);

            for (size_t j = 0; j < drawRanges.size(); ++j) {

                int primitiveOffset = (drawRanges[j].firstIndex - patch.firstIndex)/patch.desc.GetPatchSize();
                if (patch.desc.type == OpenSubdiv::kGregory || patch.desc.type == OpenSubdiv::kBoundaryGregory){
                    glUniform1i(uniformGregoryQuadOffset, patch.gregoryQuadOffsetBase + primitiveOffset*4);
                }
                glUniform1i(uniformLevelBase, patch.levelBase + primitiveOffset);

                glDrawElements(primType, drawRanges[j].numIndices, GL_UNSIGNED_INT,
                               (void *)(drawRanges[j].firstIndex * sizeof(unsigned int)));
            }
        }
        if (g_wire == 0) {
            glEnable(GL_CULL_FACE);
        }
    }

    glEndQuery(GL_PRIMITIVES_GENERATED);

    glBindVertexArray(0);

    glUseProgram(0);

    s.Stop();
    float drawCpuTime = float(s.GetElapsed() * 1000.0f);
    s.Start();

    glFinish();
    GLuint numPrimsGenerated = 0;
    glGetQueryObjectuiv(g_primQuery, GL_QUERY_RESULT, &numPrimsGenerated);

    s.Stop();
    float drawGpuTime = float(s.GetElapsed() * 1000.0f);

    if (g_hud.IsVisible()) {
        g_fpsTimer.Stop();
        g_totalTime += g_fpsTimer.GetElapsed();
        double fps = 1.0/g_fpsTimer.GetElapsed();
        g_fpsTimer.Start();

        float subdTableSize = g_farMesh->GetSubdivisionTables()->GetMemoryUsed()/1024.0f/1024.0f;
        float vboSize = g_mesh->GetVBOMemoryUsed()/1024.0f/1024.0f;

        g_hud.DrawString(10, -240, "Subd Table : %.2f MB", subdTableSize);
        g_hud.DrawString(10, -220, "Subd VBO   : %.2f MB", vboSize);
        g_hud.DrawString(10, -180, "Tess level : %d", g_tessLevel);
        g_hud.DrawString(10, -160, "Primitives : %d", numPrimsGenerated);
//        g_hud.DrawString(10, -140, "Vertices   : %d", g_cpuGLVertexBuffer->GetNumVertices());
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
mouse(GLFWwindow *, int button, int state) {
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

    glDeleteVertexArrays(1, &g_vao);

    delete g_farMesh;
    g_farMesh = NULL;

    delete g_mesh;
    g_mesh = NULL;

#ifdef OPENSUBDIV_HAS_CUDA
    cudaDeviceReset();
#endif
#ifdef OPENSUBDIV_HAS_OPENCL
    uninitCL(g_clContext, g_clQueue);
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
static void toggleFullScreen() {
#if 0
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
#endif
}

//------------------------------------------------------------------------------
static void
#if GLFW_VERSION_MAJOR>=3
keyboard(GLFWwindow *, int key, int event) {
#else
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
        case '-':  g_tessLevel = std::max(g_tessLevelMin, g_tessLevel-1); rebuildOsd(); break;
        case 'I': g_modelCount = std::max(g_modelCount/2, 1); rebuildFar(); break;
        case 'O': g_modelCount = std::min(g_modelCount*2, MAX_MODELS); rebuildFar(); break;
        case GLFW_KEY_ESC: g_hud.SetVisible(!g_hud.IsVisible()); break;
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
        cudaGLSetGLDevice( cutGetMaxGflopsDeviceId() );
    }
#endif

    rebuildOsd();
}

static void
callbackLevel(int l)
{
    g_level = l;
    rebuildFar();
}

static void
callbackModel(int m)
{
    if (m < 0)
        m = 0;

    if (m >= (int)g_defaultShapes.size())
        m = (int)g_defaultShapes.size() - 1;

    g_currentShape = m;

    rebuildFar();
}

static void
callbackCheckBox(bool checked, int button)
{
    switch(button) {
    case HUD_CB_DRAW_AT_ONCE:
        g_drawAtOnce = checked;
        break;
    case HUD_CB_ADAPTIVE:
        if (OpenSubdiv::OsdGLDrawContext::SupportsAdaptiveTessellation()) {
            g_adaptive = checked;
            rebuildFar();
        }
        break;
    case HUD_CB_ANIMATE_VERTICES:
        g_moveScale = checked;
        break;
    case HUD_CB_DISPLAY_PATCH_COLOR:
        g_displayPatchColor = checked;
        break;
    case HUD_CB_VIEW_LOD:
        g_screenSpaceTess = checked;
        break;
    case HUD_CB_FREEZE:
        g_freeze = checked;
        break;
    }
}

static void
initHUD()
{
    g_hud.Init(g_width, g_height);

    g_hud.AddRadioButton(0, "CPU (K)", g_kernel == kCPU, 10, 10, callbackKernel, kCPU, 'k');
#ifdef OPENSUBDIV_HAS_OPENMP
    g_hud.AddRadioButton(0, "OPENMP", g_kernel == kOPENMP, 10, 30, callbackKernel, kOPENMP, 'k');
#endif
#ifdef OPENSUBDIV_HAS_GCD
    g_hud.AddRadioButton(0, "GCD", g_kernel == kGCD, 10, 30, callbackKernel, kGCD, 'k');
#endif
#ifdef OPENSUBDIV_HAS_CUDA
    g_hud.AddRadioButton(0, "CUDA",   g_kernel == kCUDA, 10, 50, callbackKernel, kCUDA, 'k');
#endif
#ifdef OPENSUBDIV_HAS_OPENCL
    g_hud.AddRadioButton(0, "OPENCL", g_kernel == kCL, 10, 70, callbackKernel, kCL, 'k');
#endif
#ifdef OPENSUBDIV_HAS_GLSL_TRANSFORM_FEEDBACK
    g_hud.AddRadioButton(0, "GLSL TransformFeedback",   g_kernel == kGLSL, 10, 90, callbackKernel, kGLSL, 'k');
#endif
#ifdef OPENSUBDIV_HAS_GLSL_COMPUTE
    // Must also check at run time for OpenGL 4.3
//    if (GLEW_VERSION_4_3) {
        g_hud.AddRadioButton(0, "GLSL Compute", g_kernel == kGLSLCompute, 10, 110, callbackKernel, kGLSLCompute, 'k');
//    }
#endif

    g_hud.AddRadioButton(1, "Wire (W)",    g_wire == 0,  200, 10, callbackWireframe, 0, 'w');
    g_hud.AddRadioButton(1, "Shaded",      g_wire == 1, 200, 30, callbackWireframe, 1, 'w');
    g_hud.AddRadioButton(1, "Wire+Shaded", g_wire == 2, 200, 50, callbackWireframe, 2, 'w');

    g_hud.AddCheckBox("Draw at once (A)", g_drawAtOnce != 0, 350, 10, callbackCheckBox, HUD_CB_DRAW_AT_ONCE, 'a');
    g_hud.AddCheckBox("Animate vertices (M)", g_moveScale != 0, 350, 30, callbackCheckBox, HUD_CB_ANIMATE_VERTICES, 'm');
    g_hud.AddCheckBox("Patch Color (P)",      true, 350, 50, callbackCheckBox, HUD_CB_DISPLAY_PATCH_COLOR, 'p');
    g_hud.AddCheckBox("Screen space LOD (V)", g_screenSpaceTess != 0, 350, 70, callbackCheckBox, HUD_CB_VIEW_LOD, 'v');
    g_hud.AddCheckBox("Freeze (spc)",         false, 350, 90, callbackCheckBox, HUD_CB_FREEZE, ' ');

    if (OpenSubdiv::OsdGLDrawContext::SupportsAdaptiveTessellation())
        g_hud.AddCheckBox("Adaptive (`)", g_adaptive!=0, 10, 150, callbackCheckBox, HUD_CB_ADAPTIVE, '`');

    for (int i = 1; i < 11; ++i) {
        char level[16];
        sprintf(level, "Lv. %d", i);
        g_hud.AddRadioButton(3, level, i==g_level, 10, 170+i*20, callbackLevel, i, '0'+(i%10));
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
            GLFWvidmode vidmode = glfwGetVideoMode(g_primary);
            g_width = vidmode.width;
            g_height = vidmode.height;
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
#endif

#if not defined(__APPLE__)
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
    callbackModel(g_currentShape);

    g_fpsTimer.Start();
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
