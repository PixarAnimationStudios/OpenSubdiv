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

#include <osdutil/batch.h>
#ifdef OPENSUBDIV_HAS_OPENCL
    #include <osdutil/batchCL.h>
#endif
#include <osdutil/drawItem.h>
#include <osdutil/drawController.h>
#include "delegate.h"
#include "effect.h"

#include <common/shape_utils.h>
#include "../common/stopwatch.h"
#include "../common/simple_math.h"
#include "../common/gl_hud.h"

#include <cfloat>
#include <vector>
#include <fstream>
#include <sstream>

template <class CONTROLLER> class Controller
{
public:
    static CONTROLLER *GetInstance() {
        static CONTROLLER *instance = NULL;
        if (not instance) instance = new CONTROLLER();
        return instance;
    }
};

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
#endif

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

enum HudCheckBox { HUD_CB_BATCHING,
                   HUD_CB_ADAPTIVE,
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

OpenSubdiv::OsdUtilMeshBatchBase<MyDrawContext> *g_batch = NULL;

MyDrawDelegate g_drawDelegate;
MyEffect g_effect;

std::vector<SimpleShape> g_defaultShapes;

int g_currentShape = 0;

int   g_frame = 0,
      g_repeatCount = 0;

// GUI variables
int   g_fullscreen = 0,
      g_freeze = 0,
      g_displayStyle = kWireShaded,
      g_adaptive = 1,
      g_batching = 1,
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
#define MAX_MODELS 600
int g_modelCount = 4;
int g_moveModels = g_modelCount;

GLuint g_queries[2] = {0, 0};

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
#include "shapes.h"

static void
updateGeom(bool forceAll) {

    float r = (float)sin(g_totalTime);

    for (int j = 0; j < g_modelCount * g_modelCount; ++j) {
        if (forceAll == false && j >= g_moveModels) break;
        int nverts = (int)g_positions[j].size()/3;

        std::vector<float> vertex, varying;
        vertex.reserve(nverts * 3);

        if (g_displayStyle == kVaryingColor)
            varying.reserve(nverts * 3);

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
            vertex.push_back(v[0]);
            vertex.push_back(v[1]);
            vertex.push_back(v[2]);

            if (g_displayStyle == kVaryingColor) {
                varying.push_back(p[2]);
                varying.push_back(p[1]);
                varying.push_back(p[0]);
            }

            p += 3;
        }
        
        g_batch->UpdateCoarseVertices(j, &vertex[0], nverts);

        if (g_displayStyle == kVaryingColor) {
            g_batch->UpdateCoarseVaryings(j, &varying[0], nverts);
        }
    }

    Stopwatch s;
    s.Start();

    g_batch->FinalizeUpdate();

    s.Stop();
    g_cpuTime = float(s.GetElapsed() * 1000.0f);
    s.Start();

    if (g_kernel == kCPU) Controller<OpenSubdiv::OsdCpuComputeController>::GetInstance()->Synchronize();
#ifdef OPENSUBDIV_HAS_OPENMP
    else if (g_kernel == kOPENMP) Controller<OpenSubdiv::OsdOmpComputeController>::GetInstance()->Synchronize();
#endif
#ifdef OPENSUBDIV_HAS_OPENCL
    else if (g_kernel == kCL) Controller<OpenSubdiv::OsdCLComputeController>::GetInstance()->Synchronize();
#endif
#ifdef OPENSUBDIV_HAS_CUDA
    else if (g_kernel == kCUDA) Controller<OpenSubdiv::OsdCudaComputeController>::GetInstance()->Synchronize();
#endif
#ifdef OPENSUBDIV_HAS_GLSL_TRANSFORM_FEEDBACK
    else if (g_kernel == kGLSL) Controller<OpenSubdiv::OsdGLSLTransformFeedbackComputeController>::GetInstance()->Synchronize();
#endif

    s.Stop();
    g_gpuTime = float(s.GetElapsed() * 1000.0f);
}

//------------------------------------------------------------------------------
static OpenSubdiv::FarMesh<OpenSubdiv::OsdVertex> *
createFarMesh( const char * shape, int level, bool adaptive, Scheme scheme=kCatmark ) {

    checkGLErrors("create osd enter");
    // generate Hbr representation from "obj" description
    std::vector<float> positions;
    OsdHbrMesh * hmesh = simpleHbr<OpenSubdiv::OsdVertex>(shape, scheme, positions,
                                                          g_displayStyle == kFaceVaryingColor);

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
    OpenSubdiv::FarMesh<OpenSubdiv::OsdVertex> *farMesh = meshFactory.Create(true);

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
rebuild()
{
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

    delete g_batch;
    g_batch = NULL;

    int numVertexElements = 3;
    int numVaryingElements = g_displayStyle == kVaryingColor ? 3 : 0;
    bool requireFVarData = (g_displayStyle == kFaceVaryingColor);

    // create multimesh batch
    if (g_kernel == kCPU) {
        g_batch = OpenSubdiv::OsdUtilMeshBatch<OpenSubdiv::OsdCpuGLVertexBuffer,
            MyDrawContext, OpenSubdiv::OsdCpuComputeController>::Create(
            Controller<OpenSubdiv::OsdCpuComputeController>::GetInstance(),
            farMeshes, numVertexElements, numVaryingElements, 0, requireFVarData);
#ifdef OPENSUBDIV_HAS_OPENMP
    } else if (g_kernel == kOPENMP) {
        g_batch = OpenSubdiv::OsdUtilMeshBatch<OpenSubdiv::OsdCpuGLVertexBuffer,
            MyDrawContext,
            OpenSubdiv::OsdOmpComputeController>::Create(
            Controller<OpenSubdiv::OsdOmpComputeController>::GetInstance(),
            farMeshes, numVertexElements, numVaryingElements, 0, requireFVarData);
#endif
#ifdef OPENSUBDIV_HAS_OPENCL
    } else if (g_kernel == kCL) {
        g_batch = OpenSubdiv::OsdUtilMeshBatch<OpenSubdiv::OsdCLGLVertexBuffer,
            MyDrawContext, OpenSubdiv::OsdCLComputeController>::Create(
            Controller<OpenSubdiv::OsdCLComputeController>::GetInstance(),
            farMeshes, numVertexElements, numVaryingElements, 0, requireFVarData);
#endif
#ifdef OPENSUBDIV_HAS_CUDA
    } else if (g_kernel == kCUDA) {
        g_batch = OpenSubdiv::OsdUtilMeshBatch<OpenSubdiv::OsdCudaGLVertexBuffer,
            MyDrawContext, OpenSubdiv::OsdCudaComputeController>::Create(
            Controller<OpenSubdiv::OsdCudaComputeController>::GetInstance(),
            farMeshes, numVertexElements, numVaryingElements, 0, requireFVarData);
#endif
#ifdef OPENSUBDIV_HAS_GLSL_TRANSFORM_FEEDBACK
    } else if (g_kernel == kGLSL) {
        g_batch = OpenSubdiv::OsdUtilMeshBatch<OpenSubdiv::OsdGLVertexBuffer,
            MyDrawContext, OpenSubdiv::OsdGLSLTransformFeedbackComputeController>::Create(
            Controller<OpenSubdiv::OsdGLSLTransformFeedbackComputeController>::GetInstance(),
            farMeshes, numVertexElements, numVaryingElements, 0, requireFVarData);
#endif
    } else {
        assert(false);
    }

    assert(g_batch);

    g_scheme = scheme;

    for (size_t i = 0; i < farMeshes.size(); ++i) {
        delete farMeshes[i];
    }

    g_tessLevelMin = 0;

    g_tessLevel = std::max(g_tessLevel,g_tessLevelMin);

    updateGeom(true);
}

//------------------------------------------------------------------------------
static void
fitFrame() {

    g_pan[0] = g_pan[1] = 0;
    g_dolly = g_size;
}

//------------------------------------------------------------------------------
static void
display() {

    // set effect
    g_effect.displayStyle = g_displayStyle;
    g_effect.screenSpaceTess = (g_screenSpaceTess != 0);
    g_effect.displayPatchColor = (g_displayPatchColor != 0);

    // prepare view matrix
    float aspect = g_width/(float)g_height;
    float modelview[16], projection[16];
    identity(modelview);
    translate(modelview, -g_pan[0], -g_pan[1], -g_dolly);
    rotate(modelview, g_rotate[1], 1, 0, 0);
    rotate(modelview, g_rotate[0], 0, 1, 0);
    rotate(modelview, -90, 1, 0, 0);
    translate(modelview, -g_center[0], -g_center[1], -g_center[2]);
    perspective(projection, 45.0f, aspect, 0.01f, 500.0f);

    g_effect.SetMatrix(modelview, projection);
    g_effect.SetTessLevel((float)(1 << g_tessLevel));
    g_effect.SetLighting();

    // -----------------------------------------------------------------------
    // prepare draw items

    Stopwatch s;
    s.Start();

    OpenSubdiv::OsdUtilDrawItem<MyDrawDelegate::EffectHandle, MyDrawContext>::Collection items;
    OpenSubdiv::OsdUtilDrawItem<MyDrawDelegate::EffectHandle, MyDrawContext>::Collection cachedDrawItems;

    int numModels = g_modelCount*g_modelCount;
    items.reserve(numModels);

    for (int i = 0; i < numModels; ++i) {
        // Here, client can pack arbitrary mesh and effect into drawItems.

        items.push_back(OpenSubdiv::OsdUtilDrawItem<MyDrawDelegate::EffectHandle, MyDrawContext>(
                            g_batch, &g_effect, g_batch->GetPatchArrays(i)));
    }

    if (g_batching) {
        // create cached draw items
        OpenSubdiv::OsdUtil::OptimizeDrawItem(items, cachedDrawItems, &g_drawDelegate);
    }

    s.Stop();
    float prepCpuTime = float(s.GetElapsed() * 1000.0f);

    // -----------------------------------------------------------------------
    // draw items
    s.Start();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glViewport(0, 0, g_width, g_height);

    // primitive counting
    glBeginQuery(GL_PRIMITIVES_GENERATED, g_queries[0]);
#if defined(GL_VERSION_3_3)
    glBeginQuery(GL_TIME_ELAPSED, g_queries[1]);
#endif
    g_drawDelegate.ResetNumDrawCalls();

    if (g_displayStyle == kWire) glDisable(GL_CULL_FACE);

    if (g_batching) {
        OpenSubdiv::OsdUtil::DrawCollection(cachedDrawItems, &g_drawDelegate);
    } else {
        OpenSubdiv::OsdUtil::DrawCollection(items, &g_drawDelegate);
    }

    if (g_displayStyle == kWire) glEnable(GL_CULL_FACE);

    glEndQuery(GL_PRIMITIVES_GENERATED);
#if defined(GL_VERSION_3_3)
    glEndQuery(GL_TIME_ELAPSED);
#endif

    s.Stop();
    float drawCpuTime = float(s.GetElapsed() * 1000.0f);

    // -----------------------------------------------------------------------

    GLuint numPrimsGenerated = 0;
    GLuint timeElapsed = 0;
    glGetQueryObjectuiv(g_queries[0], GL_QUERY_RESULT, &numPrimsGenerated);
#if defined(GL_VERSION_3_3)
    glGetQueryObjectuiv(g_queries[1], GL_QUERY_RESULT, &timeElapsed);
#endif
    float drawGpuTime = timeElapsed / 1000.0f / 1000.0f;

    if (g_hud.IsVisible()) {
        g_fpsTimer.Stop();
        g_totalTime += g_fpsTimer.GetElapsed();
        double fps = 1.0/g_fpsTimer.GetElapsed();
        g_fpsTimer.Start();
        g_hud.DrawString(10, -200, "Draw Calls : %d", g_drawDelegate.GetNumDrawCalls());
        g_hud.DrawString(10, -180, "Tess level : %d", g_tessLevel);
        g_hud.DrawString(10, -160, "Primitives : %d", numPrimsGenerated);
        g_hud.DrawString(10, -120, "GPU Kernel : %.3f ms", g_gpuTime);
        g_hud.DrawString(10, -100, "CPU Kernel : %.3f ms", g_cpuTime);
        g_hud.DrawString(10, -80,  "GPU Draw   : %.3f ms", drawGpuTime);
        g_hud.DrawString(10, -60,  "CPU Draw   : %.3f ms", drawCpuTime);
        g_hud.DrawString(10, -40,  "CPU Prep   : %.3f ms", prepCpuTime);
        g_hud.DrawString(10, -20,  "FPS        : %3.1f", fps);

        g_hud.Flush();
    }

    checkGLErrors("display leave");
    glFinish();
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

    glDeleteQueries(2, g_queries);

    delete g_batch;
    g_batch = NULL;

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
        case '+':  
        case '=': g_tessLevel++; break;
        case '-': g_tessLevel = std::max(g_tessLevelMin, g_tessLevel-1); break;
        case '.': g_moveModels = std::max(g_moveModels*2, 1); break;
        case ',': g_moveModels = std::max(g_moveModels/2, 0); break;
        case 'I': g_modelCount = std::max(g_modelCount/2, 1); rebuild(); break;
        case 'O': g_modelCount = std::min(g_modelCount*2, MAX_MODELS); rebuild(); break;
        case GLFW_KEY_ESCAPE: g_hud.SetVisible(!g_hud.IsVisible()); break;
    }
}

//------------------------------------------------------------------------------
static void
callbackDisplayStyle(int b)
{
    if (g_displayStyle == kVaryingColor or b == kVaryingColor or
        g_displayStyle == kFaceVaryingColor or b == kFaceVaryingColor) {
        // need to rebuild for varying reconstruct
        g_displayStyle = b;
        rebuild();
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

    rebuild();
}

static void
callbackLevel(int l)
{
    g_level = l;
    rebuild();
}

static void
callbackModel(int m)
{
    if (m < 0)
        m = 0;

    if (m >= (int)g_defaultShapes.size())
        m = (int)g_defaultShapes.size() - 1;

    g_currentShape = m;

    rebuild();
}

static void
callbackCheckBox(bool checked, int button)
{
    switch(button) {
    case HUD_CB_BATCHING:
        g_batching = checked;
        break;
    case HUD_CB_ADAPTIVE:
        if (OpenSubdiv::OsdGLDrawContext::SupportsAdaptiveTessellation()) {
            g_adaptive = checked;
            rebuild();
        }
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

    g_hud.AddRadioButton(1, "Wire (W)", g_displayStyle == kWire,
                         200, 10, callbackDisplayStyle, kWire, 'w');
    g_hud.AddRadioButton(1, "Shaded", g_displayStyle == kShaded,
                         200, 30, callbackDisplayStyle, kShaded, 'w');
    g_hud.AddRadioButton(1, "Wire+Shaded", g_displayStyle == kWireShaded,
                         200, 50, callbackDisplayStyle, kWireShaded, 'w');
    g_hud.AddRadioButton(1, "Varying color", g_displayStyle == kVaryingColor,
                         200, 70, callbackDisplayStyle, kVaryingColor, 'w');
    g_hud.AddRadioButton(1, "Face varying color", g_displayStyle == kFaceVaryingColor,
                         200, 90, callbackDisplayStyle, kFaceVaryingColor, 'w');

    g_hud.AddCheckBox("Batching (B)", g_batching != 0, 350, 10, callbackCheckBox, HUD_CB_BATCHING, 'b');
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

    glGenQueries(2, g_queries);
}

//------------------------------------------------------------------------------
static void
idle() {

    if (not g_freeze) {
        g_frame++;
        updateGeom(false);
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

    static const char windowTitle[] = "OpenSubdiv glBatchViewer";
    
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
    }

    uninitGL();
    glfwTerminate();
}

//------------------------------------------------------------------------------
