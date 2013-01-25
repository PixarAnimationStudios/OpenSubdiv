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
    #include <stdlib.h>
    #include <GL/glew.h>
    #if defined(WIN32)
        #include <GL/wglew.h>
    #endif
    #include <GL/glut.h>
#endif

#include "../../regression/common/mutex.h" // XXX

#include <osd/error.h>
#include <osd/vertex.h>
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

#include <common/shape_utils.h>
#include "../common/stopwatch.h"
#include "../common/simple_math.h"
#include "../common/gl_hud.h"

static const char *shaderSource =
#include "shader.inc"
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
                  kCUDA = 2,
                  kCL = 3,
                  kGLSL = 4,
                  kGLSLCompute = 5 };

struct SimpleShape {
    std::string  name;
    Scheme       scheme;
    char const * data;

    SimpleShape() { }
    SimpleShape( char const * idata, char const * iname, Scheme ischeme )
        : name(iname), scheme(ischeme), data(idata) { }
};

std::vector<SimpleShape> g_defaultShapes;

int g_currentShape = 0;

int   g_frame = 0,
      g_repeatCount = 0;

// GUI variables
int   g_fullscreen=0,
      g_freeze = 0,
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

GLuint g_primQuery = 0;

std::vector<int> g_coarseEdges;
std::vector<float> g_coarseEdgeSharpness;
std::vector<float> g_coarseVertexSharpness;

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
static void
initializeShapes( ) {

#include <shapes/bilinear_cube.h>
    g_defaultShapes.push_back(SimpleShape(bilinear_cube, "bilinear_cube", kBilinear));

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
        //float move = 0.05f*cosf(p[0]*20+g_frame*0.01f);
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

    g_mesh->UpdateVertexBuffer(&vertex[0], nverts);

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
createOsdMesh( const char * shape, int level, int kernel, Scheme scheme=kCatmark ) {

    // generate Hbr representation from "obj" description
    OsdHbrMesh * hmesh = simpleHbr<OpenSubdiv::OsdVertex>(shape, scheme, g_orgPositions);

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

    if (kernel == kCPU) {
        g_mesh = new OpenSubdiv::OsdMesh<OpenSubdiv::OsdCpuGLVertexBuffer,
                                         OpenSubdiv::OsdCpuComputeController,
                                         OpenSubdiv::OsdGLDrawContext>(hmesh, 6, level, bits);
#ifdef OPENSUBDIV_HAS_OPENMP
    } else if (kernel == kOPENMP) {
        g_mesh = new OpenSubdiv::OsdMesh<OpenSubdiv::OsdCpuGLVertexBuffer,
                                         OpenSubdiv::OsdOmpComputeController,
                                         OpenSubdiv::OsdGLDrawContext>(hmesh, 6, level, bits);
#endif
#ifdef OPENSUBDIV_HAS_OPENCL
    } else if(kernel == kCL) {
        g_mesh = new OpenSubdiv::OsdMesh<OpenSubdiv::OsdCLGLVertexBuffer,
                                         OpenSubdiv::OsdCLComputeController,
                                         OpenSubdiv::OsdGLDrawContext>(hmesh, 6, level, bits, g_clContext, g_clQueue);
#endif
#ifdef OPENSUBDIV_HAS_CUDA
    } else if(kernel == kCUDA) {
        g_mesh = new OpenSubdiv::OsdMesh<OpenSubdiv::OsdCudaGLVertexBuffer,
                                         OpenSubdiv::OsdCudaComputeController,
                                         OpenSubdiv::OsdGLDrawContext>(hmesh, 6, level, bits);
#endif
#ifdef OPENSUBDIV_HAS_GLSL_TRANSFORM_FEEDBACK
    } else if(kernel == kGLSL) {
        g_mesh = new OpenSubdiv::OsdMesh<OpenSubdiv::OsdGLVertexBuffer,
                                         OpenSubdiv::OsdGLSLTransformFeedbackComputeController,
                                         OpenSubdiv::OsdGLDrawContext>(hmesh, 6, level, bits);
#endif
#ifdef OPENSUBDIV_HAS_GLSL_COMPUTE
    } else if(kernel == kGLSLCompute) {
        g_mesh = new OpenSubdiv::OsdMesh<OpenSubdiv::OsdGLVertexBuffer,
                                         OpenSubdiv::OsdGLSLComputeController,
                                         OpenSubdiv::OsdGLDrawContext>(hmesh, 6, level, bits);
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
}

//------------------------------------------------------------------------------
static void
fitFrame() {

    g_pan[0] = g_pan[1] = 0;
    g_dolly = g_size;
}

//------------------------------------------------------------------------------
static void
drawNormals() {

#if 0
    float * data=0;
    int datasize = g_vertexBuffer->GetNumVertices() * g_vertexBuffer->GetNumElements();

    data = new float[datasize];

    glBindBuffer(GL_ARRAY_BUFFER, g_vertexBuffer->BindVBO());

    glGetBufferSubData(GL_ARRAY_BUFFER,0,datasize*sizeof(float),data);

    glDisable(GL_LIGHTING);
    glColor3f(0.0f, 0.0f, 0.5f);
    glBegin(GL_LINES);

    int start = g_farmesh->GetSubdivisionTables()->GetFirstVertexOffset(g_level) *
                g_vertexBuffer->GetNumElements();

    for (int i=start; i<datasize; i+=6) {
        glVertex3f( data[i  ],
                    data[i+1],
                    data[i+2] );

        float n[3] = { data[i+3], data[i+4], data[i+5] };
        normalize(n);

        glVertex3f( data[i  ]+n[0]*0.2f,
                    data[i+1]+n[1]*0.2f,
                    data[i+2]+n[2]*0.2f );
    }
    glEnd();

    delete [] data;
#endif
}

static inline void
setSharpnessColor(float s)
{
    //  0.0       2.0       4.0
    // green --- yellow --- red
    float r = std::min(1.0f, s * 0.5f);
    float g = std::min(1.0f, 2.0f - s*0.5f);
    glColor3f(r, g, 0.0f);
}

static void
drawCageEdges() {

    glDisable(GL_LIGHTING);
    glLineWidth(2.0f);
    glBegin(GL_LINES);
    for(int i=0; i<(int)g_coarseEdges.size(); i+=2) {
        setSharpnessColor(g_coarseEdgeSharpness[i/2]);
        glVertex3fv(&g_positions[g_coarseEdges[i]*3]);
        glVertex3fv(&g_positions[g_coarseEdges[i+1]*3]);
    }
    glEnd();
    glLineWidth(1.0f);
} 

static void
drawCageVertices() {

    glDisable(GL_LIGHTING);
    glPointSize(10.0f);
    glBegin(GL_POINTS);
    for(int i=0; i<(int)g_positions.size()/3; ++i) {
        setSharpnessColor(g_coarseVertexSharpness[i]);
        glVertex3fv(&g_positions[i*3]);
    }
    glEnd();
    glPointSize(1.0f);
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

//    sconfig->commonShader.AddDefine("OSD_ENABLE_PATCH_CULL");
//    sconfig->commonShader.AddDefine("OSD_ENABLE_SCREENSPACE_TESSELLATION");

    if (desc.first.type != OpenSubdiv::kNonPatch) {

        if (effect == kQuadWire) effect = kTriWire;
        if (effect == kQuadFill) effect = kTriFill;
        if (effect == kQuadLine) effect = kTriLine;
        sconfig->geometryShader.AddDefine("SMOOTH_NORMALS");

    } else {
        sconfig->vertexShader.source = shaderSource;
        sconfig->vertexShader.version = "#version 410\n";
        sconfig->vertexShader.AddDefine("VERTEX_SHADER");
    }
    assert(sconfig);

    sconfig->geometryShader.source = shaderSource;
    sconfig->geometryShader.version = "#version 410\n";
    sconfig->geometryShader.AddDefine("GEOMETRY_SHADER");

    sconfig->fragmentShader.source = shaderSource;
    sconfig->fragmentShader.version = "#version 410\n";
    sconfig->fragmentShader.AddDefine("FRAGMENT_SHADER");

    switch (effect) {
    case kQuadWire:
        sconfig->geometryShader.AddDefine("PRIM_QUAD");
        sconfig->geometryShader.AddDefine("GEOMETRY_OUT_WIRE");
        sconfig->fragmentShader.AddDefine("PRIM_QUAD");
        sconfig->fragmentShader.AddDefine("GEOMETRY_OUT_WIRE");
        break;
    case kQuadFill:
        sconfig->geometryShader.AddDefine("PRIM_QUAD");
        sconfig->geometryShader.AddDefine("GEOMETRY_OUT_FILL");
        sconfig->fragmentShader.AddDefine("PRIM_QUAD");
        sconfig->fragmentShader.AddDefine("GEOMETRY_OUT_FILL");
        break;
    case kQuadLine:
        sconfig->geometryShader.AddDefine("PRIM_QUAD");
        sconfig->geometryShader.AddDefine("GEOMETRY_OUT_LINE");
        sconfig->fragmentShader.AddDefine("PRIM_QUAD");
        sconfig->fragmentShader.AddDefine("GEOMETRY_OUT_LINE");
        break;
    case kTriWire:
        sconfig->geometryShader.AddDefine("PRIM_TRI");
        sconfig->geometryShader.AddDefine("GEOMETRY_OUT_WIRE");
        sconfig->fragmentShader.AddDefine("PRIM_TRI");
        sconfig->fragmentShader.AddDefine("GEOMETRY_OUT_WIRE");
        break;
    case kTriFill:
        sconfig->geometryShader.AddDefine("PRIM_TRI");
        sconfig->geometryShader.AddDefine("GEOMETRY_OUT_FILL");
        sconfig->fragmentShader.AddDefine("PRIM_TRI");
        sconfig->fragmentShader.AddDefine("GEOMETRY_OUT_FILL");
        break;
    case kTriLine:
        sconfig->geometryShader.AddDefine("PRIM_TRI");
        sconfig->geometryShader.AddDefine("GEOMETRY_OUT_LINE");
        sconfig->fragmentShader.AddDefine("PRIM_TRI");
        sconfig->fragmentShader.AddDefine("GEOMETRY_OUT_LINE");
        break;
    case kPoint:
        sconfig->geometryShader.AddDefine("PRIM_POINT");
        sconfig->fragmentShader.AddDefine("PRIM_POINT");
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
static GLuint
bindProgram(Effect effect, OpenSubdiv::OsdPatchArray const & patch)
{
    EffectDesc effectDesc(patch.desc, effect);
    EffectDrawRegistry::ConfigType *
        config = effectRegistry.GetDrawConfig(effectDesc);

    GLuint program = config->program;

    glUseProgram(program);

    // Update and bind transform state
    struct Transform {
        float ModelViewMatrix[16];
        float ProjectionMatrix[16];
        float ModelViewProjectionMatrix[16];
    } transformData;
    glGetFloatv(GL_MODELVIEW_MATRIX, transformData.ModelViewMatrix);
    glGetFloatv(GL_PROJECTION_MATRIX, transformData.ProjectionMatrix);
    multMatrix(transformData.ModelViewProjectionMatrix,
               transformData.ModelViewMatrix,
               transformData.ProjectionMatrix);

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

    checkGLErrors("bindProgram leave");

    return program;
}

//------------------------------------------------------------------------------
static void
display() {

    Stopwatch s;
    s.Start();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glViewport(0, 0, g_width, g_height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glDisable(GL_LIGHTING);
    glColor3f(1, 1, 1);
    glBegin(GL_QUADS);
    glColor3f(0.5f, 0.5f, 0.5f);
    glVertex3f(-1, -1, 1);
    glVertex3f( 1, -1, 1);
    glColor3f(0, 0, 0);
    glVertex3f( 1,  1, 1);
    glVertex3f(-1,  1, 1);
    glEnd();

    double aspect = g_width/(double)g_height;
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0, aspect, 0.01, 500.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(-g_pan[0], -g_pan[1], -g_dolly);
    glRotatef(g_rotate[1], 1, 0, 0);
    glRotatef(g_rotate[0], 0, 1, 0);
    glTranslatef(-g_center[0], -g_center[2], g_center[1]); // z-up model
    glRotatef(-90, 1, 0, 0); // z-up model

    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, g_mesh->BindVertexBuffer());

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof (GLfloat) * 6, 0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof (GLfloat) * 6, (float*)12);

    OpenSubdiv::OsdPatchArrayVector const & patches = g_mesh->GetDrawContext()->patchArrays;

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_mesh->GetDrawContext()->patchIndexBuffer);

    // cv drawing
    if (g_drawPatchCVs) {
        glPointSize(3.0);

        bindProgram(kPoint, OpenSubdiv::OsdPatchArray());

        for (int i=0; i<(int)patches.size(); ++i) {
            OpenSubdiv::OsdPatchArray const & patch = patches[i];

            glDrawElements(GL_POINTS,
                           patch.numIndices, GL_UNSIGNED_INT,
                           (void *)(patch.firstIndex * sizeof(unsigned int)));
        }
    }

    // patch drawing
    int patchTypeCount[9]; // enum OsdPatchType (osd/drawCountext.h)
    int transitionPatchTypeCount[3][5][4];
    memset(patchTypeCount, 0, sizeof(patchTypeCount));
    memset(transitionPatchTypeCount, 0, sizeof(transitionPatchTypeCount));

    // primitive counting
    glBeginQuery(GL_PRIMITIVES_GENERATED, g_primQuery);

    for (int i=0; i<(int)patches.size(); ++i) {
        OpenSubdiv::OsdPatchArray const & patch = patches[i];

        OpenSubdiv::OsdPatchType patchType = patch.desc.type;
        int patchPattern = patch.desc.pattern;
        int patchRotation = patch.desc.rotation;

        if (patch.desc.subpatch == 0) {
            if (patchType == OpenSubdiv::kTransitionRegular) 
                transitionPatchTypeCount[0][patchPattern][patchRotation] += patch.numIndices / patch.patchSize;
            else if (patchType == OpenSubdiv::kTransitionBoundary) 
                transitionPatchTypeCount[1][patchPattern][patchRotation] += patch.numIndices / patch.patchSize;
            else if (patchType == OpenSubdiv::kTransitionBoundary) 
                transitionPatchTypeCount[2][patchPattern][patchRotation] += patch.numIndices / patch.patchSize;
            else
                patchTypeCount[patchType] += patch.numIndices / patch.patchSize;
        }

        GLenum primType;

        if (g_mesh->GetDrawContext()->IsAdaptive()) {

            primType = GL_PATCHES;
            glPatchParameteri(GL_PATCH_VERTICES, patch.patchSize);

        } else {
            if (g_scheme == kLoop) {
                primType = GL_TRIANGLES;
            } else {
                primType = GL_LINES_ADJACENCY; // GL_QUADS is deprecated
            }
        }

        GLuint program = bindProgram(GetEffect(), patch);

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

    glEndQuery(GL_PRIMITIVES_GENERATED);

    GLuint numPrimsGenerated = 0;
    glGetQueryObjectuiv(g_primQuery, GL_QUERY_RESULT, &numPrimsGenerated);

    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);

    glUseProgram(0);

    if (g_drawNormals)
        drawNormals();
    
    if (g_drawCageEdges)
        drawCageEdges();

    if (g_drawCageVertices)
        drawCageVertices();

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glDisableClientState(GL_VERTEX_ARRAY);

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
                         patchTypeCount[OpenSubdiv::kNonPatch]);
        g_hud.DrawString(x, -340, "Regular          : %d",
                         patchTypeCount[OpenSubdiv::kRegular]);
        g_hud.DrawString(x, -320, "Boundary         : %d",
                         patchTypeCount[OpenSubdiv::kBoundary]);
        g_hud.DrawString(x, -300, "Corner           : %d",
                         patchTypeCount[OpenSubdiv::kCorner]);
        g_hud.DrawString(x, -280, "Gregory          : %d",
                         patchTypeCount[OpenSubdiv::kGregory]);
        g_hud.DrawString(x, -260, "Boundary Gregory : %d",
                         patchTypeCount[OpenSubdiv::kBoundaryGregory]);
        g_hud.DrawString(x, -240, "Trans. Regular   : %d %d %d %d %d",
                         transitionPatchTypeCount[0][0][0],
                         transitionPatchTypeCount[0][1][0],
                         transitionPatchTypeCount[0][2][0],
                         transitionPatchTypeCount[0][3][0],
                         transitionPatchTypeCount[0][4][0]);
        for (int i=0; i < 5; i++) 
            g_hud.DrawString(x, -220+i*20, "Trans. Boundary%d : %d %d %d %d", i,
                             transitionPatchTypeCount[1][i][0],
                             transitionPatchTypeCount[1][i][1],
                             transitionPatchTypeCount[1][i][2],
                             transitionPatchTypeCount[1][i][3]);
        for (int i=0; i < 5; i++)
            g_hud.DrawString(x, -100+i*20, "Trans. Corner%d  : %d %d %d %d", i,
                             transitionPatchTypeCount[2][i][0],
                             transitionPatchTypeCount[2][i][1],
                             transitionPatchTypeCount[2][i][2],
                             transitionPatchTypeCount[2][i][3]);

        g_hud.DrawString(10, -180, "Tess level : %d", g_tessLevel);
        g_hud.DrawString(10, -160, "Primitives : %d", numPrimsGenerated);
        g_hud.DrawString(10, -140, "Vertices   : %d", g_mesh->GetNumVertices());
        g_hud.DrawString(10, -120, "Scheme     : %s", g_scheme==kBilinear ? "BILINEAR" : (g_scheme == kLoop ? "LOOP" : "CATMARK"));
        g_hud.DrawString(10, -100, "GPU Kernel : %.3f ms", g_gpuTime);
        g_hud.DrawString(10, -80,  "CPU Kernel : %.3f ms", g_cpuTime);
        g_hud.DrawString(10, -60,  "GPU Draw   : %.3f ms", drawGpuTime);
        g_hud.DrawString(10, -40,  "CPU Draw   : %.3f ms", drawCpuTime);
        g_hud.DrawString(10, -20,  "FPS        : %3.1f", fps);
    }

    g_hud.Flush();

    glFinish();
    glutSwapBuffers();
    glFinish();

    checkGLErrors("display leave");
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

    if (button == 0 && state == 1 && g_hud.MouseClick(x, y)) return;

    if (button < 3) {
        g_prev_x = float(x);
        g_prev_y = float(y);
        g_mbutton[button] = !state;
    }
}

//------------------------------------------------------------------------------
static void
quit() {

    glDeleteQueries(1, &g_primQuery);

    if (g_mesh)
        delete g_mesh;

#ifdef OPENSUBDIV_HAS_CUDA
    cudaDeviceReset();
#endif

#ifdef OPENSUBDIV_HAS_OPENCL
    uninitCL(g_clContext, g_clQueue);
#endif

    exit(0);
}

//------------------------------------------------------------------------------
static void
reshape(int width, int height) {

    g_width = width;
    g_height = height;
    
    g_hud.Rebuild(width, height);
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
static void
keyboard(unsigned char key, int x, int y) {

    if (g_hud.KeyDown(key)) return;

    switch (key) {
        case 'q': quit();
        case 'f': fitFrame(); break;
        case '\t': toggleFullScreen(); break;
        case '+':
        case '=':  g_tessLevel++; break;
        case '-':  g_tessLevel = std::max(g_tessLevelMin, g_tessLevel-1); break;
        case 0x1b: g_hud.SetVisible(!g_hud.IsVisible()); break;
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
    g_hud.AddRadioButton(0, "GLSL TransformFeedback",   false, 10, 90, callbackKernel, kGLSL, 'k');
#endif
#ifdef OPENSUBDIV_HAS_GLSL_COMPUTE
    // Must also check at run time for OpenGL 4.3
    if (GLEW_VERSION_4_3) {
        g_hud.AddRadioButton(0, "GLSL Compute",   false, 10, 110, callbackKernel, kGLSLCompute, 'k');
    }
#endif

    g_hud.AddRadioButton(1, "Wire (W)",    g_wire == 0,  200, 10, callbackWireframe, 0, 'w');
    g_hud.AddRadioButton(1, "Shaded",      g_wire == 1, 200, 30, callbackWireframe, 1, 'w');
    g_hud.AddRadioButton(1, "Wire+Shaded", g_wire == 2, 200, 50, callbackWireframe, 2, 'w');

    g_hud.AddCheckBox("Cage Edges (H)",    true,  350, 10, callbackDisplayCageEdges, 0, 'h');
    g_hud.AddCheckBox("Cage Verts (J)", false, 350, 30, callbackDisplayCageVertices, 0, 'j');
    g_hud.AddCheckBox("Patch CVs (L)", false, 350, 50, callbackDisplayPatchCVs, 0, 'l');
    g_hud.AddCheckBox("Show normal vector (E)", false, 350, 70, callbackDisplayNormal, 0, 'e');
    g_hud.AddCheckBox("Animate vertices (M)", g_moveScale != 0, 350, 90, callbackAnimate, 0, 'm');
    g_hud.AddCheckBox("Patch Color (P)",   true, 350, 110, callbackDisplayPatchColor, 0, 'p');
    g_hud.AddCheckBox("Freeze (spc)", false, 350, 130, callbackFreeze, 0, ' ');

    g_hud.AddCheckBox("Adaptive (`)", true, 10, 150, callbackAdaptive, 0, '`');

    for (int i = 1; i < 11; ++i) {
        char level[16];
        sprintf(level, "Lv. %d", i);
        g_hud.AddRadioButton(3, level, i==2, 10, 170+i*20, callbackLevel, i, '0'+(i%10));
    }

    for(int i = 0; i < (int)g_defaultShapes.size(); ++i){
        g_hud.AddRadioButton(4, g_defaultShapes[i].name.c_str(), i==0, -220, 10+i*16, callbackModel, i, 'n');
    }

    callbackModel(g_currentShape);
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
}

//------------------------------------------------------------------------------
static void
idle() {

    if (not g_freeze)
        g_frame++;

    updateGeom();
    glutPostRedisplay();

    if (g_repeatCount != 0 and g_frame >= g_repeatCount)
        quit();
}

//------------------------------------------------------------------------------
static void
callbackError(OpenSubdiv::OsdErrorType err, const char *message)
{
    printf("OsdError: %d\n", err);
    printf(message);
}

//------------------------------------------------------------------------------
int main(int argc, char ** argv)
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA |GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(g_width, g_height);
    glutCreateWindow("OpenSubdiv glutViewer");
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutMouseFunc(mouse);
    glutKeyboardFunc(keyboard);
    glutMotionFunc(motion);

    std::string str;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-d"))
            g_level = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-c"))
            g_repeatCount = atoi(argv[++i]);
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

    glewInit();

    initGL();

#ifdef WIN32
    wglSwapIntervalEXT(0);
#endif

    initHUD();

    glutIdleFunc(idle);
    glutMainLoop();

    quit();
}

//------------------------------------------------------------------------------
