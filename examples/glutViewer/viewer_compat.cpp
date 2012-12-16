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

#include <osd/glMesh.h>
OpenSubdiv::OsdGLMeshInterface *g_mesh;

#include <common/shape_utils.h>
#include "../common/stopwatch.h"
#include "../common/simple_math.h"

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
                  kGLSL = 4 };

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

// GLUT GUI variables
int   g_fullscreen=0,
      g_freeze = 0,
      g_wire = 2,
      g_drawCageEdges = 1,
      g_drawCageVertices = 0,
      g_drawNormals = 0,
      g_drawHUD = 1,
      g_mbutton[3] = {0, 0, 0};

float g_rotate[2] = {0, 0},
      g_prev_x = 0,
      g_prev_y = 0,
      g_dolly = 5,
      g_pan[2] = {0, 0},
      g_center[3] = {0, 0, 0},
      g_size = 0;

int   g_width = 1024,
      g_height = 1024;

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
int g_kernel = kCPU;
float g_moveScale = 0.0f;


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
//    g_defaultShapes.push_back(SimpleShape(bilinear_cube, "bilinear_cube", kBilinear));

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

    OpenSubdiv::OsdMeshBitset bits;
    bits.set(OpenSubdiv::MeshAdaptive, false);

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

    updateGeom();
}

//------------------------------------------------------------------------------
static void
fitFrame() {

    g_pan[0] = g_pan[1] = 0;
    g_dolly = g_size;
}

//------------------------------------------------------------------------------

#if _MSC_VER
    #define snprintf _snprintf
#endif

#define drawString(x, y, ...)               \
    { char line[1024]; \
      snprintf(line, 1024, __VA_ARGS__); \
      char *p = line; \
      glWindowPos2f(x, y); \
      while(*p) { glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, *p++); } }

//------------------------------------------------------------------------------
void
drawNormals() {

#if 0
    float * data=0;
    int datasize = g_osdmesh->GetTotalVertices() * g_vertexBuffer->GetNumElements();

    data = new float[datasize];

    glBindBuffer(GL_ARRAY_BUFFER, g_vertexBuffer->GetGpuBuffer());
    glGetBufferSubData(GL_ARRAY_BUFFER,0,datasize*sizeof(float),data);

    glDisable(GL_LIGHTING);
    glColor3f(0.0f, 0.0f, 0.5f);
    glBegin(GL_LINES);

    int start = g_osdmesh->GetFarMesh()->GetSubdivisionTables()->GetFirstVertexOffset(g_level) *
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

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

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

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_NORMAL_ARRAY);

    glBindBuffer(GL_ARRAY_BUFFER, g_mesh->BindVertexBuffer());

    glVertexPointer(3, GL_FLOAT, sizeof (GLfloat) * 6, 0);
    glNormalPointer(GL_FLOAT, sizeof (GLfloat) * 6, (float*)12);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_mesh->GetDrawContext()->patchIndexBuffer);
    int numIndices = g_mesh->GetDrawContext()->patchArrays[0].numIndices;

    GLenum primType = g_scheme == kLoop ? GL_TRIANGLES : GL_QUADS;

    glEnable(GL_LIGHTING);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    if (g_wire > 0) {
        glDrawElements(primType, numIndices, GL_UNSIGNED_INT, NULL);
    }
    glDisable(GL_LIGHTING);
    
    if (g_wire == 0 || g_wire == 2) {
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        if (g_wire == 2) {
            glColor4f(0, 0, 0.5, 1);
        } else {
            glColor4f(1, 1, 1, 1);
        }
        glDrawElements(primType, numIndices, GL_UNSIGNED_INT, NULL);
    }
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);

    if (g_drawNormals)
        drawNormals();
    
    if (g_drawCageEdges)
        drawCageEdges();

    if (g_drawCageVertices)
        drawCageVertices();

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    s.Stop();
    float drawCpuTime = float(s.GetElapsed() * 1000.0f);
    s.Start();
    glFinish();
    s.Stop();
    float drawGpuTime = float(s.GetElapsed() * 1000.0f);

    if (g_drawHUD) {
        g_fpsTimer.Stop();
        double fps = 1.0/g_fpsTimer.GetElapsed();
        g_fpsTimer.Start();

        glColor3f(1, 1, 1);
        drawString(10, 10,  "LEVEL = %d", g_level);
        drawString(10, 30,  "# of Vertices = %d", g_mesh->GetNumVertices());
        drawString(10, 50,  "KERNEL = %s", getKernelName(g_kernel));
        drawString(10, 70,  "FPS        = %3.1f", fps);
        drawString(10, 90,  "GPU Draw   = %.3f ms", drawGpuTime);
        drawString(10, 110, "CPU Draw   = %.3f ms", drawCpuTime);
        drawString(10, 130, "GPU Kernel = %.3f ms", g_gpuTime);
        drawString(10, 150, "CPU Kernel = %.3f ms", g_cpuTime);
        drawString(10, 170, "SUBDIVISION = %s", g_scheme==kBilinear ? "BILINEAR" : (g_scheme == kLoop ? "LOOP" : "CATMARK"));
        
        drawString(10, g_height-30, "w:   toggle wireframe");
        drawString(10, g_height-50, "e:   display normal vector");
        drawString(10, g_height-70, "m:   toggle vertex deforming");
        drawString(10, g_height-90, "h:   display cage edges");
        drawString(10, g_height-110, "j:   display cage verts");
        drawString(10, g_height-130, "n/p: change model");
        drawString(10, g_height-150, "1-7: subdivision level");
        drawString(10, g_height-170, "space: freeze/unfreeze time");
    }

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

    g_prev_x = float(x);
    g_prev_y = float(y);
    g_mbutton[button] = !state;
}

//------------------------------------------------------------------------------
static void
quit() {

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
}

//------------------------------------------------------------------------------
void kernelMenu(int k) {

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

//------------------------------------------------------------------------------
void
modelMenu(int m) {

    if (m < 0)
        m = 0;

    if (m >= (int)g_defaultShapes.size())
        m = g_defaultShapes.size() - 1;

    g_currentShape = m;

    glutSetWindowTitle( g_defaultShapes[m].name.c_str() );

    createOsdMesh( g_defaultShapes[m].data, g_level, g_kernel, g_defaultShapes[ g_currentShape ].scheme );
}

//------------------------------------------------------------------------------
void
levelMenu(int l) {

    g_level = l;

    createOsdMesh( g_defaultShapes[g_currentShape].data, g_level, g_kernel, g_defaultShapes[ g_currentShape ].scheme );
}

//------------------------------------------------------------------------------
void
menu(int m) {

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

    switch (key) {
        case 'q': quit();
        case ' ': g_freeze = (g_freeze+1)%2; break;
        case 'w': g_wire = (g_wire+1)%3; break;
        case 'e': g_drawNormals = (g_drawNormals+1)%2; break;
        case 'f': fitFrame(); break;
        case '\t': toggleFullScreen(); break;
        case 'm': g_moveScale = 1.0f - g_moveScale; break;
        case 'h': g_drawCageEdges = !g_drawCageEdges; break;
        case 'j': g_drawCageVertices = !g_drawCageVertices; break;
        case '1':
        case '2':
        case '3':
        case '4':
        case '5':
        case '6':
        case '7': levelMenu(key-'0'); break;
        case 'n': modelMenu(++g_currentShape); break;
        case 'p': modelMenu(--g_currentShape); break;
        case 0x1b: g_drawHUD = (g_drawHUD+1)%2; break;
    }
}

//------------------------------------------------------------------------------
static void
initGL() {

    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glEnable(GL_LIGHT0);
    glColor3f(1, 1, 1);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);

    GLfloat color[4] = {1, 1, 1, 1};
    GLfloat position[4] = {5, 5, 10, 1};
    GLfloat ambient[4] = {0.1f, 0.1f, 0.1f, 1.0f};
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
int main(int argc, char ** argv) {

    glutInit(&argc, argv);

    glutInitDisplayMode(GLUT_RGBA |GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(1024, 1024);
    glutCreateWindow("OpenSubdiv test");

    std::string str;
    if (argc > 1) {
        std::ifstream ifs(argv[1]);
        if (ifs) {
            std::stringstream ss;
            ss << ifs.rdbuf();
            ifs.close();
            str = ss.str();

            g_defaultShapes.push_back(SimpleShape(str.c_str(), argv[1], kCatmark));
        }
    }

    initializeShapes();

    int smenu = glutCreateMenu(modelMenu);
    for(int i = 0; i < (int)g_defaultShapes.size(); ++i){
        glutAddMenuEntry( g_defaultShapes[i].name.c_str(), i);
    }

    int lmenu = glutCreateMenu(levelMenu);
    for(int i = 1; i < 8; ++i){
        char level[16];
        sprintf(level, "Level %d\n", i);
        glutAddMenuEntry(level, i);
    }

    int kmenu = glutCreateMenu(kernelMenu);
    glutAddMenuEntry(getKernelName(kCPU), kCPU);
#ifdef OPENSUBDIV_HAS_OPENMP
    glutAddMenuEntry(getKernelName(kOPENMP), kOPENMP);
#endif
#ifdef OPENSUBDIV_HAS_OPENCL
    glutAddMenuEntry(getKernelName(kCL), kCL);
#endif
#ifdef OPENSUBDIV_HAS_CUDA
    glutAddMenuEntry(getKernelName(kCUDA), kCUDA);
#endif
#ifdef OPENSUBDIV_HAS_GLSL_TRANSFORM_FEEDBACK
    glutAddMenuEntry(getKernelName(kGLSL), kGLSL);
#endif

    glutCreateMenu(menu);
    glutAddSubMenu("Level", lmenu);
    glutAddSubMenu("Model", smenu);
    glutAddSubMenu("Kernel", kmenu);
    glutAttachMenu(GLUT_RIGHT_BUTTON);

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutMouseFunc(mouse);
    glutKeyboardFunc(keyboard);
    glutMotionFunc(motion);
#if not defined(__APPLE__)
    glewInit();
#endif
    initGL();

    const char *filename = NULL;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-d"))
            g_level = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-c"))
            g_repeatCount = atoi(argv[++i]);
        else
            filename = argv[i];
    }

    modelMenu(0);

    glutIdleFunc(idle);
    glutMainLoop();

    quit();
}

//------------------------------------------------------------------------------
