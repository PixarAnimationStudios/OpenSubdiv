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
    #if defined(_WIN32)
        // XXX Must include windows.h here or GLFW pollutes the global namespace
        #define WIN32_LEAN_AND_MEAN
        #include <windows.h>
    #endif
#endif

#if defined(GLFW_VERSION_3)
    #include <GLFW/glfw3.h>
    GLFWwindow* g_window=0;
#else
    #include <GL/glfw.h>
#endif

#include <stdio.h>
#include <cassert>

#include <far/meshFactory.h>

#include <osd/vertex.h>
#include <osd/cpuVertexBuffer.h>
#include <osd/cpuComputeController.h>
#include <osd/cpuComputeContext.h>

#include <osd/cpuGLVertexBuffer.h>

#ifdef OPENSUBDIV_HAS_CUDA
#endif

#ifdef OPENSUBDIV_HAS_OPENCL
    #include <osd/clComputeContext.h>
    #include <osd/clComputeController.h>
    #include <osd/clGLVertexBuffer.h>
    static cl_context g_clContext;
    static cl_command_queue g_clQueue;
    #include "../../examples/common/clInit.h" // XXXX TODO move file out of examples
#endif

#include "../common/shape_utils.h"

//
// Regression testing matching Osd to Hbr
//
// Notes:
// - precision is currently held at 1e-6
//
// - results cannot be bitwise identical as some vertex interpolations
//   are not happening in the same order.
//
// - only vertex interpolation is being tested at the moment.
//
#define PRECISION 1e-6

//------------------------------------------------------------------------------
enum BackendType {
    kBackendCPU   = 0, // raw CPU
    kBackendCPUGL = 1, // CPU with GL-backed buffer
    kBackendCL    = 2, // OpenCL
    kBackendCount
};

static const char* g_BackendNames[kBackendCount] = {
    "CPU",
    "CPUGL",
    "CL",
};

static int g_Backend = -1;

//------------------------------------------------------------------------------
// Vertex class implementation
struct xyzVV {

    xyzVV() { }

    xyzVV( int /*i*/ ) { }

    xyzVV( float x, float y, float z ) { _pos[0]=x; _pos[1]=y; _pos[2]=z; }

    xyzVV( const xyzVV & src ) { _pos[0]=src._pos[0]; _pos[1]=src._pos[1]; _pos[2]=src._pos[2]; }

   ~xyzVV( ) { }

    void     AddWithWeight(const xyzVV& src, float weight, void * =0 ) { 
        _pos[0]+=weight*src._pos[0]; 
        _pos[1]+=weight*src._pos[1]; 
        _pos[2]+=weight*src._pos[2]; 
    }

    void     AddVaryingWithWeight(const xyzVV& , float, void * =0 ) { }

    void     Clear( void * =0 ) { _pos[0]=_pos[1]=_pos[2]=0.0f; }

    void     SetPosition(float x, float y, float z) { _pos[0]=x; _pos[1]=y; _pos[2]=z; }

    void     ApplyVertexEdit(const OpenSubdiv::HbrVertexEdit<xyzVV> & edit) {
                 const float *src = edit.GetEdit();
                 switch(edit.GetOperation()) {
                   case OpenSubdiv::HbrHierarchicalEdit<xyzVV>::Set:
                     _pos[0] = src[0];
                     _pos[1] = src[1];
                     _pos[2] = src[2];
                     break;
                   case OpenSubdiv::HbrHierarchicalEdit<xyzVV>::Add:
                     _pos[0] += src[0];
                     _pos[1] += src[1];
                     _pos[2] += src[2];
                     break;
                   case OpenSubdiv::HbrHierarchicalEdit<xyzVV>::Subtract:
                     _pos[0] -= src[0];
                     _pos[1] -= src[1];
                     _pos[2] -= src[2];
                     break;
                 }
             }

    void ApplyVertexEdit(OpenSubdiv::FarVertexEdit const & edit) {
        const float *src = edit.GetEdit();
        switch(edit.GetOperation()) {
          case OpenSubdiv::FarVertexEdit::Set:
            _pos[0] = src[0];
            _pos[1] = src[1];
            _pos[2] = src[2];
            break;
          case OpenSubdiv::FarVertexEdit::Add:
            _pos[0] += src[0];
            _pos[1] += src[1];
            _pos[2] += src[2];
            break;
        }
    }
    
    void     ApplyMovingVertexEdit(const OpenSubdiv::HbrMovingVertexEdit<xyzVV> &) { }

    const float * GetPos() const { return _pos; }

private:
    float _pos[3];
};

//------------------------------------------------------------------------------

class xyzFV;
typedef OpenSubdiv::HbrMesh<xyzVV>           xyzmesh;
typedef OpenSubdiv::HbrFace<xyzVV>           xyzface;
typedef OpenSubdiv::HbrVertex<xyzVV>         xyzvertex;
typedef OpenSubdiv::HbrHalfedge<xyzVV>       xyzhalfedge;
typedef OpenSubdiv::HbrFaceOperator<xyzVV>   xyzFaceOperator;
typedef OpenSubdiv::HbrVertexOperator<xyzVV> xyzVertexOperator;

typedef OpenSubdiv::HbrMesh<OpenSubdiv::OsdVertex>     OsdHbrMesh;
typedef OpenSubdiv::HbrVertex<OpenSubdiv::OsdVertex>   OsdHbrVertex;
typedef OpenSubdiv::HbrFace<OpenSubdiv::OsdVertex>     OsdHbrFace;
typedef OpenSubdiv::HbrHalfedge<OpenSubdiv::OsdVertex> OsdHbrHalfedge;

//------------------------------------------------------------------------------
// Returns true if a vertex or any of its parents is on a boundary
bool 
VertexOnBoundary( xyzvertex const * v ) {

    if (not v)
        return false;

    if (v->OnBoundary())
        return true;

    xyzvertex const * pv = v->GetParentVertex();
    if (pv)
        return VertexOnBoundary(pv);
    else {
        xyzhalfedge const * pe = v->GetParentEdge();
        if (pe) {
              return VertexOnBoundary(pe->GetOrgVertex()) or
                     VertexOnBoundary(pe->GetDestVertex());
        } else {
            xyzface const * pf = v->GetParentFace(), * rootf = pf;
            while (pf) {
                pf = pf->GetParent();
                if (pf)
                    rootf=pf;
            }
            if (rootf)
                for (int i=0; i<rootf->GetNumVertices(); ++i)
                    if (rootf->GetVertex(i)->OnBoundary())
                        return true;
        }
    }
    return false;
}

//------------------------------------------------------------------------------
int 
checkVertexBuffer( xyzmesh * hmesh, const float * vbData, int numElements, std::vector<int> const & remap) {

    int count=0;
    float deltaAvg[3] = {0.0f, 0.0f, 0.0f},
          deltaCnt[3] = {0.0f, 0.0f, 0.0f};

    int nverts = hmesh->GetNumVertices();
    for (int i=0; i<nverts; ++i) {

        xyzvertex * hv = hmesh->GetVertex(i);

        const float * ov = & vbData[ remap[ hv->GetID() ] * numElements ];

        // boundary interpolation rules set to "none" produce "undefined" vertices on
        // boundary vertices : far does not match hbr for those, so skip comparison.
        if ( hmesh->GetInterpolateBoundaryMethod()==xyzmesh::k_InterpolateBoundaryNone and
             VertexOnBoundary(hv) )
             continue;


        if ( hv->GetData().GetPos()[0] != ov[0] )
            deltaCnt[0]++;
        if ( hv->GetData().GetPos()[1] != ov[1] )
            deltaCnt[1]++;
        if ( hv->GetData().GetPos()[2] != ov[2] )
            deltaCnt[2]++;

        float delta[3] = { hv->GetData().GetPos()[0] - ov[0],
                           hv->GetData().GetPos()[1] - ov[1],
                           hv->GetData().GetPos()[2] - ov[2] };

        deltaAvg[0]+=delta[0];
        deltaAvg[1]+=delta[1];
        deltaAvg[2]+=delta[2];

        float dist = sqrtf( delta[0]*delta[0]+delta[1]*delta[1]+delta[2]*delta[2]);
        if ( dist > PRECISION ) {
            printf("// HbrVertex<T> %d fails : dist=%.10f (%.10f %.10f %.10f)"
                   " (%.10f %.10f %.10f)\n", i, dist, hv->GetData().GetPos()[0],
                                                      hv->GetData().GetPos()[1],
                                                      hv->GetData().GetPos()[2],
                                                      ov[0],
                                                      ov[1],
                                                      ov[2] );
           count++;
        }
    }

    if (deltaCnt[0])
        deltaAvg[0]/=deltaCnt[0];
    if (deltaCnt[1])
        deltaAvg[1]/=deltaCnt[1];
    if (deltaCnt[2])
        deltaAvg[2]/=deltaCnt[2];

    printf("    delta ratio : (%d/%d %d/%d %d/%d)\n", (int)deltaCnt[0], nverts,
                                                      (int)deltaCnt[1], nverts,
                                                      (int)deltaCnt[2], nverts );
    printf("    average delta : (%.10f %.10f %.10f)\n", deltaAvg[0],
                                                        deltaAvg[1],
                                                        deltaAvg[2] );
    if (count==0)
        printf("  success !\n");

    return count;
}

//------------------------------------------------------------------------------
static void 
refine( xyzmesh * mesh, int maxlevel ) {

    for (int l=0; l<maxlevel; ++l ) {
        int nfaces = mesh->GetNumFaces();
        for (int i=0; i<nfaces; ++i) {
            xyzface * f = mesh->GetFace(i);
            if (f->GetDepth()==l)
                f->Refine();
        }
    }
}

//------------------------------------------------------------------------------
static int 
checkMeshCPU( OpenSubdiv::FarMesh<OpenSubdiv::OsdVertex>* farmesh,
              const std::vector<float>& coarseverts,
              xyzmesh * refmesh,
              const std::vector<int>& remap) {
                  
    static OpenSubdiv::OsdCpuComputeController *controller = new OpenSubdiv::OsdCpuComputeController();
    
    OpenSubdiv::OsdCpuComputeContext *context = OpenSubdiv::OsdCpuComputeContext::Create(farmesh);
    
    OpenSubdiv::OsdCpuVertexBuffer * vb = OpenSubdiv::OsdCpuVertexBuffer::Create(3, farmesh->GetNumVertices());
    
    vb->UpdateData( & coarseverts[0], 0, (int)coarseverts.size()/3 );
    
    controller->Refine( context, farmesh->GetKernelBatches(), vb );
    
    return checkVertexBuffer(refmesh, vb->BindCpuBuffer(), vb->GetNumElements(), remap);
}

//------------------------------------------------------------------------------
static int 
checkMeshCPUGL( OpenSubdiv::FarMesh<OpenSubdiv::OsdVertex>* farmesh,
                const std::vector<float>& coarseverts,
                xyzmesh * refmesh,
                const std::vector<int>& remap) {
                    
    static OpenSubdiv::OsdCpuComputeController *controller = new OpenSubdiv::OsdCpuComputeController();
    
    OpenSubdiv::OsdCpuComputeContext *context = OpenSubdiv::OsdCpuComputeContext::Create(farmesh);
    
    OpenSubdiv::OsdCpuGLVertexBuffer * vb = OpenSubdiv::OsdCpuGLVertexBuffer::Create(3, farmesh->GetNumVertices());
    
    vb->UpdateData( & coarseverts[0], 0, (int)coarseverts.size()/3 );
    
    controller->Refine( context, farmesh->GetKernelBatches(), vb );
    
    return checkVertexBuffer(refmesh, vb->BindCpuBuffer(), vb->GetNumElements(), remap);
}

//------------------------------------------------------------------------------
static int 
checkMeshCL( OpenSubdiv::FarMesh<OpenSubdiv::OsdVertex>* farmesh,
             const std::vector<float>& coarseverts,
             xyzmesh * refmesh,
             const std::vector<int>& remap ) {

#ifdef OPENSUBDIV_HAS_OPENCL

    static OpenSubdiv::OsdCLComputeController *controller = new OpenSubdiv::OsdCLComputeController(g_clContext, g_clQueue);
    
    OpenSubdiv::OsdCLComputeContext *context = OpenSubdiv::OsdCLComputeContext::Create(farmesh, g_clContext);
    
    OpenSubdiv::OsdCLGLVertexBuffer * vb = OpenSubdiv::OsdCLGLVertexBuffer::Create(3, farmesh->GetNumVertices(), g_clContext);
    
    vb->UpdateData( & coarseverts[0], 0, (int)coarseverts.size()/3, g_clQueue );
    
    controller->Refine( context, farmesh->GetKernelBatches(), vb );

    // read data back from CL buffer
    size_t dataSize = vb->GetNumVertices() * vb->GetNumElements();
    float* data = new float[dataSize];
    
    clEnqueueReadBuffer (g_clQueue, vb->BindCLBuffer(g_clQueue), CL_TRUE, 0, dataSize * sizeof(float), data, 0, NULL, NULL);
    
    int result = checkVertexBuffer(refmesh, data, vb->GetNumElements(), remap);
    
    delete[] data;
    
    return result;
#else
    return 0;
#endif
}

//------------------------------------------------------------------------------
static int 
checkMesh( char const * msg, std::string const & shape, int levels, Scheme scheme, int backend ) {

    int result =0;

    printf("- %s (scheme=%d)\n", msg, scheme);

    xyzmesh * refmesh = simpleHbr<xyzVV>(shape.c_str(), scheme, 0);

    refine( refmesh, levels );


    std::vector<float> coarseverts;

    OsdHbrMesh * hmesh = simpleHbr<OpenSubdiv::OsdVertex>(shape.c_str(), scheme, coarseverts);

    OpenSubdiv::FarMeshFactory<OpenSubdiv::OsdVertex> meshFactory(hmesh, levels);

    OpenSubdiv::FarMesh<OpenSubdiv::OsdVertex> * farmesh = meshFactory.Create();

    std::vector<int> remap = meshFactory.GetRemappingTable();

    switch (backend) {
        case kBackendCPU   : result = checkMeshCPU(farmesh, coarseverts, refmesh, remap); break;
        case kBackendCPUGL : result = checkMeshCPUGL(farmesh, coarseverts, refmesh, remap); break;
        case kBackendCL    : result = checkMeshCL(farmesh, coarseverts, refmesh, remap); break;
    }

    delete hmesh;

    return result;
}

//------------------------------------------------------------------------------
int checkBackend(int backend, int levels) {

    printf("*** checking backend : %s\n", g_BackendNames[backend]);

    if (backend == kBackendCL) {
#ifdef OPENSUBDIV_HAS_OPENCL
        if (initCL(&g_clContext, &g_clQueue) == false) {
            printf("  Cannot initialize OpenCL, skipping...\n");
            return 0;
        }
#else
        printf("  No OpenCL available, skipping...\n");
        return 0;
#endif
    }

    int total = 0;

#define test_catmark_edgeonly
#define test_catmark_edgecorner
#define test_catmark_flap
#define test_catmark_pyramid
#define test_catmark_pyramid_creases0
#define test_catmark_pyramid_creases1
#define test_catmark_cube
#define test_catmark_cube_creases0
#define test_catmark_cube_creases1
#define test_catmark_cube_corner0
#define test_catmark_cube_corner1
#define test_catmark_cube_corner2
#define test_catmark_cube_corner3
#define test_catmark_cube_corner4
#define test_catmark_dart_edgeonly
#define test_catmark_dart_edgecorner
#define test_catmark_tent
#define test_catmark_tent_creases0
#define test_catmark_tent_creases1
#define test_catmark_square_hedit0
#define test_catmark_square_hedit1
#define test_catmark_square_hedit2
#define test_catmark_square_hedit3

#define test_loop_triangle_edgeonly
#define test_loop_triangle_edgecorner
#define test_loop_icosahedron
#define test_loop_cube
#define test_loop_cube_creases0
#define test_loop_cube_creases1

#define test_bilinear_cube


#ifdef test_catmark_edgeonly
#include "../shapes/catmark_edgeonly.h"
    total += checkMesh( "test_catmark_edgeonly", catmark_edgeonly, levels, kCatmark, backend );
#endif

#ifdef test_catmark_edgecorner
#include "../shapes/catmark_edgecorner.h"
    total += checkMesh( "test_catmark_edgeonly", catmark_edgecorner, levels, kCatmark, backend );
#endif

#ifdef test_catmark_flap
#include "../shapes/catmark_flap.h"
    total += checkMesh( "test_catmark_flap", catmark_flap, levels, kCatmark, backend );
#endif

#ifdef test_catmark_pyramid
#include "../shapes/catmark_pyramid.h"
    total += checkMesh( "test_catmark_pyramid", catmark_pyramid, levels, kCatmark, backend );
#endif

#ifdef test_catmark_pyramid_creases0
#include "../shapes/catmark_pyramid_creases0.h"
    total += checkMesh( "test_catmark_pyramid_creases0", catmark_pyramid_creases0, levels, kCatmark, backend );
#endif

#ifdef test_catmark_pyramid_creases1
#include "../shapes/catmark_pyramid_creases1.h"
    total += checkMesh( "test_catmark_pyramid_creases1", catmark_pyramid_creases1, levels, kCatmark, backend );
#endif

#ifdef test_catmark_cube
#include "../shapes/catmark_cube.h"
    total += checkMesh( "test_catmark_cube", catmark_cube, levels, kCatmark, backend );
#endif

#ifdef test_catmark_cube_creases0
#include "../shapes/catmark_cube_creases0.h"
    total += checkMesh( "test_catmark_cube_creases0", catmark_cube_creases0, levels, kCatmark, backend );
#endif

#ifdef test_catmark_cube_creases1
#include "../shapes/catmark_cube_creases1.h"
    total += checkMesh( "test_catmark_cube_creases1", catmark_cube_creases1, levels, kCatmark, backend );
#endif

#ifdef test_catmark_cube_corner0
#include "../shapes/catmark_cube_corner0.h"
    total += checkMesh( "test_catmark_cube_corner0", catmark_cube_corner0, levels, kCatmark, backend );
#endif

#ifdef test_catmark_cube_corner1
#include "../shapes/catmark_cube_corner1.h"
    total += checkMesh( "test_catmark_cube_corner1", catmark_cube_corner1, levels, kCatmark, backend );
#endif

#ifdef test_catmark_cube_corner2
#include "../shapes/catmark_cube_corner2.h"
    total += checkMesh( "test_catmark_cube_corner2", catmark_cube_corner2, levels, kCatmark, backend );
#endif

#ifdef test_catmark_cube_corner3
#include "../shapes/catmark_cube_corner3.h"
    total += checkMesh( "test_catmark_cube_corner3", catmark_cube_corner3, levels, kCatmark, backend );
#endif

#ifdef test_catmark_cube_corner4
#include "../shapes/catmark_cube_corner4.h"
    total += checkMesh( "test_catmark_cube_corner4", catmark_cube_corner4, levels, kCatmark, backend );
#endif

#ifdef test_catmark_dart_edgecorner
#include "../shapes/catmark_dart_edgecorner.h"
    total += checkMesh( "test_catmark_dart_edgecorner", catmark_dart_edgecorner, levels, kCatmark, backend );
#endif

#ifdef test_catmark_dart_edgeonly
#include "../shapes/catmark_dart_edgeonly.h"
    total += checkMesh( "test_catmark_dart_edgeonly", catmark_dart_edgeonly, levels, kCatmark, backend );
#endif

#ifdef test_catmark_tent
#include "../shapes/catmark_tent.h"
    total += checkMesh( "test_catmark_tent", catmark_tent, levels, kCatmark, backend );
#endif

#ifdef test_catmark_tent_creases0
#include "../shapes/catmark_tent_creases0.h"
    total += checkMesh( "test_catmark_tent_creases0", catmark_tent_creases0, levels, kCatmark, backend );
#endif

#ifdef test_catmark_tent_creases1
#include "../shapes/catmark_tent_creases1.h"
    total += checkMesh( "test_catmark_tent_creases1", catmark_tent_creases1, levels, kCatmark, backend );
#endif

#ifdef test_catmark_square_hedit0
#include "../shapes/catmark_square_hedit0.h"
    total += checkMesh( "test_catmark_square_hedit0", catmark_square_hedit0, levels, kCatmark, backend );
#endif

#ifdef test_catmark_square_hedit1
#include "../shapes/catmark_square_hedit1.h"
    total += checkMesh( "test_catmark_square_hedit1", catmark_square_hedit1, levels, kCatmark, backend );
#endif

#ifdef test_catmark_square_hedit2
#include "../shapes/catmark_square_hedit2.h"
    total += checkMesh( "test_catmark_square_hedit2", catmark_square_hedit2, levels, kCatmark, backend );
#endif

#ifdef test_catmark_square_hedit3
#include "../shapes/catmark_square_hedit3.h"
    total += checkMesh( "test_catmark_square_hedit3", catmark_square_hedit3, levels, kCatmark, backend );
#endif


#ifdef test_loop_triangle_edgeonly
#include "../shapes/loop_triangle_edgeonly.h"
    total += checkMesh( "test_loop_triangle_edgeonly", loop_triangle_edgeonly, levels, kLoop, backend );
#endif

#ifdef test_loop_triangle_edgecorner
#include "../shapes/loop_triangle_edgecorner.h"
    total += checkMesh( "test_loop_triangle_edgecorner", loop_triangle_edgecorner, levels, kLoop, backend );
#endif

#ifdef test_loop_saddle_edgeonly
#include "../shapes/loop_saddle_edgeonly.h"
    total += checkMesh( "test_loop_saddle_edgeonly", loop_saddle_edgeonly, levels, kLoop, backend );
#endif

#ifdef test_loop_saddle_edgecorner
#include "../shapes/loop_saddle_edgecorner.h"
    total += checkMesh( "test_loop_saddle_edgecorner", loop_saddle_edgecorner, levels, kLoop, backend );
#endif

#ifdef test_loop_icosahedron
#include "../shapes/loop_icosahedron.h"
    total += checkMesh( "test_loop_icosahedron", loop_icosahedron, levels, kLoop, backend );
#endif

#ifdef test_loop_cube
#include "../shapes/loop_cube.h"
    total += checkMesh( "test_loop_cube", loop_cube, levels, kLoop, backend );
#endif

#ifdef test_loop_cube_creases0
#include "../shapes/loop_cube_creases0.h"
    total += checkMesh( "test_loop_cube_creases0", loop_cube_creases0,levels, kLoop, backend );
#endif

#ifdef test_loop_cube_creases1
#include "../shapes/loop_cube_creases1.h"
    total += checkMesh( "test_loop_cube_creases1", loop_cube_creases1, levels, kLoop, backend );
#endif



#ifdef test_bilinear_cube
#include "../shapes/bilinear_cube.h"
    total += checkMesh( "test_bilinear_cube", bilinear_cube, levels, kBilinear, backend );
#endif


    if (backend == kBackendCL) {
#ifdef OPENSUBDIV_HAS_OPENCL
        uninitCL(g_clContext, g_clQueue);
#endif
    }

    return total;
}

//------------------------------------------------------------------------------
static void
usage(char ** argv) {
    printf("%s [<options>]\n\n", argv[0]);
    printf("    Options :\n");
    
    printf("        -compute <backend>\n");
    printf("        Compute backend applied (");
    for (int i=0; i < kBackendCount; ++i)
        printf("%s ", g_BackendNames[i]);
    printf(").\n");
    
    printf("        -help / -h\n");
    printf("        Displays usage information.");
      
}

//------------------------------------------------------------------------------
static void 
parseArgs(int argc, char ** argv) {

    for (int i=1; i<argc; ++i) {
        if (not strcmp(argv[i],"-compute")) {
        
            const char * backend = NULL;
            
            if (i<(argc-1))
                backend = argv[++i];

            if (not strcmp(backend, "all")) {
              g_Backend = -1;
            } else {
              bool found = false;
              for (int i = 0; i < kBackendCount; ++i) {
                if (not strcmp(backend, g_BackendNames[i])) {
                  g_Backend = i;
                  found = true;
                  break;
                }
              }
              if (not found) {
                printf("-compute : must be 'all' or one of: ");
                for (int i = 0; i < kBackendCount; ++i)
                    printf("%s ", g_BackendNames[i]);
                printf("\n");
                exit(0);
              }
            }
        } else if ( (not strcmp(argv[i],"-help")) or
                    (not strcmp(argv[i],"-h")) ) {
            usage(argv);
            exit(1);
        } else {
            usage(argv);
            exit(0);
        }
    }
}

//------------------------------------------------------------------------------
int 
main(int argc, char ** argv) {

    // Run with no args tests default (CPU) backend.
    // "-backend all" tests all available backends.
    // "-backend <name>" tests one backend.
    parseArgs(argc, argv);

    // Make sure we have an OpenGL context : create dummy GLFW window
    if (not glfwInit()) {
        printf("Failed to initialize GLFW\n");
        return 1;
    }

    int width=10, height=10;
    
#if GLFW_VERSION_MAJOR>=3
    static const char windowTitle[] = "OpenSubdiv OSD regression";
    if (not (g_window=glfwCreateWindow(width, height, windowTitle, NULL, NULL))) {
        printf("Failed to open window.\n");
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(g_window);
#else
    if (glfwOpenWindow(width, height, 8, 8, 8, 8, 24, 8,GLFW_WINDOW) == GL_FALSE) {
        printf("Failed to open window.\n");
        glfwTerminate();
        return 1;
    }
#endif
    
#if defined(OSD_USES_GLEW)
    if (GLenum r = glewInit() != GLEW_OK) {
        printf("Failed to initialize glew. error = %d\n", r);
        exit(1);
    }
#endif

    printf("precision : %f\n",PRECISION);

    int levels=5, total=0;

    if (g_Backend == -1) {
        for (int i = 0; i < kBackendCount; ++i)
            total += checkBackend (i, levels);
    } else {
        total += checkBackend (g_Backend, levels);
    }

    glfwTerminate();

    if (total==0)
      printf("All tests passed.\n");
    else
      printf("Total failures : %d\n", total);
}
