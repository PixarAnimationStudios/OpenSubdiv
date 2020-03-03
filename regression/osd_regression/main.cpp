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
GLFWwindow* g_window=0;

#include <stdio.h>
#include <cassert>

#include <opensubdiv/osd/cpuEvaluator.h>
#include <opensubdiv/osd/cpuVertexBuffer.h>
#include <opensubdiv/osd/cpuGLVertexBuffer.h>
#include <opensubdiv/far/stencilTableFactory.h>

#include "../common/cmp_utils.h"
#include "../common/hbr_utils.h"
#include "../common/far_utils.h"

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

using namespace OpenSubdiv;    

//------------------------------------------------------------------------------
enum BackendType {
    kBackendCPU   = 0, // raw CPU
    kBackendCPUGL = 1, // CPU with GL-backed buffer
    kBackendCount
};

static const char* g_BackendNames[kBackendCount] = {
    "CPU",
    "CPUGL",
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

typedef OpenSubdiv::Far::TopologyRefiner FarTopologyRefiner;


//------------------------------------------------------------------------------
int 
checkVertexBuffer( 
    const FarTopologyRefiner &refiner, xyzmesh * hmesh, 
    const float * vbData, int numElements) {

    int count=0;
    float deltaAvg[3] = {0.0f, 0.0f, 0.0f},
          deltaCnt[3] = {0.0f, 0.0f, 0.0f};

    std::vector<xyzVV> hbrVertexData;
    std::vector<bool>  hbrVertexOnBoundaryData;

    // Only care about vertex on boundary conditions if the interpolate boundary
    // is 'none'
    std::vector<bool> *hbrVertexOnBoundaryPtr =
        (hmesh->GetInterpolateBoundaryMethod() == 
            xyzmesh::k_InterpolateBoundaryNone)
        ? &hbrVertexOnBoundaryData
        : NULL;


    GetReorderedHbrVertexData(refiner, *hmesh, &hbrVertexData, 
        hbrVertexOnBoundaryPtr);

    //int nverts = hmesh->GetNumVertices();
    int nverts = (int)hbrVertexData.size();

    for (int i=0; i<nverts; ++i) {

        const float * ov = & vbData[ i * numElements ];

        // boundary interpolation rules set to "none" produce "undefined" 
        // vertices on boundary vertices : far does not match hbr for those,
        // so skip comparison.
        if (hbrVertexOnBoundaryPtr && (*hbrVertexOnBoundaryPtr)[i])
             continue;

        const float *hbrPos = hbrVertexData[i].GetPos();


        if ( hbrPos[0] != ov[0] )
            deltaCnt[0]++;
        if ( hbrPos[1] != ov[1] )
            deltaCnt[1]++;
        if ( hbrPos[2] != ov[2] )
            deltaCnt[2]++;

        float delta[3] = { hbrPos[0] - ov[0],
                           hbrPos[1] - ov[1],
                           hbrPos[2] - ov[2] };

        deltaAvg[0]+=delta[0];
        deltaAvg[1]+=delta[1];
        deltaAvg[2]+=delta[2];

        float dist = sqrtf( delta[0]*delta[0]+delta[1]*delta[1]+delta[2]*delta[2]);
        if ( dist > PRECISION ) {
            printf("// HbrVertex<T> %d fails : dist=%.10f (%.10f %.10f %.10f)"
                   " (%.10f %.10f %.10f)\n", i, dist, hbrPos[0],
                                                      hbrPos[1],
                                                      hbrPos[2],
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
buildStencilTable(
    const FarTopologyRefiner &refiner,
    Far::StencilTable const **vertexStencils,
    Far::StencilTable const **varyingStencils)
{
    Far::StencilTableFactory::Options soptions;
    soptions.generateOffsets = true;
    soptions.generateIntermediateLevels = true;

    *vertexStencils = Far::StencilTableFactory::Create(refiner, soptions);

    soptions.interpolationMode = Far::StencilTableFactory::INTERPOLATE_VARYING;
    *varyingStencils = Far::StencilTableFactory::Create(refiner, soptions);
}



//------------------------------------------------------------------------------
static int 
checkMeshCPU( FarTopologyRefiner *refiner,
              const std::vector<xyzVV>& coarseverts,
              xyzmesh * refmesh) {

    Far::StencilTable const *vertexStencils;
    Far::StencilTable const *varyingStencils;
    buildStencilTable(*refiner, &vertexStencils, &varyingStencils);

    assert(coarseverts.size() == (size_t)refiner->GetNumVerticesTotal());
    
    Osd::CpuVertexBuffer * vb = 
        Osd::CpuVertexBuffer::Create(3, refiner->GetNumVerticesTotal());
    
    vb->UpdateData( coarseverts[0].GetPos(), 0, (int)coarseverts.size() );
    
    Osd::CpuEvaluator::EvalStencils(
        vb, Osd::BufferDescriptor(0, 3, 3),
        vb, Osd::BufferDescriptor(refiner->GetLevel(0).GetNumVertices()*3, 3, 3),
        vertexStencils);

    int result = checkVertexBuffer(*refiner, refmesh, vb->BindCpuBuffer(), 
        vb->GetNumElements());

    delete vertexStencils;
    delete varyingStencils;
    delete vb;

    return result;
}

//------------------------------------------------------------------------------
static int 
checkMeshCPUGL(FarTopologyRefiner *refiner,
               const std::vector<xyzVV>& coarseverts,
               xyzmesh * refmesh) {

    Far::StencilTable const *vertexStencils;
    Far::StencilTable const *varyingStencils;
    buildStencilTable(*refiner, &vertexStencils, &varyingStencils);
    
    Osd::CpuGLVertexBuffer *vb = Osd::CpuGLVertexBuffer::Create(3, 
        refiner->GetNumVerticesTotal());
    
    vb->UpdateData( coarseverts[0].GetPos(), 0, (int)coarseverts.size() );

    Osd::CpuEvaluator::EvalStencils(
        vb, Osd::BufferDescriptor(0, 3, 3),
        vb, Osd::BufferDescriptor(refiner->GetLevel(0).GetNumVertices()*3, 3, 3),
        vertexStencils);

    int result = checkVertexBuffer(*refiner, refmesh, 
        vb->BindCpuBuffer(), vb->GetNumElements());

    delete vertexStencils;
    delete varyingStencils;
    delete vb;
    
    return result;
}

//------------------------------------------------------------------------------
static int 
checkMesh( char const * msg, std::string const & shape, int levels, Scheme scheme, int backend ) {

    int result =0;

    printf("- %s (scheme=%d)\n", msg, scheme);

    xyzmesh * refmesh = 
        interpolateHbrVertexData<xyzVV>(shape.c_str(), scheme, levels);

    std::vector<xyzVV> farVertexData;

    FarTopologyRefiner *refiner =
        InterpolateFarVertexData(shape.c_str(), scheme, levels, 
            farVertexData);

    switch (backend) {
        case kBackendCPU:
            result = checkMeshCPU(refiner, farVertexData, refmesh); 
            break;
        case kBackendCPUGL: 
            result = checkMeshCPUGL(refiner, farVertexData, refmesh); 
            break;
    }

    delete refmesh;
    delete refiner;

    return result;
}

//------------------------------------------------------------------------------
int checkBackend(int backend, int levels) {

    printf("*** checking backend : %s\n", g_BackendNames[backend]);

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


// hedits don't work.
//#define test_catmark_square_hedit0
//#define test_catmark_square_hedit1
//#define test_catmark_square_hedit2
//#define test_catmark_square_hedit3

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

    for (int argi=1; argi<argc; ++argi) {
        if (! strcmp(argv[argi],"-compute")) {
        
            const char * backend = NULL;
            
            if (argi<(argc-1))
                backend = argv[++argi];

            if (! strcmp(backend, "all")) {
              g_Backend = -1;
            } else {
              bool found = false;
              for (int i = 0; i < kBackendCount; ++i) {
                if (! strcmp(backend, g_BackendNames[i])) {
                  g_Backend = i;
                  found = true;
                  break;
                }
              }
              if (! found) {
                printf("-compute : must be 'all' or one of: ");
                for (int i = 0; i < kBackendCount; ++i)
                    printf("%s ", g_BackendNames[i]);
                printf("\n");
                exit(0);
              }
            }
        } else if ( (! strcmp(argv[argi],"-help")) ||
                    (! strcmp(argv[argi],"-h")) ) {
            usage(argv);
            exit(1);
        } else {
            usage(argv);
            exit(0);
        }
    }
}

void _glfw_error_callback(int error, const char *description)
{
    printf("GLFW reported error %d: %s\n",
            error, description);
}

//------------------------------------------------------------------------------
int 
main(int argc, char ** argv) {

    // Run with no args tests default (CPU) backend.
    // "-backend all" tests all available backends.
    // "-backend <name>" tests one backend.
    parseArgs(argc, argv);

    glfwSetErrorCallback(_glfw_error_callback);

    // Make sure we have an OpenGL context : create dummy GLFW window
    if (! glfwInit()) {
        printf("DISPLAY set to '%s'\n", getenv("DISPLAY"));
        printf("Failed to initialize GLFW\n");
        return 1;
    }

    int width=10, height=10;
    
    static const char windowTitle[] = "OpenSubdiv OSD regression";
    if (! (g_window=glfwCreateWindow(width, height, windowTitle, NULL, NULL))) {
        printf("Failed to open window.\n");
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(g_window);
    
    OpenSubdiv::internal::GLLoader::applicationInitializeGL();

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
