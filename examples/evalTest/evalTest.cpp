// CPU Subdivision with OpenSubdiv 
// -------------------------------
// In this example program, we will setup an OpenGL application that uses OSD to
// subdivide an animated mesh. It is intended to be as simple as possible and 
// not necessarily efficient. It is also intended as a learning tool for 
// understanding the OSD internals. Unlike the other OSD examples, the common 
// code infrastructure has been removed for clarity.
//
// ### Program Structure 
//
// This example program is structured as follows:
//
// 1. Setup static mesh topology (OsdHbrMesh)
// 2. Convert the topology into a subdividable mesh (OsdMesh)
// 3. On each frame: 
//      * Animate the coarse mesh points and update the OsdMesh
//      * Subdivide the updated mesh
//      * Draw the subdivided mesh and wire frame
//
// If you are completely new to OSD, you should read the following sections to 
// get a basic understanding of how it works.
//
// ### OSD Architecture Basics
// As a client, you will primarily be interacting with the Osd and Hbr classes, 
// however it's good to be aware of all three layers. The following describes
// these layers from lowest level (Hbr) to highest (Osd):
//
// **Hbr: Halfedge Boundary Representation.**
// This layer represents the mesh topology as meshes, vertices and edges. It is
// the core that provides the structure for subdivision and provides an 
// abstraction for dealing with topology in a type-agnostic way (i.e. everything
// is templated).
//
// **Far: Feature Adaptive Representation.** 
// Far uses hbr to create and cache fast run time data structures for table 
// driven subdivision. Feature-adaptive refinement logic is used to adaptively 
// refine coarse topology only as much as needed. The FarMesh does hold vertex 
// objects but the topology has been baked into FarSubdivisionTables. It also
// provides the underpinnings for generic dispatch of subdivision evaluation, so
// subdivision can be preformed with different mechanisms (GLSL, Cuda, etc.),
// the concrete implementations are specified at the next layer up.
//
// **Osd: Open Subdiv.**
// Osd contains client level code that uses Far to create concrete instances of 
// meshes and compute patch CVs with different back ends for table driven 
// subdivision. Currently, the following are supported in Osd:
//
//  * CPU / C++ with single or multiple threads 
//  * GLSL kernels with transform feedback into VBOs 
//  * OpenCL kernels 
//  * CUDA kernels
//
// The amount of hardware specific computation code is small, ~300 lines of code,
// so it isn't a large effort to support multiple different ones for different 
// clients. In the future, it is conceivable that additional dispatchers will be
// developed to target mobile devices.
//

/*
Copyright (C) Pixar. All rights reserved.

This license governs use of the accompanying software. If you
use the software, you accept this license. If you do not accept
the license, do not use the software.

1. Definitions
The terms "reproduce," "reproduction," "derivative works," and
"distribution" have the same meaning here as under U.S.
copyright law.  A "contribution" is the original software, or
any additions or changes to the software.
A "contributor" is any person or entity that distributes its
contribution under this license.
"Licensed patents" are a contributor's patent claims that read
directly on its contribution.

2. Grant of Rights
(A) Copyright Grant- Subject to the terms of this license,
including the license conditions and limitations in section 3,
each contributor grants you a non-exclusive, worldwide,
royalty-free copyright license to reproduce its contribution,
prepare derivative works of its contribution, and distribute
its contribution or any derivative works that you create.
(B) Patent Grant- Subject to the terms of this license,
including the license conditions and limitations in section 3,
each contributor grants you a non-exclusive, worldwide,
royalty-free license under its licensed patents to make, have
made, use, sell, offer for sale, import, and/or otherwise
dispose of its contribution in the software or derivative works
of the contribution in the software.

3. Conditions and Limitations
(A) No Trademark License- This license does not grant you
rights to use any contributor's name, logo, or trademarks.
(B) If you bring a patent claim against any contributor over
patents that you claim are infringed by the software, your
patent license from such contributor to the software ends
automatically.
(C) If you distribute any portion of the software, you must
retain all copyright, patent, trademark, and attribution
notices that are present in the software.
(D) If you distribute any portion of the software in source
code form, you may do so only under this license by including a
complete copy of this license with your distribution. If you
distribute any portion of the software in compiled or object
code form, you may only do so under a license that complies
with this license.
(E) The software is licensed "as-is." You bear the risk of
using it. The contributors give no express warranties,
guarantees or conditions. You may have additional consumer
rights under your local laws which this license cannot change.
To the extent permitted under your local laws, the contributors
exclude the implied warranties of merchantability, fitness for
a particular purpose and non-infringement.
*/

// ### Helper Includes

// Vector algebra and common GL machinations that have been isolated for
// clarity of the core OSD code.
//

#include "glhelpers.h"

//
// ### OpenSubdiv Includes

// The mutex header provides a cross platform mutex implementation; the vertex 
// and mesh headers provide abstract representations of verts and meshes; the
// element array buffer provides an abstract representation of an index buffer; 
// and finally, the cpu dispatcher is how subdivision work is dispatched to the
// CPU.
//


// XXX: Fixme 
#include "../../regression/common/mutex.h"

#include <far/meshFactory.h>
 
#include <osd/vertex.h>
#include <osd/glDrawContext.h>
#include <osd/cpuDispatcher.h>
#include <osd/cpuVertexBuffer.h>
#include <osd/cpuComputeController.h>
#include <osd/cpuComputeContext.h>

using namespace OpenSubdiv;

// 
// ### Global Variables & Declarations
//

// The screen width & height; current frame for animation; and the desired 
// subdivision level.
//
int g_width = 0,
    g_height = 0,
    g_frame = 0,
    g_level = 10;

//
// A center point for the view matrix and the object size for framing
//
float g_center[3] = {0.0f, 0.0f, 0.0f},
      g_size = 0.0f;


//
// The coarse mesh positions are saved externally and deformed
// during playback.
//
std::vector<float> g_orgPositions;

// Global cache of subdivided positions and triangle mesh
std::vector<float> g_refinedPositions;
std::vector<unsigned int> g_refinedTriangleIndices;

GLuint g_refinedPositionsBuf;
GLuint g_refinedTriangleIndicesBuf;

//
// The OSD state: a mesh, vertex buffer and element array
//
//#include <osd/glMesh.h>
//OpenSubdiv::OsdGLMeshInterface *g_mesh = 0;;

typedef OpenSubdiv::HbrMesh<OpenSubdiv::OsdVertex>     OsdHbrMesh;
typedef OpenSubdiv::HbrVertex<OpenSubdiv::OsdVertex>   OsdHbrVertex;
typedef OpenSubdiv::HbrFace<OpenSubdiv::OsdVertex>     OsdHbrFace;
typedef OpenSubdiv::HbrHalfedge<OpenSubdiv::OsdVertex> OsdHbrHalfedge;

// 
// Forward declarations. These functions will be described below as they are 
// defined.
//
void idle();
void reshape(int width, int height);
void createOsdMesh(int level);
void display();
void updateGeom();

//
// ### The main program entry point
//

// register the Osd CPU kernel, 
// call createOsdMesh (see below), init glew and one-time GL state and enter the
// main glfw loop.
//
void initOsd() 
{
    // Initialize OpenGL in glhelpers.h, specify true for "adaptive" so the 
    // glsl shaders for simple adaptive subdivision will be compiled and linked
    initGL();

    // 
    // Dispatchers are created from a kernel enumeration via the factory pattern,
    // calling register here ensures that the CPU dispatcher will be available
    // for construction when it is requested via the kCPU enumeration inside the
    // function createOsdMesh.
    //
    createOsdMesh(g_level);
}




void
Univar4x4(float u, float *B, float *D)
{
    float t = u;
    float s = 1.0f - u;

    float A0 = s * s;
    float A1 = 2 * s * t;
    float A2 = t * t;

    B[0] = s * A0;
    B[1] = t * A0 + s * A1;
    B[2] = t * A1 + s * A2;
    B[3] = t * A2;

    D[0] =    - A0;
    D[1] = A0 - A1;
    D[2] = A1 - A2;
    D[3] = A2;
}

struct simpleVec3
{
    float x, y, z;

    simpleVec3() {}

    simpleVec3( float a )
    {
        x = a;
        y = a;
        z = a;
    }
    simpleVec3( float a, float b, float c )
    {
        x = a;
        y = b;
        z = c;
    }

    simpleVec3 & operator+=( simpleVec3  const & v )
        { x += v.x; y += v.y; z += v.z; return *this; }    
};

inline simpleVec3 operator*( simpleVec3  const & v, float const & s ) { return simpleVec3( v.x * s, v.y * s, v.z * s ); }
inline simpleVec3 operator*( float const & s, simpleVec3 const &v ) { return simpleVec3( v.x * s, v.y * s, v.z * s ); }

void EvalBezier(float u,
                float v,
                // all vertex positions for the subdiv
                simpleVec3 *vertexBuffer, 
                // vector of 16 indices into vertexBuffer
                const unsigned int *indices,     
                simpleVec3 &position,
                simpleVec3 &utangent,
                simpleVec3 &vtangent)
{
    float B[4], D[4];

    Univar4x4(u, B, D);

    simpleVec3 *cp = vertexBuffer;

    simpleVec3 BUCP[4], DUCP[4];

    for (int i=0; i<4; ++i) {
        BUCP[i] = simpleVec3(0,0,0);
        DUCP[i] = simpleVec3(0,0,0);

        for (int j=0; j<4; ++j) {
#if ROTATE == 1
            simpleVec3 A = cp[indices[4*(3-j) + (3-i)]];
#elif ROTATE == 2
            simpleVec3 A = cp[indices[4*i + (3-j)]];
#elif ROTATE == 3
            simpleVec3 A = cp[indices[4*j + i]];
#else
            simpleVec3 A = cp[indices[4*i + j]];
#endif
            BUCP[i] += A * B[j];
            DUCP[i] += A * D[j];
        }
    }

    position = simpleVec3(0,0,0);
    utangent = simpleVec3(0,0,0);
    vtangent = simpleVec3(0,0,0);

    Univar4x4(v, B, D);

    for (int i=0; i<4; ++i) {
        position += B[i] * BUCP[i];
        utangent += B[i] * DUCP[i];
        vtangent += D[i] * BUCP[i];
    }
}



void
EvalCubicBSpline(float u, float B[4], float BU[4])
{
    float t = u;
    float s = 1.0 - u;

    float C0 =                     s * (0.5f * s);
    float C1 = t * (s + 0.5f * t) + s * (0.5f * s + t);
    float C2 = t * (    0.5f * t);

    B[0] =                                     1.f/3.f * s                * C0;
    B[1] = (2.f/3.f * s +           t) * C0 + (2.f/3.f * s + 1.f/3.f * t) * C1;
    B[2] = (1.f/3.f * s + 2.f/3.f * t) * C1 + (          s + 2.f/3.f * t) * C2;
    B[3] =                1.f/3.f * t  * C2;

    BU[0] =    - C0;
    BU[1] = C0 - C1;
    BU[2] = C1 - C2;
    BU[3] = C2;
}

void
EvalBSpline(float u, float v,
            // all vertex positions for the subdiv
            simpleVec3 *vertexBuffer, 
            // vector of 16 indices into vertexBuffer
            const unsigned int *indices,                 
            simpleVec3 *position, simpleVec3 *utangent, simpleVec3 *vtangent)
{
    float B[4], D[4];

    EvalCubicBSpline(u, B, D);

    simpleVec3 BUCP[4], DUCP[4];

    simpleVec3 *cp = vertexBuffer;

    for (int i=0; i<4; ++i) {
        BUCP[i] = simpleVec3(0,0,0);
        DUCP[i] = simpleVec3(0,0,0);

        for (int j=0; j<4; ++j) {
            simpleVec3 A = cp[indices[i + j*4]];

            BUCP[i] += A * B[j];
            DUCP[i] += A * D[j];
        }
    }

    *position = simpleVec3(0,0,0);
    *utangent = simpleVec3(0,0,0);
    *vtangent = simpleVec3(0,0,0);

    EvalCubicBSpline(v, B, D);

    for (int i=0; i<4; ++i) {
        *position += B[i] * BUCP[i];
        *utangent += B[i] * DUCP[i];
        *vtangent += D[i] * BUCP[i];
    }
}



class MyPatch {
public:
    
    // Enum describing the number and arrangment of control vertices
    // in the patch.
    //
    // Regular patches will have 16 CVs, boundary = 12, corner = 9,
    // gregory = 4.       
    enum PatchType {
        Regular,
        Boundary,
        Corner,
        Gregory
    };

    // Note that MyPatch retains a pointer to CVs and depends
    // on it remaining allocated.
    MyPatch(const unsigned int *CVs, PatchType patchType) {
        
        _patchType = patchType;
        
        // These tables map the 9, 12, or 16 input control points onto the
        // canonical 16 control points for a regular patch.
        const int pRegular[16] = 
            {0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        const int pBoundary[16] =
            {0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
        const int pCorner[16] =         
            {0, 1, 2, 2, 0, 1, 2, 2, 3, 4, 5, 5, 6, 7, 8, 8 };            
        
        const int *p=NULL;
        switch (_patchType) {
        case Regular:  p=pRegular;  break;
        case Boundary: p=pBoundary; break;
        case Corner:   p=pCorner;   break;
        case Gregory: return; // XXX not yet implemented
        }

        // These tables account for patch rotation 
        const int r0[16] =
            { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
/*        
        const int r1[16] =
            { 12, 8, 4, 0, 13, 9, 5, 1, 14, 10, 6, 2, 15, 11, 7, 3 };
        const int r2[16] =
            { 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 };
        const int r3[16] =
            { 3, 7, 11, 15, 2, 6, 10, 14, 1, 5, 9, 13, 0, 4, 8, 12 };
*/            

        // XXX deal with rotation here
        const int *r = r0;

        // Expand and rotate control points using remapping tables
        // above.  Now the CVs will be expressed as a consistent 16 cv
        // arrangment for EvalBSpline use in Eval.
        //
        for (int i=0;i<16; ++i) {
            _cvs[i] = CVs[p[r[i]]];
        }        

    }
    void Eval( float u, float v, simpleVec3 *vertexBuffer, 
               simpleVec3 *position,  simpleVec3 *utangent,
               simpleVec3 *vtangent) {
        if (_patchType != Gregory) {
            EvalBSpline(u, v, vertexBuffer, _cvs,
                        position, utangent, vtangent);
        }
    }

    static int RingSizeForPatchType(PatchType patchType) {
        switch (patchType) {
        case Regular:  return 16;
        case Boundary: return 12;
        case Corner:   return 9; 
        case Gregory:  return 4; 
        }
    }   
    
private:
    
    
    //  Packed patch control vertices
    //   0  1  2  3 
    //   4  5  6  7
    //   8  9 10 11
    //  12 13 14 15
    unsigned int _cvs[16]; 

    PatchType _patchType;
};
 



static bool
TestPatchEvaluation()
{

    // This method creates a 4x4 cubic patch with limit
    // surface from 0->1 in x and y
    
    float positions[16*3] =   {
        -1.0f, -1.0f, 0.0f,
        0.00f, -1.0f, 0.0f,
        1.00f, -1.0f, 0.0f,
        2.00f, -1.0f, 0.0f,
        
        -1.0f, 0.00f, 0.0f,
        0.00f, 0.00f, 0.0f,
        1.00f, 0.00f, 0.0f,
        2.00f, 0.00f, 0.0f,
        
        -1.0f, 1.00f, 0.0f,
        0.00f, 1.00f, 0.0f,
        1.00f, 1.00f, 0.0f,
        2.00f, 1.00f, 0.0f,
        
        -1.0f, 2.00f, 0.0f,
        0.00f, 2.00f, 0.0f,
        1.00f, 2.00f, 0.0f,
        2.00f, 2.00f, 0.0f};

    unsigned int faceIndices[16] = {
        0,1,2,3,
        4,5,6,7,
        8,9,10,11,
        12,13,14,15};

    MyPatch patch(faceIndices, MyPatch::Regular);

    for (float u=0; u<=1.0; u+= 0.2) {
        for (float v=0; v<=1.0; v+= 0.2) {
            simpleVec3 position;
            simpleVec3 uTangent;
            simpleVec3 vTangent;
            patch.Eval(u, v, (simpleVec3*)positions,
                       &position, &uTangent, &vTangent);
            
            std::cout << u << "," << v << "( " <<
                position.x << ", " <<
                position.y << ", " <<
                position.z << "), ";
        }
        std::cout << "\n";
    }

    return true;
}
        
        
        


class OsdEvalContext : OpenSubdiv::OsdNonCopyable<OsdEvalContext> {
public:

    explicit OsdEvalContext(
        OpenSubdiv::FarMesh<OpenSubdiv::OsdVertex> *farmesh,
        unsigned int elementsPerVertex);
    
    virtual ~OsdEvalContext();

    // vertexData contains one float for each "elementsPerVertex",
    // for each vertex.  For position it'd be three floats for each vertex.
    void UpdateData(float *vertexData);

    void TessellateIntoTriangles(
        std::vector<unsigned int> *elementArray,
        std::vector<float> *vertices);
    
     OpenSubdiv::FarMesh<OpenSubdiv::OsdVertex> *GetFarMesh() { return _farMesh; }


  /*
  GetPatch(faceID) {
    if (regular) {
    }
      
  }
  Eval(int faceID,  float u,  float v, float *return, const OsdVertexBuffer &buf) {
    // gimme 16 indices into the vertex buffer and remap u and v for me into the local parameter space of the patch.
    // XXX do something a little different for boundary and corner.
    16indices indices = GetPatch(faceID, &u, &v);
    u = remap(u);
    v = remap(v);

    EvalBSpline(u,v , cp, WorldPos, Tangent, BiTangent);

    
  }
  */

    OpenSubdiv::FarTable<unsigned int> _patchTable;


private:

    // Create evaluation patches given a table describing a bunch of
    // patches, and the "ringSize" or number of CVs per patch.
    //
    // Regular patches will have ringSize = 16,  boundary = 12,
    // corner = 9, gregory = 4.
    // 
    void _AppendPatchArray(
        const OpenSubdiv::FarTable<unsigned int> &ptable,
        MyPatch::PatchType patchType) {
        
        // Iterate over all patches in this table.  Don't worry about
        // markers here, those would tell use what level of
        // subdivision the patch was created on.
        //
        // Just iterate over all patches, this is in blocks of
        // 16, 12, 9, or 4 unsigned ints per patch depending
        // on patchType
        //
        const unsigned int *vertIndices = ptable[0];
        int ringSize = MyPatch::RingSizeForPatchType(patchType);
            
        for (int i=0; i<ptable.GetSize(); i+=ringSize) {
            // Create a patch object from the next block
            // of 16 control point indices stored in
            // the patch table.
            MyPatch patch(vertIndices + i, patchType);
            _patches.push_back(patch);
        }
    }
       
    
    OpenSubdiv::FarMesh<OpenSubdiv::OsdVertex> *_farMesh;
    std::vector<MyPatch> _patches;

    // ElementsPerVertex would be 3 for xyz position, 6 for xyz
    // position + xyz normal, or more for shading data.  Here elements
    // represents the data specified per vertex and interpolated on the
    // surface, e.g. vertex varying
    //
    unsigned int _elementsPerVertex;

    OpenSubdiv::OsdCpuComputeContext *_osdComputeContext;
    OpenSubdiv::OsdCpuComputeController _osdComputeController;
    OpenSubdiv::OsdCpuVertexBuffer *_osdVertexBuffer;
};


OsdEvalContext::OsdEvalContext(
    OpenSubdiv::FarMesh<OpenSubdiv::OsdVertex> *farMesh,
    unsigned int elementsPerVertex)
    : _farMesh(farMesh),
      _elementsPerVertex(elementsPerVertex)
{

    const FarPatchTables *patchTables = farMesh->GetPatchTables();
    
    if (not patchTables) {
        std::cout << "Oooops- patchTables not constructed\n";
        return;
    } else {
        std::cout << "Have some patchTables\n";
    }

    // Iterate over the patches generated by feature
    // adaptive refinement and create MyPatch objects.
    //
    _AppendPatchArray(patchTables->GetFullRegularPatches(), MyPatch::Regular);
    
    _AppendPatchArray(patchTables->GetFullBoundaryPatches(), MyPatch::Boundary);
    _AppendPatchArray(patchTables->GetFullCornerPatches(), MyPatch::Corner);
    
    _AppendPatchArray(patchTables->GetFullGregoryPatches(), MyPatch::Gregory);
    _AppendPatchArray(patchTables->GetFullBoundaryGregoryPatches(),
                      MyPatch::Gregory);

    for (int p=0; p<5; ++p) {
        _AppendPatchArray(patchTables->GetTransitionRegularPatches(p),
                          MyPatch::Regular);
        
        for (int r=0; r<4; ++r) {           
            _AppendPatchArray(patchTables->GetTransitionBoundaryPatches(p, r),
                              MyPatch::Boundary);
            _AppendPatchArray(patchTables->GetTransitionCornerPatches(p, r),
                              MyPatch::Corner);
        }
     }        

                   
    
    
    std::cout << "Made " << _patches.size() << " patches\n";

    // Now create OpenSubdiv compute objects that will be used
    // later for limit surface computation

    _osdComputeContext = OpenSubdiv::OsdCpuComputeContext::Create(_farMesh);
    _osdVertexBuffer =
        OpenSubdiv::OsdCpuVertexBuffer::Create(
            3 /* 3 floats for position*/ , _farMesh->GetNumVertices());
}


OsdEvalContext::~OsdEvalContext()
{
    if (_osdComputeContext)
        delete _osdComputeContext;

    if (_osdVertexBuffer) 
        delete _osdVertexBuffer;
}



void
OsdEvalContext::UpdateData(float *vertexData)
{
    // Now compute control point positions and shading data on the
    // refined surface.
    //

    //
    // Send the animated coarse positions to the vertex buffer.
    //
    _osdVertexBuffer->UpdateData(vertexData, _farMesh->GetNumVertices());
    
    //
    // Dispatch subdivision work based on the coarse vertex buffer. At this 
    // point, the assigned dispatcher will queue up work, potentially in many
    // worker threads. If the subdivided data is required for further
    // processing a call to Synchronize() will allow you to block until
    // the worker threads complete.
    //
    _osdComputeController.Refine(_osdComputeContext, _osdVertexBuffer);

    //
    // Is the call to Synchronize() needed here in order to call BindCpuBuffer
    // later?
    //
    _osdComputeController.Synchronize();
}


void
OsdEvalContext::TessellateIntoTriangles(
    std::vector<unsigned int> *elementArray,
    std::vector<float> *vertices)
{
    float *points = _osdVertexBuffer->BindCpuBuffer();

    unsigned int N = 5;
    unsigned int N1 = N+1;    
    float delta = 1.0/(float)N;

    unsigned int debugPatch = -1;
    
    for (unsigned int i=0; i< _patches.size(); ++i) {

        // Add points for this patch first, record the starting
        // index of the points for this patch within vertices
        // before adding point positions
        int baseIndex = vertices->size()/3;
        
        for (float u=0; u<=1.0; u+=delta) {
            for (float v=0; v<=1.0; v+=delta) {                
                simpleVec3 position, utangent, vtangent;
                _patches[i].Eval(u, v, (simpleVec3*)points,
                                 &position, &utangent, &vtangent);
                vertices->push_back(position.x);
                vertices->push_back(position.y);
                vertices->push_back(position.z);
                if (i==debugPatch) {
                    std::cout << "\tPoint " << position.x << ", " << position.y << ", " << position.z << std::endl;
                }
            }
        }

        if (i==debugPatch)
            std::cout << "Num points = " << vertices->size()/3 << "\n";

        // Now add indexing for triangles
        for (unsigned int u=baseIndex; u< N*N + baseIndex; u+=N1) {
            for (unsigned int v=0; v< N; ++v) {
                // Add the indices for two triangles that get their
                // point positions from the vertices array
                elementArray->push_back(u      + v    );
                elementArray->push_back(u + N1 + v    );
                elementArray->push_back(u + N1 + v + 1);

                elementArray->push_back(u      + v);
                elementArray->push_back(u + N1 + v + 1);
                elementArray->push_back(u      + v + 1);

                if ((u+N1+v+1)  > vertices->size()) {
                    std::cout << "ERROR: " <<
                        u << "," <<
                        N1 << "," <<
                        v << "," <<
                        vertices->size() << "\n";
                }
                if (i==debugPatch) {
                    std::cout << "triIndices: ";
                    for (int j=0; j<6; ++j) {
                        std::cout << " " << (*elementArray)[elementArray->size() - (6-j)]; 
                    }
                    std::cout << "\n";
                }

            }
            if (i==debugPatch)
                std::cout << "\n";


        }
    }
}


OsdEvalContext *g_evalContext = NULL;

//
// ### Construct the OSD Mesh 

// Here is where the real meat of the OSD setup happens. The mesh topology is 
// created and stored for later use. Actual subdivision happens in updateGeom 
// which gets called at the end of this function and on frame change.
//
void
createOsdMesh(int level)
{
    std::cout << "Start createOsdMesh\n";


    TestPatchEvaluation();
    
    // 
    // Setup an OsdHbr mesh based on the desired subdivision scheme
    //
    static OpenSubdiv::HbrCatmarkSubdivision<OpenSubdiv::OsdVertex>  _catmark;
    OsdHbrMesh *hmesh(new OsdHbrMesh(&_catmark));


    //
    // Now that we have a mesh, we need to add verticies and define the
    // topology.
    //
    // Here, we've declared the raw vertex data in-line, for simplicity
    //
    float verts[] = {    0.000000f, -1.414214f, 1.000000f,
                        1.414214f, 0.000000f, 1.000000f,
                        -1.414214f, 0.000000f, 1.000000f,
                        0.000000f, 1.414214f, 1.000000f,
                        -1.414214f, 0.000000f, -1.000000f,
                        0.000000f, 1.414214f, -1.000000f,
                        0.000000f, -1.414214f, -1.000000f,
                        1.414214f, 0.000000f, -1.000000f
                        };

    //
    // The cube faces are also in-lined, here they are specified as quads
    //
    int faces[] = {
                        0,1,3,2,
                        2,3,5,4,
                        4,5,7,6,
                        6,7,1,0,
                        1,7,5,3,
                        6,0,2,4
                        };
                        

    //
    // Record the original vertex positions and add verts to the mesh.
    //
    // OsdVertex is really just a place holder, it doesn't care what the 
    // position of the vertex is, it's just being used here as a means of
    // defining the mesh topology.
    //
    for (unsigned i = 0; i < sizeof(verts)/sizeof(float); i += 3) {
        g_orgPositions.push_back(verts[i+0]);
        g_orgPositions.push_back(verts[i+1]);
        g_orgPositions.push_back(verts[i+2]);
        
        OpenSubdiv::OsdVertex vert;
        hmesh->NewVertex(i/3, vert);
    }

    //
    // Now specify the actual mesh topology by processing the faces array 
    //
    const unsigned VERTS_PER_FACE = 4;
    for (unsigned i = 0; i < sizeof(faces)/sizeof(int); i += VERTS_PER_FACE) {
        //
        // Do some sanity checking. It is a good idea to keep this in your 
        // code for your personal sanity as well.
        //
        // Note that this loop is not changing the HbrMesh, it's purely validating
        // the topology that is about to be created below.
        //
        for (unsigned j = 0; j < VERTS_PER_FACE; j++) {
            OsdHbrVertex * origin      = hmesh->GetVertex(faces[i+j]);
            OsdHbrVertex * destination = hmesh->GetVertex(faces[i+((j+1)%VERTS_PER_FACE)]);
            OsdHbrHalfedge * opposite  = destination->GetEdge(origin);

            if(origin==NULL || destination==NULL) {
                std::cerr << 
                    " An edge was specified that connected a nonexistent vertex"
                    << std::endl;
                exit(1);
            }

            if(origin == destination) {
                std::cerr << 
                    " An edge was specified that connected a vertex to itself" 
                    << std::endl;
                exit(1);
            }

            if(opposite && opposite->GetOpposite() ) {
                std::cerr << 
                    " A non-manifold edge incident to more than 2 faces was found" 
                    << std::endl;
                exit(1);
            }

            if(origin->GetEdge(destination)) {
                std::cerr << 
                    " An edge connecting two vertices was specified more than once."
                    " It's likely that an incident face was flipped" 
                    << std::endl;
                exit(1);
            }
        }
        // 
        // Now, create current face given the number of verts per face and the 
        // face index data.
        //
//        OsdHbrFace * face = hmesh->NewFace(VERTS_PER_FACE, faces+i, 0);
        hmesh->NewFace(VERTS_PER_FACE, faces+i, 0);        

        //
        // If you had ptex data, you would set it here, for example
        //
        /* face->SetPtexIndex(ptexIndex) */

    }

    //
    // Apply some tags to drive the subdivision algorithm. Here we set the 
    // default boundary interpolation mode along with a corner sharpness. See 
    // the API and the renderman spec for the full list of available operations.
    //
    hmesh->SetInterpolateBoundaryMethod( OsdHbrMesh::k_InterpolateBoundaryEdgeOnly );
   
    OsdHbrVertex * v = hmesh->GetVertex(0);
    v->SetSharpness(2.7f);

    //
    // Finalize the mesh object. The Finish() call is a signal to the internals 
    // that optimizations can be made on the mesh data. 
    //
    hmesh->Finish();

    // has 3 elements per vertex (3 floats for position), is defined by the topology
    // in hmesh to level subdivisions, and has a bitset that indicates osd should use
    // adaptive subdivision.
    //
    OpenSubdiv::FarMeshFactory<OpenSubdiv::OsdVertex> meshFactory(hmesh, level, true);

    OpenSubdiv::FarMesh<OpenSubdiv::OsdVertex> *farMesh = 
        meshFactory.Create(false /*ptex data*/,  false /*fvar data*/);

    // hmesh is no longer needed
    delete hmesh;


    g_evalContext = new OsdEvalContext(farMesh, 3);

    // 
    // Setup camera positioning based on object bounds. This really has nothing
    // to do with OSD.
    //
    computeCenterAndSize(g_orgPositions, g_center, &g_size);

    //
    // Finally, make an explicit call to updateGeom() to force creation of the 
    // initial buffer objects for the first draw call.
    //
    g_evalContext->UpdateData(&g_orgPositions[0]);

    //
    std::cout << "about to tessellate\n";
    g_evalContext->TessellateIntoTriangles( &g_refinedTriangleIndices,
                                            &g_refinedPositions);
    std::cout << "done tessellating\n";

//    for (unsigned int i=0; i<g_refinedTriangleIndices.size(); ++i) {
//        std::cout << g_refinedTriangleIndices[i]<<"\n ";
//    }

    std::cout << "total triangle indices = " << g_refinedTriangleIndices.size() << " Last one is " <<  g_refinedTriangleIndices[g_refinedTriangleIndices.size()-1] << "\n";

//    for (unsigned int i=0; i<g_refinedPositions.size(); ++i) {
//        std::cout << g_refinedPositions[i]<<"\n ";
//    } 

    std::cout << "total refined positions = " << g_refinedPositions.size()/3 << "\n";


    //
    // The OsdVertexBuffer provides GL identifiers which can be bound in the 
    // standard way. Here we setup a single VAO and enable points
    // as an attribute on the vertex buffer and set the index buffer.
    //
    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glGenBuffers(1, &g_refinedPositionsBuf);
    glGenBuffers(1, &g_refinedTriangleIndicesBuf);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,  g_refinedTriangleIndicesBuf);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 g_refinedTriangleIndices.size() * sizeof(unsigned int),
                 &(g_refinedTriangleIndices[0]), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);    

    glBindBuffer(GL_ARRAY_BUFFER, g_refinedPositionsBuf);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof (GLfloat) * 3, 0);
    glBufferData(GL_ARRAY_BUFFER, g_refinedPositions.size() * sizeof(float) * 3,
                 NULL, GL_STREAM_DRAW);
    
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    checkGLErrors("End createOsdMesh");    
    std::cout << "End createOsdMesh\n";    
}

//
// ### Update Geometry and Subdivide

// This is where the magic happens. Given the initial subdivision table stored 
// in the OsdMesh, on every frame we can now send coarse point position updates 
// and recompute the subdivided surface based on the coarse animation.
//
void
updateGeom() 
{
//    std::cout << "Start updateGeom\n";    
    int nverts = (int)g_orgPositions.size() / 3;

    std::vector<float> vertex;
    vertex.reserve(nverts*3);

    const float *p = &g_orgPositions[0];

    //
    // Apply a simple deformer to the coarse mesh. We save the deformed points 
    // into a separate buffer to avoid accumulation of error. This 
    // loop really has nothing to do with OSD.
    // 
    float r = sin(g_frame*0.01f);
    for (int i = 0; i < nverts; ++i) {
//        float move = 0.05f*cosf(p[0]*20+g_frame*0.001f);
        float ct = cos(p[2] * r);
        float st = sin(p[2] * r);
        
        vertex.push_back(p[0]*ct + p[1]*st);
        vertex.push_back(-p[0]*st + p[1]*ct);
        vertex.push_back(p[2]);

        p += 3;
    }

    //
    // Send the animated coarse positions to the eval context,
    // it'll do the refinement
    //    
    g_evalContext->UpdateData(&vertex[0]);

    // re-get our refined triangle mesh
    g_refinedTriangleIndices.clear();
    g_refinedPositions.clear();         
    g_evalContext->TessellateIntoTriangles( &g_refinedTriangleIndices,
                                            &g_refinedPositions);

//    std::cout << "End updateGeom\n";        
}


//
// ### Draw the Mesh 

// Display handles all drawing per frame. We first call the setupForDisplay 
// helper method to setup some uninteresting GL state and then bind the mesh
// using the buffers provided by our OSD objects
//
void
display() 
{
//    std::cout << "Start display\n";
    setupForDisplay(g_width, g_height, g_size, g_center);

    //
    // Bind the GL vertex and index buffers
    //
    // Bind the GL vertex buffer and send newly
    // refined points down the pipe
    glBindBuffer(GL_ARRAY_BUFFER, g_refinedPositionsBuf);
    glBufferSubData(GL_ARRAY_BUFFER, 0,
                    g_refinedPositions.size() * sizeof(float) * 3,
                    &g_refinedPositions[0]);


    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_refinedTriangleIndicesBuf);

//    glColor3f(0.5, 0.5, 0.5);    
    glDrawElements(GL_TRIANGLES, g_refinedTriangleIndices.size()/3,
                   GL_UNSIGNED_INT, NULL);

//    glColor3f(0.2, 0.2, 0.2);            
//    glDrawElements(GL_LINES, g_refinedTriangleIndices.size()/2,
//                   GL_UNSIGNED_INT, NULL);

//    glColor3f(1.0, 1.0, 1.0);        
    glDrawElements(GL_POINTS, g_refinedTriangleIndices.size()/3,
                   GL_UNSIGNED_INT, NULL);    
    


    //
    // This isn't strictly necessary, but unbind the GL state
    //
    glUseProgram(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);    

    //glDisableClientState(GL_VERTEX_ARRAY);

    //
    // Draw the HUD/status text
    //
    //glColor3f(1, 1, 1);
    drawString(10, 10, "LEVEL = %d", (int)g_level);
    drawString(10, 30, "# of Vertices = %d", (int)g_refinedPositions.size()/3);
    drawString(10, 50, "KERNEL = CPU");
    drawString(10, 70, "SUBDIVISION = %s", "CATMARK");

    //
    // Finish the current frame
    //
    glFinish();

    checkGLErrors("End display");
//    std::cout << "End display\n";    
}


