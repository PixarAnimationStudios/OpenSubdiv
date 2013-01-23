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
    g_level = 7;

//
// A center point for the view matrix and the object size for framing
//
float g_center[3] = {0.0f, 0.0f, 0.0f},
      g_size = 0.0f;

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
// The coarse mesh positions and saved externally and deformed
// during playback.
//
std::vector<float> g_orgPositions;

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
// main glut loop.
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

void EvalBSpline(float u,
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



class MyPatch {
public:

    // Note that MyPatch retains a pointer to CVs and depends
    // on it remaining allocated.
    MyPatch(const unsigned int *CVs) {
        _cvs = CVs;
    }

    void Eval( float u, float v, simpleVec3 *vertexBuffer, 
               simpleVec3 *position,  simpleVec3 *utangent, simpleVec3 *vtangent) {
        EvalBSpline(u, v, vertexBuffer, _cvs,
                    *position, *utangent, *vtangent);
    }

    
    
    //  Packed patch control vertices
    //   0  1  2  3 
    //   4  5  6  7
    //   8  9 10 11
    //  12 13 14 15
    const unsigned int *_cvs; // 
};
 



class OsdEvalContext : OpenSubdiv::OsdNonCopyable<OsdEvalContext> {
public:
  explicit OsdEvalContext(OpenSubdiv::FarMesh<OpenSubdiv::OsdVertex> *farmesh);
    virtual ~OsdEvalContext() {}

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
  OpenSubdiv::FarMesh<OpenSubdiv::OsdVertex> *_farMesh;
};


OsdEvalContext::OsdEvalContext(OpenSubdiv::FarMesh<OpenSubdiv::OsdVertex> *farMesh)
    : _farMesh(farMesh)
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
    
    const OpenSubdiv::FarTable<unsigned int> &ptable =
        patchTables->GetFullRegularPatches();

    std::vector<MyPatch> patches;
    
    // Iterate over all patches in this table.  Don't worry about markers here,
    // those would tell use what level of subdivision the patch was created on.
    // Just iterate over all patches, this is in blocks of 16 unsigned ints
    // per patch.
    const unsigned int *vertIndices = ptable[0];
    for (int i=0; i<ptable.GetSize(); i+=16) {
        std::cout << "patch at index " << i << "\n";

        // Create a patch object from the next block
        // of 16 control point indices stored in
        // the patch table.
        MyPatch patch(vertIndices + i);
        patches.push_back(patch);
    }

    

    std::cout << "Made " << patches.size() << " patches\n";


    // Now compute control point positions and shading data on the
    // refined surface.
    //
    // First, create OpenSubdiv compute objects 

    OpenSubdiv::OsdCpuComputeContext *osdComputeContext =
        OpenSubdiv::OsdCpuComputeContext::Create(_farMesh);
    OpenSubdiv::OsdCpuComputeController osdComputeController;
    OpenSubdiv::OsdCpuVertexBuffer *osdVertexBuffer = 
        OpenSubdiv::OsdCpuVertexBuffer::Create(
            3 /* 3 floats for position*/ , _farMesh->GetNumVertices());
    

    //
    // Send the animated coarse positions to the vertex buffer.
    //
    osdVertexBuffer->UpdateData(vertexData, numVertices);

    
    //
    // Dispatch subdivision work based on the coarse vertex buffer. At this 
    // point, the assigned dispatcher will queue up work, potentially in many
    // worker threads. If the subdivided data is required for further
    // processing a call to Synchronize() will allow you to block until
    // the worker threads complete.
    //
    osdComputeController.Refine(osdComputeContext, osdVertexBuffer);


    //
    // Is the call to Synchronize() needed here?
    //
    osdComputeController.Synchronize();
    
    float *points = osdVertexBuffer->BindCpuBuffer();

    for (int i=0; i< patches.size(); ++i) {
        for (float u=0; u<1.0; u+=0.1) {
            std::cout << "\t";            
            for (float v=0; v<1.0; v+=0.1) {
                simpleVec3 position, utangent, vtangent;
                patches[i].Eval(u, v, (simpleVec3*)points,
                                &position, &utangent, &vtangent);
                std::cout << "(" << position[0] << "," <<
                    position[1] << "," <<  position[2] << "), \n";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }

    
    delete osdVertexBuffer;
}


//
// ### Construct the OSD Mesh 

// Here is where the real meat of the OSD setup happens. The mesh topology is 
// created and stored for later use. Actual subdivision happens in updateGeom 
// which gets called at the end of this function and on frame change.
//
void
createOsdMesh(int level)
{
    // 
    // Setup an OsdHbr mesh based on the desired subdivision scheme
    //
    static OpenSubdiv::HbrCatmarkSubdivision<OpenSubdiv::OsdVertex>  _catmark;
    OsdHbrMesh *hmesh(new OsdHbrMesh(&_catmark));

    //
    // Now that we have a mesh, we need to add verticies and define the topology.
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
        OsdHbrFace * face = hmesh->NewFace(VERTS_PER_FACE, faces+i, 0);

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

    // 
    // At this point, we no longer need the topological structure of the mesh, 
    // so we bake it down into subdivision tables and cubic patches by converting 
    // the HBR mesh  into an OSD mesh. Note that this is just storing the initial 
    // subdivision tables, which will be used later during the actual subdivision 
    // process.
    //
    // Again, no vertex positions are being stored here, the point data will be 
    // sent to the mesh in updateGeom().
    //

    // Create an OpenSubdiv mesh that uses a single thread on the CPU to compute,
    // has 3 elements per vertex (3 floats for position), is defined by the topology
    // in hmesh to level subdivisions, and has a bitset that indicates osd should use
    // adaptive subdivision.
    //
    OpenSubdiv::FarMeshFactory<OpenSubdiv::OsdVertex> meshFactory(hmesh, level, true);

    farmesh = meshFactory.Create(false /*ptex data*/,  false /*fvar data*/);

    // hmesh is no longer needed
    delete hmesh;


    OsdEvalContext evalContext(farmesh);

    // 
    // Setup camera positioning based on object bounds. This really has nothing
    // to do with OSD.
    //
    computeCenterAndSize(g_orgPositions, g_center, &g_size);

    //
    // Finally, make an explicit call to updateGeom() to force creation of the 
    // initial buffer objects for the first draw call.
    //
    updateGeom();

    //
    // The OsdVertexBuffer provides GL identifiers which can be bound in the 
    // standard way. Here we setup a single VAO and enable points
    // as an attribute on the vertex buffer and set the index buffer.
    //
    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, g_vertexBuffer->BindVBO());
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof (GLfloat) * 3, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_drawContext->patchIndexBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
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
    int nverts = (int)g_orgPositions.size() / 3;

    std::vector<float> vertex;
    vertex.reserve(nverts*6);

    const float *p = &g_orgPositions[0];

    //
    // Apply a simple deformer to the coarse mesh. We save the deformed points 
    // into a separate buffer to avoid accumulation of error. This 
    // loop really has nothing to do with OSD.
    // 
    float r = sin(g_frame*0.01f);
    for (int i = 0; i < nverts; ++i) {
        float move = 0.05f*cosf(p[0]*20+g_frame*0.01f);
        float ct = cos(p[2] * r);
        float st = sin(p[2] * r);
        
        vertex.push_back(p[0]*ct + p[1]*st);
        vertex.push_back(-p[0]*st + p[1]*ct);
        vertex.push_back(p[2]);

        p += 3;
    }

    //
    // Send the animated coarse positions to the vertex buffer.
    //
    g_vertexBuffer->UpdateData(&vertex[0], nverts);

    //
    // Dispatch subdivision work based on the coarse vertex buffer. At this 
    // point, the assigned dispatcher will queue up work, potentially in many
    // worker threads. If the subdivided data is required for further processing
    // a call to Synchronize() will allow you to block until the worker threads
    // complete.
    //
    g_osdComputeController->Refine(g_osdComputeContext, g_vertexBuffer);

    //
    // The call to Synchronize() is not actually necessary, it's being used
    // here only for illustration. 
    //
    // g_mesh->Synchronize();
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
    setupForDisplay(g_width, g_height, g_size, g_center);

    //
    // Bind the GL vertex and index buffers
    //
    glBindBuffer(GL_ARRAY_BUFFER, g_vertexBuffer->BindVBO());
    
    OpenSubdiv::OsdPatchArrayVector const & patches = 
      g_drawContext->patchArrays;

    for (int i=0; i<(int)patches.size(); ++i) {
        OpenSubdiv::OsdPatchArray const & patch = patches[i];

        //
        // Bind the solid shaded program and draw elements based on the buffer contents
        //
        bindProgram(g_quadFillProgram);

        glDrawElements(GL_POINTS, patch.numIndices,
                       GL_UNSIGNED_INT, 
                       (void *)(patch.firstIndex * sizeof(unsigned int)));
    }

    //
    // This isn't strictly necessary, but unbind the GL state
    //
    glUseProgram(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    //glDisableClientState(GL_VERTEX_ARRAY);

    //
    // Draw the HUD/status text
    //
    //glColor3f(1, 1, 1);
    drawString(10, 10, "LEVEL = %d", g_level);
    drawString(10, 30, "# of Vertices = %d", g_farmesh->GetNumVertices());
    drawString(10, 50, "KERNEL = CPU");
    drawString(10, 70, "SUBDIVISION = %s", "CATMARK");

    //
    // Finish the current frame
    //
    glFinish();
}


