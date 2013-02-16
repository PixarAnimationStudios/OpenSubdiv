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
#include <osd/cpuGLVertexBuffer.h>
#include <osd/cpuComputeController.h>
#include <osd/cpuComputeContext.h>

// 
// ### Global Variables & Declarations
//

// The screen width & height; current frame for animation; and the desired 
// subdivision level.
//
int g_width = 0,
    g_height = 0,
    g_frame = 0,
    g_level = 4;

//
// A center point for the view matrix and the object size for framing
//
float g_center[3] = {0.0f, 0.0f, 0.0f},
      g_size = 0.0f;

//
// The OSD state: a mesh, vertex buffer and element array
//
OpenSubdiv::FarMesh<OpenSubdiv::OsdVertex> * g_farmesh = 0;
OpenSubdiv::OsdCpuGLVertexBuffer * g_vertexBuffer = 0;
OpenSubdiv::OsdGLDrawContext * g_drawContext = 0;
OpenSubdiv::OsdCpuComputeContext * g_osdComputeContext = 0;
OpenSubdiv::OsdCpuComputeController * g_osdComputeController = 0;

typedef OpenSubdiv::HbrMesh<OpenSubdiv::OsdVertex>     OsdHbrMesh;
typedef OpenSubdiv::HbrVertex<OpenSubdiv::OsdVertex>   OsdHbrVertex;
typedef OpenSubdiv::HbrFace<OpenSubdiv::OsdVertex>     OsdHbrFace;
typedef OpenSubdiv::HbrHalfedge<OpenSubdiv::OsdVertex> OsdHbrHalfedge;

//
// The coarse mesh positions and normals are saved externally and deformed
// during playback.
//
std::vector<float> g_orgPositions,
                   g_normals;

// 
// Forward declarations. These functions will be described below as they are 
// defined.
//
void idle();
void reshape(int width, int height);
void createOsdContext(int level);
void display();
void updateGeom();
static void calcNormals(OsdHbrMesh * mesh, 
                        std::vector<float> const & pos, 
                        std::vector<float> & result );

//
// ### The main program entry point
//

// register the Osd CPU kernel, 
// call createOsdMesh (see below), init glew and one-time GL state and enter the
// main drawing loop.
//
void initOsd() 
{
    initGL();
    // 
    // Dispatchers are created from a kernel enumeration via the factory pattern,
    // calling register here ensures that the CPU dispatcher will be available
    // for construction when it is requested via the kCPU enumeration inside the
    // function createOsdMesh.
    //
//    OpenSubdiv::OsdCpuKernelDispatcher::Register();
    g_osdComputeController = new OpenSubdiv::OsdCpuComputeController();

    //
    // The following method will populate the g_osdMesh object, which will 
    // contain the precomputed subdivision tables.
    //
    createOsdContext(g_level);

}

//
// ### Construct the OSD Mesh 

// Here is where the real meat of the OSD setup happens. The mesh topology is 
// created and stored for later use. Actual subdivision happens in updateGeom 
// which gets called at the end of this function and on frame change.
//
void
createOsdContext(int level)
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
        /* OsdHbrFace * face = */ hmesh->NewFace(VERTS_PER_FACE, faces+i, 0);

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
    // Setup some raw vectors of data. Remember that the actual point values were
    // not stored in the OsdVertex, so we keep track of them here instead
    //
    g_normals.resize(g_orgPositions.size(),0.0f);
    calcNormals( hmesh, g_orgPositions, g_normals );

    // 
    // At this point, we no longer need the topological structure of the mesh, 
    // so we bake it down into subdivision tables by converting the HBR mesh 
    // into an OSD mesh. Note that this is just storing the initial subdivision
    // tables, which will be used later during the actual subdivision process.
    //
    // Again, no vertex positions are being stored here, the point data will be 
    // sent to the mesh in updateGeom().
    //
    OpenSubdiv::FarMeshFactory<OpenSubdiv::OsdVertex> meshFactory(hmesh, level);

    g_farmesh = meshFactory.Create();

    g_osdComputeContext = OpenSubdiv::OsdCpuComputeContext::Create(g_farmesh);

    delete hmesh;

    // 
    // Initialize draw context and vertex buffer
    //
    g_vertexBuffer = 
        OpenSubdiv::OsdCpuGLVertexBuffer::Create(6,  /* 3 floats for position, 
                                                            +
                                                            3 floats for normal*/
                                                 g_farmesh->GetNumVertices());

    g_drawContext =
        OpenSubdiv::OsdGLDrawContext::Create(g_farmesh, g_vertexBuffer);

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
    // standard way. Here we setup a single VAO and enable points and normals 
    // as attributes on the vertex buffer and set the index buffer.
    //
    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, g_vertexBuffer->BindVBO());
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof (GLfloat) * 6, 0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof (GLfloat) * 6, (float*)12);
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
    const float *n = &g_normals[0];

    //
    // Apply a simple deformer to the coarse mesh. We save the deformed points 
    // and normals into a separate buffer to avoid accumulation of error. This 
    // loop really has nothing to do with OSD.
    // 
    float r = sin(g_frame*0.001f);
    for (int i = 0; i < nverts; ++i) {
        //float move = 0.05f*cosf(p[0]*20+g_frame*0.01f);
        float ct = cos(p[2] * r);
        float st = sin(p[2] * r);
        
        vertex.push_back(p[0]*ct + p[1]*st);
        vertex.push_back(-p[0]*st + p[1]*ct);
        vertex.push_back(p[2]);

        //
        // To be completely accurate, we should deform the normals here too, but
        // the original undeformed normals are sufficient for this example 
        //
        vertex.push_back(n[0]);
        vertex.push_back(n[1]);
        vertex.push_back(n[2]);
 
        p += 3;
        n += 3;
    }

    //
    // Send the animated coarse positions and normals to the vertex buffer.
    //
    std::cout << vertex.size() << " - " << nverts << std::endl;
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
    // g_osdComputeController->Synchronize();
}

//
// ### Calculate Face Normals

// A helper function to calculate face normals. It is included here to illustrate
// how to inspect the coarse mesh, give an HbrMesh pointer.
//
static void
calcNormals(OsdHbrMesh * mesh, 
            std::vector<float> const & pos,
            std::vector<float> & result ) 
{
    //
    // Get the number of vertices and faces. Notice the naming convention is 
    // different between coarse Vertices and Faces. This may change in the 
    // future (it an artifact of the original renderman code).
    //
    int nverts = mesh->GetNumVertices();
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
        normalize(&result[i*3]);
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

    OpenSubdiv::OsdPatchArrayVector const & patches = g_drawContext->patchArrays;
    for (int i=0; i<(int)patches.size(); ++i) {
        OpenSubdiv::OsdPatchArray const & patch = patches[i];

        //
        // Bind the solid shaded program and draw elements based on the buffer contents
        //
        bindProgram(g_quadFillProgram);

        glDrawElements(GL_LINES_ADJACENCY, patch.numIndices,
                       GL_UNSIGNED_INT, NULL);

        //
        // Draw the wire frame over the solid shaded mesh
        //
        bindProgram(g_quadLineProgram);
        glUniform4f(glGetUniformLocation(g_quadLineProgram, "fragColor"), 
                    0, 0, 0.5, 1);
        glDrawElements(GL_LINES_ADJACENCY, patch.numIndices,
                       GL_UNSIGNED_INT, NULL);
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


