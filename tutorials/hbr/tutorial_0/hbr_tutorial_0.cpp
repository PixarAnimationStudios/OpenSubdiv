//
//   Copyright 2013 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//


//------------------------------------------------------------------------------
// Tutorial description:
//
// This tutorial presents in a very succinct way the requisite steps to
// instantiate an Hbr mesh from simple topological data.
//

#include <opensubdiv/hbr/mesh.h>
#include <opensubdiv/hbr/catmark.h>

#include <cstdio>

//------------------------------------------------------------------------------
// Vertex container implementation.
//
// The HbrMesh<T> class is a templated interface that expects a vertex class to
// perform interpolation on arbitrary vertex data.
//
// For the template specialization of the HbrMesh interface to be met, our
// Vertex object to implement a minimal set of constructors and member
// functions.
//
// Since we are not going to subdivide the mesh, the struct presented here has
// been left minimalistic. The only customization added to our container was to
// provide storage and accessors for the position of a 3D vertex.
//
struct Vertex {

    // Hbr minimal required interface ----------------------
    Vertex() { }

    Vertex(int /*i*/) { }

    Vertex(Vertex const & src) {
        _position[0] = src._position[0];
        _position[1] = src._position[1];
        _position[2] = src._position[2];
    }

    Vertex & operator= (Vertex const & other) = default;

    void Clear( void * =0 ) { }

    void AddWithWeight(Vertex const &, float ) { }

    void AddVaryingWithWeight(Vertex const &, float) { }

    // Public interface ------------------------------------
    void SetPosition(float x, float y, float z) {
        _position[0]=x;
        _position[1]=y;
        _position[2]=z;
    }

    const float * GetPosition() const {
        return _position;
    }

private:
    float _position[3];
};

typedef OpenSubdiv::HbrMesh<Vertex>      Hmesh;
typedef OpenSubdiv::HbrFace<Vertex>      Hface;
typedef OpenSubdiv::HbrVertex<Vertex>    Hvertex;
typedef OpenSubdiv::HbrHalfedge<Vertex>  Hhalfedge;


//------------------------------------------------------------------------------
// Pyramid geometry from catmark_pyramid.h
static float verts[5][3] = {{ 0.0f,  0.0f,  2.0f},
                            { 0.0f, -2.0f,  0.0f},
                            { 2.0f,  0.0f,  0.0f},
                            { 0.0f,  2.0f,  0.0f},
                            {-2.0f,  0.0f,  0.0f}};

static int nverts = 5,
           nfaces = 5;

static int facenverts[5] = { 3, 3, 3, 3, 4 };

static int faceverts[16] = { 0, 1, 2,
                             0, 2, 3,
                             0, 3, 4,
                             0, 4, 1,
                             4, 3, 2, 1 };

//------------------------------------------------------------------------------
int main(int, char **) {

    // Create a subdivision scheme (Catmull-Clark here)
    OpenSubdiv::HbrCatmarkSubdivision<Vertex> * catmark =
        new OpenSubdiv::HbrCatmarkSubdivision<Vertex>();

    // Create an empty Hbr mesh
    Hmesh * hmesh = new Hmesh(catmark);

    // Populate the vertices
    Vertex v;
    for (int i=0; i<nverts; ++i) {

        // Primitive variable data must be set here: in our case we set
        // the 3D position of the vertex.
        v.SetPosition(verts[i][0], verts[i][1], verts[i][2]);

        // Add the vertex to the mesh.
        hmesh->NewVertex(i, v);
    }

    // Create the topology
    int * fv = faceverts;
    for (int i=0; i<nfaces; ++i) {

        int nv = facenverts[i];

        hmesh->NewFace(nv, fv, 0);

        fv+=nv;
    }

    // Set subdivision options
    //
    // By default vertex interpolation is set to "none" on boundaries, which
    // can produce un-expected results, so we change it to "edge-only".
    //
    hmesh->SetInterpolateBoundaryMethod(Hmesh::k_InterpolateBoundaryEdgeOnly);

    // Call 'Finish' to finalize the data structures before using the mesh.
    hmesh->Finish();


    printf("Created a pyramid with %d faces and %d vertices.\n",
        hmesh->GetNumFaces(), hmesh->GetNumVertices());

    delete hmesh;
    delete catmark;
}

//------------------------------------------------------------------------------
