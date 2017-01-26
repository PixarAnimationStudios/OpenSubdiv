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


//------------------------------------------------------------------------------
// Tutorial description:
//
// This tutorial shows how to subdivide uniformly a simple Hbr mesh. We are
// building upon previous tutorials and assuming a fully instantiated mesh:
// we start with an HbrMesh pointer initialized from the same pyramid shape
// used in hbr_tutorial_0.
//
// We then apply the Refine() function sequentially to all the faces in the
// mesh to generate several levels of uniform subdivision. The resulting data
// is then dumped to the terminal in Wavefront OBJ format for inspection.
// 

#include <opensubdiv/hbr/mesh.h>
#include <opensubdiv/hbr/catmark.h>

#include <cassert>
#include <cstdio>


//------------------------------------------------------------------------------
//
// For this tutorial, we have to flesh out the Vertex class further. Note that now
// the copy constructor, Clear() and AddwithWeight() methods have been
// implemented to interpolate our float3 position data.
//
// This vertex specialization pattern leaves client-code free to implement
// arbitrary vertex primvar data schemes (or none at all to conserve efficiency)
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

    void Clear( void * =0 ) {
        _position[0]=_position[1]=_position[2]=0.0f;
    }

    void AddWithWeight(Vertex const & src, float weight) {
        _position[0]+=weight*src._position[0];
        _position[1]+=weight*src._position[1];
        _position[2]+=weight*src._position[2];
    }

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

Hmesh * createMesh();

//------------------------------------------------------------------------------
int main(int, char **) {

    Hmesh * hmesh = createMesh();

    int maxlevel=2,    // 2 levels of subdivision
        firstface=0,   // marker to the first face index of level 2
        firstvertex=0; // marker to the first vertex index of level 2

    // Refine the mesh to 'maxlevel'
    for (int level=0; level<maxlevel; ++level) {

        // Total number of faces in the mesh, across all levels
        //
        // Note: this function iterates over the list of faces and can be slow
        int nfaces = hmesh->GetNumFaces();

        if (level==(maxlevel-1)) {
            // Save our vertex marker
            firstvertex = hmesh->GetNumVertices();
        }

        // Iterate over the faces of the current level of subdivision
        for (int face=firstface; face<nfaces; ++face) {

            Hface * f = hmesh->GetFace(face);

            // Note: hole tags would have to be dealt with here.
            f->Refine();
        }

        // Save our face index marker for the next level
        firstface = nfaces;
    }

    { // Output OBJ of the highest level refined -----------

        // Print vertex positions
        int nverts = hmesh->GetNumVertices();
        for (int vert=firstvertex; vert<nverts; ++vert) {
            float const * pos = hmesh->GetVertex(vert)->GetData().GetPosition();
            printf("v %f %f %f\n", pos[0], pos[1], pos[2]);
        }

        // Print faces
        for (int face=firstface; face<hmesh->GetNumFaces(); ++face) {

            Hface * f = hmesh->GetFace(face);

            assert(f->GetNumVertices()==4 );

            printf("f ");
            for (int vert=0; vert<4; ++vert) {

                // OBJ uses 1-based arrays
                printf("%d ", f->GetVertex(vert)->GetID() - firstvertex + 1);
            }
            printf("\n");
        }
    }

}

//------------------------------------------------------------------------------
// Creates an Hbr mesh
//
// see hbr_tutorial_0 and hbr_tutorial_1 for more details
//
Hmesh *
createMesh() {

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

    OpenSubdiv::HbrCatmarkSubdivision<Vertex> * catmark =
        new OpenSubdiv::HbrCatmarkSubdivision<Vertex>();

    Hmesh * hmesh = new Hmesh(catmark);

    // Populate the vertices
    Vertex v;
    for (int i=0; i<nverts; ++i) {
        v.SetPosition(verts[i][0], verts[i][1], verts[i][2]);
        hmesh->NewVertex(i, v);
    }

    // Create the topology
    int * fv = faceverts;
    for (int i=0; i<nfaces; ++i) {

        int nv = facenverts[i];

        bool valid = true;

        for(int j=0;j<nv;j++) {

            Hvertex const * origin      = hmesh->GetVertex(fv[j]),
                          * destination = hmesh->GetVertex(fv[(j+1)%nv]);
            Hhalfedge const * opposite = destination->GetEdge(origin);

            // Make sure that the vertices exist in the mesh
            if (origin==NULL || destination==NULL) {
                printf(" An edge was specified that connected a nonexistent vertex\n");
                valid=false;
                break;
            }

            // Check for a degenerate edge
            if (origin == destination) {
                printf(" An edge was specified that connected a vertex to itself\n");
                valid=false;
                break;
            }

            // Check that no more than 2 faces are adjacent to the edge
            if (opposite && opposite->GetOpposite() ) {
                printf(" A non-manifold edge incident to more than 2 faces was found\n");
                valid=false;
                break;
            }

            // Check that the edge is unique and oriented properly
            if (origin->GetEdge(destination)) {
                printf(" An edge connecting two vertices was specified more than once."
                       " It's likely that an incident face was flipped\n");
                valid=false;
                break;
            }
        }

        if (valid) {
            hmesh->NewFace(nv, fv, 0);
        } else {
            printf(" Skipped face %d\n", i);
        }

        fv+=nv;
    }

    hmesh->SetInterpolateBoundaryMethod(Hmesh::k_InterpolateBoundaryEdgeOnly);

    hmesh->Finish();

    return hmesh;
}


//------------------------------------------------------------------------------
