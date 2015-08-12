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
// This tutorial shows how to safely create Hbr meshes from arbitrary topology.
// Because Hbr is a half-edge data structure, it cannot represent non-manifold
// topology. Ensuring that the geometry used is manifold is a requirement to use
// Hbr safely. This tutorial presents some simple tests to detect inappropriate
// topology.
//

#include <opensubdiv/hbr/mesh.h>
#include <opensubdiv/hbr/catmark.h>

#include <cstdio>

//------------------------------------------------------------------------------
struct Vertex {

    // Hbr minimal required interface ----------------------
    Vertex() { }

    Vertex(int /*i*/) { }

    Vertex(Vertex const & src) {
        _position[0] = src._position[0];
        _position[1] = src._position[1];
        _position[2] = src._position[2];
    }

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
// Non-manifold geometry from catmark_fan.h
//
//                     o
//                    /|
//                   / |
//                  /  |
//                 /   |
//                o    |
//                | f2 |
//                |    |
//       o--------+----o------------o
//      /         |   /            /
//     /          |  /            /
//    /    f0     | /     f1     /
//   /            |/            /
//  o------------ o------------o
//
// The shared edge of a fan is adjacent to 3 faces, and therefore non-manifold.
//
static float verts[8][3] = {{-1.0,  0.0, -1.0},
                            {-1.0,  0.0,  0.0},
                            { 0.0,  0.0,  0.0},
                            { 0.0,  0.0, -1.0},
                            { 1.0,  0.0,  0.0},
                            { 1.0,  0.0, -1.0},
                            { 0.0,  1.0,  0.0},
                            { 0.0,  1.0, -1.0}};

static int nverts = 8,
           nfaces = 3;

static int facenverts[3] = { 4, 4, 4 };

static int faceverts[12] = { 0, 1, 2, 3,
                             3, 2, 4, 5,
                             3, 2, 6, 7 };

//------------------------------------------------------------------------------
int main(int, char **) {

    OpenSubdiv::HbrCatmarkSubdivision<Vertex> * catmark =
        new OpenSubdiv::HbrCatmarkSubdivision<Vertex>();

    Hmesh * hmesh = new Hmesh(catmark);

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

    printf("Created a fan with %d faces and %d vertices.\n",
        hmesh->GetNumFaces(), hmesh->GetNumVertices());

    delete hmesh;
    delete catmark;
}

//------------------------------------------------------------------------------
