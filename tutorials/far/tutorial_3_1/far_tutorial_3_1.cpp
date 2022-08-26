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
// This tutorial shows how to interface a high-level topology representation
// with Far for better efficiency. In tutorial 0, we showed how to instantiate
// topology from a simple face-vertex list. Here we will show how to take
// advantage of more complex data structures.
//
// Many client applications that manipulate geometry use advanced data structures
// such as half-edge, quad-edge or winged-edge in order to represent complex
// topological relationships beyond the usual face-vertex lists. We can take
// advantage of this information.
//
// Far provides an advanced interface that allows such a client application to
// communicate advanced component relationships directly and avoid having Far
// rebuilding them redundantly.
//

#include <opensubdiv/far/topologyRefinerFactory.h>
#include <opensubdiv/far/primvarRefiner.h>

#include <cstdio>

//------------------------------------------------------------------------------

using namespace OpenSubdiv;

//------------------------------------------------------------------------------
//
// For this tutorial, we provide the complete topological representation of a
// simple pyramid. In our case, we store it as a simple sequence of integers,
// with the understanding that client-code would provide a fully implemented
// data-structure such as quad-edges or winged-edges.
//
// Pyramid geometry from catmark_pyramid.h - extended for this tutorial
//
static int g_nverts = 5,
           g_nedges = 8,
           g_nfaces = 5;

// vertex positions
static float g_verts[5][3] = {{ 0.0f,  0.0f,  2.0f},
                              { 0.0f, -2.0f,  0.0f},
                              { 2.0f,  0.0f,  0.0f},
                              { 0.0f,  2.0f,  0.0f},
                              {-2.0f,  0.0f,  0.0f}};

// number of vertices in each face
static int g_facenverts[5] = { 3, 3, 3, 3, 4 };

// index of face vertices
static int g_faceverts[16] = { 0, 1, 2,
                               0, 2, 3,
                               0, 3, 4,
                               0, 4, 1,
                               4, 3, 2, 1 };

// index of edge vertices (2 per edge)
static int g_edgeverts[16] = { 0, 1,
                               1, 2,
                               2, 0,
                               2, 3,
                               3, 0,
                               3, 4,
                               4, 0,
                               4, 1 };


// index of face edges
static int g_faceedges[16] = { 0, 1, 2,
                               2, 3, 4,
                               4, 5, 6,
                               6, 7, 0,
                               5, 3, 1, 7 };

// number of faces adjacent to each edge
static int g_edgenfaces[8] = { 2, 2, 2, 2, 2, 2, 2, 2 };

// index of faces incident to a given edge
static int g_edgefaces[16] = { 0, 3,
                               0, 4,
                               0, 1,
                               1, 4,
                               1, 2,
                               2, 4,
                               2, 3,
                               3, 4 };

// number of faces incident to each vertex
static int g_vertexnfaces[5] = { 4, 3, 3, 3, 3 };

// index of faces incident to each vertex
static int g_vertexfaces[25] = { 0, 1, 2, 3,
                                 0, 3, 4,
                                 0, 4, 1,
                                 1, 4, 2,
                                 2, 4, 3 };


// number of edges incident to each vertex
static int g_vertexnedges[5] = { 4, 3, 3, 3, 3 };

// index of edges incident to each vertex
static int g_vertexedges[25] = { 0, 2, 4, 6,
                                 1, 0, 7,
                                 2, 1, 3,
                                 4, 3, 5,
                                 6, 5, 7 };

// Edge crease sharpness
static float g_edgeCreases[8] = { 0.0f,
                                  2.5f,
                                  0.0f,
                                  2.5f,
                                  0.0f,
                                  2.5f,
                                  0.0f,
                                  2.5f };

//------------------------------------------------------------------------------
//
// Because existing client-code may not provide an exact match for the
// topological queries required by Far's interface, we can provide a converter
// class. This can be particularly useful for instance if the client
// data-structure requires additional relationships to be mapped. For instance,
// half-edge representations do not store unique edge indices and it can be
// difficult to traverse edges or faces adjacent to a given vertex.
//
// Using an intermediate wrapper class allows us to leverage existing
// relationships information from a mesh, and generate the missing components
// temporarily.
//
// For a practical example, you can look at the file 'hbr_to_vtr.h' in the same
// tutorial directory. This example implements a 'OsdHbrConverter' class as a
// way of interfacing PRman's half-edge representation to Far.
//
struct Converter {

public:

    Sdc::SchemeType GetType() const {
        return Sdc::SCHEME_CATMARK;
    }

    Sdc::Options GetOptions() const {
        Sdc::Options options;
        options.SetVtxBoundaryInterpolation(Sdc::Options::VTX_BOUNDARY_EDGE_ONLY);
        return options;
    }

    int GetNumFaces() const { return g_nfaces; }

    int GetNumEdges() const { return g_nedges; }

    int GetNumVertices() const { return g_nverts; }

    //
    // Face relationships
    //
    int GetNumFaceVerts(int face) const { return g_facenverts[face]; }

    int const * GetFaceVerts(int face) const { return g_faceverts+getCompOffset(g_facenverts, face); }

    int const * GetFaceEdges(int face) const { return g_faceedges+getCompOffset(g_facenverts, face); }


    //
    // Edge relationships
    //
    int const * GetEdgeVertices(int edge) const { return g_edgeverts+edge*2; }

    int GetNumEdgeFaces(int edge) const { return g_edgenfaces[edge]; }

    int const * GetEdgeFaces(int edge) const { return g_edgefaces+getCompOffset(g_edgenfaces, edge); }

    //
    // Vertex relationships
    //
    int GetNumVertexEdges(int vert) const { return g_vertexnedges[vert]; }

    int const * GetVertexEdges(int vert) const { return g_vertexedges+getCompOffset(g_vertexnedges, vert); }

    int GetNumVertexFaces(int vert) const { return g_vertexnfaces[vert]; }

    int const * GetVertexFaces(int vert) const { return g_vertexfaces+getCompOffset(g_vertexnfaces, vert); }

private:

    int getCompOffset(int const * comps, int comp) const {
        int ofs=0;
        for (int i=0; i<comp; ++i) {
            ofs += comps[i];
        }
        return ofs;
    }

};

//------------------------------------------------------------------------------

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Far {

template <>
bool
TopologyRefinerFactory<Converter>::resizeComponentTopology(
    TopologyRefiner & refiner, Converter const & conv) {

    // Faces and face-verts
    int nfaces = conv.GetNumFaces();
    setNumBaseFaces(refiner, nfaces);
    for (int face=0; face<nfaces; ++face) {

        int nv = conv.GetNumFaceVerts(face);
        setNumBaseFaceVertices(refiner, face, nv);
    }

   // Edges and edge-faces
    int nedges = conv.GetNumEdges();
    setNumBaseEdges(refiner, nedges);
    for (int edge=0; edge<nedges; ++edge) {

        int nf = conv.GetNumEdgeFaces(edge);
        setNumBaseEdgeFaces(refiner, edge, nf);
    }

    // Vertices and vert-faces and vert-edges
    int nverts = conv.GetNumVertices();
    setNumBaseVertices(refiner, nverts);
    for (int vert=0; vert<nverts; ++vert) {

        int ne = conv.GetNumVertexEdges(vert),
            nf = conv.GetNumVertexFaces(vert);
        setNumBaseVertexEdges(refiner, vert, ne);
        setNumBaseVertexFaces(refiner, vert, nf);
    }
    return true;
}

template <>
bool
TopologyRefinerFactory<Converter>::assignComponentTopology(
    TopologyRefiner & refiner, Converter const & conv) {

    using Far::IndexArray;

    { // Face relations:
        int nfaces = conv.GetNumFaces();
        for (int face=0; face<nfaces; ++face) {

            IndexArray dstFaceVerts = getBaseFaceVertices(refiner, face);
            IndexArray dstFaceEdges = getBaseFaceEdges(refiner, face);

            int const * faceverts = conv.GetFaceVerts(face);
            int const * faceedges = conv.GetFaceEdges(face);

            for (int vert=0; vert<conv.GetNumFaceVerts(face); ++vert) {
                dstFaceVerts[vert] = faceverts[vert];
                dstFaceEdges[vert] = faceedges[vert];
            }
        }
    }

    { // Edge relations
      //
      // Note: if your representation is unable to provide edge relationships
      //       (ex: half-edges), you can comment out this section and Far will
      //       automatically generate the missing information.
      //
        int nedges = conv.GetNumEdges();
        for (int edge=0; edge<nedges; ++edge) {

            //  Edge-vertices:
            IndexArray dstEdgeVerts = getBaseEdgeVertices(refiner, edge);
            dstEdgeVerts[0] = conv.GetEdgeVertices(edge)[0];
            dstEdgeVerts[1] = conv.GetEdgeVertices(edge)[1];

            //  Edge-faces
            IndexArray dstEdgeFaces = getBaseEdgeFaces(refiner, edge);
            for (int face=0; face<conv.GetNumEdgeFaces(face); ++face) {
                dstEdgeFaces[face] = conv.GetEdgeFaces(edge)[face];
            }
        }
    }

    { // Vertex relations
        int nverts = conv.GetNumVertices();
        for (int vert=0; vert<nverts; ++vert) {

            //  Vert-Faces:
            IndexArray vertFaces = getBaseVertexFaces(refiner, vert);
            //LocalIndexArray vertInFaceIndices = getBaseVertexFaceLocalIndices(refiner, vert);
            for (int face=0; face<conv.GetNumVertexFaces(vert); ++face) {
                vertFaces[face] = conv.GetVertexFaces(vert)[face];
            }

            //  Vert-Edges:
            IndexArray vertEdges = getBaseVertexEdges(refiner, vert);
            //LocalIndexArray vertInEdgeIndices = getBaseVertexEdgeLocalIndices(refiner, vert);
            for (int edge=0; edge<conv.GetNumVertexEdges(vert); ++edge) {
                vertEdges[edge] = conv.GetVertexEdges(vert)[edge];
            }
        }
    }

    populateBaseLocalIndices(refiner);

    return true;
};

template <>
bool
TopologyRefinerFactory<Converter>::assignComponentTags(
    TopologyRefiner & refiner, Converter const & conv) {

    // arbitrarily sharpen the 4 bottom edges of the pyramid to 2.5f
    for (int edge=0; edge<conv.GetNumEdges(); ++edge) {
        setBaseEdgeSharpness(refiner, edge, g_edgeCreases[edge]);
    }
    return true;
}

#ifdef _MSC_VER
template <>
void
TopologyRefinerFactory<Converter>::reportInvalidTopology(
    TopologyError /* errCode */, char const * msg, Converter const& /* mesh */) {

    //
    //  Optional topology validation error reporting:
    //      This method is called whenever the factory encounters topology validation
    //  errors. By default, nothing is reported
    //
    Warning(msg);
}
template <>
bool
TopologyRefinerFactory<Converter>::assignFaceVaryingTopology(
    TopologyRefiner & /* refiner */, Converter const & /* conv */) {

    // Because of the way MSVC++ specializes templated functions, we had to
    // remove the default stubs in Far::TopologyRefinerFactory. In this
    // example, no face-varying data is being added, but we still need to
    // implement a template specialization or MSVC++ linker fails.
    return true;
}
#endif

} // namespace Far

} // namespace OPENSUBDIV_VERSION
} // namespace OpenSubdiv

//------------------------------------------------------------------------------
//
// Vertex container implementation.
//
struct Vertex {

    // Minimal required interface ----------------------
    Vertex() { }

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

//------------------------------------------------------------------------------
int main(int, char **) {

    Converter conv;

    Far::TopologyRefiner * refiner =
        Far::TopologyRefinerFactory<Converter>::Create(conv,
                Far::TopologyRefinerFactory<Converter>::Options(conv.GetType(), conv.GetOptions()));


    int maxlevel = 5;

    // Uniformly refine the topology up to 'maxlevel'
    refiner->RefineUniform(Far::TopologyRefiner::UniformOptions(maxlevel));


    // Allocate a buffer for vertex primvar data. The buffer length is set to
    // be the sum of all children vertices up to the highest level of refinement.
    std::vector<Vertex> vbuffer(refiner->GetNumVerticesTotal());
    Vertex * verts = &vbuffer[0];


    // Initialize coarse mesh positions
    int nCoarseVerts = g_nverts;
    for (int i=0; i<nCoarseVerts; ++i) {
        verts[i].SetPosition(g_verts[i][0], g_verts[i][1], g_verts[i][2]);
    }


    // Interpolate vertex primvar data
    Far::PrimvarRefiner primvarRefiner(*refiner);

    Vertex * src = verts;
    for (int level = 1; level <= maxlevel; ++level) {
        Vertex * dst = src + refiner->GetLevel(level-1).GetNumVertices();
        primvarRefiner.Interpolate(level, src, dst);
        src = dst;
    }


    { // Output OBJ of the highest level refined -----------

        Far::TopologyLevel const & refLastLevel = refiner->GetLevel(maxlevel);

        int nverts = refLastLevel.GetNumVertices();
        int nfaces = refLastLevel.GetNumFaces();

        // Print vertex positions
        int firstOfLastVerts = refiner->GetNumVerticesTotal() - nverts;

        for (int vert = 0; vert < nverts; ++vert) {
            float const * pos = verts[firstOfLastVerts + vert].GetPosition();
            printf("v %f %f %f\n", pos[0], pos[1], pos[2]);
        }

        // Print faces
        for (int face = 0; face < nfaces; ++face) {

            Far::ConstIndexArray fverts = refLastLevel.GetFaceVertices(face);

            // all refined Catmark faces should be quads
            assert(fverts.size()==4);

            printf("f ");
            for (int vert=0; vert<fverts.size(); ++vert) {
                printf("%d ", fverts[vert]+1); // OBJ uses 1-based arrays...
            }
            printf("\n");
        }
    }

    delete refiner;
    return EXIT_SUCCESS;
}

//------------------------------------------------------------------------------
