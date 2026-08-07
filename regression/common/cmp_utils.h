//
//   Copyright 2015 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#ifndef CMP_UTILS_H
#define CMP_UTILS_H

#include <opensubdiv/far/topologyRefinerFactory.h>

#include "hbr_utils.h"

//------------------------------------------------------------------------------

namespace {
    template <class Face, class Edge, class Vertex>
    struct LevelMapT {
        std::vector<Face *>     faces;
        std::vector<Edge *>     edges;
        std::vector<Vertex *>   verts;
    };
};


// Copies vertex data from hbrMesh into hbrVertexData reordered to match
// the given refiner and subdivision level.  This is used for later easy 
// comparison between the two.
template<class T>
void
GetReorderedHbrVertexData(
    const OpenSubdiv::Far::TopologyRefiner &farRefiner,
    const OpenSubdiv::HbrMesh<T> &hbrMesh,
    std::vector<T> *hbrVertexData,
    std::vector<bool> *hbrVertexOnBoundaryData = NULL)
{
    typedef OpenSubdiv::HbrVertex<T>   Hvertex;
    typedef OpenSubdiv::HbrFace<T>     Hface;
    typedef OpenSubdiv::HbrHalfedge<T> Hhalfedge;

    struct Mapper {
        typedef LevelMapT<Hface, Hhalfedge, Hvertex> LevelMap;
        std::vector<LevelMap> maps;

        Mapper(const OpenSubdiv::Far::TopologyRefiner &refiner, 
               const OpenSubdiv::HbrMesh<T> &hmesh) {

            bool schemeIsLoop = (refiner.GetSchemeType() == OpenSubdiv::Sdc::SCHEME_LOOP);

            maps.resize(refiner.GetMaxLevel()+1);

            typedef OpenSubdiv::Far::Index Index;
            typedef OpenSubdiv::Far::ConstIndexArray ConstIndexArray;

            {   // Populate base level
                // note : topological ordering is identical between Hbr and Far
                // for the base level
                OpenSubdiv::Far::TopologyLevel const & refBaseLevel = refiner.GetLevel(0);

                int nfaces = refBaseLevel.GetNumFaces(),
                    nedges = refBaseLevel.GetNumEdges(),
                    nverts = refBaseLevel.GetNumVertices();

                maps[0].faces.resize(nfaces, 0);
                maps[0].edges.resize(nedges, 0);
                maps[0].verts.resize(nverts, 0);

                for (int face=0; face<nfaces; ++face) {
                    maps[0].faces[face] = hmesh.GetFace(face);
                }

                for (int edge = 0; edge <nedges; ++edge) {

                    ConstIndexArray farVerts = refBaseLevel.GetEdgeVertices(edge);

                    Hvertex const * v0 = hmesh.GetVertex(farVerts[0]),
                                  * v1 = hmesh.GetVertex(farVerts[1]);

                    Hhalfedge * e = v0->GetEdge(v1);
                    if (! e) {
                        e = v1->GetEdge(v0);
                    }
                    assert(e);

                    maps[0].edges[edge] = e;
                }

                for (int vert = 0; vert<nverts; ++vert) {
                    maps[0].verts[vert] = hmesh.GetVertex(vert);
                }
            }

            // Populate refined levels
            for (int level=1; level<=refiner.GetMaxLevel(); ++level) {

                LevelMap & previous = maps[level-1],
                         & current = maps[level];

                OpenSubdiv::Far::TopologyLevel const & refLevel     = refiner.GetLevel(level);
                OpenSubdiv::Far::TopologyLevel const & refPrevLevel = refiner.GetLevel(level-1);

                current.faces.resize(refLevel.GetNumFaces(), 0);
                current.edges.resize(refLevel.GetNumEdges(), 0);
                current.verts.resize(refLevel.GetNumVertices(), 0);

                for (int face=0; face < refPrevLevel.GetNumFaces(); ++face) {

                    // populate child faces
                    Hface * f = previous.faces[face];

                    ConstIndexArray childFaces = refPrevLevel.GetFaceChildFaces(face);
                    for (int i=0; i<childFaces.size(); ++i) {
                        current.faces[childFaces[i]] = f->GetChild(i);
                    }

                    // populate child face-verts -- when present (none for Loop subdivision)
                    if (!schemeIsLoop) {
                        Hvertex * v = f->Subdivide();
                        Index childVert = refPrevLevel.GetFaceChildVertex(face);
                        assert(v->GetParentFace());
                        current.verts[childVert] = v;
                    }
                }

                for (int edge=0; edge < refPrevLevel.GetNumEdges(); ++edge) {
                    // populate child edge-verts
                    Index childVert = refPrevLevel.GetEdgeChildVertex(edge);
                    Hhalfedge * e = previous.edges[edge];
                    Hvertex * v = e->Subdivide();
                    assert(v->GetParentEdge());
                    current.verts[childVert] = v;
                }

                for (int vert = 0; vert < refPrevLevel.GetNumVertices(); ++vert) {
                    // populate child vert-verts
                    Index childVert = refPrevLevel.GetVertexChildVertex(vert);
                    Hvertex * v = previous.verts[vert]->Subdivide();
                    current.verts[childVert] = v;
                    assert(v->GetParentVertex());
                }

                // populate child edges
                for (int edge=0; edge < refLevel.GetNumEdges(); ++edge) {

                    ConstIndexArray farVerts = refLevel.GetEdgeVertices(edge);

                    Hvertex const * v0 = current.verts[farVerts[0]],
                                  * v1 = current.verts[farVerts[1]];
                    assert(v0 && v1);

                    Hhalfedge * e= v0->GetEdge(v1);
                    if (! e) {
                        e = v1->GetEdge(v0);
                    }
                    assert(e);
                    current.edges[edge] = e;
                }
            }
        }
    };

    Mapper mapper(farRefiner, hbrMesh);

    int nverts = hbrMesh.GetNumVertices();
    assert( nverts==farRefiner.GetNumVerticesTotal() );

    hbrVertexData->resize(nverts);

    for (int level=0, ofs=0; level<(farRefiner.GetMaxLevel()+1); ++level) {

       typename Mapper::LevelMap & map = mapper.maps[level];
       for (int i=0; i<(int)map.verts.size(); ++i) {
            Hvertex * v = map.verts[i];
            if (hbrVertexOnBoundaryData) {
                (*hbrVertexOnBoundaryData)[ofs] = hbrVertexOnBoundary(v);
            }
            (*hbrVertexData)[ofs++] = v->GetData();
       }
    }

}

//------------------------------------------------------------------------------

#endif /* CMP_UTILS_H */
