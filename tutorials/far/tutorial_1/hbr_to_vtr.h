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

#include <hbr/mesh.h>
#include <hbr/bilinear.h>
#include <hbr/loop.h>
#include <hbr/catmark.h>
#include <hbr/vertexEdit.h>
#include <hbr/cornerEdit.h>
#include <hbr/holeEdit.h>

#include <opensubdiv/far/topologyRefinerFactory.h>

#include <typeinfo>
#include <cassert>

//------------------------------------------------------------------------------

inline bool
compareType(std::type_info const & t1, std::type_info const & t2) {

    if (t1==t2) {
        return true;
    }

    // On some systems, distinct instances of \c type_info objects compare equal if
    // their name() functions return equivalent strings.  On other systems, distinct
    // type_info objects never compare equal.  The latter can cause problems in the
    // presence of plugins loaded without RTLD_GLOBAL, because typeid(T) returns
    // different \c type_info objects for the same T in the two plugins.
    for (char const * p1 = t1.name(), *p2 = t2.name(); *p1 == *p2; ++p1, ++p2)
        if (*p1 == '\0')
            return true;
    return false;
}

//------------------------------------------------------------------------------
// Translates subdivision options from Hbr to SdcOptions
//
template <class T>
static OpenSubdiv::SdcType
getSdcOptions( OpenSubdiv::HbrMesh<T> const & mesh, OpenSubdiv::SdcOptions * options) {

    typedef OpenSubdiv::SdcOptions SdcOptions;

    typedef OpenSubdiv::HbrMesh<T>               HMesh;
    typedef OpenSubdiv::HbrSubdivision<T>        HSubdiv;
    typedef OpenSubdiv::HbrCatmarkSubdivision<T> HCatmarkSubdiv;

    OpenSubdiv::SdcType type;

    SdcOptions::TriangleSubdivision tris=SdcOptions::TRI_SUB_NORMAL;

    HSubdiv const * subdivision = mesh.GetSubdivision();

           if (compareType(typeid(*subdivision), typeid(OpenSubdiv::HbrBilinearSubdivision<T>))) {
        type = OpenSubdiv::TYPE_BILINEAR;
    } else if (compareType(typeid(*subdivision), typeid(OpenSubdiv::HbrCatmarkSubdivision<T>))) {
        type = OpenSubdiv::TYPE_CATMARK;
        HCatmarkSubdiv const * catmarkSudiv = dynamic_cast<HCatmarkSubdiv const *>(subdivision);
        switch (catmarkSudiv->GetTriangleSubdivisionMethod()) {
            case HCatmarkSubdiv::k_Normal: tris=SdcOptions::TRI_SUB_NORMAL; break;
            case HCatmarkSubdiv::k_Old:    tris=SdcOptions::TRI_SUB_OLD; break;
            case HCatmarkSubdiv::k_New:    tris=SdcOptions::TRI_SUB_NEW; break;
        }
    } else if (compareType(typeid(*subdivision), typeid(OpenSubdiv::HbrLoopSubdivision<T>))) {
        type = OpenSubdiv::TYPE_LOOP;
    } else
        assert(0);

    OpenSubdiv::SdcOptions::VtxBoundaryInterpolation  vvbi;
    switch (mesh.GetInterpolateBoundaryMethod()) {
        case HMesh::k_InterpolateBoundaryNone:          vvbi=SdcOptions::VTX_BOUNDARY_NONE; break;
        case HMesh::k_InterpolateBoundaryEdgeOnly:      vvbi=SdcOptions::VTX_BOUNDARY_EDGE_ONLY; break;
        case HMesh::k_InterpolateBoundaryEdgeAndCorner: vvbi=SdcOptions::VTX_BOUNDARY_EDGE_AND_CORNER; break;
        default:
            assert(0);
    }

    OpenSubdiv::SdcOptions::FVarBoundaryInterpolation  fvbi;
    switch (mesh.GetInterpolateBoundaryMethod()) {
        case HMesh::k_InterpolateBoundaryNone:          fvbi=SdcOptions::FVAR_BOUNDARY_BILINEAR; break;
        case HMesh::k_InterpolateBoundaryEdgeOnly:      fvbi=SdcOptions::FVAR_BOUNDARY_EDGE_ONLY; break;
        case HMesh::k_InterpolateBoundaryEdgeAndCorner: fvbi=SdcOptions::FVAR_BOUNDARY_EDGE_AND_CORNER; break;
        case HMesh::k_InterpolateBoundaryAlwaysSharp:   fvbi=SdcOptions::FVAR_BOUNDARY_ALWAYS_SHARP; break;
        default:
            assert(0);
    }

    OpenSubdiv::SdcOptions::CreasingMethod creaseMethod;
    switch (subdivision->GetCreaseSubdivisionMethod()) {
        case HSubdiv::k_CreaseNormal:  creaseMethod=OpenSubdiv::SdcOptions::CREASE_UNIFORM; break;
        case HSubdiv::k_CreaseChaikin: creaseMethod=OpenSubdiv::SdcOptions::CREASE_CHAIKIN; break;
    };

    options->SetVtxBoundaryInterpolation(vvbi);
    options->SetFVarBoundaryInterpolation(fvbi);
    options->SetCreasingMethod(creaseMethod);
    options->SetTriangleSubdivision(tris);
    options->SetNonManifoldInterpolation(OpenSubdiv::SdcOptions::NON_MANIFOLD_NONE);

    return type;
}

//------------------------------------------------------------------------------
// The HbrConverter is used to specialize FarTopologyRefinerFactory for the Hbr rep.
//
// Hbr is a half-edge topo rep, which by definition does not index edges
// uniquely. To remedy the problem, the converter uses a std::map to connect
// Hbr's half-edge pointers to unique indices.
//
// This remapping code is provided as an example of efficient implementation of
// the translation from an arbitrary topo rep to Vtr.
//
// Even though Vtr is capable of re-generating edge and vertex relationships on
// its own, this requires costly work that may be redundant if these relationships
// can be translated from the host rep.
//
template <class T> class HbrConverter {

public:
    // Constructor
    HbrConverter(OpenSubdiv::HbrMesh<T> const & hmesh) : _hmesh(hmesh) {

        _nfaces = _hmesh.GetNumFaces();
        _nverts = _hmesh.GetNumVertices();
        _type = getSdcOptions<OpenSubdiv::OsdVertex>(_hmesh, &_options);
    }

    // Returns the type of mesh (Bilinear, Catmark, Loop)
    OpenSubdiv::SdcType const & GetType() const {
        return _type;
    }

    // Returns subdivision options
    OpenSubdiv::SdcOptions const & GetOptions() const {
        return _options;
    }

    // The HbrMesh being converted
    OpenSubdiv::HbrMesh<T> const & GetHbrMesh() const {
        return _hmesh;
    }

    // Number of faces in the mesh (cached for efficiency)
    int GetNumFaces() const {
        return _nfaces;
    }

    // Number of vertices in the mesh (cached for efficiency)
    int GetNumVertices() const {
        return _nverts;
    }

    // Number of edges in the mesh
    int GetNumEdges() const {
        return (int)_edgeset.size();
    }

    // Returns a pointer to the Hbr halfege of index 'idx'
    OpenSubdiv::HbrHalfedge<T> const * GetEdge(int idx) const {
        return _edgeids[idx];
    }

    typedef std::map<OpenSubdiv::HbrHalfedge<T> const *, int> EdgeMap;

    // A map of unique edges between vertices
    EdgeMap & GetEdges() {
        return _edgeset;
    }

    // Must be called after the edge map has been populated
    void FinishEdgeMap() const {
        // Fudge const-ness because resizeComponentTopology() passes the converter
        // as a const. The alternative is to add an iteration loop over Hbr before,
        // which would waste time.
        EdgeVec * edges = const_cast<EdgeVec *>(&_edgeids);
        edges->resize(_edgeset.size());
        for (typename EdgeMap::const_iterator it=_edgeset.begin(); it!=_edgeset.end(); ++it) {
            (*edges)[it->second] = it->first;
        }
    }

    // Returns a unique edge index for a given half-edge
    int GetEdgeIndex(OpenSubdiv::HbrHalfedge<T> const * e) const {
        assert(e);
        typename EdgeMap::const_iterator it = _edgeset.find(e);
        if (it==_edgeset.end()) {
            assert(e->GetOpposite());
            it = _edgeset.find(e->GetOpposite());
        }
        assert(it!=_edgeset.end());
        return it->second;
    }

    // Returns the edgeVertIndex for a given Hbr halfedge 'e' and vertex 'v'
    int GetEdgeVertIndex(OpenSubdiv::HbrHalfedge<T> const * e, OpenSubdiv::HbrVertex<T> const * v) const {
        assert(e && v);
        if (_edgeset.find(e)==_edgeset.end()) {
            e = e->GetOpposite();
            assert(e);
        }
        return (v==e->GetOrgVertex() ? 0:1);
    }

private:

    typedef std::vector<OpenSubdiv::HbrHalfedge<T> const *> EdgeVec;

    OpenSubdiv::SdcType    _type;
    OpenSubdiv::SdcOptions _options;

    OpenSubdiv::HbrMesh<T> const & _hmesh;

    int _nfaces,
        _nverts;

    EdgeMap _edgeset;
    EdgeVec _edgeids;
};

typedef HbrConverter<OpenSubdiv::OsdVertex>               OsdHbrConverter;

typedef OpenSubdiv::HbrMesh<OpenSubdiv::OsdVertex>        OsdHbrMesh;
typedef OpenSubdiv::HbrSubdivision<OpenSubdiv::OsdVertex> OsdHbrSubdivision;
typedef OpenSubdiv::HbrVertex<OpenSubdiv::OsdVertex>      OsdHbrVertex;
typedef OpenSubdiv::HbrFace<OpenSubdiv::OsdVertex>        OsdHbrFace;
typedef OpenSubdiv::HbrHalfedge<OpenSubdiv::OsdVertex>    OsdHbrHalfedge;

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

template <>
void
FarTopologyRefinerFactory<OsdHbrConverter>::resizeComponentTopology(
    FarTopologyRefiner & refiner, OsdHbrConverter const & conv) {

    OsdHbrMesh const & hmesh = conv.GetHbrMesh();

    int nfaces = hmesh.GetNumFaces(),
        nverts = hmesh.GetNumVertices();

    OsdHbrConverter::EdgeMap & edges = const_cast<OsdHbrConverter &>(conv).GetEdges();
    assert(edges.size()==0);

    // Faces and face-verts
    setNumBaseFaces(refiner, nfaces);
    for (int i=0; i<nfaces; ++i) {

        OsdHbrFace const * f = hmesh.GetFace(i);

        int nv = f->GetNumVertices();
        assert(nv==4); // temporary until n-gons are supported

        setNumBaseFaceVertices(refiner, i, nv);

        for (int j=0; j<nv; ++j) {

            // index Hbr edge pointers in the map
            OsdHbrHalfedge const * e = f->GetEdge(j);
            if (e->IsBoundary() || (e->GetRightFace()->GetID()>f->GetID())) {
                int id = (int)edges.size();
                edges[e] = id;
            }
        }
    }

    conv.FinishEdgeMap();

    // Edges and edge-faces
    setNumBaseEdges(refiner, (int)edges.size());
    for (int i=0; i!=conv.GetNumEdges(); ++i) {
        OsdHbrHalfedge const * e = conv.GetEdge(i);
        setNumBaseEdgeFaces(refiner, i, e->GetRightFace() ? 2 : 1);
    }

    // Vertices and vert-faces and vert-edges
    setNumBaseVertices(refiner, nverts);
    for (int i=0; i<nverts; ++i) {

        OsdHbrVertex * v = hmesh.GetVertex(i);

        class GatherOperator : public OpenSubdiv::HbrHalfedgeOperator<OpenSubdiv::OsdVertex> {

                OsdHbrVertex const * _v;
        public:
            int vertEdgeCount,
                vertFaceCount;

            GatherOperator(OsdHbrVertex const * v) : _v(v), vertEdgeCount(0), vertFaceCount(0) { }

            virtual void operator() (OsdHbrHalfedge &e) {
                if (e.GetOrgVertex()==_v && e.GetFace())
                    ++vertFaceCount;
                ++vertEdgeCount;
            }
        };

        GatherOperator op(v);
        v->ApplyOperatorSurroundingEdges(op);

        setNumBaseVertexEdges(refiner, i, op.vertEdgeCount);
        setNumBaseVertexFaces(refiner, i, op.vertFaceCount);
    }
}

template <>
void
FarTopologyRefinerFactory<OsdHbrConverter>::assignComponentTopology(
    FarTopologyRefiner & refiner, OsdHbrConverter const & conv) {

    typedef FarTopologyRefiner::Index           Index;
    typedef FarTopologyRefiner::IndexArray      IndexArray;
    typedef FarTopologyRefiner::LocalIndex      LocalIndex;

    OsdHbrMesh const & hmesh = conv.GetHbrMesh();

    OsdHbrConverter::EdgeMap & edges = const_cast<OsdHbrConverter &>(conv).GetEdges();

    { // Face relations:
        int nfaces = getNumBaseFaces(refiner);
        for (int i=0; i < nfaces; ++i) {

            IndexArray dstFaceVerts = getBaseFaceVertices(refiner, i);
            IndexArray dstFaceEdges = getBaseFaceEdges(refiner, i);

            OsdHbrFace * f = hmesh.GetFace(i);

            for (int j = 0; j < dstFaceVerts.size(); ++j) {

                dstFaceVerts[j] = (int)f->GetVertex(j)->GetID();
                dstFaceEdges[j] = conv.GetEdgeIndex(f->GetEdge(j));
            }
        }
    }

    { // Edge relations
        for (OsdHbrConverter::EdgeMap::const_iterator it=edges.begin(); it!=edges.end(); ++it) {

            OsdHbrHalfedge const * e = it->first;
            int eidx = it->second;

            //  Edge-vertices:
            IndexArray dstEdgeVerts = getBaseEdgeVertices(refiner, eidx);
            dstEdgeVerts[0] = e->GetOrgVertex()->GetID();
            dstEdgeVerts[1] = e->GetDestVertex()->GetID();

            //  Edge-faces
            IndexArray dstEdgeFaces = getBaseEdgeFaces(refiner, eidx);
            dstEdgeFaces[0] = e->GetLeftFace()->GetID();
            // half-edges only have 2 faces incident to an edge (no non-manifold)
            if (e->GetRightFace()) {
                dstEdgeFaces[1] = e->GetRightFace()->GetID();
            }
        }
    }

    { // Vert relations
        for (int i=0; i<getNumBaseVertices(refiner); ++i) {

            OsdHbrVertex const * v = hmesh.GetVertex(i);

            // The Hbr operator gathers the indices of the faces and edges incident
            // to a vertex and populates the refiner topological relationships.
            class GatherOperator : public OpenSubdiv::HbrHalfedgeOperator<OpenSubdiv::OsdVertex> {

                OsdHbrConverter const & _conv;
                OsdHbrVertex const * _v;

                Index * _dstVertFaces,
                      * _dstVertEdges;

                LocalIndex * _dstVertInFaceIndices,
                           * _dstVertInEdgeIndices;
            public:

                GatherOperator(FarTopologyRefiner & refiner, OsdHbrConverter const & conv,
                    OsdHbrVertex const * v, int idx) : _conv(conv), _v(v) {

                    _dstVertFaces = getBaseVertexFaces(refiner, idx).begin(),
                    _dstVertEdges = getBaseVertexEdges(refiner, idx).begin();

                    _dstVertInFaceIndices = getBaseVertexFaceLocalIndices(refiner, idx).begin(),
                    _dstVertInEdgeIndices = getBaseVertexEdgeLocalIndices(refiner, idx).begin();
                }

                virtual void operator() (OsdHbrHalfedge &e) {

                    OsdHbrFace * f=e.GetFace();
                    if (f && (e.GetOrgVertex()==_v)) {
                        *_dstVertFaces++ = f->GetID();
                        for (int j=0; j<f->GetNumVertices(); ++j) {
                            if (f->GetVertex(j)==_v) {
                                *_dstVertInFaceIndices++ = j;
                                break;
                            }
                        }
                    }
                    *_dstVertEdges++ = _conv.GetEdgeIndex(&e);
                    *_dstVertInEdgeIndices++ = _conv.GetEdgeVertIndex(&e, _v);
                }
            };

            GatherOperator op(refiner, conv, v, i);
            v->ApplyOperatorSurroundingEdges(op);
        }
    }
}

template <>
void
FarTopologyRefinerFactory<OsdHbrConverter>::assignComponentTags(
    FarTopologyRefiner & refiner, OsdHbrConverter const & conv) {

    OsdHbrMesh const & hmesh = conv.GetHbrMesh();

    OsdHbrConverter::EdgeMap & edges = const_cast<OsdHbrConverter &>(conv).GetEdges();

    // Initialize edge sharpness
    for (OsdHbrConverter::EdgeMap::const_iterator it=edges.begin(); it!=edges.end(); ++it) {

        OsdHbrHalfedge const * e = it->first;

        float sharpness = e->GetSharpness();
        if (e->GetOpposite()) {
            sharpness = std::max(sharpness, e->GetOpposite()->GetSharpness());

        }
        setBaseEdgeSharpness(refiner, it->second, sharpness);
    }

    // Initialize vertex sharpness
    for (int i=0; i<getNumBaseVertices(refiner); ++i) {
        setBaseVertexSharpness(refiner, i, hmesh.GetVertex(i)->GetSharpness());
    }

    // XXXX Initialize h-edits
}

} // namespace OPENSUBDIV_VERSION
} // namespace OpenSubdiv

