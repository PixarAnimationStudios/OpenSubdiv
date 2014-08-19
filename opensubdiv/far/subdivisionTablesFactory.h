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

#ifndef FAR_SUBDIVISION_TABLES_FACTORY_H
#define FAR_SUBDIVISION_TABLES_FACTORY_H

#include "../version.h"

// note: currently, this file has to be included from meshFactory.h
#include "../far/subdivisionTables.h"
#include "../far/kernelBatch.h"

#include <cassert>
#include <utility>
#include <vector>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

template <class T, class U> class  FarBilinearSubdivisionTablesFactory;
template <class T, class U> class  FarCatmarkSubdivisionTablesFactory;
template <class T, class U> class  FarLoopSubdivisionTablesFactory;

/// \brief A specialized factory for FarSubdivisionTables
///
/// This factory is private to Far and should not be used by client code.
///
template <class T, class U> class FarSubdivisionTablesFactory {

public:
    typedef std::vector<FarMesh<U> const *> FarMeshVector;

    /// \brief Splices subdivision tables and batches from multiple meshes and returns them
    /// Client code is responsible for deallocation.
    static FarSubdivisionTables *Splice(FarMeshVector const &meshes, FarKernelBatchVector *resultBatches);

protected:
    friend class FarBilinearSubdivisionTablesFactory<T,U>;
    friend class FarCatmarkSubdivisionTablesFactory<T,U>;
    friend class FarLoopSubdivisionTablesFactory<T,U>;

    template <class X, class Y> friend class FarMeshFactory;

    typedef bool (*CompareVerticesOperator)(const HbrVertex<T> *, const HbrVertex<T> *);

    // This factory accumulates vertex topology data that will be shared among the
    // specialized subdivision scheme factories (Bilinear / Catmark / Loop).
    // It also populates the FarMeshFactory vertex remapping vector that ties the
    // Hbr vertex indices to the FarVertexEdit tables.
    FarSubdivisionTablesFactory( HbrMesh<T> const * mesh, int maxlevel, std::vector<int> & remapTable, CompareVerticesOperator compareVertices = CompareVertices );

    // Returns the number of coarse vertices found in the mesh
    int GetNumCoarseVertices() const {
        return (int)(_vertVertsList[0].size());
    }

    // Total number of face vertices up to 'level'
    int GetNumFaceVerticesTotal(int level) const {
        return sumList<HbrVertex<T> *>(_faceVertsList, level);
    }

    // Total number of edge vertices up to 'level'
    int GetNumEdgeVerticesTotal(int level) const {
        return sumList<HbrVertex<T> *>(_edgeVertsList, level);
    }

    // Total number of vertex vertices up to 'level'
    int GetNumVertexVerticesTotal(int level) const {
        return sumList<HbrVertex<T> *>(_vertVertsList, level);
    }

    // Valence summation up to 'level'
    int GetFaceVertsValenceSum() const { return _faceVertsValenceSum; }

    // Valence summation for face vertices
    int GetVertVertsValenceSum() const { return _vertVertsValenceSum; }

    // Minimum valence for coarse faces
    int GetMinCoarseFaceValence() const { return _minCoarseFaceValence; }

    // Maximum valence for coarse faces
    int GetMaxCoarseFaceValence() const { return _maxCoarseFaceValence; }

    // Number of coarse triangle faces
    int GetNumCoarseTriangleFaces() const { return _numCoarseTriangleFaces; }

    // Returns an integer based on the order in which the kernels are applied
    static int GetMaskRanking( unsigned char mask0, unsigned char mask1 );

    bool HasFractionalEdgeSharpness() const { return _hasFractionalEdgeSharpness; }

    bool HasFractionalVertexSharpness() const { return _hasFractionalVertexSharpness; }

    // Compares vertices based on their topological configuration
    // (see subdivisionTables::GetMaskRanking for more details)
    static bool CompareVertices( HbrVertex<T> const *x, HbrVertex<T> const *y );

    // Compare vertices operator
    CompareVerticesOperator _compareVertices;

    // Per-level counters and offsets for each type of vertex (face,edge,vert)
    std::vector<int> _faceVertIdx,
                     _edgeVertIdx,
                     _vertVertIdx;

    // Number of indices required for the face-vert and vertex-vert
    // iteration tables at each level
    int _faceVertsValenceSum,
        _vertVertsValenceSum;

    // lists of vertices sorted by type and level
    std::vector<std::vector< HbrVertex<T> *> > _faceVertsList,
                                               _edgeVertsList,
                                               _vertVertsList;

    // Minimum and maximum valence for coarse faces
    int _minCoarseFaceValence,
        _maxCoarseFaceValence;

    // Number of coarse triangle faces
    int _numCoarseTriangleFaces;

    // Indicates if an edge or vertex has a fractional (non-integer) sharpness
    bool _hasFractionalEdgeSharpness,
         _hasFractionalVertexSharpness;

private:

    // Returns the subdivision level of a vertex
    static int getVertexDepth(HbrVertex<T> * v);

    template <class Type> static int sumList( std::vector<std::vector<Type> > const & list, int level );

    // Sums the number of adjacent vertices required to interpolate a Vert-Vertex
    static int sumVertVertexValence(HbrVertex<T> * vertex);
};

template <class T, class U>
FarSubdivisionTablesFactory<T,U>::FarSubdivisionTablesFactory( HbrMesh<T> const * mesh, int maxlevel, std::vector<int> & remapTable, CompareVerticesOperator compareVertices ) :
    _compareVertices(compareVertices),
    _faceVertIdx(maxlevel+1,0),
    _edgeVertIdx(maxlevel+1,0),
    _vertVertIdx(maxlevel+1,0),
    _faceVertsValenceSum(0),
    _vertVertsValenceSum(0),
    _faceVertsList(maxlevel+1),
    _edgeVertsList(maxlevel+1),
    _vertVertsList(maxlevel+1),
    _minCoarseFaceValence(0),
    _maxCoarseFaceValence(0),
    _numCoarseTriangleFaces(0),
    _hasFractionalEdgeSharpness(false),
    _hasFractionalVertexSharpness(false)
 {
    assert( mesh );

    int numVertices = mesh->GetNumVertices();

    std::vector<int> faceCounts(maxlevel+1,0),
                     edgeCounts(maxlevel+1,0),
                     vertCounts(maxlevel+1,0);

    // First pass (vertices) : count the vertices of each type for each depth
    // up to maxlevel (values are dependent on topology).
    int maxvertid=-1;
    for (int i=0; i<numVertices; ++i) {

        HbrVertex<T> * v = mesh->GetVertex(i);
        assert(v);

        if (not v->IsConnected()) {
            continue;
        }

        int depth = getVertexDepth( v );

        if (depth>maxlevel)
            continue;

        if (depth==0 )
            vertCounts[depth]++;

        if (v->GetID()>maxvertid)
            maxvertid = v->GetID();

        if (v->GetParentFace()) {
            faceCounts[depth]++;
            int valence = v->GetParentFace()->GetNumVertices();
            _faceVertsValenceSum += valence;

            if (depth == 1) {
                _minCoarseFaceValence = (_minCoarseFaceValence == 0 ? valence : std::min(_minCoarseFaceValence, valence));
                _maxCoarseFaceValence = (_maxCoarseFaceValence == 0 ? valence : std::max(_maxCoarseFaceValence, valence));
                if (valence == 3)
                    ++_numCoarseTriangleFaces;
            }
        } else if (v->GetParentEdge()) {
            edgeCounts[depth]++;

            // Determine if any edges have fractional sharpness.
            float sharpness = v->GetParentEdge()->GetSharpness();
            if (sharpness > HbrHalfedge<T>::k_Smooth and sharpness < HbrHalfedge<T>::k_Sharp)
                _hasFractionalEdgeSharpness = true;
        } else if (v->GetParentVertex()) {
            vertCounts[depth]++;
            _vertVertsValenceSum+=sumVertVertexValence(v);
            float sharpness = v->GetParentVertex()->GetSharpness();
            if (sharpness > 0.0f and sharpness < 1.0f)
                _hasFractionalVertexSharpness = true;
        }
    }

    int nsingulars = (int)mesh->GetSplitVertices().size();
    vertCounts[0] -= nsingulars;

    // Per-level offset to the first vertex of each type in the global vertex map
    _vertVertsList[0].reserve( vertCounts[0] );
    for (int l=1; l<(maxlevel+1); ++l) {
        _faceVertIdx[l]= _vertVertIdx[l-1]+vertCounts[l-1];
        _edgeVertIdx[l]= _faceVertIdx[l]+faceCounts[l];
        _vertVertIdx[l]= _edgeVertIdx[l]+edgeCounts[l];

        _faceVertsList[l].reserve( faceCounts[l] );
        _edgeVertsList[l].reserve( edgeCounts[l] );
        _vertVertsList[l].reserve( vertCounts[l] );
    }

    // reset counters
    faceCounts.assign(maxlevel+1,0);
    edgeCounts.assign(maxlevel+1,0);

    remapTable.resize( maxvertid+1, -1);

    // Second pass (vertices) : calculate the starting indices of the sub-tables
    // (face, edge, verts...) and populate the remapping table.
    for (int i=0; i<numVertices; ++i) {

        HbrVertex<T> * v = mesh->GetVertex(i);
        assert(v);

        if (not v->IsConnected()) {
            remapTable[ v->GetID() ] = v->GetID();
            continue;
        }

        int depth = getVertexDepth( v );

        if (depth>maxlevel)
            continue;

        assert( remapTable[ v->GetID() ] == -1 );

        if (depth==0) {
            _vertVertsList[ depth ].push_back( v );
            remapTable[ v->GetID() ] = v->GetID();
        } else if (v->GetParentFace()) {
            remapTable[ v->GetID() ]=_faceVertIdx[depth]+faceCounts[depth]++;
            _faceVertsList[ depth ].push_back( v );
        } else if (v->GetParentEdge()) {
            remapTable[ v->GetID() ]=_edgeVertIdx[depth]+edgeCounts[depth]++;
            _edgeVertsList[ depth ].push_back( v );
        } else if (v->GetParentVertex()) {
            // vertices need to be sorted separately based on compute kernel :
            // the remapping step is done just after this
            _vertVertsList[ depth ].push_back( v );
        }
    }

    // Sort the the vertices that are the child of a vertex based on their weight
    // mask. The masks combinations are ordered so as to minimize the compute
    // kernel switching.(see subdivisionTables::GetMaskRanking for more details)
    for (size_t i=1; i<_vertVertsList.size(); ++i)
        std::sort( _vertVertsList[i].begin(), _vertVertsList[i].end(), _compareVertices );


    // These vertices still need a remapped index
    for (int l=1; l<(maxlevel+1); ++l)
        for (size_t i=0; i<_vertVertsList[l].size(); ++i)
            remapTable[ _vertVertsList[l][i]->GetID() ]=_vertVertIdx[l]+(int)i;

    // Remap singular vertices to their origin vertices
    std::vector<std::pair<int, int> > const & singulars = mesh->GetSplitVertices();
    for (int i=0; i<(int)singulars.size(); ++i) {
        remapTable[singulars[i].first]=singulars[i].second;
    }
}


template <class T, class U> int
FarSubdivisionTablesFactory<T,U>::getVertexDepth(HbrVertex<T> * v) {

    if (v->IsConnected()) {
        return v->GetFace()->GetDepth();
    } else {
        // Un-connected vertices do not have a face pointer, so we have to seek
        // the parent. Note : subdivision tables can only work with face-vertices,
        // so we assert out of the other types.
        HbrFace<T> * parent = v->GetParentFace();
        assert(parent);
        return parent->GetDepth()+1;
    }
}

template <class T, class U>
    template <class Type> int
FarSubdivisionTablesFactory<T,U>::sumList( std::vector<std::vector<Type> > const & list, int level) {

    level = std::min(level, (int)list.size()-1);
    int total = 0;
    for (int i=0; i<=level; ++i)
        total += (int)list[i].size();
    return total;
}

// The ranking matrix defines the order of execution for the various combinations
// of Corner, Crease, Dart and Smooth topological configurations. This matrix is
// somewhat arbitrary as it is possible to perform some permutations in the
// ordering without adverse effects, but it does try to minimize kernel switching
// during the exececution of Apply(). This table is identical for both the Loop
// and Catmull-Clark schemes.
//
// The matrix is derived from this table :
// Rules     +----+----+----+----+----+----+----+----+----+----+
//   Pass 0  | Dt | Sm | Sm | Dt | Sm | Dt | Sm | Cr | Co | Cr |
//   Pass 1  |    |    |    | Co | Co | Cr | Cr | Co |    |    |
// Kernel    +----+----+----+----+----+----+----+----+----+----+
//   Pass 0  | B  | B  | B  | B  | B  | B  | B  | A  | A  | A  |
//   Pass 1  |    |    |    | A  | A  | A  | A  | A  |    |    |
//           +----+----+----+----+----+----+----+----+----+----+
// Rank      | 0  | 1  | 2  | 3  | 4  | 5  | 6  | 7  | 8  | 9  |
//           +----+----+----+----+----+----+----+----+----+----+
// with :
//     - A : compute kernel applying k_Crease / k_Corner rules
//     - B : compute kernel applying k_Smooth / k_Dart rules
template <class T, class U> int
FarSubdivisionTablesFactory<T,U>::GetMaskRanking( unsigned char mask0, unsigned char mask1 ) {
    static short masks[4][4] = { {    0,    1,    6,    4 },
                                 { 0xFF,    2,    5,    3 },
                                 { 0xFF, 0xFF,    9,    7 },
                                 { 0xFF, 0xFF, 0xFF,    8 } };
    return masks[mask0][mask1];
}

// Sums the number of adjacent vertices required to interpolate a Vert-Vertex
template <class T, class U> int
FarSubdivisionTablesFactory<T,U>::sumVertVertexValence(HbrVertex<T> * vertex) {
    int masks[2], npasses=1, result=0;

    HbrVertex<T> * pv = vertex->GetParentVertex();
    assert(pv);

    masks[0] = pv->GetMask(false);
    masks[1] = pv->GetMask(true);

    if (masks[0] != masks[1]and (
                not (masks[0]==HbrVertex<T>::k_Smooth and
                     masks[1]==HbrVertex<T>::k_Dart)))
        npasses = 2;

    int valence = pv->GetValence();
    for (int i=0; i<npasses; ++i)
        switch (masks[i]) {
            case HbrVertex<T>::k_Smooth:
            case HbrVertex<T>::k_Dart: result+=valence; break;
            default: break;
        }

    return result;
}

// Compare the weight masks of 2 vertices using the following ordering table.
//
// Assuming 2 computer kernels :
//  - A handles the k_Crease and K_Corner rules
//  - B handles the K_Smooth and K_Dart rules
// The vertices should be sorted so as to minimize the number execution calls of
// these kernels to match the 2 pass interpolation scheme used in Hbr.
template <class T, class U> bool
FarSubdivisionTablesFactory<T,U>::CompareVertices( HbrVertex<T> const * x, HbrVertex<T> const * y ) {

    // Masks of the parent vertex decide for the current vertex.
    HbrVertex<T> * px=x->GetParentVertex(),
                 * py=y->GetParentVertex();

    assert( (GetMaskRanking(px->GetMask(false), px->GetMask(true) )!=0xFF) and
            (GetMaskRanking(py->GetMask(false), py->GetMask(true) )!=0xFF) );

    return GetMaskRanking(px->GetMask(false), px->GetMask(true) ) <
           GetMaskRanking(py->GetMask(false), py->GetMask(true) );
}

// splice subdivision tables
template <typename V, typename IT> static IT
copyWithOffset(IT dst_iterator, V const &src, int offset) {
    return std::transform(src.begin(), src.end(), dst_iterator,
                          std::bind2nd(std::plus<typename V::value_type>(), offset));
}

template <typename V, typename IT> static IT
copyWithPtexFaceOffset(IT dst_iterator, V const &src, int start, int count, int offset) {
    for (typename V::const_iterator it = src.begin()+start; it != src.begin()+start+count; ++it) {
        typename V::value_type ptexCoord = *it;
        ptexCoord.faceIndex += offset;
        *dst_iterator++ = ptexCoord;
    }
    return dst_iterator;
}

template <typename V, typename IT> static IT
copyWithOffsetF_ITa(IT dst_iterator, V const &src, int offset) {
    for (typename V::const_iterator it = src.begin(); it != src.end();) {
        *dst_iterator++ = *it++ + offset;   // offset to F_IT
        *dst_iterator++ = *it++;            // valence
    }
    return dst_iterator;
}

template <typename V, typename IT> static IT
copyWithOffsetE_IT(IT dst_iterator, V const &src, int offset) {
    for (typename V::const_iterator it = src.begin(); it != src.end(); ++it) {
        *dst_iterator++ = (*it == -1) ? -1 : (*it + offset);
    }
    return dst_iterator;
}

template <typename V, typename IT> static IT
copyWithOffsetV_ITa(IT dst_iterator, V const &src, int tableOffset, int vertexOffset) {
    for (typename V::const_iterator it = src.begin(); it != src.end();) {
        *dst_iterator++ = *it++ + tableOffset;   // offset to V_IT
        *dst_iterator++ = *it++;                 // valence
        *dst_iterator++ = (*it == -1) ? -1 : (*it + vertexOffset); ++it;
        *dst_iterator++ = (*it == -1) ? -1 : (*it + vertexOffset); ++it;
        *dst_iterator++ = (*it == -1) ? -1 : (*it + vertexOffset); ++it;
    }
    return dst_iterator;
}

template <class T, class U> FarSubdivisionTables*
FarSubdivisionTablesFactory<T,U>::Splice(FarMeshVector const &meshes, FarKernelBatchVector *batches ) {

    // count total table size
    size_t total_F_ITa = 0, total_F_IT = 0;
    size_t total_E_IT = 0, total_E_W = 0;
    size_t total_V_ITa = 0, total_V_IT = 0, total_V_W = 0;
    FarSubdivisionTables::Scheme scheme = FarSubdivisionTables::UNDEFINED;
    int maxLevel = 0;
    for (size_t i = 0; i < meshes.size(); ++i) {
        FarSubdivisionTables const * tables = meshes[i]->GetSubdivisionTables();
        assert(tables);

        total_F_ITa += tables->Get_F_ITa().size();
        total_F_IT  += tables->Get_F_IT().size();
        total_E_IT  += tables->Get_E_IT().size();
        total_E_W   += tables->Get_E_W().size();
        total_V_ITa += tables->Get_V_ITa().size();
        total_V_IT  += tables->Get_V_IT().size();
        total_V_W   += tables->Get_V_W().size();

        maxLevel = std::max(maxLevel, tables->GetMaxLevel()-1);
        if (scheme == FarSubdivisionTables::UNDEFINED) {
            scheme = tables->GetScheme();
        } else {
            assert(scheme == tables->GetScheme());
        }
    }

    // pad E_W to align with E_IT when only some meshes use CATMARK_RESTRICTED_EDGE_VERTEX kernel
    if (total_E_W != 0)
        total_E_W = total_E_IT / 2;

    FarSubdivisionTables *result = new FarSubdivisionTables(maxLevel, scheme);

    result->_F_ITa.resize(total_F_ITa);
    result->_F_IT.resize(total_F_IT);
    result->_E_IT.resize(total_E_IT);
    result->_E_W.resize(total_E_W);
    result->_V_ITa.resize(total_V_ITa);
    result->_V_IT.resize(total_V_IT);
    result->_V_W.resize(total_V_W);

    // compute table offsets;
    std::vector<int> vertexOffsets;
    std::vector<int> fvOffsets;
    std::vector<int> evOffsets;
    std::vector<int> vvOffsets;
    std::vector<int> F_IToffsets;
    std::vector<int> V_IToffsets;

    {
        int vertexOffset = 0;
        int F_IToffset = 0;
        int V_IToffset = 0;
        int fvOffset = 0;
        int evOffset = 0;
        int vvOffset = 0;
        for (size_t i = 0; i < meshes.size(); ++i) {
            FarSubdivisionTables const * tables = meshes[i]->GetSubdivisionTables();
            assert(tables);

            vertexOffsets.push_back(vertexOffset);
            F_IToffsets.push_back(F_IToffset);
            V_IToffsets.push_back(V_IToffset);
            fvOffsets.push_back(fvOffset);
            evOffsets.push_back(evOffset);
            vvOffsets.push_back(vvOffset);

            vertexOffset += meshes[i]->GetNumVertices();
            F_IToffset += (int)tables->Get_F_IT().size();
            fvOffset += (int)tables->Get_F_ITa().size()/2;
            V_IToffset += (int)tables->Get_V_IT().size();

            if (scheme == FarSubdivisionTables::CATMARK or
                scheme == FarSubdivisionTables::LOOP) {

                evOffset += (int)tables->Get_E_IT().size()/4;
                vvOffset += (int)tables->Get_V_ITa().size()/5;
            } else {

                evOffset += (int)tables->Get_E_IT().size()/2;
                vvOffset += (int)tables->Get_V_ITa().size();
            }
        }
    }

    // concat F_IT and V_IT
    std::vector<unsigned int>::iterator F_IT = result->_F_IT.begin();
    std::vector<unsigned int>::iterator V_IT = result->_V_IT.begin();

    for (size_t i = 0; i < meshes.size(); ++i) {
        FarSubdivisionTables const * tables = meshes[i]->GetSubdivisionTables();

        int vertexOffset = vertexOffsets[i];
        // remap F_IT, V_IT tables
        F_IT = copyWithOffset(F_IT, tables->Get_F_IT(), vertexOffset);
        V_IT = copyWithOffset(V_IT, tables->Get_V_IT(), vertexOffset);
    }

    // merge other tables
    std::vector<int>::iterator F_ITa = result->_F_ITa.begin();
    std::vector<int>::iterator E_IT  = result->_E_IT.begin();
    std::vector<float>::iterator E_W = result->_E_W.begin();
    std::vector<float>::iterator V_W = result->_V_W.begin();
    std::vector<int>::iterator V_ITa = result->_V_ITa.begin();

    for (size_t i = 0; i < meshes.size(); ++i) {
        FarSubdivisionTables const * tables = meshes[i]->GetSubdivisionTables();

        // copy face tables
        F_ITa = copyWithOffsetF_ITa(F_ITa, tables->Get_F_ITa(), F_IToffsets[i]);

        // copy edge tables
        E_IT = copyWithOffsetE_IT(E_IT, tables->Get_E_IT(), vertexOffsets[i]);
        if (not tables->Get_E_W().empty())
            E_W = copyWithOffset(E_W, tables->Get_E_W(), 0);
        else
            E_W += tables->Get_E_IT().size() / 2;

        // copy vert tables
        if (scheme == FarSubdivisionTables::CATMARK or
            scheme == FarSubdivisionTables::LOOP) {

            V_ITa = copyWithOffsetV_ITa(V_ITa, tables->Get_V_ITa(), V_IToffsets[i], vertexOffsets[i]);
        } else {

            V_ITa = copyWithOffset(V_ITa, tables->Get_V_ITa(), vertexOffsets[i]);
        }
        V_W = copyWithOffset(V_W, tables->Get_V_W(), 0);
    }

    // merge batch, model by model
    int editTableIndexOffset = 0;
    for (int i = 0; i < (int)meshes.size(); ++i) {
        int numBatches = (int)meshes[i]->GetKernelBatches().size();
        for (int j = 0; j < numBatches; ++j) {
            FarKernelBatch batch = meshes[i]->GetKernelBatches()[j];
            batch._meshIndex = i;
            batch._vertexOffset += vertexOffsets[i];

            if (batch._kernelType == FarKernelBatch::CATMARK_FACE_VERTEX or
                batch._kernelType == FarKernelBatch::BILINEAR_FACE_VERTEX) {

                batch._tableOffset += fvOffsets[i];

            } else if (batch._kernelType == FarKernelBatch::CATMARK_QUAD_FACE_VERTEX or
                       batch._kernelType == FarKernelBatch::CATMARK_TRI_QUAD_FACE_VERTEX) {

                batch._tableOffset += F_IToffsets[i];

            } else if (batch._kernelType == FarKernelBatch::CATMARK_EDGE_VERTEX or
                       batch._kernelType == FarKernelBatch::CATMARK_RESTRICTED_EDGE_VERTEX or
                       batch._kernelType == FarKernelBatch::LOOP_EDGE_VERTEX or
                       batch._kernelType == FarKernelBatch::BILINEAR_EDGE_VERTEX) {

                batch._tableOffset += evOffsets[i];

            } else if (batch._kernelType == FarKernelBatch::CATMARK_VERT_VERTEX_A1 or
                       batch._kernelType == FarKernelBatch::CATMARK_VERT_VERTEX_A2 or
                       batch._kernelType == FarKernelBatch::CATMARK_VERT_VERTEX_B or
                       batch._kernelType == FarKernelBatch::CATMARK_RESTRICTED_VERT_VERTEX_A or
                       batch._kernelType == FarKernelBatch::CATMARK_RESTRICTED_VERT_VERTEX_B1 or
                       batch._kernelType == FarKernelBatch::CATMARK_RESTRICTED_VERT_VERTEX_B2 or
                       batch._kernelType == FarKernelBatch::LOOP_VERT_VERTEX_A1 or
                       batch._kernelType == FarKernelBatch::LOOP_VERT_VERTEX_A2 or
                       batch._kernelType == FarKernelBatch::LOOP_VERT_VERTEX_B or
                       batch._kernelType == FarKernelBatch::BILINEAR_VERT_VERTEX) {

                batch._tableOffset += vvOffsets[i];

            } else if (batch._kernelType == FarKernelBatch::HIERARCHICAL_EDIT) {

                batch._tableIndex += editTableIndexOffset;
            }
            batches->push_back(batch);
        }
        editTableIndexOffset += meshes[i]->GetVertexEditTables() ?
            meshes[i]->GetVertexEditTables()->GetNumBatches() : 0;
    }

    // count verts offsets
    result->_vertsOffsets.resize(maxLevel+2);
    for (size_t i = 0; i < meshes.size(); ++i) {
        FarSubdivisionTables const * tables = meshes[i]->GetSubdivisionTables();
        for (size_t j = 0; j < tables->_vertsOffsets.size(); ++j) {
            result->_vertsOffsets[j] += tables->_vertsOffsets[j];
        }
    }

    return result;
}




} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* FAR_SUBDIVISION_TABLES_H */
