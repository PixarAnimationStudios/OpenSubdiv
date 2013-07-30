//
//     Copyright 2013 Pixar
//
//     Licensed under the Apache License, Version 2.0 (the "License");
//     you may not use this file except in compliance with the License
//     and the following modification to it: Section 6 Trademarks.
//     deleted and replaced with:
//
//     6. Trademarks. This License does not grant permission to use the
//     trade names, trademarks, service marks, or product names of the
//     Licensor and its affiliates, except as required for reproducing
//     the content of the NOTICE file.
//
//     You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
//     Unless required by applicable law or agreed to in writing,
//     software distributed under the License is distributed on an
//     "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
//     either express or implied.  See the License for the specific
//     language governing permissions and limitations under the
//     License.
//

#ifndef FAR_SUBDIVISION_TABLES_FACTORY_H
#define FAR_SUBDIVISION_TABLES_FACTORY_H

#include "../version.h"

#include "../far/meshFactory.h"
#include "../far/subdivisionTables.h"

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

protected:
    friend class FarBilinearSubdivisionTablesFactory<T,U>;
    friend class FarCatmarkSubdivisionTablesFactory<T,U>;
    friend class FarLoopSubdivisionTablesFactory<T,U>;

    template <class X, class Y> friend class FarMeshFactory;
    
    // This factory accumulates vertex topology data that will be shared among the
    // specialized subdivision scheme factories (Bilinear / Catmark / Loop).
    // It also populates the FarMeshFactory vertex remapping vector that ties the
    // Hbr vertex indices to the FarVertexEdit tables.
    FarSubdivisionTablesFactory( HbrMesh<T> const * mesh, int maxlevel, std::vector<int> & remapTable );

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

    // Returns an integer based on the order in which the kernels are applied
    static int GetMaskRanking( unsigned char mask0, unsigned char mask1 );

    // Per-level counters and offsets for each type of vertex (face,edge,vert)
    std::vector<int> _faceVertIdx,
                     _edgeVertIdx,
                     _vertVertIdx;

    // Mumber of indices required for the face-vert and vertex-vert
    // iteration tables at each level
    int _faceVertsValenceSum, 
        _vertVertsValenceSum;

    // lists of vertices sorted by type and level
    std::vector<std::vector< HbrVertex<T> *> > _faceVertsList,
                                               _edgeVertsList,
                                               _vertVertsList;
private:

    // Returns the subdivision level of a vertex
    static int getVertexDepth(HbrVertex<T> * v);

    template <class Type> static int sumList( std::vector<std::vector<Type> > const & list, int level );

    // Sums the number of adjacent vertices required to interpolate a Vert-Vertex 
    static int sumVertVertexValence(HbrVertex<T> * vertex);

    // Compares vertices based on their topological configuration 
    // (see subdivisionTables::GetMaskRanking for more details)
    static bool compareVertices( HbrVertex<T> const *x, HbrVertex<T> const *y );
};

template <class T, class U> 
FarSubdivisionTablesFactory<T,U>::FarSubdivisionTablesFactory( HbrMesh<T> const * mesh, int maxlevel, std::vector<int> & remapTable ) :
    _faceVertIdx(maxlevel+1,0),
    _edgeVertIdx(maxlevel+1,0),
    _vertVertIdx(maxlevel+1,0),
    _faceVertsValenceSum(0), 
    _vertVertsValenceSum(0),
    _faceVertsList(maxlevel+1),
    _edgeVertsList(maxlevel+1),
    _vertVertsList(maxlevel+1)
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

        int depth = getVertexDepth( v );

        if (depth>maxlevel)
            continue;

        if (depth==0 )
            vertCounts[depth]++;

        if (v->GetID()>maxvertid)
            maxvertid = v->GetID();

        if (v->GetParentFace()) {
            faceCounts[depth]++;
            _faceVertsValenceSum += v->GetParentFace()->GetNumVertices();
        } else if (v->GetParentEdge())
            edgeCounts[depth]++;
        else if (v->GetParentVertex()) {
            vertCounts[depth]++;
            _vertVertsValenceSum+=sumVertVertexValence(v);
        }
    }

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
        std::sort( _vertVertsList[i].begin(), _vertVertsList[i].end(), compareVertices );


    // These vertices still need a remapped index
    for (int l=1; l<(maxlevel+1); ++l)
        for (size_t i=0; i<_vertVertsList[l].size(); ++i)
            remapTable[ _vertVertsList[l][i]->GetID() ]=_vertVertIdx[l]+(int)i;


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
FarSubdivisionTablesFactory<T,U>::compareVertices( HbrVertex<T> const * x, HbrVertex<T> const * y ) {

    // Masks of the parent vertex decide for the current vertex.
    HbrVertex<T> * px=x->GetParentVertex(),
                 * py=y->GetParentVertex();

    assert( (GetMaskRanking(px->GetMask(false), px->GetMask(true) )!=0xFF) and
            (GetMaskRanking(py->GetMask(false), py->GetMask(true) )!=0xFF) );

    return GetMaskRanking(px->GetMask(false), px->GetMask(true) ) <
           GetMaskRanking(py->GetMask(false), py->GetMask(true) );
}


} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* FAR_SUBDIVISION_TABLES_H */
