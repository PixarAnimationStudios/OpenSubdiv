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

#ifndef FAR_SUBDIVISION_TABLES_H
#define FAR_SUBDIVISION_TABLES_H

#include "../version.h"

#include <cassert>
#include <utility>
#include <vector>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

template <class U> class FarMesh;

/// \brief FarSubdivisionTables are a serialized topological data representation.
///
/// Subdivision tables store the indexing tables required in order to compute
/// the refined positions of a mesh without the help of a hierarchical data
/// structure. The advantage of this representation is its ability to be executed
/// in a massively parallel environment without data dependencies.
///
/// The vertex indexing tables require the vertex buffer to be sorted based on the
/// nature of the parent of a given vertex : either a face, an edge, or a vertex.
/// (note : the Loop subdivision scheme does not create vertices as a child of a
/// face).
///
/// Each type of vertex in the buffer is associated the following tables :
/// - _<T>_IT : indices of all the adjacent vertices required by the compute kernels
/// - _<T>_W : fractional weight of the vertex (based on sharpness & topology)
/// - _<T>_ITa : codex for the two previous tables
/// (where T denotes a face-vertex / edge-vertex / vertex-vertex)
///
///
/// Because each subdivision scheme (Catmark / Loop / Bilinear) introduces variations
/// in the subdivision rules, a derived class specialization is associated with
/// each scheme.
///
/// For more details see : "Feature Adaptive GPU Rendering of Catmull-Clark
/// Subdivision Surfaces"  (p.3 - par. 3.2)
///
template <class U> class FarSubdivisionTables {

public:
    enum TableType {
        E_IT,  ///< edge-vertices adjacency indexing table
        E_W,   ///< edge-vertices weights
        
        V_ITa, ///< vertex-vertices adjacency indexing table
        V_IT,  ///< vertex-vertices indexing table
        V_W,   ///< vertex-vertices weights
        
        F_ITa, ///< face-vertices adjacency indexing table
        F_IT,  ///< face-vertices indexing table
        
        TABLE_TYPES_COUNT  // number of different types of tables
    };

    /// \brief Destructor
    virtual ~FarSubdivisionTables<U>() {}

    /// \brief Return the highest level of subdivision possible with these tables
    int GetMaxLevel() const { return (int)(_vertsOffsets.size()-1); }

    /// \brief Memory required to store the indexing tables
    int GetMemoryUsed() const;

    /// \brief Pointer back to the mesh owning the table
    FarMesh<U> * GetMesh() { return _mesh; }

    /// \brief The index of the first vertex that belongs to the level of subdivision
    /// represented by this set of FarCatmarkSubdivisionTables
    int GetFirstVertexOffset( int level ) const;

    /// \brief Returns the total number of vertex adressed by the tables (this is the
    /// length that a vertex buffer object should be allocating
    int GetNumVertices( ) const;

    /// \brief Returns the number of vertices at a given level
    int GetNumVertices( int level ) const;

    /// \brief Returns the summation of the number of vertices up to a given level
    int GetNumVerticesTotal( int level ) const;

    // Indexing tables accessors

    /// \brief Returns the face vertices codex table
    std::vector<int> const &          Get_F_ITa( ) const { return _F_ITa; }

    /// \brief Returns the face vertices indexing table
    std::vector<unsigned int> const & Get_F_IT( ) const { return _F_IT; }

    /// \brief Returns the edge vertices indexing table
    std::vector<int> const &          Get_E_IT() const { return _E_IT; }

    /// \brief Returns the edge vertices weights table
    std::vector<float> const &        Get_E_W() const { return _E_W; }

    /// \brief Returns the vertex vertices codex table
    std::vector<int> const &          Get_V_ITa() const { return _V_ITa; }

    /// \brief Returns the vertex vertices indexing table
    std::vector<unsigned int> const & Get_V_IT() const { return _V_IT; }

    /// \brief Returns the vertex vertices weights table
    std::vector<float> const &        Get_V_W() const { return _V_W; }

    /// \brief Returns the number of indexing tables needed to represent this particular
    /// subdivision scheme.
    virtual int GetNumTables() const { return 5; }

protected:
    template <class X, class Y> friend class FarMeshFactory;
    template <class X, class Y> friend class FarMultiMeshFactory;

    FarSubdivisionTables<U>( FarMesh<U> * mesh, int maxlevel );

    // mesh that owns this subdivisionTable
    FarMesh<U> * _mesh;

    std::vector<int>          _F_ITa; // vertices from face refinement
    std::vector<unsigned int> _F_IT;  // indices of face vertices

    std::vector<int>          _E_IT;  // vertices from edge refinement
    std::vector<float>        _E_W;   // weigths

    std::vector<int>          _V_ITa; // vertices from vertex refinement
    std::vector<unsigned int> _V_IT;  // indices of adjacent vertices
    std::vector<float>        _V_W;   // weights

    std::vector<int> _vertsOffsets; // offset to the first vertex of each level
};

template <class U>
FarSubdivisionTables<U>::FarSubdivisionTables( FarMesh<U> * mesh, int maxlevel ) :
    _mesh(mesh),
    _vertsOffsets(maxlevel+2, 0)
{
    assert( maxlevel > 0 );
}

template <class U> int
FarSubdivisionTables<U>::GetFirstVertexOffset( int level ) const {
    assert(level>=0 and level<(int)_vertsOffsets.size());
    return _vertsOffsets[level];
}

template <class U> int
FarSubdivisionTables<U>::GetNumVertices( ) const {
    if (_vertsOffsets.empty()) {
        return 0;
    } else {
        // _vertsOffsets contains an extra offset at the end that is the position
        // of the first vertex 1 level above that of the tables
        return *_vertsOffsets.rbegin();
    }
}

template <class U> int
FarSubdivisionTables<U>::GetNumVertices( int level ) const {
    assert(level>=0 and level<((int)_vertsOffsets.size()-1));
    return _vertsOffsets[level+1] - _vertsOffsets[level];
}

template <class U> int
FarSubdivisionTables<U>::GetNumVerticesTotal( int level ) const {
    assert(level>=0 and level<((int)_vertsOffsets.size()-1));
    return _vertsOffsets[level+1];
}

template <class U> int
FarSubdivisionTables<U>::GetMemoryUsed() const {
    return (int)(_F_ITa.size() * sizeof(int) +
                 _F_IT.size() * sizeof(unsigned int) +
                 _E_IT.size() * sizeof(int) +
                 _E_W.size() * sizeof(float) +
                 _V_ITa.size() * sizeof(int) +
                 _V_IT.size() * sizeof(unsigned int) +
                 _V_W.size() * sizeof(float));
}

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* FAR_SUBDIVISION_TABLES_H */
