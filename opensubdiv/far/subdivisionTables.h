//
//     Copyright (C) Pixar. All rights reserved.
//
//     This license governs use of the accompanying software. If you
//     use the software, you accept this license. If you do not accept
//     the license, do not use the software.
//
//     1. Definitions
//     The terms "reproduce," "reproduction," "derivative works," and
//     "distribution" have the same meaning here as under U.S.
//     copyright law.  A "contribution" is the original software, or
//     any additions or changes to the software.
//     A "contributor" is any person or entity that distributes its
//     contribution under this license.
//     "Licensed patents" are a contributor's patent claims that read
//     directly on its contribution.
//
//     2. Grant of Rights
//     (A) Copyright Grant- Subject to the terms of this license,
//     including the license conditions and limitations in section 3,
//     each contributor grants you a non-exclusive, worldwide,
//     royalty-free copyright license to reproduce its contribution,
//     prepare derivative works of its contribution, and distribute
//     its contribution or any derivative works that you create.
//     (B) Patent Grant- Subject to the terms of this license,
//     including the license conditions and limitations in section 3,
//     each contributor grants you a non-exclusive, worldwide,
//     royalty-free license under its licensed patents to make, have
//     made, use, sell, offer for sale, import, and/or otherwise
//     dispose of its contribution in the software or derivative works
//     of the contribution in the software.
//
//     3. Conditions and Limitations
//     (A) No Trademark License- This license does not grant you
//     rights to use any contributor's name, logo, or trademarks.
//     (B) If you bring a patent claim against any contributor over
//     patents that you claim are infringed by the software, your
//     patent license from such contributor to the software ends
//     automatically.
//     (C) If you distribute any portion of the software, you must
//     retain all copyright, patent, trademark, and attribution
//     notices that are present in the software.
//     (D) If you distribute any portion of the software in source
//     code form, you may do so only under this license by including a
//     complete copy of this license with your distribution. If you
//     distribute any portion of the software in compiled or object
//     code form, you may only do so under a license that complies
//     with this license.
//     (E) The software is licensed "as-is." You bear the risk of
//     using it. The contributors give no express warranties,
//     guarantees or conditions. You may have additional consumer
//     rights under your local laws which this license cannot change.
//     To the extent permitted under your local laws, the contributors
//     exclude the implied warranties of merchantability, fitness for
//     a particular purpose and non-infringement.
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
        F_ITa, // face-vertices adjacency indexing table
        F_IT,  // face-vertices indexing table
        
        E_IT,  // edge-vertices adjacency indexing table
        E_W,   // edge-vertices weights
        
        V_ITa, // vertex-vertices adjacency indexing table
        V_IT,  // vertex-vertices indexing table
        V_W,   // vertex-vertices weights
        
        TABLE_TYPES_COUNT  // number of different types of tables
    };

    /// Destructor
    virtual ~FarSubdivisionTables<U>() {}

    /// Return the highest level of subdivision possible with these tables
    int GetMaxLevel() const { return (int)(_vertsOffsets.size()-1); }

    /// Memory required to store the indexing tables
    int GetMemoryUsed() const;

    /// Pointer back to the mesh owning the table
    FarMesh<U> * GetMesh() { return _mesh; }

    /// The index of the first vertex that belongs to the level of subdivision
    /// represented by this set of FarCatmarkSubdivisionTables
    int GetFirstVertexOffset( int level ) const;

    /// Number of vertices at a given level
    int GetNumVertices( int level ) const;

    /// Total number of vertices at a given level
    int GetNumVerticesTotal( int level ) const;

    /// Indexing tables accessors

    /// Returns the face vertices codex table
    std::vector<int> const &          Get_F_ITa( ) const { return _F_ITa; }

    /// Returns the face vertices indexing table
    std::vector<unsigned int> const & Get_F_IT( ) const { return _F_IT; }

    /// Returns the edge vertices indexing table
    std::vector<int> const &          Get_E_IT() const { return _E_IT; }

    /// Returns the edge vertices weights table
    std::vector<float> const &        Get_E_W() const { return _E_W; }

    /// Returns the vertex vertices codex table
    std::vector<int> const &          Get_V_ITa() const { return _V_ITa; }

    /// Returns the vertex vertices indexing table
    std::vector<unsigned int> const & Get_V_IT() const { return _V_IT; }

    /// Returns the vertex vertices weights table
    std::vector<float> const &        Get_V_W() const { return _V_W; }

    /// Returns the number of indexing tables needed to represent this particular
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
