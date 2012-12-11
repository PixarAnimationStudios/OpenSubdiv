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

#include "../far/table.h"

#include <cassert>
#include <utility>
#include <vector>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

template <class U> class FarMesh;
template <class U> class FarDispatcher;

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

    /// Destructor
    virtual ~FarSubdivisionTables<U>() {}

    /// Return the highest level of subdivision possible with these tables
    int GetMaxLevel() const { return (int)(_vertsOffsets.size()); }

    /// Memory required to store the indexing tables
    virtual int GetMemoryUsed() const;

    /// Compute the positions of refined vertices using the specified kernels
    virtual void Apply( int level, FarDispatcher<U> const *dispatch, void * data=0 ) const=0;

    /// Pointer back to the mesh owning the table
    FarMesh<U> * GetMesh() { return _mesh; }

    /// The index of the first vertex that belongs to the level of subdivision
    /// represented by this set of FarCatmarkSubdivisionTables
    int GetFirstVertexOffset( int level ) const;

    /// Number of vertices children of a face at a given level (always 0 for Loop)
    int GetNumFaceVertices( int level ) const;

    /// Number of vertices children of an edge at a given level
    int GetNumEdgeVertices( int level ) const;

    /// Number of vertices children of a vertex at a given level
    int GetNumVertexVertices( int level ) const;

    // Total number of vertices at a given level
    int GetNumVertices( int level ) const;

    /// Indexing tables accessors

    /// Returns the edge vertices indexing table
    FarTable<int> const &          Get_E_IT() const { return _E_IT; }

    /// Returns the edge vertices weights table
    FarTable<float> const &        Get_E_W() const { return _E_W; }

    /// Returns the vertex vertices codex table
    FarTable<int> const &          Get_V_ITa() const { return _V_ITa; }

    /// Returns the vertex vertices indexing table
    FarTable<unsigned int> const & Get_V_IT() const { return _V_IT; }

    /// Returns the vertex vertices weights table
    FarTable<float> const &        Get_V_W() const { return _V_W; }

    /// Returns the number of indexing tables needed to represent this particular
    /// subdivision scheme.
    virtual int GetNumTables() const { return 5; }

protected:
    template <class X, class Y> friend class FarMeshFactory;

    FarSubdivisionTables<U>( FarMesh<U> * mesh, int maxlevel );

#if defined(__clang__)
    // XXX(jcowles): seems like there is a compiler bug in clang that requires
    //               this struct to be public
public:
#endif
    struct VertexKernelBatch {
        int kernelF; // number of face vertices
        int kernelE; // number of edge vertices

        std::pair<int,int> kernelB;  // first / last vertex vertex batch (kernel B)
        std::pair<int,int> kernelA1; // first / last vertex vertex batch (kernel A pass 1)
        std::pair<int,int> kernelA2; // first / last vertex vertex batch (kernel A pass 2)

        VertexKernelBatch() : kernelF(0), kernelE(0) { }

        void InitVertexKernels(int a, int b) {
            kernelB.first = kernelA1.first = kernelA2.first = a;
            kernelB.second = kernelA1.second = kernelA2.second = b;
        }

        void AddVertex( int index, int rank ) {
            // expand the range of kernel batches based on vertex index and rank
            if (rank<7) {
                if (index < kernelB.first)
                    kernelB.first=index;
                if (index > kernelB.second)
                    kernelB.second=index;
            }
            if ((rank>2) and (rank<8)) {
                if (index < kernelA2.first)
                    kernelA2.first=index;
                if (index > kernelA2.second)
                    kernelA2.second=index;
            }
            if (rank>6) {
                if (index < kernelA1.first)
                    kernelA1.first=index;
                if (index > kernelA1.second)
                    kernelA1.second=index;
            }
        }
    };
#if defined(__clang__)
protected:
#endif

    // Returns the range of vertex indices of each of the 3 batches of VertexPoint
    // compute Kernels (kernel application order is : B / A / A)
    std::vector<VertexKernelBatch> & getKernelBatches() const { return _batches; }

protected:
    // mesh that owns this subdivisionTable
    FarMesh<U> * _mesh;

    FarTable<int>          _E_IT;  // vertices from edge refinement
    FarTable<float>        _E_W;   // weigths

    FarTable<int>          _V_ITa; // vertices from vertex refinement
    FarTable<unsigned int> _V_IT;  // indices of adjacent vertices
    FarTable<float>        _V_W;   // weights

    std::vector<VertexKernelBatch> _batches; // batches of vertices for kernel execution

    std::vector<int> _vertsOffsets; // offset to the first vertex of each level
    
    unsigned int _numCoarseVertices;
private:
};

template <class U>
FarSubdivisionTables<U>::FarSubdivisionTables( FarMesh<U> * mesh, int maxlevel ) :
    _mesh(mesh),
    _E_IT(maxlevel+1),
    _E_W(maxlevel+1),
    _V_ITa(maxlevel+1),
    _V_IT(maxlevel+1),
    _V_W(maxlevel+1),
    _batches(maxlevel),
    _vertsOffsets(maxlevel+1,0),
    _numCoarseVertices(0)
{
    assert( maxlevel > 0 );
}

template <class U> int
FarSubdivisionTables<U>::GetFirstVertexOffset( int level ) const {
    assert(level>=0 and level<=(int)_vertsOffsets.size());
    return _vertsOffsets[level];
}

template <class U> int
FarSubdivisionTables<U>::GetNumFaceVertices( int level ) const {
    assert(level>=0 and level<=(int)_batches.size());
    return _batches[level-1].kernelF;
}

template <class U> int
FarSubdivisionTables<U>::GetNumEdgeVertices( int level ) const {
    assert(level>=0 and level<=(int)_batches.size());
    return _batches[level-1].kernelE;
}

template <class U> int
FarSubdivisionTables<U>::GetNumVertexVertices( int level ) const {
    assert(level>=0 and level<=(int)_batches.size());
    if (level==0)
        return _numCoarseVertices;
    else
        return std::max( _batches[level-1].kernelB.second,
                   std::max(_batches[level-1].kernelA1.second,
                       _batches[level-1].kernelA2.second));
}

template <class U> int
FarSubdivisionTables<U>::GetNumVertices( int level ) const {
    assert(level>=0 and level<=(int)_batches.size());
    if (level==0)
        return GetNumVertexVertices(0);
    else
        return GetNumFaceVertices(level)+
               GetNumEdgeVertices(level)+
               GetNumVertexVertices(level);
}

template <class U> int
FarSubdivisionTables<U>::GetMemoryUsed() const {
    return _E_IT.GetMemoryUsed()+
           _E_W.GetMemoryUsed()+
           _V_ITa.GetMemoryUsed()+
           _V_IT.GetMemoryUsed()+
           _V_W.GetMemoryUsed();
}

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* FAR_SUBDIVISION_TABLES_H */
