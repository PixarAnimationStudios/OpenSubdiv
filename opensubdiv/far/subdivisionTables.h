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

#include <assert.h>
#include <utility>
#include <vector>

#include "../version.h"
#include "../far/table.h"

template <class T> class HbrFace;
template <class T> class HbrHalfedge;
template <class T> class HbrVertex;
template <class T> class HbrMesh;

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

template <class T, class U> class FarMesh;
template <class T, class U> class FarMeshFactory;

// Catmull-Clark tables store the indexing tables required in order to compute
// the refined positions of a mesh without the help of a hierarchical data
// structure. The advantage of this representation is its ability to be executed
// in a massively parallel environment without data dependencies.
//
// The vertex indexing tables require the vertex buffer to be sorted based on the
// nature of the parent of a given vertex : either a face, an edge, or a vertex.
//
// [...Child of a Face...]|[... Child of an Edge ...]|[... Child of a Vertex ...]
//
// Each segment of the buffer is associated the following tables (<T> is the type):
// _<T>_IT : indices of all the adjacent vertices required by the compute kernels
// _<T>_W : fractional weight of the vertex (based on sharpness & topology)
// _<T>_ITa : codex for the two previous tables

// For more details see : "Feature Adaptive GPU Rendering of Catmull-Clark
// Subdivision Surfaces"  p.3 - par. 3.2
template <class T, class U=T> class FarSubdivisionTables {
public:

    // Destructor
    virtual ~FarSubdivisionTables<T,U>() {}

    // Return the highest level of subdivision possible with these tables
    int GetMaxLevel() const { return (int)(_vertsOffsets.size()); }

    // Memory required to store the indexing tables
    virtual int GetMemoryUsed() const;

    // Compute the positions of refined vertices using the specified kernels
    virtual void Apply( int level, void * clientdata=0 ) const=0;

    // Pointer back to the mesh owning the table
    FarMesh<T,U> * GetMesh() { return _mesh; }

    // The index of the first vertex that belongs to the level of subdivision
    // represented by this set of FarCatmarkSubdivisionTables
    int GetFirstVertexOffset( int level ) const;

    // Number of vertices children of a face at a given level (always 0 for Loop)
    int GetNumFaceVertices( int level ) const;

    // Number of vertices children of an edge at a given level
    int GetNumEdgeVertices( int level ) const;

    // Number of vertices children of a vertex at a given level
    int GetNumVertexVertices( int level ) const;

    // Total number of vertices at a given level
    int GetNumVertices( int level ) const;

    // Indexing tables accessors

    // Returns the edge vertices indexing table
    FarTable<int> const &          Get_E_IT() const { return _E_IT; }

    // Returns the edge vertices weights table
    FarTable<float> const &        Get_E_W() const { return _E_W; }

    // Returns the vertex vertices codex table
    FarTable<int> const &          Get_V_ITa() const { return _V_ITa; }

    // Returns the vertex vertices indexing table
    FarTable<unsigned int> const & Get_V_IT() const { return _V_IT; }

    // Returns the vertex vertices weights table
    FarTable<float> const &        Get_V_W() const { return _V_W; }

    // Returns the number of indexing tables needed to represent this particular
    // subdivision scheme.
    virtual int GetNumTables() const { return 5; }

protected:
    friend class FarMeshFactory<T,U>;

    FarSubdivisionTables<T,U>( FarMesh<T,U> * mesh, int maxlevel );

    // Returns an integer based on the order in which the kernels are applied
    static int getMaskRanking( unsigned char mask0, unsigned char mask1 );

    // Compares to vertices based on the ranking of their hbr masks combination
    static bool compareVertices( HbrVertex<T> const * x, HbrVertex<T> const * y );

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

    // Returns the range of vertex indices of each of the 3 batches of VertexPoint
    // compute Kernels (kernel application order is : B / A / A)
    std::vector<VertexKernelBatch> & getKernelBatches() const { return _batches; }

protected:
    // mesh that owns this subdivisionTable
    FarMesh<T,U> * _mesh;

    FarTable<int>          _E_IT;  // vertices from edge refinement
    FarTable<float>        _E_W;   // weigths

    FarTable<int>          _V_ITa; // vertices from vertex refinement
    FarTable<unsigned int> _V_IT;  // indices of adjacent vertices
    FarTable<float>        _V_W;   // weights

    std::vector<VertexKernelBatch> _batches; // batches of vertices for kernel execution

    std::vector<int> _vertsOffsets; // offset to the first vertex of each level
private:
};

template <class T, class U>
FarSubdivisionTables<T,U>::FarSubdivisionTables( FarMesh<T,U> * mesh, int maxlevel ) :
    _mesh(mesh),
    _E_IT(maxlevel+1),
    _E_W(maxlevel+1),
    _V_ITa(maxlevel+1),
    _V_IT(maxlevel+1),
    _V_W(maxlevel+1),
    _batches(maxlevel),
    _vertsOffsets(maxlevel+1,0)
{
    assert( maxlevel > 0 );
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
FarSubdivisionTables<T,U>::getMaskRanking( unsigned char mask0, unsigned char mask1 ) {
    static short masks[4][4] = { {    0,    1,    6,    4 },
                                 { 0xFF,    2,    5,    3 },
                                 { 0xFF, 0xFF,    9,    7 },
                                 { 0xFF, 0xFF, 0xFF,    8 } };
    return masks[mask0][mask1];
}

// Compare the weight masks of 2 vertices using the following ordering table.
//
// Assuming 2 computer kernels :
//  - A handles the k_Crease and K_Corner rules
//  - B handles the K_Smooth and K_Dart rules
// The vertices should be sorted so as to minimize the number execution calls of
// these kernels to match the 2 pass interpolation scheme used in Hbr.
template <class T, class U> bool
FarSubdivisionTables<T,U>::compareVertices( HbrVertex<T> const * x, HbrVertex<T> const * y ) {

    // Masks of the parent vertex decide for the current vertex.
    HbrVertex<T> * px=x->GetParentVertex(),
                 * py=y->GetParentVertex();

    assert( (getMaskRanking(px->GetMask(false), px->GetMask(true) )!=0xFF) and
            (getMaskRanking(py->GetMask(false), py->GetMask(true) )!=0xFF) );

    return getMaskRanking(px->GetMask(false), px->GetMask(true) ) <
           getMaskRanking(py->GetMask(false), py->GetMask(true) );
}

template <class T, class U> int
FarSubdivisionTables<T,U>::GetFirstVertexOffset( int level ) const {
    assert(level>=0 and level<=(int)_vertsOffsets.size());
    return _vertsOffsets[level];
}

template <class T, class U> int
FarSubdivisionTables<T,U>::GetNumFaceVertices( int level ) const {
    assert(level>=0 and level<=(int)_batches.size());
    return _batches[level-1].kernelF;
}

template <class T, class U> int
FarSubdivisionTables<T,U>::GetNumEdgeVertices( int level ) const {
    assert(level>=0 and level<=(int)_batches.size());
    return _batches[level-1].kernelE;
}

template <class T, class U> int
FarSubdivisionTables<T,U>::GetNumVertexVertices( int level ) const {
    assert(level>=0 and level<=(int)_batches.size());
    if (level==0)
        return _mesh->GetNumCoarseVertices();
    else
        return std::max( _batches[level-1].kernelB.second,
                   std::max(_batches[level-1].kernelA1.second,
                       _batches[level-1].kernelA2.second));
}

template <class T, class U> int
FarSubdivisionTables<T,U>::GetNumVertices( int level ) const {
    assert(level>=0 and level<=(int)_batches.size());
    if (level==0)
        return GetNumVertexVertices(0);
    else
        return GetNumFaceVertices(level)+
               GetNumEdgeVertices(level)+
               GetNumVertexVertices(level);
}

template <class T, class U> int
FarSubdivisionTables<T,U>::GetMemoryUsed() const {
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
