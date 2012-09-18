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
#ifndef FAR_MESH_FACTORY_H
#define FAR_MESH_FACTORY_H

#include <typeinfo>

#include "../version.h"

#include "../hbr/mesh.h"
#include "../hbr/bilinear.h"
#include "../hbr/catmark.h"
#include "../hbr/loop.h"

#include "../far/mesh.h"
#include "../far/dispatcher.h"
#include "../far/bilinearSubdivisionTablesFactory.h"
#include "../far/catmarkSubdivisionTablesFactory.h"
#include "../far/loopSubdivisionTablesFactory.h"
#include "../far/vertexEditTablesFactory.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {


// The meshFactory institutes a 2 steps process in the conversion of a mesh from
// an HbrMesh<T>. The main reason is that client code may want to have access
// to the remapping table that correlates vertices from both meshes for reasons
// of their own. This is also useful to the unit-test code which can match the
// subdivision results of both code paths for correctness.

template <class T, class U=T> class FarMeshFactory {

public:

    // Constructor for the factory : analyzes the HbrMesh and stores
    // transient data used to create the adaptive patch representation.
    // Once the new rep has been instantiated with 'Create', this factory
    // object can be deleted safely.
    FarMeshFactory(HbrMesh<T> * mesh, int maxlevel);

    // Create a table-based mesh representation
    FarMesh<U> * Create( FarDispatcher<U> * dispatch=0 );

    // Maximum level of subidivision supported by this factory
    int GetMaxLevel() const { return _maxlevel; }

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
    int GetNumAdjacentVertVerticesTotal(int level) const;

    // Total number of faces across up to a level
    int GetNumFacesTotal(int level) const {
        return sumList<HbrFace<T> *>(_facesList, level);
    }

    // Return the corresponding index of the HbrVertex<T> in the new mesh
    int GetVertexID( HbrVertex<T> * v );

    // Returns a the mapping between HbrVertex<T>->GetID() and Far vertices indices
    std::vector<int> const & GetRemappingTable( ) const { return _remapTable; }

private:
    friend struct FarBilinearSubdivisionTablesFactory<T,U>;
    friend struct FarCatmarkSubdivisionTablesFactory<T,U>;
    friend struct FarLoopSubdivisionTablesFactory<T,U>;
    friend struct FarVertexEditTablesFactory<T,U>;

    // Non-copyable, so these are not implemented:
    FarMeshFactory( FarMeshFactory const & );
    FarMeshFactory<T,U> & operator=(FarMeshFactory<T,U> const &);

    static bool isBilinear(HbrMesh<T> * mesh);

    static bool isCatmark(HbrMesh<T> * mesh);

    static bool isLoop(HbrMesh<T> * mesh);

    void generatePtexCoordinates( std::vector<int> & vec, int level );

    void copyTopology( std::vector<int> & vec, int level );

    static  bool compareVertices( HbrVertex<T> const *x, HbrVertex<T> const *y );

    static void refine( HbrMesh<T> * mesh, int maxlevel );

    template <class Type> static int sumList( std::vector<std::vector<Type> > const & list, int level );

private:
    HbrMesh<T> * _hbrMesh;

    int _maxlevel,
        _numVertices,
        _numFaces;

    // per-level counters and offsets for each type of vertex (face,edge,vert)
    std::vector<int> _faceVertIdx,
                     _edgeVertIdx,
                     _vertVertIdx;

    // number of indices required for the vertex iteration table at each level
    std::vector<int> _vertVertsListSize;

    // remapping table to translate vertex ID's between Hbr indices and the
    // order of the same vertices in the tables
    std::vector<int> _remapTable;

    // lists of vertices sorted by type and level
    std::vector<std::vector< HbrVertex<T> *> > _faceVertsList,
                                               _edgeVertsList,
                                               _vertVertsList;

    // list of faces sorted by level
    std::vector<std::vector< HbrFace<T> *> > _facesList;
};

template <class T, class U>
    template <class Type> int
FarMeshFactory<T,U>::sumList( std::vector<std::vector<Type> > const & list, int level) {

    level = std::min(level, (int)list.size()-1);
    int total = 0;
    for (int i=0; i<=level; ++i)
        total += (int)list[i].size();
    return total;
}

template <class T, class U> int
FarMeshFactory<T,U>::GetNumAdjacentVertVerticesTotal(int level) const {

    level = std::min(level, GetMaxLevel());
    int total = 0;
    for (int i=0; i<=level; ++i)
        total += _vertVertsListSize[i];
    return total;
}

template <class T, class U> void
FarMeshFactory<T,U>::refine( HbrMesh<T> * mesh, int maxlevel ) {

    for (int l=0; l<maxlevel; ++l ) {
        int nfaces = mesh->GetNumFaces();
        for (int i=0; i<nfaces; ++i) {
            HbrFace<T> * f = mesh->GetFace(i);
            if (f->GetDepth()==l)
                f->Refine();
        }
    }

}

// Compare the weight masks of 2 vertices using the following ordering table.
//
// Assuming 2 computer kernels :
//  - A handles the k_Crease and K_Corner rules
//  - B handles the K_Smooth and K_Dart rules
// The vertices should be sorted so as to minimize the number execution calls of
// these kernels to match the 2 pass interpolation scheme used in Hbr.
template <class T, class U> bool
FarMeshFactory<T,U>::compareVertices( HbrVertex<T> const * x, HbrVertex<T> const * y ) {

    // Masks of the parent vertex decide for the current vertex.
    HbrVertex<T> * px=x->GetParentVertex(),
                 * py=y->GetParentVertex();

    assert( (FarSubdivisionTables<U>::getMaskRanking(px->GetMask(false), px->GetMask(true) )!=0xFF) and
            (FarSubdivisionTables<U>::getMaskRanking(py->GetMask(false), py->GetMask(true) )!=0xFF) );

    return FarSubdivisionTables<U>::getMaskRanking(px->GetMask(false), px->GetMask(true) ) <
           FarSubdivisionTables<U>::getMaskRanking(py->GetMask(false), py->GetMask(true) );
}

// Assumption : the order of the vertices in the HbrMesh could be set in any
// random order, so the builder runs 2 passes over the entire vertex list to
// gather the counters needed to generate the indexing tables.
template <class T, class U>
FarMeshFactory<T,U>::FarMeshFactory( HbrMesh<T> * mesh, int maxlevel ) :
    _hbrMesh(mesh),
    _maxlevel(maxlevel),
    _numVertices(-1),
    _numFaces(-1),
    _faceVertIdx(maxlevel+1,0),
    _edgeVertIdx(maxlevel+1,0),
    _vertVertIdx(maxlevel+1,0),
    _vertVertsListSize(maxlevel+1,0),
    _faceVertsList(maxlevel+1),
    _edgeVertsList(maxlevel+1),
    _vertVertsList(maxlevel+1),
    _facesList(maxlevel+1)
{
    // non-adaptive subdivision of the Hbr mesh up to maxlevel
    refine( mesh, maxlevel);

    int numVertices = mesh->GetNumVertices();
    int numFaces = mesh->GetNumFaces();

    std::vector<int> faceCounts(maxlevel+1,0),
                     edgeCounts(maxlevel+1,0),
                     vertCounts(maxlevel+1,0);

    // First pass (vertices) : count the vertices of each type for each depth
    // up to maxlevel (values are dependent on topology).
    int maxvertid=-1;
    for (int i=0; i<numVertices; ++i) {

        HbrVertex<T> * v = mesh->GetVertex(i);
        assert(v);

        int depth = v->GetFace()->GetDepth();

        if (depth>maxlevel)
            continue;

        if (depth==0 )
            vertCounts[depth]++;

        if (v->GetID()>maxvertid)
            maxvertid = v->GetID();

        if (not v->OnBoundary())
            _vertVertsListSize[depth] += v->GetValence();
        else if (v->GetValence()!=2)
            _vertVertsListSize[depth] ++;

        if (v->GetParentFace())
            faceCounts[depth]++;
        else if (v->GetParentEdge())
            edgeCounts[depth]++;
        else if (v->GetParentVertex())
            vertCounts[depth]++;
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

    _remapTable.resize( maxvertid+1, -1);

    // Second pass (vertices) : calculate the starting indices of the sub-tables
    // (face, edge, verts...) and populate the remapping table.
    for (int i=0; i<numVertices; ++i) {

        HbrVertex<T> * v = mesh->GetVertex(i);
        assert(v);

        int depth = v->GetFace()->GetDepth();

        if (depth>maxlevel)
            continue;

        assert( _remapTable[ v->GetID() ] = -1 );

        if (depth==0) {
            _vertVertsList[ depth ].push_back( v );
            _remapTable[ v->GetID() ] = v->GetID();
        } else if (v->GetParentFace()) {
            _remapTable[ v->GetID() ]=_faceVertIdx[depth]+faceCounts[depth]++;
            _faceVertsList[ depth ].push_back( v );
        } else if (v->GetParentEdge()) {
            _remapTable[ v->GetID() ]=_edgeVertIdx[depth]+edgeCounts[depth]++;
            _edgeVertsList[ depth ].push_back( v );
        } else if (v->GetParentVertex()) {
            // vertices need to be sorted separately based on compute kernel :
            // the remapping step is done just after this
            _vertVertsList[ depth ].push_back( v );
        }
    }

    // Sort the the vertices that are the child of a vertex based on their weight
    // mask. The masks combinations are ordered so as to minimize the compute
    // kernel switching.
    for (size_t i=1; i<_vertVertsList.size(); ++i)
        std::sort(_vertVertsList[i].begin(), _vertVertsList[i].end(),compareVertices);

    // These vertices still need a remapped index
    for (int l=1; l<(maxlevel+1); ++l)
        for (size_t i=0; i<_vertVertsList[l].size(); ++i)
            _remapTable[ _vertVertsList[l][i]->GetID() ]=_vertVertIdx[l]+(int)i;


    // Third pass (faces) : populate the face lists.
    int fsize=0;
    for (int i=0; i<numFaces; ++i) {
        HbrFace<T> * f = mesh->GetFace(i);
        assert(f);
        if (f->GetDepth()==0)
            fsize += mesh->GetSubdivision()->GetFaceChildrenCount( f->GetNumVertices() );
    }

    _facesList[0].reserve(mesh->GetNumCoarseFaces());
    _facesList[1].reserve(fsize);
    for (int l=2; l<=maxlevel; ++l)
        _facesList[l].reserve( _facesList[l-1].capacity()*4 );

    for (int i=0; i<numFaces; ++i) {
        HbrFace<T> * f = mesh->GetFace(i);
        if (f->GetDepth()<=maxlevel)
            _facesList[ f->GetDepth() ].push_back(f);
    }

    _numFaces = GetNumFacesTotal(maxlevel);
    _numVertices = GetNumFaceVerticesTotal(maxlevel) +
                   GetNumEdgeVerticesTotal(maxlevel) +
                   GetNumVertexVerticesTotal(maxlevel);
}

template <class T, class U> bool
FarMeshFactory<T,U>::isBilinear(HbrMesh<T> * mesh) {
    return typeid(*(mesh->GetSubdivision()))==typeid(HbrBilinearSubdivision<T>);
}

template <class T, class U> bool
FarMeshFactory<T,U>::isCatmark(HbrMesh<T> * mesh) {
    return typeid(*(mesh->GetSubdivision()))==typeid(HbrCatmarkSubdivision<T>);
}

template <class T, class U> bool
FarMeshFactory<T,U>::isLoop(HbrMesh<T> * mesh) {
    return typeid(*(mesh->GetSubdivision()))==typeid(HbrLoopSubdivision<T>);
}

template <class T, class U> void
FarMeshFactory<T,U>::copyTopology( std::vector<int> & vec, int level ) {

    assert( _hbrMesh );

    int nv=-1;
    if ( isCatmark(_hbrMesh) or isBilinear(_hbrMesh) )
        nv=4;
    else if ( isLoop(_hbrMesh) )
        nv=3;

    assert(nv>0);

    vec.resize( nv * _facesList[level].size(), -1 );

    for (int i=0; i<(int)_facesList[level].size(); ++i) {
        HbrFace<T> * f = _facesList[level][i];
        assert( f and f->GetNumVertices()==nv);
        for (int j=0; j<f->GetNumVertices(); ++j)
            vec[nv*i+j]=_remapTable[f->GetVertex(j)->GetID()];
    }
}

template <class T, class U> void
copyVertex( T & dest, U const & src ) {
}

template <class T> void
copyVertex( T & dest, T const & src ) {
    dest = src;
}

// XXXX : this currently only supports Catmark / Bilinear schemes.
template <class T, class U> void
FarMeshFactory<T,U>::generatePtexCoordinates( std::vector<int> & vec, int level ) {

    assert( _hbrMesh );

    if (_facesList[level][0]->GetPtexIndex() == -1) return;

    vec.resize( _facesList[level].size()*2, -1 );

    for (int i=0; i<(int)_facesList[level].size(); ++i) {

        HbrFace<T> const * f = _facesList[level][i];
        assert(f);

        short u,v;
        unsigned short ofs = 1, depth;
        bool quad = true;

        // track upwards towards coarse face, accumulating u,v indices
        HbrFace<T> const * p = f->GetParent();
        for ( u=v=depth=0;  p!=NULL; depth++ ) {

            int nverts = p->GetNumVertices();
            if ( nverts != 4 ) {           // non-quad coarse face : stop accumulating offsets
                quad = false;              // invert the coarse face index
                break;
            }

            for (unsigned char i=0; i<nverts; ++i)
                if ( p->GetChild( i )==f ) {
                    switch ( i ) {
                      case 0 :                     break;
                      case 1 : { u+=ofs;         } break;
                      case 2 : { u+=ofs; v+=ofs; } break;
                      case 3 : {         v+=ofs; } break;
                    }
                    break;
                }
            ofs = ofs << 1;
            f = p;
            p = f->GetParent();
        }

        vec[2*i] = quad ? f->GetPtexIndex() : -f->GetPtexIndex();
        vec[2*i+1] = (int)u << 16;
        vec[2*i+1] += v;
    }
}

template <class T, class U> FarMesh<U> *
FarMeshFactory<T,U>::Create( FarDispatcher<U> * dispatch ) {

    assert( _hbrMesh );

    if (_maxlevel<1)
        return 0;

    FarMesh<U> * result = new FarMesh<U>();

    if (dispatch)
        result->_dispatcher = dispatch;
    else
        result->_dispatcher = & FarDispatcher<U>::_DefaultDispatcher;

    if ( isBilinear( _hbrMesh ) ) {
        result->_subdivisionTables = FarBilinearSubdivisionTablesFactory<T,U>::Create( this, result, _maxlevel );
    } else if ( isCatmark( _hbrMesh ) ) {
        result->_subdivisionTables = FarCatmarkSubdivisionTablesFactory<T,U>::Create( this, result, _maxlevel );
    } else if ( isLoop(_hbrMesh) ) {
        result->_subdivisionTables = FarLoopSubdivisionTablesFactory<T,U>::Create( this, result, _maxlevel );
    } else
        assert(0);
    assert(result->_subdivisionTables);
    
    result->_numCoarseVertices = (int)_vertVertsList[0].size();

    // Copy the data of the coarse vertices into the vertex buffer.
    // XXXX : we should figure out a test to make sure that the vertex
    //        class is not an empty placeholder (ex. non-interleaved data)
    result->_vertices.resize( _numVertices );
    for (int i=0; i<result->GetNumCoarseVertices(); ++i)
        copyVertex(result->_vertices[i], _hbrMesh->GetVertex(i)->GetData());

    // Populate topology (face verts indices)
    // XXXX : only k_BilinearQuads support for now - adaptive bicubic patches to come
    result->_patchtype = FarMesh<U>::k_BilinearQuads;

    // XXXX : we should let the client control what to copy, most of this may be irrelevant
    result->_faceverts.resize(_maxlevel+1);
    for (int l=1; l<=_maxlevel; ++l)
        copyTopology(result->_faceverts[l], l);

    result->_ptexcoordinates.resize(_maxlevel+1);
    for (int l=1; l<=_maxlevel; ++l)
        generatePtexCoordinates(result->_ptexcoordinates[l], l);

    // Create VertexEditTables if necessary
    if (_hbrMesh->HasVertexEdits()) {
        result->_vertexEditTables = FarVertexEditTablesFactory<T,U>::Create( this, result, _maxlevel );
        assert(result->_vertexEditTables);
    }
    
    return result;
}

template <class T, class U> int
FarMeshFactory<T,U>::GetVertexID( HbrVertex<T> * v ) {
    assert( v  and (v->GetID() < _remapTable.size()) );
    return _remapTable[ v->GetID() ];
}

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* FAR_MESH_FACTORY_H */
