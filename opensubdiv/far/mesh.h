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
#ifndef FAR_MESH_H
#define FAR_MESH_H

#include <cassert>
#include <vector>

#include "../version.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

template <class T, class U> class FarMeshFactory;
template <class T, class U> class FarSubdivisionTables;
template <class T, class U> class FarDispatcher;

// Core serialized subdivision mesh class.
//
// In order to support both interleaved & non-interleaved vertex access,
// classes are dual-templated : T corresponds to the Hbr vertex representation
// while U correcsponds to this library's vertex representation. In some cases,
// the actual positions of the vertices are irrelevant, so passing an "empty"
// vertex class to Hbr is perfectly acceptable and saves some data-copy steps.

template <class T, class U=T> class FarMesh {
public:
    
    ~FarMesh();

    // returns the subdivision method
    FarSubdivisionTables<T,U> const * GetSubdivision() const { return _subdivision; }

    // returns the compute dispatcher
    FarDispatcher<T,U> const * GetDispatcher() const { return _dispatcher; }

    enum PatchType {
        k_BilinearTriangles,
        k_BilinearQuads,
        k_Triangles,
    };
    
    // returns the type of patches described by the face vertices list 
    PatchType GetPatchType() const { return _patchtype; }

    // returns the list of vertices in the mesh (from subdiv level 0 to N)
    std::vector<U> & GetVertices() { return _vertices; }
    
    U & GetVertex(int index) { return _vertices[index]; }

    // returns the list of indices of the vertices of the faces in the mesh
    std::vector<int> const & GetFaceVertices(int level) const;

    // returns the number of coarse vertices held at the beginning of the vertex
    // buffer.
    int GetNumCoarseVertices() const;

    // returns the total number of vertices in the mesh across across all depths
    int GetNumVertices() const { return (int)(_vertices.size()); }

    // apply the subdivision tables to compute the positions of the vertices up 
    // to 'level'
    void Subdivide(int level=-1);

private:    
    friend class FarMeshFactory<T,U>;

    FarMesh() : _subdivision(0), _dispatcher(0) { }

    // non-copyable, so these are not implemented:
    FarMesh(FarMesh<T,U> const &);
    FarMesh<T,U> & operator = (FarMesh<T,U> const &);

    // subdivision method used in this mesh
    FarSubdivisionTables<T,U> * _subdivision;

    // customizable compute dispatcher class
    FarDispatcher<T,U> * _dispatcher;
    
    // list of vertices (up to N levels of subdivision)
    std::vector<U> _vertices;

    // list of vertex indices for each face
    std::vector< std::vector<int> > _faceverts;
    
    // XXX stub for adaptive work
    PatchType _patchtype;

    // number of vertices at level 0 of subdivision
    int _numCoarseVertices;
};

template <class T, class U>
FarMesh<T,U>::~FarMesh()
{ 
    delete _subdivision;
}

template <class T, class U> int
FarMesh<T,U>::GetNumCoarseVertices() const {
    return _numCoarseVertices;
}

template <class T, class U> std::vector<int> const & 
FarMesh<T,U>::GetFaceVertices(int level) const {
    if ( (level>=0) and (level<(int)_faceverts.size()) )
        return _faceverts[level];
    return _faceverts[0];
}

template <class T, class U> void
FarMesh<T,U>::Subdivide(int maxlevel) {

    assert(_subdivision);

    if ( (maxlevel < 0) )
        maxlevel=_subdivision->GetMaxLevel();
    else
        maxlevel = std::min(maxlevel, _subdivision->GetMaxLevel());

    for (int i=1; i<maxlevel; ++i)
        _subdivision->Refine(i);
}

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* FAR_MESH_H */
