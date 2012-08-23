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
#ifndef FAR_VERTEX_EDIT_TABLES_H
#define FAR_VERTEX_EDIT_TABLES_H

#include <assert.h>
#include <utility>
#include <vector>

#include "../version.h"
#include "../far/table.h"
#include "../far/dispatcher.h"
#include "../hbr/vertexEdit.h"

template <class T> class HbrFace;
template <class T> class HbrHalfedge;
template <class T> class HbrVertex;
template <class T> class HbrMesh;

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

template <class T, class U> class FarMesh;
template <class T, class U> class FarMeshFactory;

template <class T, class U=T> class FarVertexEditTables {
public:
    FarVertexEditTables( FarMesh<T,U> * mesh, int maxlevel);

    // type of edit operation. This enum matches to HbrHiearachicalEdit<T>::Operation
    enum Operation {
        Set,
        Add
    };

    // Compute the positions of edited vertices
    void Apply(int level, void * clientdata=0) const;

    int GetNumBatches() const {
        return (int)_batches.size();
    }

    // this class holds a batch for vertex edit. each batch has unique index/width/operation
    class VertexEdit {
    public:
        VertexEdit(int index, int width, Operation operation);

        // copy vertex id and edit values into table
        void Append(int level, int vertexID, const float *values, bool negate);

        // Compute-kernel applied to vertices
        void ApplyVertexEdit(U * vsrc, int level) const;

        // Edit tables accessors

        // Returns the edit offset table
        FarTable<unsigned int> const & Get_Offsets() const { return _offsets; }

        // Returns the edit values table
        FarTable<float> const & Get_Values() const { return _values; }

        Operation GetOperation() const { return _operation; }

        int GetPrimvarOffset() const { return _index; }

        int GetPrimvarWidth() const { return _width; } 

    private:
        friend class FarMeshFactory<T,U>;

        FarTable<unsigned int> _offsets;  // absolute vertex index array for edits
        FarTable<float>        _values;   // edit values array

        int _index;                       // primvar offset in vertex
        int _width;                       // numElements per vertex in values
        Operation _operation;             // edit operation (Set, Add)
    };

    VertexEdit const & GetBatch(int index) const {
        return _batches[index];
    }

protected:
    friend class FarMeshFactory<T,U>;
    friend class FarDispatcher<T,U>;

    // Compute-kernel applied to vertices
    void editVertex(int level, void *clientdata) const;

    // mesh that owns this vertexEditTable
    FarMesh<T,U> * _mesh;

    std::vector<VertexEdit> _batches;
};

template <class T, class U>
FarVertexEditTables<T,U>::VertexEdit::VertexEdit(int index, int width, Operation operation) :
    _index(index),
    _width(width),
    _operation(operation) {
}

template <class T, class U>
void
FarVertexEditTables<T,U>::VertexEdit::ApplyVertexEdit(U * vsrc, int level) const
{
    int n = _offsets.GetNumElements(level-1);
    const unsigned int * offsets = _offsets[level-1];
    const float * values = _values[level-1];

    for(int i=0; i<n; ++i) {
        U * vdst = vsrc + offsets[i];

        // XXXX: tentative.
        // consider adding new interface to vertex class without HbrVertexEdit,
        // such as vdst->ApplyVertexEditAdd(const float *), vdst->ApplyVertexEditSet(const float *)
        if (_operation == FarVertexEditTables<T,U>::Set) {
            HbrVertexEdit<T> vedit(0, 0, 0, 0, 0, _width, false, HbrVertexEdit<T>::Set, const_cast<float*>(&values[i*_width]));
            vdst->ApplyVertexEdit(vedit);
        } else {
            HbrVertexEdit<T> vedit(0, 0, 0, 0, 0, _width, false, HbrVertexEdit<T>::Add, const_cast<float*>(&values[i*_width]));
            vdst->ApplyVertexEdit(vedit);
        }
    }
}

template <class T, class U>
FarVertexEditTables<T,U>::FarVertexEditTables( FarMesh<T,U> * mesh, int maxlevel) :
    _mesh(mesh) {
}


template <class T, class U> void
FarVertexEditTables<T,U>::Apply( int level, void * clientdata ) const {

    assert(this->_mesh and level>0);

    FarDispatcher<T,U> const * dispatch = this->_mesh->GetDispatcher();
    assert(dispatch);

    dispatch->ApplyVertexEdit(this->_mesh, 0, level, clientdata);
}

template <class T, class U> void
FarVertexEditTables<T,U>::editVertex(int level, void *clientdata) const {

    assert(this->_mesh);

    U * vsrc = &this->_mesh->GetVertices().at(0);

    for(int i=0; i<(int)_batches.size(); ++i) {
        _batches[i].ApplyVertexEdit(vsrc, level);
    }
}

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* FAR_VERTEX_EDIT_TABLES_H */
