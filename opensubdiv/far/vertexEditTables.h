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

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

template <class U> class FarMesh;
template <class U> class FarDispatcher;

/// \brief A serialized container for hierarchical edits.
///
/// Some of the hierarchical edits are resolved into the vertex weights computed
/// into the FarSubdivision tables. Certain edits however have to be "post-processed"
/// after the tables are applied to the vertices and require their HBR representation
/// to be serialized into a specific container.
///
class FarVertexEdit {
public:
    /// Type of edit operation - equivalent to HbrHiearachicalEdit<T>::Operation
    enum Operation {
        Set,
        Add
        /// Note : subtract edits are converted to Add edits for better serialization
    };

    /// Get the type of operation
    Operation GetOperation() const { return _op; }

    /// Return index of variable this edit applies to
    int GetIndex() const { return _index; }

    /// Return width of the variable
    int GetWidth() const { return _width; }
    
    /// Get the numerical value of the edit
    const float* GetEdit() const { return _edit; }

private:
    template <class U> friend class FarVertexEditTables;
    
    FarVertexEdit(Operation op, int index, int width) :
        _op(op), _edit(0), _index(index), _width(width)
    { }
    
    void SetEdit(float const * edit) { _edit=edit; }

    Operation     _op;
    float const * _edit;
    int           _index,
                  _width;
};

template <class U> class FarVertexEditTables {
public:
    FarVertexEditTables( FarMesh<U> * mesh, int maxlevel);

    // Note : Subtract type edits are converted into Adds in order to save kernel calls.

    // Compute the positions of edited vertices
    void Apply(int level, void * clientdata=0) const;

    int GetNumBatches() const {
        return (int)_batches.size();
    }

    // This class holds an array of edits. each batch has unique index/width/operation
    class VertexEditBatch {
    public:
        VertexEditBatch(int index, int width, FarVertexEdit::Operation operation);

        // copy vertex id and edit values into table
        void Append(int level, int vertexID, const float *values, bool negate);

        // Compute-kernel applied to vertices
        void ApplyVertexEdits(U * vsrc, int level) const;

        // Edit tables accessors

        // Returns the edit offset table
        FarTable<unsigned int> const & GetVertexIndices() const { return _vertIndices; }

        // Returns the edit values table
        FarTable<float> const & GetValues() const { return _edits; }

        FarVertexEdit::Operation GetOperation() const { return _op; }

        int GetPrimvarIndex() const { return _primvarIndex; }

        int GetPrimvarWidth() const { return _primvarWidth; } 

    private:
        template <class X, class Y> friend struct FarVertexEditTablesFactory;
        friend class FarDispatcher<U>;

        FarTable<unsigned int>    _vertIndices;  // absolute vertex index array for edits
        FarTable<float>           _edits;        // edit values array

        int                       _primvarIndex, // primvar offset in vertex
                                  _primvarWidth; // numElements per vertex in values
        FarVertexEdit::Operation  _op;           // edit operation (Set, Add)
    };

    VertexEditBatch const & GetBatch(int index) const {
        return _batches[index];
    }

private:
    template <class X, class Y> friend struct FarVertexEditTablesFactory;
    friend class FarDispatcher<U>;

    // Compute-kernel that applies the edits
    void computeVertexEdits(int level, void *clientdata) const;

    // mesh that owns this vertexEditTable
    FarMesh<U> * _mesh;

    std::vector<VertexEditBatch> _batches;
};

template <class U>
FarVertexEditTables<U>::VertexEditBatch::VertexEditBatch(int index, int width, FarVertexEdit::Operation op) :
    _primvarIndex(index),
    _primvarWidth(width),
    _op(op) {
}

template <class U>
void
FarVertexEditTables<U>::VertexEditBatch::ApplyVertexEdits(U * vsrc, int level) const
{
    int n = _vertIndices.GetNumElements(level-1);
    const unsigned int * offsets = _vertIndices[level-1];
    const float * values = _edits[level-1];

    FarVertexEdit edit( GetOperation(), GetPrimvarIndex(), GetPrimvarWidth() );

    for(int i=0; i<n; ++i) {
        U * vdst = vsrc + offsets[i];

        edit.SetEdit( const_cast<float*>(&values[i*GetPrimvarWidth()]) );
        vdst->ApplyVertexEdit( edit );
    }
}

template <class U>
FarVertexEditTables<U>::FarVertexEditTables( FarMesh<U> * mesh, int maxlevel) :
    _mesh(mesh) {
}


template <class U> void
FarVertexEditTables<U>::Apply( int level, void * clientdata ) const {

    assert(this->_mesh and level>0);

    FarDispatcher<U> const * dispatch = this->_mesh->GetDispatcher();
    assert(dispatch);

    dispatch->ApplyVertexEdits(this->_mesh, 0, level, clientdata);
}

template <class U> void
FarVertexEditTables<U>::computeVertexEdits(int level, void *clientdata) const {

    assert(this->_mesh);

    U * vsrc = &this->_mesh->GetVertices().at(0);

    for(int i=0; i<(int)_batches.size(); ++i)
        _batches[i].ApplyVertexEdits(vsrc, level);
}

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* FAR_VERTEX_EDIT_TABLES_H */
