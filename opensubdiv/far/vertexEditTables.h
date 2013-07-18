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

#ifndef FAR_VERTEX_EDIT_TABLES_H
#define FAR_VERTEX_EDIT_TABLES_H

#include "../version.h"

#include <assert.h>
#include <utility>
#include <vector>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

template <class U> class FarMesh;

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

    /// \brief Get the type of operation
    Operation GetOperation() const { return _op; }

    /// \brief Return index of variable this edit applies to
    int GetIndex() const { return _index; }

    /// \brief Return width of the variable
    int GetWidth() const { return _width; }
    
    /// \brief Get the numerical value of the edit
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
    /// \brief Constructor
    FarVertexEditTables( FarMesh<U> * mesh );

    // Note : Subtract type edits are converted into Adds in order to save kernel calls.

    /// \brief This class holds an array of edits. each batch has unique index/width/operation
    class VertexEditBatch {
    public:
        /// \brief Constructor
        VertexEditBatch(int index, int width, FarVertexEdit::Operation operation);

        /// \brief Copy vertex id and edit values into table
        void Append(int level, int vertexID, const float *values, bool negate);

        /// \brief Compute-kernel applied to vertices
        void ApplyVertexEdits(U * vsrc, int offset, int tableOffset, int start, int end) const;

        // Edit tables accessors

        /// \brief Returns the edit offset table
        const std::vector<unsigned int> &GetVertexIndices() const { return _vertIndices; }

        /// \brief Returns the edit values table
        const std::vector<float> &GetValues() const { return _edits; }

        /// \brief Returns the edit operand (Set / Add)
        FarVertexEdit::Operation GetOperation() const { return _op; }

        /// \brief Returns the index of the primvar affected by the edit
        int GetPrimvarIndex() const { return _primvarIndex; }

        /// \brief Returns the width of the primvar (number of elements)
        int GetPrimvarWidth() const { return _primvarWidth; } 

    private:
        template <class X, class Y> friend class FarVertexEditTablesFactory;
        template <class X, class Y> friend class FarMultiMeshFactory;

        std::vector<unsigned int> _vertIndices;  // absolute vertex index array for edits
        std::vector<float>        _edits;        // edit values array

        int                       _primvarIndex, // primvar offset in vertex
                                  _primvarWidth; // numElements per vertex in values
        FarVertexEdit::Operation  _op;           // edit operation (Set, Add)
    };

    /// \brief Returns the number of edit batches
    int GetNumBatches() const {
        return (int)_batches.size();
    }

    /// \brief Returns a batch of vertex edits
    ///
    /// @param index  batch index
    ///
    VertexEditBatch const & GetBatch(int index) const {
        return _batches[index];
    }

private:
    template <class X, class Y> friend class FarVertexEditTablesFactory;
    template <class X, class Y> friend class FarMultiMeshFactory;
    template <class CONTROLLER> friend class FarComputeController;

    // Compute-kernel that applies the edits
    void computeVertexEdits(int tableIndex, int offset, int tableOffset, int start, int end, void *clientdata) const;

    // mesh that owns this vertexEditTable
    FarMesh<U> * _mesh;

#if defined(__GNUC__)
    // XXX(dyu): seems like there is a compiler bug in g++ that requires
    //               this struct to be public
public:
#endif
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
FarVertexEditTables<U>::VertexEditBatch::ApplyVertexEdits(U * vsrc, int vertexOffset, int tableOffset, int start, int end) const
{
    int primvarWidth = GetPrimvarWidth();
    assert(tableOffset+end <= (int)_vertIndices.size());
    const unsigned int * vindices = &_vertIndices[tableOffset];
    const float * values = &_edits[tableOffset * primvarWidth];
    FarVertexEdit edit( GetOperation(), GetPrimvarIndex(), GetPrimvarWidth() );

    for (int i=start; i<end; ++i) {
        U * vdst = vsrc + vindices[i] + vertexOffset;

        edit.SetEdit(const_cast<float*>(&values[i*primvarWidth]));
        vdst->ApplyVertexEdit(edit);
    }
}

template <class U>
FarVertexEditTables<U>::FarVertexEditTables( FarMesh<U> * mesh ) :
    _mesh(mesh) {
}

template <class U> void
FarVertexEditTables<U>::computeVertexEdits(int tableIndex, int offset, int tableOffset, int start, int end, void *clientdata) const {

    assert(this->_mesh);

    U * vsrc = &this->_mesh->GetVertices().at(0);

    _batches[tableIndex].ApplyVertexEdits(vsrc, offset, tableOffset, start, end);
}

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* FAR_VERTEX_EDIT_TABLES_H */
