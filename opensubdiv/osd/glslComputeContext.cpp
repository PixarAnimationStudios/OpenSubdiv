//
//   Copyright 2013 Pixar
//
//   Licensed under the Apache License, Version 2.0 (the "Apache License")
//   with the following modification; you may not use this file except in
//   compliance with the Apache License and the following modification to it:
//   Section 6. Trademarks. is deleted and replaced with:
//
//   6. Trademarks. This License does not grant permission to use the trade
//      names, trademarks, service marks, or product names of the Licensor
//      and its affiliates, except as required to comply with Section 4(c) of
//      the License and to reproduce the content of the NOTICE file.
//
//   You may obtain a copy of the Apache License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the Apache License with the above modification is
//   distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
//   KIND, either express or implied. See the Apache License for the specific
//   language governing permissions and limitations under the Apache License.
//

#include "../osd/debug.h"
#include "../osd/error.h"
#include "../osd/glslComputeContext.h"
#include "../osd/glslKernelBundle.h"

#include "../osd/opengl.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

void
OsdGLSLComputeTable::createBuffer(size_t size, const void *ptr) {

    glGenBuffers(1, &_devicePtr);

#if defined(GL_EXT_direct_state_access)
    if (glNamedBufferDataEXT) {
        glNamedBufferDataEXT(_devicePtr, size, ptr, GL_STATIC_DRAW);
    } else {
#else
    {
#endif
        GLint prev = 0;
        glGetIntegerv(GL_SHADER_STORAGE_BUFFER_BINDING, &prev);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, _devicePtr);
        glBufferData(GL_SHADER_STORAGE_BUFFER, size, ptr, GL_STATIC_DRAW);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, prev);
    }
/*
  CHECK_GL_ERROR("UpdateTable tableIndex %d, size %ld, buffer =%d\n",
  tableIndex, size, _tableBuffers[tableIndex]);
*/
}

OsdGLSLComputeTable::~OsdGLSLComputeTable() {

    glDeleteBuffers(1, &_devicePtr);
}

GLuint
OsdGLSLComputeTable::GetBuffer() const {

    return _devicePtr;
}

// ----------------------------------------------------------------------------

OsdGLSLComputeHEditTable::OsdGLSLComputeHEditTable(
    const FarVertexEditTables::VertexEditBatch &batch)
    : _primvarIndicesTable(new OsdGLSLComputeTable(batch.GetVertexIndices())),
      _editValuesTable(new OsdGLSLComputeTable(batch.GetValues())) {

    _operation = batch.GetOperation();
    _primvarOffset = batch.GetPrimvarIndex();
    _primvarWidth = batch.GetPrimvarWidth();
}

OsdGLSLComputeHEditTable::~OsdGLSLComputeHEditTable() {

    delete _primvarIndicesTable;
    delete _editValuesTable;
}

const OsdGLSLComputeTable *
OsdGLSLComputeHEditTable::GetPrimvarIndices() const {

    return _primvarIndicesTable;
}

const OsdGLSLComputeTable *
OsdGLSLComputeHEditTable::GetEditValues() const {

    return _editValuesTable;
}

int
OsdGLSLComputeHEditTable::GetOperation() const {

    return _operation;
}

int
OsdGLSLComputeHEditTable::GetPrimvarOffset() const {

    return _primvarOffset;
}

int
OsdGLSLComputeHEditTable::GetPrimvarWidth() const {

    return _primvarWidth;
}

// ----------------------------------------------------------------------------

OsdGLSLComputeContext::OsdGLSLComputeContext(
    FarSubdivisionTables const *subdivisionTables,
    FarVertexEditTables const *vertexEditTables) {

    // allocate 5 or 7 tables
    // XXXtakahito: Although _tables size depends on table type, F_IT is set
    // to NULL even in loop case, to determine the condition in
    // bindShaderStorageBuffer()...
    _tables.resize(7, 0);

    _tables[FarSubdivisionTables::E_IT]  = new OsdGLSLComputeTable(subdivisionTables->Get_E_IT());
    _tables[FarSubdivisionTables::V_IT]  = new OsdGLSLComputeTable(subdivisionTables->Get_V_IT());
    _tables[FarSubdivisionTables::V_ITa] = new OsdGLSLComputeTable(subdivisionTables->Get_V_ITa());
    _tables[FarSubdivisionTables::E_W]   = new OsdGLSLComputeTable(subdivisionTables->Get_E_W());
    _tables[FarSubdivisionTables::V_W]   = new OsdGLSLComputeTable(subdivisionTables->Get_V_W());

    if (subdivisionTables->GetNumTables() > 5) {
        // catmark, bilinear
        _tables[FarSubdivisionTables::F_IT]  = new OsdGLSLComputeTable(subdivisionTables->Get_F_IT());
        _tables[FarSubdivisionTables::F_ITa] = new OsdGLSLComputeTable(subdivisionTables->Get_F_ITa());
    } else {
        // loop
        _tables[FarSubdivisionTables::F_IT] = NULL;
        _tables[FarSubdivisionTables::F_ITa] = NULL;
    }

    // create hedit tables
    if (vertexEditTables) {
        int numEditBatches = vertexEditTables->GetNumBatches();
        _editTables.reserve(numEditBatches);
        for (int i = 0; i < numEditBatches; ++i) {
            const FarVertexEditTables::VertexEditBatch & edit =
                vertexEditTables->GetBatch(i);
            _editTables.push_back(new OsdGLSLComputeHEditTable(edit));
        }
    }
}

OsdGLSLComputeContext::~OsdGLSLComputeContext() {

    for (size_t i = 0; i < _tables.size(); ++i) {
        delete _tables[i];
    }
    for (size_t i = 0; i < _editTables.size(); ++i) {
        delete _editTables[i];
    }
}

const OsdGLSLComputeTable *
OsdGLSLComputeContext::GetTable(int tableIndex) const {

    return _tables[tableIndex];
}

int
OsdGLSLComputeContext::GetNumEditTables() const {

    return static_cast<int>(_editTables.size());
}

const OsdGLSLComputeHEditTable *
OsdGLSLComputeContext::GetEditTable(int tableIndex) const {

    return _editTables[tableIndex];
}

OsdGLSLComputeContext *
OsdGLSLComputeContext::Create(FarSubdivisionTables const *subdivisionTables,
                              FarVertexEditTables const *vertexEditTables) {

    return new OsdGLSLComputeContext(subdivisionTables, vertexEditTables);
}

void
OsdGLSLComputeContext::BindEditShaderStorageBuffers(int editIndex) const {

    const OsdGLSLComputeHEditTable * edit = _editTables[editIndex];
    const OsdGLSLComputeTable * primvarIndices = edit->GetPrimvarIndices();
    const OsdGLSLComputeTable * editValues = edit->GetEditValues();

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 9,
                     primvarIndices->GetBuffer());
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 10,
                     editValues->GetBuffer());
}

void
OsdGLSLComputeContext::UnbindEditShaderStorageBuffers() const {

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 9, 0);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 10, 0);
}

void
OsdGLSLComputeContext::BindShaderStorageBuffers() const {

    // 0 and 1 are reserved for vertex/varying buffer bindings.
    if (_tables[FarSubdivisionTables::F_IT]) {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2,
                         _tables[FarSubdivisionTables::F_IT]->GetBuffer());
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3,
                         _tables[FarSubdivisionTables::F_ITa]->GetBuffer());
    }

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4,
                     _tables[FarSubdivisionTables::E_IT]->GetBuffer());
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5,
                     _tables[FarSubdivisionTables::V_IT]->GetBuffer());
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6,
                     _tables[FarSubdivisionTables::V_ITa]->GetBuffer());
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7,
                     _tables[FarSubdivisionTables::E_W]->GetBuffer());
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8,
                     _tables[FarSubdivisionTables::V_W]->GetBuffer());
}

void
OsdGLSLComputeContext::UnbindShaderStorageBuffers() const {

    // 0 and 1 are reserved for vertex/varying buffer bindings.
    if (_tables[FarSubdivisionTables::F_IT]) {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, 0);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, 0);
    }
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, 0);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, 0);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, 0);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, 0);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, 0);

    glUseProgram(0);

    OSD_DEBUG_CHECK_GL_ERROR("UnbindTextures");
}

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
