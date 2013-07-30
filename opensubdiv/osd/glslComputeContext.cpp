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

#include "../far/mesh.h"
#include "../osd/debug.h"
#include "../osd/error.h"
#include "../osd/glslComputeContext.h"
#include "../osd/glslKernelBundle.h"

#include "../osd/opengl.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

void
OsdGLSLComputeTable::createBuffer(size_t size, const void *ptr) {

    GLint prev = 0;
    glGenBuffers(1, &_devicePtr);
    glGetIntegerv(GL_SHADER_STORAGE_BUFFER_BINDING, &prev);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _devicePtr);
    glBufferData(GL_SHADER_STORAGE_BUFFER, size, ptr, GL_STATIC_DRAW);
/*
  CHECK_GL_ERROR("UpdateTable tableIndex %d, size %ld, buffer =%d\n",
  tableIndex, size, _tableBuffers[tableIndex]);
*/
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, prev);
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
    const FarVertexEditTables<OsdVertex>::VertexEditBatch &batch)
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
    FarMesh<OsdVertex> const *farMesh)
    : _vertexTexture(0), _varyingTexture(0) {

    FarSubdivisionTables<OsdVertex> const * farTables =
        farMesh->GetSubdivisionTables();

    // allocate 5 or 7 tables
    // XXXtakahito: Although _tables size depends on table type, F_IT is set
    // to NULL even in loop case, to determine the condition in
    // bindShaderStorageBuffer()...
    _tables.resize(7, 0);

    _tables[FarSubdivisionTables<OsdVertex>::E_IT]  = new OsdGLSLComputeTable(farTables->Get_E_IT());
    _tables[FarSubdivisionTables<OsdVertex>::V_IT]  = new OsdGLSLComputeTable(farTables->Get_V_IT());
    _tables[FarSubdivisionTables<OsdVertex>::V_ITa] = new OsdGLSLComputeTable(farTables->Get_V_ITa());
    _tables[FarSubdivisionTables<OsdVertex>::E_W]   = new OsdGLSLComputeTable(farTables->Get_E_W());
    _tables[FarSubdivisionTables<OsdVertex>::V_W]   = new OsdGLSLComputeTable(farTables->Get_V_W());

    if (farTables->GetNumTables() > 5) {
        // catmark, bilinear
        _tables[FarSubdivisionTables<OsdVertex>::F_IT]  = new OsdGLSLComputeTable(farTables->Get_F_IT());
        _tables[FarSubdivisionTables<OsdVertex>::F_ITa] = new OsdGLSLComputeTable(farTables->Get_F_ITa());
    } else {
        // loop
        _tables[FarSubdivisionTables<OsdVertex>::F_IT] = NULL;
        _tables[FarSubdivisionTables<OsdVertex>::F_ITa] = NULL;
    }

    // create hedit tables
    FarVertexEditTables<OsdVertex> const *editTables = farMesh->GetVertexEdit();
    if (editTables) {
        int numEditBatches = editTables->GetNumBatches();
        _editTables.reserve(numEditBatches);
        for (int i = 0; i < numEditBatches; ++i) {
            const FarVertexEditTables<OsdVertex>::VertexEditBatch & edit =
                editTables->GetBatch(i);
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

GLuint
OsdGLSLComputeContext::GetCurrentVertexBuffer() const {

    return _currentVertexBuffer;
}

GLuint
OsdGLSLComputeContext::GetCurrentVaryingBuffer() const {

    return _currentVaryingBuffer;
}

OsdGLSLComputeKernelBundle *
OsdGLSLComputeContext::GetKernelBundle() const {

    return _kernelBundle;
}

void
OsdGLSLComputeContext::SetKernelBundle(
    OsdGLSLComputeKernelBundle *kernelBundle) {

    _kernelBundle = kernelBundle;
}

OsdGLSLComputeContext *
OsdGLSLComputeContext::Create(FarMesh<OsdVertex> const *farmesh) {

    return new OsdGLSLComputeContext(farmesh);
}

void
OsdGLSLComputeContext::BindEditShaderStorageBuffers(int editIndex) {

    const OsdGLSLComputeHEditTable * edit = _editTables[editIndex];
    const OsdGLSLComputeTable * primvarIndices = edit->GetPrimvarIndices();
    const OsdGLSLComputeTable * editValues = edit->GetEditValues();

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 9,
                     primvarIndices->GetBuffer());
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 10,
                     editValues->GetBuffer());
}

void
OsdGLSLComputeContext::UnbindEditShaderStorageBuffers() {

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 9, 0);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 10, 0);
}

void
OsdGLSLComputeContext::bindShaderStorageBuffers() {

    _kernelBundle->UseProgram();

    if (_currentVertexBuffer)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _currentVertexBuffer);

    if (_currentVaryingBuffer)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _currentVaryingBuffer);

    // XXX: should be better handling for loop subdivision.
    if (_tables[FarSubdivisionTables<OsdVertex>::F_IT]) {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2,
                         _tables[FarSubdivisionTables<OsdVertex>::F_IT]->GetBuffer());
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3,
                         _tables[FarSubdivisionTables<OsdVertex>::F_ITa]->GetBuffer());
    }

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4,
                     _tables[FarSubdivisionTables<OsdVertex>::E_IT]->GetBuffer());
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5,
                     _tables[FarSubdivisionTables<OsdVertex>::V_IT]->GetBuffer());
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6,
                     _tables[FarSubdivisionTables<OsdVertex>::V_ITa]->GetBuffer());
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7,
                     _tables[FarSubdivisionTables<OsdVertex>::E_W]->GetBuffer());
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8,
                     _tables[FarSubdivisionTables<OsdVertex>::V_W]->GetBuffer());
}

void
OsdGLSLComputeContext::unbindShaderStorageBuffers() {

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, 0);
    if (_tables[FarSubdivisionTables<OsdVertex>::F_IT]) {
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
