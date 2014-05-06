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

#include "../version.h"

#include "../far/subdivisionTables.h"
#include "../osd/debug.h"
#include "../osd/glslTransformFeedbackComputeContext.h"
#include "../osd/glslTransformFeedbackKernelBundle.h"

#include "../osd/opengl.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

void
OsdGLSLTransformFeedbackTable::createTextureBuffer(size_t size, const void *ptr, GLenum type) {

    GLuint buffer;
    glGenBuffers(1, &buffer);
    glGenTextures(1, &_texture);

#if defined(GL_EXT_direct_state_access)
    if (glNamedBufferDataEXT and glTextureBufferEXT) {
        glNamedBufferDataEXT(buffer, size, ptr, GL_STATIC_DRAW);
        glTextureBufferEXT(_texture, GL_TEXTURE_BUFFER, type, buffer);
    } else {
#else
    {
#endif
        GLint prev = 0;

        glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &prev);
        glBindBuffer(GL_ARRAY_BUFFER, buffer);
        glBufferData(GL_ARRAY_BUFFER, size, ptr, GL_STATIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, prev);

        glGetIntegerv(GL_TEXTURE_BINDING_BUFFER, &prev);
        glBindTexture(GL_TEXTURE_BUFFER, _texture);
        glTexBuffer(GL_TEXTURE_BUFFER, type, buffer);
        glBindTexture(GL_TEXTURE_BUFFER, prev);
    }

    glDeleteBuffers(1, &buffer);
}

OsdGLSLTransformFeedbackTable::~OsdGLSLTransformFeedbackTable() {

    glDeleteTextures(1, &_texture);
}

GLuint
OsdGLSLTransformFeedbackTable::GetTexture() const {

    return _texture;
}

// ----------------------------------------------------------------------------

OsdGLSLTransformFeedbackHEditTable::OsdGLSLTransformFeedbackHEditTable(
    const FarVertexEditTables::VertexEditBatch &batch)
    : _primvarIndicesTable(new OsdGLSLTransformFeedbackTable(batch.GetVertexIndices(), GL_R32UI)),
      _editValuesTable(new OsdGLSLTransformFeedbackTable(batch.GetValues(), GL_R32F)) {

    _operation = batch.GetOperation();
    _primvarOffset = batch.GetPrimvarIndex();
    _primvarWidth = batch.GetPrimvarWidth();
}

OsdGLSLTransformFeedbackHEditTable::~OsdGLSLTransformFeedbackHEditTable() {

    delete _primvarIndicesTable;
    delete _editValuesTable;
}

const OsdGLSLTransformFeedbackTable *
OsdGLSLTransformFeedbackHEditTable::GetPrimvarIndices() const {

    return _primvarIndicesTable;
}

const OsdGLSLTransformFeedbackTable *
OsdGLSLTransformFeedbackHEditTable::GetEditValues() const {

    return _editValuesTable;
}

int
OsdGLSLTransformFeedbackHEditTable::GetOperation() const {

    return _operation;
}

int
OsdGLSLTransformFeedbackHEditTable::GetPrimvarOffset() const {

    return _primvarOffset;
}

int
OsdGLSLTransformFeedbackHEditTable::GetPrimvarWidth() const {

    return _primvarWidth;
}

// ----------------------------------------------------------------------------

OsdGLSLTransformFeedbackComputeContext::OsdGLSLTransformFeedbackComputeContext(
    FarSubdivisionTables const *subdivisionTables,
    FarVertexEditTables const *vertexEditTables) {

    // allocate 5 or 7 tables
    _tables.resize(7, 0);

    _tables[FarSubdivisionTables::E_IT]  = new OsdGLSLTransformFeedbackTable(subdivisionTables->Get_E_IT(), GL_R32I);
    _tables[FarSubdivisionTables::V_IT]  = new OsdGLSLTransformFeedbackTable(subdivisionTables->Get_V_IT(), GL_R32UI);
    _tables[FarSubdivisionTables::V_ITa] = new OsdGLSLTransformFeedbackTable(subdivisionTables->Get_V_ITa(), GL_R32I);
    _tables[FarSubdivisionTables::E_W]   = new OsdGLSLTransformFeedbackTable(subdivisionTables->Get_E_W(), GL_R32F);
    _tables[FarSubdivisionTables::V_W]   = new OsdGLSLTransformFeedbackTable(subdivisionTables->Get_V_W(), GL_R32F);

    if (subdivisionTables->GetNumTables() > 5) {
        // catmark, bilinear
        _tables[FarSubdivisionTables::F_IT]  = new OsdGLSLTransformFeedbackTable(subdivisionTables->Get_F_IT(), GL_R32UI);
        _tables[FarSubdivisionTables::F_ITa] = new OsdGLSLTransformFeedbackTable(subdivisionTables->Get_F_ITa(), GL_R32I);
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
            const FarVertexEditTables::VertexEditBatch & edit = vertexEditTables->GetBatch(i);
            _editTables.push_back(new OsdGLSLTransformFeedbackHEditTable(edit));
        }
    }
}

OsdGLSLTransformFeedbackComputeContext::~OsdGLSLTransformFeedbackComputeContext() {

    for (size_t i = 0; i < _tables.size(); ++i) {
        delete _tables[i];
    }
    for (size_t i = 0; i < _editTables.size(); ++i) {
        delete _editTables[i];
    }
}

const OsdGLSLTransformFeedbackTable *
OsdGLSLTransformFeedbackComputeContext::GetTable(int tableIndex) const {

    return _tables[tableIndex];
}

int
OsdGLSLTransformFeedbackComputeContext::GetNumEditTables() const {

    return static_cast<int>(_editTables.size());
}

const OsdGLSLTransformFeedbackHEditTable *
OsdGLSLTransformFeedbackComputeContext::GetEditTable(int tableIndex) const {

    return _editTables[tableIndex];
}

OsdGLSLTransformFeedbackComputeContext *
OsdGLSLTransformFeedbackComputeContext::Create(FarSubdivisionTables const *subdivisionTables,
                                               FarVertexEditTables const *vertexEditTables) {

    return new OsdGLSLTransformFeedbackComputeContext(subdivisionTables, vertexEditTables);
}

void
OsdGLSLTransformFeedbackComputeContext::BindEditTextures(
    int editIndex, OsdGLSLTransformFeedbackKernelBundle const *kernelBundle) const {

    const OsdGLSLTransformFeedbackHEditTable * edit = _editTables[editIndex];
    const OsdGLSLTransformFeedbackTable * primvarIndices = edit->GetPrimvarIndices();
    const OsdGLSLTransformFeedbackTable * editValues = edit->GetEditValues();

    bindTexture(kernelBundle->GetEditIndicesUniformLocation(),
                primvarIndices->GetTexture(), 9);
    bindTexture(kernelBundle->GetEditValuesUniformLocation(),
                editValues->GetTexture(), 10);
}

void
OsdGLSLTransformFeedbackComputeContext::UnbindEditTextures() const {

    unbindTexture(9);
    unbindTexture(10);
}

void
OsdGLSLTransformFeedbackComputeContext::bindTexture(GLint samplerUniform, GLuint texture, int unit) const {

    if (samplerUniform == -1) return;
    glUniform1i(samplerUniform, unit);
    glActiveTexture(GL_TEXTURE0 + unit);
    glBindTexture(GL_TEXTURE_BUFFER, texture);
    glActiveTexture(GL_TEXTURE0);
}

void
OsdGLSLTransformFeedbackComputeContext::unbindTexture(GLuint unit) const {

    glActiveTexture(GL_TEXTURE0 + unit);
    glBindTexture(GL_TEXTURE_BUFFER, 0);
}

void
OsdGLSLTransformFeedbackComputeContext::BindTableTextures(
    OsdGLSLTransformFeedbackKernelBundle const *kernelBundle) const {

    // XXX: loop...
    if (_tables[FarSubdivisionTables::F_IT]) {
        bindTexture(kernelBundle->GetTableUniformLocation(FarSubdivisionTables::F_IT),
                    _tables[FarSubdivisionTables::F_IT]->GetTexture(),  2);
        bindTexture(kernelBundle->GetTableUniformLocation(FarSubdivisionTables::F_ITa),
                    _tables[FarSubdivisionTables::F_ITa]->GetTexture(), 3);
    }

    bindTexture(kernelBundle->GetTableUniformLocation(FarSubdivisionTables::E_IT),
                _tables[FarSubdivisionTables::E_IT]->GetTexture(),  4);
    bindTexture(kernelBundle->GetTableUniformLocation(FarSubdivisionTables::V_IT),
                _tables[FarSubdivisionTables::V_IT]->GetTexture(),  5);
    bindTexture(kernelBundle->GetTableUniformLocation(FarSubdivisionTables::V_ITa),
                _tables[FarSubdivisionTables::V_ITa]->GetTexture(), 6);
    bindTexture(kernelBundle->GetTableUniformLocation(FarSubdivisionTables::E_W),
                _tables[FarSubdivisionTables::E_W]->GetTexture(),   7);
    bindTexture(kernelBundle->GetTableUniformLocation(FarSubdivisionTables::V_W),
                _tables[FarSubdivisionTables::V_W]->GetTexture(),   8);
}

void
OsdGLSLTransformFeedbackComputeContext::UnbindTableTextures() const {

    for (int i = 8; i >= 2; --i) {
        unbindTexture(i);
    }
}

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
