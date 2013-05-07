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

#if defined(__APPLE__)
    #include "TargetConditionals.h"
    #if TARGET_OS_IPHONE or TARGET_IPHONE_SIMULATOR
        #include <OpenGLES/ES2/gl.h>
    #else
        #include <OpenGL/gl3.h>
    #endif
#elif defined(ANDROID)
    #include <GLES2/gl2.h>
#else
    #if defined(_WIN32)
        #include <windows.h>
    #endif
    #include <GL/glew.h>
#endif

#include "../far/mesh.h"
#include "../far/subdivisionTables.h"
#include "../osd/glslTransformFeedbackComputeContext.h"
#include "../osd/glslTransformFeedbackKernelBundle.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

void
OsdGLSLTransformFeedbackTable::createTextureBuffer(size_t size, const void *ptr, GLenum type) {

    glGenBuffers(1, &_devicePtr);
    glGenTextures(1, &_texture);

    GLint prev = 0;
    glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &prev);
    glBindBuffer(GL_ARRAY_BUFFER, _devicePtr);
    glBufferData(GL_ARRAY_BUFFER, size, ptr, GL_STATIC_DRAW);
/*
  CHECK_GL_ERROR("UpdateTable tableIndex %d, size %ld, buffer =%d\n",
  tableIndex, size, _tableBuffers[tableIndex]);
*/
    glBindBuffer(GL_ARRAY_BUFFER, prev);

    glGetIntegerv(GL_TEXTURE_BINDING_BUFFER, &prev);
    glBindTexture(GL_TEXTURE_BUFFER, _texture);
    glTexBuffer(GL_TEXTURE_BUFFER, type, _devicePtr);
    glBindTexture(GL_TEXTURE_BUFFER, prev);

    glDeleteBuffers(1, &_devicePtr);
}

OsdGLSLTransformFeedbackTable::~OsdGLSLTransformFeedbackTable() {

    glDeleteTextures(1, &_texture);
}

GLuint
OsdGLSLTransformFeedbackTable::GetTexture() const {

    return _texture;
}

// ----------------------------------------------------------------------------

OsdGLSLTransformFeedbackHEditTable::OsdGLSLTransformFeedbackHEditTable(const FarVertexEditTables<OsdVertex>::VertexEditBatch &batch)
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
    FarMesh<OsdVertex> *farMesh) :
    _vertexTexture(0), _varyingTexture(0) {

    FarSubdivisionTables<OsdVertex> const * farTables =
        farMesh->GetSubdivisionTables();

    // allocate 5 or 7 tables
    _tables.resize(7, 0);

    _tables[FarSubdivisionTables<OsdVertex>::E_IT]  = new OsdGLSLTransformFeedbackTable(farTables->Get_E_IT(), GL_R32I);
    _tables[FarSubdivisionTables<OsdVertex>::V_IT]  = new OsdGLSLTransformFeedbackTable(farTables->Get_V_IT(), GL_R32UI);
    _tables[FarSubdivisionTables<OsdVertex>::V_ITa] = new OsdGLSLTransformFeedbackTable(farTables->Get_V_ITa(), GL_R32I);
    _tables[FarSubdivisionTables<OsdVertex>::E_W]   = new OsdGLSLTransformFeedbackTable(farTables->Get_E_W(), GL_R32F);
    _tables[FarSubdivisionTables<OsdVertex>::V_W]   = new OsdGLSLTransformFeedbackTable(farTables->Get_V_W(), GL_R32F);

    if (farTables->GetNumTables() > 5) {
        // catmark, bilinear
        _tables[FarSubdivisionTables<OsdVertex>::F_IT]  = new OsdGLSLTransformFeedbackTable(farTables->Get_F_IT(), GL_R32UI);
        _tables[FarSubdivisionTables<OsdVertex>::F_ITa] = new OsdGLSLTransformFeedbackTable(farTables->Get_F_ITa(), GL_R32I);
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
            const FarVertexEditTables<OsdVertex>::VertexEditBatch & edit = editTables->GetBatch(i);
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
    if (_vertexTexture) glDeleteTextures(1, &_vertexTexture);
    if (_varyingTexture) glDeleteTextures(1, &_varyingTexture);
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

GLuint
OsdGLSLTransformFeedbackComputeContext::GetCurrentVertexBuffer() const {

    return _currentVertexBuffer;
}

GLuint
OsdGLSLTransformFeedbackComputeContext::GetCurrentVaryingBuffer() const {

    return _currentVaryingBuffer;
}

int
OsdGLSLTransformFeedbackComputeContext::GetNumCurrentVertexElements() const {

    return _numVertexElements;
}

int
OsdGLSLTransformFeedbackComputeContext::GetNumCurrentVaryingElements() const {
    
    return _numVaryingElements;
}

OsdGLSLTransformFeedbackKernelBundle *
OsdGLSLTransformFeedbackComputeContext::GetKernelBundle() const {

    return _kernelBundle;
}

void
OsdGLSLTransformFeedbackComputeContext::SetKernelBundle(OsdGLSLTransformFeedbackKernelBundle *kernelBundle) {

    _kernelBundle = kernelBundle;
}

OsdGLSLTransformFeedbackComputeContext *
OsdGLSLTransformFeedbackComputeContext::Create(FarMesh<OsdVertex> *farmesh) {

    return new OsdGLSLTransformFeedbackComputeContext(farmesh);
}

void
OsdGLSLTransformFeedbackComputeContext::BindEditTextures(int editIndex) {

    const OsdGLSLTransformFeedbackHEditTable * edit = _editTables[editIndex];
    const OsdGLSLTransformFeedbackTable * primvarIndices = edit->GetPrimvarIndices();
    const OsdGLSLTransformFeedbackTable * editValues = edit->GetEditValues();

    bindTexture(_kernelBundle->GetEditIndicesUniformLocation(),
                primvarIndices->GetTexture(), 9);
    bindTexture(_kernelBundle->GetEditValuesUniformLocation(),
                editValues->GetTexture(), 10);
}

void
OsdGLSLTransformFeedbackComputeContext::UnbindEditTextures() {

    unbindTexture(9);
    unbindTexture(10);
}

void
OsdGLSLTransformFeedbackComputeContext::bindTexture(GLuint sampler, GLuint texture, int unit) {

    glUniform1i(sampler, unit);
    glActiveTexture(GL_TEXTURE0 + unit);
    glBindTexture(GL_TEXTURE_BUFFER, texture);
    glActiveTexture(GL_TEXTURE0);
}

void
OsdGLSLTransformFeedbackComputeContext::unbindTexture(GLuint unit) {

    glActiveTexture(GL_TEXTURE0 + unit);
    glBindTexture(GL_TEXTURE_BUFFER, 0);
}

void
OsdGLSLTransformFeedbackComputeContext::bindTextures() {

    glEnable(GL_RASTERIZER_DISCARD);
    _kernelBundle->UseProgram();

    // bind vertex texture
    if (_currentVertexBuffer) {
        if (not _vertexTexture) glGenTextures(1, &_vertexTexture);
        glBindTexture(GL_TEXTURE_BUFFER, _vertexTexture);
        glTexBuffer(GL_TEXTURE_BUFFER, GL_R32F, _currentVertexBuffer);
        glBindTexture(GL_TEXTURE_BUFFER, 0);
    }
    bindTexture(_kernelBundle->GetVertexUniformLocation(), _vertexTexture, 0);

    if (_currentVaryingBuffer) {
        if (not _varyingTexture) glGenTextures(1, &_varyingTexture);
        glBindTexture(GL_TEXTURE_BUFFER, _varyingTexture);
        glTexBuffer(GL_TEXTURE_BUFFER, GL_R32F, _currentVaryingBuffer);
        glBindTexture(GL_TEXTURE_BUFFER, 0);
    }
    bindTexture(_kernelBundle->GetVaryingUniformLocation(), _varyingTexture, 1);

    // XXX: loop...
    if (_tables[FarSubdivisionTables<OsdVertex>::F_IT]) {
        bindTexture(_kernelBundle->GetTableUniformLocation(FarSubdivisionTables<OsdVertex>::F_IT),
                    _tables[FarSubdivisionTables<OsdVertex>::F_IT]->GetTexture(),  2);
        bindTexture(_kernelBundle->GetTableUniformLocation(FarSubdivisionTables<OsdVertex>::F_ITa),
                    _tables[FarSubdivisionTables<OsdVertex>::F_ITa]->GetTexture(), 3);
    }

    bindTexture(_kernelBundle->GetTableUniformLocation(FarSubdivisionTables<OsdVertex>::E_IT),
                _tables[FarSubdivisionTables<OsdVertex>::E_IT]->GetTexture(),  4);
    bindTexture(_kernelBundle->GetTableUniformLocation(FarSubdivisionTables<OsdVertex>::V_IT),
                _tables[FarSubdivisionTables<OsdVertex>::V_IT]->GetTexture(),  5);
    bindTexture(_kernelBundle->GetTableUniformLocation(FarSubdivisionTables<OsdVertex>::V_ITa),
                _tables[FarSubdivisionTables<OsdVertex>::V_ITa]->GetTexture(), 6);
    bindTexture(_kernelBundle->GetTableUniformLocation(FarSubdivisionTables<OsdVertex>::E_W),
                _tables[FarSubdivisionTables<OsdVertex>::E_W]->GetTexture(),   7);
    bindTexture(_kernelBundle->GetTableUniformLocation(FarSubdivisionTables<OsdVertex>::V_W),
                _tables[FarSubdivisionTables<OsdVertex>::V_W]->GetTexture(),   8);

    // bind texture image (for edit kernel)
    glUniform1i(_kernelBundle->GetVertexBufferImageUniformLocation(), 0);
    glBindImageTexture(0, _vertexTexture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);
}

void
OsdGLSLTransformFeedbackComputeContext::unbindTextures() {

    for (int i = 8; i >= 0; --i) {
        unbindTexture(i);
    }
    glDisable(GL_RASTERIZER_DISCARD);
    glUseProgram(0);
}

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
