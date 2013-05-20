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
#include "../osd/debug.h"
#include "../osd/error.h"
#include "../osd/glslComputeContext.h"
#include "../osd/glslKernelBundle.h"

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
