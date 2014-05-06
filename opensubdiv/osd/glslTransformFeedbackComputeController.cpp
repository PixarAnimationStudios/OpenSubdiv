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

#include "../osd/glslTransformFeedbackComputeController.h"
#include "../osd/glslTransformFeedbackComputeContext.h"
#include "../osd/glslTransformFeedbackKernelBundle.h"

#include "../osd/opengl.h"

#include <algorithm>
#include <cassert>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

OsdGLSLTransformFeedbackComputeController::OsdGLSLTransformFeedbackComputeController() :
    _vertexTexture(0), _varyingTexture(0),
    _currentVertexBuffer(0), _currentVaryingBuffer(0),
    _currentKernelBundle(NULL) {
}

OsdGLSLTransformFeedbackComputeController::~OsdGLSLTransformFeedbackComputeController() {

    for (std::vector<OsdGLSLTransformFeedbackKernelBundle*>::iterator it =
             _kernelRegistry.begin();
         it != _kernelRegistry.end(); ++it) {
        delete *it;
    }
    if (_vertexTexture) glDeleteTextures(1, &_vertexTexture);
    if (_varyingTexture) glDeleteTextures(1, &_varyingTexture);
}

void
OsdGLSLTransformFeedbackComputeController::Synchronize() {

    glFinish();
}

OsdGLSLTransformFeedbackKernelBundle *
OsdGLSLTransformFeedbackComputeController::getKernels(int numVertexElements,
                                     int numVaryingElements) {

    std::vector<OsdGLSLTransformFeedbackKernelBundle*>::iterator it =
        std::find_if(_kernelRegistry.begin(), _kernelRegistry.end(),
                     OsdGLSLTransformFeedbackKernelBundle::Match(numVertexElements,
                                                numVaryingElements));
    if (it != _kernelRegistry.end()) {
        return *it;
    } else {
        OsdGLSLTransformFeedbackKernelBundle *kernelBundle = new OsdGLSLTransformFeedbackKernelBundle();
        _kernelRegistry.push_back(kernelBundle);
        kernelBundle->Compile(numVertexElements, numVaryingElements);
        return kernelBundle;
    }
}

static void
bindTexture(GLint samplerUniform, GLuint texture, int unit) {

    if (samplerUniform == -1) return;
    glUniform1i(samplerUniform, unit);
    glActiveTexture(GL_TEXTURE0 + unit);
    glBindTexture(GL_TEXTURE_BUFFER, texture);
    glActiveTexture(GL_TEXTURE0);
}

void
OsdGLSLTransformFeedbackComputeController::bindTextures() {

    glEnable(GL_RASTERIZER_DISCARD);
    _currentKernelBundle->UseProgram();

    // bind vertex texture
    if (_currentVertexBuffer) {
        if (not _vertexTexture) glGenTextures(1, &_vertexTexture);
#if defined(GL_EXT_direct_state_access)
        if (glTextureBufferEXT) {
            glTextureBufferEXT(_vertexTexture, GL_TEXTURE_BUFFER, GL_R32F, _currentVertexBuffer);
        } else {
#else
        {
#endif
            glBindTexture(GL_TEXTURE_BUFFER, _vertexTexture);
            glTexBuffer(GL_TEXTURE_BUFFER, GL_R32F, _currentVertexBuffer);
            glBindTexture(GL_TEXTURE_BUFFER, 0);
        }
    }

    if (_currentVaryingBuffer) {
        if (not _varyingTexture) glGenTextures(1, &_varyingTexture);
#if defined(GL_EXT_direct_state_access)
        if (glTextureBufferEXT) {
            glTextureBufferEXT(_varyingTexture, GL_TEXTURE_BUFFER, GL_R32F, _currentVaryingBuffer);
        } else {
#else
        {
#endif
            glBindTexture(GL_TEXTURE_BUFFER, _varyingTexture);
            glTexBuffer(GL_TEXTURE_BUFFER, GL_R32F, _currentVaryingBuffer);
            glBindTexture(GL_TEXTURE_BUFFER, 0);
        }
    }

    if (_vertexTexture)
        bindTexture(_currentKernelBundle->GetVertexUniformLocation(), _vertexTexture, 0);
    if (_varyingTexture)
        bindTexture(_currentKernelBundle->GetVaryingUniformLocation(), _varyingTexture, 1);

    // bind vertex texture image (for edit kernel)
    glUniform1i(_currentKernelBundle->GetVertexBufferImageUniformLocation(), 0);
    glBindImageTexture(0, _vertexTexture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);
}

void
OsdGLSLTransformFeedbackComputeController::unbindTextures() {

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_BUFFER, 0);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_BUFFER, 0);

    // unbind vertex texture image
    glBindImageTexture(0, 0, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);

    glDisable(GL_RASTERIZER_DISCARD);
    glUseProgram(0);
    glActiveTexture(GL_TEXTURE0);
}

void
OsdGLSLTransformFeedbackComputeController::ApplyBilinearFaceVerticesKernel(
    FarKernelBatch const &batch, OsdGLSLTransformFeedbackComputeContext const *context) const {

    assert(context);

    _currentKernelBundle->ApplyBilinearFaceVerticesKernel(
        _currentVertexBuffer, _vdesc.numVertexElements,
        _currentVaryingBuffer, _vdesc.numVaryingElements,
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdGLSLTransformFeedbackComputeController::ApplyBilinearEdgeVerticesKernel(
    FarKernelBatch const &batch, OsdGLSLTransformFeedbackComputeContext const *context) const {

    assert(context);

    _currentKernelBundle->ApplyBilinearEdgeVerticesKernel(
        _currentVertexBuffer, _vdesc.numVertexElements,
        _currentVaryingBuffer, _vdesc.numVaryingElements,
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdGLSLTransformFeedbackComputeController::ApplyBilinearVertexVerticesKernel(
    FarKernelBatch const &batch, OsdGLSLTransformFeedbackComputeContext const *context) const {

    assert(context);

    _currentKernelBundle->ApplyBilinearVertexVerticesKernel(
        _currentVertexBuffer, _vdesc.numVertexElements,
        _currentVaryingBuffer, _vdesc.numVaryingElements,
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdGLSLTransformFeedbackComputeController::ApplyCatmarkFaceVerticesKernel(
    FarKernelBatch const &batch, OsdGLSLTransformFeedbackComputeContext const *context) const {

    assert(context);

    _currentKernelBundle->ApplyCatmarkFaceVerticesKernel(
        _currentVertexBuffer, _vdesc.numVertexElements,
        _currentVaryingBuffer, _vdesc.numVaryingElements,
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}



void
OsdGLSLTransformFeedbackComputeController::ApplyCatmarkEdgeVerticesKernel(
    FarKernelBatch const &batch, OsdGLSLTransformFeedbackComputeContext const *context) const {

    assert(context);

    _currentKernelBundle->ApplyCatmarkEdgeVerticesKernel(
        _currentVertexBuffer, _vdesc.numVertexElements,
        _currentVaryingBuffer, _vdesc.numVaryingElements,
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdGLSLTransformFeedbackComputeController::ApplyCatmarkVertexVerticesKernelB(
    FarKernelBatch const &batch, OsdGLSLTransformFeedbackComputeContext const *context) const {

    assert(context);

    _currentKernelBundle->ApplyCatmarkVertexVerticesKernelB(
        _currentVertexBuffer, _vdesc.numVertexElements,
        _currentVaryingBuffer, _vdesc.numVaryingElements,
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdGLSLTransformFeedbackComputeController::ApplyCatmarkVertexVerticesKernelA1(
    FarKernelBatch const &batch, OsdGLSLTransformFeedbackComputeContext const *context) const {

    assert(context);

    _currentKernelBundle->ApplyCatmarkVertexVerticesKernelA(
        _currentVertexBuffer, _vdesc.numVertexElements,
        _currentVaryingBuffer, _vdesc.numVaryingElements,
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd(), false);
}

void
OsdGLSLTransformFeedbackComputeController::ApplyCatmarkVertexVerticesKernelA2(
    FarKernelBatch const &batch, OsdGLSLTransformFeedbackComputeContext const *context) const {

    assert(context);

    _currentKernelBundle->ApplyCatmarkVertexVerticesKernelA(
        _currentVertexBuffer, _vdesc.numVertexElements,
        _currentVaryingBuffer, _vdesc.numVaryingElements,
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd(), true);
}

void
OsdGLSLTransformFeedbackComputeController::ApplyLoopEdgeVerticesKernel(
    FarKernelBatch const &batch, OsdGLSLTransformFeedbackComputeContext const *context) const {

    assert(context);

    _currentKernelBundle->ApplyLoopEdgeVerticesKernel(
        _currentVertexBuffer, _vdesc.numVertexElements,
        _currentVaryingBuffer, _vdesc.numVaryingElements,
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdGLSLTransformFeedbackComputeController::ApplyLoopVertexVerticesKernelB(
    FarKernelBatch const &batch, OsdGLSLTransformFeedbackComputeContext const *context) const {

    assert(context);

    _currentKernelBundle->ApplyLoopVertexVerticesKernelB(
        _currentVertexBuffer, _vdesc.numVertexElements,
        _currentVaryingBuffer, _vdesc.numVaryingElements,
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdGLSLTransformFeedbackComputeController::ApplyLoopVertexVerticesKernelA1(
    FarKernelBatch const &batch, OsdGLSLTransformFeedbackComputeContext const *context) const {

    assert(context);

    _currentKernelBundle->ApplyLoopVertexVerticesKernelA(
        _currentVertexBuffer, _vdesc.numVertexElements,
        _currentVaryingBuffer, _vdesc.numVaryingElements,
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd(), false);
}

void
OsdGLSLTransformFeedbackComputeController::ApplyLoopVertexVerticesKernelA2(
    FarKernelBatch const &batch, OsdGLSLTransformFeedbackComputeContext const *context) const {

    assert(context);

    _currentKernelBundle->ApplyLoopVertexVerticesKernelA(
        _currentVertexBuffer, _vdesc.numVertexElements,
        _currentVaryingBuffer, _vdesc.numVaryingElements,
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd(), true);
}

void
OsdGLSLTransformFeedbackComputeController::ApplyVertexEdits(
    FarKernelBatch const &batch, OsdGLSLTransformFeedbackComputeContext const *context) const {

    assert(context);

    const OsdGLSLTransformFeedbackHEditTable * edit = context->GetEditTable(batch.GetTableIndex());
    assert(edit);

    context->BindEditTextures(batch.GetTableIndex(), _currentKernelBundle);

    int primvarOffset = edit->GetPrimvarOffset();
    int primvarWidth = edit->GetPrimvarWidth();

    if (edit->GetOperation() == FarVertexEdit::Add) {
        _currentKernelBundle->ApplyEditAdd(
            _currentVertexBuffer, _vdesc.numVertexElements,
            _currentVaryingBuffer, _vdesc.numVaryingElements,
            primvarOffset, primvarWidth,
            batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
    } else {
        // XXX: edit SET is not implemented yet.
    }
    
    context->UnbindEditTextures();
}

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
