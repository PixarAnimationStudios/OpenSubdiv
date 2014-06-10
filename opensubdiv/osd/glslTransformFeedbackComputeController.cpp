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
    _vertexTexture(0), _varyingTexture(0), _vao(0) {
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
OsdGLSLTransformFeedbackComputeController::getKernels(
    OsdVertexBufferDescriptor const &vertexDesc,
    OsdVertexBufferDescriptor const &varyingDesc,
    bool interleaved) {

    std::vector<OsdGLSLTransformFeedbackKernelBundle*>::iterator it =
        std::find_if(_kernelRegistry.begin(), _kernelRegistry.end(),
                     OsdGLSLTransformFeedbackKernelBundle::Match(
                         vertexDesc, varyingDesc, interleaved));

    if (it != _kernelRegistry.end()) {
        return *it;
    } else {
        OsdGLSLTransformFeedbackKernelBundle *kernelBundle =
            new OsdGLSLTransformFeedbackKernelBundle();
        _kernelRegistry.push_back(kernelBundle);
        kernelBundle->Compile(vertexDesc, varyingDesc, interleaved);
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
OsdGLSLTransformFeedbackComputeController::bindResources() {

    glEnable(GL_RASTERIZER_DISCARD);
    _currentBindState.kernelBundle->UseProgram(_currentBindState.vertexDesc.offset, _currentBindState.varyingDesc.offset);

    // bind vertex texture
    if (_currentBindState.vertexBuffer) {
        if (not _vertexTexture) glGenTextures(1, &_vertexTexture);
#if defined(GL_EXT_direct_state_access)
        if (glTextureBufferEXT) {
            glTextureBufferEXT(_vertexTexture, GL_TEXTURE_BUFFER, GL_R32F, _currentBindState.vertexBuffer);
        } else {
#else
        {
#endif
            glBindTexture(GL_TEXTURE_BUFFER, _vertexTexture);
            glTexBuffer(GL_TEXTURE_BUFFER, GL_R32F, _currentBindState.vertexBuffer);
            glBindTexture(GL_TEXTURE_BUFFER, 0);
        }
    }

    if (_currentBindState.varyingBuffer) {
        if (not _varyingTexture) glGenTextures(1, &_varyingTexture);
#if defined(GL_EXT_direct_state_access)
        if (glTextureBufferEXT) {
            glTextureBufferEXT(_varyingTexture, GL_TEXTURE_BUFFER, GL_R32F, _currentBindState.varyingBuffer);
        } else {
#else
        {
#endif
            glBindTexture(GL_TEXTURE_BUFFER, _varyingTexture);
            glTexBuffer(GL_TEXTURE_BUFFER, GL_R32F, _currentBindState.varyingBuffer);
            glBindTexture(GL_TEXTURE_BUFFER, 0);
        }
    }

    if (_vertexTexture)
        bindTexture(_currentBindState.kernelBundle->GetVertexUniformLocation(), _vertexTexture, 0);
    if (_varyingTexture)
        bindTexture(_currentBindState.kernelBundle->GetVaryingUniformLocation(), _varyingTexture, 1);

    // bind vertex texture image (for edit kernel)
    glUniform1i(_currentBindState.kernelBundle->GetVertexBufferImageUniformLocation(), 0);
    glBindImageTexture(0, _vertexTexture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);

    // bind vertex array
    // always create new one, to be safe with multiple contexts.
    glGenVertexArrays(1, &_vao);
    glBindVertexArray(_vao);
}

void
OsdGLSLTransformFeedbackComputeController::unbindResources() {

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_BUFFER, 0);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_BUFFER, 0);

    // unbind vertex texture image
    glBindImageTexture(0, 0, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);

    glDisable(GL_RASTERIZER_DISCARD);
    glUseProgram(0);
    glActiveTexture(GL_TEXTURE0);

    // unbind vertex array
    glBindVertexArray(0);
    glDeleteVertexArrays(1, &_vao);
}

void
OsdGLSLTransformFeedbackComputeController::ApplyBilinearFaceVerticesKernel(
    FarKernelBatch const &batch, OsdGLSLTransformFeedbackComputeContext const *context) const {

    assert(context);

    _currentBindState.kernelBundle->ApplyBilinearFaceVerticesKernel(
        _currentBindState.vertexBuffer, _currentBindState.varyingBuffer,
        _currentBindState.vertexDesc.offset, _currentBindState.varyingDesc.offset,
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdGLSLTransformFeedbackComputeController::ApplyBilinearEdgeVerticesKernel(
    FarKernelBatch const &batch, OsdGLSLTransformFeedbackComputeContext const *context) const {

    assert(context);

    _currentBindState.kernelBundle->ApplyBilinearEdgeVerticesKernel(
        _currentBindState.vertexBuffer, _currentBindState.varyingBuffer,
        _currentBindState.vertexDesc.offset, _currentBindState.varyingDesc.offset,
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdGLSLTransformFeedbackComputeController::ApplyBilinearVertexVerticesKernel(
    FarKernelBatch const &batch, OsdGLSLTransformFeedbackComputeContext const *context) const {

    assert(context);

    _currentBindState.kernelBundle->ApplyBilinearVertexVerticesKernel(
        _currentBindState.vertexBuffer, _currentBindState.varyingBuffer,
        _currentBindState.vertexDesc.offset, _currentBindState.varyingDesc.offset,
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdGLSLTransformFeedbackComputeController::ApplyCatmarkFaceVerticesKernel(
    FarKernelBatch const &batch, OsdGLSLTransformFeedbackComputeContext const *context) const {

    assert(context);

    _currentBindState.kernelBundle->ApplyCatmarkFaceVerticesKernel(
        _currentBindState.vertexBuffer, _currentBindState.varyingBuffer,
        _currentBindState.vertexDesc.offset, _currentBindState.varyingDesc.offset,
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdGLSLTransformFeedbackComputeController::ApplyCatmarkQuadFaceVerticesKernel(
    FarKernelBatch const &batch, OsdGLSLTransformFeedbackComputeContext const *context) const {

    assert(context);

    _currentBindState.kernelBundle->ApplyCatmarkQuadFaceVerticesKernel(
        _currentBindState.vertexBuffer, _currentBindState.varyingBuffer,
        _currentBindState.vertexDesc.offset, _currentBindState.varyingDesc.offset,
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdGLSLTransformFeedbackComputeController::ApplyCatmarkTriQuadFaceVerticesKernel(
    FarKernelBatch const &batch, OsdGLSLTransformFeedbackComputeContext const *context) const {

    assert(context);

    _currentBindState.kernelBundle->ApplyCatmarkTriQuadFaceVerticesKernel(
        _currentBindState.vertexBuffer, _currentBindState.varyingBuffer,
        _currentBindState.vertexDesc.offset, _currentBindState.varyingDesc.offset,
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdGLSLTransformFeedbackComputeController::ApplyCatmarkEdgeVerticesKernel(
    FarKernelBatch const &batch, OsdGLSLTransformFeedbackComputeContext const *context) const {

    assert(context);

    _currentBindState.kernelBundle->ApplyCatmarkEdgeVerticesKernel(
        _currentBindState.vertexBuffer, _currentBindState.varyingBuffer,
        _currentBindState.vertexDesc.offset, _currentBindState.varyingDesc.offset,
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdGLSLTransformFeedbackComputeController::ApplyCatmarkRestrictedEdgeVerticesKernel(
    FarKernelBatch const &batch, OsdGLSLTransformFeedbackComputeContext const *context) const {

    assert(context);

    _currentBindState.kernelBundle->ApplyCatmarkRestrictedEdgeVerticesKernel(
        _currentBindState.vertexBuffer, _currentBindState.varyingBuffer,
        _currentBindState.vertexDesc.offset, _currentBindState.varyingDesc.offset,
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdGLSLTransformFeedbackComputeController::ApplyCatmarkVertexVerticesKernelB(
    FarKernelBatch const &batch, OsdGLSLTransformFeedbackComputeContext const *context) const {

    assert(context);

    _currentBindState.kernelBundle->ApplyCatmarkVertexVerticesKernelB(
        _currentBindState.vertexBuffer, _currentBindState.varyingBuffer,
        _currentBindState.vertexDesc.offset, _currentBindState.varyingDesc.offset,
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdGLSLTransformFeedbackComputeController::ApplyCatmarkVertexVerticesKernelA1(
    FarKernelBatch const &batch, OsdGLSLTransformFeedbackComputeContext const *context) const {

    assert(context);

    _currentBindState.kernelBundle->ApplyCatmarkVertexVerticesKernelA(
        _currentBindState.vertexBuffer, _currentBindState.varyingBuffer,
        _currentBindState.vertexDesc.offset, _currentBindState.varyingDesc.offset,
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd(), false);
}

void
OsdGLSLTransformFeedbackComputeController::ApplyCatmarkVertexVerticesKernelA2(
    FarKernelBatch const &batch, OsdGLSLTransformFeedbackComputeContext const *context) const {

    assert(context);

    _currentBindState.kernelBundle->ApplyCatmarkVertexVerticesKernelA(
        _currentBindState.vertexBuffer, _currentBindState.varyingBuffer,
        _currentBindState.vertexDesc.offset, _currentBindState.varyingDesc.offset,
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd(), true);
}

void
OsdGLSLTransformFeedbackComputeController::ApplyCatmarkRestrictedVertexVerticesKernelB1(
    FarKernelBatch const &batch, OsdGLSLTransformFeedbackComputeContext const *context) const {

    assert(context);

    _currentBindState.kernelBundle->ApplyCatmarkRestrictedVertexVerticesKernelB1(
        _currentBindState.vertexBuffer, _currentBindState.varyingBuffer,
        _currentBindState.vertexDesc.offset, _currentBindState.varyingDesc.offset,
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdGLSLTransformFeedbackComputeController::ApplyCatmarkRestrictedVertexVerticesKernelB2(
    FarKernelBatch const &batch, OsdGLSLTransformFeedbackComputeContext const *context) const {

    assert(context);

    _currentBindState.kernelBundle->ApplyCatmarkRestrictedVertexVerticesKernelB2(
        _currentBindState.vertexBuffer, _currentBindState.varyingBuffer,
        _currentBindState.vertexDesc.offset, _currentBindState.varyingDesc.offset,
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdGLSLTransformFeedbackComputeController::ApplyCatmarkRestrictedVertexVerticesKernelA(
    FarKernelBatch const &batch, OsdGLSLTransformFeedbackComputeContext const *context) const {

    assert(context);

    _currentBindState.kernelBundle->ApplyCatmarkRestrictedVertexVerticesKernelA(
        _currentBindState.vertexBuffer, _currentBindState.varyingBuffer,
        _currentBindState.vertexDesc.offset, _currentBindState.varyingDesc.offset,
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdGLSLTransformFeedbackComputeController::ApplyLoopEdgeVerticesKernel(
    FarKernelBatch const &batch, OsdGLSLTransformFeedbackComputeContext const *context) const {

    assert(context);

    _currentBindState.kernelBundle->ApplyLoopEdgeVerticesKernel(
        _currentBindState.vertexBuffer, _currentBindState.varyingBuffer,
        _currentBindState.vertexDesc.offset, _currentBindState.varyingDesc.offset,
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdGLSLTransformFeedbackComputeController::ApplyLoopVertexVerticesKernelB(
    FarKernelBatch const &batch, OsdGLSLTransformFeedbackComputeContext const *context) const {

    assert(context);

    _currentBindState.kernelBundle->ApplyLoopVertexVerticesKernelB(
        _currentBindState.vertexBuffer, _currentBindState.varyingBuffer,
        _currentBindState.vertexDesc.offset, _currentBindState.varyingDesc.offset,
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdGLSLTransformFeedbackComputeController::ApplyLoopVertexVerticesKernelA1(
    FarKernelBatch const &batch, OsdGLSLTransformFeedbackComputeContext const *context) const {

    assert(context);

    _currentBindState.kernelBundle->ApplyLoopVertexVerticesKernelA(
        _currentBindState.vertexBuffer, _currentBindState.varyingBuffer,
        _currentBindState.vertexDesc.offset, _currentBindState.varyingDesc.offset,
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd(), false);
}

void
OsdGLSLTransformFeedbackComputeController::ApplyLoopVertexVerticesKernelA2(
    FarKernelBatch const &batch, OsdGLSLTransformFeedbackComputeContext const *context) const {

    assert(context);

    _currentBindState.kernelBundle->ApplyLoopVertexVerticesKernelA(
        _currentBindState.vertexBuffer, _currentBindState.varyingBuffer,
        _currentBindState.vertexDesc.offset, _currentBindState.varyingDesc.offset,
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd(), true);
}

void
OsdGLSLTransformFeedbackComputeController::ApplyVertexEdits(
    FarKernelBatch const &batch, OsdGLSLTransformFeedbackComputeContext const *context) const {

    assert(context);

    const OsdGLSLTransformFeedbackHEditTable * edit = context->GetEditTable(batch.GetTableIndex());
    assert(edit);

    context->BindEditTextures(batch.GetTableIndex(), _currentBindState.kernelBundle);

    int primvarOffset = edit->GetPrimvarOffset();
    int primvarWidth = edit->GetPrimvarWidth();

    if (edit->GetOperation() == FarVertexEdit::Add) {
        _currentBindState.kernelBundle->ApplyEditAdd(
            _currentBindState.vertexBuffer, _currentBindState.varyingBuffer,
            _currentBindState.vertexDesc.offset, _currentBindState.varyingDesc.offset,
            primvarOffset, primvarWidth,
            batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
    } else {
        // XXX: edit SET is not implemented yet.
    }
    
    context->UnbindEditTextures();
}

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
