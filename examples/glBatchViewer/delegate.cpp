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

#include "delegate.h"

#include <osd/opengl.h>

MyDrawContext::MyDrawContext() {
    glGenVertexArrays(1, &_vao);
}

MyDrawContext::~MyDrawContext() {
    glDeleteVertexArrays(1, &_vao);
}

MyDrawContext*
MyDrawContext::Create(OpenSubdiv::FarPatchTables const *patchTables, bool requireFVarData)
{
    MyDrawContext * result = new MyDrawContext();

    if (patchTables) {
        if (result->create(patchTables, requireFVarData)) {
            return result;
        } else {
            delete result;
        }
    }
    return NULL;
}

// ----------------------------------------------------------------------------

void
MyDrawDelegate::Bind(OpenSubdiv::OsdUtilMeshBatchBase<MyDrawContext> *batch, EffectHandle const &effect) {

    if (batch != _currentBatch) {
        // bind batch
        _currentBatch = batch;
        MyDrawContext *drawContext = batch->GetDrawContext();

        // bind vao
        glBindVertexArray(drawContext->GetVertexArray());

        // bind vbo state
        glBindBuffer(GL_ARRAY_BUFFER, batch->BindVertexBuffer());
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, drawContext->GetPatchIndexBuffer());

        // vertex attrib
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof (GLfloat) * 3, 0);

        if (effect->displayStyle == kVaryingColor) {
            glBindBuffer(GL_ARRAY_BUFFER, batch->BindVaryingBuffer());
            glEnableVertexAttribArray(1);
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof (GLfloat) * 3, 0);
        }

        // bind other builtin texture buffers
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_BUFFER, drawContext->GetVertexTextureBuffer());
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_BUFFER, drawContext->GetVertexValenceTextureBuffer());
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_BUFFER, drawContext->GetQuadOffsetsTextureBuffer());
        glActiveTexture(GL_TEXTURE3);
        glBindTexture(GL_TEXTURE_BUFFER, drawContext->GetPatchParamTextureBuffer());

        if (drawContext->GetFvarDataTextureBuffer()) {
            glActiveTexture(GL_TEXTURE4);
            glBindTexture(GL_TEXTURE_BUFFER, drawContext->GetFvarDataTextureBuffer());
        }
    }
    if (effect != _currentEffect) {
        _currentEffect = effect;

        // bind effect
    }
}

void
MyDrawDelegate::Unbind(OpenSubdiv::OsdUtilMeshBatchBase<MyDrawContext> *batch, EffectHandle const &effect) {
}

void
MyDrawDelegate::Begin() {
    _currentBatch = NULL;
    _currentEffect = NULL;
}

void
MyDrawDelegate::End() {
    _currentBatch = NULL;
    _currentEffect = NULL;
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
}

void
MyDrawDelegate::DrawElements(OpenSubdiv::OsdDrawContext::PatchArray const &patchArray) {

    // bind patchType-wise effect state
    // can be skipped (if config is not changed)
    MyDrawConfig *config = GetDrawConfig(_currentEffect, patchArray.GetDescriptor());

    GLuint program = config->program;

    if (true /* if config is different from previous call */) {
        glUseProgram(program);
#if defined(GL_ARB_tessellation_shader) || defined(GL_VERSION_4_0)
        glPatchParameteri(GL_PATCH_VERTICES, patchArray.GetDescriptor().GetNumControlVertices());
#endif
        // bind patchArray state and draw
    }
    
    // apply patch color
    _currentEffect->BindDrawConfig(config, patchArray.GetDescriptor());
    
    glUniform1i(config->primitiveIdBaseUniform, patchArray.GetPatchIndex());
    if (patchArray.GetDescriptor().GetType() == OpenSubdiv::FarPatchTables::GREGORY ||
        patchArray.GetDescriptor().GetType() == OpenSubdiv::FarPatchTables::GREGORY_BOUNDARY){
        glUniform1i(config->gregoryQuadOffsetBaseUniform, patchArray.GetQuadOffsetIndex());
    }
    
    if (patchArray.GetDescriptor().GetType() == OpenSubdiv::FarPatchTables::QUADS) {
        glDrawElements(GL_LINES_ADJACENCY, patchArray.GetNumIndices(), GL_UNSIGNED_INT,
                       (void*)(patchArray.GetVertIndex()*sizeof(GLuint)));
    } else {
#if defined(GL_ARB_tessellation_shader) || defined(GL_VERSION_4_0)
        glDrawElements(GL_PATCHES, patchArray.GetNumIndices(), GL_UNSIGNED_INT,
                       (void*)(patchArray.GetVertIndex()*sizeof(GLuint)));
#endif
    }
    _numDrawCalls++;
}

bool
MyDrawDelegate::IsCombinable(EffectHandle const &a, EffectHandle const &b) const {
    return a == b;
}

MyDrawConfig *
MyDrawDelegate::GetDrawConfig(EffectHandle &effect, OpenSubdiv::OsdDrawContext::PatchDescriptor desc) {

    return _effectRegistry.GetDrawConfig(effect->GetEffectDescriptor(), desc);
}
