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
    
    glUniform1i(config->levelBaseUniform, patchArray.GetPatchIndex());
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
