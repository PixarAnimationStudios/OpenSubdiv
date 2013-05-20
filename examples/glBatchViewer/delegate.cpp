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

#include "delegate.h"

MyDrawContext::MyDrawContext() {
    glGenVertexArrays(1, &vao);
}

MyDrawContext::~MyDrawContext() {
    glDeleteVertexArrays(1, &vao);
}

MyDrawContext*
MyDrawContext::Create(OpenSubdiv::FarMesh<OpenSubdiv::OsdVertex> const *farMesh, bool requireFVarData)
{
    MyDrawContext *instance = new MyDrawContext();

    OpenSubdiv::FarPatchTables const * patchTables = farMesh->GetPatchTables();

    if (patchTables) {
        return Create(patchTables, requireFVarData);
    }

    delete instance;
    return NULL;
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
MyDrawDelegate::BindBatch(OpenSubdiv::OsdUtilMeshBatchBase<MyDrawContext> *batch) {

    MyDrawContext *drawContext = batch->GetDrawContext();

    // bind vao
    glBindVertexArray(drawContext->vao);

    // bind vbo state
    // glBindVertexArray(batch->vao);
    glBindBuffer(GL_ARRAY_BUFFER, batch->BindVertexBuffer());
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, drawContext->patchIndexBuffer);

    // vertex attrib
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof (GLfloat) * 3, 0);

    // bind other builtin texture buffers
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_BUFFER, drawContext->vertexTextureBuffer);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_BUFFER, drawContext->vertexValenceTextureBuffer);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_BUFFER, drawContext->quadOffsetTextureBuffer);
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_BUFFER, drawContext->ptexCoordinateTextureBuffer);
}

void
MyDrawDelegate::UnbindBatch(OpenSubdiv::OsdUtilMeshBatchBase<MyDrawContext> *batch) {

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void
MyDrawDelegate::BindEffect(MyEffect &effect) {
    // cross-patch state
    // bind ptex etc
}

void
MyDrawDelegate::UnbindEffect(MyEffect &effect) {
}

void
MyDrawDelegate::DrawElements(MyEffect &effect, OpenSubdiv::OsdDrawContext::PatchArray const &patchArray) {

    // bind patchType-wise effect state
    // can be skipped (if config is not changed)
    MyDrawConfig *config = GetDrawConfig(effect, patchArray.GetDescriptor());

    GLuint program = config->program;

    if (true /* if config is different from previous call */) {
        glUseProgram(program);
#if defined(GL_ARB_tessellation_shader) || defined(GL_VERSION_4_0)
        glPatchParameteri(GL_PATCH_VERTICES, patchArray.GetDescriptor().GetNumControlVertices());
#endif
        // bind patchArray state and draw
    }
    
    // apply patch color
    effect.BindDrawConfig(config, patchArray.GetDescriptor());
    
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

MyDrawConfig *
MyDrawDelegate::GetDrawConfig(MyEffect &effect, OpenSubdiv::OsdDrawContext::PatchDescriptor desc) {

    return _effectRegistry.GetDrawConfig(effect.GetEffectDescriptor(), desc);
}
