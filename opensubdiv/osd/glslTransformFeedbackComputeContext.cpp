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

#include "../far/stencilTables.h"

#include "../osd/error.h"
//#define OSD_DEBUG_BUILD
//#include "../osd/debug.h"
#include "../osd/glslTransformFeedbackComputeContext.h"
#include "../osd/opengl.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

// -----------------------------------------------------------------------------

template <class T> GLuint
createGLTextureBuffer(std::vector<T> const & src, GLenum type) {

    int size = (int)src.size()*sizeof(T);
    void const * ptr = &src.at(0);

    GLuint buffer;
    glGenBuffers(1, &buffer);

    GLuint devicePtr;
    glGenTextures(1, &devicePtr);

#if defined(GL_EXT_direct_state_access)
    if (glNamedBufferDataEXT and glTextureBufferEXT) {
        glNamedBufferDataEXT(buffer, size, ptr, GL_STATIC_DRAW);
        glTextureBufferEXT(devicePtr, GL_TEXTURE_BUFFER, type, buffer);
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
        glBindTexture(GL_TEXTURE_BUFFER, devicePtr);
        glTexBuffer(GL_TEXTURE_BUFFER, type, buffer);
        glBindTexture(GL_TEXTURE_BUFFER, prev);
    }

    glDeleteBuffers(1, &buffer);

    //OSD_DEBUG_CHECK_GL_ERROR("createGLTextureBuffer end\n");
    return devicePtr;
}

// -----------------------------------------------------------------------------

class GLSLTransformFeedbackComputeContext::GLStencilTables {

public:

    GLStencilTables(Far::StencilTables const & stencilTables) {
        _sizes = createGLTextureBuffer(stencilTables.GetSizes(), GL_R8UI);
        _offsets = createGLTextureBuffer(stencilTables.GetOffsets(), GL_R32I);
        _indices = createGLTextureBuffer(stencilTables.GetControlIndices(), GL_R32I);
        _weights = createGLTextureBuffer(stencilTables.GetWeights(), GL_R32F);
    }

    ~GLStencilTables() {
        glDeleteTextures(1, &_sizes);
        glDeleteTextures(1, &_offsets);
        glDeleteTextures(1, &_weights);
        glDeleteTextures(1, &_indices);
    }

    bool IsValid() const {
        return _sizes and _offsets and _indices and _weights;
    }

    GLuint GetSizes() const {
        return _sizes;
    }

    GLuint GetOffsets() const {
        return _offsets;
    }

    GLuint GetIndices() const {
        return _indices;
    }

    GLuint GetWeights() const {
        return _weights;
    }

private:

    GLuint _sizes,
           _offsets,
           _indices,
           _weights;
};

// -----------------------------------------------------------------------------

GLSLTransformFeedbackComputeContext::GLSLTransformFeedbackComputeContext(
    Far::StencilTables const * vertexStencilTables,
        Far::StencilTables const * varyingStencilTables) :
            _vertexStencilTables(0), _varyingStencilTables(0),
                _numControlVertices(0) {

    if (vertexStencilTables) {
        _vertexStencilTables = new GLStencilTables(*vertexStencilTables);
        _numControlVertices = vertexStencilTables->GetNumControlVertices();
    }

    if (varyingStencilTables) {
        _varyingStencilTables = new GLStencilTables(*varyingStencilTables);

        if (_numControlVertices) {
            assert(_numControlVertices==varyingStencilTables->GetNumControlVertices());
        } else {
            _numControlVertices = varyingStencilTables->GetNumControlVertices();
        }
    }
}

GLSLTransformFeedbackComputeContext::~GLSLTransformFeedbackComputeContext() {
    delete _vertexStencilTables;
    delete _varyingStencilTables;
}

// ----------------------------------------------------------------------------

bool
GLSLTransformFeedbackComputeContext::HasVertexStencilTables() const {
    return _vertexStencilTables ? _vertexStencilTables->IsValid() : false;
}

bool
GLSLTransformFeedbackComputeContext::HasVaryingStencilTables() const {
    return _varyingStencilTables ? _varyingStencilTables->IsValid() : false;
}

// ----------------------------------------------------------------------------

GLuint
GLSLTransformFeedbackComputeContext::GetVertexStencilTablesSizes() const {
    return _vertexStencilTables ? _vertexStencilTables->GetSizes() : 0;
}

GLuint
GLSLTransformFeedbackComputeContext::GetVertexStencilTablesOffsets() const {
    return _vertexStencilTables ? _vertexStencilTables->GetOffsets() : 0;
}

GLuint
GLSLTransformFeedbackComputeContext::GetVertexStencilTablesIndices() const {
    return _vertexStencilTables ? _vertexStencilTables->GetIndices() : 0;
}

GLuint
GLSLTransformFeedbackComputeContext::GetVertexStencilTablesWeights() const {
    return _vertexStencilTables ? _vertexStencilTables->GetWeights() : 0;
}

// ----------------------------------------------------------------------------

GLuint
GLSLTransformFeedbackComputeContext::GetVaryingStencilTablesSizes() const {
    return _varyingStencilTables ? _varyingStencilTables->GetSizes() : 0;
}

GLuint
GLSLTransformFeedbackComputeContext::GetVaryingStencilTablesOffsets() const {
    return _varyingStencilTables ? _varyingStencilTables->GetOffsets() : 0;
}

GLuint
GLSLTransformFeedbackComputeContext::GetVaryingStencilTablesIndices() const {
    return _varyingStencilTables ? _varyingStencilTables->GetIndices() : 0;
}

GLuint
GLSLTransformFeedbackComputeContext::GetVaryingStencilTablesWeights() const {
    return _varyingStencilTables ? _varyingStencilTables->GetWeights() : 0;
}


// -----------------------------------------------------------------------------

GLSLTransformFeedbackComputeContext *
GLSLTransformFeedbackComputeContext::Create(
    Far::StencilTables const * vertexStencilTables,
        Far::StencilTables const * varyingStencilTables) {

    GLSLTransformFeedbackComputeContext *result =
        new GLSLTransformFeedbackComputeContext(
            vertexStencilTables, varyingStencilTables);

    return result;
}


// -----------------------------------------------------------------------------

}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
