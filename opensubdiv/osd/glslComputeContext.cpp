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
//#include "../osd/debug.h"
#include "../osd/glslComputeContext.h"
#include "../osd/opengl.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

// -----------------------------------------------------------------------------

template <class T> GLuint
createGLSLBuffer(std::vector<T> const & src) {

    GLuint devicePtr=0;

    glGenBuffers(1, &devicePtr);

#if defined(GL_EXT_direct_state_access)
    if (glNamedBufferDataEXT) {
        glNamedBufferDataEXT(devicePtr, src.size()*sizeof(T), &src.at(0), GL_STATIC_DRAW);
    } else {
#else
    {
#endif
        GLint prev = 0;
        glGetIntegerv(GL_SHADER_STORAGE_BUFFER_BINDING, &prev);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, devicePtr);
        glBufferData(GL_SHADER_STORAGE_BUFFER, src.size()*sizeof(T), &src.at(0), GL_STATIC_DRAW);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, prev);
    }

    //OSD_DEBUG_CHECK_GL_ERROR("createGLSLBuffer size %ld", src.size());
    return devicePtr;
}

// -----------------------------------------------------------------------------

class GLSLComputeContext::GLSLStencilTables {

public:

    GLSLStencilTables(Far::StencilTables const & stencilTables) {
        _sizes = createGLSLBuffer(stencilTables.GetSizes());
        _offsets = createGLSLBuffer(stencilTables.GetOffsets());
        _indices = createGLSLBuffer(stencilTables.GetControlIndices());
        _weights = createGLSLBuffer(stencilTables.GetWeights());
    }

    ~GLSLStencilTables() {
        glDeleteBuffers(1, &_sizes);
        glDeleteBuffers(1, &_offsets);
        glDeleteBuffers(1, &_weights);
        glDeleteBuffers(1, &_indices);
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

    void Bind() const {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _sizes);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _offsets);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _indices);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _weights);
    }

    static void Unbind() {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, 0);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, 0);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, 0);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, 0);

        glUseProgram(0);
    }

private:

    GLuint _sizes,
           _offsets,
           _indices,
           _weights;
};

// -----------------------------------------------------------------------------

GLSLComputeContext::GLSLComputeContext(
    Far::StencilTables const * vertexStencilTables,
        Far::StencilTables const * varyingStencilTables) :
            _vertexStencilTables(0), _varyingStencilTables(0),
                _numControlVertices(0) {

    if (vertexStencilTables) {
        _vertexStencilTables = new GLSLStencilTables(*vertexStencilTables);
        _numControlVertices = vertexStencilTables->GetNumControlVertices();
    }

    if (varyingStencilTables) {
        _varyingStencilTables = new GLSLStencilTables(*varyingStencilTables);

        if (_numControlVertices) {
            assert(_numControlVertices==varyingStencilTables->GetNumControlVertices());
        } else {
            _numControlVertices = varyingStencilTables->GetNumControlVertices();
        }
    }
}

GLSLComputeContext::~GLSLComputeContext() {
    delete _vertexStencilTables;
    delete _varyingStencilTables;
}

// ----------------------------------------------------------------------------

bool
GLSLComputeContext::HasVertexStencilTables() const {
    return _vertexStencilTables ? _vertexStencilTables->IsValid() : false;
}

bool
GLSLComputeContext::HasVaryingStencilTables() const {
    return _varyingStencilTables ? _varyingStencilTables->IsValid() : false;
}

// ----------------------------------------------------------------------------


void
GLSLComputeContext::BindVertexStencilTables() const {
    if (_vertexStencilTables) {
        _vertexStencilTables->Bind();
    }
}

void
GLSLComputeContext::BindVaryingStencilTables() const {
    if (_varyingStencilTables) {
        _varyingStencilTables->Bind();
    }
}

void
GLSLComputeContext::UnbindStencilTables() const {
    GLSLStencilTables::Unbind();
}


// -----------------------------------------------------------------------------

GLSLComputeContext *
GLSLComputeContext::Create(Far::StencilTables const * vertexStencilTables,
                              Far::StencilTables const * varyingStencilTables) {

    GLSLComputeContext *result =
        new GLSLComputeContext(vertexStencilTables, varyingStencilTables);

    return result;
}

// -----------------------------------------------------------------------------

}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
