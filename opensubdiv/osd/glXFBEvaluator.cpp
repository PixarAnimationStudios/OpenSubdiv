//
//   Copyright 2015 Pixar
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

#include "../osd/glXFBEvaluator.h"

#include <sstream>
#include <string>
#include <vector>
#include <cstdio>

#include "../far/error.h"
#include "../far/stencilTable.h"

#if _MSC_VER
    #define snprintf _snprintf
#endif

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

static const char *shaderSource =
#include "../osd/glslXFBKernel.gen.h"
;

template <class T> GLuint
createGLTextureBuffer(std::vector<T> const & src, GLenum type) {
    GLint size = static_cast<int>(src.size()*sizeof(T));
    void const * ptr = &src.at(0);

    GLuint buffer;
    glGenBuffers(1, &buffer);

    GLuint devicePtr;
    glGenTextures(1, &devicePtr);

#if defined(GL_EXT_direct_state_access)
    if (glNamedBufferDataEXT && glTextureBufferEXT) {
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

    return devicePtr;
}

GLStencilTableTBO::GLStencilTableTBO(
    Far::StencilTable const *stencilTable) {

    _numStencils = stencilTable->GetNumStencils();
    if (_numStencils > 0) {
        _sizes   = createGLTextureBuffer(stencilTable->GetSizes(), GL_R32UI);
        _offsets = createGLTextureBuffer(
            stencilTable->GetOffsets(), GL_R32I);
        _indices = createGLTextureBuffer(
            stencilTable->GetControlIndices(), GL_R32I);
        _weights = createGLTextureBuffer(stencilTable->GetWeights(), GL_R32F);
    } else {
        _sizes = _offsets = _indices = _weights = 0;
    }
}

GLStencilTableTBO::~GLStencilTableTBO() {
    if (_sizes) glDeleteTextures(1, &_sizes);
    if (_offsets) glDeleteTextures(1, &_offsets);
    if (_weights) glDeleteTextures(1, &_weights);
    if (_indices) glDeleteTextures(1, &_indices);
}

// ---------------------------------------------------------------------------


GLXFBEvaluator::GLXFBEvaluator() :
    _program(0), _srcBufferTexture(0),
    _uniformSrcBufferTexture(0), _uniformSizesTexture(0),
    _uniformOffsetsTexture(0), _uniformIndicesTexture(0),
    _uniformWeightsTexture(0), _uniformStart(0), _uniformEnd(0),
    _uniformSrcOffset(0) {
}

GLXFBEvaluator::~GLXFBEvaluator() {
    if (_program) {
        glDeleteProgram(_program);
    }
    if (_srcBufferTexture) {
        glDeleteTextures(1, &_srcBufferTexture);
    }
}

bool
GLXFBEvaluator::Compile(VertexBufferDescriptor const &srcDesc,
                        VertexBufferDescriptor const &dstDesc) {
    if (_program) {
        glDeleteProgram(_program);
        _program = 0;
    }
    _program = glCreateProgram();

    GLuint shader = glCreateShader(GL_VERTEX_SHADER);

    std::ostringstream defines;
    defines << "#define LENGTH " << srcDesc.length << "\n"
            << "#define SRC_STRIDE " << srcDesc.stride << "\n";
    std::string defineStr = defines.str();

    const char *shaderSources[3] = {"#version 410\n", NULL, NULL};

    shaderSources[1] = defineStr.c_str();
    shaderSources[2] = shaderSource;
    glShaderSource(shader, 3, shaderSources, NULL);
    glCompileShader(shader);
    glAttachShader(_program, shader);

    std::vector<std::string> outputs;
    std::vector<const char *> pOutputs;
    {
        // vertex data (may include custom vertex data) and varying data
        // are stored into the same buffer, interleaved.
        //
        // (gl_SkipComponents1)
        // outVertexData[0]
        // outVertexData[1]
        // outVertexData[2]
        // (gl_SkipComponents1)
        //
        // note that "primvarOffset" in shader is still needed to read
        // interleaved components even if gl_SkipComponents is used.
        //
        char attrName[32];
        int primvarOffset = (dstDesc.offset % dstDesc.stride);
        for (int i = 0; i < primvarOffset; ++i) {
            outputs.push_back("gl_SkipComponents1");
        }
        for (int i = 0; i < dstDesc.length; ++i) {
            snprintf(attrName, sizeof(attrName), "outVertexBuffer[%d]", i);
            outputs.push_back(attrName);
        }
        for (int i = primvarOffset + dstDesc.length; i < dstDesc.stride; ++i) {
            outputs.push_back("gl_SkipComponents1");
        }

        // convert to char* array
        for (size_t i = 0; i < outputs.size(); ++i) {
            pOutputs.push_back(&outputs[i][0]);
        }
    }

    glTransformFeedbackVaryings(_program, (GLsizei)outputs.size(),
                                &pOutputs[0], GL_INTERLEAVED_ATTRIBS);

    GLint linked = 0;
    glLinkProgram(_program);
    glGetProgramiv(_program, GL_LINK_STATUS, &linked);

    if (linked == GL_FALSE) {
        char buffer[1024];
        glGetShaderInfoLog(shader, 1024, NULL, buffer);
        Far::Error(Far::FAR_RUNTIME_ERROR, buffer);

        glGetProgramInfoLog(_program, 1024, NULL, buffer);
        Far::Error(Far::FAR_RUNTIME_ERROR, buffer);

        glDeleteProgram(_program);
        _program = 0;
        return false;
    }

    glDeleteShader(shader);

    // set uniform locations for compute kernels
    _uniformSrcBufferTexture  = glGetUniformLocation(_program, "vertexBuffer");

    _uniformSizesTexture   = glGetUniformLocation(_program, "sizes");
    _uniformOffsetsTexture = glGetUniformLocation(_program, "offsets");
    _uniformIndicesTexture = glGetUniformLocation(_program, "indices");
    _uniformWeightsTexture = glGetUniformLocation(_program, "weights");

    _uniformStart   = glGetUniformLocation(_program, "batchStart");
    _uniformEnd     = glGetUniformLocation(_program, "batchEnd");

    _uniformSrcOffset = glGetUniformLocation(_program, "srcOffset");

    // create a texture for input buffer
    if (!_srcBufferTexture) {
        glGenTextures(1, &_srcBufferTexture);
    }
    return true;
}

/* static */
void
GLXFBEvaluator::Synchronize(void * /*kernel*/) {
    // XXX: this is currently just for the test purpose.
    // need to be reimplemented by fence and sync.
    glFinish();
}

static void
bindTexture(GLint sampler, GLuint texture, int unit) {
    if (sampler == -1) {
        return;
    }
    glUniform1i(sampler, unit);
    glActiveTexture(GL_TEXTURE0 + unit);
    glBindTexture(GL_TEXTURE_BUFFER, texture);
    glActiveTexture(GL_TEXTURE0);
}

bool
GLXFBEvaluator::EvalStencils(GLuint srcBuffer,
                             VertexBufferDescriptor const &srcDesc,
                             GLuint dstBuffer,
                             VertexBufferDescriptor const &dstDesc,
                             GLuint sizesTexture,
                             GLuint offsetsTexture,
                             GLuint indicesTexture,
                             GLuint weightsTexture,
                             int start,
                             int end) const {
    if (!_program) return false;
    int count = end - start;
    if (count <= 0) {
        return true;
    }

    // bind vertex array
    // always create new one, to be safe with multiple contexts (slow though)
    GLuint vao = 0;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glEnable(GL_RASTERIZER_DISCARD);
    glUseProgram(_program);

    // Set input VBO as a texture buffer.
    glBindTexture(GL_TEXTURE_BUFFER, _srcBufferTexture);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_R32F, srcBuffer);
    glBindTexture(GL_TEXTURE_BUFFER, 0);

    bindTexture(_uniformSrcBufferTexture, _srcBufferTexture, 0);

    // bind stencil table textures.
    bindTexture(_uniformSizesTexture,   sizesTexture, 1);
    bindTexture(_uniformOffsetsTexture, offsetsTexture, 2);
    bindTexture(_uniformIndicesTexture, indicesTexture, 3);
    bindTexture(_uniformWeightsTexture, weightsTexture, 4);

    // set batch range
    glUniform1i(_uniformStart,     start);
    glUniform1i(_uniformEnd,       end);
    glUniform1i(_uniformSrcOffset, srcDesc.offset);

    // The destination buffer is bound at vertex boundary.
    //
    // Example: When we have a batched and interleaved vertex buffer
    //
    //  Obj  X    |    Obj Y                                  |
    // -----------+-------------------------------------------+-------
    //            |    vtx 0      |    vtx 1      |           |
    // -----------+---------------+---------------+-----------+-------
    //            | x y z r g b a | x y z r g b a | ....      |
    // -----------+---------------+---------------+-----------+-------
    //                    ^
    //                    srcDesc.offset for Obj Y color
    //
    //            ^-------------------------------------------^
    //                    XFB destination buffer range
    //              S S S * * * *
    //              k k k
    //              i i i
    //              p p p
    //
    //  We use gl_SkipComponents to skip the first 3 XYZ so the
    //  buffer itself needs to be bound for entire section of ObjY.
    //
    //  Note that for the source buffer (texture) we bind the whole
    //  buffer (all VBO range) and use srcOffset=srcDesc.offset for
    //  indexing.
    //
    int dstBufferBindOffset =
        dstDesc.offset - (dstDesc.offset % dstDesc.stride);

    // bind destination buffer
    glBindBufferRange(GL_TRANSFORM_FEEDBACK_BUFFER,
                      0, dstBuffer,
                      dstBufferBindOffset * sizeof(float),
                      count * dstDesc.stride * sizeof(float));

    glBeginTransformFeedback(GL_POINTS);
    glDrawArrays(GL_POINTS, 0, count);
    glEndTransformFeedback();

    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, 0);

    for (int i = 0; i < 5; ++i) {
        glActiveTexture(GL_TEXTURE0 + i);
        glBindTexture(GL_TEXTURE_BUFFER, 0);
    }

    glDisable(GL_RASTERIZER_DISCARD);
    glUseProgram(0);
    glActiveTexture(GL_TEXTURE0);

    // revert vao
    glBindVertexArray(0);
    glDeleteVertexArrays(1, &vao);


    return true;
}

}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
