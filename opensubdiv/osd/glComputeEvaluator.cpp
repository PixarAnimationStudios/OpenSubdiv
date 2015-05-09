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

#include "../osd/glComputeEvaluator.h"

#include <cassert>
#include <sstream>
#include <string>
#include <vector>

#include "../far/error.h"
#include "../far/stencilTables.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

static const char *shaderSource =
#include "../osd/glslComputeKernel.gen.h"
;

template <class T> GLuint
createSSBO(std::vector<T> const & src) {
    GLuint devicePtr = 0;
    glGenBuffers(1, &devicePtr);

#if defined(GL_EXT_direct_state_access)
    if (glNamedBufferDataEXT) {
        glNamedBufferDataEXT(devicePtr, src.size()*sizeof(T),
                             &src.at(0), GL_STATIC_DRAW);
    } else {
#else
    {
#endif
        GLint prev = 0;
        glGetIntegerv(GL_SHADER_STORAGE_BUFFER_BINDING, &prev);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, devicePtr);
        glBufferData(GL_SHADER_STORAGE_BUFFER, src.size()*sizeof(T),
                     &src.at(0), GL_STATIC_DRAW);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, prev);
    }

    return devicePtr;
}

GLStencilTablesSSBO::GLStencilTablesSSBO(
    Far::StencilTables const *stencilTables) {
    _numStencils = stencilTables->GetNumStencils();
    if (_numStencils > 0) {
        _sizes   = createSSBO(stencilTables->GetSizes());
        _offsets = createSSBO(stencilTables->GetOffsets());
        _indices = createSSBO(stencilTables->GetControlIndices());
        _weights = createSSBO(stencilTables->GetWeights());
    } else {
        _sizes = _offsets = _indices = _weights = 0;
    }
}

GLStencilTablesSSBO::~GLStencilTablesSSBO() {
    if (_sizes)   glDeleteBuffers(1, &_sizes);
    if (_offsets) glDeleteBuffers(1, &_offsets);
    if (_weights) glDeleteBuffers(1, &_weights);
    if (_indices) glDeleteBuffers(1, &_indices);
}

// ---------------------------------------------------------------------------


GLComputeEvaluator::GLComputeEvaluator() :
    _program(0), _workGroupSize(64) {
}

GLComputeEvaluator::~GLComputeEvaluator() {
    if (_program) {
        glDeleteProgram(_program);
    }
}

bool
GLComputeEvaluator::Compile(VertexBufferDescriptor const &srcDesc,
                            VertexBufferDescriptor const &dstDesc) {
    if (srcDesc.length > dstDesc.length) {
        Far::Error(Far::FAR_RUNTIME_ERROR,
                   "srcDesc length must be less than or equal to "
                   "dstDesc length.\n");
        return false;
    }

    if (_program) {
        glDeleteProgram(_program);
        _program = 0;
    }
    _program = glCreateProgram();

    GLuint shader = glCreateShader(GL_COMPUTE_SHADER);

    std::ostringstream defines;
    defines << "#define LENGTH "     << srcDesc.length << "\n"
            << "#define SRC_STRIDE " << srcDesc.stride << "\n"
            << "#define DST_STRIDE " << dstDesc.stride << "\n"
            << "#define WORK_GROUP_SIZE " << _workGroupSize << "\n";
    std::string defineStr = defines.str();

    const char *shaderSources[3] = {"#version 430\n", 0, 0};
    shaderSources[1] = defineStr.c_str();
    shaderSources[2] = shaderSource;
    glShaderSource(shader, 3, shaderSources, NULL);
    glCompileShader(shader);
    glAttachShader(_program, shader);

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

    // store uniform locations for the compute kernel program.
    _uniformSizes   = glGetUniformLocation(_program, "stencilSizes");
    _uniformOffsets = glGetUniformLocation(_program, "stencilOffsets");
    _uniformIndices = glGetUniformLocation(_program, "stencilIndices");
    _uniformWeights = glGetUniformLocation(_program, "stencilIWeights");

    _uniformStart   = glGetUniformLocation(_program, "batchStart");
    _uniformEnd     = glGetUniformLocation(_program, "batchEnd");

    _uniformSrcOffset = glGetUniformLocation(_program, "srcOffset");
    _uniformDstOffset = glGetUniformLocation(_program, "dstOffset");

    return true;
}

/* static */
void
GLComputeEvaluator::Synchronize(void * /*kernel*/) {
    // XXX: this is currently just for the performance measuring purpose.
    // need to be reimplemented by fence and sync.
    glFinish();
}

bool
GLComputeEvaluator::EvalStencils(GLuint srcBuffer,
                                 VertexBufferDescriptor const &srcDesc,
                                 GLuint dstBuffer,
                                 VertexBufferDescriptor const &dstDesc,
                                 GLuint sizesBuffer,
                                 GLuint offsetsBuffer,
                                 GLuint indicesBuffer,
                                 GLuint weightsBuffer,
                                 int start,
                                 int end) const {
    if (!_program) return false;
    int count = end - start;
    if (count <= 0) {
        return true;
    }

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, srcBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, dstBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, sizesBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, offsetsBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, indicesBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, weightsBuffer);

    glUseProgram(_program);

    glUniform1i(_uniformStart,     start);
    glUniform1i(_uniformEnd,       end);
    glUniform1i(_uniformSrcOffset, srcDesc.offset);
    glUniform1i(_uniformDstOffset, dstDesc.offset);

    glDispatchCompute((count + _workGroupSize - 1) / _workGroupSize, 1, 1);

    glUseProgram(0);

    glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, 0);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, 0);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, 0);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, 0);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, 0);

    return true;
}

}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
