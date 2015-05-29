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
#include "../far/stencilTable.h"

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

GLStencilTableSSBO::GLStencilTableSSBO(
    Far::StencilTable const *stencilTable) {
    _numStencils = stencilTable->GetNumStencils();
    if (_numStencils > 0) {
        _sizes   = createSSBO(stencilTable->GetSizes());
        _offsets = createSSBO(stencilTable->GetOffsets());
        _indices = createSSBO(stencilTable->GetControlIndices());
        _weights = createSSBO(stencilTable->GetWeights());
        _duWeights = _dvWeights = 0;
    } else {
        _sizes = _offsets = _indices = _weights = 0;
        _duWeights = _dvWeights = 0;
    }
}

GLStencilTableSSBO::GLStencilTableSSBO(
    Far::LimitStencilTable const *limitStencilTable) {
    _numStencils = limitStencilTable->GetNumStencils();
    if (_numStencils > 0) {
        _sizes   = createSSBO(limitStencilTable->GetSizes());
        _offsets = createSSBO(limitStencilTable->GetOffsets());
        _indices = createSSBO(limitStencilTable->GetControlIndices());
        _weights = createSSBO(limitStencilTable->GetWeights());
        _duWeights = createSSBO(limitStencilTable->GetDuWeights());
        _dvWeights = createSSBO(limitStencilTable->GetDvWeights());
    } else {
        _sizes = _offsets = _indices = _weights = 0;
        _duWeights = _dvWeights = 0;
    }
}

GLStencilTableSSBO::~GLStencilTableSSBO() {
    if (_sizes)   glDeleteBuffers(1, &_sizes);
    if (_offsets) glDeleteBuffers(1, &_offsets);
    if (_indices) glDeleteBuffers(1, &_indices);
    if (_weights) glDeleteBuffers(1, &_weights);
    if (_duWeights) glDeleteBuffers(1, &_duWeights);
    if (_dvWeights) glDeleteBuffers(1, &_dvWeights);
}

// ---------------------------------------------------------------------------


GLComputeEvaluator::GLComputeEvaluator() : _workGroupSize(64) {
    memset (&_stencilKernel, 0, sizeof(_stencilKernel));
    memset (&_patchKernel, 0, sizeof(_patchKernel));
}

GLComputeEvaluator::~GLComputeEvaluator() {
    if (_stencilKernel.program) {
        glDeleteProgram(_stencilKernel.program);
    }
    if (_patchKernel.program) {
        glDeleteProgram(_patchKernel.program);
    }
}

static GLuint
compileKernel(BufferDescriptor const &srcDesc,
              BufferDescriptor const &dstDesc,
              BufferDescriptor const & /* duDesc */,
              BufferDescriptor const & /* dvDesc */,
              const char *kernelDefine,
              int workGroupSize) {
    GLuint program = glCreateProgram();

    GLuint shader = glCreateShader(GL_COMPUTE_SHADER);

    std::ostringstream defines;
    defines << "#define LENGTH "     << srcDesc.length << "\n"
            << "#define SRC_STRIDE " << srcDesc.stride << "\n"
            << "#define DST_STRIDE " << dstDesc.stride << "\n"
            << "#define WORK_GROUP_SIZE " << workGroupSize << "\n"
            << kernelDefine << "\n";
    std::string defineStr = defines.str();

    const char *shaderSources[3] = {"#version 430\n", 0, 0};
    shaderSources[1] = defineStr.c_str();
    shaderSources[2] = shaderSource;
    glShaderSource(shader, 3, shaderSources, NULL);
    glCompileShader(shader);
    glAttachShader(program, shader);

    GLint linked = 0;
    glLinkProgram(program);
    glGetProgramiv(program, GL_LINK_STATUS, &linked);

    if (linked == GL_FALSE) {
        char buffer[1024];
        glGetShaderInfoLog(shader, 1024, NULL, buffer);
        Far::Error(Far::FAR_RUNTIME_ERROR, buffer);

        glGetProgramInfoLog(program, 1024, NULL, buffer);
        Far::Error(Far::FAR_RUNTIME_ERROR, buffer);

        glDeleteProgram(program);
        return 0;
    }

    glDeleteShader(shader);

    return program;
}

bool
GLComputeEvaluator::Compile(BufferDescriptor const &srcDesc,
                            BufferDescriptor const &dstDesc,
                            BufferDescriptor const &duDesc,
                            BufferDescriptor const &dvDesc) {

    // create a stencil kernel
    if (!_stencilKernel.Compile(srcDesc, dstDesc, duDesc, dvDesc,
                                _workGroupSize)) {
        return false;
    }

    // create a patch kernel
    if (!_patchKernel.Compile(srcDesc, dstDesc, duDesc, dvDesc,
                              _workGroupSize)) {
        return false;
    }

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
GLComputeEvaluator::EvalStencils(
    GLuint srcBuffer, BufferDescriptor const &srcDesc,
    GLuint dstBuffer, BufferDescriptor const &dstDesc,
    GLuint duBuffer,  BufferDescriptor const &duDesc,
    GLuint dvBuffer,  BufferDescriptor const &dvDesc,
    GLuint sizesBuffer,
    GLuint offsetsBuffer,
    GLuint indicesBuffer,
    GLuint weightsBuffer,
    GLuint duWeightsBuffer,
    GLuint dvWeightsBuffer,
    int start, int end) const {

    if (!_stencilKernel.program) return false;
    int count = end - start;
    if (count <= 0) {
        return true;
    }

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, srcBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, dstBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, duBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, dvBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, sizesBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, offsetsBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, indicesBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, weightsBuffer);
    if (duWeightsBuffer)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, duWeightsBuffer);
    if (dvWeightsBuffer)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 9, dvWeightsBuffer);

    glUseProgram(_stencilKernel.program);

    glUniform1i(_stencilKernel.uniformStart,     start);
    glUniform1i(_stencilKernel.uniformEnd,       end);
    glUniform1i(_stencilKernel.uniformSrcOffset, srcDesc.offset);
    glUniform1i(_stencilKernel.uniformDstOffset, dstDesc.offset);
    if (_stencilKernel.uniformDuDesc > 0) {
        glUniform3i(_stencilKernel.uniformDuDesc,
                    duDesc.offset, duDesc.length, duDesc.stride);
    }
    if (_stencilKernel.uniformDvDesc > 0) {
        glUniform3i(_stencilKernel.uniformDvDesc,
                    dvDesc.offset, dvDesc.length, dvDesc.stride);
    }

    glDispatchCompute((count + _workGroupSize - 1) / _workGroupSize, 1, 1);

    glUseProgram(0);

    glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
    for (int i = 0; i < 10; ++i) {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, 0);
    }

    return true;
}

bool
GLComputeEvaluator::EvalPatches(
    GLuint srcBuffer, BufferDescriptor const &srcDesc,
    GLuint dstBuffer, BufferDescriptor const &dstDesc,
    GLuint duBuffer,  BufferDescriptor const &duDesc,
    GLuint dvBuffer,  BufferDescriptor const &dvDesc,
    int numPatchCoords,
    GLuint patchCoordsBuffer,
    const PatchArrayVector &patchArrays,
    GLuint patchIndexBuffer,
    GLuint patchParamsBuffer) const {

    if (!_patchKernel.program) return false;

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, srcBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, dstBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, duBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, dvBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, patchCoordsBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, patchIndexBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, patchParamsBuffer);

    glUseProgram(_patchKernel.program);

    glUniform1i(_patchKernel.uniformSrcOffset, srcDesc.offset);
    glUniform1i(_patchKernel.uniformDstOffset, dstDesc.offset);
    glUniform4iv(_patchKernel.uniformPatchArray, (int)patchArrays.size(),
                 (const GLint*)&patchArrays[0]);
    glUniform3i(_patchKernel.uniformDuDesc, duDesc.offset, duDesc.length, duDesc.stride);
    glUniform3i(_patchKernel.uniformDvDesc, dvDesc.offset, dvDesc.length, dvDesc.stride);

    glDispatchCompute((numPatchCoords + _workGroupSize - 1) / _workGroupSize, 1, 1);

    glUseProgram(0);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, 0);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, 0);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, 0);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, 0);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, 0);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, 0);

    return true;
}
// ---------------------------------------------------------------------------

GLComputeEvaluator::_StencilKernel::_StencilKernel() : program(0) {
}
GLComputeEvaluator::_StencilKernel::~_StencilKernel() {
    if (program) {
        glDeleteProgram(program);
    }
}

bool
GLComputeEvaluator::_StencilKernel::Compile(BufferDescriptor const &srcDesc,
                                            BufferDescriptor const &dstDesc,
                                            BufferDescriptor const &duDesc,
                                            BufferDescriptor const &dvDesc,
                                            int workGroupSize) {
    // create stencil kernel
    if (program) {
        glDeleteProgram(program);
    }

    bool derivatives = (duDesc.length > 0 || dvDesc.length > 0);
    const char *kernelDef = derivatives
        ? "#define OPENSUBDIV_GLSL_COMPUTE_KERNEL_EVAL_STENCILS\n"
          "#define OPENSUBDIV_GLSL_COMPUTE_USE_DERIVATIVES\n"
        : "#define OPENSUBDIV_GLSL_COMPUTE_KERNEL_EVAL_STENCILS\n";

    if (program) {
        glDeleteProgram(program);
    }
    program = compileKernel(srcDesc, dstDesc, duDesc, dvDesc, kernelDef,
                            workGroupSize);
    if (program == 0) return false;

    // cache uniform locations (TODO: use uniform block)
    uniformStart     = glGetUniformLocation(program, "batchStart");
    uniformEnd       = glGetUniformLocation(program, "batchEnd");
    uniformSrcOffset = glGetUniformLocation(program, "srcOffset");
    uniformDstOffset = glGetUniformLocation(program, "dstOffset");
    uniformDuDesc    = glGetUniformLocation(program, "duDesc");
    uniformDvDesc    = glGetUniformLocation(program, "dvDesc");

    return true;
}

// ---------------------------------------------------------------------------

GLComputeEvaluator::_PatchKernel::_PatchKernel() : program(0) {
}
GLComputeEvaluator::_PatchKernel::~_PatchKernel() {
    if (program) {
        glDeleteProgram(program);
    }
}

bool
GLComputeEvaluator::_PatchKernel::Compile(BufferDescriptor const &srcDesc,
                                          BufferDescriptor const &dstDesc,
                                          BufferDescriptor const &duDesc,
                                          BufferDescriptor const &dvDesc,
                                          int workGroupSize) {
    // create stencil kernel
    if (program) {
        glDeleteProgram(program);
    }

    bool derivatives = (duDesc.length > 0 || dvDesc.length > 0);
    const char *kernelDef = derivatives
        ? "#define OPENSUBDIV_GLSL_COMPUTE_KERNEL_EVAL_PATCHES\n"
          "#define OPENSUBDIV_GLSL_COMPUTE_USE_DERIVATIVES\n"
        : "#define OPENSUBDIV_GLSL_COMPUTE_KERNEL_EVAL_PATCHES\n";

    if (program) {
        glDeleteProgram(program);
    }
    program = compileKernel(srcDesc, dstDesc, duDesc, dvDesc, kernelDef,
                            workGroupSize);
    if (program == 0) return false;

    // cache uniform locations
    uniformSrcOffset  = glGetUniformLocation(program, "srcOffset");
    uniformDstOffset  = glGetUniformLocation(program, "dstOffset");
    uniformPatchArray = glGetUniformLocation(program, "patchArray");
    uniformDuDesc     = glGetUniformLocation(program, "duDesc");
    uniformDvDesc     = glGetUniformLocation(program, "dvDesc");

    return true;
}

}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
