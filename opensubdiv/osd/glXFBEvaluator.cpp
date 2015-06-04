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
        _duWeights = _dvWeights = 0;
    } else {
        _sizes = _offsets = _indices = _weights = 0;
        _duWeights = _dvWeights = 0;
    }
}

GLStencilTableTBO::GLStencilTableTBO(
    Far::LimitStencilTable const *limitStencilTable) {

    _numStencils = limitStencilTable->GetNumStencils();
    if (_numStencils > 0) {
        _sizes   = createGLTextureBuffer(
            limitStencilTable->GetSizes(), GL_R32UI);
        _offsets = createGLTextureBuffer(
            limitStencilTable->GetOffsets(), GL_R32I);
        _indices = createGLTextureBuffer(
            limitStencilTable->GetControlIndices(), GL_R32I);
        _weights = createGLTextureBuffer(
            limitStencilTable->GetWeights(), GL_R32F);
        _duWeights = createGLTextureBuffer(
            limitStencilTable->GetDuWeights(), GL_R32F);
        _dvWeights = createGLTextureBuffer(
            limitStencilTable->GetDvWeights(), GL_R32F);
    } else {
        _sizes = _offsets = _indices = _weights = 0;
        _duWeights = _dvWeights = 0;
    }
}

GLStencilTableTBO::~GLStencilTableTBO() {
    if (_sizes) glDeleteTextures(1, &_sizes);
    if (_offsets) glDeleteTextures(1, &_offsets);
    if (_indices) glDeleteTextures(1, &_indices);
    if (_weights) glDeleteTextures(1, &_weights);
    if (_duWeights) glDeleteTextures(1, &_duWeights);
    if (_dvWeights) glDeleteTextures(1, &_dvWeights);
}

// ---------------------------------------------------------------------------

GLXFBEvaluator::GLXFBEvaluator() : _srcBufferTexture(0) {
}

GLXFBEvaluator::~GLXFBEvaluator() {
    if (_srcBufferTexture) {
        glDeleteTextures(1, &_srcBufferTexture);
    }
}

static GLuint
compileKernel(BufferDescriptor const &srcDesc,
              BufferDescriptor const &dstDesc,
              BufferDescriptor const &duDesc,
              BufferDescriptor const &dvDesc,
              const char *kernelDefine) {

    GLuint program = glCreateProgram();

    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);

    std::ostringstream defines;
    defines << "#define LENGTH " << srcDesc.length << "\n"
            << "#define SRC_STRIDE " << srcDesc.stride << "\n"
            << "#define VERTEX_SHADER\n"
            << kernelDefine << "\n";
    std::string defineStr = defines.str();

    const char *shaderSources[3] = {"#version 410\n", NULL, NULL};

    shaderSources[1] = defineStr.c_str();
    shaderSources[2] = shaderSource;
    glShaderSource(vertexShader, 3, shaderSources, NULL);
    glCompileShader(vertexShader);
    glAttachShader(program, vertexShader);

    std::vector<std::string> outputs;
    char attrName[32];
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
    }
    if (duDesc.length) {
        //
        // For derivatives, we use another buffer bindings so gl_NextBuffer
        // is inserted here to switch the destination of transform feedback.
        //
        // Note that the destination buffers may or may not be shared between
        // vertex and each derivatives. gl_NextBuffer seems still works well
        // in either case.
        //
        outputs.push_back("gl_NextBuffer");
        int primvarOffset = (duDesc.offset % duDesc.stride);
        for (int i = 0; i < primvarOffset; ++i) {
            outputs.push_back("gl_SkipComponents1");
        }
        for (int i = 0; i < duDesc.length; ++i) {
            snprintf(attrName, sizeof(attrName), "outDuBuffer[%d]", i);
            outputs.push_back(attrName);
        }
        for (int i = primvarOffset + duDesc.length; i < duDesc.stride; ++i) {
            outputs.push_back("gl_SkipComponents1");
        }
    }
    if (dvDesc.length) {
        outputs.push_back("gl_NextBuffer");
        int primvarOffset = (dvDesc.offset % dvDesc.stride);
        for (int i = 0; i < primvarOffset; ++i) {
            outputs.push_back("gl_SkipComponents1");
        }
        for (int i = 0; i < dvDesc.length; ++i) {
            snprintf(attrName, sizeof(attrName), "outDvBuffer[%d]", i);
            outputs.push_back(attrName);
        }
        for (int i = primvarOffset + dvDesc.length; i < dvDesc.stride; ++i) {
            outputs.push_back("gl_SkipComponents1");
        }
    }
    // convert to char* array
    std::vector<const char *> pOutputs;
    for (size_t i = 0; i < outputs.size(); ++i) {
        pOutputs.push_back(&outputs[i][0]);
    }

    glTransformFeedbackVaryings(program, (GLsizei)outputs.size(),
                                &pOutputs[0], GL_INTERLEAVED_ATTRIBS);

    GLint linked = 0;
    glLinkProgram(program);
    glGetProgramiv(program, GL_LINK_STATUS, &linked);

    if (linked == GL_FALSE) {
        char buffer[1024];
        glGetShaderInfoLog(vertexShader, 1024, NULL, buffer);
        Far::Error(Far::FAR_RUNTIME_ERROR, buffer);

        glGetProgramInfoLog(program, 1024, NULL, buffer);
        Far::Error(Far::FAR_RUNTIME_ERROR, buffer);

        glDeleteProgram(program);
        program = 0;
    }

    glDeleteShader(vertexShader);

    return program;
}

bool
GLXFBEvaluator::Compile(BufferDescriptor const &srcDesc,
                        BufferDescriptor const &dstDesc,
                        BufferDescriptor const &duDesc,
                        BufferDescriptor const &dvDesc) {

    // create a stencil kernel
    _stencilKernel.Compile(srcDesc, dstDesc, duDesc, dvDesc);

    // create a patch kernel
    _patchKernel.Compile(srcDesc, dstDesc, duDesc, dvDesc);

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
GLXFBEvaluator::EvalStencils(
    GLuint srcBuffer, BufferDescriptor const &srcDesc,
    GLuint dstBuffer, BufferDescriptor const &dstDesc,
    GLuint duBuffer,  BufferDescriptor const &duDesc,
    GLuint dvBuffer,  BufferDescriptor const &dvDesc,
    GLuint sizesTexture,
    GLuint offsetsTexture,
    GLuint indicesTexture,
    GLuint weightsTexture,
    GLuint duWeightsTexture,
    GLuint dvWeightsTexture,
    int start, int end) const {

    if (!_stencilKernel.program) return false;
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
    glUseProgram(_stencilKernel.program);

    // Set input VBO as a texture buffer.
    glBindTexture(GL_TEXTURE_BUFFER, _srcBufferTexture);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_R32F, srcBuffer);
    glBindTexture(GL_TEXTURE_BUFFER, 0);

    bindTexture(_stencilKernel.uniformSrcBufferTexture, _srcBufferTexture, 0);

    // bind stencil table textures.
    bindTexture(_stencilKernel.uniformSizesTexture,   sizesTexture, 1);
    bindTexture(_stencilKernel.uniformOffsetsTexture, offsetsTexture, 2);
    bindTexture(_stencilKernel.uniformIndicesTexture, indicesTexture, 3);
    bindTexture(_stencilKernel.uniformWeightsTexture, weightsTexture, 4);
    if (_stencilKernel.uniformDuWeightsTexture >= 0 && duWeightsTexture)
        bindTexture(_stencilKernel.uniformDuWeightsTexture, duWeightsTexture, 5);
    if (_stencilKernel.uniformDvWeightsTexture >= 0 && dvWeightsTexture)
        bindTexture(_stencilKernel.uniformDvWeightsTexture, dvWeightsTexture, 6);

    // set batch range
    glUniform1i(_stencilKernel.uniformStart,     start);
    glUniform1i(_stencilKernel.uniformEnd,       end);
    glUniform1i(_stencilKernel.uniformSrcOffset, srcDesc.offset);

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
    int dstBufferBindOffset = dstDesc.stride ?
        (dstDesc.offset - (dstDesc.offset % dstDesc.stride)) : 0;
    int duBufferBindOffset = duDesc.stride ?
        (duDesc.offset - (duDesc.offset % duDesc.stride)) : 0;
    int dvBufferBindOffset = dvDesc.stride ?
        (dvDesc.offset - (dvDesc.offset % dvDesc.stride)) : 0;

    // bind destination buffer
    glBindBufferRange(GL_TRANSFORM_FEEDBACK_BUFFER,
                      0, dstBuffer,
                      dstBufferBindOffset * sizeof(float),
                      count * dstDesc.stride * sizeof(float));

    if (duDesc.length > 0) {
        glBindBufferRange(GL_TRANSFORM_FEEDBACK_BUFFER,
                          1, duBuffer,
                          duBufferBindOffset * sizeof(float),
                          count * duDesc.stride * sizeof(float));
    }

    if (dvDesc.length > 0) {
        glBindBufferRange(GL_TRANSFORM_FEEDBACK_BUFFER,
                          2, dvBuffer,
                          dvBufferBindOffset * sizeof(float),
                          count * dvDesc.stride * sizeof(float));
    }

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


bool
GLXFBEvaluator::EvalPatches(
    GLuint srcBuffer, BufferDescriptor const &srcDesc,
    GLuint dstBuffer, BufferDescriptor const &dstDesc,
    GLuint duBuffer,  BufferDescriptor const &duDesc,
    GLuint dvBuffer,  BufferDescriptor const &dvDesc,
    int numPatchCoords,
    GLuint patchCoordsBuffer,
    const PatchArrayVector &patchArrays,
    GLuint patchIndexTexture,
    GLuint patchParamTexture) const {

    bool derivatives = (duDesc.length > 0 || dvDesc.length > 0);

    if (!_patchKernel.program) return false;

    // bind vertex array
    // always create new one, to be safe with multiple contexts (slow though)
    GLuint vao = 0;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glEnable(GL_RASTERIZER_DISCARD);
    glUseProgram(_patchKernel.program);

    // Set input VBO as a texture buffer.
    glBindTexture(GL_TEXTURE_BUFFER, _srcBufferTexture);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_R32F, srcBuffer);
    glBindTexture(GL_TEXTURE_BUFFER, 0);

    bindTexture(_patchKernel.uniformSrcBufferTexture, _srcBufferTexture, 0);

    // bind patch index and patch param textures.
    bindTexture(_patchKernel.uniformPatchParamTexture, patchParamTexture, 1);
    bindTexture(_patchKernel.uniformPatchIndexTexture, patchIndexTexture, 2);

    // set other uniforms
    glUniform4iv(_patchKernel.uniformPatchArray, (int)patchArrays.size(),
                 (const GLint*)&patchArrays[0]);
    glUniform1i(_patchKernel.uniformSrcOffset, srcDesc.offset);

    // input patchcoords
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    int stride = sizeof(int) * 5; // patchcoord = int*5 struct
    glBindBuffer(GL_ARRAY_BUFFER, patchCoordsBuffer);
    glVertexAttribIPointer(0, 3, GL_UNSIGNED_INT, stride, (void*)0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, (void*)(sizeof(int)*3));

    int dstBufferBindOffset =
        dstDesc.offset - (dstDesc.offset % dstDesc.stride);
    int duBufferBindOffset = duDesc.stride
        ? (duDesc.offset - (duDesc.offset % duDesc.stride))
        : 0;
    int dvBufferBindOffset = dvDesc.stride
        ? (dvDesc.offset - (dvDesc.offset % dvDesc.stride))
        : 0;

    // bind destination buffer
    glBindBufferRange(GL_TRANSFORM_FEEDBACK_BUFFER,
                      0, dstBuffer,
                      dstBufferBindOffset * sizeof(float),
                      numPatchCoords * dstDesc.stride * sizeof(float));

    if (derivatives) {
        glBindBufferRange(GL_TRANSFORM_FEEDBACK_BUFFER,
                          1, duBuffer,
                          duBufferBindOffset * sizeof(float),
                          numPatchCoords * duDesc.stride * sizeof(float));

        glBindBufferRange(GL_TRANSFORM_FEEDBACK_BUFFER,
                          2, dvBuffer,
                          dvBufferBindOffset * sizeof(float),
                          numPatchCoords * dvDesc.stride * sizeof(float));

    }

    glBeginTransformFeedback(GL_POINTS);
    glDrawArrays(GL_POINTS, 0, numPatchCoords);
    glEndTransformFeedback();

    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, 0);

    // unbind textures
    for (int i = 0; i < 3; ++i) {
        glActiveTexture(GL_TEXTURE0 + i);
        glBindTexture(GL_TEXTURE_BUFFER, 0);
    }

    glDisable(GL_RASTERIZER_DISCARD);
    glUseProgram(0);
    glActiveTexture(GL_TEXTURE0);

    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);

    // revert vao
    glBindVertexArray(0);
    glDeleteVertexArrays(1, &vao);


    return true;
}

// ---------------------------------------------------------------------------

GLXFBEvaluator::_StencilKernel::_StencilKernel() : program(0) {
}
GLXFBEvaluator::_StencilKernel::~_StencilKernel() {
    if (program) {
        glDeleteProgram(program);
    }
}

bool
GLXFBEvaluator::_StencilKernel::Compile(BufferDescriptor const &srcDesc,
                                        BufferDescriptor const &dstDesc,
                                        BufferDescriptor const &duDesc,
                                        BufferDescriptor const &dvDesc) {
    // create stencil kernel
    if (program) {
        glDeleteProgram(program);
    }

    bool derivatives = (duDesc.length > 0 || dvDesc.length > 0);
    const char *kernelDef = derivatives
        ? "#define OPENSUBDIV_GLSL_XFB_KERNEL_EVAL_STENCILS\n"
          "#define OPENSUBDIV_GLSL_XFB_USE_DERIVATIVES\n"
        : "#define OPENSUBDIV_GLSL_XFB_KERNEL_EVAL_STENCILS\n";

    program = compileKernel(srcDesc, dstDesc, duDesc, dvDesc, kernelDef);
    if (program == 0) return false;

    // cache uniform locations (TODO: use uniform block)
    uniformSrcBufferTexture = glGetUniformLocation(program, "vertexBuffer");
    uniformSrcOffset        = glGetUniformLocation(program, "srcOffset");
    uniformSizesTexture     = glGetUniformLocation(program, "sizes");
    uniformOffsetsTexture   = glGetUniformLocation(program, "offsets");
    uniformIndicesTexture   = glGetUniformLocation(program, "indices");
    uniformWeightsTexture   = glGetUniformLocation(program, "weights");
    uniformDuWeightsTexture = glGetUniformLocation(program, "duWeights");
    uniformDvWeightsTexture = glGetUniformLocation(program, "dvWeights");
    uniformStart            = glGetUniformLocation(program, "batchStart");
    uniformEnd              = glGetUniformLocation(program, "batchEnd");

    return true;
}

// ---------------------------------------------------------------------------

GLXFBEvaluator::_PatchKernel::_PatchKernel() : program(0) {
}
GLXFBEvaluator::_PatchKernel::~_PatchKernel() {
    if (program) {
        glDeleteProgram(program);
    }
}

bool
GLXFBEvaluator::_PatchKernel::Compile(BufferDescriptor const &srcDesc,
                                      BufferDescriptor const &dstDesc,
                                      BufferDescriptor const &duDesc,
                                      BufferDescriptor const &dvDesc) {
    // create stencil kernel
    if (program) {
        glDeleteProgram(program);
    }

    bool derivatives = (duDesc.length > 0 || dvDesc.length > 0);
    const char *kernelDef = derivatives
        ? "#define OPENSUBDIV_GLSL_XFB_KERNEL_EVAL_PATCHES\n"
          "#define OPENSUBDIV_GLSL_XFB_USE_DERIVATIVES\n"
        : "#define OPENSUBDIV_GLSL_XFB_KERNEL_EVAL_PATCHES\n";

    program = compileKernel(srcDesc, dstDesc, duDesc, dvDesc, kernelDef);
    if (program == 0) return false;

    // cache uniform locations
    uniformSrcBufferTexture  = glGetUniformLocation(program, "vertexBuffer");
    uniformSrcOffset         = glGetUniformLocation(program, "srcOffset");
    uniformPatchArray        = glGetUniformLocation(program, "patchArray");
    uniformPatchParamTexture = glGetUniformLocation(program, "patchParamBuffer");
    uniformPatchIndexTexture = glGetUniformLocation(program, "patchIndexBuffer");

    return true;
}


}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
