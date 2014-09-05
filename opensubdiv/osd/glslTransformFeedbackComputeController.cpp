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

#include "../osd/error.h"
//#define OSD_DEBUG_BUILD
#include "../osd/debug.h"
#include "../osd/glslTransformFeedbackComputeController.h"
#include "../osd/glslTransformFeedbackComputeContext.h"
#include "../osd/opengl.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <iostream>
#include <sstream>

#if _MSC_VER
    #define snprintf _snprintf
#endif

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

static const char *shaderSource =
#include "../osd/glslTransformFeedbackKernel.gen.h"
;

static const char *shaderDefines = ""
#ifdef OPT_CATMARK_V_IT_VEC2
"#define OPT_CATMARK_V_IT_VEC2\n"
#endif
#ifdef OPT_E0_IT_VEC4
"#define OPT_E0_IT_VEC4\n"
#endif
#ifdef OPT_E0_S_VEC2
"#define OPT_E0_S_VEC2\n"
#endif
;

// ----------------------------------------------------------------------------
static void
bindTexture(GLint sampler, GLuint texture, int unit) {
    if (sampler==-1) {
        return;
    }
    glUniform1i(sampler, unit);
    glActiveTexture(GL_TEXTURE0 + unit);
    glBindTexture(GL_TEXTURE_BUFFER, texture);
    glActiveTexture(GL_TEXTURE0);
}

// ----------------------------------------------------------------------------

class GLSLTransformFeedbackComputeController::KernelBundle :
    NonCopyable<GLSLTransformFeedbackComputeController::KernelBundle> {

public:

    KernelBundle() :
        _program(0),
        _uniformSizes(0),
        _uniformOffsets(0),
        _uniformIndices(0),
        _uniformWeights(0),
        _uniformStart(0),
        _uniformEnd(0),
        _uniformOffset(0) { }

    ~KernelBundle() {
        if (_program) {
            glDeleteProgram(_program);
        }
    }

    void UseProgram(int primvarOffset) const {
        glUseProgram(_program);
        glUniform1i(_uniformOffset, primvarOffset);
    }

    bool Compile(VertexBufferDescriptor const & desc) {

        _desc = VertexBufferDescriptor(0, desc.length, desc.stride);

        if (_program) {
            glDeleteProgram(_program);
            _program=0;
        }
        _program = glCreateProgram();

        GLuint shader = glCreateShader(GL_VERTEX_SHADER);

        std::ostringstream defines;
        defines << "#define LENGTH " << desc.length << "\n"
                << "#define STRIDE " << desc.stride << "\n";
        std::string defineStr = defines.str();

        const char *shaderSources[3];
        shaderSources[0] = defineStr.c_str();
        shaderSources[1] = shaderDefines;
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
            char attrName[32];
            for (int i = 0; i < desc.offset; ++i) {
                outputs.push_back("gl_SkipComponents1");
            }
            for (int i = 0; i < desc.length; ++i) {
                snprintf(attrName, 32, "outVertexBuffer[%d]", i);
                outputs.push_back(attrName);
            }
            for (int i = desc.offset + desc.length; i < desc.stride; ++i) {
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
            Error(OSD_GLSL_LINK_ERROR, buffer);

            glGetProgramInfoLog(_program, 1024, NULL, buffer);
            Error(OSD_GLSL_LINK_ERROR, buffer);

            glDeleteProgram(_program);
            _program = 0;
            return false;
        }

        glDeleteShader(shader);

        _subStencilKernel = glGetSubroutineIndex(
            _program, GL_VERTEX_SHADER, "computeStencil");

        // set uniform locations for compute kernels
        _primvarBuffer  = glGetUniformLocation(_program, "vertexBuffer");

        _uniformSizes   = glGetUniformLocation(_program, "sizes");
        _uniformOffsets = glGetUniformLocation(_program, "offsets");
        _uniformIndices = glGetUniformLocation(_program, "indices");
        _uniformWeights = glGetUniformLocation(_program, "weights");

        _uniformStart   = glGetUniformLocation(_program, "batchStart");
        _uniformEnd     = glGetUniformLocation(_program, "batchEnd");

        _uniformOffset  = glGetUniformLocation(_program, "primvarOffset");

        OSD_DEBUG_CHECK_GL_ERROR("KernelBundle::Compile");

        return true;
    }

    GLint GetPrimvarBufferLocation() const {
        return _primvarBuffer;
    }

    GLint GetSizesLocation() const {
        return _uniformSizes;
    }

    GLint GetOffsetsLocation() const {
        return _uniformOffsets;
    }
    GLint GetIndicesLocation() const {
        return _uniformIndices;
    }
    GLint GetWeightsLocation() const {
        return _uniformWeights;
    }

    void TransformPrimvarBuffer(GLuint primvarBuffer,
        int offset, int numCVs, int start, int end) const {

        assert(end >= start);

        // set batch range
        glUniform1i(_uniformStart,  start);
        glUniform1i(_uniformEnd,    end);
        glUniform1i(_uniformOffset, offset);

        int count = end - start,
            stride = _desc.stride*sizeof(float);

        glBindBufferRange(GL_TRANSFORM_FEEDBACK_BUFFER,
            0, primvarBuffer, (start + numCVs)*stride, count*stride);

        glBeginTransformFeedback(GL_POINTS);

        glDrawArrays(GL_POINTS, 0, count);

        glEndTransformFeedback();

        glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, 0);

        //OSD_DEBUG_CHECK_GL_ERROR("TransformPrimvarBuffer\n");
    }

    void ApplyStencilTableKernel(Far::KernelBatch const &batch,
        GLuint primvarBuffer, int offset, int numCVs) const {

        glUniformSubroutinesuiv(GL_VERTEX_SHADER, 1, &_subStencilKernel);

        TransformPrimvarBuffer(primvarBuffer,
            offset, numCVs, batch.start, batch.end);
    }

    struct Match {

        Match(VertexBufferDescriptor const & d) : desc(d) { }

        bool operator() (KernelBundle const * kernel) {
            return (desc.length==kernel->_desc.length and
                    desc.stride==kernel->_desc.stride);
        }

        VertexBufferDescriptor desc;
    };

private:

    GLuint _program;

    GLuint _subStencilKernel; // stencil compute kernel GLSL subroutine

    GLint _primvarBuffer;

    GLint _uniformSizes,     // uniform paramaeters for kernels
          _uniformOffsets,
          _uniformIndices,
          _uniformWeights,

          _uniformStart,     // batch
          _uniformEnd,

          _uniformOffset;    // GL primvar buffer descriptor

    VertexBufferDescriptor _desc; // primvar buffer descriptor
};

// ----------------------------------------------------------------------------
void
GLSLTransformFeedbackComputeController::bindBufferAndProgram(
    GLuint & feedbackTexture) {

    glEnable(GL_RASTERIZER_DISCARD);
    _currentBindState.kernelBundle->UseProgram(/*primvarOffset*/0);

    if (not feedbackTexture) {
        glGenTextures(1, &feedbackTexture);
#if defined(GL_EXT_direct_state_access)
        if (glTextureBufferEXT) {
            glTextureBufferEXT(feedbackTexture, GL_TEXTURE_BUFFER, GL_R32F,
                _currentBindState.buffer);
        } else {
#else
        {
#endif
            glBindTexture(GL_TEXTURE_BUFFER, feedbackTexture);
            glTexBuffer(GL_TEXTURE_BUFFER, GL_R32F, _currentBindState.buffer);
            glBindTexture(GL_TEXTURE_BUFFER, 0);
        }
    }

    bindTexture(
        _currentBindState.kernelBundle->GetPrimvarBufferLocation(), feedbackTexture, 0);

    // bind vertex array
    // always create new one, to be safe with multiple contexts.
    glGenVertexArrays(1, &_vao);
    glBindVertexArray(_vao);
}

// ----------------------------------------------------------------------------

void
GLSLTransformFeedbackComputeController::bindContextStencilTables(
    ComputeContext const *context, bool varying) {

    GLint sizesLocation   = _currentBindState.kernelBundle->GetSizesLocation(),
          offsetsLocation = _currentBindState.kernelBundle->GetOffsetsLocation(),
          indicesLocation = _currentBindState.kernelBundle->GetIndicesLocation(),
          weightsLocation = _currentBindState.kernelBundle->GetWeightsLocation();

    if (not varying) {
        bindTexture(sizesLocation,   context->GetVertexStencilTablesSizes(), 1);
        bindTexture(offsetsLocation, context->GetVertexStencilTablesOffsets(), 2);
        bindTexture(indicesLocation, context->GetVertexStencilTablesIndices(), 3);
        bindTexture(weightsLocation, context->GetVertexStencilTablesWeights(), 4);
    } else {
        bindTexture(sizesLocation,   context->GetVaryingStencilTablesSizes(), 1);
        bindTexture(offsetsLocation, context->GetVaryingStencilTablesOffsets(), 2);
        bindTexture(indicesLocation, context->GetVaryingStencilTablesIndices(), 3);
        bindTexture(weightsLocation, context->GetVaryingStencilTablesWeights(), 4);
    }
}

// ----------------------------------------------------------------------------

void
GLSLTransformFeedbackComputeController::unbindResources() {

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_BUFFER, 0);

    glDisable(GL_RASTERIZER_DISCARD);
    glUseProgram(0);
    glActiveTexture(GL_TEXTURE0);

    glBindVertexArray(0);
    glDeleteVertexArrays(1, &_vao);
}

// ----------------------------------------------------------------------------

GLSLTransformFeedbackComputeController::KernelBundle const *
GLSLTransformFeedbackComputeController::getKernel(
    VertexBufferDescriptor const &desc) {

    KernelRegistry::iterator it =
        std::find_if(_kernelRegistry.begin(), _kernelRegistry.end(),
            KernelBundle::Match(desc));

    if (it != _kernelRegistry.end()) {
        return *it;
    } else {
        KernelBundle * kernelBundle = new KernelBundle();
        kernelBundle->Compile(desc);
        _kernelRegistry.push_back(kernelBundle);
        return kernelBundle;
    }
}

// ----------------------------------------------------------------------------

void
GLSLTransformFeedbackComputeController::ApplyStencilTableKernel(
    Far::KernelBatch const &batch,
        GLSLTransformFeedbackComputeContext const *context) const {

    assert(context);

    _currentBindState.kernelBundle->ApplyStencilTableKernel(batch,
        _currentBindState.buffer, _currentBindState.desc.offset,
            context->GetNumControlVertices());
}


// ----------------------------------------------------------------------------

GLSLTransformFeedbackComputeController::GLSLTransformFeedbackComputeController() :
    _vertexTexture(0), _varyingTexture(0), _vao(0) {
}

GLSLTransformFeedbackComputeController::~GLSLTransformFeedbackComputeController() {

    for (KernelRegistry::iterator it = _kernelRegistry.begin();
        it != _kernelRegistry.end(); ++it) {
        delete *it;
    }
    if (_vertexTexture) {
        glDeleteTextures(1, &_vertexTexture);
    }
    if (_varyingTexture) {
        glDeleteTextures(1, &_varyingTexture);
    }
}

// ----------------------------------------------------------------------------

void
GLSLTransformFeedbackComputeController::Synchronize() {
    glFinish();
}



}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
