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

#include "../osd/glslComputeController.h"
#include "../osd/vertexDescriptor.h"
//#include "../osd/debug.h"
#include "../osd/error.h"
#include "../osd/opengl.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <sstream>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

static const char *shaderSource =
#include "../osd/glslComputeKernel.gen.h"
;

// ----------------------------------------------------------------------------

class GLSLComputeController::KernelBundle :
    NonCopyable<GLSLComputeController::KernelBundle> {

public:

    KernelBundle() :
        _program(0),
        _uniformSizes(0),
        _uniformOffsets(0),
        _uniformIndices(0),
        _uniformWeights(0),
        _uniformStart(0),
        _uniformEnd(0),
        _uniformOffset(0),
        _uniformNumCVs(0),
        _workGroupSize(64) { }

    ~KernelBundle() {
        if (_program) {
            glDeleteProgram(_program);
        }
    }

    void UseProgram(int primvarOffset) const {
        glUseProgram(_program);
        glUniform1i(_uniformOffset, primvarOffset);

        //OSD_DEBUG_CHECK_GL_ERROR("UseProgram");
    }

    bool Compile(VertexBufferDescriptor const & desc) {

        _desc = VertexBufferDescriptor(0, desc.length, desc.stride);

        if (_program) {
            glDeleteProgram(_program);
            _program=0;
        }
        _program = glCreateProgram();

        GLuint shader = glCreateShader(GL_COMPUTE_SHADER);

        std::ostringstream defines;
        defines << "#define OFFSET " << _desc.offset << "\n"
                << "#define LENGTH " << _desc.length << "\n"
                << "#define STRIDE " << _desc.stride << "\n"
                << "#define WORK_GROUP_SIZE " << _workGroupSize << "\n";
        std::string defineStr = defines.str();

        const char *shaderSources[3];
        shaderSources[0] = defineStr.c_str();
        shaderSources[1] = shaderSource;
        glShaderSource(shader, 2, shaderSources, NULL);
        glCompileShader(shader);
        glAttachShader(_program, shader);

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

        _subStencilKernel = glGetSubroutineIndex(_program, GL_COMPUTE_SHADER, "computeStencil");

        // set uniform locations for compute kernels
        _uniformSizes   = glGetUniformLocation(_program, "sterncilSizes");
        _uniformOffsets = glGetUniformLocation(_program, "sterncilOffsets");
        _uniformIndices = glGetUniformLocation(_program, "sterncilIndices");
        _uniformWeights = glGetUniformLocation(_program, "sterncilIWeights");

        _uniformStart   = glGetUniformLocation(_program, "batchStart");
        _uniformEnd     = glGetUniformLocation(_program, "batchEnd");

        _uniformOffset  = glGetUniformLocation(_program, "primvarOffset");
        _uniformNumCVs  = glGetUniformLocation(_program, "numCVs");

        //OSD_DEBUG_CHECK_GL_ERROR("Compile");

        return true;
    }

    void ApplyStencilTableKernel(Far::KernelBatch const &batch, int offset, int numCVs) const {

        // select stencil GLSL subroutine
        glUniformSubroutinesuiv(GL_COMPUTE_SHADER, 1, &_subStencilKernel);

        dispatchCompute(offset, numCVs, batch.start, batch.end);
    }

    struct Match {

        Match(VertexBufferDescriptor const & d) : desc(d) { }

        bool operator() (KernelBundle const * kernel) {
            return (desc.length==kernel->_desc.length and 
                    desc.stride==kernel->_desc.stride);
        }

        VertexBufferDescriptor desc;
    };

protected:

    void dispatchCompute(int offset, int numCVs, int start, int end) const {
    
        int count = end - start;
        if (count<=0) {
            return;
        }
        

        glUniform1i(_uniformStart, start);
        glUniform1i(_uniformEnd, end);

        glUniform1i(_uniformOffset, offset);
        glUniform1i(_uniformNumCVs, numCVs);
        
        glDispatchCompute(count/_workGroupSize + 1, 1, 1);

        // sync for later reading.
        // XXX: in theory, just SHADER_STORAGE_BARRIER is needed here. However
        // we found a problem (issue #295) with nvidia driver 331.49 / Quadro4000
        // resulting in invalid vertices.
        // Apparently adding TEXTURE_FETCH_BARRIER after a kernel fixes it.
        // The workaroud is commented out, since it looks fixed as of driver 334.xx.
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        //OSD_DEBUG_CHECK_GL_ERROR("dispatchCompute");
    }

private:

    GLuint _program;

    GLuint _subStencilKernel; // stencil compute kernel GLSL subroutine

    GLuint _uniformSizes,     // uniform paramaeters for kernels
           _uniformOffsets,
           _uniformIndices,
           _uniformWeights,

           _uniformStart,     // batch
           _uniformEnd,

           _uniformOffset,    // GL primvar buffer descriptor
           _uniformNumCVs;    // number of const control vertices padded at
                              // the beginning of the buffer

    VertexBufferDescriptor _desc; // primvar buffer descriptor

    int _workGroupSize;
};

// ----------------------------------------------------------------------------

void
GLSLComputeController::ApplyStencilTableKernel(
    Far::KernelBatch const &batch, ComputeContext const *context) const {

    assert(context);
    
    _currentBindState.kernelBundle->ApplyStencilTableKernel(
        batch, _currentBindState.desc.offset, context->GetNumControlVertices());
}

// ----------------------------------------------------------------------------

GLSLComputeController::GLSLComputeController() { }

GLSLComputeController::~GLSLComputeController() {
    for (KernelRegistry::iterator it = _kernelRegistry.begin();
        it != _kernelRegistry.end(); ++it) {
        delete *it;
    }
}

// ----------------------------------------------------------------------------

void
GLSLComputeController::Synchronize() {

    glFinish();
}

// ----------------------------------------------------------------------------
GLSLComputeController::KernelBundle const *
GLSLComputeController::getKernel(VertexBufferDescriptor const &desc) {

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

void
GLSLComputeController::bindBufferAndProgram() {

    if (_currentBindState.buffer)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _currentBindState.buffer);

    _currentBindState.kernelBundle->UseProgram(/*primvarOffset*/0);

    glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
}

void
GLSLComputeController::unbindBufferAndProgram() {

    glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0);
    glUseProgram(0);
}

// ----------------------------------------------------------------------------

}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
