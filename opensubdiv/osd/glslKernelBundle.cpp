//
//     Copyright 2013 Pixar
//
//     Licensed under the Apache License, Version 2.0 (the "License");
//     you may not use this file except in compliance with the License
//     and the following modification to it: Section 6 Trademarks.
//     deleted and replaced with:
//
//     6. Trademarks. This License does not grant permission to use the
//     trade names, trademarks, service marks, or product names of the
//     Licensor and its affiliates, except as required for reproducing
//     the content of the NOTICE file.
//
//     You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
//     Unless required by applicable law or agreed to in writing,
//     software distributed under the License is distributed on an
//     "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
//     either express or implied.  See the License for the specific
//     language governing permissions and limitations under the
//     License.
//

#include <stdio.h>
#ifdef _MSC_VER
#define snprintf _snprintf
#endif

#include "../osd/debug.h"
#include "../osd/error.h"
#include "../osd/glslKernelBundle.h"
#include "../osd/vertex.h"

#include "../far/subdivisionTables.h"

#include "../osd/opengl.h"

#include <cassert>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

static const char *shaderSource =
#include "../osd/glslComputeKernel.inc"
;

OsdGLSLComputeKernelBundle::OsdGLSLComputeKernelBundle()
    : _program(0) {

    // XXX: too rough!
    _workGroupSize = 64;
}

OsdGLSLComputeKernelBundle::~OsdGLSLComputeKernelBundle() {
    if (_program)
        glDeleteProgram(_program);
}

bool
OsdGLSLComputeKernelBundle::Compile(int numVertexElements, int numVaryingElements) {

    _vdesc.Set(numVertexElements, numVaryingElements );

    if (_program) {
        glDeleteProgram(_program);
        _program = 0;
    }
    _program = glCreateProgram();

    GLuint shader = glCreateShader(GL_COMPUTE_SHADER);

    char constantDefine[256];
    snprintf(constantDefine, 256,
             "#define NUM_VERTEX_ELEMENTS %d\n"
             "#define NUM_VARYING_ELEMENTS %d\n"
             "#define WORK_GROUP_SIZE %d\n",
             numVertexElements, numVaryingElements, _workGroupSize);

    const char *shaderSources[3];
    shaderSources[0] = constantDefine;
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
        OsdError(OSD_GLSL_LINK_ERROR, buffer);

        glGetProgramInfoLog(_program, 1024, NULL, buffer);
        OsdError(OSD_GLSL_LINK_ERROR, buffer);

        glDeleteProgram(_program);
        _program = 0;
        // XXX ERROR HANDLE
        printf("%s\n", constantDefine);
        assert(false);
        return false;
    }

    glDeleteShader(shader);

    _subComputeFace           = glGetSubroutineIndex(_program,
                                                     GL_COMPUTE_SHADER,
                                                     "catmarkComputeFace");
    _subComputeEdge           = glGetSubroutineIndex(_program,
                                                     GL_COMPUTE_SHADER,
                                                     "catmarkComputeEdge");
    _subComputeBilinearEdge   = glGetSubroutineIndex(_program,
                                                     GL_COMPUTE_SHADER,
                                                     "bilinearComputeEdge");
    _subComputeVertex         = glGetSubroutineIndex(_program,
                                                     GL_COMPUTE_SHADER,
                                                     "bilinearComputeVertex");
    _subComputeVertexA        = glGetSubroutineIndex(_program,
                                                     GL_COMPUTE_SHADER,
                                                     "catmarkComputeVertexA");
    _subComputeCatmarkVertexB = glGetSubroutineIndex(_program,
                                                     GL_COMPUTE_SHADER,
                                                     "catmarkComputeVertexB");
    _subComputeLoopVertexB    = glGetSubroutineIndex(_program,
                                                     GL_COMPUTE_SHADER,
                                                     "loopComputeVertexB");

    // set uniform locations for compute
    _uniformVertexPass   = glGetUniformLocation(_program, "vertexPass");
    _uniformVertexOffset = glGetUniformLocation(_program, "vertexOffset");
    _uniformTableOffset  = glGetUniformLocation(_program, "tableOffset");
    _uniformIndexStart   = glGetUniformLocation(_program, "indexStart");
    _uniformIndexEnd     = glGetUniformLocation(_program, "indexEnd");

    _tableUniforms[FarSubdivisionTables<OsdVertex>::F_IT]  = glGetUniformLocation(_program, "_F0_IT");
    _tableUniforms[FarSubdivisionTables<OsdVertex>::F_ITa] = glGetUniformLocation(_program, "_F0_ITa");
    _tableUniforms[FarSubdivisionTables<OsdVertex>::E_IT]  = glGetUniformLocation(_program, "_E0_IT");
    _tableUniforms[FarSubdivisionTables<OsdVertex>::V_IT]  = glGetUniformLocation(_program, "_V0_IT");
    _tableUniforms[FarSubdivisionTables<OsdVertex>::V_ITa] = glGetUniformLocation(_program, "_V0_ITa");
    _tableUniforms[FarSubdivisionTables<OsdVertex>::E_W]   = glGetUniformLocation(_program, "_E0_S");
    _tableUniforms[FarSubdivisionTables<OsdVertex>::V_W]   = glGetUniformLocation(_program, "_V0_S");

    // set unfiorm locations for edit
    _subEditAdd               = glGetSubroutineIndex(_program,
                                                     GL_COMPUTE_SHADER,
                                                     "editAdd");

    _uniformEditPrimVarOffset = glGetUniformLocation(_program, "editPrimVarOffset");
    _uniformEditPrimVarWidth  = glGetUniformLocation(_program, "editPrimVarWidth");


    return true;
}

void
OsdGLSLComputeKernelBundle::dispatchCompute(
    int vertexOffset, int tableOffset, int start, int end) const {

    int count = end - start;
    if (count <= 0) return;

    // set batch range
    glUniform1i(_uniformVertexOffset, vertexOffset);
    glUniform1i(_uniformTableOffset, tableOffset);
    glUniform1i(_uniformIndexStart, start);
    glUniform1i(_uniformIndexEnd, end);

    // execute
    glDispatchCompute(count/_workGroupSize + 1, 1, 1);

    // sync for later reading (slow..)
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void
OsdGLSLComputeKernelBundle::ApplyBilinearFaceVerticesKernel(
    int vertexOffset, int tableOffset, int start, int end) {

    glUniformSubroutinesuiv(GL_COMPUTE_SHADER, 1, &_subComputeFace);
    dispatchCompute(vertexOffset, tableOffset, start, end);
}

void
OsdGLSLComputeKernelBundle::ApplyBilinearEdgeVerticesKernel(
    int vertexOffset, int tableOffset, int start, int end) {

    glUniformSubroutinesuiv(GL_COMPUTE_SHADER, 1, &_subComputeBilinearEdge);
    dispatchCompute(vertexOffset, tableOffset, start, end);
}

void
OsdGLSLComputeKernelBundle::ApplyBilinearVertexVerticesKernel(
    int vertexOffset, int tableOffset, int start, int end) {

    glUniformSubroutinesuiv(GL_COMPUTE_SHADER, 1, &_subComputeVertex);
    dispatchCompute(vertexOffset, tableOffset, start, end);
}


void
OsdGLSLComputeKernelBundle::ApplyCatmarkFaceVerticesKernel(
    int vertexOffset, int tableOffset, int start, int end) {

    glUniformSubroutinesuiv(GL_COMPUTE_SHADER, 1, &_subComputeFace);
    dispatchCompute(vertexOffset, tableOffset, start, end);
}

void
OsdGLSLComputeKernelBundle::ApplyCatmarkEdgeVerticesKernel(
    int vertexOffset, int tableOffset, int start, int end) {

    glUniformSubroutinesuiv(GL_COMPUTE_SHADER, 1, &_subComputeEdge);
    dispatchCompute(vertexOffset, tableOffset, start, end);
}

void
OsdGLSLComputeKernelBundle::ApplyCatmarkVertexVerticesKernelB(
    int vertexOffset, int tableOffset, int start, int end) {

    glUniformSubroutinesuiv(GL_COMPUTE_SHADER, 1, &_subComputeCatmarkVertexB);
    dispatchCompute(vertexOffset, tableOffset, start, end);
}

void
OsdGLSLComputeKernelBundle::ApplyCatmarkVertexVerticesKernelA(
    int vertexOffset, int tableOffset, int start, int end, bool pass) {

    glUniformSubroutinesuiv(GL_COMPUTE_SHADER, 1, &_subComputeVertexA);
    glUniform1i(_uniformVertexPass, pass ? 1 : 0);
    dispatchCompute(vertexOffset, tableOffset, start, end);
}

void
OsdGLSLComputeKernelBundle::ApplyLoopEdgeVerticesKernel(
    int vertexOffset, int tableOffset, int start, int end) {

    glUniformSubroutinesuiv(GL_COMPUTE_SHADER, 1, &_subComputeEdge);
    dispatchCompute(vertexOffset, tableOffset, start, end);
}

void
OsdGLSLComputeKernelBundle::ApplyLoopVertexVerticesKernelB(
    int vertexOffset, int tableOffset, int start, int end) {

    glUniformSubroutinesuiv(GL_COMPUTE_SHADER, 1, &_subComputeLoopVertexB);
    dispatchCompute(vertexOffset, tableOffset, start, end);
}

void
OsdGLSLComputeKernelBundle::ApplyLoopVertexVerticesKernelA(
    int vertexOffset, int tableOffset, int start, int end, bool pass) {

    glUniformSubroutinesuiv(GL_COMPUTE_SHADER, 1, &_subComputeVertexA);
    glUniform1i(_uniformVertexPass, pass ? 1 : 0);
    dispatchCompute(vertexOffset, tableOffset, start, end);
}

void
OsdGLSLComputeKernelBundle::ApplyEditAdd(
    int primvarOffset, int primvarWidth,
    int vertexOffset, int tableOffset, int start, int end) {

    glUniformSubroutinesuiv(GL_COMPUTE_SHADER, 1, &_subEditAdd);
    glUniform1i(_uniformEditPrimVarOffset, primvarOffset);
    glUniform1i(_uniformEditPrimVarWidth, primvarWidth);
    dispatchCompute(vertexOffset, tableOffset, start, end);
}

void
OsdGLSLComputeKernelBundle::UseProgram() const
{
    glUseProgram(_program);
}

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
