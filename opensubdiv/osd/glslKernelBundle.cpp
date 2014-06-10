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
#include <sstream>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

static const char *shaderSource =
#include "../osd/glslComputeKernel.gen.h"
;

OsdGLSLComputeKernelBundle::OsdGLSLComputeKernelBundle()
    : _program(0),
      _numVertexElements(0),
      _vertexStride(0),
      _numVaryingElements(0),
      _varyingStride(0) {

    // XXX: too rough!
    _workGroupSize = 64;
}

OsdGLSLComputeKernelBundle::~OsdGLSLComputeKernelBundle() {
    if (_program)
        glDeleteProgram(_program);
}

bool
OsdGLSLComputeKernelBundle::Compile(
    OsdVertexBufferDescriptor const &vertexDesc,
    OsdVertexBufferDescriptor const &varyingDesc) {

    _numVertexElements = vertexDesc.length;
    _vertexStride = vertexDesc.stride;
    _numVaryingElements = varyingDesc.length;
    _varyingStride = varyingDesc.stride;

    if (_program) {
        glDeleteProgram(_program);
        _program = 0;
    }
    _program = glCreateProgram();

    GLuint shader = glCreateShader(GL_COMPUTE_SHADER);

    std::ostringstream defines;
    defines << "#define NUM_VERTEX_ELEMENTS "  << _numVertexElements << "\n"
            << "#define VERTEX_STRIDE "        << _vertexStride << "\n"
            << "#define NUM_VARYING_ELEMENTS " << _numVaryingElements << "\n"
            << "#define VARYING_STRIDE "       << _varyingStride << "\n"
            << "#define WORK_GROUP_SIZE "      << _workGroupSize << "\n";
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
        OsdError(OSD_GLSL_LINK_ERROR, buffer);

        glGetProgramInfoLog(_program, 1024, NULL, buffer);
        OsdError(OSD_GLSL_LINK_ERROR, buffer);

        glDeleteProgram(_program);
        _program = 0;
        return false;
    }

    glDeleteShader(shader);

    _subComputeFace               = glGetSubroutineIndex(_program,
                                                         GL_COMPUTE_SHADER,
                                                         "catmarkComputeFace");
    _subComputeQuadFace           = glGetSubroutineIndex(_program,
                                                         GL_COMPUTE_SHADER,
                                                         "catmarkComputeQuadFace");
    _subComputeTriQuadFace        = glGetSubroutineIndex(_program,
                                                         GL_COMPUTE_SHADER,
                                                         "catmarkComputeTriQuadFace");
    _subComputeEdge               = glGetSubroutineIndex(_program,
                                                         GL_COMPUTE_SHADER,
                                                         "catmarkComputeEdge");
    _subComputeRestrictedEdge     = glGetSubroutineIndex(_program,
                                                         GL_COMPUTE_SHADER,
                                                         "catmarkComputeRestrictedEdge");
    _subComputeRestrictedVertexA  = glGetSubroutineIndex(_program,
                                                         GL_COMPUTE_SHADER,
                                                         "catmarkComputeRestrictedVertexA");
    _subComputeRestrictedVertexB1 = glGetSubroutineIndex(_program,
                                                         GL_COMPUTE_SHADER,
                                                         "catmarkComputeRestrictedVertexB1");
    _subComputeRestrictedVertexB2 = glGetSubroutineIndex(_program,
                                                         GL_COMPUTE_SHADER,
                                                         "catmarkComputeRestrictedVertexB2");
    _subComputeBilinearEdge       = glGetSubroutineIndex(_program,
                                                         GL_COMPUTE_SHADER,
                                                         "bilinearComputeEdge");
    _subComputeVertex             = glGetSubroutineIndex(_program,
                                                         GL_COMPUTE_SHADER,
                                                         "bilinearComputeVertex");
    _subComputeVertexA            = glGetSubroutineIndex(_program,
                                                         GL_COMPUTE_SHADER,
                                                         "catmarkComputeVertexA");
    _subComputeCatmarkVertexB     = glGetSubroutineIndex(_program,
                                                         GL_COMPUTE_SHADER,
                                                         "catmarkComputeVertexB");
    _subComputeLoopVertexB        = glGetSubroutineIndex(_program,
                                                         GL_COMPUTE_SHADER,
                                                         "loopComputeVertexB");

    // set uniform locations for compute
    _uniformVertexPass        = glGetUniformLocation(_program, "vertexPass");
    _uniformVertexOffset      = glGetUniformLocation(_program, "vertexOffset");
    _uniformTableOffset       = glGetUniformLocation(_program, "tableOffset");
    _uniformIndexStart        = glGetUniformLocation(_program, "indexStart");
    _uniformIndexEnd          = glGetUniformLocation(_program, "indexEnd");
    _uniformVertexBaseOffset  = glGetUniformLocation(_program, "vertexBaseOffset");
    _uniformVaryingBaseOffset = glGetUniformLocation(_program, "varyingBaseOffset");

    _tableUniforms[FarSubdivisionTables::F_IT]  = glGetUniformLocation(_program, "_F0_IT");
    _tableUniforms[FarSubdivisionTables::F_ITa] = glGetUniformLocation(_program, "_F0_ITa");
    _tableUniforms[FarSubdivisionTables::E_IT]  = glGetUniformLocation(_program, "_E0_IT");
    _tableUniforms[FarSubdivisionTables::V_IT]  = glGetUniformLocation(_program, "_V0_IT");
    _tableUniforms[FarSubdivisionTables::V_ITa] = glGetUniformLocation(_program, "_V0_ITa");
    _tableUniforms[FarSubdivisionTables::E_W]   = glGetUniformLocation(_program, "_E0_S");
    _tableUniforms[FarSubdivisionTables::V_W]   = glGetUniformLocation(_program, "_V0_S");

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

    // sync for later reading.
    // XXX: in theory, just SHADER_STORAGE_BARRIER is needed here. However
    // we found a problem (issue #295) with nvidia driver 331.49 / Quadro4000
    // resulting invalid vertices.
    // Apparently adding TEXTURE_FETCH_BARRIER after face kernel fixes it.
    // The workaroud is commented out, since it looks fixed at driver 334.xx.
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void
OsdGLSLComputeKernelBundle::ApplyBilinearFaceVerticesKernel(
    int vertexOffset, int tableOffset, int start, int end) {

    glUniformSubroutinesuiv(GL_COMPUTE_SHADER, 1, &_subComputeFace);
    dispatchCompute(vertexOffset, tableOffset, start, end);

    // glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
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

    // see the comment in dispatchCompute()
    // this workaround causes a performance problem.
    // glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
}

void
OsdGLSLComputeKernelBundle::ApplyCatmarkQuadFaceVerticesKernel(
    int vertexOffset, int tableOffset, int start, int end) {

    glUniformSubroutinesuiv(GL_COMPUTE_SHADER, 1, &_subComputeQuadFace);
    dispatchCompute(vertexOffset, tableOffset, start, end);
}

void
OsdGLSLComputeKernelBundle::ApplyCatmarkTriQuadFaceVerticesKernel(
    int vertexOffset, int tableOffset, int start, int end) {

    glUniformSubroutinesuiv(GL_COMPUTE_SHADER, 1, &_subComputeTriQuadFace);
    dispatchCompute(vertexOffset, tableOffset, start, end);
}

void
OsdGLSLComputeKernelBundle::ApplyCatmarkEdgeVerticesKernel(
    int vertexOffset, int tableOffset, int start, int end) {

    glUniformSubroutinesuiv(GL_COMPUTE_SHADER, 1, &_subComputeEdge);
    dispatchCompute(vertexOffset, tableOffset, start, end);
}

void
OsdGLSLComputeKernelBundle::ApplyCatmarkRestrictedEdgeVerticesKernel(
    int vertexOffset, int tableOffset, int start, int end) {

    glUniformSubroutinesuiv(GL_COMPUTE_SHADER, 1, &_subComputeRestrictedEdge);
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
OsdGLSLComputeKernelBundle::ApplyCatmarkRestrictedVertexVerticesKernelB1(
    int vertexOffset, int tableOffset, int start, int end) {

    glUniformSubroutinesuiv(GL_COMPUTE_SHADER, 1, &_subComputeRestrictedVertexB1);
    dispatchCompute(vertexOffset, tableOffset, start, end);
}

void
OsdGLSLComputeKernelBundle::ApplyCatmarkRestrictedVertexVerticesKernelB2(
    int vertexOffset, int tableOffset, int start, int end) {

    glUniformSubroutinesuiv(GL_COMPUTE_SHADER, 1, &_subComputeRestrictedVertexB2);
    dispatchCompute(vertexOffset, tableOffset, start, end);
}

void
OsdGLSLComputeKernelBundle::ApplyCatmarkRestrictedVertexVerticesKernelA(
    int vertexOffset, int tableOffset, int start, int end) {

    glUniformSubroutinesuiv(GL_COMPUTE_SHADER, 1, &_subComputeRestrictedVertexA);
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
OsdGLSLComputeKernelBundle::UseProgram(int vertexBaseOffset,
                                       int varyingBaseOffset) const
{
    glUseProgram(_program);

    glUniform1i(_uniformVertexBaseOffset, vertexBaseOffset);
    glUniform1i(_uniformVaryingBaseOffset, varyingBaseOffset);
}

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
