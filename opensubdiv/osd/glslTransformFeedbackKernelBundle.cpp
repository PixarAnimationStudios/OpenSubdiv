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

#include "../version.h"

#include <stdio.h>
#ifdef _MSC_VER
#define snprintf _snprintf
#endif

#include "../osd/debug.h"
#include "../osd/error.h"
#include "../osd/glslTransformFeedbackKernelBundle.h"
#include "../osd/vertex.h"

#include "../far/subdivisionTables.h"

#include "../osd/opengl.h"

#include <cassert>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

static const char *shaderSource =
#include "../osd/glslTransformFeedbackKernel.inc"
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

OsdGLSLTransformFeedbackKernelBundle::OsdGLSLTransformFeedbackKernelBundle()
    : _program(0) {
}

OsdGLSLTransformFeedbackKernelBundle::~OsdGLSLTransformFeedbackKernelBundle() {
    if (_program)
        glDeleteProgram(_program);
}

bool
OsdGLSLTransformFeedbackKernelBundle::Compile(int numVertexElements, int numVaryingElements) {

    assert(numVertexElements >= 3); // at least xyz required (for performance reason)

    _vdesc.Set(numVertexElements, numVaryingElements);
    
    _program = glCreateProgram();

    GLuint shader = glCreateShader(GL_VERTEX_SHADER);

    char constantDefine[256];
    snprintf(constantDefine, 256,
             "#define NUM_VERTEX_ELEMENTS %d\n"
             "#define NUM_VARYING_ELEMENTS %d\n",
             numVertexElements, numVaryingElements);

    const char *shaderSources[3];
    shaderSources[0] = constantDefine;
    shaderSources[1] = shaderDefines;
    shaderSources[2] = shaderSource;
    glShaderSource(shader, 3, shaderSources, NULL);
    glCompileShader(shader);
    glAttachShader(_program, shader);

    const char *outputs[4];
    int nOutputs = 0;

    // position and custom vertex data are stored same buffer whereas varying data
    // exists on another buffer. "gl_NextBuffer" identifier helps to split them.
    if (numVertexElements > 0)
        outputs[nOutputs++] = "outVertexData";
    if (numVaryingElements > 0) {
        if (nOutputs > 0)
            outputs[nOutputs++] = "gl_NextBuffer";
        outputs[nOutputs++] = "outVaryingData";
    }

    glTransformFeedbackVaryings(_program, nOutputs, outputs, GL_INTERLEAVED_ATTRIBS);

    OSD_DEBUG_CHECK_GL_ERROR("Transform feedback initialize\n");

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
        return false;
    }

    glDeleteShader(shader);

    _uniformVertexBuffer         = glGetUniformLocation(_program, "vertexData");
    _uniformVaryingBuffer        = glGetUniformLocation(_program, "varyingData");

    _subComputeFace = glGetSubroutineIndex(_program, GL_VERTEX_SHADER, "catmarkComputeFace");
    _subComputeEdge = glGetSubroutineIndex(_program, GL_VERTEX_SHADER, "catmarkComputeEdge");
    _subComputeBilinearEdge = glGetSubroutineIndex(_program, GL_VERTEX_SHADER, "bilinearComputeEdge");
    _subComputeVertex = glGetSubroutineIndex(_program, GL_VERTEX_SHADER, "bilinearComputeVertex");
    _subComputeVertexA = glGetSubroutineIndex(_program, GL_VERTEX_SHADER, "catmarkComputeVertexA");
    _subComputeCatmarkVertexB = glGetSubroutineIndex(_program, GL_VERTEX_SHADER, "catmarkComputeVertexB");
    _subComputeLoopVertexB = glGetSubroutineIndex(_program, GL_VERTEX_SHADER, "loopComputeVertexB");

    _uniformVertexPass   = glGetUniformLocation(_program, "vertexPass");
    _uniformVertexOffset = glGetUniformLocation(_program, "vertexOffset");
    _uniformTableOffset  = glGetUniformLocation(_program, "tableOffset");
    _uniformIndexStart   = glGetUniformLocation(_program, "indexStart");

    _uniformTables[FarSubdivisionTables<OsdVertex>::F_IT]  = glGetUniformLocation(_program, "_F0_IT");
    _uniformTables[FarSubdivisionTables<OsdVertex>::F_ITa] = glGetUniformLocation(_program, "_F0_ITa");
    _uniformTables[FarSubdivisionTables<OsdVertex>::E_IT]  = glGetUniformLocation(_program, "_E0_IT");
    _uniformTables[FarSubdivisionTables<OsdVertex>::V_IT]  = glGetUniformLocation(_program, "_V0_IT");
    _uniformTables[FarSubdivisionTables<OsdVertex>::V_ITa] = glGetUniformLocation(_program, "_V0_ITa");
    _uniformTables[FarSubdivisionTables<OsdVertex>::E_W]   = glGetUniformLocation(_program, "_E0_S");
    _uniformTables[FarSubdivisionTables<OsdVertex>::V_W]   = glGetUniformLocation(_program, "_V0_S");

    // set unfiorm locations for edit
    _subEditAdd               = glGetSubroutineIndex(_program,
                                                     GL_VERTEX_SHADER, "editAdd");

    _uniformEditPrimVarOffset = glGetUniformLocation(_program, "editPrimVarOffset");
    _uniformEditPrimVarWidth  = glGetUniformLocation(_program, "editPrimVarWidth");

    _uniformEditIndices       = glGetUniformLocation(_program, "_editIndices");
    _uniformEditValues        = glGetUniformLocation(_program, "_editValues");
    _uniformVertexBufferImage = glGetUniformLocation(_program, "_vertexBufferImage");

    return true;
}

void
OsdGLSLTransformFeedbackKernelBundle::transformGpuBufferData(
    GLuint vertexBuffer, int numVertexElements,
    GLuint varyingBuffer, int numVaryingElements,
    int vertexOffset, int tableOffset, int start, int end) const {

    int count = end - start;
    if (count <= 0) return;

    // set batch range
    glUniform1i(_uniformIndexStart, start);
    glUniform1i(_uniformVertexOffset, vertexOffset);
    glUniform1i(_uniformTableOffset, tableOffset);
    // XXX: end is not used here now
    OSD_DEBUG_CHECK_GL_ERROR("Uniform index set at offset=%d. start=%d\n",
                             vertexOffset, start);

    // set transform feedback buffer
    if (vertexBuffer) {
        int vertexStride = numVertexElements*sizeof(float);
        glBindBufferRange(GL_TRANSFORM_FEEDBACK_BUFFER, 0, vertexBuffer,
                          (start + vertexOffset)*vertexStride, count*vertexStride);
    }

    if (varyingBuffer){
        int varyingStride = numVaryingElements*sizeof(float);
        glBindBufferRange(GL_TRANSFORM_FEEDBACK_BUFFER, 1, varyingBuffer,
                          (start + vertexOffset)*varyingStride, count*varyingStride);
    }

    OSD_DEBUG_CHECK_GL_ERROR("transformGpuBufferData glBindBufferRange\n");

    glBeginTransformFeedback(GL_POINTS);

    OSD_DEBUG_CHECK_GL_ERROR("transformGpuBufferData glBeginTransformFeedback\n");

    // draw array -----------------------------------------
    glDrawArrays(GL_POINTS, 0, count);

    OSD_DEBUG_CHECK_GL_ERROR("transformGpuBufferData DrawArray (%d)\n", count);

    glEndTransformFeedback();
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, 0);

    GLsync sync = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    glWaitSync(sync, 0, GL_TIMEOUT_IGNORED);
    glDeleteSync(sync);
}


void
OsdGLSLTransformFeedbackKernelBundle::ApplyBilinearFaceVerticesKernel(
    GLuint vertexBuffer, int numVertexElements,
    GLuint varyingBuffer, int numVaryingElements,
    int vertexOffset, int tableOffset, int start, int end) {

    glUniformSubroutinesuiv(GL_VERTEX_SHADER, 1, &_subComputeFace);
    transformGpuBufferData(vertexBuffer, numVertexElements,
                           varyingBuffer, numVaryingElements,
                           vertexOffset, tableOffset, start, end);
}

void
OsdGLSLTransformFeedbackKernelBundle::ApplyBilinearEdgeVerticesKernel(
    GLuint vertexBuffer, int numVertexElements,
    GLuint varyingBuffer, int numVaryingElements,
    int vertexOffset, int tableOffset, int start, int end) {

    glUniformSubroutinesuiv(GL_VERTEX_SHADER, 1, &_subComputeBilinearEdge);
    transformGpuBufferData(vertexBuffer, numVertexElements,
                           varyingBuffer, numVaryingElements,
                           vertexOffset, tableOffset, start, end);
}

void
OsdGLSLTransformFeedbackKernelBundle::ApplyBilinearVertexVerticesKernel(
    GLuint vertexBuffer, int numVertexElements,
    GLuint varyingBuffer, int numVaryingElements,
    int vertexOffset, int tableOffset, int start, int end) {

    glUniformSubroutinesuiv(GL_VERTEX_SHADER, 1, &_subComputeVertex);
    transformGpuBufferData(vertexBuffer, numVertexElements,
                           varyingBuffer, numVaryingElements,
                           vertexOffset, tableOffset, start, end);
}


void
OsdGLSLTransformFeedbackKernelBundle::ApplyCatmarkFaceVerticesKernel(
    GLuint vertexBuffer, int numVertexElements,
    GLuint varyingBuffer, int numVaryingElements,
    int vertexOffset, int tableOffset, int start, int end) {

    glUniformSubroutinesuiv(GL_VERTEX_SHADER, 1, &_subComputeFace);
    transformGpuBufferData(vertexBuffer, numVertexElements,
                           varyingBuffer, numVaryingElements,
                           vertexOffset, tableOffset, start, end);
}

void
OsdGLSLTransformFeedbackKernelBundle::ApplyCatmarkEdgeVerticesKernel(
    GLuint vertexBuffer, int numVertexElements,
    GLuint varyingBuffer, int numVaryingElements,
    int vertexOffset, int tableOffset, int start, int end) {

    glUniformSubroutinesuiv(GL_VERTEX_SHADER, 1, &_subComputeEdge);
    transformGpuBufferData(vertexBuffer, numVertexElements,
                           varyingBuffer, numVaryingElements,
                           vertexOffset, tableOffset, start, end);
}

void
OsdGLSLTransformFeedbackKernelBundle::ApplyCatmarkVertexVerticesKernelB(
    GLuint vertexBuffer, int numVertexElements,
    GLuint varyingBuffer, int numVaryingElements,
    int vertexOffset, int tableOffset, int start, int end) {

    glUniformSubroutinesuiv(GL_VERTEX_SHADER, 1, &_subComputeCatmarkVertexB);
    transformGpuBufferData(vertexBuffer, numVertexElements,
                           varyingBuffer, numVaryingElements,
                           vertexOffset, tableOffset, start, end);
}

void
OsdGLSLTransformFeedbackKernelBundle::ApplyCatmarkVertexVerticesKernelA(
    GLuint vertexBuffer, int numVertexElements,
    GLuint varyingBuffer, int numVaryingElements,
    int vertexOffset, int tableOffset, int start, int end, bool pass) {

    glUniformSubroutinesuiv(GL_VERTEX_SHADER, 1, &_subComputeVertexA);
    glUniform1i(_uniformVertexPass, pass ? 1 : 0);
    transformGpuBufferData(vertexBuffer, numVertexElements,
                           varyingBuffer, numVaryingElements,
                           vertexOffset, tableOffset, start, end);
}

void
OsdGLSLTransformFeedbackKernelBundle::ApplyLoopEdgeVerticesKernel(
    GLuint vertexBuffer, int numVertexElements,
    GLuint varyingBuffer, int numVaryingElements,
    int vertexOffset, int tableOffset, int start, int end) {

    glUniformSubroutinesuiv(GL_VERTEX_SHADER, 1, &_subComputeEdge);
    transformGpuBufferData(vertexBuffer, numVertexElements,
                           varyingBuffer, numVaryingElements,
                           vertexOffset, tableOffset, start, end);
}

void
OsdGLSLTransformFeedbackKernelBundle::ApplyLoopVertexVerticesKernelB(
    GLuint vertexBuffer, int numVertexElements,
    GLuint varyingBuffer, int numVaryingElements,
    int vertexOffset, int tableOffset, int start, int end) {

    glUniformSubroutinesuiv(GL_VERTEX_SHADER, 1, &_subComputeLoopVertexB);
    transformGpuBufferData(vertexBuffer, numVertexElements,
                           varyingBuffer, numVaryingElements,
                           vertexOffset, tableOffset, start, end);
}

void
OsdGLSLTransformFeedbackKernelBundle::ApplyLoopVertexVerticesKernelA(
    GLuint vertexBuffer, int numVertexElements,
    GLuint varyingBuffer, int numVaryingElements,
    int vertexOffset, int tableOffset, int start, int end, bool pass) {

    glUniformSubroutinesuiv(GL_VERTEX_SHADER, 1, &_subComputeVertexA);
    glUniform1i(_uniformVertexPass, pass ? 1 : 0);
    transformGpuBufferData(vertexBuffer, numVertexElements,
                           varyingBuffer, numVaryingElements,
                           vertexOffset, tableOffset, start, end);
}

void
OsdGLSLTransformFeedbackKernelBundle::ApplyEditAdd(
    GLuint vertexBuffer, int numVertexElements,
    GLuint varyingBuffer, int numVaryingElements,
    int primvarOffset, int primvarWidth,
    int vertexOffset, int tableOffset, int start, int end) {
    
    if (end - start <= 0) return;
    glUniformSubroutinesuiv(GL_VERTEX_SHADER, 1, &_subEditAdd);
    glUniform1i(_uniformEditPrimVarOffset, primvarOffset);
    glUniform1i(_uniformEditPrimVarWidth, primvarWidth);

    glUniform1i(_uniformIndexStart, start);
    glUniform1i(_uniformVertexOffset, vertexOffset);
    glUniform1i(_uniformTableOffset, tableOffset);
    glDrawArrays(GL_POINTS, 0, end - start);
}

void
OsdGLSLTransformFeedbackKernelBundle::UseProgram() const
{
    glUseProgram(_program);
}


}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
