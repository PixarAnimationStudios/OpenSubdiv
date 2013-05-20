//
//     Copyright (C) Pixar. All rights reserved.
//
//     This license governs use of the accompanying software. If you
//     use the software, you accept this license. If you do not accept
//     the license, do not use the software.
//
//     1. Definitions
//     The terms "reproduce," "reproduction," "derivative works," and
//     "distribution" have the same meaning here as under U.S.
//     copyright law.  A "contribution" is the original software, or
//     any additions or changes to the software.
//     A "contributor" is any person or entity that distributes its
//     contribution under this license.
//     "Licensed patents" are a contributor's patent claims that read
//     directly on its contribution.
//
//     2. Grant of Rights
//     (A) Copyright Grant- Subject to the terms of this license,
//     including the license conditions and limitations in section 3,
//     each contributor grants you a non-exclusive, worldwide,
//     royalty-free copyright license to reproduce its contribution,
//     prepare derivative works of its contribution, and distribute
//     its contribution or any derivative works that you create.
//     (B) Patent Grant- Subject to the terms of this license,
//     including the license conditions and limitations in section 3,
//     each contributor grants you a non-exclusive, worldwide,
//     royalty-free license under its licensed patents to make, have
//     made, use, sell, offer for sale, import, and/or otherwise
//     dispose of its contribution in the software or derivative works
//     of the contribution in the software.
//
//     3. Conditions and Limitations
//     (A) No Trademark License- This license does not grant you
//     rights to use any contributor's name, logo, or trademarks.
//     (B) If you bring a patent claim against any contributor over
//     patents that you claim are infringed by the software, your
//     patent license from such contributor to the software ends
//     automatically.
//     (C) If you distribute any portion of the software, you must
//     retain all copyright, patent, trademark, and attribution
//     notices that are present in the software.
//     (D) If you distribute any portion of the software in source
//     code form, you may do so only under this license by including a
//     complete copy of this license with your distribution. If you
//     distribute any portion of the software in compiled or object
//     code form, you may only do so under a license that complies
//     with this license.
//     (E) The software is licensed "as-is." You bear the risk of
//     using it. The contributors give no express warranties,
//     guarantees or conditions. You may have additional consumer
//     rights under your local laws which this license cannot change.
//     To the extent permitted under your local laws, the contributors
//     exclude the implied warranties of merchantability, fitness for
//     a particular purpose and non-infringement.
//

#if defined(__APPLE__)
    #include <OpenGL/gl3.h>
#else
    #include <GL/glew.h>
#endif

#include <stdio.h>
#ifdef _MSC_VER
#define snprintf _snprintf
#endif

#include "../osd/debug.h"
#include "../osd/error.h"
#include "../osd/glslTransformFeedbackKernelBundle.h"
#include "../osd/vertex.h"

#include "../far/subdivisionTables.h"

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
    outputs[nOutputs++] = "outPosition";

    // position and custom vertex data are stored same buffer whereas varying data
    // exists on another buffer. "gl_NextBuffer" identifier helps to split them.
    if (numVertexElements > 3) {
        outputs[nOutputs++] = "outVertexData";
    }
    if (numVaryingElements > 0) {
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

    _uniformVertexBuffer         = glGetUniformLocation(_program, "vertex");
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
                             offset, start);

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
