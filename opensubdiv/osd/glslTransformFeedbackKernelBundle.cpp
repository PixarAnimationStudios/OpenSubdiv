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
#include <string>
#include <sstream>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

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

OsdGLSLTransformFeedbackKernelBundle::OsdGLSLTransformFeedbackKernelBundle()
    : _program(0),
      _numVertexElements(0),
      _vertexStride(0),
      _numVaryingElements(0),
      _varyingStride(0),
      _interleaved(false) {
}

OsdGLSLTransformFeedbackKernelBundle::~OsdGLSLTransformFeedbackKernelBundle() {
    if (_program)
        glDeleteProgram(_program);
}

bool
OsdGLSLTransformFeedbackKernelBundle::Compile(
    OsdVertexBufferDescriptor const &vertexDesc,
    OsdVertexBufferDescriptor const &varyingDesc,
    bool interleaved) {

    _numVertexElements = vertexDesc.length;
    _vertexStride = vertexDesc.stride;
    _numVaryingElements = varyingDesc.length;
    _varyingStride = varyingDesc.stride;
    _interleaved = interleaved;

    // modulo of vbo offset
    _vertexOffsetMod = (_vertexStride ? vertexDesc.offset % _vertexStride : 0);
    _varyingOffsetMod = (_varyingStride ? varyingDesc.offset % _varyingStride : 0);

    _program = glCreateProgram();

    GLuint shader = glCreateShader(GL_VERTEX_SHADER);

    std::ostringstream defines;
    defines << "#define NUM_VERTEX_ELEMENTS "  << _numVertexElements << "\n"
            << "#define VERTEX_STRIDE "        << _vertexStride << "\n"
            << "#define NUM_VARYING_ELEMENTS " << _numVaryingElements << "\n"
            << "#define VARYING_STRIDE "       << _varyingStride << "\n";
    std::string defineStr = defines.str();

    const char *shaderSources[3];
    shaderSources[0] = defineStr.c_str();
    shaderSources[1] = shaderDefines;
    shaderSources[2] = shaderSource;
    glShaderSource(shader, 3, shaderSources, NULL);
    glCompileShader(shader);
    glAttachShader(_program, shader);

    std::vector<std::string> outputs;

    /*
      output attribute array

      - interleaved
      outVertexData[0]
      outVertexData[1]
      outVertexData[2]
      (gl_SkipComponents1)
      outVaryingData[0]
      outVaryingData[1]
      outVaryingData[2]
      outVaryingData[3]
      (gl_SkipComponents1)
      ...


      - non-interleaved
      outVertexData[0]
      outVertexData[1]
      outVertexData[2]
      gl_NextBuffer
      outVaryingData[0]
      outVaryingData[1]
      outVaryingData[2]
      outVaryingData[3]

     */

    if (_interleaved) {
        assert(_vertexStride == _varyingStride);
        assert(_numVertexElements + _numVaryingElements <= _vertexStride);
        char attrName[32];

        for (int i = 0; i < _vertexStride; ++i) {
            int vertexElem = i - _vertexOffsetMod;
            int varyingElem = i - _varyingOffsetMod;

            if (vertexElem >= 0 and vertexElem < _numVertexElements) {
                snprintf(attrName, 32, "outVertexData[%d]", vertexElem);
                outputs.push_back(attrName);
            } else if (varyingElem >= 0 and varyingElem <= _numVaryingElements) {
                snprintf(attrName, 32, "outVaryingData[%d]", varyingElem);
                outputs.push_back(attrName);
            } else {
                outputs.push_back("gl_SkipComponents1");
            }
        }
    } else {
        // non-interleaved
        char attrName[32];

        // vertex data (may include custom vertex data) and varying data
        // are stored into the same buffer, interleaved.
        for (int i = 0; i < _vertexOffsetMod; ++i)
            outputs.push_back("gl_SkipComponents1");
        for (int i = 0; i < _numVertexElements; ++i) {
            snprintf(attrName, 32, "outVertexData[%d]", i);
            outputs.push_back(attrName);
        }
        for (int i = _numVertexElements + _vertexOffsetMod; i < _vertexStride; ++i)
            outputs.push_back("gl_SkipComponents1");

        // varying
        if (_numVaryingElements) {
            outputs.push_back("gl_NextBuffer");
        }
        for (int i = 0; i < _varyingOffsetMod; ++i) {
            outputs.push_back("gl_SkipComponents1");
        }
        for (int i = 0; i < _numVaryingElements; ++i) {
            snprintf(attrName, 32, "outVaryingData[%d]", i);
            outputs.push_back(attrName);
        }
        for (int i = _numVaryingElements + _varyingOffsetMod; i < _varyingStride; ++i) {
            outputs.push_back("gl_SkipComponents1");
        }
    }

    // convert to char* array
    std::vector<const char *> pOutputs;
    for (size_t i = 0; i < outputs.size(); ++i) {
        pOutputs.push_back(&outputs[i][0]);
    }

    glTransformFeedbackVaryings(_program, (GLsizei)outputs.size(),
                                &pOutputs[0], GL_INTERLEAVED_ATTRIBS);

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
    _subComputeQuadFace = glGetSubroutineIndex(_program, GL_VERTEX_SHADER, "catmarkComputeQuadFace");
    _subComputeTriQuadFace = glGetSubroutineIndex(_program, GL_VERTEX_SHADER, "catmarkComputeTriQuadFace");
    _subComputeEdge = glGetSubroutineIndex(_program, GL_VERTEX_SHADER, "catmarkComputeEdge");
    _subComputeRestrictedEdge = glGetSubroutineIndex(_program, GL_VERTEX_SHADER, "catmarkComputeRestrictedEdge");
    _subComputeBilinearEdge = glGetSubroutineIndex(_program, GL_VERTEX_SHADER, "bilinearComputeEdge");
    _subComputeVertex = glGetSubroutineIndex(_program, GL_VERTEX_SHADER, "bilinearComputeVertex");
    _subComputeVertexA = glGetSubroutineIndex(_program, GL_VERTEX_SHADER, "catmarkComputeVertexA");
    _subComputeCatmarkVertexB = glGetSubroutineIndex(_program, GL_VERTEX_SHADER, "catmarkComputeVertexB");
    _subComputeRestrictedVertexA = glGetSubroutineIndex(_program, GL_VERTEX_SHADER, "catmarkComputeRestrictedVertexA");
    _subComputeRestrictedVertexB1 = glGetSubroutineIndex(_program, GL_VERTEX_SHADER, "catmarkComputeRestrictedVertexB1");
    _subComputeRestrictedVertexB2 = glGetSubroutineIndex(_program, GL_VERTEX_SHADER, "catmarkComputeRestrictedVertexB2");
    _subComputeLoopVertexB = glGetSubroutineIndex(_program, GL_VERTEX_SHADER, "loopComputeVertexB");

    _uniformVertexPass   = glGetUniformLocation(_program, "vertexPass");
    _uniformVertexOffset = glGetUniformLocation(_program, "vertexOffset");
    _uniformTableOffset  = glGetUniformLocation(_program, "tableOffset");
    _uniformIndexStart   = glGetUniformLocation(_program, "indexStart");
    _uniformVertexBaseOffset  = glGetUniformLocation(_program, "vertexBaseOffset");
    _uniformVaryingBaseOffset = glGetUniformLocation(_program, "varyingBaseOffset");

    _uniformTables[FarSubdivisionTables::F_IT]  = glGetUniformLocation(_program, "_F0_IT");
    _uniformTables[FarSubdivisionTables::F_ITa] = glGetUniformLocation(_program, "_F0_ITa");
    _uniformTables[FarSubdivisionTables::E_IT]  = glGetUniformLocation(_program, "_E0_IT");
    _uniformTables[FarSubdivisionTables::V_IT]  = glGetUniformLocation(_program, "_V0_IT");
    _uniformTables[FarSubdivisionTables::V_ITa] = glGetUniformLocation(_program, "_V0_ITa");
    _uniformTables[FarSubdivisionTables::E_W]   = glGetUniformLocation(_program, "_E0_S");
    _uniformTables[FarSubdivisionTables::V_W]   = glGetUniformLocation(_program, "_V0_S");

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
    GLuint vertexBuffer, GLuint varyingBuffer,
    int vertexOffset, int varyingOffset,
    int offset, int tableOffset, int start, int end) const {

    int count = end - start;
    if (count <= 0) return;

    // set batch range
    glUniform1i(_uniformIndexStart, start);
    glUniform1i(_uniformVertexOffset, offset);
    glUniform1i(_uniformTableOffset, tableOffset);

    // XXX: end is not used here now
    OSD_DEBUG_CHECK_GL_ERROR("Uniform index set at offset=%d. start=%d\n",
                             offset, start);

    int vertexOrigin = vertexOffset - _vertexOffsetMod;
    int varyingOrigin = varyingOffset - _varyingOffsetMod;

    // set transform feedback buffer
    if (_interleaved) {
        int vertexStride = _vertexStride*sizeof(float);
        glBindBufferRange(GL_TRANSFORM_FEEDBACK_BUFFER, 0, vertexBuffer,
                          (start + offset)*vertexStride + vertexOrigin*sizeof(float),
                          count*vertexStride);
    } else {
        if (vertexBuffer) {
            int vertexStride = _vertexStride*sizeof(float);
            glBindBufferRange(GL_TRANSFORM_FEEDBACK_BUFFER, 0, vertexBuffer,
                              (start + offset)*vertexStride + vertexOrigin*sizeof(float),
                              count*vertexStride);
        }
        if (varyingBuffer){
            int varyingStride = _varyingStride*sizeof(float);
            glBindBufferRange(GL_TRANSFORM_FEEDBACK_BUFFER, 1, varyingBuffer,
                              (start + offset)*varyingStride + varyingOrigin*sizeof(float),
                              count*varyingStride);
        }
    }

    OSD_DEBUG_CHECK_GL_ERROR("transformGpuBufferData glBindBufferRange\n");

    glBeginTransformFeedback(GL_POINTS);

    OSD_DEBUG_CHECK_GL_ERROR("transformGpuBufferData glBeginTransformFeedback\n");

    // draw array -----------------------------------------
    glDrawArrays(GL_POINTS, 0, count);

    OSD_DEBUG_CHECK_GL_ERROR("transformGpuBufferData DrawArray (%d)\n", count);

    glEndTransformFeedback();
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, 0);
}


void
OsdGLSLTransformFeedbackKernelBundle::ApplyBilinearFaceVerticesKernel(
    GLuint vertexBuffer, GLuint varyingBuffer,
    int vertexOffset, int varyingOffset,
    int offset, int tableOffset, int start, int end) {

    glUniformSubroutinesuiv(GL_VERTEX_SHADER, 1, &_subComputeFace);
    transformGpuBufferData(vertexBuffer, varyingBuffer,
                           vertexOffset, varyingOffset,
                           offset, tableOffset, start, end);
}

void
OsdGLSLTransformFeedbackKernelBundle::ApplyBilinearEdgeVerticesKernel(
    GLuint vertexBuffer, GLuint varyingBuffer,
    int vertexOffset, int varyingOffset,
    int offset, int tableOffset, int start, int end) {

    glUniformSubroutinesuiv(GL_VERTEX_SHADER, 1, &_subComputeBilinearEdge);
    transformGpuBufferData(vertexBuffer, varyingBuffer,
                           vertexOffset, varyingOffset,
                           offset, tableOffset, start, end);
}

void
OsdGLSLTransformFeedbackKernelBundle::ApplyBilinearVertexVerticesKernel(
    GLuint vertexBuffer, GLuint varyingBuffer,
    int vertexOffset, int varyingOffset,
    int offset, int tableOffset, int start, int end) {

    glUniformSubroutinesuiv(GL_VERTEX_SHADER, 1, &_subComputeVertex);
    transformGpuBufferData(vertexBuffer, varyingBuffer,
                           vertexOffset, varyingOffset,
                           offset, tableOffset, start, end);
}


void
OsdGLSLTransformFeedbackKernelBundle::ApplyCatmarkQuadFaceVerticesKernel(
    GLuint vertexBuffer, GLuint varyingBuffer,
    int vertexOffset, int varyingOffset,
    int offset, int tableOffset, int start, int end) {

    glUniformSubroutinesuiv(GL_VERTEX_SHADER, 1, &_subComputeQuadFace);
    transformGpuBufferData(vertexBuffer, varyingBuffer,
                           vertexOffset, varyingOffset,
                           offset, tableOffset, start, end);
}

void
OsdGLSLTransformFeedbackKernelBundle::ApplyCatmarkTriQuadFaceVerticesKernel(
    GLuint vertexBuffer, GLuint varyingBuffer,
    int vertexOffset, int varyingOffset,
    int offset, int tableOffset, int start, int end) {

    glUniformSubroutinesuiv(GL_VERTEX_SHADER, 1, &_subComputeTriQuadFace);
    transformGpuBufferData(vertexBuffer, varyingBuffer,
                           vertexOffset, varyingOffset,
                           offset, tableOffset, start, end);
}

void
OsdGLSLTransformFeedbackKernelBundle::ApplyCatmarkFaceVerticesKernel(
    GLuint vertexBuffer, GLuint varyingBuffer,
    int vertexOffset, int varyingOffset,
    int offset, int tableOffset, int start, int end) {

    glUniformSubroutinesuiv(GL_VERTEX_SHADER, 1, &_subComputeFace);
    transformGpuBufferData(vertexBuffer, varyingBuffer,
                           vertexOffset, varyingOffset,
                           offset, tableOffset, start, end);
}

void
OsdGLSLTransformFeedbackKernelBundle::ApplyCatmarkEdgeVerticesKernel(
    GLuint vertexBuffer, GLuint varyingBuffer,
    int vertexOffset, int varyingOffset,
    int offset, int tableOffset, int start, int end) {

    glUniformSubroutinesuiv(GL_VERTEX_SHADER, 1, &_subComputeEdge);
    transformGpuBufferData(vertexBuffer, varyingBuffer,
                           vertexOffset, varyingOffset,
                           offset, tableOffset, start, end);
}

void
OsdGLSLTransformFeedbackKernelBundle::ApplyCatmarkRestrictedEdgeVerticesKernel(
    GLuint vertexBuffer, GLuint varyingBuffer,
    int vertexOffset, int varyingOffset,
    int offset, int tableOffset, int start, int end) {

    glUniformSubroutinesuiv(GL_VERTEX_SHADER, 1, &_subComputeRestrictedEdge);
    transformGpuBufferData(vertexBuffer, varyingBuffer,
                           vertexOffset, varyingOffset,
                           offset, tableOffset, start, end);
}

void
OsdGLSLTransformFeedbackKernelBundle::ApplyCatmarkVertexVerticesKernelB(
    GLuint vertexBuffer, GLuint varyingBuffer,
    int vertexOffset, int varyingOffset,
    int offset, int tableOffset, int start, int end) {

    glUniformSubroutinesuiv(GL_VERTEX_SHADER, 1, &_subComputeCatmarkVertexB);
    transformGpuBufferData(vertexBuffer, varyingBuffer,
                           vertexOffset, varyingOffset,
                           offset, tableOffset, start, end);
}

void
OsdGLSLTransformFeedbackKernelBundle::ApplyCatmarkVertexVerticesKernelA(
    GLuint vertexBuffer, GLuint varyingBuffer,
    int vertexOffset, int varyingOffset,
    int offset, int tableOffset, int start, int end, bool pass) {

    glUniformSubroutinesuiv(GL_VERTEX_SHADER, 1, &_subComputeVertexA);
    glUniform1i(_uniformVertexPass, pass ? 1 : 0);
    transformGpuBufferData(vertexBuffer, varyingBuffer,
                           vertexOffset, varyingOffset,
                           offset, tableOffset, start, end);
}

void
OsdGLSLTransformFeedbackKernelBundle::ApplyCatmarkRestrictedVertexVerticesKernelB1(
    GLuint vertexBuffer, GLuint varyingBuffer,
    int vertexOffset, int varyingOffset,
    int offset, int tableOffset, int start, int end) {

    glUniformSubroutinesuiv(GL_VERTEX_SHADER, 1, &_subComputeRestrictedVertexB1);
    transformGpuBufferData(vertexBuffer, varyingBuffer,
                           vertexOffset, varyingOffset,
                           offset, tableOffset, start, end);
}

void
OsdGLSLTransformFeedbackKernelBundle::ApplyCatmarkRestrictedVertexVerticesKernelB2(
    GLuint vertexBuffer, GLuint varyingBuffer,
    int vertexOffset, int varyingOffset,
    int offset, int tableOffset, int start, int end) {

    glUniformSubroutinesuiv(GL_VERTEX_SHADER, 1, &_subComputeRestrictedVertexB2);
    transformGpuBufferData(vertexBuffer, varyingBuffer,
                           vertexOffset, varyingOffset,
                           offset, tableOffset, start, end);
}

void
OsdGLSLTransformFeedbackKernelBundle::ApplyCatmarkRestrictedVertexVerticesKernelA(
    GLuint vertexBuffer, GLuint varyingBuffer,
    int vertexOffset, int varyingOffset,
    int offset, int tableOffset, int start, int end) {

    glUniformSubroutinesuiv(GL_VERTEX_SHADER, 1, &_subComputeRestrictedVertexA);
    transformGpuBufferData(vertexBuffer, varyingBuffer,
                           vertexOffset, varyingOffset,
                           offset, tableOffset, start, end);
}

void
OsdGLSLTransformFeedbackKernelBundle::ApplyLoopEdgeVerticesKernel(
    GLuint vertexBuffer, GLuint varyingBuffer,
    int vertexOffset, int varyingOffset,
    int offset, int tableOffset, int start, int end) {

    glUniformSubroutinesuiv(GL_VERTEX_SHADER, 1, &_subComputeEdge);
    transformGpuBufferData(vertexBuffer, varyingBuffer,
                           vertexOffset, varyingOffset,
                           offset, tableOffset, start, end);
}

void
OsdGLSLTransformFeedbackKernelBundle::ApplyLoopVertexVerticesKernelB(
    GLuint vertexBuffer, GLuint varyingBuffer,
    int vertexOffset, int varyingOffset,
    int offset, int tableOffset, int start, int end) {

    glUniformSubroutinesuiv(GL_VERTEX_SHADER, 1, &_subComputeLoopVertexB);
    transformGpuBufferData(vertexBuffer, varyingBuffer,
                           vertexOffset, varyingOffset,
                           offset, tableOffset, start, end);
}

void
OsdGLSLTransformFeedbackKernelBundle::ApplyLoopVertexVerticesKernelA(
    GLuint vertexBuffer, GLuint varyingBuffer,
    int vertexOffset, int varyingOffset,
    int offset, int tableOffset, int start, int end, bool pass) {

    glUniformSubroutinesuiv(GL_VERTEX_SHADER, 1, &_subComputeVertexA);
    glUniform1i(_uniformVertexPass, pass ? 1 : 0);
    transformGpuBufferData(vertexBuffer, varyingBuffer,
                           vertexOffset, varyingOffset,
                           offset, tableOffset, start, end);
}

void
OsdGLSLTransformFeedbackKernelBundle::ApplyEditAdd(
    GLuint /* vertexBuffer */, GLuint /* varyingBuffer */,
    int /* vertexOffset */, int /* varyingOffset */,
    int primvarOffset, int primvarWidth,
    int offset, int tableOffset, int start, int end) {
    
    if (end - start <= 0) return;
    glUniformSubroutinesuiv(GL_VERTEX_SHADER, 1, &_subEditAdd);
    glUniform1i(_uniformEditPrimVarOffset, primvarOffset);
    glUniform1i(_uniformEditPrimVarWidth, primvarWidth);

    glUniform1i(_uniformIndexStart, start);
    glUniform1i(_uniformVertexOffset, offset);
    glUniform1i(_uniformTableOffset, tableOffset);
    glDrawArrays(GL_POINTS, 0, end - start);
}

void
OsdGLSLTransformFeedbackKernelBundle::UseProgram(int vertexBaseOffset,
                                                 int varyingBaseOffset) const
{
    glUseProgram(_program);

    glUniform1i(_uniformVertexBaseOffset, vertexBaseOffset);
    glUniform1i(_uniformVaryingBaseOffset, varyingBaseOffset);
}


}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
