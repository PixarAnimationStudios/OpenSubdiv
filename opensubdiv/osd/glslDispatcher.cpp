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
#include "../version.h"
#include "../osd/glslDispatcher.h"
#include "../osd/local.h"

#include <GL/glew.h>
#include <stdlib.h>
#include <string.h>
#include <functional>
#include <algorithm>

#define OPT_E0_IT_VEC4
#define OPT_E0_S_VEC2

#ifdef _MSC_VER
#define snprintf _snprintf
#endif

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

static const char *shaderSource =
#include "../osd/glslKernel.inc"
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

std::vector<OsdGlslKernelDispatcher::ComputeShader> OsdGlslKernelDispatcher::shaderRegistry;
    
OsdGlslKernelDispatcher::OsdGlslKernelDispatcher(int levels)
    : OsdKernelDispatcher(levels)
{
    _currentVertexBuffer = 0;
    _currentVaryingBuffer = 0;
    _shader = 0;

    glGenTextures(1, &_vertexTexture);
    glGenTextures(1, &_varyingTexture);

    _tableBuffers.resize(TABLE_MAX);
    _tableTextures.resize(TABLE_MAX);

    glGenBuffers(TABLE_MAX, &_tableBuffers[0]);
    glGenTextures(TABLE_MAX, &_tableTextures[0]);

}

OsdGlslKernelDispatcher::~OsdGlslKernelDispatcher() {

    glDeleteTextures(1, &_vertexTexture);
    glDeleteTextures(1, &_varyingTexture);

    glDeleteBuffers(TABLE_MAX, &_tableBuffers[0]);
}

void
OsdGlslKernelDispatcher::CopyTable(int tableIndex, size_t size, const void *ptr) {

    glBindBuffer(GL_ARRAY_BUFFER, _tableBuffers[tableIndex]);
    glBufferData(GL_ARRAY_BUFFER, size, ptr, GL_STATIC_DRAW);
    CHECK_GL_ERROR("UpdateTable tableIndex %d, size %ld, buffer =%d\n",
        tableIndex, size, _tableBuffers[tableIndex]);
}

void
OsdGlslKernelDispatcher::OnKernelLaunch() {

    glEnable(GL_RASTERIZER_DISCARD);
    _shader->UseProgram();

//XXX what if loop..
    bindTextureBuffer(_shader->GetTableUniform(F_IT),  _tableBuffers[F_IT],
                      _tableTextures[F_IT],  GL_R32UI, 2);
    bindTextureBuffer(_shader->GetTableUniform(F_ITa), _tableBuffers[F_ITa],
                      _tableTextures[F_ITa], GL_R32I,  3);

#ifdef OPT_E0_IT_VEC4
    bindTextureBuffer(_shader->GetTableUniform(E_IT),  _tableBuffers[E_IT],
                      _tableTextures[E_IT],  GL_RGBA32UI, 4);
#else
    bindTextureBuffer(_shader->GetTableUniform(E_IT),  _tableBuffers[E_IT],
                      _tableTextures[E_IT],  GL_R32UI, 4);
#endif

#ifdef OPT_CATMARK_V_IT_VEC2
    bindTextureBuffer(_shader->GetTableUniform(V_IT),  _tableBuffers[V_IT],
                      _tableTextures[V_IT],  GL_RG32UI, 5);
#else
    bindTextureBuffer(_shader->GetTableUniform(V_IT),  _tableBuffers[V_IT],
                      _tableTextures[V_IT],  GL_R32UI, 5);
#endif
    bindTextureBuffer(_shader->GetTableUniform(V_ITa), _tableBuffers[V_ITa],
                      _tableTextures[V_ITa], GL_R32I,  6);
#ifdef OPT_E0_S_VEC2
    bindTextureBuffer(_shader->GetTableUniform(E_W),   _tableBuffers[E_W],
                      _tableTextures[E_W],   GL_RG32F,  7);
#else
    bindTextureBuffer(_shader->GetTableUniform(E_W),   _tableBuffers[E_W],
                      _tableTextures[E_W],   GL_R32F,  7);
#endif
    bindTextureBuffer(_shader->GetTableUniform(V_W),   _tableBuffers[V_W],
                      _tableTextures[V_W],   GL_R32F,  8);

}

void
OsdGlslKernelDispatcher::OnKernelFinish() {

    unbindTextureBuffer(2);
    unbindTextureBuffer(3);
    unbindTextureBuffer(4);
    unbindTextureBuffer(5);
    unbindTextureBuffer(6);
    unbindTextureBuffer(7);
    unbindTextureBuffer(8);

    glDisable(GL_RASTERIZER_DISCARD);
    glUseProgram(0);
}

OsdVertexBuffer *
OsdGlslKernelDispatcher::InitializeVertexBuffer(int numElements, int numVertices)
{
    return new OsdGpuVertexBuffer(numElements, numVertices);
}

void
OsdGlslKernelDispatcher::BindVertexBuffer(OsdVertexBuffer *vertex, OsdVertexBuffer *varying) {

    if (vertex)
        _currentVertexBuffer = dynamic_cast<OsdGpuVertexBuffer *>(vertex);
    else 
        _currentVertexBuffer = NULL;

    if (varying)
        _currentVaryingBuffer = dynamic_cast<OsdGpuVertexBuffer *>(varying);
    else
        _currentVaryingBuffer = NULL;

    int numVertexElements = vertex ? vertex->GetNumElements() : 0;
    int numVaryingElements = varying ? varying->GetNumElements() : 0;

    // find appropriate shader program from registry (compile it if needed)
    std::vector<ComputeShader>::iterator it =
        std::find_if(shaderRegistry.begin(), shaderRegistry.end(),
                     ComputeShader::Match(numVertexElements, numVaryingElements));

    _shader = NULL;
    if (it != shaderRegistry.end()) {
        _shader = &(*it);
    } else {
        shaderRegistry.push_back(ComputeShader());
        _shader = &shaderRegistry.back();
        _shader->Compile(numVertexElements, numVaryingElements);
    }

    _shader->UseProgram(); // need to bind textures

    // bind vertex texture
    if (_currentVertexBuffer) {
        bindTextureBuffer(_shader->GetVertexUniform(), _currentVertexBuffer->GetGpuBuffer(), _vertexTexture, GL_RGB32F, 0);
    }

    if (_currentVaryingBuffer) {
        bindTextureBuffer(_shader->GetVaryingUniform(), _currentVaryingBuffer->GetGpuBuffer(), _varyingTexture, GL_R32F, 1);
    }

#if 0  // experiment to use image store function
    glActiveTexture(GL_TEXTURE0 + 0);
    glBindImageTextureEXT(0, _vertexTexture, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F);

    if (_numVarying > 0) {
        glBindImageTextureEXT(1, _vertexTexture, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F);
    }
#endif

    CHECK_GL_ERROR("BindVertexBuffer \n");

}

void
OsdGlslKernelDispatcher::UnbindVertexBuffer()
{
    if (_currentVertexBuffer) {
        unbindTextureBuffer(0);
    }
    if (_currentVaryingBuffer) {
        unbindTextureBuffer(1);
    }
    _currentVertexBuffer = NULL;
    _currentVaryingBuffer = NULL;
}

void
OsdGlslKernelDispatcher::Synchronize() {
    glFinish();
}

void
OsdGlslKernelDispatcher::bindTextureBuffer(
    GLuint sampler, GLuint buffer, GLuint texture, GLenum type, int unit) const {

    if (sampler == -1) {
        OSD_ERROR("BindTextureError:: sampler = %d\n", sampler);
        return;
    }
    OSD_DEBUG("BindTextureBuffer unit=%d, sampler=%d, buffer=%d, texture = %d, E%x\n", unit, sampler, buffer, texture, glGetError());

    glUniform1i(sampler, unit);
    CHECK_GL_ERROR("BindTextureBuffer glUniform %d\n", unit);
    glActiveTexture(GL_TEXTURE0 + unit);
    CHECK_GL_ERROR("BindTextureBuffer glActiveTexture %d\n", unit);
    glBindTexture(GL_TEXTURE_BUFFER, texture);
    CHECK_GL_ERROR("BindTextureBuffer glBindTexture %d\n", texture);
    glTexBuffer(GL_TEXTURE_BUFFER, type, buffer);
    CHECK_GL_ERROR("BindTextureBuffer glTexBuffer\n");
    glActiveTexture(GL_TEXTURE0);
}


void
OsdGlslKernelDispatcher::unbindTextureBuffer(int unit) const {

    glActiveTexture(GL_TEXTURE0 + unit);
    glBindTexture(GL_TEXTURE_BUFFER, 0);
}

void
OsdGlslKernelDispatcher::ApplyCatmarkFaceVerticesKernel(
    FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const {

    _shader->ApplyCatmarkFaceVerticesKernel(_currentVertexBuffer, _currentVaryingBuffer,
                                            _tableOffsets[F_IT][level-1],
                                            _tableOffsets[F_ITa][level-1],
                                            offset, start, end);
}

void
OsdGlslKernelDispatcher::ApplyCatmarkEdgeVerticesKernel(
    FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const {
    
    _shader->ApplyCatmarkEdgeVerticesKernel(_currentVertexBuffer, _currentVaryingBuffer,
                                            _tableOffsets[E_IT][level-1],
                                            _tableOffsets[E_W][level-1],
                                            offset, start, end);
}

void
OsdGlslKernelDispatcher::ApplyCatmarkVertexVerticesKernelB(
    FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const {
    
    _shader->ApplyCatmarkVertexVerticesKernelB(_currentVertexBuffer, _currentVaryingBuffer,
                                               _tableOffsets[V_IT][level-1],
                                               _tableOffsets[V_ITa][level-1],
                                               _tableOffsets[V_W][level-1],
                                               offset, start, end);
}

void
OsdGlslKernelDispatcher::ApplyCatmarkVertexVerticesKernelA(
    FarMesh<OsdVertex> * mesh, int offset, bool pass, int level, int start, int end, void * data) const {
    
    _shader->ApplyCatmarkVertexVerticesKernelA(_currentVertexBuffer, _currentVaryingBuffer,
                                               _tableOffsets[V_ITa][level-1],
                                               _tableOffsets[V_W][level-1],
                                               offset, pass, start, end);
}

void
OsdGlslKernelDispatcher::ApplyLoopEdgeVerticesKernel(
    FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const {
    
    _shader->ApplyLoopEdgeVerticesKernel(_currentVertexBuffer, _currentVaryingBuffer,
                                         _tableOffsets[E_IT][level-1],
                                         _tableOffsets[E_W][level-1],
                                         offset, start, end);
}

void
OsdGlslKernelDispatcher::ApplyLoopVertexVerticesKernelB(
    FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const {
    
    _shader->ApplyLoopVertexVerticesKernelB(_currentVertexBuffer, _currentVaryingBuffer,
                                            _tableOffsets[V_IT][level-1],
                                            _tableOffsets[V_ITa][level-1],
                                            _tableOffsets[V_W][level-1],
                                            offset, start, end);
}

void
OsdGlslKernelDispatcher::ApplyLoopVertexVerticesKernelA(
    FarMesh<OsdVertex> * mesh, int offset, bool pass, int level, int start, int end, void * data) const {
    
    _shader->ApplyLoopVertexVerticesKernelA(_currentVertexBuffer, _currentVaryingBuffer,
                                            _tableOffsets[V_ITa][level-1],
                                            _tableOffsets[V_W][level-1],
                                            offset, pass, start, end);
}

// -------------------------------------------------------------------------------

OsdGlslKernelDispatcher::ComputeShader::ComputeShader() :
    _program(0)
{
}

OsdGlslKernelDispatcher::ComputeShader::~ComputeShader()
{
    if (_program) 
        glDeleteProgram(_program);
}

bool
OsdGlslKernelDispatcher::ComputeShader::Compile(int numVertexElements, int numVaryingElements) {

    // XXX: NOTE: current GLSL only supports numVertexElements = 6!!!
    assert(numVertexElements == 6);

    _numVertexElements = numVertexElements;
    _numVaryingElements = numVaryingElements;
    _program = glCreateProgram();

    GLuint shader = glCreateShader(GL_VERTEX_SHADER);

    char constantDefine[256];
    snprintf(constantDefine, 256,
             "#define NUM_VARYING_ELEMENTS %d\n", numVaryingElements);

    const char *shaderSources[3];
    shaderSources[0] = constantDefine;
    shaderSources[1] = shaderDefines;
    shaderSources[2] = shaderSource;
    glShaderSource(shader, 3, shaderSources, NULL);
    glCompileShader(shader);
    glAttachShader(_program, shader);

    const char *outputs[] = { "outPosition", 
                              "outNormal", 
			      "gl_NextBuffer", 
			      "outVaryingData" };

    int nOutputs = numVaryingElements > 0 ? 4 : 2;

    glTransformFeedbackVaryings(_program, nOutputs, outputs, GL_INTERLEAVED_ATTRIBS);

    CHECK_GL_ERROR("Transform feedback initialize \n");

    GLint linked = 0;
    glLinkProgram(_program);
    glGetProgramiv(_program, GL_LINK_STATUS, &linked);

    if (linked == GL_FALSE) {
        OSD_ERROR("Fail to link shader\n");

        char buffer[1024];
        glGetShaderInfoLog(shader, 1024, NULL, buffer);
        OSD_ERROR(buffer);

        glGetProgramInfoLog(_program, 1024, NULL, buffer);
        OSD_ERROR(buffer);
        
        glDeleteProgram(_program);
        _program = 0;
        // XXX ERROR HANDLE
        return false;
    }

    glDeleteShader(shader);

    _vertexUniform         = glGetUniformLocation(_program, "vertex");
    _varyingUniform        = glGetUniformLocation(_program, "varyingData");

    _subComputeFace = glGetSubroutineIndex(_program, GL_VERTEX_SHADER, "catmarkComputeFace");
    _subComputeEdge = glGetSubroutineIndex(_program, GL_VERTEX_SHADER, "catmarkComputeEdge");
    _subComputeVertexA = glGetSubroutineIndex(_program, GL_VERTEX_SHADER, "catmarkComputeVertexA");
    _subComputeVertexB = glGetSubroutineIndex(_program, GL_VERTEX_SHADER, "catmarkComputeVertexB");
    _subComputeLoopVertexB = glGetSubroutineIndex(_program, GL_VERTEX_SHADER, "loopComputeVertexB");

    _uniformVertexPass = glGetUniformLocation(_program, "vertexPass");
    _uniformIndexStart = glGetUniformLocation(_program, "indexStart");
    _uniformIndexOffset = glGetUniformLocation(_program, "indexOffset");

    _tableUniforms.resize(TABLE_MAX);
    _tableOffsetUniforms.resize(TABLE_MAX);

    _tableUniforms[F_IT]  = glGetUniformLocation(_program, "_F0_IT");
    _tableUniforms[F_ITa] = glGetUniformLocation(_program, "_F0_ITa");
    _tableUniforms[E_IT]  = glGetUniformLocation(_program, "_E0_IT");
    _tableUniforms[V_IT]  = glGetUniformLocation(_program, "_V0_IT");
    _tableUniforms[V_ITa] = glGetUniformLocation(_program, "_V0_ITa");
    _tableUniforms[E_W]   = glGetUniformLocation(_program, "_E0_S");
    _tableUniforms[V_W]   = glGetUniformLocation(_program, "_V0_S");
    _tableOffsetUniforms[F_IT]  = glGetUniformLocation(_program, "F_IT_ofs");
    _tableOffsetUniforms[F_ITa] = glGetUniformLocation(_program, "F_ITa_ofs");
    _tableOffsetUniforms[E_IT]  = glGetUniformLocation(_program, "E_IT_ofs");
    _tableOffsetUniforms[V_IT]  = glGetUniformLocation(_program, "V_IT_ofs");
    _tableOffsetUniforms[V_ITa] = glGetUniformLocation(_program, "V_ITa_ofs");
    _tableOffsetUniforms[E_W]   = glGetUniformLocation(_program, "E_W_ofs");
    _tableOffsetUniforms[V_W]   = glGetUniformLocation(_program, "V_W_ofs");

    return true;
}

void
OsdGlslKernelDispatcher::ComputeShader::transformGpuBufferData(OsdGpuVertexBuffer *vertexBuffer, OsdGpuVertexBuffer *varyingBuffer,
                                                               GLint offset, int start, int end) const {
    int count = end - start;
    if (count <= 0) return;
    OSD_DEBUG("_transformGpuBufferData offset=%d, count=%d\n", glGetError(), offset, count);

    // set batch range
    glUniform1i(_uniformIndexStart, start);
    glUniform1i(_uniformIndexOffset, offset);
    // XXX: end is not used here now
    CHECK_GL_ERROR("Uniform index set at offset=%d. start=%d\n", offset, start);

    // set transform feedback buffer
    if (vertexBuffer) {
        int vertexStride = vertexBuffer->GetNumElements()*sizeof(float);
        glBindBufferRange(GL_TRANSFORM_FEEDBACK_BUFFER, 0, vertexBuffer->GetGpuBuffer(),
                          (start+offset)*vertexStride, count*vertexStride);
    }

    if (varyingBuffer){
        int varyingStride = varyingBuffer->GetNumElements()*sizeof(float);
        glBindBufferRange(GL_TRANSFORM_FEEDBACK_BUFFER, 1, varyingBuffer->GetGpuBuffer(),
                          (start+offset)*varyingStride, count*varyingStride);
    }

    CHECK_GL_ERROR("transformGpuBufferData glBindBufferRange\n");

    glBeginTransformFeedback(GL_POINTS);

    CHECK_GL_ERROR("transformGpuBufferData glBeginTransformFeedback\n");
    
    // draw array -----------------------------------------
    glDrawArrays(GL_POINTS, 0, count);
    CHECK_GL_ERROR("transformGpuBufferData DrawArray (%d)\n", count);

    glEndTransformFeedback();
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, 0);

    GLsync sync = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    glWaitSync(sync, 0, GL_TIMEOUT_IGNORED);
    glDeleteSync(sync);
}

void
OsdGlslKernelDispatcher::ComputeShader::ApplyCatmarkFaceVerticesKernel(
    OsdGpuVertexBuffer *vertex, OsdGpuVertexBuffer *varying,
    int F_IT_ofs, int F_ITa_ofs, int offset, int start, int end) {

    glUniformSubroutinesuiv(GL_VERTEX_SHADER, 1, &_subComputeFace);
    glUniform1i(_tableOffsetUniforms[F_IT], F_IT_ofs);
    glUniform1i(_tableOffsetUniforms[F_ITa], F_ITa_ofs);
    transformGpuBufferData(vertex, varying, offset, start, end);
}

void
OsdGlslKernelDispatcher::ComputeShader::ApplyCatmarkEdgeVerticesKernel(
    OsdGpuVertexBuffer *vertex, OsdGpuVertexBuffer *varying,
    int E_IT_ofs, int E_W_ofs, int offset, int start, int end) {

    glUniformSubroutinesuiv(GL_VERTEX_SHADER, 1, &_subComputeEdge);
    glUniform1i(_tableOffsetUniforms[E_IT], E_IT_ofs);
    glUniform1i(_tableOffsetUniforms[E_W], E_W_ofs);
    transformGpuBufferData(vertex, varying, offset, start, end);
}

void
OsdGlslKernelDispatcher::ComputeShader::ApplyCatmarkVertexVerticesKernelB(
    OsdGpuVertexBuffer *vertex, OsdGpuVertexBuffer *varying,
    int V_IT_ofs, int V_ITa_ofs, int V_W_ofs, int offset, int start, int end) {
    
    glUniformSubroutinesuiv(GL_VERTEX_SHADER, 1, &_subComputeVertexB);
    glUniform1i(_tableOffsetUniforms[V_IT], V_IT_ofs);
    glUniform1i(_tableOffsetUniforms[V_ITa], V_ITa_ofs);
    glUniform1i(_tableOffsetUniforms[V_W], V_W_ofs);
    transformGpuBufferData(vertex, varying, offset, start, end);
}

void
OsdGlslKernelDispatcher::ComputeShader::ApplyCatmarkVertexVerticesKernelA(
    OsdGpuVertexBuffer *vertex, OsdGpuVertexBuffer *varying,
    int V_ITa_ofs, int V_W_ofs, int offset, bool pass, int start, int end) {
    
    glUniformSubroutinesuiv(GL_VERTEX_SHADER, 1, &_subComputeVertexA);
    glUniform1i(_uniformVertexPass, pass ? 1 : 0);
    glUniform1i(_tableOffsetUniforms[V_ITa], V_ITa_ofs);
    glUniform1i(_tableOffsetUniforms[V_W], V_W_ofs);
    transformGpuBufferData(vertex, varying, offset, start, end);
}

void
OsdGlslKernelDispatcher::ComputeShader::ApplyLoopEdgeVerticesKernel(
    OsdGpuVertexBuffer *vertex, OsdGpuVertexBuffer *varying,
    int E_IT_ofs, int E_W_ofs, int offset, int start, int end) {
    
    glUniformSubroutinesuiv(GL_VERTEX_SHADER, 1, &_subComputeEdge);
    glUniform1i(_tableOffsetUniforms[E_IT], E_IT_ofs);
    glUniform1i(_tableOffsetUniforms[E_W], E_W_ofs);
    transformGpuBufferData(vertex, varying, offset, start, end);
}

void
OsdGlslKernelDispatcher::ComputeShader::ApplyLoopVertexVerticesKernelB(
    OsdGpuVertexBuffer *vertex, OsdGpuVertexBuffer *varying,
    int V_IT_ofs, int V_ITa_ofs, int V_W_ofs, int offset, int start, int end) {
    
    glUniformSubroutinesuiv(GL_VERTEX_SHADER, 1, &_subComputeLoopVertexB);
    glUniform1i(_tableOffsetUniforms[V_IT], V_IT_ofs);
    glUniform1i(_tableOffsetUniforms[V_ITa], V_ITa_ofs);
    glUniform1i(_tableOffsetUniforms[V_W], V_W_ofs);
    transformGpuBufferData(vertex, varying, offset, start, end);
}

void
OsdGlslKernelDispatcher::ComputeShader::ApplyLoopVertexVerticesKernelA(
    OsdGpuVertexBuffer *vertex, OsdGpuVertexBuffer *varying,
    int V_ITa_ofs, int V_W_ofs, int offset, bool pass, int start, int end) {
    
    glUniformSubroutinesuiv(GL_VERTEX_SHADER, 1, &_subComputeVertexA);
    glUniform1i(_uniformVertexPass, pass ? 1 : 0);
    glUniform1i(_tableOffsetUniforms[V_ITa], V_ITa_ofs);
    glUniform1i(_tableOffsetUniforms[V_W], V_W_ofs);
    transformGpuBufferData(vertex, varying, offset, start, end);
}

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv
