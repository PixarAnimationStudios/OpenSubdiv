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

OsdGlslKernelDispatcher::OsdGlslKernelDispatcher(int levels)
    : OsdKernelDispatcher(levels)
{
    _vertexBuffer = 0;
    _varyingBuffer = 0;
    _prgKernel = 0;

    glGenTextures(1, &_vertexTexture);
    glGenTextures(1, &_varyingTexture);

    _tableBuffers.resize(TABLE_MAX);
    _tableTextures.resize(TABLE_MAX);
    _tableUniforms.resize(TABLE_MAX);
    _tableOffsetUniforms.resize(TABLE_MAX);

    glGenBuffers(TABLE_MAX, &_tableBuffers[0]);
    glGenTextures(TABLE_MAX, &_tableTextures[0]);

    subComputeFace = 0;
    subComputeEdge = 0;
    subComputeVertexA = 0;
    subComputeVertexB = 0;
    uniformVertexPass = 0;
    uniformIndexStart = 0;
    uniformIndexOffset = 0;

    compile(shaderSource, shaderDefines);

    subComputeFace = glGetSubroutineIndex(_prgKernel, GL_VERTEX_SHADER, "catmarkComputeFace");
    subComputeEdge = glGetSubroutineIndex(_prgKernel, GL_VERTEX_SHADER, "catmarkComputeEdge");
    subComputeVertexA = glGetSubroutineIndex(_prgKernel, GL_VERTEX_SHADER, "catmarkComputeVertexA");
    subComputeVertexB = glGetSubroutineIndex(_prgKernel, GL_VERTEX_SHADER, "catmarkComputeVertexB");
    subComputeLoopVertexB = glGetSubroutineIndex(_prgKernel, GL_VERTEX_SHADER, "loopComputeVertexB");

    uniformVertexPass = glGetUniformLocation(_prgKernel, "vertexPass");
    uniformIndexStart = glGetUniformLocation(_prgKernel, "indexStart");
    uniformIndexOffset = glGetUniformLocation(_prgKernel, "indexOffset");

    _tableUniforms[F_IT]  = glGetUniformLocation(_prgKernel, "_F0_IT");
    _tableUniforms[F_ITa] = glGetUniformLocation(_prgKernel, "_F0_ITa");
    _tableUniforms[E_IT]  = glGetUniformLocation(_prgKernel, "_E0_IT");
    _tableUniforms[V_IT]  = glGetUniformLocation(_prgKernel, "_V0_IT");
    _tableUniforms[V_ITa] = glGetUniformLocation(_prgKernel, "_V0_ITa");
    _tableUniforms[E_W]   = glGetUniformLocation(_prgKernel, "_E0_S");
    _tableUniforms[V_W]   = glGetUniformLocation(_prgKernel, "_V0_S");
    _tableOffsetUniforms[F_IT]  = glGetUniformLocation(_prgKernel, "F_IT_ofs");
    _tableOffsetUniforms[F_ITa] = glGetUniformLocation(_prgKernel, "F_ITa_ofs");
    _tableOffsetUniforms[E_IT]  = glGetUniformLocation(_prgKernel, "E_IT_ofs");
    _tableOffsetUniforms[V_IT]  = glGetUniformLocation(_prgKernel, "V_IT_ofs");
    _tableOffsetUniforms[V_ITa] = glGetUniformLocation(_prgKernel, "V_ITa_ofs");
    _tableOffsetUniforms[E_W]   = glGetUniformLocation(_prgKernel, "E_W_ofs");
    _tableOffsetUniforms[V_W]   = glGetUniformLocation(_prgKernel, "V_W_ofs");
}

OsdGlslKernelDispatcher::~OsdGlslKernelDispatcher() {

    if (_prgKernel) 
        glDeleteProgram(_prgKernel);

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
OsdGlslKernelDispatcher::BeginLaunchKernel() {

    glUseProgram(_prgKernel);
    glEnable(GL_RASTERIZER_DISCARD);

//XXX what if loop..
    bindTextureBuffer(_tableUniforms[F_IT],  _tableBuffers[F_IT],
                      _tableTextures[F_IT],  GL_R32UI, 2);
    bindTextureBuffer(_tableUniforms[F_ITa], _tableBuffers[F_ITa],
                      _tableTextures[F_ITa], GL_R32I,  3);

#ifdef OPT_E0_IT_VEC4
    bindTextureBuffer(_tableUniforms[E_IT],  _tableBuffers[E_IT],
                      _tableTextures[E_IT],  GL_RGBA32UI, 4);
#else
    bindTextureBuffer(_tableUniforms[E_IT],  _tableBuffers[E_IT],
                      _tableTextures[E_IT],  GL_R32UI, 4);
#endif

#ifdef OPT_CATMARK_V_IT_VEC2
    bindTextureBuffer(_tableUniforms[V_IT],  _tableBuffers[V_IT],
                      _tableTextures[V_IT],  GL_RG32UI, 5);
#else
    bindTextureBuffer(_tableUniforms[V_IT],  _tableBuffers[V_IT],
                      _tableTextures[V_IT],  GL_R32UI, 5);
#endif
    bindTextureBuffer(_tableUniforms[V_ITa], _tableBuffers[V_ITa],
                      _tableTextures[V_ITa], GL_R32I,  6);
#ifdef OPT_E0_S_VEC2
    bindTextureBuffer(_tableUniforms[E_W],   _tableBuffers[E_W],
                      _tableTextures[E_W],   GL_RG32F,  7);
#else
    bindTextureBuffer(_tableUniforms[E_W],   _tableBuffers[E_W],
                      _tableTextures[E_W],   GL_R32F,  7);
#endif
    bindTextureBuffer(_tableUniforms[V_W],   _tableBuffers[V_W],
                      _tableTextures[V_W],   GL_R32F,  8);

}

void
OsdGlslKernelDispatcher::EndLaunchKernel() {

    glDisable(GL_RASTERIZER_DISCARD);
    glUseProgram(0);

    // XXX Unbind table buffer
}

OsdVertexBuffer *
OsdGlslKernelDispatcher::InitializeVertexBuffer(int numElements, int count)
{
    return new OsdGpuVertexBuffer(numElements, count);
}

void
OsdGlslKernelDispatcher::BindVertexBuffer(OsdVertexBuffer *vertex, OsdVertexBuffer *varying) {

    OsdGpuVertexBuffer *bVertex = dynamic_cast<OsdGpuVertexBuffer *>(vertex);
    OsdGpuVertexBuffer *bVarying = dynamic_cast<OsdGpuVertexBuffer *>(varying);

    if (bVertex) {
        _vertexBuffer = bVertex->GetGpuBuffer();
        bindTextureBuffer(_vertexUniform, _vertexBuffer, _vertexTexture, GL_RGB32F, 0);
    }

    if (bVarying) {
        _varyingBuffer = bVarying->GetGpuBuffer();
        bindTextureBuffer(_varyingUniform, _varyingBuffer, _varyingTexture, GL_R32F, 0);
    }

    glUseProgram(_prgKernel);
    glUniform1i(_vertexUniform, 0);

#if 0  // experiment to use image store function
    glActiveTexture(GL_TEXTURE0 + 0);
    glBindImageTextureEXT(0, _vertexTexture, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F);

    if (_numVarying > 0) {
        glUniform1i(_varyingUniform, 1);
        glBindImageTextureEXT(1, _vertexTexture, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F);
    }
#endif
    CHECK_GL_ERROR("BindVertexBuffer \n");
}

void
OsdGlslKernelDispatcher::UnbindVertexBuffer()
{
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
    OSD_DEBUG("BindTextureBuffer sampler=%d, buffer=%d, texture = %d, E%x\n", sampler, buffer, texture, glGetError());

    glUniform1i(sampler, unit);
    glActiveTexture(GL_TEXTURE0 + unit);
    CHECK_GL_ERROR("BindTextureBuffer glActiveTexture %d\n", unit);
    glBindTexture(GL_TEXTURE_BUFFER, texture);
    CHECK_GL_ERROR("BindTextureBuffer glBindTexture %d\n", texture);
    glTexBuffer(GL_TEXTURE_BUFFER, type, buffer);
    CHECK_GL_ERROR("BindTextureBuffer glTexBuffer\n");
    glActiveTexture(GL_TEXTURE0);
}

bool
OsdGlslKernelDispatcher::compile(const char *shaderSource, const char *shaderDefine) {

    _prgKernel = glCreateProgram();

    GLuint shader = glCreateShader(GL_VERTEX_SHADER);

    char constantDefine[256];
    snprintf(constantDefine, 256,
             "#define NUM_VARYING_ELEMENTS %d\n", _numVarying);

    const char *shaderSources[3];
    shaderSources[0] = constantDefine;
    shaderSources[1] = shaderDefine;
    shaderSources[2] = shaderSource;
    glShaderSource(shader, 3, shaderSources, NULL);
    glCompileShader(shader);
    glAttachShader(_prgKernel, shader);

    const char *outputs[] = { "outPosition", 
                              "outNormal", 
			      "gl_NextBuffer", 
			      "outVaryingData" };
    int nOutputs = _numVarying > 0 ? 4 : 2;

    glTransformFeedbackVaryings(_prgKernel, nOutputs, outputs, GL_INTERLEAVED_ATTRIBS);

    CHECK_GL_ERROR("Transform feedback initialize \n");

    GLint linked = 0;
    glLinkProgram(_prgKernel);
    glGetProgramiv(_prgKernel, GL_LINK_STATUS, &linked);

    if (linked == GL_FALSE) {
        OSD_ERROR("Fail to link shader\n");

        char buffer[1024];
        glGetShaderInfoLog(shader, 1024, NULL, buffer);
        OSD_ERROR(buffer);

        glGetProgramInfoLog(_prgKernel, 1024, NULL, buffer);
        OSD_ERROR(buffer);
        
        glDeleteProgram(_prgKernel);
        _prgKernel = 0;
        // XXX ERROR HANDLE
        return false;
    }

    glDeleteShader(shader);

    _vertexUniform         = glGetUniformLocation(_prgKernel, "vertex");
    _varyingUniform        = glGetUniformLocation(_prgKernel, "varyingData");

    return true;
}

void
OsdGlslKernelDispatcher::unbindTextureBuffer(int unit) const {
    glActiveTexture(GL_TEXTURE0 + unit);
    glBindTexture(GL_TEXTURE_BUFFER, 0);
}

void
OsdGlslKernelDispatcher::transformGpuBufferData(GLuint kernel, GLint offset, int start, int end, bool vertexPass) const {
    int count = end - start;
    if (count <= 0) return;
    OSD_DEBUG("_transformGpuBufferData kernel=%d E%x, offset=%d, count=%d\n", kernel, glGetError(), offset, count);

    glUniformSubroutinesuiv(GL_VERTEX_SHADER, 1, &kernel);
    glUniform1i(uniformVertexPass, vertexPass); // XXX

    // set batch range
    glUniform1i(uniformIndexStart, start);
    glUniform1i(uniformIndexOffset, offset);
    // XXX: end is not used here now
    CHECK_GL_ERROR("Uniform index set at offset=%d. start=%d\n", offset, start);

    // set transform feedback buffer
    int vertexStride = _numVertexElements*sizeof(float);
    int varyingStride = _numVarying*sizeof(float);
    glBindBufferRange(GL_TRANSFORM_FEEDBACK_BUFFER, 0, _vertexBuffer,
                      (start+offset)*vertexStride, count*vertexStride);
    CHECK_GL_ERROR("transformGpuBufferData glBindBufferRange\n");

    if (_numVarying > 0){
        glBindBufferRange(GL_TRANSFORM_FEEDBACK_BUFFER, 1, _varyingBuffer,
                          (start+offset)*varyingStride, count*varyingStride);
    }

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
OsdGlslKernelDispatcher::ApplyCatmarkFaceVerticesKernel(
    FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const {
    
    glUniform1i(_tableOffsetUniforms[F_IT], _tableOffsets[F_IT][level-1]);
    glUniform1i(_tableOffsetUniforms[F_ITa], _tableOffsets[F_ITa][level-1]);
    transformGpuBufferData(subComputeFace, offset, start, end);
}

void
OsdGlslKernelDispatcher::ApplyCatmarkEdgeVerticesKernel(
    FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const {
    
    glUniform1i(_tableOffsetUniforms[E_IT], _tableOffsets[E_IT][level-1]);
    glUniform1i(_tableOffsetUniforms[E_W], _tableOffsets[E_W][level-1]);
    transformGpuBufferData(subComputeEdge, offset, start, end);
}

void
OsdGlslKernelDispatcher::ApplyCatmarkVertexVerticesKernelB(
    FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const {
    
    glUniform1i(_tableOffsetUniforms[V_IT], _tableOffsets[V_IT][level-1]);
    glUniform1i(_tableOffsetUniforms[V_ITa], _tableOffsets[V_ITa][level-1]);
    glUniform1i(_tableOffsetUniforms[V_W], _tableOffsets[V_W][level-1]);
    transformGpuBufferData(subComputeVertexB, offset, start, end);
}

void
OsdGlslKernelDispatcher::ApplyCatmarkVertexVerticesKernelA(
    FarMesh<OsdVertex> * mesh, int offset, bool pass, int level, int start, int end, void * data) const {
    
    glUniform1i(_tableOffsetUniforms[V_ITa], _tableOffsets[V_ITa][level-1]);
    glUniform1i(_tableOffsetUniforms[V_W], _tableOffsets[V_W][level-1]);
    transformGpuBufferData(subComputeVertexA, offset, start, end, pass);
}

void
OsdGlslKernelDispatcher::ApplyLoopEdgeVerticesKernel(
    FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const {
    
    glUniform1i(_tableOffsetUniforms[E_IT], _tableOffsets[E_IT][level-1]);
    glUniform1i(_tableOffsetUniforms[E_W], _tableOffsets[E_W][level-1]);
    transformGpuBufferData(subComputeEdge, offset, start, end);
}

void
OsdGlslKernelDispatcher::ApplyLoopVertexVerticesKernelB(
    FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const {
    
    glUniform1i(_tableOffsetUniforms[V_IT], _tableOffsets[V_IT][level-1]);
    glUniform1i(_tableOffsetUniforms[V_ITa], _tableOffsets[V_ITa][level-1]);
    glUniform1i(_tableOffsetUniforms[V_W], _tableOffsets[V_W][level-1]);
    transformGpuBufferData(subComputeLoopVertexB, offset, start, end);
}

void
OsdGlslKernelDispatcher::ApplyLoopVertexVerticesKernelA(
    FarMesh<OsdVertex> * mesh, int offset, bool pass, int level, int start, int end, void * data) const {
    
    glUniform1i(_tableOffsetUniforms[V_ITa], _tableOffsets[V_ITa][level-1]);
    glUniform1i(_tableOffsetUniforms[V_W], _tableOffsets[V_W][level-1]);
    transformGpuBufferData(subComputeVertexA, offset, start, end, pass);
}

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv
