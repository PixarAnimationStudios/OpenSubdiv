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

#if not defined(__APPLE__)
    #include <GL/glew.h>
#else
    #include <OpenGL/gl3.h>
#endif

#include "../osd/vertexBuffer.h"

#include <iostream>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

OsdVertexBuffer::~OsdVertexBuffer() {
}

OsdGpuVertexBuffer::OsdGpuVertexBuffer(int numElements, int numVertices) :
    OsdVertexBuffer(numElements), _vbo(0)
{
    int size = numElements * numVertices * sizeof(float);
    glGenBuffers(1, &_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, _vbo);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_STREAM_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void
OsdGpuVertexBuffer::UpdateData(const float *src, int numVertices) {

    glBindBuffer(GL_ARRAY_BUFFER, GetGpuBuffer());
    float * pointer = (float*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
    memcpy(pointer, src, GetNumElements() * numVertices * sizeof(float));
    glUnmapBuffer(GL_ARRAY_BUFFER);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void
OsdGpuVertexBuffer::GetBufferData(float * data, int firstVert, int numVerts) {

    glBindBuffer(GL_ARRAY_BUFFER, GetGpuBuffer());

    glGetBufferSubData(GL_ARRAY_BUFFER, GetNumElements() * firstVert * sizeof(float),
                                        GetNumElements() * numVerts * sizeof(float),
                                        data);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

OsdGpuVertexBuffer::~OsdGpuVertexBuffer() {

    glDeleteBuffers(1, &_vbo);
}

OsdCpuVertexBuffer::OsdCpuVertexBuffer(int numElements, int numVertices) :
    OsdVertexBuffer(numElements), _cpuVbo(NULL), _vboSize(0), _vbo(0)
{
    _vboSize = numElements * numVertices;
    _cpuVbo = new float[numElements * numVertices];
}

OsdCpuVertexBuffer::~OsdCpuVertexBuffer() {

    delete [] _cpuVbo;
    if (_vbo)
        glDeleteBuffers(1, &_vbo);
}

void
OsdCpuVertexBuffer::UpdateData(const float *src, int numVertices) {

    memcpy(_cpuVbo, src, _numElements * numVertices * sizeof(float));
}

GLuint
OsdCpuVertexBuffer::GetGpuBuffer() {

    if (!_vbo)
        glGenBuffers(1, &_vbo);

    glBindBuffer(GL_ARRAY_BUFFER, _vbo);
    glBufferData(GL_ARRAY_BUFFER, _vboSize * sizeof(float), _cpuVbo, GL_STREAM_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    return _vbo;
}

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv

