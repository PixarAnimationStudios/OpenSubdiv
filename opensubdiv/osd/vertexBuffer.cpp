#include "../version.h"

#include <GL/glew.h>

#include "vertexBuffer.h"

#include <GL/gl.h>
#include <GL/glext.h>

#include <iostream>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

OsdVertexBuffer::~OsdVertexBuffer()
{
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
OsdGpuVertexBuffer::UpdateData(const float *src, int numVertices)
{
    glBindBuffer(GL_ARRAY_BUFFER, _vbo);
    float * pointer = (float*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
    memcpy(pointer, src, _numElements * numVertices * sizeof(float));
    glUnmapBuffer(GL_ARRAY_BUFFER);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

OsdGpuVertexBuffer::~OsdGpuVertexBuffer()
{
    glDeleteBuffers(1, &_vbo);

}


OsdCpuVertexBuffer::OsdCpuVertexBuffer(int numElements, int numVertices) :
    OsdVertexBuffer(numElements), _cpuVbo(NULL), _vboSize(0), _vbo(0)
{
    _vboSize = numElements * numVertices;
    _cpuVbo = new float[numElements * numVertices];
}

OsdCpuVertexBuffer::~OsdCpuVertexBuffer()
{
    delete [] _cpuVbo;
    if (_vbo)
        glDeleteBuffers(1, &_vbo);
}

void
OsdCpuVertexBuffer::UpdateData(const float *src, int numVertices)
{
    memcpy(_cpuVbo, src, _numElements * numVertices * sizeof(float));
}

GLuint
OsdCpuVertexBuffer::GetGpuBuffer()
{
    if (!_vbo)
        glGenBuffers(1, &_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, _vbo);
    glBufferData(GL_ARRAY_BUFFER, _vboSize * sizeof(float), _cpuVbo, GL_STREAM_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    return _vbo;
}

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv

