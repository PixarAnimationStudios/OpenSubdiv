#ifndef OSD_VERTEX_BUFFER_H
#define OSD_VERTEX_BUFFER_H

#include <GL/glew.h>
#include <string.h> // memcpy (tobe moved to cpp)

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

class OsdVertexBuffer {
public:
    OsdVertexBuffer(int numElements) : _numElements(numElements) {}
    virtual ~OsdVertexBuffer() {}

    virtual void UpdateData(const float *src, int numVertices) = 0;

    virtual GLuint GetGpuBuffer() = 0;

    int GetNumElements() const {
        return _numElements;
    }

protected:
    int _numElements;
};

class OsdGpuVertexBuffer : public OsdVertexBuffer {
public:
    OsdGpuVertexBuffer(int numElements, int numVertices) : OsdVertexBuffer(numElements), _vbo(0) {
        int size = numElements * numVertices * sizeof(float);
        glGenBuffers(1, &_vbo);
        glBindBuffer(GL_ARRAY_BUFFER, _vbo);
        glBufferData(GL_ARRAY_BUFFER, size, 0, GL_STREAM_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }
    virtual ~OsdGpuVertexBuffer() {
        glDeleteBuffers(1, &_vbo);
    }

    virtual void UpdateData(const float *src, int numVertices) {
        glBindBuffer(GL_ARRAY_BUFFER, _vbo);
        float * pointer = (float*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
        memcpy(pointer, src, _numElements * numVertices * sizeof(float));
        glUnmapBuffer(GL_ARRAY_BUFFER);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    virtual GLuint GetGpuBuffer() {
        return _vbo;
    }

protected:
    GLuint _vbo;
};

class OsdCpuVertexBuffer : public OsdVertexBuffer {
public:
    OsdCpuVertexBuffer(int numElements, int numVertices) : OsdVertexBuffer(numElements), _cpuVbo(NULL), _vboSize(0), _vbo(0) {
        _vboSize = numElements * numVertices;
        _cpuVbo = new float[numElements * numVertices];
    }
    virtual ~OsdCpuVertexBuffer() {
        delete [] _cpuVbo;
        if (_vbo)
            glDeleteBuffers(1, &_vbo);
    }

    virtual void UpdateData(const float *src, int numVertices) {
        memcpy(_cpuVbo, src, _numElements * numVertices * sizeof(float));
    }

    float *GetCpuBuffer() {
        return _cpuVbo;
    }

    // XXX: this method name is missleading
    virtual GLuint GetGpuBuffer() {
        if (!_vbo)
            glGenBuffers(1, &_vbo);
        glBindBuffer(GL_ARRAY_BUFFER, _vbo);
        glBufferData(GL_ARRAY_BUFFER, _vboSize * sizeof(float), _cpuVbo, GL_STREAM_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        return _vbo;
    }

protected:
    float *_cpuVbo;
    int _vboSize;
    GLuint _vbo;
};

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif // OSD_VERTEX_BUFFER_H
