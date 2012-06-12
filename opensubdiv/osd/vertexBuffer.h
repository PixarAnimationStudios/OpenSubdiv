#ifndef OSD_VERTEX_BUFFER_H
#define OSD_VERTEX_BUFFER_H

#include <GL/glew.h>
#include <string.h> // memcpy (tobe moved to cpp)

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

class OsdVertexBuffer {
public:
    virtual ~OsdVertexBuffer() {}
    virtual void UpdateData(const float *src, int count) = 0;
    virtual GLuint GetGpuBuffer() = 0;
};

class OsdGpuVertexBuffer : public OsdVertexBuffer {
public:
    OsdGpuVertexBuffer(int numElements, int count) : _vbo(0), _numElements(numElements) {
        int stride = numElements * count * sizeof(float);
        glGenBuffers(1, &_vbo);
        glBindBuffer(GL_ARRAY_BUFFER, _vbo);
        glBufferData(GL_ARRAY_BUFFER, stride, 0, GL_STREAM_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }
    virtual ~OsdGpuVertexBuffer() {
        glDeleteBuffers(1, &_vbo);
    }
    virtual void UpdateData(const float *src, int count) {
        glBindBuffer(GL_ARRAY_BUFFER, _vbo);
        float * pointer = (float*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
        memcpy(pointer, src, _numElements * count * sizeof(float));
        glUnmapBuffer(GL_ARRAY_BUFFER);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }
    virtual GLuint GetGpuBuffer() {
        return _vbo;
    }
private:
    GLuint _vbo;
    int _numElements;
};

class OsdCpuVertexBuffer : public OsdVertexBuffer {
public:
    OsdCpuVertexBuffer(int numElements, int count) : _cpuVbo(NULL), _vboSize(0), _numElements(numElements), _vbo(0) {
        _cpuVbo = new float[numElements * count];
        _vboSize = numElements * count;
    }
    virtual ~OsdCpuVertexBuffer() {
        if(_cpuVbo) delete[] _cpuVbo;
        if(_vbo) glDeleteBuffers(1, &_vbo);
    }
    virtual void UpdateData(const float *src, int count) {
        memcpy(_cpuVbo, src, _numElements * count * sizeof(float));
    }
    float *GetCpuBuffer() {
        return _cpuVbo;
    }
    virtual GLuint GetGpuBuffer() {
        if(!_vbo) glGenBuffers(1, &_vbo);
        glBindBuffer(GL_ARRAY_BUFFER, _vbo);
        glBufferData(GL_ARRAY_BUFFER, _vboSize * sizeof(float), _cpuVbo, GL_STREAM_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        return _vbo;
    }

    int GetNumElements() const {
        return _numElements;
    }
private:
    float *_cpuVbo;
    int _vboSize;
    int _numElements;
    GLuint _vbo;
};

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif // OSD_VERTEX_BUFFER_H
