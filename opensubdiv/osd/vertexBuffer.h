#ifndef OSD_VERTEX_BUFFER_H
#define OSD_VERTEX_BUFFER_H

#include <GL/glu.h>

#include <string.h> // memcpy (tobe moved to cpp)

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

class OsdVertexBuffer {
public:
    OsdVertexBuffer(int numElements) : _numElements(numElements) {}
    virtual ~OsdVertexBuffer();

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
    OsdGpuVertexBuffer(int numElements, int numVertices);

    virtual ~OsdGpuVertexBuffer();

    virtual void UpdateData(const float *src, int numVertices);

    virtual GLuint GetGpuBuffer() {
        return _vbo;
    }

protected:
    GLuint _vbo;
};

class OsdCpuVertexBuffer : public OsdVertexBuffer {
public:
    OsdCpuVertexBuffer(int numElements, int numVertices);
    
    virtual ~OsdCpuVertexBuffer();

    virtual void UpdateData(const float *src, int numVertices);

    float *GetCpuBuffer() {
        return _cpuVbo;
    }

    virtual GLuint GetGpuBuffer();

protected:
    float *_cpuVbo;
    int _vboSize;
    GLuint _vbo;
};

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif // OSD_VERTEX_BUFFER_H
