

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

#include "../osd/clGLVertexBuffer.h"

#include <cassert>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

OsdCLGLVertexBuffer::OsdCLGLVertexBuffer(int numElements,
                                         int numVertices,
                                         cl_context clContext)
    : _numElements(numElements), _numVertices(numVertices),
      _vbo(0), _clQueue(0), _clMemory(0), _clMapped(false) {

}

OsdCLGLVertexBuffer::~OsdCLGLVertexBuffer() {

    unmap();
    clReleaseMemObject(_clMemory);
    glDeleteBuffers(1, &_vbo);
}

OsdCLGLVertexBuffer *
OsdCLGLVertexBuffer::Create(int numElements, int numVertices, cl_context clContext)
{
    OsdCLGLVertexBuffer *instance =
        new OsdCLGLVertexBuffer(numElements, numVertices, clContext);
    if (instance->allocate(clContext)) return instance;
    delete instance;
    return NULL;
}

void
OsdCLGLVertexBuffer::UpdateData(const float *src, int numVertices,
                                cl_command_queue queue) {

    size_t size = numVertices * _numElements * sizeof(float);

    map(queue);
    clEnqueueWriteBuffer(queue, _clMemory, true, 0, size, src, 0, NULL, NULL);
}

int
OsdCLGLVertexBuffer::GetNumElements() const {

    return _numElements;
}

int
OsdCLGLVertexBuffer::GetNumVertices() const {

    return _numVertices;
}

cl_mem
OsdCLGLVertexBuffer::BindCLBuffer(cl_command_queue queue) {

    map(queue);
    return _clMemory;
}

GLuint
OsdCLGLVertexBuffer::BindVBO() {

    unmap();
    return _vbo;
}

bool
OsdCLGLVertexBuffer::allocate(cl_context clContext) {

    assert(clContext);

    // create GL buffer first
    int size = _numElements * _numVertices * sizeof(float);
    GLint prev = 0;

    glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &prev);
    glGenBuffers(1, &_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, _vbo);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_STREAM_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, prev);

    if (glGetError() != GL_NO_ERROR) return false;

    // register vbo as cl memory
    cl_int err;
    _clMemory = clCreateFromGLBuffer(clContext,
                                     CL_MEM_READ_WRITE, _vbo, &err);

    if (err != CL_SUCCESS) return false;
    return true;
}

void
OsdCLGLVertexBuffer::map(cl_command_queue queue) {

    if (_clMapped) return;    // XXX: what if another queue is given?
    _clQueue = queue;
    clEnqueueAcquireGLObjects(queue, 1, &_clMemory, 0, 0, 0);
    _clMapped = true;
}

void
OsdCLGLVertexBuffer::unmap() {

    if (not _clMapped) return;
    clEnqueueReleaseGLObjects(_clQueue, 1, &_clMemory, 0, 0, 0);
    _clMapped = false;
}

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv

