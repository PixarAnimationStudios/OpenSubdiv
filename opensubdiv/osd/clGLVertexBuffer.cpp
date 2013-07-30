//
//     Copyright 2013 Pixar
//
//     Licensed under the Apache License, Version 2.0 (the "License");
//     you may not use this file except in compliance with the License
//     and the following modification to it: Section 6 Trademarks.
//     deleted and replaced with:
//
//     6. Trademarks. This License does not grant permission to use the
//     trade names, trademarks, service marks, or product names of the
//     Licensor and its affiliates, except as required for reproducing
//     the content of the NOTICE file.
//
//     You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
//     Unless required by applicable law or agreed to in writing,
//     software distributed under the License is distributed on an
//     "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
//     either express or implied.  See the License for the specific
//     language governing permissions and limitations under the
//     License.
//

#include "../osd/clGLVertexBuffer.h"

#include "../osd/opengl.h"

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
OsdCLGLVertexBuffer::UpdateData(const float *src, int startVertex, int numVertices, cl_command_queue queue) {

    size_t size = numVertices * _numElements * sizeof(float);
    size_t offset = startVertex * _numElements * sizeof(float);

    map(queue);
    clEnqueueWriteBuffer(queue, _clMemory, true, offset, size, src, 0, NULL, NULL);
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

