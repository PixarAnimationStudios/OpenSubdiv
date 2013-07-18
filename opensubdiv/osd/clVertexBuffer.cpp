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

#include "../osd/clVertexBuffer.h"

#include <cassert>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

OsdCLVertexBuffer::OsdCLVertexBuffer(int numElements, int numVertices,
                                     cl_context clContext)
    : _numElements(numElements), _numVertices(numVertices),
      _clQueue(NULL), _clMemory(NULL) {

}

OsdCLVertexBuffer::~OsdCLVertexBuffer() {

    clReleaseMemObject(_clMemory);
}

OsdCLVertexBuffer *
OsdCLVertexBuffer::Create(int numElements, int numVertices,
                          cl_context clContext) {
    OsdCLVertexBuffer *instance =
        new OsdCLVertexBuffer(numElements, numVertices, clContext);
    if (instance->allocate(clContext)) return instance;
    delete instance;
    return NULL;
}

void
OsdCLVertexBuffer::UpdateData(const float *src, int startVertex, int numVertices, cl_command_queue queue) {

    size_t size = _numElements * numVertices * sizeof(float);
    size_t offset = startVertex * _numElements * sizeof(float);

    clEnqueueWriteBuffer(queue, _clMemory, true, offset, size, src, 0, NULL, NULL);
}

int
OsdCLVertexBuffer::GetNumElements() const {

    return _numElements;
}

int
OsdCLVertexBuffer::GetNumVertices() const {

    return _numVertices;
}

cl_mem
OsdCLVertexBuffer::BindCLBuffer(cl_command_queue queue) {

    return _clMemory;
}

bool
OsdCLVertexBuffer::allocate(cl_context clContext) {
    assert(clContext);
    int size = _numVertices * _numElements * sizeof(float);
    cl_int err;

    // XXX: do we really need a dummy buffer?
    float *ptr = new float[_numVertices * _numElements];
    _clMemory = clCreateBuffer(clContext, CL_MEM_READ_WRITE, size, ptr, &err);
    delete[] ptr;

    if (err != CL_SUCCESS) return false;
    return true;
}

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv

