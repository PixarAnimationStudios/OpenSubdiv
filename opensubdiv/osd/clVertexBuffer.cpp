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

#include "../osd/clVertexBuffer.h"

#include <cassert>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

OsdCLVertexBuffer::OsdCLVertexBuffer(int numElements, int numVertices,
                                     cl_context clContext)
    : _numElements(numElements), _numVertices(numVertices),
      _clMemory(NULL), _clQueue(NULL) {

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
OsdCLVertexBuffer::UpdateData(const float *src, int numVertices, cl_command_queue queue) {

    size_t size = _numElements * numVertices * sizeof(float);

    clEnqueueWriteBuffer(queue, _clMemory, true, 0, size, src, 0, NULL, NULL);
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

