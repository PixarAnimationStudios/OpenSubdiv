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

#include "../osd/cudaGLVertexBuffer.h"
#include "../osd/error.h"

#include "../osd/opengl.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <cassert>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

OsdCudaGLVertexBuffer::OsdCudaGLVertexBuffer(int numElements, int numVertices)
    : _numElements(numElements), _numVertices(numVertices),
      _vbo(0), _devicePtr(0), _cudaResource(0) {
}

OsdCudaGLVertexBuffer::~OsdCudaGLVertexBuffer() {

    cudaThreadSynchronize();
    unmap();
    cudaGraphicsUnregisterResource(_cudaResource);
    cudaThreadSynchronize();
    glDeleteBuffers(1, &_vbo);
}

OsdCudaGLVertexBuffer *
OsdCudaGLVertexBuffer::Create(int numElements, int numVertices) {
    OsdCudaGLVertexBuffer *instance =
        new OsdCudaGLVertexBuffer(numElements, numVertices);
    if (instance->allocate()) return instance;
    OsdError(OSD_CUDA_GL_ERROR,"OsdCudaGLVertexBuffer::Create failed.\n");
    delete instance;
    return NULL;
}

void
OsdCudaGLVertexBuffer::UpdateData(const float *src, int startVertex, int numVertices) {

    map();
    cudaMemcpy((float*)_devicePtr + _numElements * startVertex,
               src,
               _numElements * numVertices * sizeof(float),
               cudaMemcpyHostToDevice);
}

int
OsdCudaGLVertexBuffer::GetNumElements() const {

    return _numElements;
}

int
OsdCudaGLVertexBuffer::GetNumVertices() const {

    return _numVertices;
}

float *
OsdCudaGLVertexBuffer::BindCudaBuffer() {

    map();
    return static_cast<float*>(_devicePtr);
}

GLuint
OsdCudaGLVertexBuffer::BindVBO() {

    unmap();
    return _vbo;
}

bool
OsdCudaGLVertexBuffer::allocate() {

    int size = _numElements * _numVertices * sizeof(float);
    
    glGenBuffers(1, &_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, _vbo);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_STREAM_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    cudaThreadSynchronize();
    // register vbo as cuda resource
    cudaError_t err = cudaGraphicsGLRegisterBuffer(
        &_cudaResource, _vbo, cudaGraphicsMapFlagsNone);

    if (err != cudaSuccess) return false;
    return true;
}

void
OsdCudaGLVertexBuffer::map() {

    if (_devicePtr) return;
    size_t num_bytes;
    void *ptr;

    cudaThreadSynchronize();
    cudaError_t err = cudaGraphicsMapResources(1, &_cudaResource, 0);
    if (err != cudaSuccess)
        OsdError(OSD_CUDA_GL_ERROR, "OsdCudaGLVertexBuffer::map failed.\n%s\n", cudaGetErrorString(err));
    err = cudaGraphicsResourceGetMappedPointer(&ptr, &num_bytes, _cudaResource);
    if (err != cudaSuccess)
        OsdError(OSD_CUDA_GL_ERROR, "OsdCudaGLVertexBuffer::map failed.\n%s\n", cudaGetErrorString(err));
    _devicePtr = ptr;
}

void
OsdCudaGLVertexBuffer::unmap() {

    cudaThreadSynchronize();
    if (_devicePtr == NULL) return;
    cudaError_t err = cudaGraphicsUnmapResources(1, &_cudaResource, 0);
    if (err != cudaSuccess)
        OsdError(OSD_CUDA_GL_ERROR, "OsdCudaGLVertexBuffer::unmap failed.\n%s\n", cudaGetErrorString(err));
    _devicePtr = NULL;
}


}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv

