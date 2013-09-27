//
//   Copyright 2013 Pixar
//
//   Licensed under the Apache License, Version 2.0 (the "Apache License")
//   with the following modification; you may not use this file except in
//   compliance with the Apache License and the following modification to it:
//   Section 6. Trademarks. is deleted and replaced with:
//
//   6. Trademarks. This License does not grant permission to use the trade
//      names, trademarks, service marks, or product names of the Licensor
//      and its affiliates, except as required to comply with Section 4(c) of
//      the License and to reproduce the content of the NOTICE file.
//
//   You may obtain a copy of the Apache License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the Apache License with the above modification is
//   distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
//   KIND, either express or implied. See the Apache License for the specific
//   language governing permissions and limitations under the Apache License.
//

#include "../osd/cudaVertexBuffer.h"

#include <cuda_runtime.h>
#include <cassert>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

OsdCudaVertexBuffer::OsdCudaVertexBuffer(int numElements, int numVertices)
    : _numElements(numElements),
      _numVertices(numVertices),
      _cudaMem(0) {
}

OsdCudaVertexBuffer::~OsdCudaVertexBuffer() {
    if (_cudaMem) cudaFree(_cudaMem);
}

OsdCudaVertexBuffer *
OsdCudaVertexBuffer::Create(int numElements, int numVertices) {
    OsdCudaVertexBuffer *instance =
        new OsdCudaVertexBuffer(numElements, numVertices);
    if (instance->allocate()) return instance;
    delete instance;
    return NULL;
}

void
OsdCudaVertexBuffer::UpdateData(const float *src, int startVertex, int numVertices) {

    size_t size = _numElements * numVertices * sizeof(float);

    cudaMemcpy((float*)_cudaMem + _numElements * startVertex,
               src, size, cudaMemcpyHostToDevice);
}

int
OsdCudaVertexBuffer::GetNumElements() const {

    return _numElements;
}

int
OsdCudaVertexBuffer::GetNumVertices() const {

    return _numVertices;
}

float *
OsdCudaVertexBuffer::BindCudaBuffer() {

    return static_cast<float*>(_cudaMem);
}

bool
OsdCudaVertexBuffer::allocate() {
    int size = _numElements * _numVertices * sizeof(float);

    cudaError_t err = cudaMalloc(&_cudaMem, size);

    if (err != cudaSuccess) return false;
    return true;
}

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv

