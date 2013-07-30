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

#include "../osd/cpuVertexBuffer.h"

#include <string.h>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

OsdCpuVertexBuffer::OsdCpuVertexBuffer(int numElements, int numVertices)
    : _numElements(numElements),
      _numVertices(numVertices),
      _cpuBuffer(NULL) {

    _cpuBuffer = new float[numElements * numVertices];
}

OsdCpuVertexBuffer::~OsdCpuVertexBuffer() {

    delete[] _cpuBuffer;
}

OsdCpuVertexBuffer *
OsdCpuVertexBuffer::Create(int numElements, int numVertices) {

    return new OsdCpuVertexBuffer(numElements, numVertices);
}

void
OsdCpuVertexBuffer::UpdateData(const float *src, int startVertex, int numVertices) {

    memcpy(_cpuBuffer + startVertex * _numElements,
           src, GetNumElements() * numVertices * sizeof(float));
}

int
OsdCpuVertexBuffer::GetNumElements() const {

    return _numElements;
}

int
OsdCpuVertexBuffer::GetNumVertices() const {

    return _numVertices;
}

float*
OsdCpuVertexBuffer::BindCpuBuffer() {

    return _cpuBuffer;
}

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv

