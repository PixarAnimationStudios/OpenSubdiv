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

#include "../osd/glVertexBuffer.h"

#include "../osd/opengl.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

OsdGLVertexBuffer::OsdGLVertexBuffer(int numElements, int numVertices)
    : _numElements(numElements),
      _numVertices(numVertices),
      _vbo(0)
{
}

OsdGLVertexBuffer::~OsdGLVertexBuffer() {

    glDeleteBuffers(1, &_vbo);
}

OsdGLVertexBuffer *
OsdGLVertexBuffer::Create(int numElements, int numVertices) {

    OsdGLVertexBuffer *instance =
        new OsdGLVertexBuffer(numElements, numVertices);
    if (instance->allocate()) return instance;
    delete instance;
    return 0;
}

void
OsdGLVertexBuffer::UpdateData(const float *src, int startVertex, int numVertices) {

    glBindBuffer(GL_ARRAY_BUFFER, _vbo);
    int size = numVertices * _numElements * sizeof(float);
    glBufferSubData(GL_ARRAY_BUFFER, startVertex * _numElements * sizeof(float), size, src);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

int
OsdGLVertexBuffer::GetNumElements() const {

    return _numElements;
}

int
OsdGLVertexBuffer::GetNumVertices() const {

    return _numVertices;
}

GLuint
OsdGLVertexBuffer::BindVBO() {

    return _vbo;
}

bool
OsdGLVertexBuffer::allocate() {
    
    int size = _numElements * _numVertices * sizeof(float);
    GLint prev = 0;

    glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &prev);
    glGenBuffers(1, &_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, _vbo);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_STREAM_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, prev);

//    if (glGetError() != GL_NO_ERROR) return false;
    return true;
}

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv

