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
#ifndef OSD_GL_VERTEX_BUFFER_H
#define OSD_GL_VERTEX_BUFFER_H

#include "../version.h"

#include "../osd/opengl.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

///
/// \brief Concrete vertex buffer class for GLSL subvision and OpenGL drawing.
///
/// OsdGLVertexBuffer implements OsdGLVertexBufferInterface. An instance
/// of this buffer class can be passed to OsdGLComputeController
/// and OsdGLDrawController
///
class OsdGLVertexBuffer {
public:
    /// Creator. Returns NULL if error.
    static OsdGLVertexBuffer * Create(int numElements, int numVertices);

    /// Destructor.
    ~OsdGLVertexBuffer();

    /// This method is meant to be used in client code in order to provide coarse
    /// vertices data to Osd.
    void UpdateData(const float *src, int startVertex, int numVertices);

    /// Returns how many elements defined in this vertex buffer.
    int GetNumElements() const;

    /// Returns how many vertices allocated in this vertex buffer.
    int GetNumVertices() const;

    /// Returns the GL buffer object.
    GLuint BindVBO();

protected:
    /// Constructor.
    OsdGLVertexBuffer(int numElements, int numVertices);

    /// Allocates VBO for this buffer.
    /// Returns true if success.
    bool allocate();

private:
    int _numElements;
    int _numVertices;
    GLuint _vbo;
};

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OSD_GL_VERTEX_BUFFER_H
