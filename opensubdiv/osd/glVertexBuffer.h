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

#ifndef OSD_GL_VERTEX_BUFFER_H
#define OSD_GL_VERTEX_BUFFER_H

#include "../version.h"

#include "../osd/opengl.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

///
/// \brief Concrete vertex buffer class for GLSL subvision and OpenGL drawing.
///
/// GLVertexBuffer implements GLVertexBufferInterface. An instance
/// of this buffer class can be passed to OsdGLComputeController
/// and OsdGLDrawController
///
class GLVertexBuffer {
public:
    /// Creator. Returns NULL if error.
    static GLVertexBuffer * Create(int numElements, int numVertices);

    /// Destructor.
    ~GLVertexBuffer();

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
    GLVertexBuffer(int numElements, int numVertices);

    /// Allocates VBO for this buffer.
    /// Returns true if success.
    bool allocate();

private:
    int _numElements;
    int _numVertices;
    GLuint _vbo;
};

}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OSD_GL_VERTEX_BUFFER_H
