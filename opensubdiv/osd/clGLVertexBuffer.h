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
#ifndef OSD_CL_GL_VERTEX_BUFFER_H
#define OSD_CL_GL_VERTEX_BUFFER_H

#include "../version.h"

#include "../osd/opengl.h"

#if defined(__APPLE__)
    #include <OpenCL/opencl.h>
#else
    #include <CL/opencl.h>
#endif

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

///
/// \brief Concrete vertex buffer class for OpenCL subvision and OpenGL drawing.
///
/// OsdCLGLVertexBuffer implements OsdCLVertexBufferInterface and
/// OsdGLVertexBufferInterface.
///
/// The buffer interop between OpenCL and GL is handled automatically when a
/// client calls BindCLBuffer and BindVBO methods.
///
class OsdCLGLVertexBuffer {
public:
    /// Creator. Returns NULL if error.
    static OsdCLGLVertexBuffer * Create(int numElements, 
                                        int numVertices, 
                                        cl_context clContext);

    /// Destructor.
    ~OsdCLGLVertexBuffer();

    /// This method is meant to be used in client code in order to provide coarse
    /// vertices data to Osd.
    void UpdateData(const float *src, int startVertex, int numVertices, cl_command_queue clQueue);

    /// Returns how many elements defined in this vertex buffer.
    int GetNumElements() const;

    /// Returns how many vertices allocated in this vertex buffer.
    int GetNumVertices() const;

    /// Returns the CL memory object. GL buffer will be mapped to CL memory
    /// space if necessary.
    cl_mem BindCLBuffer(cl_command_queue queue);

    /// Returns the GL buffer object. If the buffer is mapped to CL memory
    /// space, it will be unmapped back to GL.
    GLuint BindVBO();

protected:
    /// Constructor.
    OsdCLGLVertexBuffer(int numElements, int numVertices, cl_context clContext);

    /// Allocates VBO for this buffer and register as a CL resource.
    /// Returns true if success.
    bool allocate(cl_context clContext);

    /// Acqures a resource from GL.
    void map(cl_command_queue queue);

    /// Releases a resource to GL.
    void unmap();

private:
    int _numElements;
    int _numVertices;
    GLuint _vbo;
    cl_command_queue _clQueue;
    cl_mem _clMemory;

    bool _clMapped;

};

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OSD_CL_GL_VERTEX_BUFFER_H
