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
