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

#if defined(__APPLE__)
    #include "TargetConditionals.h"
    #if TARGET_OS_IPHONE or TARGET_IPHONE_SIMULATOR
        #include <OpenGLES/ES2/gl.h>
    #else
        #include <OpenGL/gl3.h>
    #endif
#elif defined(ANDROID)
    #include <GLES2/gl2.h>
#else
    #if defined(_WIN32)
        #include <windows.h>
    #endif
    #include <GL/glew.h>
#endif

#include "../osd/glVertexBuffer.h"

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

