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
#ifndef OSD_GL_DRAW_CONTEXT_H
#define OSD_GL_DRAW_CONTEXT_H

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
    #include <GL/gl.h>
#endif

#include "../version.h"

#include "../far/mesh.h"
#include "../osd/drawContext.h"
#include "../osd/drawRegistry.h"
#include "../osd/vertex.h"

#include <map>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

/// \brief OpenGL specialized DrawContext class
///
/// OsdGLDrawContext implements the OSD drawing interface with the OpenGL API.
/// Some functionality may be disabled depending on compile and run-time driver
/// support.
///
/// Contexts interface the serialized topological data pertaining to the 
/// geometric primitives with the capabilities of the selected discrete 
/// compute device.
///
class OsdGLDrawContext : public OsdDrawContext {
public:
    typedef GLuint VertexBufferBinding;

    virtual ~OsdGLDrawContext();

    /// \brief Create an OsdGLDraContext from FarPatchTables
    ///
    /// @param patchTables      a valid set of FarPatchTables
    ///
    /// @param requireFVarData  set to true to enable face-varying data to be 
    ///                         carried over from the Far data structures.
    ///
    static OsdGLDrawContext * Create(FarPatchTables const * patchTables, bool requireFVarData);

    /// Set vbo as a vertex texture (for gregory patch drawing)
    ///
    /// @param vbo  the vertex buffer object to update
    ///
    template<class VERTEX_BUFFER>
    void UpdateVertexTexture(VERTEX_BUFFER *vbo) {
        updateVertexTexture(vbo->BindVBO(), vbo->GetNumElements());
    }

    /// true if the GL version detected supports shader tessellation
    static bool SupportsAdaptiveTessellation();

    GLuint patchIndexBuffer;

#if defined(GL_ES_VERSION_2_0)
    GLuint patchTrianglesIndexBuffer;
#endif

    GLuint ptexCoordinateTextureBuffer;
    GLuint fvarDataTextureBuffer;

    GLuint vertexTextureBuffer;
    GLuint vertexValenceTextureBuffer;
    GLuint quadOffsetTextureBuffer;

protected:
    OsdGLDrawContext();

    // allocate buffers from patchTables
    bool create(FarPatchTables const *patchTables, bool requireFVarData);

    void updateVertexTexture(GLuint vbo, int numElements);
};

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* OSD_DRAW_CONTEXT_H */
