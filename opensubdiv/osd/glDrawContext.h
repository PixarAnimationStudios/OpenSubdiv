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

#ifndef OSD_GL_DRAW_CONTEXT_H
#define OSD_GL_DRAW_CONTEXT_H

#include "../version.h"

#include "../far/patchTables.h"
#include "../osd/drawContext.h"
#include "../osd/drawRegistry.h"
#include "../osd/vertex.h"

#include "../osd/opengl.h"

#include <map>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

/// \brief OpenGL specialized DrawContext class
///
/// GLDrawContext implements the OSD drawing interface with the OpenGL API.
/// Some functionality may be disabled depending on compile and run-time driver
/// support.
///
/// Contexts interface the serialized topological data pertaining to the 
/// geometric primitives with the capabilities of the selected discrete 
/// compute device.
///
class GLDrawContext : public DrawContext {
public:
    typedef GLuint VertexBufferBinding;

    virtual ~GLDrawContext();

    /// \brief Create an GLDraContext from Far::PatchTables
    ///
    /// @param patchTables          a valid set of Far::PatchTables
    ///
    /// @param numVertexElements    the number of vertex elements
    ///
    static GLDrawContext * Create(Far::PatchTables const * patchTables, int numVertexElements);

    /// Set vbo as a vertex texture (for gregory patch drawing)
    ///
    /// @param vbo  the vertex buffer object to update
    ///
    template<class VERTEX_BUFFER>
    void UpdateVertexTexture(VERTEX_BUFFER *vbo) {
        if (vbo)
            updateVertexTexture(vbo->BindVBO());
    }

    /// true if the GL version detected supports shader tessellation
    static bool SupportsAdaptiveTessellation();

    /// Returns the GL texture buffer containing the patch control vertices array
    GLuint GetPatchIndexBuffer() const {
        return _patchIndexBuffer;
    }

    /// Returns the GL texture buffer containing the patch local parameterization
    /// data
    GLuint GetPatchParamTextureBuffer() const {
        return _patchParamTextureBuffer;
    }

    /// Returns the GL texture buffer containing the vertex data
    GLuint GetVertexTextureBuffer() const {
        return _vertexTextureBuffer;
    }

    /// Returns the GL texture buffer containing patch vertex valence data (only
    /// used by Gregory patches)
    GLuint GetVertexValenceTextureBuffer() const {
        return _vertexValenceTextureBuffer;
    }

    /// Returns the GL texture buffer containing patch quad offsets data (only
    /// used by Gregory patches)
    GLuint GetQuadOffsetsTextureBuffer() const {
        return _quadOffsetsTextureBuffer;
    }

    /// Returns the GL texture buffer containing fvar data
    GLuint GetFvarDataTextureBuffer() const {
        return _fvarDataTextureBuffer;
    }

    /// Sets face-varying data buffer
    ///
    /// @param patchTables      A valid set of Far::PatchTables
    ///
    /// @param fvarWidth        Total face-varying primvar data width in fvarData
    ///
    /// @param fvarData         Vector containing the face-varying data
    ///
    /// @return                 True if the operation was successful
    ///
    bool SetFVarDataTexture(Far::PatchTables const & patchTables,
                            int fvarWidth, FVarData const & fvarData);

protected:

    GLuint _patchIndexBuffer;

    GLuint _patchParamTextureBuffer;
    GLuint _fvarDataTextureBuffer;

    GLuint _vertexTextureBuffer;
    GLuint _vertexValenceTextureBuffer;
    GLuint _quadOffsetsTextureBuffer;

    GLDrawContext();

    // allocate buffers from patchTables
    bool create(Far::PatchTables const & patchTables, int numElements);

    void updateVertexTexture(GLuint vbo);
};

}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* OSD_DRAW_CONTEXT_H */
