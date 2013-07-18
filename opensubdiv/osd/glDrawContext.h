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
#ifndef OSD_GL_DRAW_CONTEXT_H
#define OSD_GL_DRAW_CONTEXT_H

#include "../version.h"

#include "../far/mesh.h"
#include "../osd/drawContext.h"
#include "../osd/drawRegistry.h"
#include "../osd/vertex.h"

#include "../osd/opengl.h"

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

    /// Returns the GL texture buffer containing the patch control vertices array
    GLuint GetPatchIndexBuffer() const {
        return _patchIndexBuffer;
    }

#if defined(GL_ES_VERSION_2_0)
    /// Returns the GL a VBO containing a triangulated version of the mesh
    GLuint GetPatchTrianglesIndexBUffer() const {
        return _patchTrianglesIndexBuffer;
    }
#endif

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

protected:
    GLuint _patchIndexBuffer;

#if defined(GL_ES_VERSION_2_0)
    GLuint _patchTrianglesIndexBuffer;
#endif

    GLuint _patchParamTextureBuffer;
    GLuint _fvarDataTextureBuffer;

    GLuint _vertexTextureBuffer;
    GLuint _vertexValenceTextureBuffer;
    GLuint _quadOffsetsTextureBuffer;

    OsdGLDrawContext();

    // allocate buffers from patchTables
    bool create(FarPatchTables const *patchTables, bool requireFVarData);

    void updateVertexTexture(GLuint vbo, int numElements);
};

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* OSD_DRAW_CONTEXT_H */
