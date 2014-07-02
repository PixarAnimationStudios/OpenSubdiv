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

#include "../far/dispatcher.h"
#include "../osd/glDrawRegistry.h"
#include "../osd/glDrawContext.h"

#include "../osd/opengl.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

OsdGLDrawContext::OsdGLDrawContext() :
    _patchIndexBuffer(0), _patchParamTextureBuffer(0), _fvarDataTextureBuffer(0),
    _vertexTextureBuffer(0), _vertexValenceTextureBuffer(0), _quadOffsetsTextureBuffer(0)
{
}

OsdGLDrawContext::~OsdGLDrawContext()
{
    glDeleteBuffers(1, &_patchIndexBuffer);
    glDeleteTextures(1, &_vertexTextureBuffer);
    glDeleteTextures(1, &_vertexValenceTextureBuffer);
    glDeleteTextures(1, &_quadOffsetsTextureBuffer);
    glDeleteTextures(1, &_patchParamTextureBuffer);
    glDeleteTextures(1, &_fvarDataTextureBuffer);
}

bool
OsdGLDrawContext::SupportsAdaptiveTessellation()
{
#ifdef OSD_USES_GLEW
    // XXX: uncomment here to try tessellation on OSX
    // if (GLEW_ARB_tessellation_shader)
    //    return true;
#endif
    static const GLubyte *version = glGetString(GL_VERSION);
    if (version and version[0] == '4')
        return true;

    return false;
}

template <typename T> static GLuint 
createTextureBuffer(T const &data, GLint format, int offset=0)
{
    GLuint buffer = 0, texture = 0;

#if defined(GL_ARB_texture_buffer_object) || defined(GL_VERSION_3_1)
    glGenTextures(1, &texture);
    glGenBuffers(1, &buffer);

#if defined(GL_EXT_direct_state_access)
    if (glNamedBufferDataEXT and glTextureBufferEXT) {
        glNamedBufferDataEXT(buffer, (data.size()-offset) * sizeof(typename T::value_type),
                             &data[offset], GL_STATIC_DRAW);
        glTextureBufferEXT(texture, GL_TEXTURE_BUFFER, format, buffer);
    } else
#endif
#if defined(GLEW_ARB_texture_buffer_object)
    if (GLEW_ARB_texture_buffer_object) {
#else
    {
#endif
        glBindBuffer(GL_ARRAY_BUFFER, buffer);
        glBufferData(GL_ARRAY_BUFFER, (data.size()-offset) * sizeof(typename T::value_type),
                     &data[offset], GL_STATIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        glBindTexture(GL_TEXTURE_BUFFER, texture);
        glTexBuffer(GL_TEXTURE_BUFFER, format, buffer);
        glBindTexture(GL_TEXTURE_BUFFER, 0);
    }
    glDeleteBuffers(1, &buffer);

#endif

    return texture;
}

OsdGLDrawContext *
OsdGLDrawContext::Create(FarPatchTables const * patchTables, int numVertexElements, bool requireFVarData) {

    if (patchTables) {
        
        OsdGLDrawContext * result = new OsdGLDrawContext();
        
        if (result->create(patchTables, numVertexElements, requireFVarData)) {
            return result;
        } else {
            delete result;
        }
    }
    return NULL;
}

bool
OsdGLDrawContext::create(FarPatchTables const * patchTables, int numVertexElements, bool requireFVarData) {

    assert(patchTables);
         
    _isAdaptive = patchTables->IsFeatureAdaptive();
    
    // Process PTable
    FarPatchTables::PTable const & ptables = patchTables->GetPatchTable();

    glGenBuffers(1, &_patchIndexBuffer);

#if defined(GL_EXT_direct_state_access)
    if (glNamedBufferDataEXT) {
        glNamedBufferDataEXT(_patchIndexBuffer,
                             ptables.size() * sizeof(unsigned int), &ptables[0], GL_STATIC_DRAW);
    } else {
#else
    {
#endif
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _patchIndexBuffer);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                     ptables.size() * sizeof(unsigned int), &ptables[0], GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    }
    
    OsdDrawContext::ConvertPatchArrays(patchTables->GetPatchArrayVector(),
        patchArrays, patchTables->GetMaxValence(), numVertexElements);

    // allocate and initialize additional buffer data

#if defined(GL_ARB_texture_buffer_object) || defined(GL_VERSION_3_1)

#if defined(GLEW_ARB_texture_buffer_object)
    if (GLEW_ARB_texture_buffer_object) {
#else
    {
#endif

        // create vertex valence buffer and vertex texture
        FarPatchTables::VertexValenceTable const &
            valenceTable = patchTables->GetVertexValenceTable();

        if (not valenceTable.empty()) {
            _vertexValenceTextureBuffer = createTextureBuffer(valenceTable, GL_R32I);

            // also create vertex texture buffer (will be updated in UpdateVertexTexture())
            glGenTextures(1, &_vertexTextureBuffer);
        }


        // create quad offset table buffer
        FarPatchTables::QuadOffsetTable const &
            quadOffsetTable = patchTables->GetQuadOffsetTable();

        if (not quadOffsetTable.empty())
            _quadOffsetsTextureBuffer = createTextureBuffer(quadOffsetTable, GL_R32I);


        // create ptex coordinate buffer
        FarPatchTables::PatchParamTable const &
            patchParamTables = patchTables->GetPatchParamTable();

        if (not patchParamTables.empty())
            _patchParamTextureBuffer = createTextureBuffer(patchParamTables, GL_RG32I);


        // create fvar data buffer if requested
        std::vector<float> const &
            fvarData = patchTables->GetFVarData().GetAllData();

        if (requireFVarData and not fvarData.empty())
            _fvarDataTextureBuffer = createTextureBuffer(fvarData, GL_R32F);

        glBindBuffer(GL_TEXTURE_BUFFER, 0);
    }
#endif

    return true;
}

void
OsdGLDrawContext::updateVertexTexture(GLuint vbo)
{
#if defined(GL_ARB_texture_buffer_object) || defined(GL_VERSION_3_1)

#if defined(GL_EXT_direct_state_access)
    if (glTextureBufferEXT) {
        glTextureBufferEXT(_vertexTextureBuffer, GL_TEXTURE_BUFFER, GL_R32F, vbo);
    } else
#endif
#if defined(GLEW_ARB_texture_buffer_object)
    if (GLEW_ARB_texture_buffer_object) {
#else
    {
#endif
        glBindTexture(GL_TEXTURE_BUFFER, _vertexTextureBuffer);
        glTexBuffer(GL_TEXTURE_BUFFER, GL_R32F, vbo);
        glBindTexture(GL_TEXTURE_BUFFER, 0);
    }

#endif
}


} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv
