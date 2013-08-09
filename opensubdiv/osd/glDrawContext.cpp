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

#include "../far/dispatcher.h"
#include "../far/loopSubdivisionTables.h"
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
    glBindBuffer(GL_ARRAY_BUFFER, buffer);
    glBufferData(GL_ARRAY_BUFFER, (data.size()-offset) * sizeof(typename T::value_type),
                 &data[offset], GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindTexture(GL_TEXTURE_BUFFER, texture);
    glTexBuffer(GL_TEXTURE_BUFFER, format, buffer);
    glBindTexture(GL_TEXTURE_BUFFER, 0);
    glDeleteBuffers(1, &buffer);
#endif

    return texture;
}

OsdGLDrawContext *
OsdGLDrawContext::Create(FarPatchTables const * patchTables, bool requireFVarData) {

    if (patchTables) {
        
        OsdGLDrawContext * result = new OsdGLDrawContext();
        
        if (result->create(patchTables, requireFVarData)) {
            return result;
        } else {
            delete result;
        }
    }
    return NULL;
}

bool
OsdGLDrawContext::create(FarPatchTables const * patchTables, bool requireFVarData) {

    assert(patchTables);
         
    _isAdaptive = patchTables->IsFeatureAdaptive();
    
    // Process PTable
    FarPatchTables::PTable const & ptables = patchTables->GetPatchTable();

    glGenBuffers(1, &_patchIndexBuffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _patchIndexBuffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 ptables.size() * sizeof(unsigned int), &ptables[0], GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    
    OsdDrawContext::ConvertPatchArrays(patchTables->GetPatchArrayVector(),
        patchArrays, patchTables->GetMaxValence(), 0);

/*    
#if defined(GL_ES_VERSION_2_0)
        // XXX: farmesh should have FarDensePatchTable for dense mesh indices.
        //      instead of GetFaceVertices().
        const FarSubdivisionTables<OsdVertex> *tables = farMesh->GetSubdivisionTables();
        int level = tables->GetMaxLevel();
        const std::vector<int> &indices = farMesh->GetFaceVertices(level-1);

        int numIndices = (int)indices.size();

        // Allocate and fill index buffer.
        glGenBuffers(1, &_patchIndexBuffer);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _patchIndexBuffer);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                     numIndices * sizeof(unsigned int), &(indices[0]), GL_STATIC_DRAW);


        // OpenGLES 2 supports only triangle topologies for filled
        // primitives i.e. not QUADS or PATCHES or LINES_ADJACENCY
        // For the convenience of clients build build a triangles
        // index buffer by splitting quads.
        int numQuads = indices.size() / 4;
        int numTrisIndices = numQuads * 6;

        std::vector<short> trisIndices;
        trisIndices.reserve(numTrisIndices);
        for (int i=0; i<numQuads; ++i) {
            const int * quad = &indices[i*4];
            trisIndices.push_back(short(quad[0]));
            trisIndices.push_back(short(quad[1]));
            trisIndices.push_back(short(quad[2]));

            trisIndices.push_back(short(quad[2]));
            trisIndices.push_back(short(quad[3]));
            trisIndices.push_back(short(quad[0]));
        }

        // Allocate and fill triangles index buffer.
        glGenBuffers(1, &patchTrianglesIndexBuffer);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, patchTrianglesIndexBuffer);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                     numTrisIndices * sizeof(short), &(trisIndices[0]), GL_STATIC_DRAW);
#endif
*/    
    
    // allocate and initialize additional buffer data

#if defined(GL_ARB_texture_buffer_object) || defined(GL_VERSION_3_1)
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
    FarPatchTables::FVarDataTable const &
        fvarTables = patchTables->GetFVarDataTable();

    if (requireFVarData and not fvarTables.empty())
        _fvarDataTextureBuffer = createTextureBuffer(fvarTables, GL_R32F);
#endif

    glBindBuffer(GL_TEXTURE_BUFFER, 0);

    return true;
}

void
OsdGLDrawContext::updateVertexTexture(GLuint vbo, int numVertexElements)
{
#if defined(GL_ARB_texture_buffer_object) || defined(GL_VERSION_3_1)
    glBindTexture(GL_TEXTURE_BUFFER, _vertexTextureBuffer);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_R32F, vbo);
    glBindTexture(GL_TEXTURE_BUFFER, 0);
#endif

    // XXX: consider moving this proc to base class
    // updating num elements in descriptor with new vbo specs
    for (int i = 0; i < (int)patchArrays.size(); ++i) {
        PatchArray &parray = patchArrays[i];
        PatchDescriptor desc = parray.GetDescriptor();
        desc.SetNumElements(numVertexElements);
        parray.SetDescriptor(desc);
    }
}


} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv
