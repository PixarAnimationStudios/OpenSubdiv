//
//   Copyright 2015 Pixar
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

#include "../osd/glLegacyGregoryPatchTable.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

GLLegacyGregoryPatchTable::GLLegacyGregoryPatchTable() :
    _vertexTextureBuffer(0), _vertexValenceTextureBuffer(0),
    _quadOffsetsTextureBuffer(0) {
    _quadOffsetsBase[0] = _quadOffsetsBase[1] = 0;
}

GLLegacyGregoryPatchTable::~GLLegacyGregoryPatchTable() {
    if (_vertexTextureBuffer)
        glDeleteTextures(1, &_vertexTextureBuffer);
    if (_vertexValenceTextureBuffer)
        glDeleteTextures(1, &_vertexValenceTextureBuffer);
    if (_quadOffsetsTextureBuffer)
        glDeleteTextures(1, &_quadOffsetsTextureBuffer);
}

GLLegacyGregoryPatchTable *
GLLegacyGregoryPatchTable::Create(Far::PatchTables const *farPatchTables) {

    GLLegacyGregoryPatchTable *result = new GLLegacyGregoryPatchTable();
    glGenTextures(1, &result->_vertexTextureBuffer);
    glGenTextures(1, &result->_vertexValenceTextureBuffer);
    glGenTextures(1, &result->_quadOffsetsTextureBuffer);

    Far::PatchTables::VertexValenceTable const &
        valenceTable = farPatchTables->GetVertexValenceTable();
    Far::PatchTables::QuadOffsetsTable const &
        quadOffsetsTable = farPatchTables->GetQuadOffsetsTable();

    GLuint buffers[2];
    glGenBuffers(2, buffers);
    glBindBuffer(GL_ARRAY_BUFFER, buffers[0]);
    glBufferData(GL_ARRAY_BUFFER, valenceTable.size() * sizeof(int),
                 &valenceTable[0], GL_STATIC_DRAW);

    glBindTexture(GL_TEXTURE_BUFFER, result->_vertexValenceTextureBuffer);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_R32I, buffers[0]);
    glBindTexture(GL_TEXTURE_BUFFER, 0);

    glBindBuffer(GL_ARRAY_BUFFER, buffers[1]);
    glBufferData(GL_ARRAY_BUFFER, quadOffsetsTable.size() * sizeof(int),
                 &quadOffsetsTable[0], GL_STATIC_DRAW);

    glBindTexture(GL_TEXTURE_BUFFER, result->_quadOffsetsTextureBuffer);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_R32I, buffers[1]);
    glBindTexture(GL_TEXTURE_BUFFER, 0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glDeleteBuffers(2, buffers);

    result->_quadOffsetsBase[0] = 0;
    result->_quadOffsetsBase[1] = 0;
    // scan patchtable to find quadOffsetsBase.
    for (int i = 0; i < farPatchTables->GetNumPatchArrays(); ++i) {
        // GREGORY_BOUNDARY's quadoffsets come after GREGORY's.
        if (farPatchTables->GetPatchArrayDescriptor(i) ==
            Far::PatchDescriptor::GREGORY) {
            result->_quadOffsetsBase[1] = farPatchTables->GetNumPatches(i) * 4;
            break;
        }
    }
    return result;
}

void
GLLegacyGregoryPatchTable::UpdateVertexBuffer(GLuint vbo) {
    glBindTexture(GL_TEXTURE_BUFFER, _vertexTextureBuffer);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_R32F, vbo);
    glBindTexture(GL_TEXTURE_BUFFER, 0);
}

}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
