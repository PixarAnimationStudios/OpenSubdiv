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

#include "../osd/glPatchTable.h"

#include "../far/patchTable.h"
#include "../osd/opengl.h"
#include "../osd/cpuPatchTable.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

GLPatchTable::GLPatchTable() :
    _patchIndexBuffer(0), _patchParamBuffer(0),
    _patchIndexTexture(0), _patchParamTexture(0) {
}

GLPatchTable::~GLPatchTable() {
    if (_patchIndexBuffer) glDeleteBuffers(1, &_patchIndexBuffer);
    if (_patchParamBuffer) glDeleteBuffers(1, &_patchParamBuffer);
    if (_patchIndexTexture) glDeleteTextures(1, &_patchIndexTexture);
    if (_patchParamTexture) glDeleteTextures(1, &_patchParamTexture);
}

GLPatchTable *
GLPatchTable::Create(Far::PatchTable const *farPatchTable,
                     void * /*deviceContext*/) {
    GLPatchTable *instance = new GLPatchTable();
    if (instance->allocate(farPatchTable)) return instance;
    delete instance;
    return 0;
}

bool
GLPatchTable::allocate(Far::PatchTable const *farPatchTable) {
    glGenBuffers(1, &_patchIndexBuffer);
    glGenBuffers(1, &_patchParamBuffer);

    CpuPatchTable patchTable(farPatchTable);

    size_t numPatchArrays = patchTable.GetNumPatchArrays();
    GLsizei indexSize = (GLsizei)patchTable.GetPatchIndexSize();
    GLsizei patchParamSize = (GLsizei)patchTable.GetPatchParamSize();

    // copy patch array
    _patchArrays.assign(patchTable.GetPatchArrayBuffer(),
                        patchTable.GetPatchArrayBuffer() + numPatchArrays);

    // copy index buffer
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _patchIndexBuffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 indexSize * sizeof(GLint),
                 patchTable.GetPatchIndexBuffer(),
                 GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    // copy patchparam buffer
    glBindBuffer(GL_ARRAY_BUFFER, _patchParamBuffer);
    glBufferData(GL_ARRAY_BUFFER,
                 patchParamSize * sizeof(PatchParam),
                 patchTable.GetPatchParamBuffer(),
                 GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // make both buffer as texture buffers too.
    glGenTextures(1, &_patchIndexTexture);
    glGenTextures(1, &_patchParamTexture);

    GLuint buffer;
    glGenBuffers(1, &buffer);
    glBindBuffer(GL_ARRAY_BUFFER, buffer);
    glBufferData(GL_ARRAY_BUFFER,
                 indexSize * sizeof(GLint),
                 patchTable.GetPatchIndexBuffer(),
                 GL_STATIC_DRAW);

    glBindTexture(GL_TEXTURE_BUFFER, _patchIndexTexture);
//    glTexBuffer(GL_TEXTURE_BUFFER, GL_R32I, _patchIndexBuffer);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_R32I, buffer);

    glBindTexture(GL_TEXTURE_BUFFER, _patchParamTexture);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_RGB32I, _patchParamBuffer);
    glBindTexture(GL_TEXTURE_BUFFER, 0);

    return true;
}


}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv

