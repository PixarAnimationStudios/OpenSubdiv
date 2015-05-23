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

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

GLPatchTable::GLPatchTable() :
    _indexBuffer(0), _patchParamTexture(0) {
}

GLPatchTable::~GLPatchTable() {
    if (_indexBuffer) glDeleteBuffers(1, &_indexBuffer);
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
    glGenBuffers(1, &_indexBuffer);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _indexBuffer);
    std::vector<int> buffer;
    std::vector<unsigned int> ppBuffer;

    // needs reserve?

    int nPatchArrays = farPatchTable->GetNumPatchArrays();

    // for each patchArray
    for (int j = 0; j < nPatchArrays; ++j) {
        PatchArray patchArray(farPatchTable->GetPatchArrayDescriptor(j),
                              farPatchTable->GetNumPatches(j),
                              (int)buffer.size(),
                              (int)ppBuffer.size()/3);
        _patchArrays.push_back(patchArray);

        // indices
        Far::ConstIndexArray indices = farPatchTable->GetPatchArrayVertices(j);
        for (int k = 0; k < indices.size(); ++k) {
            buffer.push_back(indices[k]);
        }

        // patchParams
#if 0
        // XXX: we need sharpness interface for patcharray or put sharpness
        //      into patchParam.
        Far::ConstPatchParamArray patchParams =
            farPatchTable->GetPatchParams(j);
        for (int k = 0; k < patchParams.size(); ++k) {
            float sharpness = 0.0;
            ppBuffer.push_back(patchParams[k].faceIndex);
            ppBuffer.push_back(patchParams[k].bitField.field);
            ppBuffer.push_back(*((unsigned int *)&sharpness));
        }
#else
        // XXX: workaround. GetPatchParamTable() will be deprecated though.
        Far::PatchParamTable const & patchParamTable =
            farPatchTable->GetPatchParamTable();
        std::vector<Far::Index> const &sharpnessIndexTable =
            farPatchTable->GetSharpnessIndexTable();
        int numPatches = farPatchTable->GetNumPatches(j);
        for (int k = 0; k < numPatches; ++k) {
            float sharpness = 0.0;
            int patchIndex = (int)ppBuffer.size()/3;
            if (patchIndex < (int)sharpnessIndexTable.size()) {
                int sharpnessIndex = sharpnessIndexTable[patchIndex];
                if (sharpnessIndex >= 0)
                    sharpness = farPatchTable->GetSharpnessValues()[sharpnessIndex];
            }
            ppBuffer.push_back(patchParamTable[patchIndex].faceIndex);
            ppBuffer.push_back(patchParamTable[patchIndex].bitField.field);
            ppBuffer.push_back(*((unsigned int *)&sharpness));
        }
#endif
    }

    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 (int)buffer.size()*sizeof(int), &buffer[0], GL_STATIC_DRAW);

    // patchParam is currently expected to be texture (it can be SSBO)
    GLuint texBuffer = 0;
    glGenBuffers(1, &texBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, texBuffer);
    glBufferData(GL_ARRAY_BUFFER, ppBuffer.size()*sizeof(unsigned int),
                 &ppBuffer[0], GL_STATIC_DRAW);

    glGenTextures(1, &_patchParamTexture);
    glBindTexture(GL_TEXTURE_BUFFER, _patchParamTexture);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_RGB32I, texBuffer);
    glBindTexture(GL_TEXTURE_BUFFER, 0);

    glDeleteBuffers(1, &texBuffer);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    return true;
}


}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv

