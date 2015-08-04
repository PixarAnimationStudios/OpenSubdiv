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

#include "../osd/cpuPatchTable.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

CpuPatchTable::CpuPatchTable(const Far::PatchTable *farPatchTable) {
    int nPatchArrays = farPatchTable->GetNumPatchArrays();

    // count
    int numPatches = 0;
    int numIndices = 0;
    for (int j = 0; j < nPatchArrays; ++j) {
        int nPatch = farPatchTable->GetNumPatches(j);
        int nCV = farPatchTable->GetPatchArrayDescriptor(j).GetNumControlVertices();
        numPatches += nPatch;
        numIndices += nPatch * nCV;
    }
    _patchArrays.reserve(nPatchArrays);
    _indexBuffer.reserve(numIndices);
    _patchParamBuffer.reserve(numPatches);

    // for each patchArray
    for (int j = 0; j < nPatchArrays; ++j) {
        PatchArray patchArray(farPatchTable->GetPatchArrayDescriptor(j),
                              farPatchTable->GetNumPatches(j),
                              (int)_indexBuffer.size(),
                              (int)_patchParamBuffer.size());
        _patchArrays.push_back(patchArray);

        // indices
        Far::ConstIndexArray indices = farPatchTable->GetPatchArrayVertices(j);
        for (int k = 0; k < indices.size(); ++k) {
            _indexBuffer.push_back(indices[k]);
        }

        // patchParams bundling
        // XXX: this process won't be needed if Far::PatchParam includes
        // sharpness.
#if 0
        // XXX: we need sharpness interface for patcharray or put sharpness
        //      into patchParam.
        Far::ConstPatchParamArray patchParams =
            farPatchTable->GetPatchParams(j);
        for (int k = 0; k < patchParams.size(); ++k) {
            float sharpness = 0.0;
            _patchParamBuffer.push_back(patchParams[k].field0);
            _patchParamBuffer.push_back(patchParams[k].field1);
            _patchParamBuffer.push_back(*((unsigned int *)&sharpness));
        }
#else
        // XXX: workaround. GetPatchParamTable() will be deprecated though.
        Far::PatchParamTable const & patchParamTable =
            farPatchTable->GetPatchParamTable();
        std::vector<Far::Index> const &sharpnessIndexTable =
            farPatchTable->GetSharpnessIndexTable();
        int numPatchesJ = farPatchTable->GetNumPatches(j);
        for (int k = 0; k < numPatchesJ; ++k) {
            float sharpness = 0.0;
            int patchIndex = (int)_patchParamBuffer.size();
            if (patchIndex < (int)sharpnessIndexTable.size()) {
                int sharpnessIndex = sharpnessIndexTable[patchIndex];
                if (sharpnessIndex >= 0)
                    sharpness = farPatchTable->GetSharpnessValues()[sharpnessIndex];
            }
            PatchParam param;
            //param.patchParam = patchParamTable[patchIndex];
            param.field0 = patchParamTable[patchIndex].field0;
            param.field1 = patchParamTable[patchIndex].field1;
            param.sharpness = sharpness;
            _patchParamBuffer.push_back(param);
        }
#endif
    }
}

}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv

