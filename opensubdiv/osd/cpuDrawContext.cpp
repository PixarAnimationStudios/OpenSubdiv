//
//     Copyright (c) 2013, Motorola Mobility LLC.  All Rights Reserved.
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

#include "../osd/cpuDrawContext.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

OsdCpuDrawContext *
OsdCpuDrawContext::Create(FarPatchTables const * patchTables,
    bool requireFVarData)
{
    if (patchTables) {
        OsdCpuDrawContext * result = new OsdCpuDrawContext();
        
        if (result->create(patchTables, requireFVarData)) {
            return result;
        } else {
            delete result;
        }
    }
    return NULL;
}

OsdCpuDrawContext::~OsdCpuDrawContext()
{
}

OsdCpuDrawContext::OsdCpuDrawContext()
{
}

bool
OsdCpuDrawContext::create(FarPatchTables const * patchTables,
    bool requireFVarData)
{
    assert(patchTables);
         
    _isAdaptive = patchTables->IsFeatureAdaptive();

    OsdDrawContext::ConvertPatchArrays(patchTables->GetPatchArrayVector(),
        patchArrays, patchTables->GetMaxValence(), 0);

    _patchIndexBuffer = patchTables->GetPatchTable();
    _vertexValenceBuffer = patchTables->GetVertexValenceTable();
    _quadOffsetBuffer = patchTables->GetQuadOffsetTable();
    _patchParamBuffer = reinterpret_cast<const std::vector<unsigned int>&>(
        patchTables->GetPatchParamTable());
    _fvarDataBuffer = patchTables->GetFVarDataTable();

    return true;
}

}  // namespace OPENSUBDIV_VERSION
}  // namespace OpenSubdiv
