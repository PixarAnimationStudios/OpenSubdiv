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

#include "../osd/cpuEvalLimitContext.h"
#include "../osd/vertexDescriptor.h"

#include <string.h>
#include <cassert>
#include <cstdio>
#include <cmath>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

OsdCpuEvalLimitContext *
OsdCpuEvalLimitContext::Create(FarMesh<OsdVertex> const * farmesh, bool requireFVarData) {

    assert(farmesh);
    
    // we do not support uniform yet
    if (not farmesh->GetPatchTables())
        return NULL;
                                          
    return new OsdCpuEvalLimitContext(farmesh, requireFVarData);
}

OsdCpuEvalLimitContext::OsdCpuEvalLimitContext(FarMesh<OsdVertex> const * farmesh, bool requireFVarData) :
    OsdEvalLimitContext(farmesh) {
    
    FarPatchTables const * patchTables = farmesh->GetPatchTables();
    assert(patchTables);

    // copy the data from the FarTables
    _patches = patchTables->GetPatchTable();

    _patchArrays = patchTables->GetPatchArrayVector();
    
    _vertexValenceTable = patchTables->GetVertexValenceTable();
    
    _quadOffsetTable = patchTables->GetQuadOffsetTable();
    
    _maxValence = patchTables->GetMaxValence();
    
    // Copy the bitfields, the faceId will be the key to our map
    int npatches = patchTables->GetNumPatches();
    
    _patchBitFields.reserve(npatches);

    FarPatchTables::PatchParamTable const & ptxTable =
        patchTables->GetPatchParamTable();

    if ( not ptxTable.empty() ) {

        FarPatchParam const * pptr = &ptxTable[0];

        for (int arrayId = 0; arrayId < (int)_patchArrays.size(); ++arrayId) {

            FarPatchTables::PatchArray const & pa = _patchArrays[arrayId];

            for (unsigned int j=0; j < pa.GetNumPatches(); ++j) {
                _patchBitFields.push_back( pptr++->bitField );
            }
        }
    }
    
    // Copy the face-varying table if necessary    
    if (requireFVarData) {
        _fvarwidth = farmesh->GetTotalFVarWidth();
        if (_fvarwidth>0) {
            _fvarData = patchTables->GetFVarDataTable();
        }
    }
    
    _patchMap = new FarPatchMap( *patchTables );
}

OsdCpuEvalLimitContext::~OsdCpuEvalLimitContext() {
    delete _patchMap;
}

void 
OsdCpuEvalLimitContext::VertexData::Unbind() {

    inDesc.Reset();
    in.Unbind();

    outDesc.Reset();
    out.Unbind();
    outDu.Unbind();
    outDv.Unbind();
}

void 
OsdCpuEvalLimitContext::VaryingData::Unbind() {

    inDesc.Reset();
    in.Unbind();

    outDesc.Reset();
    out.Unbind();
}

void 
OsdCpuEvalLimitContext::FaceVaryingData::Unbind() {

    inDesc.Reset();

    outDesc.Reset();
    out.Unbind();
}

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv
