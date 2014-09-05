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

namespace Osd {

CpuEvalLimitContext *
CpuEvalLimitContext::Create(Far::PatchTables const & patchTables, bool requireFVarData) {

    // there is no limit with uniform subdivision
    if (not patchTables.IsFeatureAdaptive())
        return NULL;
                                          
    return new CpuEvalLimitContext(patchTables, requireFVarData);
}

CpuEvalLimitContext::CpuEvalLimitContext(Far::PatchTables const & patchTables, bool requireFVarData) :
    EvalLimitContext(patchTables) {
    
    // copy the data from the FarTables
    _patches = patchTables.GetPatchTable();

    _patchArrays = patchTables.GetPatchArrayVector();
    
    _vertexValenceTable = patchTables.GetVertexValenceTable();
    
    _quadOffsetTable = patchTables.GetQuadOffsetTable();
    
    _maxValence = patchTables.GetMaxValence();
    
    // Copy the bitfields, the faceId will be the key to our map
    int npatches = patchTables.GetNumPatches();
    
    _patchBitFields.reserve(npatches);

    Far::PatchTables::PatchParamTable const & ptxTable =
        patchTables.GetPatchParamTable();

    if ( not ptxTable.empty() ) {

        Far::PatchParam const * pptr = &ptxTable[0];

        for (int arrayId = 0; arrayId < (int)_patchArrays.size(); ++arrayId) {

            Far::PatchTables::PatchArray const & pa = _patchArrays[arrayId];

            for (unsigned int j=0; j < pa.GetNumPatches(); ++j) {
                _patchBitFields.push_back( pptr++->bitField );
            }
        }
    }
    
    // Copy the face-varying table if necessary    
    if (requireFVarData) {
/* XXXX manuelk do fvar stuff here
        _fvarwidth = patchTables.GetFVarData().GetFVarWidth();
        if (_fvarwidth>0) {
            _fvarData = patchTables.GetFVarData().GetAllData();
        }
*/        
    }
    
    _patchMap = new Far::PatchMap( patchTables );
}

CpuEvalLimitContext::~CpuEvalLimitContext() {
    delete _patchMap;
}

} // end namespace Osd

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv
