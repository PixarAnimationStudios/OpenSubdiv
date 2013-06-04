//
//     Copyright (C) Pixar. All rights reserved.
//
//     This license governs use of the accompanying software. If you
//     use the software, you accept this license. If you do not accept
//     the license, do not use the software.
//
//     1. Definitions
//     The terms "reproduce," "reproduction," "derivative works," and
//     "distribution" have the same meaning here as under U.S.
//     copyright law.  A "contribution" is the original software, or
//     any additions or changes to the software.
//     A "contributor" is any person or entity that distributes its
//     contribution under this license.
//     "Licensed patents" are a contributor's patent claims that read
//     directly on its contribution.
//
//     2. Grant of Rights
//     (A) Copyright Grant- Subject to the terms of this license,
//     including the license conditions and limitations in section 3,
//     each contributor grants you a non-exclusive, worldwide,
//     royalty-free copyright license to reproduce its contribution,
//     prepare derivative works of its contribution, and distribute
//     its contribution or any derivative works that you create.
//     (B) Patent Grant- Subject to the terms of this license,
//     including the license conditions and limitations in section 3,
//     each contributor grants you a non-exclusive, worldwide,
//     royalty-free license under its licensed patents to make, have
//     made, use, sell, offer for sale, import, and/or otherwise
//     dispose of its contribution in the software or derivative works
//     of the contribution in the software.
//
//     3. Conditions and Limitations
//     (A) No Trademark License- This license does not grant you
//     rights to use any contributor's name, logo, or trademarks.
//     (B) If you bring a patent claim against any contributor over
//     patents that you claim are infringed by the software, your
//     patent license from such contributor to the software ends
//     automatically.
//     (C) If you distribute any portion of the software, you must
//     retain all copyright, patent, trademark, and attribution
//     notices that are present in the software.
//     (D) If you distribute any portion of the software in source
//     code form, you may do so only under this license by including a
//     complete copy of this license with your distribution. If you
//     distribute any portion of the software in compiled or object
//     code form, you may only do so under a license that complies
//     with this license.
//     (E) The software is licensed "as-is." You bear the risk of
//     using it. The contributors give no express warranties,
//     guarantees or conditions. You may have additional consumer
//     rights under your local laws which this license cannot change.
//     To the extent permitted under your local laws, the contributors
//     exclude the implied warranties of merchantability, fitness for
//     a particular purpose and non-infringement.
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
OsdCpuEvalLimitContext::Create(FarMesh<OsdVertex> const * farmesh) {

    assert(farmesh);
    
    // we do not support uniform yet
    if (not farmesh->GetPatchTables())
        return NULL;
                                          
    return new OsdCpuEvalLimitContext(farmesh);
}

OsdCpuEvalLimitContext::OsdCpuEvalLimitContext(FarMesh<OsdVertex> const * farmesh) :
    OsdEvalLimitContext(farmesh) {
    
    FarPatchTables const * patchTables = farmesh->GetPatchTables();
    assert(patchTables);

    // copy the data from the FarTables
    _patches = patchTables->GetPatchTable();

    _patchArrays = patchTables->GetPatchArrayVector();
    
    _vertexValenceBuffer = patchTables->GetVertexValenceTable();
    
    _quadOffsetBuffer = patchTables->GetQuadOffsetTable();
    
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
    

    _patchMap = new FarPatchTables::PatchMap( *patchTables );
}

OsdCpuEvalLimitContext::~OsdCpuEvalLimitContext() {
    delete _patchMap;
}

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv
