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

    _patchMap = new OsdCpuEvalLimitContext::PatchMap( *patchTables );

    size_t  npatches = patchTables->GetNumPatches(),
           nvertices = patchTables->GetNumControlVertices();

    _patchArrays.reserve(npatches);

    _patchBitFields.reserve(npatches);
    
    _patchBuffer.reserve(nvertices);


    // Full regular patches
    _AppendPatchArray(OsdPatchDescriptor(kRegular, 0, 0, 0, 0),
                      patchTables->GetFullRegularPatches(),
                      patchTables->GetFullRegularPtexCoordinates());

    _AppendPatchArray(OsdPatchDescriptor(kBoundary, 0, 0, 0, 0),
                      patchTables->GetFullBoundaryPatches(),
                      patchTables->GetFullBoundaryPtexCoordinates());

    _AppendPatchArray(OsdPatchDescriptor(kCorner, 0, 0, 0, 0),
                      patchTables->GetFullCornerPatches(),
                      patchTables->GetFullCornerPtexCoordinates());

    _AppendPatchArray(OsdPatchDescriptor(kGregory, 0, 0, patchTables->GetMaxValence(), 0),
                      patchTables->GetFullGregoryPatches(),
                      patchTables->GetFullGregoryPtexCoordinates());

    _AppendPatchArray(OsdPatchDescriptor(kBoundaryGregory, 0, 0, patchTables->GetMaxValence(), 0),
                      patchTables->GetFullBoundaryGregoryPatches(),
                      patchTables->GetFullBoundaryGregoryPtexCoordinates());

    _vertexValenceBuffer = patchTables->GetVertexValenceTable();
    _quadOffsetBuffer = patchTables->GetQuadOffsetTable();

    // Transition patches
    for (int p=0; p<5; ++p) {
    
        _AppendPatchArray(OsdPatchDescriptor(kTransitionRegular, p, 0, 0, 0),
                          patchTables->GetTransitionRegularPatches(p),
                          patchTables->GetTransitionRegularPtexCoordinates(p));

        for (int r=0; r<4; ++r) {
            _AppendPatchArray(OsdPatchDescriptor(kTransitionBoundary, p, r, 0, 0),
                              patchTables->GetTransitionBoundaryPatches(p, r),
                              patchTables->GetTransitionBoundaryPtexCoordinates(p, r));

            _AppendPatchArray(OsdPatchDescriptor(kTransitionCorner, p, r, 0, 0),
                              patchTables->GetTransitionCornerPatches(p, r),
                              patchTables->GetTransitionCornerPtexCoordinates(p, r));
        }
    }

}

OsdCpuEvalLimitContext::~OsdCpuEvalLimitContext() {
    delete _patchMap;
}

int
OsdCpuEvalLimitContext::_AppendPatchArray(
    OsdPatchDescriptor const & desc,
    FarPatchTables::PTable const & pTable,
    FarPatchTables::PtexCoordinateTable const & ptxTable )
{
    if (pTable.first.empty() or ptxTable.empty())
        return 0;

    OsdPatchArray array;
    array.desc = desc;
    array.firstIndex = (int)_patchBuffer.size();
    array.numIndices = (int)pTable.first.size();
    array.gregoryQuadOffsetBase = 0;
    //array.gregoryVertexValenceBase = gregoryQuadOffsetBase;

    // copy patch array descriptor
    _patchArrays.push_back(array);
    
    // copy patches bitfields
    for (int i=0; i<(int)ptxTable.size(); ++i) {
        _patchBitFields.push_back( ptxTable[i].bitField );
    }

    // copy control vertices indices
    _patchBuffer.insert( _patchBuffer.end(), pTable.first.begin(), pTable.first.end());
    
    return 1;
}

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv
