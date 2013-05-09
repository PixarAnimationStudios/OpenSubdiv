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

#include "../osd/drawContext.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

OsdDrawContext::~OsdDrawContext() {}

void
OsdDrawContext::createPatchArrays(FarPatchTables const * patchTables, int numElements) {

    int maxValence = patchTables->GetMaxValence();

    // create patch arrays for drawing (while duplicating subpatches for transition patch arrays)
    static int subPatchCounts[] = { 1, 3, 4, 4, 4, 2 }; // number of subpatches for patterns

    FarPatchTables::PatchArrayVector const & srcPatchArrays = patchTables->GetAllPatchArrays();

    int numTotalPatchArrays = 0;
    for (int i = 0; i < (int)srcPatchArrays.size(); ++i) {
        FarPatchTables::TransitionPattern pattern = srcPatchArrays[i].GetDescriptor().GetPattern();
        numTotalPatchArrays += subPatchCounts[(int)pattern];
    }

    // allocate drawing patch arrays
    patchArrays.reserve(numTotalPatchArrays);
    for (int i = 0; i < (int)srcPatchArrays.size(); ++i) {
        FarPatchTables::TransitionPattern pattern = srcPatchArrays[i].GetDescriptor().GetPattern();
        int numSubPatches = subPatchCounts[(int)pattern];

        FarPatchTables::PatchArray const &parray = srcPatchArrays[i];
        FarPatchTables::Descriptor srcDesc = parray.GetDescriptor();

        for (int j = 0; j < numSubPatches; ++j) {
            FarPatchTables::Descriptor desc(srcDesc.GetType(),
                                            srcDesc.GetPattern(),
                                            srcDesc.GetRotation(),
                                            maxValence,
                                            j,
                                            numElements);

            patchArrays.push_back(FarPatchTables::PatchArray(desc,
                                                             parray.GetVertIndex(),
                                                             parray.GetPatchIndex(),
                                                             parray.GetNumPatches(),
                                                             parray.GetQuadOffsetIndex()));
        }
    }
}

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv


