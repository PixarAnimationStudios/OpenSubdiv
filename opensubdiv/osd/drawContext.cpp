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

#include "../osd/drawContext.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

OsdDrawContext::~OsdDrawContext() {}

void
OsdDrawContext::ConvertPatchArrays(FarPatchTables::PatchArrayVector const &farPatchArrays,
                                   OsdDrawContext::PatchArrayVector &osdPatchArrays,
                                   int maxValence, int numElements)
{
    // create patch arrays for drawing (while duplicating subpatches for transition patch arrays)
    static int subPatchCounts[] = { 1, 3, 4, 4, 4, 2 }; // number of subpatches for patterns

    int numTotalPatchArrays = 0;
    for (int i = 0; i < (int)farPatchArrays.size(); ++i) {
        FarPatchTables::TransitionPattern pattern = farPatchArrays[i].GetDescriptor().GetPattern();
        numTotalPatchArrays += subPatchCounts[(int)pattern];
    }

    // allocate drawing patch arrays
    osdPatchArrays.clear();
    osdPatchArrays.reserve(numTotalPatchArrays);

    for (int i = 0; i < (int)farPatchArrays.size(); ++i) {
        FarPatchTables::TransitionPattern pattern = farPatchArrays[i].GetDescriptor().GetPattern();
        int numSubPatches = subPatchCounts[(int)pattern];

        FarPatchTables::PatchArray const &parray = farPatchArrays[i];
        FarPatchTables::Descriptor srcDesc = parray.GetDescriptor();

        for (int j = 0; j < numSubPatches; ++j) {
            PatchDescriptor desc(srcDesc, maxValence, j, numElements);

            osdPatchArrays.push_back(PatchArray(desc, parray.GetArrayRange()));
        }
    }
}

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv


