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

#include <cstring>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

DrawContext::~DrawContext() {}

void
DrawContext::ConvertPatchArrays(Far::PatchTables const &patchTables,
    PatchArrayVector &osdPatchArrays, int maxValence, int numElements) {

    // create patch arrays for drawing (while duplicating subpatches for transition patch arrays)
    static int subPatchCounts[] = { 1, 3, 4, 4, 4, 2 }; // number of subpatches for patterns

    int numTotalPatchArrays = 0;
    for (int array=0; array < patchTables.GetNumPatchArrays(); ++array) {

        Far::PatchDescriptor::TransitionPattern pattern =
            patchTables.GetPatchArrayDescriptor(array).GetPattern();

        numTotalPatchArrays += subPatchCounts[(int)pattern];
    }

    // allocate drawing patch arrays
    osdPatchArrays.clear();
    osdPatchArrays.reserve(numTotalPatchArrays);

    int narrays = patchTables.GetNumPatchArrays();
    for (int array=0, pidx=0, vidx=0, qidx=0; array<narrays; ++array) {

        Far::PatchDescriptor srcDesc = patchTables.GetPatchArrayDescriptor(array);

        int npatches = patchTables.GetNumPatches(array),
            nsubpatches = subPatchCounts[(int)srcDesc.GetPattern()],
            nverts = srcDesc.GetNumControlVertices();

        for (int i = 0; i < nsubpatches; ++i) {

            PatchDescriptor desc(srcDesc, maxValence, i, numElements);

            osdPatchArrays.push_back(PatchArray(desc, npatches, vidx, pidx, qidx));
        }

        vidx += npatches * nverts;
        pidx += npatches;
        qidx += (srcDesc.GetType() == Far::PatchDescriptor::GREGORY) ? npatches*nverts  : 0;
    }
}

// note : it is likely that Far::PatchTables::GetPatchControlVerticesTable()
//        will eventually be deprecated if control vertices cannot be kept
//        in a single linear array of indices. This function will help
//        packing patch control vertices for GPU buffers.
void
DrawContext::packPatchVerts(Far::PatchTables const & patchTables,
    std::vector<Index> & dst) {

    dst.resize(patchTables.GetNumControlVerticesTotal());
    Index * ptr = &dst[0];

    int narrays = patchTables.GetNumPatchArrays();
    for (int array=0; array<narrays; ++array) {
        Far::ConstIndexArray verts = patchTables.GetPatchArrayVertices(array);
        memcpy(ptr, verts.begin(), verts.size()*sizeof(Index));
        ptr += verts.size();
    }
}

void
DrawContext::packFVarData(Far::PatchTables const & patchTables,
    int fvarWidth, FVarData const & src, FVarData & dst) {

    assert(fvarWidth and (not src.empty()));

    Far::PatchTables::FVarPatchTables const * fvarPatchTables =
        patchTables.GetFVarPatchTables();
    assert(fvarPatchTables);

    // OsdMesh only accesses channel 0
    std::vector<Far::Index> const & indices = fvarPatchTables->GetPatchVertices(0);

    dst.resize(indices.size() * fvarWidth);
    float * ptr = &dst[0];

    for (int fvert=0; fvert<(int)indices.size(); ++fvert, ptr+=fvarWidth) {

        int index = indices[fvert] * fvarWidth;
        assert(index<(int)src.size());

        memcpy(ptr, &src[index], fvarWidth*sizeof(float));
    }
}

}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv
