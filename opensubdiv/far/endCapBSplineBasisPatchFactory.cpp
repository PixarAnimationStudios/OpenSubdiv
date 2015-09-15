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

#include "../far/gregoryBasis.h"
#include "../far/endCapBSplineBasisPatchFactory.h"
#include "../far/error.h"
#include "../far/stencilTableFactory.h"
#include "../far/topologyRefiner.h"

#include <cassert>
#include <cmath>
#include <cstring>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Far {

namespace {
#ifdef __INTEL_COMPILER
#pragma warning (push)
#pragma warning disable 1572
#endif
    inline bool isWeightNonZero(float w) { return (w != 0.0f); }
#ifdef __INTEL_COMPILER
#pragma warning (pop)
#endif
}

EndCapBSplineBasisPatchFactory::EndCapBSplineBasisPatchFactory(
    TopologyRefiner const & refiner) :
    _refiner(&refiner), _numVertices(0), _numPatches(0) {

    // Sanity check: the mesh must be adaptively refined
    assert(not refiner.IsUniform());

    // Reserve the patch point stencils. Ideally topology refiner
    // would have an API to return how many endcap patches will be required.
    // Instead we conservatively estimate by the number of patches at the
    // finest level.
    int numMaxLevelFaces = refiner.GetLevel(refiner.GetMaxLevel()).GetNumFaces();

    _vertexStencils.reserve(numMaxLevelFaces*20);
    _varyingStencils.reserve(numMaxLevelFaces*20);
}

ConstIndexArray
EndCapBSplineBasisPatchFactory::GetPatchPoints(
    Vtr::internal::Level const * level, Index faceIndex,
    PatchTableFactory::PatchFaceTag const * /*levelPatchTags*/,
    int levelVertOffset) {

    // XXX: For now, always create new 16 indices for each patch.
    // we'll optimize later to share all regular control points with
    // other patches as well as to try to make extra ordinary verts watertight.

    int offset = _refiner->GetNumVerticesTotal();
    for (int i = 0; i < 16; ++i) {
        _patchPoints.push_back(_numVertices + offset);
        ++_numVertices;
    }

    // XXX: temporary hack. we should traverse topology and find existing
    //      vertices if available
    //
    // Reorder gregory basis stencils into regular bezier
    GregoryBasis::ProtoBasis basis(*level, faceIndex, levelVertOffset, -1);

    GregoryBasis::Point const *bezierCP[16];

    bezierCP[0] = &basis.P[0];
    bezierCP[1] = &basis.Ep[0];
    bezierCP[2] = &basis.Em[1];
    bezierCP[3] = &basis.P[1];

    bezierCP[4] = &basis.Em[0];
    bezierCP[5] = &basis.Fp[0]; // arbitrary
    bezierCP[6] = &basis.Fp[1]; // arbitrary
    bezierCP[7] = &basis.Ep[1];

    bezierCP[8]  = &basis.Ep[3];
    bezierCP[9]  = &basis.Fp[3]; // arbitrary
    bezierCP[10] = &basis.Fp[2]; // arbitrary
    bezierCP[11] = &basis.Em[2];

    bezierCP[12] = &basis.P[3];
    bezierCP[13] = &basis.Em[3];
    bezierCP[14] = &basis.Ep[2];
    bezierCP[15] = &basis.P[2];

    // all stencils should have the same capacity.
    int stencilCapacity = basis.P[0].GetCapacity();

    // Apply basis conversion from bezier to b-spline
    float Q[4][4] = {{ 6, -7,  2, 0},
                     { 0,  2, -1, 0},
                     { 0, -1,  2, 0},
                     { 0,  2, -7, 6} };
    Vtr::internal::StackBuffer<GregoryBasis::Point, 16> H(16);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            H[i*4+j].Clear(stencilCapacity);
            for (int k = 0; k < 4; ++k) {
                if (isWeightNonZero(Q[i][k])) {
                    H[i*4+j].AddWithWeight(*bezierCP[j+k*4], Q[i][k]);
                }
            }
        }
    }
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            GregoryBasis::Point p(stencilCapacity);
            for (int k = 0; k < 4; ++k) {
                if (isWeightNonZero(Q[j][k])) {
                    p.AddWithWeight(H[i*4+k], Q[j][k]);
                }
            }
            _vertexStencils.push_back(p);
        }
    }

    int varyingIndices[] = { 0, 0, 1, 1,
                             0, 0, 1, 1,
                             3, 3, 2, 2,
                             3, 3, 2, 2,};
    for (int i = 0; i < 16; ++i) {
        _varyingStencils.push_back(basis.V[varyingIndices[i]]);
    }

    ++_numPatches;
    return ConstIndexArray(&_patchPoints[(_numPatches-1)*16], 16);
}

} // end namespace Far

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv
