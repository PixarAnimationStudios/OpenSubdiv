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
    std::vector<GregoryBasis::Point> bezierCP;
    bezierCP.reserve(16);

    bezierCP.push_back(basis.P[0]);
    bezierCP.push_back(basis.Ep[0]);
    bezierCP.push_back(basis.Em[1]);
    bezierCP.push_back(basis.P[1]);

    bezierCP.push_back(basis.Em[0]);
    bezierCP.push_back(basis.Fp[0]); // arbitrary
    bezierCP.push_back(basis.Fp[1]); // arbitrary
    bezierCP.push_back(basis.Ep[1]);

    bezierCP.push_back(basis.Ep[3]);
    bezierCP.push_back(basis.Fp[3]); // arbitrary
    bezierCP.push_back(basis.Fp[2]); // arbitrary
    bezierCP.push_back(basis.Em[2]);

    bezierCP.push_back(basis.P[3]);
    bezierCP.push_back(basis.Em[3]);
    bezierCP.push_back(basis.Ep[2]);
    bezierCP.push_back(basis.P[2]);

    // Apply basis conversion from bezier to b-spline
    float Q[4][4] = {{ 6, -7,  2, 0},
                     { 0,  2, -1, 0},
                     { 0, -1,  2, 0},
                     { 0,  2, -7, 6} };
    std::vector<GregoryBasis::Point> H(16);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 4; ++k) {            
                if (isWeightNonZero(Q[i][k])) H[i*4+j] += bezierCP[j+k*4] * Q[i][k];
            }
        }
    }
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            GregoryBasis::Point p;
            for (int k = 0; k < 4; ++k) {
                if (isWeightNonZero(Q[j][k])) p += H[i*4+k] * Q[j][k];
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
