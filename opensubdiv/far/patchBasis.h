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

#ifndef OPENSUBDIV3_FAR_PATCH_BASIS_H
#define OPENSUBDIV3_FAR_PATCH_BASIS_H

#include "../version.h"

#include "../far/patchParam.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Far {
namespace internal {

//
// XXXX barfowl:  These functions are being kept in place while more complete
// underlying support for all patch types is being worked out.  That support
// will include a larger set of patch types (eventually including triangular
// patches for Loop) and arbitrary differentiation of all (to support second
// derivatives and other needs).
//
// So this interface will be changing in future.
//

void GetBilinearWeights(PatchParam const & patchParam,
    float s, float t, float wP[4], float wDs[4], float wDt[4], float wDss[4] = 0, float wDst[4] = 0, float wDtt[4] = 0);

void GetBezierWeights(PatchParam const & patchParam,
    float s, float t, float wP[16], float wDs[16], float wDt[16], float wDss[16] = 0, float wDst[16] = 0, float wDtt[16] = 0);

void GetBSplineWeights(PatchParam const & patchParam,
    float s, float t, float wP[16], float wDs[16], float wDt[16], float wDss[16] = 0, float wDst[16] = 0, float wDtt[16] = 0);

void GetGregoryWeights(PatchParam const & patchParam,
    float s, float t, float wP[20], float wDs[20], float wDt[20], float wDss[20] = 0, float wDst[20] = 0, float wDtt[20] = 0);


} // end namespace internal
} // end namespace Far

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* OPENSUBDIV3_FAR_PATCH_BASIS_H */
