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

//
// Quad patch types:
//
template <typename REAL>
int GetBilinearWeights(PatchParam const & patchParam, REAL s, REAL t,
    REAL wP[4], REAL wDs[4], REAL wDt[4], REAL wDss[4] = 0, REAL wDst[4] = 0, REAL wDtt[4] = 0);

template <typename REAL>
int GetBezierWeights(PatchParam const & patchParam, REAL s, REAL t,
    REAL wP[16], REAL wDs[16], REAL wDt[16], REAL wDss[16] = 0, REAL wDst[16] = 0, REAL wDtt[16] = 0);

template <typename REAL>
int GetBSplineWeights(PatchParam const & patchParam, REAL s, REAL t,
    REAL wP[16], REAL wDs[16], REAL wDt[16], REAL wDss[16] = 0, REAL wDst[16] = 0, REAL wDtt[16] = 0);

template <typename REAL>
int GetGregoryWeights(PatchParam const & patchParam, REAL s, REAL t,
    REAL wP[20], REAL wDs[20], REAL wDt[20], REAL wDss[20] = 0, REAL wDst[20] = 0, REAL wDtt[20] = 0);

//
// Triangle patch types:
//
template <typename REAL>
int GetLinearTriWeights(PatchParam const & patchParam, REAL s, REAL t,
    REAL wP[3], REAL wDs[3], REAL wDt[3], REAL wDss[3] = 0, REAL wDst[3] = 0, REAL wDtt[3] = 0);

template <typename REAL>
int GetBezierTriWeights(PatchParam const & patchParam, REAL s, REAL t,
    REAL wP[12], REAL wDs[12], REAL wDt[12], REAL wDss[12] = 0, REAL wDst[12] = 0, REAL wDtt[12] = 0);

template <typename REAL>
int GetBoxSplineTriWeights(PatchParam const & patchParam, REAL s, REAL t,
    REAL wP[12], REAL wDs[12], REAL wDt[12], REAL wDss[12] = 0, REAL wDst[12] = 0, REAL wDtt[12] = 0);

template <typename REAL>
int GetGregoryTriWeights(PatchParam const & patchParam, REAL s, REAL t,
    REAL wP[15], REAL wDs[15], REAL wDt[15], REAL wDss[15] = 0, REAL wDst[15] = 0, REAL wDtt[15] = 0);

} // end namespace internal
} // end namespace Far

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* OPENSUBDIV3_FAR_PATCH_BASIS_H */
