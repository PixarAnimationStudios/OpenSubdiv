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

#ifndef FAR_INTERPOLATE_H
#define FAR_INTERPOLATE_H

#include "../version.h"

#include "../far/patchParam.h"
#include "../far/stencilTables.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Far {

// XXXX 
//
// Note 1 : interpolation functions will eventually need to be augmented to
//          handle second order derivatives
//
// Note 2 : evaluation of derivatives need to be ptional (passing NULL to 
//          dQ1 and dQ2 to the weights functions will do this)

void GetBilinearWeights(PatchParam::BitField bits,
    float s, float t, float point[4], float deriv1[4], float deriv2[4]);

void GetBezierWeights(PatchParam::BitField bits,
    float s, float t, float point[16], float deriv1[16], float deriv2[16]);

void GetBSplineWeights(PatchParam::BitField bits,
    float s, float t, float point[16], float deriv1[16], float deriv2[16]);


/// \brief Interpolate the (s,t) parametric location of a bilinear (quad)
/// patch
///
/// @param cvs     Array of 16 control vertex indices
///
/// @param Q       Array of 16 bicubic weights for the control vertices
///
/// @param Qd1     Array of 16 bicubic 's' tangent weights for the control
///                vertices
///
/// @param Qd2     Array of 16 bicubic 't' tangent weights for the control
///                vertices
///
/// @param src     Source primvar buffer (control vertices data)
///
/// @param dst     Destination primvar buffer (limit surface data)
///
template <class T, class U>
inline void
InterpolateBilinearPatch(Index const * cvs,
    float const * Q, float const *Qd1, float const *Qd2,
        T const & src, U & dst) {

    //
    //  v0 -- v1
    //   |.....|
    //   |.....|
    //  v3 -- v2
    //
    for (int k=0; k<4; ++k) {
        dst.AddWithWeight(src[cvs[k]], Q[k], Qd1[k], Qd2[k]);
    }
}

/// \brief Interpolate the (s,t) parametric location of a regular bicubic
///        patch
///
/// @param cvs     Array of 16 control vertex indices
///
/// @param Q       Array of 16 bicubic weights for the control vertices
///
/// @param Qd1     Array of 16 bicubic 's' tangent weights for the control
///                vertices
///
/// @param Qd2     Array of 16 bicubic 't' tangent weights for the control
///                vertices
///
/// @param src     Source primvar buffer (control vertices data)
///
/// @param dst     Destination primvar buffer (limit surface data)
///
template <class T, class U>
inline void
InterpolateRegularPatch(Index const * cvs,
    float const * Q, float const *Qd1, float const *Qd2,
        T const & src, U & dst) {

    //
    //  v0 -- v1 -- v2 -- v3
    //   |.....|.....|.....|
    //   |.....|.....|.....|
    //  v4 -- v5 -- v6 -- v7
    //   |.....|.....|.....|
    //   |.....|.....|.....|
    //  v8 -- v9 -- v10-- v11
    //   |.....|.....|.....|
    //   |.....|.....|.....|
    //  v12-- v13-- v14-- v15
    //
    for (int k=0; k<16; ++k) {
        dst.AddWithWeight(src[cvs[k]], Q[k], Qd1[k], Qd2[k]);
    }
}

/// \brief Interpolate the (s,t) parametric location of a boundary bicubic
///        patch
///
/// @param cvs     Array of 12 control vertex indices
///
/// @param Q       Array of 12 bicubic weights for the control vertices
///
/// @param Qd1     Array of 12 bicubic 's' tangent weights for the control
///                vertices
///
/// @param Qd2     Array of 12 bicubic 't' tangent weights for the control
///                vertices
///
/// @param src     Source primvar buffer (control vertices data)
///
/// @param dst     Destination primvar buffer (limit surface data)
///
template <class T, class U>
inline void
InterpolateBoundaryPatch(Index const * cvs,
    float const * Q, float const *Qd1, float const *Qd2,
        T const & src, U & dst) {

    // mirror the missing vertices (M)
    //
    //  M0 -- M1 -- M2 -- M3 (corner)
    //   |     |     |     |
    //   |     |     |     |
    //  v0 -- v1 -- v2 -- v3    M : mirrored
    //   |.....|.....|.....|
    //   |.....|.....|.....|
    //  v4 -- v5 -- v6 -- v7    v : original Cv
    //   |.....|.....|.....|
    //   |.....|.....|.....|
    //  v8 -- v9 -- v10-- v11
    //
    for (int k=0; k<4; ++k) { // M0 - M3
        dst.AddWithWeight(src[cvs[k]],    2.0f*Q[k],  2.0f*Qd1[k],  2.0f*Qd2[k]);
        dst.AddWithWeight(src[cvs[k+4]], -1.0f*Q[k], -1.0f*Qd1[k], -1.0f*Qd2[k]);
    }
    for (int k=0; k<12; ++k) {
        dst.AddWithWeight(src[cvs[k]], Q[k+4], Qd1[k+4], Qd2[k+4]);
    }
}

/// \brief Interpolate the (s,t) parametric location of a corner bicubic
///        patch
///
/// @param cvs     Array of 9 control vertex indices
///
/// @param Q       Array of 9 bicubic weights for the control vertices
///
/// @param Qd1     Array of 9 bicubic 's' tangent weights for the control
///                vertices
///
/// @param Qd2     Array of 9 bicubic 't' tangent weights for the control
///                vertices
///
/// @param src     Source primvar buffer (control vertices data)
///
/// @param dst     Destination primvar buffer (limit surface data)
///
template <class T, class U>
inline void
InterpolateCornerPatch(Index const * cvs,
    float const * Q, float const *Qd1, float const *Qd2,
        T const & src, U & dst) {

    // mirror the missing vertices (M)
    //
    //  M0 -- M1 -- M2 -- M3 (corner)
    //   |     |     |     |
    //   |     |     |     |
    //  v0 -- v1 -- v2 -- M4    M : mirrored
    //   |.....|.....|     |
    //   |.....|.....|     |
    //  v3.--.v4.--.v5 -- M5    v : original Cv
    //   |.....|.....|     |
    //   |.....|.....|     |
    //  v6 -- v7 -- v8 -- M6
    //
    for (int k=0; k<3; ++k) { // M0 - M2
        dst.AddWithWeight(src[cvs[k  ]],  2.0f*Q[k],  2.0f*Qd1[k],  2.0f*Qd2[k]);
        dst.AddWithWeight(src[cvs[k+3]], -1.0f*Q[k], -1.0f*Qd1[k], -1.0f*Qd2[k]);
    }
    for (int k=0; k<3; ++k) { // M4 - M6
        int idx = (k+1)*4 + 3;
        dst.AddWithWeight(src[cvs[k*3+2]],  2.0f*Q[idx],  2.0f*Qd1[idx],  2.0f*Qd2[idx]);
        dst.AddWithWeight(src[cvs[k*3+1]], -1.0f*Q[idx], -1.0f*Qd1[idx], -1.0f*Qd2[idx]);
    }
    // M3 = -2.v1 + 4.v2 + v4 - 2.v5
    dst.AddWithWeight(src[cvs[1]], -2.0f*Q[3], -2.0f*Qd1[3], -2.0f*Qd2[3]);
    dst.AddWithWeight(src[cvs[2]],  4.0f*Q[3],  4.0f*Qd1[3],  4.0f*Qd2[3]);
    dst.AddWithWeight(src[cvs[4]],  1.0f*Q[3],  1.0f*Qd1[3],  1.0f*Qd2[3]);
    dst.AddWithWeight(src[cvs[5]], -2.0f*Q[3], -2.0f*Qd1[3], -2.0f*Qd2[3]);
    for (int y=0; y<3; ++y) { // v0 - v8
        for (int x=0; x<3; ++x) {
            int idx = y*4+x+4;
            dst.AddWithWeight(src[cvs[y*3+x]], Q[idx], Qd1[idx], Qd2[idx]);
        }
    }
}

/// \brief Interpolate the (s,t) parametric location of a Gregory bicubic
///        patch
///
/// @param basisStencils  Stencil tables driving the 20 CV basis of the patches
///
/// @param stencilIndex   Index of the first CV stencil in the basis stencils tables
///
/// @param s              Patch coordinate (in coarse face normalized space)
///
/// @param t              Patch coordinate (in coarse face normalized space)
///
/// @param Q              Array of 9 bicubic weights for the control vertices
///
/// @param Qd1            Array of 9 bicubic 's' tangent weights for the control
///                       vertices
///
/// @param Qd2            Array of 9 bicubic 't' tangent weights for the control
///                       vertices
///
/// @param src            Source primvar buffer (control vertices data)
///
/// @param dst            Destination primvar buffer (limit surface data)
///
template <class T, class U>
inline void
InterpolateGregoryPatch(StencilTables const * basisStencils,
    int stencilIndex, float s, float t,
        float const * Q, float const *Qd1, float const *Qd2,
            T const & src, U & dst) {

    float ss = 1-s,
          tt = 1-t;
// remark #1572: floating-point equality and inequality comparisons are unreliable
#ifdef __INTEL_COMPILER
#pragma warning disable 1572
#endif
    float d11 = s+t;   if(s+t==0.0f)   d11 = 1.0f;
    float d12 = ss+t;  if(ss+t==0.0f)  d12 = 1.0f;
    float d21 = s+tt;  if(s+tt==0.0f)  d21 = 1.0f;
    float d22 = ss+tt; if(ss+tt==0.0f) d22 = 1.0f;
#ifdef __INTEL_COMPILER
#pragma warning enable 1572
#endif

    float weights[4][2] = { {  s/d11,  t/d11 },
                            { ss/d12,  t/d12 },
                            {  s/d21, tt/d21 },
                            { ss/d22, tt/d22 } };

    //
    //  P3         e3-      e2+         P2
    //     O--------O--------O--------O
    //     |        |        |        |
    //     |        |        |        |
    //     |        | f3-    | f2+    |
    //     |        O        O        |
    // e3+ O------O            O------O e2-
    //     |     f3+          f2-     |
    //     |                          |
    //     |                          |
    //     |      f0-         f1+     |
    // e0- O------O            O------O e1+
    //     |        O        O        |
    //     |        | f0+    | f1-    |
    //     |        |        |        |
    //     |        |        |        |
    //     O--------O--------O--------O
    //  P0         e0+      e1-         P1
    //
    // XXXX manuelk re-order stencils in factory and get rid of permutation ?
    int const permute[16] =
        { 0, 1, 7, 5, 2, -1, -1, 6, 16, -1, -1, 12, 15, 17, 11, 10 };

    for (int i=0, fcount=0; i<16; ++i) {

        int index = permute[i],
            offset = stencilIndex;

        if (index==-1) {

            // 0-ring vertex: blend 2 extra basis CVs
            int const fpermute[4][2] = { {3, 4}, {9, 8}, {19, 18}, {13, 14} };

            assert(fcount < 4);
            int v0 = fpermute[fcount][0],
                v1 = fpermute[fcount][1];

            Stencil s0 = basisStencils->GetStencil(offset + v0),
                    s1 = basisStencils->GetStencil(offset + v1);

            float w0=weights[fcount][0],
                  w1=weights[fcount][1];

            {
                Index const * srcIndices = s0.GetVertexIndices();
                float const * srcWeights = s0.GetWeights();
                for (int j=0; j<s0.GetSize(); ++j) {
                    dst.AddWithWeight(src[srcIndices[j]],
                        Q[i]*w0*srcWeights[j], Qd1[i]*w0*srcWeights[j],
                            Qd2[i]*w0*srcWeights[j]);
                }
            }
            {
                Index const * srcIndices = s1.GetVertexIndices();
                float const * srcWeights = s1.GetWeights();
                for (int j=0; j<s1.GetSize(); ++j) {
                    dst.AddWithWeight(src[srcIndices[j]],
                        Q[i]*w1*srcWeights[j], Qd1[i]*w1*srcWeights[j],
                            Qd2[i]*w1*srcWeights[j]);
                }
            }
            ++fcount;
        } else {
            Stencil stencil = basisStencils->GetStencil(offset + index);
            Index const * srcIndices = stencil.GetVertexIndices();
            float const * srcWeights = stencil.GetWeights();
            for (int j=0; j<stencil.GetSize(); ++j) {
                dst.AddWithWeight( src[srcIndices[j]],
                    Q[i]*srcWeights[j], Qd1[i]*srcWeights[j],
                         Qd2[i]*srcWeights[j]);
            }
        }
    }
}


} // end namespace Far

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* FAR_INTERPOLATE_H */
