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

#include "../far/patchBasis.h"

#include <cassert>
#include <cstring>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Far {
namespace internal {

enum SplineBasis {
    BASIS_BILINEAR,
    BASIS_BEZIER,
    BASIS_BSPLINE,
    BASIS_BOX_SPLINE
};

template <SplineBasis BASIS>
class Spline {

public:

    // curve weights
    static void GetWeights(float t, float point[], float deriv[]);

    // box-spline weights
    static void GetWeights(float v, float w, float point[]);

    // patch weights
    static void GetPatchWeights(PatchParam const & param,
        float s, float t, float point[], float deriv1[], float deriv2[]);

    // adjust patch weights for boundary (and corner) edges
    static void AdjustBoundaryWeights(PatchParam const & param,
        float sWeights[4], float tWeights[4]);
};

template <>
inline void Spline<BASIS_BEZIER>::GetWeights(
    float t, float point[4], float deriv[4]) {

    // The four uniform cubic Bezier basis functions (in terms of t and its
    // complement tC) evaluated at t:
    float t2 = t*t;
    float tC = 1.0f - t;
    float tC2 = tC * tC;

    assert(point);
    point[0] = tC2 * tC;
    point[1] = tC2 * t * 3.0f;
    point[2] = t2 * tC * 3.0f;
    point[3] = t2 * t;

    // Derivatives of the above four basis functions at t:
    if (deriv) {
       deriv[0] = -3.0f * tC2;
       deriv[1] =  9.0f * t2 - 12.0f * t + 3.0f;
       deriv[2] = -9.0f * t2 +  6.0f * t;
       deriv[3] =  3.0f * t2;
    }
}

template <>
inline void Spline<BASIS_BSPLINE>::GetWeights(
    float t, float point[4], float deriv[4]) {

    // The four uniform cubic B-Spline basis functions evaluated at t:
    float const one6th = 1.0f / 6.0f;

    float t2 = t * t;
    float t3 = t * t2;

    assert(point);
    point[0] = one6th * (1.0f - 3.0f*(t -      t2) -      t3);
    point[1] = one6th * (4.0f           - 6.0f*t2  + 3.0f*t3);
    point[2] = one6th * (1.0f + 3.0f*(t +      t2  -      t3));
    point[3] = one6th * (                                 t3);

    // Derivatives of the above four basis functions at t:
    if (deriv) {
        deriv[0] = -0.5f*t2 +      t - 0.5f;
        deriv[1] =  1.5f*t2 - 2.0f*t;
        deriv[2] = -1.5f*t2 +      t + 0.5f;
        deriv[3] =  0.5f*t2;
    }
}

template <>
inline void Spline<BASIS_BOX_SPLINE>::GetWeights(
    float v, float w, float point[12]) {

    float u = 1.0f - v - w;

    //
    //  The 12 basis functions of the quartic box spline (unscaled by their common
    //  factor of 1/12 until later, and formatted to make it easy to spot any
    //  typing errors):
    //
    //      15 terms for the 3 points above the triangle corners
    //       9 terms for the 3 points on faces opposite the triangle edges
    //       2 terms for the 6 points on faces opposite the triangle corners
    //
    //  Powers of each variable for notational convenience:
    float u2 = u*u;
    float u3 = u*u2;
    float u4 = u*u3;
    float v2 = v*v;
    float v3 = v*v2;
    float v4 = v*v3;
    float w2 = w*w;
    float w3 = w*w2;
    float w4 = w*w3;

    //  And now the basis functions:
    point[ 0] = u4 + 2.0f*u3*v;
    point[ 1] = u4 + 2.0f*u3*w;
    point[ 8] = w4 + 2.0f*w3*u;
    point[11] = w4 + 2.0f*w3*v;
    point[ 9] = v4 + 2.0f*v3*w;
    point[ 5] = v4 + 2.0f*v3*u;

    point[ 2] = u4 + 2.0f*u3*w + 6.0f*u3*v + 6.0f*u2*v*w + 12.0f*u2*v2 +
                v4 + 2.0f*v3*w + 6.0f*v3*u + 6.0f*v2*u*w;
    point[ 4] = w4 + 2.0f*w3*v + 6.0f*w3*u + 6.0f*w2*u*v + 12.0f*w2*u2 +
                u4 + 2.0f*u3*v + 6.0f*u3*w + 6.0f*u2*v*w;
    point[10] = v4 + 2.0f*v3*u + 6.0f*v3*w + 6.0f*v2*w*u + 12.0f*v2*w2 +
                w4 + 2.0f*w3*u + 6.0f*w3*v + 6.0f*w3*u*v;

    point[ 3] = v4 + 6*v3*w + 8*v3*u + 36*v2*w*u + 24*v2*u2 + 24*v*u3 +
                w4 + 6*w3*v + 8*w3*u + 36*w2*v*u + 24*w2*u2 + 24*w*u3 + 6*u4 + 60*u2*v*w + 12*v2*w2;
    point[ 6] = w4 + 6*w3*u + 8*w3*v + 36*w2*u*v + 24*w2*v2 + 24*w*v3 +
                u4 + 6*u3*w + 8*u3*v + 36*u2*v*w + 24*u2*v2 + 24*u*v3 + 6*v4 + 60*v2*w*u + 12*w2*u2;
    point[ 7] = u4 + 6*u3*v + 8*u3*w + 36*u2*v*w + 24*u2*w2 + 24*u*w3 +
                v4 + 6*v3*u + 8*v3*w + 36*v2*u*w + 24*v2*w2 + 24*v*w3 + 6*w4 + 60*w2*u*v + 12*u2*v2;

    for (int i = 0; i < 12; ++i) {
        point[i] *= 1.0f / 12.0f;
    }
}

template <>
inline void Spline<BASIS_BILINEAR>::GetPatchWeights(PatchParam const & param,
    float s, float t, float point[4], float derivS[4], float derivT[4]) {

    param.Normalize(s,t);

    float sC = 1.0f - s,
          tC = 1.0f - t;

    if (point) {
        point[0] = sC * tC;
        point[1] =  s * tC;
        point[2] =  s * t;
        point[3] = sC * t;
    }
    
    if (derivS and derivT) {
        float dScale = (float)(1 << param.GetDepth());

        derivS[0] = -tC * dScale;
        derivS[1] =  tC * dScale;
        derivS[2] =   t * dScale;
        derivS[3] =  -t * dScale;

        derivT[0] = -sC * dScale;
        derivT[1] =  -s * dScale;
        derivT[2] =   s * dScale;
        derivT[3] =  sC * dScale;
    }
}

template <SplineBasis BASIS>
void Spline<BASIS>::AdjustBoundaryWeights(PatchParam const & param,
    float sWeights[4], float tWeights[4]) {

    int boundary = param.GetBoundary();

    if (boundary & 1) {
        tWeights[2] -= tWeights[0];
        tWeights[1] += 2*tWeights[0];
        tWeights[0] = 0;
    }
    if (boundary & 2) {
        sWeights[1] -= sWeights[3];
        sWeights[2] += 2*sWeights[3];
        sWeights[3] = 0;
    }
    if (boundary & 4) {
        tWeights[1] -= tWeights[3];
        tWeights[2] += 2*tWeights[3];
        tWeights[3] = 0;
    }
    if (boundary & 8) {
        sWeights[2] -= sWeights[0];
        sWeights[1] += 2*sWeights[0];
        sWeights[0] = 0;
    }
}

template <SplineBasis BASIS>
void Spline<BASIS>::GetPatchWeights(PatchParam const & param,
    float s, float t, float point[16], float derivS[16], float derivT[16]) {

    float sWeights[4], tWeights[4], dsWeights[4], dtWeights[4];

    param.Normalize(s,t);

    Spline<BASIS>::GetWeights(s, point ? sWeights : 0, derivS ? dsWeights : 0);
    Spline<BASIS>::GetWeights(t, point ? tWeights : 0, derivT ? dtWeights : 0);

    if (point) {
        // Compute the tensor product weight of the (s,t) basis function
        // corresponding to each control vertex:

        AdjustBoundaryWeights(param, sWeights, tWeights);

        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                point[4*i+j] = sWeights[j] * tWeights[i];
            }
        }
    }

    if (derivS and derivT) {
        // Compute the tensor product weight of the differentiated (s,t) basis
        // function corresponding to each control vertex (scaled accordingly):

        float dScale = (float)(1 << param.GetDepth());

        AdjustBoundaryWeights(param, dsWeights, dtWeights);

        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                derivS[4*i+j] = dsWeights[j] * tWeights[i] * dScale;
                derivT[4*i+j] = sWeights[j] * dtWeights[i] * dScale;
            }
        }
    }
}

void GetBilinearWeights(PatchParam const & param,
    float s, float t, float point[4], float deriv1[4], float deriv2[4]) {

    Spline<BASIS_BILINEAR>::GetPatchWeights(param, s, t, point, deriv1, deriv2);
}

void GetBezierWeights(PatchParam const param,
    float s, float t, float point[16], float deriv1[16], float deriv2[16]) {

    Spline<BASIS_BEZIER>::GetPatchWeights(param, s, t, point, deriv1, deriv2);
}

void GetBSplineWeights(PatchParam const & param,
    float s, float t, float point[16], float deriv1[16], float deriv2[16]) {

    Spline<BASIS_BSPLINE>::GetPatchWeights(param, s, t, point, deriv1, deriv2);
}

void GetGregoryWeights(PatchParam const & param,
    float s, float t, float point[20], float deriv1[20], float deriv2[20]) {

    //
    //  P3         e3-      e2+         P2
    //     15------17-------11--------10
    //     |        |        |        |
    //     |        |        |        |
    //     |        | f3-    | f2+    |
    //     |       19       13        |
    // e3+ 16-----18           14-----12 e2-
    //     |     f3+          f2-     |
    //     |                          |
    //     |                          |
    //     |      f0-         f1+     |
    // e0- 2------4            8------6 e1+
    //     |        3        9        |
    //     |        | f0+    | f1-    |
    //     |        |        |        |
    //     |        |        |        |
    //     O--------1--------7--------5
    //  P0         e0+      e1-         P1
    //

    //  Indices of boundary and interior points and their corresponding Bezier points
    //  (this can be reduced with more direct indexing and unrolling of loops):
    //
    static int const boundaryGregory[12] = { 0, 1, 7, 5, 2, 6, 16, 12, 15, 17, 11, 10 };
    static int const boundaryBezSCol[12] = { 0, 1, 2, 3, 0, 3,  0,  3,  0,  1,  2,  3 };
    static int const boundaryBezTRow[12] = { 0, 0, 0, 0, 1, 1,  2,  2,  3,  3,  3,  3 };

    static int const interiorGregory[8] = { 3, 4,  8, 9,  13, 14,  18, 19 };
    static int const interiorBezSCol[8] = { 1, 1,  2, 2,   2,  2,   1,  1 };
    static int const interiorBezTRow[8] = { 1, 1,  1, 1,   2,  2,   2,  2 };

    //
    //  Bezier basis functions are denoted with B while the rational multipliers for the
    //  interior points will be denoted G -- so we have B(s), B(t) and G(s,t):
    //
    //  Directional Bezier basis functions B at s and t:
    float Bs[4], Bds[4];
    float Bt[4], Bdt[4];

    param.Normalize(s,t);

    Spline<BASIS_BEZIER>::GetWeights(s, Bs, deriv1 ? Bds : 0);
    Spline<BASIS_BEZIER>::GetWeights(t, Bt, deriv2 ? Bdt : 0);

    //  Rational multipliers G at s and t:
    float sC = 1.0f - s;
    float tC = 1.0f - t;

    //  Use <= here to avoid compiler warnings -- the sums should always be non-negative:
    float df0 = s  + t;   df0 = (df0 <= 0.0f) ? 1.0f : (1.0f / df0);
    float df1 = sC + t;   df1 = (df1 <= 0.0f) ? 1.0f : (1.0f / df1);
    float df2 = sC + tC;  df2 = (df2 <= 0.0f) ? 1.0f : (1.0f / df2);
    float df3 = s  + tC;  df3 = (df3 <= 0.0f) ? 1.0f : (1.0f / df3);

    float G[8] = { s*df0, t*df0,  t*df1, sC*df1,  sC*df2, tC*df2,  tC*df3, s*df3 };

    //  Combined weights for boundary and interior points:
    for (int i = 0; i < 12; ++i) {
        point[boundaryGregory[i]] = Bs[boundaryBezSCol[i]] * Bt[boundaryBezTRow[i]];
    }
    for (int i = 0; i < 8; ++i) {
        point[interiorGregory[i]] = Bs[interiorBezSCol[i]] * Bt[interiorBezTRow[i]] * G[i];
    }

    //
    //  For derivatives, the basis functions for the interior points are rational and ideally
    //  require appropriate differentiation, i.e. product rule for the combination of B and G
    //  and the quotient rule for the rational G itself.  As initially proposed by Loop et al
    //  though, the approximation using the 16 Bezier points arising from the G(s,t) has
    //  proved adequate (and is what the GPU shaders use) so we continue to use that here.
    //
    //  An implementation of the true derivatives is provided for future reference -- it is
    //  unclear if the approximations will hold up under surface analysis involving higher
    //  order differentiation.
    //
    if (deriv1 and deriv2) {
        //  Remember to include derivative scaling in all assignments below:
        float dScale = (float)(1 << param.GetDepth());

        //  Combined weights for boundary points -- simple (scaled) tensor products:
        for (int i = 0; i < 12; ++i) {
            int iDst = boundaryGregory[i];
            int tRow = boundaryBezTRow[i];
            int sCol = boundaryBezSCol[i];

            deriv1[iDst] = Bds[sCol] * Bt[tRow] * dScale;
            deriv2[iDst] = Bdt[tRow] * Bs[sCol] * dScale;
        }

#define _USE_BEZIER_PSEUDO_DERIVATIVES
#ifdef _USE_BEZIER_PSEUDO_DERIVATIVES
        //  Approximation to the true Gregory derivatives by differentiating the Bezier patch
        //  unique to the given (s,t), i.e. having F = (g^+ * f^+) + (g^- * f^-) as its four
        //  interior points:
        //
        //  Combined weights for interior points -- (scaled) tensor products with G+ or G-:
        for (int i = 0; i < 8; ++i) {
            int iDst = interiorGregory[i];
            int tRow = interiorBezTRow[i];
            int sCol = interiorBezSCol[i];

            deriv1[iDst] = Bds[sCol] * Bt[tRow] * G[i] * dScale;
            deriv2[iDst] = Bdt[tRow] * Bs[sCol] * G[i] * dScale;
        }
#else
        //  True Gregory derivatives using appropriate differentiation of composite functions:
        //
        //  Note that for G(s,t) = N(s,t) / D(s,t), all N' and D' are trivial constants (which
        //  simplifies things for higher order derivatives).  And while each pair of functions
        //  G (i.e. the G+ and G- corresponding to points f+ and f-) must sum to 1 to ensure
        //  Bezier equivalence (when f+ = f-), the pairs of G' must similarly sum to 0.  So we
        //  can potentially compute only one of the pair and negate the result for the other
        //  (and with 4 or 8 computations involving these constants, this is all very SIMD
        //  friendly...) but for now we treat all 8 independently for simplicity.
        //
        //float N[8] = {   s,     t,      t,     sC,      sC,     tC,      tC,     s };
        float D[8] = {   df0,   df0,    df1,    df1,     df2,    df2,     df3,   df3 };

        static float const Nds[8] = { 1.0f, 0.0f,  0.0f, -1.0f, -1.0f,  0.0f,  0.0f,  1.0f };
        static float const Ndt[8] = { 0.0f, 1.0f,  1.0f,  0.0f,  0.0f, -1.0f, -1.0f,  0.0f };

        static float const Dds[8] = { 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f,  1.0f,  1.0f };
        static float const Ddt[8] = { 1.0f, 1.0f,  1.0f,  1.0f, -1.0f, -1.0f, -1.0f, -1.0f };

        //  Combined weights for interior points -- (scaled) combinations of B, B', G and G':
        for (int i = 0; i < 8; ++i) {
            int iDst = interiorGregory[i];
            int tRow = interiorBezTRow[i];
            int sCol = interiorBezSCol[i];

            //  Quotient rule for G' (re-expressed in terms of G to simplify (and D = 1/D)):
            float Gds = (Nds[i] - Dds[i] * G[i]) * D[i];
            float Gdt = (Ndt[i] - Ddt[i] * G[i]) * D[i];

            //  Product rule combining B and B' with G and G' (and scaled):
            deriv1[iDst] = (Bds[sCol] * G[i] + Bs[sCol] * Gds) * Bt[tRow] * dScale;
            deriv2[iDst] = (Bdt[tRow] * G[i] + Bt[tRow] * Gdt) * Bs[sCol] * dScale;
        }
#endif
    }
}

} // end namespace internal
} // end namespace Far

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv
