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

namespace {
//
//  Evaluation functions for curves used to assemble tensor product bases:
//
template <typename REAL>
void evalBezierCurve(REAL t, REAL wP[4], REAL wDP[4], REAL wDP2[4]) {

    // The four uniform cubic Bezier basis functions (in terms of t and its
    // complement tC) evaluated at t:
    REAL t2 = t*t;
    REAL tC = 1.0f - t;
    REAL tC2 = tC * tC;

    wP[0] = tC2 * tC;
    wP[1] = tC2 * t * 3.0f;
    wP[2] = t2 * tC * 3.0f;
    wP[3] = t2 * t;

    // Derivatives of the above four basis functions at t:
    if (wDP) {
       wDP[0] = -3.0f * tC2;
       wDP[1] =  9.0f * t2 - 12.0f * t + 3.0f;
       wDP[2] = -9.0f * t2 +  6.0f * t;
       wDP[3] =  3.0f * t2;
    }

    // Second derivatives of the basis functions at t:
    if (wDP2) {
        wDP2[0] =   6.0f * tC;
        wDP2[1] =  18.0f * t - 12.0f;
        wDP2[2] = -18.0f * t +  6.0f;
        wDP2[3] =   6.0f * t;
    }
}

template <typename REAL>
void evalBSplineCurve(REAL t, REAL wP[4], REAL wDP[4], REAL wDP2[4]) {

    // The four uniform cubic B-Spline basis functions evaluated at t:
    REAL const one6th = (REAL)(1.0 / 6.0);

    REAL t2 = t * t;
    REAL t3 = t * t2;

    wP[0] = one6th * (1.0f - 3.0f*(t -      t2) -      t3);
    wP[1] = one6th * (4.0f           - 6.0f*t2  + 3.0f*t3);
    wP[2] = one6th * (1.0f + 3.0f*(t +      t2  -      t3));
    wP[3] = one6th * (                                 t3);

    // Derivatives of the above four basis functions at t:
    if (wDP) {
        wDP[0] = -0.5f*t2 +      t - 0.5f;
        wDP[1] =  1.5f*t2 - 2.0f*t;
        wDP[2] = -1.5f*t2 +      t + 0.5f;
        wDP[3] =  0.5f*t2;
    }

    // Second derivatives of the basis functions at t:
    if (wDP2) {
        wDP2[0] = -       t + 1.0f;
        wDP2[1] =  3.0f * t - 2.0f;
        wDP2[2] = -3.0f * t + 1.0f;
        wDP2[3] =         t;
    }
}
}

template <typename REAL>
void GetBoxSplineWeights(PatchParam const & param, REAL s, REAL t, REAL wP[12]) {

    float u = s;
    float v = t;
    float w = 1.0f - u - v;

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
    wP[ 0] = u4 + 2.0f*u3*v;
    wP[ 1] = u4 + 2.0f*u3*w;
    wP[ 8] = w4 + 2.0f*w3*u;
    wP[11] = w4 + 2.0f*w3*v;
    wP[ 9] = v4 + 2.0f*v3*w;
    wP[ 5] = v4 + 2.0f*v3*u;

    wP[ 2] = u4 + 2.0f*u3*w + 6.0f*u3*v + 6.0f*u2*v*w + 12.0f*u2*v2 +
                v4 + 2.0f*v3*w + 6.0f*v3*u + 6.0f*v2*u*w;
    wP[ 4] = w4 + 2.0f*w3*v + 6.0f*w3*u + 6.0f*w2*u*v + 12.0f*w2*u2 +
                u4 + 2.0f*u3*v + 6.0f*u3*w + 6.0f*u2*v*w;
    wP[10] = v4 + 2.0f*v3*u + 6.0f*v3*w + 6.0f*v2*w*u + 12.0f*v2*w2 +
                w4 + 2.0f*w3*u + 6.0f*w3*v + 6.0f*w3*u*v;

    wP[ 3] = v4 + 6*v3*w + 8*v3*u + 36*v2*w*u + 24*v2*u2 + 24*v*u3 +
                w4 + 6*w3*v + 8*w3*u + 36*w2*v*u + 24*w2*u2 + 24*w*u3 + 6*u4 + 60*u2*v*w + 12*v2*w2;
    wP[ 6] = w4 + 6*w3*u + 8*w3*v + 36*w2*u*v + 24*w2*v2 + 24*w*v3 +
                u4 + 6*u3*w + 8*u3*v + 36*u2*v*w + 24*u2*v2 + 24*u*v3 + 6*v4 + 60*v2*w*u + 12*w2*u2;
    wP[ 7] = u4 + 6*u3*v + 8*u3*w + 36*u2*v*w + 24*u2*w2 + 24*u*w3 +
                v4 + 6*v3*u + 8*v3*w + 36*v2*u*w + 24*v2*w2 + 24*v*w3 + 6*w4 + 60*w2*u*v + 12*u2*v2;

    for (int i = 0; i < 12; ++i) {
        wP[i] *= 1.0f / 12.0f;
    }
}

template <typename REAL>
void GetBilinearWeights(PatchParam const & param, REAL s, REAL t,
    REAL wP[4], REAL wDs[4], REAL wDt[4],
    REAL wDss[4], REAL wDst[4], REAL wDtt[4]) {

    param.Normalize(s,t);

    REAL sC = 1.0f - s;
    REAL tC = 1.0f - t;

    if (wP) {
        wP[0] = sC * tC;
        wP[1] =  s * tC;
        wP[2] =  s * t;
        wP[3] = sC * t;
    }

    if (wDs && wDt) {
        REAL dScale = (REAL)(1 << param.GetDepth());

        wDs[0] = -tC * dScale;
        wDs[1] =  tC * dScale;
        wDs[2] =   t * dScale;
        wDs[3] =  -t * dScale;

        wDt[0] = -sC * dScale;
        wDt[1] =  -s * dScale;
        wDt[2] =   s * dScale;
        wDt[3] =  sC * dScale;

        if (wDss && wDst && wDtt) {
            REAL d2Scale = dScale * dScale;

            for(int i=0;i<4;i++) {
                wDss[i] = 0.0f;
                wDtt[i] = 0.0f;
            }

            wDst[0] =  d2Scale;
            wDst[1] = -d2Scale;
            wDst[2] = -d2Scale;
            wDst[3] =  d2Scale;
        }
    }
}

//
//  BSpline patch evaluation -- involves adjustments to weights when boundary
//  points are missing and implicitly extrapolated.
//
namespace {
template <typename REAL>
void adjustBSplineBoundaryWeights(PatchParam const & param, REAL sWeights[4], REAL tWeights[4]) {

    int boundary = param.GetBoundary();

    if ((boundary & 1) != 0) {
        tWeights[2] -= tWeights[0];
        tWeights[1] += tWeights[0] * 2.0f;
        tWeights[0]  = 0.0f;
    }
    if ((boundary & 2) != 0) {
        sWeights[1] -= sWeights[3];
        sWeights[2] += sWeights[3] * 2.0f;
        sWeights[3]  = 0.0f;
    }
    if ((boundary & 4) != 0) {
        tWeights[1] -= tWeights[3];
        tWeights[2] += tWeights[3] * 2.0f;
        tWeights[3]  = 0.0f;
    }
    if ((boundary & 8) != 0) {
        sWeights[2] -= sWeights[0];
        sWeights[1] += sWeights[0] * 2.0f;
        sWeights[0]  = 0.0f;
    }
}
}

template <typename REAL>
void GetBSplineWeights(PatchParam const & param, REAL s, REAL t,
    REAL wP[16], REAL wDs[16], REAL wDt[16],
    REAL wDss[16], REAL wDst[16], REAL wDtt[16]) {

    REAL sWeights[4], tWeights[4], dsWeights[4], dtWeights[4], dssWeights[4], dttWeights[4];

    param.Normalize(s,t);

    evalBSplineCurve(s, wP ? sWeights : 0, wDs ? dsWeights : 0, wDss ? dssWeights : 0);
    evalBSplineCurve(t, wP ? tWeights : 0, wDt ? dtWeights : 0, wDtt ? dttWeights : 0);

    if (wP) {
        adjustBSplineBoundaryWeights(param, sWeights, tWeights);

        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                wP[4*i+j] = sWeights[j] * tWeights[i];
            }
        }
    }

    if (wDs && wDt) {
        REAL dScale = (REAL)(1 << param.GetDepth());

        adjustBSplineBoundaryWeights(param, dsWeights, dtWeights);

        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                wDs[4*i+j] = dsWeights[j] * tWeights[i] * dScale;
                wDt[4*i+j] = sWeights[j] * dtWeights[i] * dScale;
            }
        }

        if (wDss && wDst && wDtt) {
            REAL d2Scale = dScale * dScale;

            adjustBSplineBoundaryWeights(param, dssWeights, dttWeights);

            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    wDss[4*i+j] = dssWeights[j] * tWeights[i] * d2Scale;
                    wDst[4*i+j] = dsWeights[j] * dtWeights[i] * d2Scale;
                    wDtt[4*i+j] = sWeights[j] * dttWeights[i] * d2Scale;
                }
            }
        }
    }
}

template <typename REAL>
void GetBezierWeights(PatchParam const & param, REAL s, REAL t,
    REAL wP[16], REAL wDs[16], REAL wDt[16],
    REAL wDss[16], REAL wDst[16], REAL wDtt[16]) {

    REAL sWeights[4], tWeights[4], dsWeights[4], dtWeights[4], dssWeights[4], dttWeights[4];

    param.Normalize(s,t);

    evalBezierCurve(s, wP ? sWeights : 0, wDs ? dsWeights : 0, wDss ? dssWeights : 0);
    evalBezierCurve(t, wP ? tWeights : 0, wDt ? dtWeights : 0, wDtt ? dttWeights : 0);

    if (wP) {
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                wP[4*i+j] = sWeights[j] * tWeights[i];
            }
        }
    }
    if (wDs && wDt) {
        REAL dScale = (REAL)(1 << param.GetDepth());

        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                wDs[4*i+j] = dsWeights[j] * tWeights[i] * dScale;
                wDt[4*i+j] = sWeights[j] * dtWeights[i] * dScale;
            }
        }
        if (wDss && wDst && wDtt) {
            REAL d2Scale = dScale * dScale;

            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    wDss[4*i+j] = dssWeights[j] * tWeights[i] * d2Scale;
                    wDst[4*i+j] = dsWeights[j] * dtWeights[i] * d2Scale;
                    wDtt[4*i+j] = sWeights[j] * dttWeights[i] * d2Scale;
                }
            }
        }
    }
}

template <typename REAL>
void GetGregoryWeights(PatchParam const & param, REAL s, REAL t,
    REAL point[20], REAL wDs[20], REAL wDt[20],
    REAL wDss[20], REAL wDst[20], REAL wDtt[20]) {

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
    REAL Bs[4], Bds[4], Bdss[4];
    REAL Bt[4], Bdt[4], Bdtt[4];

    param.Normalize(s,t);

    evalBezierCurve(s, Bs, wDs ? Bds : 0, wDss ? Bdss : 0);
    evalBezierCurve(t, Bt, wDt ? Bdt : 0, wDtt ? Bdtt : 0);

    //  Rational multipliers G at s and t:
    REAL sC = 1.0f - s;
    REAL tC = 1.0f - t;

    //  Use <= here to avoid compiler warnings -- the sums should always be non-negative:
    REAL df0 = s  + t;   df0 = (df0 <= 0.0f) ? (REAL)1.0f : (1.0f / df0);
    REAL df1 = sC + t;   df1 = (df1 <= 0.0f) ? (REAL)1.0f : (1.0f / df1);
    REAL df2 = sC + tC;  df2 = (df2 <= 0.0f) ? (REAL)1.0f : (1.0f / df2);
    REAL df3 = s  + tC;  df3 = (df3 <= 0.0f) ? (REAL)1.0f : (1.0f / df3);

    REAL G[8] = { s*df0, t*df0,  t*df1, sC*df1,  sC*df2, tC*df2,  tC*df3, s*df3 };

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
    if (wDs && wDt) {
        bool find_second_partials = wDs && wDst && wDtt;
        //  Remember to include derivative scaling in all assignments below:
        REAL dScale = (REAL)(1 << param.GetDepth());
        REAL d2Scale = dScale * dScale;

        //  Combined weights for boundary points -- simple (scaled) tensor products:
        for (int i = 0; i < 12; ++i) {
            int iDst = boundaryGregory[i];
            int tRow = boundaryBezTRow[i];
            int sCol = boundaryBezSCol[i];

            wDs[iDst] = Bds[sCol] * Bt[tRow] * dScale;
            wDt[iDst] = Bdt[tRow] * Bs[sCol] * dScale;

            if (find_second_partials) {
                wDss[iDst] = Bdss[sCol] * Bt[tRow] * d2Scale;
                wDst[iDst] = Bds[sCol] * Bdt[tRow] * d2Scale;
                wDtt[iDst] = Bs[sCol] * Bdtt[tRow] * d2Scale;
            }
        }

        // dclyde's note: skipping half of the product rule like this does seem to change the result a lot in my tests.
        // This is not a runtime bottleneck for cloth sims anyway so I'm just using the accurate version.
#ifndef OPENSUBDIV_GREGORY_EVAL_TRUE_DERIVATIVES
        //  Approximation to the true Gregory derivatives by differentiating the Bezier patch
        //  unique to the given (s,t), i.e. having F = (g^+ * f^+) + (g^- * f^-) as its four
        //  interior points:
        //
        //  Combined weights for interior points -- (scaled) tensor products with G+ or G-:
        for (int i = 0; i < 8; ++i) {
            int iDst = interiorGregory[i];
            int tRow = interiorBezTRow[i];
            int sCol = interiorBezSCol[i];

            wDs[iDst] = Bds[sCol] * Bt[tRow] * G[i] * dScale;
            wDt[iDst] = Bdt[tRow] * Bs[sCol] * G[i] * dScale;

            if (find_second_partials) {
                wDss[iDst] = Bdss[sCol] * Bt[tRow] * G[i] * d2Scale;
                wDst[iDst] = Bds[sCol] * Bdt[tRow] * G[i] * d2Scale;
                wDtt[iDst] = Bs[sCol] * Bdtt[tRow] * G[i] * d2Scale;
            }
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
        //REAL N[8] = {   s,     t,      t,     sC,      sC,     tC,      tC,     s };
        REAL D[8] = {   df0,   df0,    df1,    df1,     df2,    df2,     df3,   df3 };

        static REAL const Nds[8] = {  1.0f,  0.0f,  0.0f, -1.0f, -1.0f,  0.0f,  0.0f,  1.0f };
        static REAL const Ndt[8] = {  0.0f,  1.0f,  1.0f,  0.0f,  0.0f, -1.0f, -1.0f,  0.0f };

        static REAL const Dds[8] = {  1.0f,  1.0f, -1.0f, -1.0f, -1.0f, -1.0f,  1.0f,  1.0f };
        static REAL const Ddt[8] = {  1.0f,  1.0f,  1.0f,  1.0f, -1.0f, -1.0f, -1.0f, -1.0f };

        //  Combined weights for interior points -- (scaled) combinations of B, B', G and G':
        for (int i = 0; i < 8; ++i) {
            int iDst = interiorGregory[i];
            int tRow = interiorBezTRow[i];
            int sCol = interiorBezSCol[i];

            //  Quotient rule for G' (re-expressed in terms of G to simplify (and D = 1/D)):
            REAL Gds = (Nds[i] - Dds[i] * G[i]) * D[i];
            REAL Gdt = (Ndt[i] - Ddt[i] * G[i]) * D[i];

            //  Product rule combining B and B' with G and G' (and scaled):
            wDs[iDst] = (Bds[sCol] * G[i] + Bs[sCol] * Gds) * Bt[tRow] * dScale;
            wDt[iDst] = (Bdt[tRow] * G[i] + Bt[tRow] * Gdt) * Bs[sCol] * dScale;

            if (find_second_partials) {
                REAL Dsqr_inv = D[i]*D[i];

                REAL Gdss = 2.0f * Dds[i] * Dsqr_inv * (G[i] * Dds[i] - Nds[i]);
                REAL Gdst = Dsqr_inv * (2.0f * G[i] * Dds[i] * Ddt[i] - Nds[i] * Ddt[i] - Ndt[i] * Dds[i]);
                REAL Gdtt = 2.0f * Ddt[i] * Dsqr_inv * (G[i] * Ddt[i] - Ndt[i]);

                wDss[iDst] = (Bdss[sCol] * G[i] + 2.0f * Bds[sCol] * Gds + Bs[sCol] * Gdss) * Bt[tRow] * d2Scale;
                wDst[iDst] = (Bt[tRow] * (Bs[sCol] * Gdst + Bds[sCol] * Gdt) + Bdt[tRow] * (Bds[sCol] * G[i] + Bs[sCol] * Gds)) * d2Scale;
                wDtt[iDst] = (Bdtt[tRow] * G[i] + 2.0f * Bdt[tRow] * Gdt + Bt[tRow] * Gdtt) * Bs[sCol] * d2Scale;
            }
        }
#endif
    }
}


//
//  Explicit float and double instantiations:
//
template void GetBilinearWeights<float>(PatchParam const & patchParam, float s, float t,
        float wP[4], float wDs[4], float wDt[4], float wDss[4], float wDst[4], float wDtt[4]);
template void GetBezierWeights<float>(PatchParam const & patchParam, float s, float t,
        float wP[16], float wDs[16], float wDt[16], float wDss[16], float wDst[16], float wDtt[16]);
template void GetBSplineWeights<float>(PatchParam const & patchParam, float s, float t,
        float wP[16], float wDs[16], float wDt[16], float wDss[16], float wDst[16], float wDtt[16]);
template void GetGregoryWeights<float>(PatchParam const & patchParam, float s, float t,
        float wP[20], float wDs[20], float wDt[20], float wDss[20], float wDst[20], float wDtt[20]);

template void GetBilinearWeights<double>(PatchParam const & patchParam, double s, double t,
        double wP[4], double wDs[4], double wDt[4], double wDss[4], double wDst[4], double wDtt[4]);
template void GetBezierWeights<double>(PatchParam const & patchParam, double s, double t,
        double wP[16], double wDs[16], double wDt[16], double wDss[16], double wDst[16], double wDtt[16]);
template void GetBSplineWeights<double>(PatchParam const & patchParam, double s, double t,
        double wP[16], double wDs[16], double wDt[16], double wDss[16], double wDst[16], double wDtt[16]);
template void GetGregoryWeights<double>(PatchParam const & patchParam, double s, double t,
        double wP[20], double wDs[20], double wDt[20], double wDss[20], double wDst[20], double wDtt[20]);

} // end namespace internal
} // end namespace Far

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv
