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

//----------------------------------------------------------
// Patches.TessControlTransition
//----------------------------------------------------------
#ifdef OSD_PATCH_TESS_CONTROL_BSPLINE_SHADER

patch out vec4 tessOuterLo, tessOuterHi;

void
GetTessLevelsUniform(vec3 cp[16], int patchParam)
{
    float tessAmount = GetTessLevel(GetPatchLevel());

    tessOuterLo = vec4(tessAmount);
    tessOuterHi = vec4(0);
}

//
// Organization of B-spline and Bezier control points.
//
// Each patch is defined by 16 control points (labeled 0-15).
//
// The patch will be evaluated across the domain from (0,0) at
// the lower-left to (1,1) at the upper-right. When computing
// adaptive tessellation metrics, we consider refined vertex-vertex
// and edge-vertex points along the transition edges of the patch
// (labeled vv* and ev* respectively).
//
// The two segments of each transition edge are labeled Lo and Hi,
// with the Lo segment occuring before the Hi segment along the
// transition edge's domain parameterization. These Lo and Hi segment
// tessellation levels determine how domain evaluation coordinates
// are remapped along transition edges. The Hi segment value will
// be zero for a non-transition edge.
//
// (0,1)                                         (1,1)
//
//   vv3                  ev23                   vv2
//        |       Lo3       |       Hi3       |
//      --O-----------O-----+-----O-----------O--
//        | 12        | 13     14 |        15 |
//        |           |           |           |
//        |           |           |           |
//    Hi0 |           |           |           | Hi2
//        |           |           |           |
//        O-----------O-----------O-----------O
//        | 8         | 9      10 |        11 |
//        |           |           |           |
// ev03 --+           |           |           +-- ev12
//        |           |           |           |
//        | 4         | 5       6 |         7 |
//        O-----------O-----------O-----------O
//        |           |           |           |
//    Lo0 |           |           |           | Lo2
//        |           |           |           |
//        |           |           |           |
//        | 0         | 1       2 |         3 |
//      --O-----------O-----+-----O-----------O--
//        |       Lo1       |       Hi1       |
//   vv0                  ev01                   vv1
//
// (0,0)                                         (1,0)
//

void
GetTessLevelsRefinedPoints(vec3 cp[16], int patchParam)
{
    // Each edge of a transition patch is adjacent to one or two patches
    // at the next refined level of subdivision. We compute the corresponding
    // vertex-vertex and edge-vertex refined points along the edges of the
    // patch using Catmull-Clark subdivision stencil weights.
    // For simplicity, we let the optimizer discard unused computation.

    vec3 vv0 = (cp[0] + cp[2] + cp[8] + cp[10]) * 0.015625 +
               (cp[1] + cp[4] + cp[6] + cp[9]) * 0.09375 + cp[5] * 0.5625;
    vec3 ev01 = (cp[1] + cp[2] + cp[9] + cp[10]) * 0.0625 +
                (cp[5] + cp[6]) * 0.375;

    vec3 vv1 = (cp[1] + cp[3] + cp[9] + cp[11]) * 0.015625 +
               (cp[2] + cp[5] + cp[7] + cp[10]) * 0.09375 + cp[6] * 0.5625;
    vec3 ev12 = (cp[5] + cp[7] + cp[9] + cp[11]) * 0.0625 +
                (cp[6] + cp[10]) * 0.375;

    vec3 vv2 = (cp[5] + cp[7] + cp[13] + cp[15]) * 0.015625 +
               (cp[6] + cp[9] + cp[11] + cp[14]) * 0.09375 + cp[10] * 0.5625;
    vec3 ev23 = (cp[5] + cp[6] + cp[13] + cp[14]) * 0.0625 +
                (cp[9] + cp[10]) * 0.375;

    vec3 vv3 = (cp[4] + cp[6] + cp[12] + cp[14]) * 0.015625 +
               (cp[5] + cp[8] + cp[10] + cp[13]) * 0.09375 + cp[9] * 0.5625;
    vec3 ev03 = (cp[4] + cp[6] + cp[8] + cp[10]) * 0.0625 +
                (cp[5] + cp[9]) * 0.375;

    tessOuterLo = vec4(1);
    tessOuterHi = vec4(0);

    if (((patchParam >> 11) & 1) != 0) {
        tessOuterLo[0] = TessAdaptive(vv0, ev03);
        tessOuterHi[0] = TessAdaptive(vv3, ev03);
    } else {
        tessOuterLo[0] = TessAdaptive(cp[5], cp[9]);
    }
    if (((patchParam >> 8) & 1) != 0) {
        tessOuterLo[1] = TessAdaptive(vv0, ev01);
        tessOuterHi[1] = TessAdaptive(vv1, ev01);
    } else {
        tessOuterLo[1] = TessAdaptive(cp[5], cp[6]);
    }
    if (((patchParam >> 9) & 1) != 0) {
        tessOuterLo[2] = TessAdaptive(vv1, ev12);
        tessOuterHi[2] = TessAdaptive(vv2, ev12);
    } else {
        tessOuterLo[2] = TessAdaptive(cp[6], cp[10]);
    }
    if (((patchParam >> 10) & 1) != 0) {
        tessOuterLo[3] = TessAdaptive(vv3, ev23);
        tessOuterHi[3] = TessAdaptive(vv2, ev23);
    } else {
        tessOuterLo[3] = TessAdaptive(cp[9], cp[10]);
    }
}

void
GetTessLevelsLimitPoints(vec3 cpBezier[16], int patchParam)
{
    // Each edge of a transition patch is adjacent to one or two patches
    // at the next refined level of subdivision. When the patch control
    // points have been converted to the Bezier basis, the control points
    // at the four corners are on the limit surface (since a Bezier patch
    // interpolates its corner control points). We can compute an adaptive
    // tessellation level for transition edges on the limit surface by
    // evaluating a limit position at the mid point of each transition edge.

    tessOuterLo = vec4(1);
    tessOuterHi = vec4(0);

    if (((patchParam >> 11) & 1) != 0) {
        vec3 ev03 = EvalBezier(cpBezier, vec2(0.0, 0.5));
        tessOuterLo[0] = TessAdaptive(cpBezier[0], ev03);
        tessOuterHi[0] = TessAdaptive(cpBezier[12], ev03);
    } else {
        tessOuterLo[0] = TessAdaptive(cpBezier[0], cpBezier[12]);
    }
    if (((patchParam >> 8) & 1) != 0) {
        vec3 ev01 = EvalBezier(cpBezier, vec2(0.5, 0.0));
        tessOuterLo[1] = TessAdaptive(cpBezier[0], ev01);
        tessOuterHi[1] = TessAdaptive(cpBezier[3], ev01);
    } else {
        tessOuterLo[1] = TessAdaptive(cpBezier[0], cpBezier[3]);
    }
    if (((patchParam >> 9) & 1) != 0) {
        vec3 ev12 = EvalBezier(cpBezier, vec2(1.0, 0.5));
        tessOuterLo[2] = TessAdaptive(cpBezier[3], ev12);
        tessOuterHi[2] = TessAdaptive(cpBezier[15], ev12);
    } else {
        tessOuterLo[2] = TessAdaptive(cpBezier[3], cpBezier[15]);
    }
    if (((patchParam >> 10) & 1) != 0) {
        vec3 ev23 = EvalBezier(cpBezier, vec2(0.5, 1.0));
        tessOuterLo[3] = TessAdaptive(cpBezier[12], ev23);
        tessOuterHi[3] = TessAdaptive(cpBezier[15], ev23);
    } else {
        tessOuterLo[3] = TessAdaptive(cpBezier[12], cpBezier[15]);
    }
}

void
GetTransitionTessLevels(vec3 cp[16], int patchParam,
                        inout vec4 outerLevel, inout vec2 innerLevel)
{
#if defined OSD_ENABLE_SCREENSPACE_TESSELLATION
    GetTessLevelsLimitPoints(cp, patchParam);
#elif defined OSD_ENABLE_SCREENSPACE_TESSELLATION_REFINED
    GetTessLevelsRefinedPoints(cp, patchParam);
#else
    GetTessLevelsUniform(cp, patchParam);
#endif

    // Outer levels are the sum of the Lo and Hi segments where the Hi
    // segments will have a length of zero for non-transition edges.
    outerLevel = tessOuterLo + tessOuterHi;

    // Inner levels are the average the corresponding outer levels.
    innerLevel[0] = (outerLevel[1] + outerLevel[3]) * 0.5;
    innerLevel[1] = (outerLevel[0] + outerLevel[2]) * 0.5;
}

#endif

//----------------------------------------------------------
// Patches.TessEvalTransition
//----------------------------------------------------------
#ifdef OSD_PATCH_TESS_EVAL_BSPLINE_SHADER

patch in vec4 tessOuterLo, tessOuterHi;

float
GetTransitionSplit(float t, float n0, float n1)
{
    float ti = round(t * (n0 + n1));

    if (ti <= n0) {
        return 0.5 * (ti / n0);
    } else {
        return 0.5 * ((ti - n0) / n1) + 0.5;
    }
}

vec2
GetTransitionParameterization()
{
    vec2 UV = gl_TessCoord.xy;
    if (UV.x == 0 && tessOuterHi[0] > 0) {
        UV.y = GetTransitionSplit(UV.y, tessOuterLo[0], tessOuterHi[0]);
    } else
    if (UV.y == 0 && tessOuterHi[1] > 0) {
        UV.x = GetTransitionSplit(UV.x, tessOuterLo[1], tessOuterHi[1]);
    } else
    if (UV.x == 1 && tessOuterHi[2] > 0) {
        UV.y = GetTransitionSplit(UV.y, tessOuterLo[2], tessOuterHi[2]);
    } else
    if (UV.y == 1 && tessOuterHi[3] > 0) {
        UV.x = GetTransitionSplit(UV.x, tessOuterLo[3], tessOuterHi[3]);
    }
    return UV;
}

#endif
