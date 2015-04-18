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
// Patches.HullTransition
//----------------------------------------------------------

float
TessAdaptiveRound(float3 p0, float3 p1)
{
    return round(TessAdaptive(p0, p1));
}

void
GetTransitionTessLevels(
        float3 cp[24], int patchParam, int patchLevel,
        inout float4 outer, inout float4 inner,
        inout float4 tessOuterLo, inout float4 tessOuterHi)
{
    // Each edge of a transition patch is adjacent to one or two patches
    // at the next refined level of subdivision. We COmpute the corresponding
    // vertex-vertex and edge-vertex refined points along the edges of the
    // patch using Catmull-Clark subdivision stencil weights.
    // For simplicity, we let the optimizer discard unused computation.
    cp[16] = (cp[0] + cp[2] + cp[8] + cp[10]) * 0.015625 +
             (cp[1] + cp[4] + cp[6] + cp[9]) * 0.09375 + cp[5] * 0.5625;
    cp[17] = (cp[1] + cp[2] + cp[9] + cp[10]) * 0.0625 + (cp[5] + cp[6]) * 0.375;

    cp[18] = (cp[1] + cp[3] + cp[9] + cp[11]) * 0.015625 +
             (cp[2] + cp[5] + cp[7] + cp[10]) * 0.09375 + cp[6] * 0.5625;
    cp[19] = (cp[5] + cp[7] + cp[9] + cp[11]) * 0.0625 + (cp[6] + cp[10]) * 0.375;

    cp[20] = (cp[5] + cp[7] + cp[13] + cp[15]) * 0.015625 +
             (cp[6] + cp[9] + cp[11] + cp[14]) * 0.09375 + cp[10] * 0.5625;
    cp[21] = (cp[5] + cp[6] + cp[13] + cp[14]) * 0.0625 + (cp[9] + cp[10]) * 0.375;

    cp[22] = (cp[4] + cp[6] + cp[12] + cp[14]) * 0.015625 +
             (cp[5] + cp[8] + cp[10] + cp[13]) * 0.09375 + cp[9] * 0.5625;
    cp[23] = (cp[4] + cp[6] + cp[8] + cp[10]) * 0.0625 + (cp[5] + cp[9]) * 0.375;
    tessOuterLo = float4(1,1,1,1);
    tessOuterHi = float4(0,0,0,0);

#ifdef OSD_ENABLE_SCREENSPACE_TESSELLATION
    if (((patchParam >> 11) & 1) != 0) {
        tessOuterLo[0] = TessAdaptiveRound(cp[23], cp[16]);
        tessOuterHi[0] = TessAdaptiveRound(cp[22], cp[23]);
    } else {
        tessOuterLo[0] = TessAdaptiveRound(cp[5], cp[9]);
    }
    if (((patchParam >> 8) & 1) != 0) {
        tessOuterLo[1] = TessAdaptiveRound(cp[16], cp[17]);
        tessOuterHi[1] = TessAdaptiveRound(cp[17], cp[18]);
    } else {
        tessOuterLo[1] = TessAdaptiveRound(cp[5], cp[6]);
    }
    if (((patchParam >> 9) & 1) != 0) {
        tessOuterLo[2] = TessAdaptiveRound(cp[18], cp[19]);
        tessOuterHi[2] = TessAdaptiveRound(cp[19], cp[20]);
    } else {
        tessOuterLo[2] = TessAdaptiveRound(cp[6], cp[10]);
    }
    if (((patchParam >> 10) & 1) != 0) {
        tessOuterLo[3] = TessAdaptiveRound(cp[21], cp[22]);
        tessOuterHi[3] = TessAdaptiveRound(cp[20], cp[21]);
    } else {
        tessOuterLo[3] = TessAdaptiveRound(cp[9], cp[10]);
    }
#else
    float tessAmount = GetTessLevel(patchLevel);

    tessOuterLo[0] = tessAmount;
    tessOuterLo[1] = tessAmount;
    tessOuterLo[2] = tessAmount;
    tessOuterLo[3] = tessAmount;
#endif

    outer = tessOuterLo + tessOuterHi;
    inner[0] = (outer[0] + outer[2]) * 0.5;
    inner[1] = (outer[1] + outer[3]) * 0.5;
}

//----------------------------------------------------------
// Patches.DomainTransition
//----------------------------------------------------------

float
GetTransitionSplit(float t, float n0, float n1)
{
    float n = round(n0 + n1);
    float ti = round(t * n);

    if (ti <= n0) {
        return 0.5 * (ti / n0);
    } else {
        return 0.5 * ((ti - n0) / n1) + 0.5;
    }
}

float2
GetTransitionParameterization(
    in HS_CONSTANT_FUNC_OUT input,
    in float2 uv)
{
    float2 UV = uv.xy;
    if (UV.x == 0 && input.tessOuterHi[0] > 0) {
        UV.y = GetTransitionSplit(UV.y, input.tessOuterLo[0], input.tessOuterHi[0]);
    } else
    if (UV.y == 0 && input.tessOuterHi[1] > 0) {
        UV.x = GetTransitionSplit(UV.x, input.tessOuterLo[1], input.tessOuterHi[1]);
    } else
    if (UV.x == 1 && input.tessOuterHi[2] > 0) {
        UV.y = GetTransitionSplit(UV.y, input.tessOuterLo[2], input.tessOuterHi[2]);
    } else
    if (UV.y == 1 && input.tessOuterHi[3] > 0) {
        UV.x = GetTransitionSplit(UV.x, input.tessOuterLo[3], input.tessOuterHi[3]);
    }
    return UV;
}
