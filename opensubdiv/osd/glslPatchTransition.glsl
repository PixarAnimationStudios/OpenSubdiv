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
GetTransitionTessLevels(vec3 cp[24], int patchParam, inout vec4 outer, inout vec2 inner)
{
    // Each edge of a transition patch is adjacent to one or two patches
    // at the next refined level of subdivision. We compute the corresponding
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

    tessOuterLo = vec4(1);
    tessOuterHi = vec4(0);

#ifdef OSD_ENABLE_SCREENSPACE_TESSELLATION
    float tessAmount = GetTessLevel(GetPatchLevel());

    if (((patchParam >> 11) & 1) != 0) {
        tessOuterLo[0] = TessAdaptive(cp[16], cp[17]);
        tessOuterHi[0] = TessAdaptive(cp[17], cp[18]);
    } else {
        tessOuterLo[0] = TessAdaptive(cp[5], cp[6]);
    }
    if (((patchParam >> 8) & 1) != 0) {
        tessOuterLo[1] = TessAdaptive(cp[18], cp[19]);
        tessOuterHi[1] = TessAdaptive(cp[19], cp[20]);
    } else {
        tessOuterLo[1] = TessAdaptive(cp[6], cp[10]);
    }
    if (((patchParam >> 9) & 1) != 0) {
        tessOuterHi[2] = TessAdaptive(cp[20], cp[21]);
        tessOuterLo[2] = TessAdaptive(cp[21], cp[22]);
    } else {
        tessOuterLo[2] = TessAdaptive(cp[9], cp[10]);
    }
    if (((patchParam >> 10) & 1) != 0) {
        tessOuterHi[3] = TessAdaptive(cp[22], cp[23]);
        tessOuterLo[3] = TessAdaptive(cp[23], cp[16]);
    } else {
        tessOuterLo[3] = TessAdaptive(cp[5], cp[9]);
    }
#else
    float tessAmount = GetTessLevel(GetPatchLevel());

    tessOuterLo[0] = tessAmount;
    tessOuterLo[1] = tessAmount;
    tessOuterLo[2] = tessAmount;
    tessOuterLo[3] = tessAmount;

    tessOuterHi[0] = tessAmount * ((patchParam >> 11) & 1);
    tessOuterHi[1] = tessAmount * ((patchParam >> 8) & 1);
    tessOuterHi[2] = tessAmount * ((patchParam >> 9) & 1);
    tessOuterHi[3] = tessAmount * ((patchParam >> 10) & 1);
#endif

    outer = tessOuterLo + tessOuterHi;
    inner[0] = (outer[0] + outer[2]) * 0.5;
    inner[1] = (outer[1] + outer[3]) * 0.5;
}

#endif

//----------------------------------------------------------
// Patches.TessEvalTransition
//----------------------------------------------------------
#ifdef OSD_PATCH_TESS_EVAL_BSPLINE_SHADER

patch in vec4 tessOuterLow, tessOuterHigh;

vec2
GetTransitionSubpatchUV()
{
#ifdef OSD_ENABLE_SCREENSPACE_TESSELLATION
    // XXXdyu-patch-drawing debug -- just split along transitions
    vec2 uv = gl_TessCoord.xy;
#else
    vec2 uv = gl_TessCoord.xy;
#endif
    return uv;
}

#endif
