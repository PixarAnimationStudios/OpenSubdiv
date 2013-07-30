//
//     Copyright 2013 Pixar
//
//     Licensed under the Apache License, Version 2.0 (the "License");
//     you may not use this file except in compliance with the License
//     and the following modification to it: Section 6 Trademarks.
//     deleted and replaced with:
//
//     6. Trademarks. This License does not grant permission to use the
//     trade names, trademarks, service marks, or product names of the
//     Licensor and its affiliates, except as required for reproducing
//     the content of the NOTICE file.
//
//     You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
//     Unless required by applicable law or agreed to in writing,
//     software distributed under the License is distributed on an
//     "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
//     either express or implied.  See the License for the specific
//     language governing permissions and limitations under the
//     License.
//

#if defined(OSD_TRANSITION_PATTERN00) || defined(OSD_TRANSITION_PATTERN01) || defined(OSD_TRANSITION_PATTERN02) || defined(OSD_TRANSITION_PATTERN10) || defined(OSD_TRANSITION_PATTERN11) || defined(OSD_TRANSITION_PATTERN12) || defined(OSD_TRANSITION_PATTERN13) || defined(OSD_TRANSITION_PATTERN21) || defined(OSD_TRANSITION_PATTERN22) || defined(OSD_TRANSITION_PATTERN23)

    #define OSD_TRANSITION_TRIANGLE_SUBPATCH

#else

    #undef OSD_TRANSITION_TRIANGLE_SUBPATCH

#endif

struct HS_CONSTANT_TRANSITION_FUNC_OUT {
#ifdef OSD_TRANSITION_TRIANGLE_SUBPATCH
    float tessLevelInner    : SV_InsideTessFactor;
    float tessLevelOuter[3] : SV_TessFactor;
#else
    float tessLevelInner[2] : SV_InsideTessFactor;
    float tessLevelOuter[4] : SV_TessFactor;
#endif
};

//----------------------------------------------------------
// Patches.HullTransition
//----------------------------------------------------------

void
SetTransitionTessLevels(inout HS_CONSTANT_TRANSITION_FUNC_OUT output, float3 cp[OSD_PATCH_INPUT_SIZE], int patchLevel)
{
#ifdef OSD_ENABLE_SCREENSPACE_TESSELLATION
    // These tables map the 9, 12, or 16 input control points onto the
    // canonical 16 control points for a regular patch.
#if defined OSD_PATCH_BOUNDARY
    const int p[16] = { 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
#elif defined OSD_PATCH_CORNER
    const int p[16] = { 0, 1, 2, 2, 0, 1, 2, 2, 3, 4, 5, 5, 6, 7, 8, 8 };
#else
    const int p[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
#endif

#if OSD_TRANSITION_ROTATE == 0
    const int r[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
#elif OSD_TRANSITION_ROTATE == 1
    const int r[16] = { 12, 8, 4, 0, 13, 9, 5, 1, 14, 10, 6, 2, 15, 11, 7, 3 };
#elif OSD_TRANSITION_ROTATE == 2
    const int r[16] = { 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 };
#elif OSD_TRANSITION_ROTATE == 3
    const int r[16] = { 3, 7, 11, 15, 2, 6, 10, 14, 1, 5, 9, 13, 0, 4, 8, 12 };
#endif

    // Expand and rotate control points using remapping tables above
    float3 pv0 = cp[p[r[0]]];
    float3 pv1 = cp[p[r[1]]];
    float3 pv2 = cp[p[r[2]]];
    float3 pv3 = cp[p[r[3]]];

    float3 pv4 = cp[p[r[4]]];
    float3 pv5 = cp[p[r[5]]];
    float3 pv6 = cp[p[r[6]]];
    float3 pv7 = cp[p[r[7]]];

    float3 pv8 = cp[p[r[8]]];
    float3 pv9 = cp[p[r[9]]];
    float3 pv10 = cp[p[r[10]]];
    float3 pv11 = cp[p[r[11]]];

    float3 pv12 = cp[p[r[12]]];
    float3 pv13 = cp[p[r[13]]];
    float3 pv14 = cp[p[r[14]]];
    float3 pv15 = cp[p[r[15]]];

    // Each edge of a transition patch is adjacent to one or two 
    // patches at the next refined level of subdivision.
    // Compute the corresponding vertex-vertex and edge-vertex refined
    // points along the edges of the patch using Catmull-Clark subdivision
    // stencil weights.
    // For simplicity, we let the optimizer discard unused computation.
    float3 vv0 = (pv0 + pv2 + pv8 + pv10) * 0.015625 +
                 (pv1 + pv4 + pv6 + pv9) * 0.09375 + pv5 * 0.5625;
    float3 ev01 = (pv1 + pv2 + pv9 + pv10) * 0.0625 + (pv5 + pv6) * 0.375;

    float3 vv1 = (pv1 + pv3 + pv9 + pv11) * 0.015625 +
                 (pv2 + pv5 + pv7 + pv10) * 0.09375 + pv6 * 0.5625;
    float3 ev12 = (pv5 + pv7 + pv9 + pv11) * 0.0625 + (pv6 + pv10) * 0.375;

    float3 vv2 = (pv5 + pv7 + pv13 + pv15) * 0.015625 +
                 (pv6 + pv9 + pv11 + pv14) * 0.09375 + pv10 * 0.5625;
    float3 ev23 = (pv5 + pv6 + pv13 + pv14) * 0.0625 + (pv9 + pv10) * 0.375;

    float3 vv3 = (pv4 + pv6 + pv12 + pv14) * 0.015625 +
                 (pv5 + pv8 + pv10 + pv13) * 0.09375 + pv9 * 0.5625;
    float3 ev30 = (pv4 + pv6 + pv8 + pv10) * 0.0625 + (pv5 + pv9) * 0.375;

    // The vertices along boundaries and at corners are refined specially.
#if defined OSD_PATCH_BOUNDARY
#if OSD_TRANSITION_ROTATE == 0
    vv0 = (pv4 + pv6) * 0.125 + pv5 * 0.75;
    vv1 = (pv5 + pv7) * 0.125 + pv6 * 0.75;
#elif OSD_TRANSITION_ROTATE == 1
    vv1 = (pv2 + pv10) * 0.125 + pv6 * 0.75;
    vv2 = (pv6 + pv14) * 0.125 + pv10 * 0.75;
#elif OSD_TRANSITION_ROTATE == 2
    vv2 = (pv9 + pv11) * 0.125 + pv10 * 0.75;
    vv3 = (pv8 + pv10) * 0.125 + pv9 * 0.75;
#elif OSD_TRANSITION_ROTATE == 3
    vv3 = (pv5 + pv13) * 0.125 + pv9 * 0.75;
    vv0 = (pv1 + pv9) * 0.125 + pv5 * 0.75;
#endif
#elif defined OSD_PATCH_CORNER
#if OSD_TRANSITION_ROTATE == 0
    vv0 = (pv4 + pv6) * 0.125 + pv5 * 0.75;
    vv1 = pv6;
    vv2 = (pv6 + pv14) * 0.125 + pv10 * 0.75;
#elif OSD_TRANSITION_ROTATE == 1
    vv1 = (pv5 + pv7) * 0.125 + pv6 * 0.75;
    vv2 = pv10;
    vv3 = (pv8 + pv10) * 0.125 + pv9 * 0.75;
#elif OSD_TRANSITION_ROTATE == 2
    vv2 = (pv6 + pv14) * 0.125 + pv10 * 0.75;
    vv3 = pv9;
    vv0 = (pv4 + pv6) * 0.125 + pv5 * 0.75;
#elif OSD_TRANSITION_ROTATE == 3
    vv3 = (pv8 + pv10) * 0.125 + pv9 * 0.75;
    vv0 = pv5;
    vv1 = (pv5 + pv7) * 0.125 + pv6 * 0.75;
#endif
#endif

#ifdef OSD_TRANSITION_PATTERN00
    output.tessLevelOuter[0] = TessAdaptive(ev01, pv9) * 0.5;
    output.tessLevelOuter[1] = TessAdaptive(ev01, pv10) * 0.5;
    output.tessLevelOuter[2] = TessAdaptive(pv9, pv10);

    output.tessLevelInner    =
        (output.tessLevelOuter[0] + output.tessLevelOuter[1] + output.tessLevelOuter[2]) * 0.5;
#endif
#ifdef OSD_TRANSITION_PATTERN01
    output.tessLevelOuter[0] = TessAdaptive(ev01, vv1);
    output.tessLevelOuter[1] = TessAdaptive(pv6, pv10);
    output.tessLevelOuter[2] = TessAdaptive(ev01, pv10) * 0.5;

    output.tessLevelInner    =
        (output.tessLevelOuter[0] + output.tessLevelOuter[1] + output.tessLevelOuter[2]) * 0.25;
#endif
#ifdef OSD_TRANSITION_PATTERN02
    output.tessLevelOuter[0] = TessAdaptive(ev01, vv0);
    output.tessLevelOuter[1] = TessAdaptive(ev01, pv9) * 0.5;
    output.tessLevelOuter[2] = TessAdaptive(pv5, pv9);

    output.tessLevelInner    =
        (output.tessLevelOuter[0] + output.tessLevelOuter[1] + output.tessLevelOuter[2]) * 0.25;
#endif


#ifdef OSD_TRANSITION_PATTERN10 
    output.tessLevelOuter[0] = TessAdaptive(pv6, pv10);
    output.tessLevelOuter[1] = TessAdaptive(ev01, pv10);
    output.tessLevelOuter[2] = TessAdaptive(ev01, vv1);

    output.tessLevelInner    =
        (output.tessLevelOuter[0] + output.tessLevelOuter[1]) * 0.25;
#endif
#ifdef OSD_TRANSITION_PATTERN11
    output.tessLevelOuter[0] = TessAdaptive(pv9, pv10);
    output.tessLevelOuter[1] = TessAdaptive(ev30, vv3);
    output.tessLevelOuter[2] = TessAdaptive(ev30, pv10);

    output.tessLevelInner    =
        (output.tessLevelOuter[0] + output.tessLevelOuter[2]) * 0.25;
#endif
#ifdef OSD_TRANSITION_PATTERN12
    output.tessLevelOuter[0] = TessAdaptive(ev30, vv0);
    output.tessLevelOuter[1] = TessAdaptive(ev01, vv0);
    output.tessLevelOuter[2] = TessAdaptive(ev01, ev30);

    output.tessLevelInner    =
        (output.tessLevelOuter[0] + output.tessLevelOuter[1] + output.tessLevelOuter[2]) * 0.25;
#endif
#ifdef OSD_TRANSITION_PATTERN13
    output.tessLevelOuter[0] = TessAdaptive(ev01, pv10);
    output.tessLevelOuter[1] = TessAdaptive(ev30, pv10);
    output.tessLevelOuter[2] = TessAdaptive(ev01, ev30);

    output.tessLevelInner    =
        (output.tessLevelOuter[0] + output.tessLevelOuter[1] + output.tessLevelOuter[2]) * 0.25;
#endif


#ifdef OSD_TRANSITION_PATTERN20
    output.tessLevelOuter[0] = TessAdaptive(pv5, pv6);
    output.tessLevelOuter[1] = TessAdaptive(ev12, vv1);
    output.tessLevelOuter[2] = TessAdaptive(ev12, ev30);
    output.tessLevelOuter[3] = TessAdaptive(ev30, vv0);

    output.tessLevelInner[0] =
        max(output.tessLevelOuter[1], output.tessLevelOuter[3]);
    output.tessLevelInner[1] =
        max(output.tessLevelOuter[0], output.tessLevelOuter[2]);
#endif
#ifdef OSD_TRANSITION_PATTERN21
    output.tessLevelOuter[0] = TessAdaptive(ev23, ev30) * 0.5;
    output.tessLevelOuter[1] = TessAdaptive(ev23, vv3);
    output.tessLevelOuter[2] = TessAdaptive(ev30, vv3);

    output.tessLevelInner    =
        (output.tessLevelOuter[1] + output.tessLevelOuter[2]) * 0.5;
#endif
#ifdef OSD_TRANSITION_PATTERN22
    output.tessLevelOuter[0] = TessAdaptive(ev12, vv2);
    output.tessLevelOuter[1] = TessAdaptive(ev23, vv2);
    output.tessLevelOuter[2] = TessAdaptive(ev12, ev23) * 0.5;

    output.tessLevelInner    =
        (output.tessLevelOuter[0] + output.tessLevelOuter[1]) * 0.5;
#endif
#ifdef OSD_TRANSITION_PATTERN23
    output.tessLevelOuter[0] = TessAdaptive(ev12, ev30);
    output.tessLevelOuter[1] = TessAdaptive(ev12, ev23) * 0.5;
    output.tessLevelOuter[2] = TessAdaptive(ev23, ev30) * 0.5;

    output.tessLevelInner    =
        (output.tessLevelOuter[0] + output.tessLevelOuter[1] + output.tessLevelOuter[2]) * 0.5;
#endif


#ifdef OSD_TRANSITION_PATTERN30
    output.tessLevelOuter[0] = TessAdaptive(ev30, ev12) * 0.5;
    output.tessLevelOuter[1] = TessAdaptive(ev30, vv0);
    output.tessLevelOuter[2] = TessAdaptive(ev01, vv0);
    output.tessLevelOuter[3] = TessAdaptive(ev01, ev23) * 0.5;
    output.tessLevelInner[0] =
        max(output.tessLevelOuter[1], output.tessLevelOuter[3]);
    output.tessLevelInner[1] =
        max(output.tessLevelOuter[0], output.tessLevelOuter[2]);
#endif
#ifdef OSD_TRANSITION_PATTERN31
    output.tessLevelOuter[0] = TessAdaptive(ev01, ev23) * 0.5;
    output.tessLevelOuter[1] = TessAdaptive(ev23, vv3);
    output.tessLevelOuter[2] = TessAdaptive(ev30, vv3);
    output.tessLevelOuter[3] = TessAdaptive(ev30, ev12) * 0.5;
    output.tessLevelInner[0] =
        max(output.tessLevelOuter[1], output.tessLevelOuter[3]);
    output.tessLevelInner[1] =
        max(output.tessLevelOuter[0], output.tessLevelOuter[2]);
#endif
#ifdef OSD_TRANSITION_PATTERN32
    output.tessLevelOuter[0] = TessAdaptive(ev23, ev01) * 0.5;
    output.tessLevelOuter[1] = TessAdaptive(ev01, vv1);
    output.tessLevelOuter[2] = TessAdaptive(ev12, vv1);
    output.tessLevelOuter[3] = TessAdaptive(ev12, ev30) * 0.5;
    output.tessLevelInner[0] =
        max(output.tessLevelOuter[1], output.tessLevelOuter[3]);
    output.tessLevelInner[1] =
        max(output.tessLevelOuter[0], output.tessLevelOuter[2]);
#endif
#ifdef OSD_TRANSITION_PATTERN33
    output.tessLevelOuter[0] = TessAdaptive(ev12, ev30) * 0.5;
    output.tessLevelOuter[1] = TessAdaptive(ev12, vv2);
    output.tessLevelOuter[2] = TessAdaptive(ev23, vv2);
    output.tessLevelOuter[3] = TessAdaptive(ev01, ev23) * 0.5;
    output.tessLevelInner[0] =
        max(output.tessLevelOuter[1], output.tessLevelOuter[3]);
    output.tessLevelInner[1] =
        max(output.tessLevelOuter[0], output.tessLevelOuter[2]);
#endif


#ifdef OSD_TRANSITION_PATTERN40
    output.tessLevelOuter[0] = TessAdaptive(ev01, vv0);
    output.tessLevelOuter[1] = TessAdaptive(ev01, ev23);
    output.tessLevelOuter[2] = TessAdaptive(ev23, vv3);
    output.tessLevelOuter[3] = TessAdaptive(pv5, pv9);

    output.tessLevelInner[0] =
        max(output.tessLevelOuter[1], output.tessLevelOuter[3]);
    output.tessLevelInner[1] =
        max(output.tessLevelOuter[0], output.tessLevelOuter[2]);
#endif
#ifdef OSD_TRANSITION_PATTERN41
    output.tessLevelOuter[0] = TessAdaptive(ev01, vv1);
    output.tessLevelOuter[1] = TessAdaptive(pv6, pv10);
    output.tessLevelOuter[2] = TessAdaptive(ev23, vv2);
    output.tessLevelOuter[3] = TessAdaptive(ev01, ev23);

    output.tessLevelInner[0] =
        max(output.tessLevelOuter[1], output.tessLevelOuter[3]);
    output.tessLevelInner[1] =
        max(output.tessLevelOuter[0], output.tessLevelOuter[2]);
#endif

#else // OSD_ENABLE_SCREENSPACE_TESSELLATION

    // XXX: HLSL compiler crashes with an internal compiler error occasionaly
    // if this shader accesses a shader resource buffer or a constant buffer
    // from this hull constant function.
    //float TessAmount = GetTessLevel(patchLevel);
    //float TessAmount = GetTessLevel(0);
    float TessAmount = 2.0;

#ifdef OSD_TRANSITION_PATTERN00
    float side = sqrt(1.25)*TessAmount;
    output.tessLevelOuter[0] = side;
    output.tessLevelOuter[1] = side;
    output.tessLevelOuter[2] = TessAmount;

    output.tessLevelInner    = TessAmount;
#endif
#ifdef OSD_TRANSITION_PATTERN01
    float side =  sqrt(1.25)*TessAmount;
    output.tessLevelOuter[0] = TessAmount/2.0;
    output.tessLevelOuter[1] = TessAmount;
    output.tessLevelOuter[2] = side;

    output.tessLevelInner    = TessAmount/2.0;
#endif
#ifdef OSD_TRANSITION_PATTERN02
    float side =  sqrt(1.25)*TessAmount;
    output.tessLevelOuter[0] = TessAmount/2.0;
    output.tessLevelOuter[1] = side;
    output.tessLevelOuter[2] = TessAmount;

    output.tessLevelInner    = TessAmount/2.0;
#endif
#ifdef OSD_TRANSITION_PATTERN10 
    float side = sqrt(1.25) * TessAmount;
    output.tessLevelOuter[0] = TessAmount;
    output.tessLevelOuter[1] = side;
    output.tessLevelOuter[2] = TessAmount/2.0;

    output.tessLevelInner    = TessAmount/2;
#endif
#ifdef OSD_TRANSITION_PATTERN11
    float side = sqrt(1.25) * TessAmount;
    output.tessLevelOuter[0] = TessAmount;
    output.tessLevelOuter[1] = TessAmount/2.0;
    output.tessLevelOuter[2] = side;

    output.tessLevelInner    = TessAmount/2;
#endif
#ifdef OSD_TRANSITION_PATTERN12
    float side = sqrt(0.125) * TessAmount;
    output.tessLevelOuter[0] = TessAmount/2.0;
    output.tessLevelOuter[1] = TessAmount/2.0;
    output.tessLevelOuter[2] = side;

    output.tessLevelInner    = TessAmount/2;
#endif
#ifdef OSD_TRANSITION_PATTERN13
    float side1 = sqrt(1.25) * TessAmount;
    float side2 = sqrt(0.125) * TessAmount;
    output.tessLevelOuter[0] = side1;
    output.tessLevelOuter[1] = side1;
    output.tessLevelOuter[2] = side2;

    output.tessLevelInner    = TessAmount/2.0*1.414;
#endif


#ifdef OSD_TRANSITION_PATTERN20
    output.tessLevelOuter[0] = TessAmount;
    output.tessLevelOuter[1] = TessAmount/2.0;
    output.tessLevelOuter[2] = TessAmount;
    output.tessLevelOuter[3] = TessAmount/2.0;

    output.tessLevelInner[0] = TessAmount/2.0;
    output.tessLevelInner[1] = TessAmount;
#endif
#ifdef OSD_TRANSITION_PATTERN21
    float side = sqrt(0.125) * TessAmount;
    output.tessLevelOuter[0] = side;
    output.tessLevelOuter[1] = TessAmount/2.0;
    output.tessLevelOuter[2] = TessAmount/2.0;

    output.tessLevelInner    = TessAmount/2.0;
#endif
#ifdef OSD_TRANSITION_PATTERN22
    float side = sqrt(0.125) * TessAmount;
    output.tessLevelOuter[0] = TessAmount/2.0;
    output.tessLevelOuter[1] = TessAmount/2.0;
    output.tessLevelOuter[2] = side;

    output.tessLevelInner    = TessAmount/2.0;
#endif
#ifdef OSD_TRANSITION_PATTERN23
    float side = sqrt(0.125) * TessAmount;
    output.tessLevelOuter[0] = TessAmount;
    output.tessLevelOuter[1] = side;
    output.tessLevelOuter[2] = side;

    output.tessLevelInner    = TessAmount/2.0;
#endif


#ifdef OSD_TRANSITION_PATTERN30
    output.tessLevelOuter[0] = output.tessLevelOuter[1] =
    output.tessLevelOuter[2] = output.tessLevelOuter[3] = TessAmount/2.0;
    output.tessLevelInner[0] = output.tessLevelInner[1] = TessAmount/2.0;
#endif
#ifdef OSD_TRANSITION_PATTERN31
    output.tessLevelOuter[0] = output.tessLevelOuter[1] =
    output.tessLevelOuter[2] = output.tessLevelOuter[3] = TessAmount/2.0;
    output.tessLevelInner[0] = output.tessLevelInner[1] = TessAmount/2.0;
#endif
#ifdef OSD_TRANSITION_PATTERN32
    output.tessLevelOuter[0] = output.tessLevelOuter[1] =
    output.tessLevelOuter[2] = output.tessLevelOuter[3] = TessAmount/2.0;
    output.tessLevelInner[0] = output.tessLevelInner[1] = TessAmount/2.0;
#endif
#ifdef OSD_TRANSITION_PATTERN33
    output.tessLevelOuter[0] = output.tessLevelOuter[1] =
    output.tessLevelOuter[2] = output.tessLevelOuter[3] = TessAmount/2.0;
    output.tessLevelInner[0] = output.tessLevelInner[1] = TessAmount/2.0;
#endif


#ifdef OSD_TRANSITION_PATTERN40
    output.tessLevelOuter[0] = TessAmount/2.0;
    output.tessLevelOuter[1] = TessAmount;
    output.tessLevelOuter[2] = TessAmount/2.0;
    output.tessLevelOuter[3] = TessAmount;

    output.tessLevelInner[0] = TessAmount;
    output.tessLevelInner[1] = TessAmount/2.0;
#endif
#ifdef OSD_TRANSITION_PATTERN41
    output.tessLevelOuter[0] = TessAmount/2.0;
    output.tessLevelOuter[1] = TessAmount;
    output.tessLevelOuter[2] = TessAmount/2.0;
    output.tessLevelOuter[3] = TessAmount;

    output.tessLevelInner[0] = TessAmount;
    output.tessLevelInner[1] = TessAmount/2.0;
#endif

#endif // OSD_ENABLE_SCREENSPACE_TESSELLATION
}

//----------------------------------------------------------
// Patches.DomainTransition
//----------------------------------------------------------

float2
GetTransitionSubpatchUV(
#ifdef OSD_TRANSITION_TRIANGLE_SUBPATCH
    in float3 uvw
#else
    in float2 uv
#endif
)
{
    float2 UV = float2(0.0, 0.0);

//  OSD_TRANSITION_PATTERN0*
//  +-------------+
//  |     /\\     |
//  | 1  /  \\  2 |
//  |   /    \\   |
//  |  /      \\  |
//  | /    0   \\ |
//  |/          \\|
//  +-------------+

#ifdef OSD_TRANSITION_PATTERN00
    UV.x = 1.0-uvw.y-uvw.z/2;
    UV.y = 1.0-uvw.z;
#endif    
#ifdef OSD_TRANSITION_PATTERN01
    UV.x = 1.0-uvw.y/2;
    UV.y = uvw.x;
#endif    
#ifdef OSD_TRANSITION_PATTERN02
    UV.x = uvw.z/2;
    UV.y = uvw.x;
#endif

// OSD_TRANSITION_PATTERN1*
//  +-------------+
//  | 0   /\\   2 |
//  |    /   \\   |
//  |   /  3   \\ |
//  |  /       /  |
//  | /    /    1 |
//  |/ /          |
//  +-------------+

#ifdef OSD_TRANSITION_PATTERN10
    UV.x = 1.0-uvw.x/2.0;
    UV.y = uvw.z;
#endif
#ifdef OSD_TRANSITION_PATTERN11
    UV.x = uvw.y;
    UV.y = 1.0-uvw.x/2.0;
#endif
#ifdef OSD_TRANSITION_PATTERN12
    UV.x = uvw.x/2.0;
    UV.y = uvw.y/2.0;
#endif
#ifdef OSD_TRANSITION_PATTERN13
    UV.x = 1.0-uvw.x-uvw.y/2.0;
    UV.y = 1.0-uvw.y-uvw.x/2.0;
#endif

//  OSD_TRANSITION_PATTERN2*
//  +-------------+
//  |             |
//  |      0      |
//  |             |
//  |-------------|
//  |\\    3    / |
//  |  \\     /   |
//  | 1  \\ /   2 |
//  +-------------+

#ifdef OSD_TRANSITION_PATTERN20
    UV.x = 1.0-uv.y;
    UV.y = uv.x/2.0;
#endif
#ifdef OSD_TRANSITION_PATTERN21
    UV.x = uvw.z/2.0;
    UV.y = 1.0-uvw.y/2.0;
#endif
#ifdef OSD_TRANSITION_PATTERN22
    UV.x = 1.0-uvw.x/2.0;
    UV.y = 1.0-uvw.y/2.0;
#endif
#ifdef OSD_TRANSITION_PATTERN23
    UV.x = 1.0-uvw.y-uvw.x/2;
    UV.y = 0.5+uvw.x/2.0;
#endif

//  OSD_TRANSITION_PATTERN3*
//  +-------------+
//  |      |      |
//  |  1   |  0   |
//  |      |      |
//  |------|------|
//  |      |      |
//  |  3   |  2   |
//  |      |      |
//  +-------------+

#ifdef OSD_TRANSITION_PATTERN30
    UV.x = uv.y/2.0;
    UV.y = 0.5 - uv.x/2.0;
#endif
#ifdef OSD_TRANSITION_PATTERN31
    UV.x = 0.5 - uv.x/2.0;
    UV.y = 1.0 - uv.y/2.0;
#endif
#ifdef OSD_TRANSITION_PATTERN32
    UV.x = 0.5 + uv.x/2.0;
    UV.y = uv.y/2.0;
#endif
#ifdef OSD_TRANSITION_PATTERN33
    UV.x = 1.0 - uv.y/2.0;
    UV.y = 0.5 + uv.x/2.0;
#endif

//  OSD_TRANSITION_PATTERN4*
//  +-------------+
//  |      |      |
//  |      |      |
//  |      |      |
//  |  1   |   0  |
//  |      |      |
//  |      |      |
//  |      |      |
//  +-------------+

#ifdef OSD_TRANSITION_PATTERN40
    UV.x = 0.5 - uv.y/2.0;
    UV.y = uv.x;
#endif
#ifdef OSD_TRANSITION_PATTERN41
    UV.x = 1.0 - uv.y/2.0;
    UV.y = uv.x;
#endif

    return UV;
}
