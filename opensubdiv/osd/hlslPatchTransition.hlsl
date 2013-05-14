//
//     Copyright (C) Pixar. All rights reserved.
//
//     This license governs use of the accompanying software. If you
//     use the software, you accept this license. If you do not accept
//     the license, do not use the software.
//
//     1. Definitions
//     The terms "reproduce," "reproduction," "derivative works," and
//     "distribution" have the same meaning here as under U.S.
//     copyright law.  A "contribution" is the original software, or
//     any additions or changes to the software.
//     A "contributor" is any person or entity that distributes its
//     contribution under this license.
//     "Licensed patents" are a contributor's patent claims that read
//     directly on its contribution.
//
//     2. Grant of Rights
//     (A) Copyright Grant- Subject to the terms of this license,
//     including the license conditions and limitations in section 3,
//     each contributor grants you a non-exclusive, worldwide,
//     royalty-free copyright license to reproduce its contribution,
//     prepare derivative works of its contribution, and distribute
//     its contribution or any derivative works that you create.
//     (B) Patent Grant- Subject to the terms of this license,
//     including the license conditions and limitations in section 3,
//     each contributor grants you a non-exclusive, worldwide,
//     royalty-free license under its licensed patents to make, have
//     made, use, sell, offer for sale, import, and/or otherwise
//     dispose of its contribution in the software or derivative works
//     of the contribution in the software.
//
//     3. Conditions and Limitations
//     (A) No Trademark License- This license does not grant you
//     rights to use any contributor's name, logo, or trademarks.
//     (B) If you bring a patent claim against any contributor over
//     patents that you claim are infringed by the software, your
//     patent license from such contributor to the software ends
//     automatically.
//     (C) If you distribute any portion of the software, you must
//     retain all copyright, patent, trademark, and attribution
//     notices that are present in the software.
//     (D) If you distribute any portion of the software in source
//     code form, you may do so only under this license by including a
//     complete copy of this license with your distribution. If you
//     distribute any portion of the software in compiled or object
//     code form, you may only do so under a license that complies
//     with this license.
//     (E) The software is licensed "as-is." You bear the risk of
//     using it. The contributors give no express warranties,
//     guarantees or conditions. You may have additional consumer
//     rights under your local laws which this license cannot change.
//     To the extent permitted under your local laws, the contributors
//     exclude the implied warranties of merchantability, fitness for
//     a particular purpose and non-infringement.
//

#if defined(CASE00) || defined(CASE01) || defined(CASE02) || defined(CASE10) || defined(CASE11) || defined(CASE12) || defined(CASE13) || defined(CASE21) || defined(CASE22) || defined(CASE23)

    #define TRIANGLE
    #define HS_DOMAIN "tri"

#else

    #undef TRIANGLE
    #define HS_DOMAIN "quad"

#endif

#if defined BOUNDARY
    #define PATCH_INPUT_SIZE 12
#elif defined CORNER
    #define PATCH_INPUT_SIZE 9
#else
    #define PATCH_INPUT_SIZE 16
#endif

struct HS_CONSTANT_TRANSITION_FUNC_OUT {
#ifdef TRIANGLE
    float tessLevelInner    : SV_InsideTessFactor;
    float tessLevelOuter[3] : SV_TessFactor;
#else
    float tessLevelInner[2] : SV_InsideTessFactor;
    float tessLevelOuter[4] : SV_TessFactor;
#endif
};

//----------------------------------------------------------
// Patches.Coefficients
//----------------------------------------------------------

static float4x4 Q = {
    1.f/6.f, 2.f/3.f, 1.f/6.f, 0.f,
    0.f,     2.f/3.f, 1.f/3.f, 0.f,
    0.f,     1.f/3.f, 2.f/3.f, 0.f,
    0.f,     1.f/6.f, 2.f/3.f, 1.f/6.f
};

// Boundary
static float4x3 B = {
    1.0f,    0.0f,    0.0f,
    2.f/3.f, 1.f/3.f, 0.0f,
    1.f/3.f, 2.f/3.f, 0.0f,
    1.f/6.f, 2.f/3.f, 1.f/6.f
};

// Corner
static float4x4 R = {
    1.f/6.f, 2.f/3.f, 1.f/6.f, 0.0f,
    0.0f,    2.f/3.f, 1.f/3.f, 0.0f,
    0.0f,    1.f/3.f, 2.f/3.f, 0.0f,
    0.0f,    0.0f,    1.0f,    0.0f
};

//----------------------------------------------------------
// Patches.Vertex
//----------------------------------------------------------

void vs_main_patches( in InputVertex input,
                      out HullVertex output )
{
    output.position = mul(ModelViewMatrix, input.position);
    OSD_PATCH_CULL_COMPUTE_CLIPFLAGS(input.position);

#if OSD_NUM_VARYINGS > 0
    for (int i = 0; i< OSD_NUM_VARYINGS; ++i)
        output.varyings[i] = input.varyings[i];
#endif
}

//----------------------------------------------------------
// Patches.HullTransition
//----------------------------------------------------------

HS_CONSTANT_TRANSITION_FUNC_OUT HSConstFunc(
    InputPatch<HullVertex, PATCH_INPUT_SIZE> patch,
    uint primitiveID : SV_PrimitiveID)
{
    HS_CONSTANT_TRANSITION_FUNC_OUT output;
    int patchLevel = GetPatchLevel(primitiveID);

#ifdef TRIANGLE
    OSD_PATCH_CULL_TRIANGLE(PATCH_INPUT_SIZE);
#else
    OSD_PATCH_CULL(PATCH_INPUT_SIZE);
#endif

#if OSD_ENABLE_SCREENSPACE_TESSELLATION
    // These tables map the 9, 12, or 16 input control points onto the
    // canonical 16 control points for a regular patch.
#if defined BOUNDARY
    const int p[16] = { 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
#elif defined CORNER
    const int p[16] = { 0, 1, 2, 2, 0, 1, 2, 2, 3, 4, 5, 5, 6, 7, 8, 8 };
#else
    const int p[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
#endif

#if ROTATE == 0
    const int r[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
#elif ROTATE == 1
    const int r[16] = { 12, 8, 4, 0, 13, 9, 5, 1, 14, 10, 6, 2, 15, 11, 7, 3 };
#elif ROTATE == 2
    const int r[16] = { 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 };
#elif ROTATE == 3
    const int r[16] = { 3, 7, 11, 15, 2, 6, 10, 14, 1, 5, 9, 13, 0, 4, 8, 12 };
#endif

    // Expand and rotate control points using remapping tables above
    float3 pv0 = patch[p[r[0]]].position.xyz;
    float3 pv1 = patch[p[r[1]]].position.xyz;
    float3 pv2 = patch[p[r[2]]].position.xyz;
    float3 pv3 = patch[p[r[3]]].position.xyz;

    float3 pv4 = patch[p[r[4]]].position.xyz;
    float3 pv5 = patch[p[r[5]]].position.xyz;
    float3 pv6 = patch[p[r[6]]].position.xyz;
    float3 pv7 = patch[p[r[7]]].position.xyz;

    float3 pv8 = patch[p[r[8]]].position.xyz;
    float3 pv9 = patch[p[r[9]]].position.xyz;
    float3 pv10 = patch[p[r[10]]].position.xyz;
    float3 pv11 = patch[p[r[11]]].position.xyz;

    float3 pv12 = patch[p[r[12]]].position.xyz;
    float3 pv13 = patch[p[r[13]]].position.xyz;
    float3 pv14 = patch[p[r[14]]].position.xyz;
    float3 pv15 = patch[p[r[15]]].position.xyz;

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
#if defined BOUNDARY
    #if ROTATE == 0
        vv0 = (pv4 + pv6) * 0.125 + pv5 * 0.75;
        vv1 = (pv5 + pv7) * 0.125 + pv6 * 0.75;
    #elif ROTATE == 1
        vv1 = (pv2 + pv10) * 0.125 + pv6 * 0.75;
        vv2 = (pv6 + pv14) * 0.125 + pv10 * 0.75;
    #elif ROTATE == 2
        vv2 = (pv9 + pv11) * 0.125 + pv10 * 0.75;
        vv3 = (pv8 + pv10) * 0.125 + pv9 * 0.75;
    #elif ROTATE == 3
        vv3 = (pv5 + pv13) * 0.125 + pv9 * 0.75;
        vv0 = (pv1 + pv9) * 0.125 + pv5 * 0.75;
    #endif
#elif defined CORNER
    #if ROTATE == 0
        vv0 = (pv4 + pv6) * 0.125 + pv5 * 0.75;
        vv1 = pv6;
        vv2 = (pv6 + pv14) * 0.125 + pv10 * 0.75;
    #elif ROTATE == 1
        vv1 = (pv5 + pv7) * 0.125 + pv6 * 0.75;
        vv2 = pv10;
        vv3 = (pv8 + pv10) * 0.125 + pv9 * 0.75;
    #elif ROTATE == 2
        vv2 = (pv6 + pv14) * 0.125 + pv10 * 0.75;
        vv3 = pv9;
        vv0 = (pv4 + pv6) * 0.125 + pv5 * 0.75;
    #elif ROTATE == 3
        vv3 = (pv8 + pv10) * 0.125 + pv9 * 0.75;
        vv0 = pv5;
        vv1 = (pv5 + pv7) * 0.125 + pv6 * 0.75;
    #endif
#endif

    #ifdef CASE00
        output.tessLevelOuter[0] = TessAdaptive(ev01, pv9, patchLevel) * 0.5;
        output.tessLevelOuter[1] = TessAdaptive(ev01, pv10, patchLevel) * 0.5;
        output.tessLevelOuter[2] = TessAdaptive(pv9, pv10, patchLevel);

        output.tessLevelInner    =
            (output.tessLevelOuter[0] + output.tessLevelOuter[1] + output.tessLevelOuter[2]) * 0.5;
    #endif
    #ifdef CASE01
        output.tessLevelOuter[0] = TessAdaptive(ev01, vv1, patchLevel+1);
        output.tessLevelOuter[1] = TessAdaptive(pv6, pv10, patchLevel);
        output.tessLevelOuter[2] = TessAdaptive(ev01, pv10, patchLevel) * 0.5;

        output.tessLevelInner    =
            (output.tessLevelOuter[0] + output.tessLevelOuter[1] + output.tessLevelOuter[2]) * 0.25;
    #endif
    #ifdef CASE02
        output.tessLevelOuter[0] = TessAdaptive(ev01, vv0, patchLevel+1);
        output.tessLevelOuter[1] = TessAdaptive(ev01, pv9, patchLevel) * 0.5;
        output.tessLevelOuter[2] = TessAdaptive(pv5, pv9, patchLevel);

        output.tessLevelInner    =
            (output.tessLevelOuter[0] + output.tessLevelOuter[1] + output.tessLevelOuter[2]) * 0.25;
    #endif


    #ifdef CASE10 
        output.tessLevelOuter[0] = TessAdaptive(pv6, pv10, patchLevel);
        output.tessLevelOuter[1] = TessAdaptive(ev01, pv10, patchLevel);
        output.tessLevelOuter[2] = TessAdaptive(ev01, vv1, patchLevel+1);

        output.tessLevelInner    =
            (output.tessLevelOuter[0] + output.tessLevelOuter[1]) * 0.25;
    #endif
    #ifdef CASE11
        output.tessLevelOuter[0] = TessAdaptive(pv9, pv10, patchLevel);
        output.tessLevelOuter[1] = TessAdaptive(ev30, vv3, patchLevel+1);
        output.tessLevelOuter[2] = TessAdaptive(ev30, pv10, patchLevel);

        output.tessLevelInner    =
            (output.tessLevelOuter[0] + output.tessLevelOuter[2]) * 0.25;
    #endif
    #ifdef CASE12
        output.tessLevelOuter[0] = TessAdaptive(ev30, vv0, patchLevel+1);
        output.tessLevelOuter[1] = TessAdaptive(ev01, vv0, patchLevel+1);
        output.tessLevelOuter[2] = TessAdaptive(ev01, ev30, patchLevel);

        output.tessLevelInner    =
            (output.tessLevelOuter[0] + output.tessLevelOuter[1] + output.tessLevelOuter[2]) * 0.25;
    #endif
    #ifdef CASE13
        output.tessLevelOuter[0] = TessAdaptive(ev01, pv10, patchLevel);
        output.tessLevelOuter[1] = TessAdaptive(ev30, pv10, patchLevel);
        output.tessLevelOuter[2] = TessAdaptive(ev01, ev30, patchLevel);

        output.tessLevelInner    =
            (output.tessLevelOuter[0] + output.tessLevelOuter[1] + output.tessLevelOuter[2]) * 0.25;
    #endif


    #ifdef CASE20
        output.tessLevelOuter[0] = TessAdaptive(ev12, ev30, patchLevel);
        output.tessLevelOuter[1] = TessAdaptive(ev30, vv0, patchLevel+1);
        output.tessLevelOuter[2] = TessAdaptive(pv5, pv6, patchLevel);
        output.tessLevelOuter[3] = TessAdaptive(ev12, vv1, patchLevel+1);

        output.tessLevelInner[0] =
            max(output.tessLevelOuter[1], output.tessLevelOuter[3]);
        output.tessLevelInner[1] =
            max(output.tessLevelOuter[0], output.tessLevelOuter[2]);
    #endif
    #ifdef CASE21
        output.tessLevelOuter[0] = TessAdaptive(ev23, ev30, patchLevel) * 0.5;
        output.tessLevelOuter[1] = TessAdaptive(ev23, vv3, patchLevel+1);
        output.tessLevelOuter[2] = TessAdaptive(ev30, vv3, patchLevel+1);

        output.tessLevelInner    =
            (output.tessLevelOuter[1] + output.tessLevelOuter[2]) * 0.5;
    #endif
    #ifdef CASE22
        output.tessLevelOuter[0] = TessAdaptive(ev12, vv2, patchLevel+1);
        output.tessLevelOuter[1] = TessAdaptive(ev23, vv2, patchLevel+1);
        output.tessLevelOuter[2] = TessAdaptive(ev12, ev23, patchLevel) * 0.5;

        output.tessLevelInner    =
            (output.tessLevelOuter[0] + output.tessLevelOuter[1]) * 0.5;
    #endif
    #ifdef CASE23
        output.tessLevelOuter[0] = TessAdaptive(ev12, ev30, patchLevel);
        output.tessLevelOuter[1] = TessAdaptive(ev12, ev23, patchLevel) * 0.5;
        output.tessLevelOuter[2] = TessAdaptive(ev23, ev30, patchLevel) * 0.5;

        output.tessLevelInner    =
            (output.tessLevelOuter[0] + output.tessLevelOuter[1] + output.tessLevelOuter[2]) * 0.5;
    #endif


    #ifdef CASE30
        output.tessLevelOuter[0] = TessAdaptive(ev30, ev12, patchLevel) * 0.5;
        output.tessLevelOuter[1] = TessAdaptive(ev30, vv0, patchLevel+1);
        output.tessLevelOuter[2] = TessAdaptive(ev01, vv0, patchLevel+1);
        output.tessLevelOuter[3] = TessAdaptive(ev01, ev23, patchLevel) * 0.5;
        output.tessLevelInner[0] =
            max(output.tessLevelOuter[1], output.tessLevelOuter[3]);
        output.tessLevelInner[1] =
            max(output.tessLevelOuter[0], output.tessLevelOuter[2]);
    #endif
    #ifdef CASE31
        output.tessLevelOuter[0] = TessAdaptive(ev01, vv1, patchLevel+1);
        output.tessLevelOuter[1] = TessAdaptive(ev12, vv1, patchLevel+1);
        output.tessLevelOuter[2] = TessAdaptive(ev12, ev30, patchLevel) * 0.5;
        output.tessLevelOuter[3] = TessAdaptive(ev01, ev23, patchLevel) * 0.5;
        output.tessLevelInner[0] =
            max(output.tessLevelOuter[1], output.tessLevelOuter[3]);
        output.tessLevelInner[1] =
            max(output.tessLevelOuter[0], output.tessLevelOuter[2]);
    #endif
    #ifdef CASE32
        output.tessLevelOuter[0] = TessAdaptive(ev01, ev23, patchLevel) * 0.5;
        output.tessLevelOuter[1] = TessAdaptive(ev12, ev30, patchLevel) * 0.5;
        output.tessLevelOuter[2] = TessAdaptive(ev23, vv3, patchLevel+1);
        output.tessLevelOuter[3] = TessAdaptive(ev30, vv3, patchLevel+1);
        output.tessLevelInner[0] =
            max(output.tessLevelOuter[1], output.tessLevelOuter[3]);
        output.tessLevelInner[1] =
            max(output.tessLevelOuter[0], output.tessLevelOuter[2]);
    #endif
    #ifdef CASE33
        output.tessLevelOuter[0] = TessAdaptive(ev01, ev23, patchLevel) * 0.5;
        output.tessLevelOuter[1] = TessAdaptive(ev12, vv2, patchLevel+1);
        output.tessLevelOuter[2] = TessAdaptive(ev23, vv2, patchLevel+1);
        output.tessLevelOuter[3] = TessAdaptive(ev12, ev30, patchLevel) * 0.5;
        output.tessLevelInner[0] =
            max(output.tessLevelOuter[1], output.tessLevelOuter[3]);
        output.tessLevelInner[1] =
            max(output.tessLevelOuter[0], output.tessLevelOuter[2]);
    #endif


    #ifdef CASE40
        output.tessLevelOuter[0] = TessAdaptive(ev01, vv0, patchLevel+1);
        output.tessLevelOuter[1] = TessAdaptive(ev01, ev23, patchLevel);
        output.tessLevelOuter[2] = TessAdaptive(ev23, vv3, patchLevel+1);
        output.tessLevelOuter[3] = TessAdaptive(pv5, pv9, patchLevel);

        output.tessLevelInner[0] =
            max(output.tessLevelOuter[1], output.tessLevelOuter[3]);
        output.tessLevelInner[1] =
            max(output.tessLevelOuter[0], output.tessLevelOuter[2]);
    #endif
    #ifdef CASE41
        output.tessLevelOuter[0] = TessAdaptive(ev01, vv1, patchLevel+1);
        output.tessLevelOuter[1] = TessAdaptive(pv6, pv10, patchLevel);
        output.tessLevelOuter[2] = TessAdaptive(ev23, vv2, patchLevel+1);
        output.tessLevelOuter[3] = TessAdaptive(ev01, ev23, patchLevel);

        output.tessLevelInner[0] =
            max(output.tessLevelOuter[1], output.tessLevelOuter[3]);
        output.tessLevelInner[1] =
            max(output.tessLevelOuter[0], output.tessLevelOuter[2]);
    #endif
#else
    // XXX: HLSL compiler crashes with an internal compiler error occasionaly
    // if this shader accesses a shader resource buffer or a constant buffer
    // from this hull constant function.
    //float TessAmount = GetTessLevel(patchLevel);
    //float TessAmount = GetTessLevel(0);
    float TessAmount = 2.0;

    #ifdef CASE00
        float side = sqrt(1.25)*TessAmount;
        output.tessLevelOuter[0] = side;
        output.tessLevelOuter[1] = side;
        output.tessLevelOuter[2] = TessAmount;

        output.tessLevelInner    = TessAmount;
    #endif
    #ifdef CASE01
        float side =  sqrt(1.25)*TessAmount;
        output.tessLevelOuter[0] = TessAmount/2.0;
        output.tessLevelOuter[1] = TessAmount;
        output.tessLevelOuter[2] = side;

        output.tessLevelInner    = TessAmount/2.0;
    #endif
    #ifdef CASE02
        float side =  sqrt(1.25)*TessAmount;
        output.tessLevelOuter[0] = TessAmount/2.0;
        output.tessLevelOuter[1] = side;
        output.tessLevelOuter[2] = TessAmount;

        output.tessLevelInner    = TessAmount/2.0;
    #endif
    #ifdef CASE10 
        float side = sqrt(1.25) * TessAmount;
        output.tessLevelOuter[0] = TessAmount;
        output.tessLevelOuter[1] = side;
        output.tessLevelOuter[2] = TessAmount/2.0;

        output.tessLevelInner    = TessAmount/2;
    #endif
    #ifdef CASE11
        float side = sqrt(1.25) * TessAmount;
        output.tessLevelOuter[0] = TessAmount;
        output.tessLevelOuter[1] = TessAmount/2.0;
        output.tessLevelOuter[2] = side;

        output.tessLevelInner    = TessAmount/2;
    #endif
    #ifdef CASE12
        float side = sqrt(0.125) * TessAmount;
        output.tessLevelOuter[0] = TessAmount/2.0;
        output.tessLevelOuter[1] = TessAmount/2.0;
        output.tessLevelOuter[2] = side;

        output.tessLevelInner    = TessAmount/2;
    #endif
    #ifdef CASE13
        float side1 = sqrt(1.25) * TessAmount;
        float side2 = sqrt(0.125) * TessAmount;
        output.tessLevelOuter[0] = side1;
        output.tessLevelOuter[1] = side1;
        output.tessLevelOuter[2] = side2;

        output.tessLevelInner    = TessAmount/2.0*1.414;
    #endif


    #ifdef CASE20
        output.tessLevelOuter[0] = TessAmount;
        output.tessLevelOuter[1] = TessAmount/2.0;
        output.tessLevelOuter[2] = TessAmount;
        output.tessLevelOuter[3] = TessAmount/2.0;

        output.tessLevelInner[0] = TessAmount/2.0;
        output.tessLevelInner[1] = TessAmount;
    #endif
    #ifdef CASE21
        float side = sqrt(0.125) * TessAmount;
        output.tessLevelOuter[0] = side;
        output.tessLevelOuter[1] = TessAmount/2.0;
        output.tessLevelOuter[2] = TessAmount/2.0;

        output.tessLevelInner    = TessAmount/2.0;
    #endif
    #ifdef CASE22
        float side = sqrt(0.125) * TessAmount;
        output.tessLevelOuter[0] = TessAmount/2.0;
        output.tessLevelOuter[1] = TessAmount/2.0;
        output.tessLevelOuter[2] = side;

        output.tessLevelInner    = TessAmount/2.0;
    #endif
    #ifdef CASE23
        float side = sqrt(0.125) * TessAmount;
        output.tessLevelOuter[0] = TessAmount;
        output.tessLevelOuter[1] = side;
        output.tessLevelOuter[2] = side;

        output.tessLevelInner    = TessAmount/2.0;
    #endif


    #ifdef CASE30
        output.tessLevelOuter[0] = output.tessLevelOuter[1] =
        output.tessLevelOuter[2] = output.tessLevelOuter[3] = TessAmount/2.0;
        output.tessLevelInner[0] = output.tessLevelInner[1] = TessAmount/2.0;
    #endif
    #ifdef CASE31
        output.tessLevelOuter[0] = output.tessLevelOuter[1] =
        output.tessLevelOuter[2] = output.tessLevelOuter[3] = TessAmount/2.0;
        output.tessLevelInner[0] = output.tessLevelInner[1] = TessAmount/2.0;
    #endif
    #ifdef CASE32
        output.tessLevelOuter[0] = output.tessLevelOuter[1] =
        output.tessLevelOuter[2] = output.tessLevelOuter[3] = TessAmount/2.0;
        output.tessLevelInner[0] = output.tessLevelInner[1] = TessAmount/2.0;
    #endif
    #ifdef CASE33
        output.tessLevelOuter[0] = output.tessLevelOuter[1] =
        output.tessLevelOuter[2] = output.tessLevelOuter[3] = TessAmount/2.0;
        output.tessLevelInner[0] = output.tessLevelInner[1] = TessAmount/2.0;
    #endif


    #ifdef CASE40
        output.tessLevelOuter[0] = TessAmount/2.0;
        output.tessLevelOuter[1] = TessAmount;
        output.tessLevelOuter[2] = TessAmount/2.0;
        output.tessLevelOuter[3] = TessAmount;

        output.tessLevelInner[0] = TessAmount;
        output.tessLevelInner[1] = TessAmount/2.0;
    #endif
    #ifdef CASE41
        output.tessLevelOuter[0] = TessAmount/2.0;
        output.tessLevelOuter[1] = TessAmount;
        output.tessLevelOuter[2] = TessAmount/2.0;
        output.tessLevelOuter[3] = TessAmount;

        output.tessLevelInner[0] = TessAmount;
        output.tessLevelInner[1] = TessAmount/2.0;
    #endif
#endif

    return output;
}

[domain(HS_DOMAIN)]
[partitioning("integer")]
[outputtopology("triangle_cw")]
[outputcontrolpoints(16)]
[patchconstantfunc("HSConstFunc")]
HullVertex hs_main_patches(
    in InputPatch<HullVertex, PATCH_INPUT_SIZE> patch,
    uint primitiveID : SV_PrimitiveID,
    in uint ID : SV_OutputControlPointID )
{
#if defined BOUNDARY
    int i = ID/4;
    int j = ID%4;

#if defined(CASE20) || defined(CASE21) || defined(CASE22) || defined(CASE23)
#else
    i = 3 - i;
#endif

    float3 H[3];
    for (int l=0; l<3; l++) {
        H[l] = float3(0,0,0);
        for (int k=0; k<4; k++) {
            float c = Q[i][k];
            H[l] += c*patch[l*4 + k].position.xyz;
        }
    }

    float3 pos = float3(0,0,0);
    for (int k=0; k<3; k++) {
        pos += B[j][k]*H[k];
    }

#elif defined CORNER
    int i = ID/4;
    int j = ID%4;

    float3 H[3];
    for (int l=0; l<3; l++) {
        H[l] = float3(0,0,0);
        for (int k=0; k<3; k++) {
            float c = B[i][2-k];
            H[l] += c*patch[l*3 + k].position.xyz;
        }
    }

    float3 pos = float3(0,0,0);
    for (int k=0; k<3; k++) {
        pos += B[j][k]*H[k];
    }

#else
    int i = ID/4;
    int j = ID%4;

    float3 H[4];
    for (int l=0; l<4; ++l) {
        H[l] = float3(0,0,0);

        for(int k=0; k<4; ++k) {
            H[l] += Q[i][k] * patch[l*4 + k].position.xyz;
        }
    }

    float3 pos = float3(0,0,0);
    for (int k=0; k<4; ++k){
        pos += Q[j][k] * H[k];
    }

#endif

    HullVertex output;
    output.position = float4(pos, 1.0);

    int patchLevel = GetPatchLevel(primitiveID);
    // +0.5 to avoid interpolation error of integer value
    output.patchCoord = float4(0, 0,
                               patchLevel+0.5,
                               primitiveID+LevelBase+0.5);

    OSD_COMPUTE_PTEX_COORD_HULL_SHADER;

    return output;
}

//----------------------------------------------------------
// Patches.DomainTransition
//----------------------------------------------------------

// B-spline basis evaluation via deBoor pyramid...
void
EvalCubicBSpline(in float u, out float B[4], out float BU[4])
{
    float t = u;
    float s = 1.0 - u;

    float C0 =                     s * (0.5 * s);
    float C1 = t * (s + 0.5 * t) + s * (0.5 * s + t);
    float C2 = t * (    0.5 * t);

    B[0] =                                     1.f/3.f * s                * C0;
    B[1] = (2.f/3.f * s +           t) * C0 + (2.f/3.f * s + 1.f/3.f * t) * C1;
    B[2] = (1.f/3.f * s + 2.f/3.f * t) * C1 + (          s + 2.f/3.f * t) * C2;
    B[3] =                1.f/3.f * t  * C2;

    BU[0] =    - C0;
    BU[1] = C0 - C1;
    BU[2] = C1 - C2;
    BU[3] = C2;
}

void
Univar4x4(in float u, out float B[4], out float D[4])
{
    float t = u;
    float s = 1.0 - u;

    float A0 = s * s;
    float A1 = 2 * s * t;
    float A2 = t * t;

    B[0] = s * A0;
    B[1] = t * A0 + s * A1;
    B[2] = t * A1 + s * A2;
    B[3] = t * A2;

    D[0] =    - A0;
    D[1] = A0 - A1;
    D[2] = A1 - A2;
    D[3] = A2;
}

[domain(HS_DOMAIN)]
void ds_main_patches(
    in HS_CONSTANT_TRANSITION_FUNC_OUT input,
    in OutputPatch<HullVertex, 16> patch,
#ifdef TRIANGLE
    in float3 uvw : SV_DomainLocation,
#else
    in float2 uv : SV_DomainLocation,
#endif
    out OutputVertex output )
{
    float2 UV = float2(0.0, 0.0);

// XXXtakahito: Tess coordinates computed below are results of heuristic hack
//              to get front facing and appropriate patch uv.
//              Revisit here to get more consistent code with patch factory!

/*  CASE0*
    +-------+
    |1 /\\2 |
    | /  \\ |
    |/ 0  \\|
    +-------+
 */

#ifdef CASE00
    UV.x = 1.0-uvw.z;
    UV.y = 1.0-uvw.y-uvw.z/2.0;
#endif    
#ifdef CASE01
    UV.x = uvw.x;
    UV.y = 1.0 - uvw.y/2;
#endif    
#ifdef CASE02
    UV.x = uvw.x;
    UV.y = uvw.z/2;
#endif

/*  CASE1*
    +------+
    |1 /\\2|
    | /3_\\|
    |/_- 0 |
    +------+
*/

#ifdef CASE10
    UV.x = uvw.z;
    UV.y = 1.0-uvw.x/2.0;
#endif
#ifdef CASE11
    UV.x = 1.0-uvw.x/2.0;
    UV.y = uvw.y;
#endif
#ifdef CASE12
    UV.x = uvw.y/2.0;
    UV.y = uvw.x/2.0;
#endif
#ifdef CASE13
    UV.x = 1.0-uvw.y-uvw.x/2.0;
    UV.y = 1.0-uvw.x-uvw.y/2.0;
#endif

/*  CASE2*
    +-------+
    |   |\\2|
    |   | \\|
    | 0 |3/ |
    |   |/ 1|
    +-------+
 */

#ifdef CASE20
    UV.x = 0.5 - uv.x/2.0;
    UV.y = uv.y;
#endif
#ifdef CASE21
    UV.x = 1.0 - 0.5 *uvw.y;
    UV.y = 0.5*uvw.z;
#endif
#ifdef CASE22
    UV.x = 1.0 - uvw.y/2.0;
    UV.y = 1.0-uvw.x/2.0;
#endif
#ifdef CASE23
    UV.x = 1.0-0.5*uvw.y-0.5*uvw.z;
    UV.y = 1-uvw.y-0.5*uvw.x;
#endif

/*  CASE3*
    +-----+
    |2 |3 |
    |--+--+
    |0 |1 |
    +-----+
*/

#ifdef CASE30
    UV.x = 0.5 - uv.x/2.0;
    UV.y = uv.y/2.0;
#endif
#ifdef CASE31
    UV.x = 0.5 + uv.x/2.0;
    UV.y = 0.5 - uv.y/2.0;
#endif
#ifdef CASE32
    UV.x = uv.x/2.0;
    UV.y = 1.0 - uv.y/2.0;
#endif
#ifdef CASE33
    UV.x = 0.5 + uv.x/2.0;
    UV.y = 1.0 - uv.y/2.0;
#endif

/*  CASE4*
    +-----+
    | 1   |
    +-----+
    | 0   |
    +-----+
*/
#ifdef CASE40
    UV.x = uv.x;
    UV.y = 0.5 - uv.y/2.0;
#endif
#ifdef CASE41
    UV.x = uv.x;
    UV.y = 1.0 - uv.y/2.0;
#endif

    float B[4], D[4];

    Univar4x4(UV.x, B, D);

    float3 BUCP[4], DUCP[4];

    for (int i=0; i<4; ++i) {
        BUCP[i] = float3(0,0,0);
        DUCP[i] = float3(0,0,0);

        for (int j=0; j<4; ++j) {
#if ROTATE == 0
            float3 A = patch[4*i + j].position.xyz;
#elif ROTATE == 1
            float3 A = patch[4*(3-j) + (3-i)].position.xyz;
#elif ROTATE == 2
            float3 A = patch[4*i + (3-j)].position.xyz;
#elif ROTATE == 3
            float3 A = patch[4*j + i].position.xyz;
#endif
            BUCP[i] += A * B[j];
            DUCP[i] += A * D[j];
        }
    }

    float3 WorldPos  = float3(0,0,0);
    float3 Tangent   = float3(0,0,0);
    float3 BiTangent = float3(0,0,0);

    Univar4x4(UV.y, B, D);

    for (int i=0; i<4; ++i) {
        WorldPos  += B[i] * BUCP[i];
        Tangent   += B[i] * DUCP[i];
        BiTangent += D[i] * BUCP[i];
    }

    float3 normal = -normalize(cross(BiTangent, Tangent));

    normal = -normal;

    output.position = float4(WorldPos, 1.0f);
    output.normal = normal;
    output.tangent = normalize(BiTangent);

    output.patchCoord = patch[0].patchCoord;

#if ROTATE == 1
    output.patchCoord.xy = float2(UV.x, 1.0-UV.y);
#elif ROTATE == 2
    output.patchCoord.xy = float2(1.0-UV.y, 1.0-UV.x);
#elif ROTATE == 3
    output.patchCoord.xy = float2(1.0-UV.x, UV.y);
#else
    output.patchCoord.xy = float2(UV.y, UV.x);
#endif

    OSD_COMPUTE_PTEX_COORD_DOMAIN_SHADER;

#ifdef ROTATE
    OSD_COMPUTE_PTEX_COMPATIBLE_TANGENT(ROTATE);
#endif

    OSD_DISPLACEMENT_CALLBACK;

    output.positionOut = mul(ProjectionMatrix, float4(WorldPos, 1.0f));
}

//----------------------------------------------------------
// Patches.Vertex
//----------------------------------------------------------

void vs_main( in InputVertex input,
              out OutputVertex output)
{
    output.positionOut = mul(ModelViewProjectionMatrix, input.position);
}

//----------------------------------------------------------
// Patches.PixelColor
//----------------------------------------------------------

cbuffer Data : register( b2 ) {
    float4 color;
};

void ps_main( in OutputVertex input,
              out float4 colorOut : SV_Target )
{
    colorOut = color;
}
