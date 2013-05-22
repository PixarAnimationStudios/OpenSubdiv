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

#extension GL_EXT_gpu_shader4 : require
#line 2

//----------------------------------------------------------
// Patches.Coefficients
//----------------------------------------------------------

#if defined(CASE00) || defined(CASE01) || defined(CASE02) || defined(CASE10) || defined(CASE11) || defined(CASE12) || defined(CASE13) || defined(CASE21) || defined(CASE22) || defined(CASE23)

    #define TRIANGLE

#else

    #undef TRIANGLE

#endif

//----------------------------------------------------------
// Patches.TessVertex
//----------------------------------------------------------
#ifdef PATCH_VERTEX_SHADER

layout (location=0) in vec4 position;

out block {
    ControlVertex v;
} output;

void main() {
    output.v.position = ModelViewMatrix * position;
    OSD_PATCH_CULL_COMPUTE_CLIPFLAGS(position);

#if OSD_NUM_VARYINGS > 0
    for (int i = 0; i < OSD_NUM_VARYINGS; ++i)
        output.v.varyings[i] = varyings[i];
#endif
}

#endif

//----------------------------------------------------------
// Patches.TessControlTransition
//----------------------------------------------------------
#ifdef PATCH_TESS_CONTROL_TRANSITION_SHADER

layout(vertices = 16) out;

in block {
    ControlVertex v;
} input[];

out block {
    ControlVertex v;
} output[];

#define ID gl_InvocationID

void main()
{
#if defined BOUNDARY
    int i = ID/4;
    int j = ID%4;

#if defined(CASE20) || defined(CASE21) || defined(CASE22) || defined(CASE23)
#else
    i = 3 - i;
#endif

    vec3 H[3];
    for (int l=0; l<3 ;l++) {
        H[l] = vec3(0,0,0);
        for (int k=0; k<4; k++) {
            float c = Q[i][k];
            H[l] += c*input[l*4 + k].v.position.xyz;
        }
    }

    vec3 pos = vec3(0,0,0);
    for (int k=0; k<3; k++) {
        pos += B[j][k]*H[k];
    }

#elif defined CORNER
    int i = ID/4;
    int j = ID%4;

    vec3 H[3];
    for (int l=0; l<3; l++) {
        H[l] = vec3(0,0,0);
        for (int k=0; k<3; k++) {
            float c = B[i][2-k];
            H[l] += c*input[l*3 + k].v.position.xyz;
        }
    }

    vec3 pos = vec3(0,0,0);
    for (int k=0; k<3; k++) {
        pos += B[j][k]*H[k];
    }

#else // not BOUNDARY, not CORNER
    int i = ID/4;
    int j = ID%4;

    vec3 H[4];
    for (int l=0; l<4; l++) {
        H[l] = vec3(0,0,0);
        for (int k=0; k<4; k++) {
            float c = Q[i][k];
            H[l] += c*input[l*4 + k].v.position.xyz;
        }
    }

    vec3 pos = vec3(0,0,0);
    for (int k=0; k<4; k++) {
        pos += Q[j][k]*H[k];
    }

#endif

    output[ID].v.position = vec4(pos, 1.0);

    int patchLevel = GetPatchLevel();
    output[ID].v.patchCoord = vec4(0, 0,
                                   patchLevel+0.5,
                                   gl_PrimitiveID+LevelBase+0.5);

    OSD_COMPUTE_PTEX_COORD_TESSCONTROL_SHADER;

    if (ID == 0) {
        OSD_PATCH_CULL(16);

#if OSD_ENABLE_SCREENSPACE_TESSELLATION
#line 1000
        // These tables map the 9, 12, or 16 input control points onto the
        // canonical 16 control points for a regular patch.
#if defined BOUNDARY
        const int p[16] = int[]( 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 );
#elif defined CORNER
        const int p[16] = int[]( 0, 1, 2, 2, 0, 1, 2, 2, 3, 4, 5, 5, 6, 7, 8, 8 );
#else
        const int p[16] = int[]( 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 );
#endif

#if ROTATE == 0
        const int r[16] = int[]( 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 );
#elif ROTATE == 1
        const int r[16] = int[]( 12, 8, 4, 0, 13, 9, 5, 1, 14, 10, 6, 2, 15, 11, 7, 3 );
#elif ROTATE == 2
        const int r[16] = int[]( 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 );
#elif ROTATE == 3
        const int r[16] = int[]( 3, 7, 11, 15, 2, 6, 10, 14, 1, 5, 9, 13, 0, 4, 8, 12 );
#endif

#line 2000
        // Expand and rotate control points using remapping tables above
        vec3 pv0 = input[p[r[0]]].v.position.xyz;
        vec3 pv1 = input[p[r[1]]].v.position.xyz;
        vec3 pv2 = input[p[r[2]]].v.position.xyz;
        vec3 pv3 = input[p[r[3]]].v.position.xyz;

        vec3 pv4 = input[p[r[4]]].v.position.xyz;
        vec3 pv5 = input[p[r[5]]].v.position.xyz;
        vec3 pv6 = input[p[r[6]]].v.position.xyz;
        vec3 pv7 = input[p[r[7]]].v.position.xyz;

        vec3 pv8 = input[p[r[8]]].v.position.xyz;
        vec3 pv9 = input[p[r[9]]].v.position.xyz;
        vec3 pv10 = input[p[r[10]]].v.position.xyz;
        vec3 pv11 = input[p[r[11]]].v.position.xyz;

        vec3 pv12 = input[p[r[12]]].v.position.xyz;
        vec3 pv13 = input[p[r[13]]].v.position.xyz;
        vec3 pv14 = input[p[r[14]]].v.position.xyz;
        vec3 pv15 = input[p[r[15]]].v.position.xyz;

        // Each edge of a transition patch is adjacent to one or two 
        // patches at the next refined level of subdivision.
        // Compute the corresponding vertex-vertex and edge-vertex refined
        // points along the edges of the patch using Catmull-Clark subdivision
        // stencil weights.
        // For simplicity, we let the optimizer discard unused computation.
        vec3 vv0 = (pv0 + pv2 + pv8 + pv10) * 0.015625 +
                     (pv1 + pv4 + pv6 + pv9) * 0.09375 + pv5 * 0.5625;
        vec3 ev01 = (pv1 + pv2 + pv9 + pv10) * 0.0625 + (pv5 + pv6) * 0.375;

        vec3 vv1 = (pv1 + pv3 + pv9 + pv11) * 0.015625 +
                     (pv2 + pv5 + pv7 + pv10) * 0.09375 + pv6 * 0.5625;
        vec3 ev12 = (pv5 + pv7 + pv9 + pv11) * 0.0625 + (pv6 + pv10) * 0.375;

        vec3 vv2 = (pv5 + pv7 + pv13 + pv15) * 0.015625 +
                     (pv6 + pv9 + pv11 + pv14) * 0.09375 + pv10 * 0.5625;
        vec3 ev23 = (pv5 + pv6 + pv13 + pv14) * 0.0625 + (pv9 + pv10) * 0.375;

        vec3 vv3 = (pv4 + pv6 + pv12 + pv14) * 0.015625 +
                     (pv5 + pv8 + pv10 + pv13) * 0.09375 + pv9 * 0.5625;
        vec3 ev30 = (pv4 + pv6 + pv8 + pv10) * 0.0625 + (pv5 + pv9) * 0.375;

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
        gl_TessLevelOuter[0] = TessAdaptive(ev01, pv9, patchLevel) * 0.5;
        gl_TessLevelOuter[1] = TessAdaptive(ev01, pv10, patchLevel) * 0.5;
        gl_TessLevelOuter[2] = TessAdaptive(pv9, pv10, patchLevel);

        gl_TessLevelInner[0] =
            (gl_TessLevelOuter[0] + gl_TessLevelOuter[1] + gl_TessLevelOuter[2]) * 0.5;
    #endif
    #ifdef CASE01
        gl_TessLevelOuter[0] = TessAdaptive(ev01, vv1, patchLevel+1);
        gl_TessLevelOuter[1] = TessAdaptive(pv6, pv10, patchLevel);
        gl_TessLevelOuter[2] = TessAdaptive(ev01, pv10, patchLevel) * 0.5;

        gl_TessLevelInner[0] =
            (gl_TessLevelOuter[0] + gl_TessLevelOuter[1] + gl_TessLevelOuter[2]) * 0.25;
    #endif
    #ifdef CASE02
        gl_TessLevelOuter[0] = TessAdaptive(ev01, vv0, patchLevel+1);
        gl_TessLevelOuter[1] = TessAdaptive(ev01, pv9, patchLevel) * 0.5;
        gl_TessLevelOuter[2] = TessAdaptive(pv5, pv9, patchLevel);

        gl_TessLevelInner[0] =
            (gl_TessLevelOuter[0] + gl_TessLevelOuter[1] + gl_TessLevelOuter[2]) * 0.25;
    #endif


    #ifdef CASE10 
        gl_TessLevelOuter[0] = TessAdaptive(pv6, pv10, patchLevel);
        gl_TessLevelOuter[1] = TessAdaptive(ev01, pv10, patchLevel);
        gl_TessLevelOuter[2] = TessAdaptive(ev01, vv1, patchLevel+1);

        gl_TessLevelInner[0] =
            (gl_TessLevelOuter[0] + gl_TessLevelOuter[1]) * 0.25;
    #endif
    #ifdef CASE11
        gl_TessLevelOuter[0] = TessAdaptive(pv9, pv10, patchLevel);
        gl_TessLevelOuter[1] = TessAdaptive(ev30, vv3, patchLevel+1);
        gl_TessLevelOuter[2] = TessAdaptive(ev30, pv10, patchLevel);

        gl_TessLevelInner[0] =
            (gl_TessLevelOuter[0] + gl_TessLevelOuter[2]) * 0.25;
    #endif
    #ifdef CASE12
        gl_TessLevelOuter[0] = TessAdaptive(ev30, vv0, patchLevel+1);
        gl_TessLevelOuter[1] = TessAdaptive(ev01, vv0, patchLevel+1);
        gl_TessLevelOuter[2] = TessAdaptive(ev01, ev30, patchLevel);

        gl_TessLevelInner[0] =
            (gl_TessLevelOuter[0] + gl_TessLevelOuter[1] + gl_TessLevelOuter[2]) * 0.25;
    #endif
    #ifdef CASE13
        gl_TessLevelOuter[0] = TessAdaptive(ev01, pv10, patchLevel);
        gl_TessLevelOuter[1] = TessAdaptive(ev30, pv10, patchLevel);
        gl_TessLevelOuter[2] = TessAdaptive(ev01, ev30, patchLevel);

        gl_TessLevelInner[0] =
            (gl_TessLevelOuter[0] + gl_TessLevelOuter[1] + gl_TessLevelOuter[2]) * 0.25;
    #endif


    #ifdef CASE20
        gl_TessLevelOuter[0] = TessAdaptive(ev12, ev30, patchLevel);
        gl_TessLevelOuter[1] = TessAdaptive(ev30, vv0, patchLevel+1);
        gl_TessLevelOuter[2] = TessAdaptive(pv5, pv6, patchLevel);
        gl_TessLevelOuter[3] = TessAdaptive(ev12, vv1, patchLevel+1);

        gl_TessLevelInner[0] =
            max(gl_TessLevelOuter[1], gl_TessLevelOuter[3]);
        gl_TessLevelInner[1] =
            max(gl_TessLevelOuter[0], gl_TessLevelOuter[2]);
    #endif
    #ifdef CASE21
        gl_TessLevelOuter[0] = TessAdaptive(ev23, ev30, patchLevel) * 0.5;
        gl_TessLevelOuter[1] = TessAdaptive(ev23, vv3, patchLevel+1);
        gl_TessLevelOuter[2] = TessAdaptive(ev30, vv3, patchLevel+1);

        gl_TessLevelInner[0] =
            (gl_TessLevelOuter[1] + gl_TessLevelOuter[2]) * 0.5;
    #endif
    #ifdef CASE22
        gl_TessLevelOuter[0] = TessAdaptive(ev12, vv2, patchLevel+1);
        gl_TessLevelOuter[1] = TessAdaptive(ev23, vv2, patchLevel+1);
        gl_TessLevelOuter[2] = TessAdaptive(ev12, ev23, patchLevel) * 0.5;

        gl_TessLevelInner[0] =
            (gl_TessLevelOuter[0] + gl_TessLevelOuter[1]) * 0.5;
    #endif
    #ifdef CASE23
        gl_TessLevelOuter[0] = TessAdaptive(ev12, ev30, patchLevel);
        gl_TessLevelOuter[1] = TessAdaptive(ev12, ev23, patchLevel) * 0.5;
        gl_TessLevelOuter[2] = TessAdaptive(ev23, ev30, patchLevel) * 0.5;

        gl_TessLevelInner[0] =
            (gl_TessLevelOuter[0] + gl_TessLevelOuter[1] + gl_TessLevelOuter[2]) * 0.5;
    #endif


    #ifdef CASE30
        gl_TessLevelOuter[0] = TessAdaptive(ev30, ev12, patchLevel) * 0.5;
        gl_TessLevelOuter[1] = TessAdaptive(ev30, vv0, patchLevel+1);
        gl_TessLevelOuter[2] = TessAdaptive(ev01, vv0, patchLevel+1);
        gl_TessLevelOuter[3] = TessAdaptive(ev01, ev23, patchLevel) * 0.5;
        gl_TessLevelInner[0] =
            max(gl_TessLevelOuter[1], gl_TessLevelOuter[3]);
        gl_TessLevelInner[1] =
            max(gl_TessLevelOuter[0], gl_TessLevelOuter[2]);
    #endif
    #ifdef CASE31
        gl_TessLevelOuter[0] = TessAdaptive(ev01, vv1, patchLevel+1);
        gl_TessLevelOuter[1] = TessAdaptive(ev12, vv1, patchLevel+1);
        gl_TessLevelOuter[2] = TessAdaptive(ev12, ev30, patchLevel) * 0.5;
        gl_TessLevelOuter[3] = TessAdaptive(ev01, ev23, patchLevel) * 0.5;
        gl_TessLevelInner[0] =
            max(gl_TessLevelOuter[1], gl_TessLevelOuter[3]);
        gl_TessLevelInner[1] =
            max(gl_TessLevelOuter[0], gl_TessLevelOuter[2]);
    #endif
    #ifdef CASE32
        gl_TessLevelOuter[0] = TessAdaptive(ev01, ev23, patchLevel) * 0.5;
        gl_TessLevelOuter[1] = TessAdaptive(ev12, ev30, patchLevel) * 0.5;
        gl_TessLevelOuter[2] = TessAdaptive(ev23, vv3, patchLevel+1);
        gl_TessLevelOuter[3] = TessAdaptive(ev30, vv3, patchLevel+1);
        gl_TessLevelInner[0] =
            max(gl_TessLevelOuter[1], gl_TessLevelOuter[3]);
        gl_TessLevelInner[1] =
            max(gl_TessLevelOuter[0], gl_TessLevelOuter[2]);
    #endif
    #ifdef CASE33
        gl_TessLevelOuter[0] = TessAdaptive(ev01, ev23, patchLevel) * 0.5;
        gl_TessLevelOuter[1] = TessAdaptive(ev12, vv2, patchLevel+1);
        gl_TessLevelOuter[2] = TessAdaptive(ev23, vv2, patchLevel+1);
        gl_TessLevelOuter[3] = TessAdaptive(ev12, ev30, patchLevel) * 0.5;
        gl_TessLevelInner[0] =
            max(gl_TessLevelOuter[1], gl_TessLevelOuter[3]);
        gl_TessLevelInner[1] =
            max(gl_TessLevelOuter[0], gl_TessLevelOuter[2]);
    #endif


    #ifdef CASE40
        gl_TessLevelOuter[0] = TessAdaptive(ev01, vv0, patchLevel+1);
        gl_TessLevelOuter[1] = TessAdaptive(ev01, ev23, patchLevel);
        gl_TessLevelOuter[2] = TessAdaptive(ev23, vv3, patchLevel+1);
        gl_TessLevelOuter[3] = TessAdaptive(pv5, pv9, patchLevel);

        gl_TessLevelInner[0] =
            max(gl_TessLevelOuter[1], gl_TessLevelOuter[3]);
        gl_TessLevelInner[1] =
            max(gl_TessLevelOuter[0], gl_TessLevelOuter[2]);
    #endif
    #ifdef CASE41
        gl_TessLevelOuter[0] = TessAdaptive(ev01, vv1, patchLevel+1);
        gl_TessLevelOuter[1] = TessAdaptive(pv6, pv10, patchLevel);
        gl_TessLevelOuter[2] = TessAdaptive(ev23, vv2, patchLevel+1);
        gl_TessLevelOuter[3] = TessAdaptive(ev01, ev23, patchLevel);

        gl_TessLevelInner[0] =
            max(gl_TessLevelOuter[1], gl_TessLevelOuter[3]);
        gl_TessLevelInner[1] =
            max(gl_TessLevelOuter[0], gl_TessLevelOuter[2]);
    #endif
#else
    float TessAmount = GetTessLevel(patchLevel);

    #ifdef CASE00
        float side = sqrt(1.25)*TessAmount;
        gl_TessLevelOuter[0] = side;
        gl_TessLevelOuter[1] = side;
        gl_TessLevelOuter[2] = TessAmount;

        gl_TessLevelInner[0] = TessAmount;
    #endif
    #ifdef CASE01
        float side =  sqrt(1.25)*TessAmount;
        gl_TessLevelOuter[0] = TessAmount/2.0;
        gl_TessLevelOuter[1] = TessAmount;
        gl_TessLevelOuter[2] = side;

        gl_TessLevelInner[0] = TessAmount/2.0;
    #endif
    #ifdef CASE02
        float side =  sqrt(1.25)*TessAmount;
        gl_TessLevelOuter[0] = TessAmount/2.0;
        gl_TessLevelOuter[1] = side;
        gl_TessLevelOuter[2] = TessAmount;

        gl_TessLevelInner[0] = TessAmount/2.0;
    #endif
    #ifdef CASE10 
        float side = sqrt(1.25) * TessAmount;
        gl_TessLevelOuter[0] = TessAmount;
        gl_TessLevelOuter[1] = side;
        gl_TessLevelOuter[2] = TessAmount/2.0;

        gl_TessLevelInner[0] = TessAmount/2;
    #endif
    #ifdef CASE11
        float side = sqrt(1.25) * TessAmount;
        gl_TessLevelOuter[0] = TessAmount;
        gl_TessLevelOuter[1] = TessAmount/2.0;
        gl_TessLevelOuter[2] = side;

        gl_TessLevelInner[0] = TessAmount/2;
    #endif
    #ifdef CASE12
        float side = sqrt(0.125) * TessAmount;
        gl_TessLevelOuter[0] = TessAmount/2.0;
        gl_TessLevelOuter[1] = TessAmount/2.0;
        gl_TessLevelOuter[2] = side;

        gl_TessLevelInner[0] = TessAmount/2;
    #endif
    #ifdef CASE13
        float side1 = sqrt(1.25) * TessAmount;
        float side2 = sqrt(0.125) * TessAmount;
        gl_TessLevelOuter[0] = side1;
        gl_TessLevelOuter[1] = side1;
        gl_TessLevelOuter[2] = side2;

        gl_TessLevelInner[0] = TessAmount/2.0*1.414;
    #endif


    #ifdef CASE20
        gl_TessLevelOuter[0] = TessAmount;
        gl_TessLevelOuter[1] = TessAmount/2.0;
        gl_TessLevelOuter[2] = TessAmount;
        gl_TessLevelOuter[3] = TessAmount/2.0;

        gl_TessLevelInner[0] = TessAmount/2.0;
        gl_TessLevelInner[1] = TessAmount;
    #endif
    #ifdef CASE21
        float side = sqrt(0.125) * TessAmount;
        gl_TessLevelOuter[0] = side;
        gl_TessLevelOuter[1] = TessAmount/2.0;
        gl_TessLevelOuter[2] = TessAmount/2.0;

        gl_TessLevelInner[0] = TessAmount/2.0;
    #endif
    #ifdef CASE22
        float side = sqrt(0.125) * TessAmount;
        gl_TessLevelOuter[0] = TessAmount/2.0;
        gl_TessLevelOuter[1] = TessAmount/2.0;
        gl_TessLevelOuter[2] = side;

        gl_TessLevelInner[0] = TessAmount/2.0;
    #endif
    #ifdef CASE23
        float side = sqrt(0.125) * TessAmount;
        gl_TessLevelOuter[0] = TessAmount;
        gl_TessLevelOuter[1] = side;
        gl_TessLevelOuter[2] = side;

        gl_TessLevelInner[0] = TessAmount/2.0;
    #endif


    #ifdef CASE30
        gl_TessLevelOuter[0] = gl_TessLevelOuter[1] =
        gl_TessLevelOuter[2] = gl_TessLevelOuter[3] = TessAmount/2.0;
        gl_TessLevelInner[0] = gl_TessLevelInner[1] = TessAmount/2.0;
    #endif
    #ifdef CASE31
        gl_TessLevelOuter[0] = gl_TessLevelOuter[1] =
        gl_TessLevelOuter[2] = gl_TessLevelOuter[3] = TessAmount/2.0;
        gl_TessLevelInner[0] = gl_TessLevelInner[1] = TessAmount/2.0;
    #endif
    #ifdef CASE32
        gl_TessLevelOuter[0] = gl_TessLevelOuter[1] =
        gl_TessLevelOuter[2] = gl_TessLevelOuter[3] = TessAmount/2.0;
        gl_TessLevelInner[0] = gl_TessLevelInner[1] = TessAmount/2.0;
    #endif
    #ifdef CASE33
        gl_TessLevelOuter[0] = gl_TessLevelOuter[1] =
        gl_TessLevelOuter[2] = gl_TessLevelOuter[3] = TessAmount/2.0;
        gl_TessLevelInner[0] = gl_TessLevelInner[1] = TessAmount/2.0;
    #endif


    #ifdef CASE40
        gl_TessLevelOuter[0] = TessAmount/2.0;
        gl_TessLevelOuter[1] = TessAmount;
        gl_TessLevelOuter[2] = TessAmount/2.0;
        gl_TessLevelOuter[3] = TessAmount;

        gl_TessLevelInner[0] = TessAmount;
        gl_TessLevelInner[1] = TessAmount/2.0;
    #endif
    #ifdef CASE41
        gl_TessLevelOuter[0] = TessAmount/2.0;
        gl_TessLevelOuter[1] = TessAmount;
        gl_TessLevelOuter[2] = TessAmount/2.0;
        gl_TessLevelOuter[3] = TessAmount;

        gl_TessLevelInner[0] = TessAmount;
        gl_TessLevelInner[1] = TessAmount/2.0;
    #endif
#endif
    }
}

#endif

//----------------------------------------------------------
// Patches.TessEvalTransition
//----------------------------------------------------------
#ifdef PATCH_TESS_EVAL_TRANSITION_SHADER

#ifdef TRIANGLE
    layout(triangles) in;
#else
    layout(quads) in;
#endif

in block {
    ControlVertex v;
} input[];

out block {
    OutputVertex v;
} output;

void main()
{
    vec2 UV = vec2(0.0, 0.0);
#ifdef TRIANGLE
    vec3 uvw = vec3(gl_TessCoord.x, gl_TessCoord.y, gl_TessCoord.z);
#else
    vec2 uv = vec2(gl_TessCoord.x, gl_TessCoord.y);
#endif

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

    vec3 WorldPos, Tangent, BiTangent;
    vec3 cp[16];
    for(int i = 0; i < 16; ++i) cp[i] = input[i].v.position.xyz;
    EvalBSpline(UV, cp, WorldPos, Tangent, BiTangent);

    vec3 normal = normalize(cross(BiTangent, Tangent));

    output.v.position = vec4(WorldPos, 1.0f);
    output.v.normal = normal;
    output.v.tangent = BiTangent;

    output.v.patchCoord = input[0].v.patchCoord;

#if ROTATE == 1
    output.v.patchCoord.xy = vec2(UV.x, 1.0-UV.y);
#elif ROTATE == 2
    output.v.patchCoord.xy = vec2(1.0-UV.y, 1.0-UV.x);
#elif ROTATE == 3
    output.v.patchCoord.xy = vec2(1.0-UV.x, UV.y);
#else
    output.v.patchCoord.xy = vec2(UV.y, UV.x);
#endif

    OSD_COMPUTE_PTEX_COORD_TESSEVAL_SHADER;

    OSD_COMPUTE_PTEX_COMPATIBLE_TANGENT(ROTATE);

    OSD_DISPLACEMENT_CALLBACK;

    gl_Position = (ProjectionMatrix * vec4(WorldPos, 1.0f));
}

#endif

//----------------------------------------------------------
// Patches.Vertex
//----------------------------------------------------------
#ifdef VERTEX_SHADER

layout (location=0) in vec4 position;
layout (location=1) in vec3 normal;
layout (location=2) in vec4 color;

out block {
    OutputVertex v;
} output;

void main() {
    gl_Position = ModelViewProjectionMatrix * position;
    output.v.color = color;
}

#endif

//----------------------------------------------------------
// Patches.FragmentColor
//----------------------------------------------------------
#ifdef FRAGMENT_SHADER

in block {
    OutputVertex v;
} input;

void main() {
    gl_FragColor = input.v.color;
}
#endif
