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
// Patches.TessControlBoundary
//----------------------------------------------------------
#ifdef PATCH_TESS_CONTROL_BOUNDARY_SHADER

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
    int i = ID/4;
    int j = ID%4;

    i = 3 - i;

    vec3 H[3];
    for (int l=0; l<3; l++) {
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

    output[ID].v.position = vec4(pos, 1.0);

    int patchLevel = GetPatchLevel();
    // +0.5 to avoid interpolation error of integer value
    output[ID].v.patchCoord = vec4(0, 0,
                                   patchLevel+0.5,
                                   gl_PrimitiveID+LevelBase+0.5);

    OSD_COMPUTE_PTEX_COORD_TESSCONTROL_SHADER;

    if (ID == 0) {
        OSD_PATCH_CULL(12);

#if OSD_ENABLE_SCREENSPACE_TESSELLATION
        gl_TessLevelOuter[0] =
            TessAdaptive(input[1].v.position.xyz, input[2].v.position.xyz, patchLevel);
        gl_TessLevelOuter[1] =
            TessAdaptive(input[2].v.position.xyz, input[6].v.position.xyz, patchLevel);
        gl_TessLevelOuter[2] =
            TessAdaptive(input[5].v.position.xyz, input[6].v.position.xyz, patchLevel);
        gl_TessLevelOuter[3] =
            TessAdaptive(input[1].v.position.xyz, input[5].v.position.xyz, patchLevel);
        gl_TessLevelInner[0] =
            max(gl_TessLevelOuter[1], gl_TessLevelOuter[3]);
        gl_TessLevelInner[1] =
            max(gl_TessLevelOuter[0], gl_TessLevelOuter[2]);
#else
        gl_TessLevelInner[0] = GetTessLevel(patchLevel);
        gl_TessLevelInner[1] = GetTessLevel(patchLevel);
        gl_TessLevelOuter[0] = GetTessLevel(patchLevel);
        gl_TessLevelOuter[1] = GetTessLevel(patchLevel);
        gl_TessLevelOuter[2] = GetTessLevel(patchLevel);
        gl_TessLevelOuter[3] = GetTessLevel(patchLevel);
#endif
    }
}

#endif

//----------------------------------------------------------
// Patches.TessEvalBoundary
//----------------------------------------------------------
#ifdef PATCH_TESS_EVAL_BOUNDARY_SHADER

layout(quads) in;
layout(equal_spacing) in;

in block {
    ControlVertex v;
    int clipFlag;
} input[];

out block {
    OutputVertex v;
} output;

void main()
{
    float u = gl_TessCoord.x,
          v = gl_TessCoord.y;

/*
    float B[4], D[4];

    Univar4x4(u, B, D);

    vec3 BUCP[4], DUCP[4];

    for (int i=0; i<4; ++i) {
        BUCP[i] = vec3(0.0f, 0.0f, 0.0f);
        DUCP[i] = vec3(0.0f, 0.0f, 0.0f);

        for (int j=0; j<4; ++j) {
            vec3 A = input[4*i + j].v.position.xyz;

            BUCP[i] += A * B[j];
            DUCP[i] += A * D[j];
        }
    }

    vec3 WorldPos  = vec3(0.0f, 0.0f, 0.0f);
    vec3 Tangent   = vec3(0.0f, 0.0f, 0.0f);
    vec3 BiTangent = vec3(0.0f, 0.0f, 0.0f);

    Univar4x4(v, B, D);

    for (int i=0; i<4; ++i) {
        WorldPos  += B[i] * BUCP[i];
        Tangent   += B[i] * DUCP[i];
        BiTangent += D[i] * BUCP[i];
    }
*/
    vec3 WorldPos, Tangent, BiTangent;
    vec3 cp[16];
    for(int i = 0; i < 16; ++i) cp[i] = input[i].v.position.xyz;
    EvalBSpline(gl_TessCoord.xy, cp, WorldPos, Tangent, BiTangent);

    vec3 normal = normalize(cross(Tangent, BiTangent));

    output.v.position = vec4(WorldPos, 1.0f);
    output.v.normal = normal;

    BiTangent = -BiTangent;  // BiTangent will be used in following macro
    output.v.tangent = BiTangent;

    output.v.patchCoord = input[0].v.patchCoord;
    output.v.patchCoord.xy = vec2(1.0-v, u);

    OSD_COMPUTE_PTEX_COORD_TESSEVAL_SHADER;

    OSD_COMPUTE_PTEX_COMPATIBLE_TANGENT(0);

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
