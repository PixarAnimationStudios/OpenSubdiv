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
// Patches.TessVertex
//----------------------------------------------------------
#ifdef PATCH_VERTEX_SHADER

layout (location=0) in vec4 position;

out block {
    ControlVertex v;
} oOutput;

void main() {
    oOutput.v.position = ModelViewMatrix * position;
    OSD_PATCH_CULL_COMPUTE_CLIPFLAGS(position);

#if OSD_NUM_VARYINGS > 0
    for (int i = 0; i < OSD_NUM_VARYINGS; ++i)
        oOutput.v.varyings[i] = varyings[i];
#endif
}

#endif

//----------------------------------------------------------
// Patches.TessControlRegular
//----------------------------------------------------------
#ifdef PATCH_TESS_CONTROL_REGULAR_SHADER

layout(vertices = 16) out;

in block {
    ControlVertex v;
} iInput[];

out block {
    ControlVertex v;
} oOutput[] ;

uniform isamplerBuffer g_patchLevelBuffer;
OSD_DECLARE_PTEX_INDICES_BUFFER

#define ID gl_InvocationID

void main()
{
    int i = ID/4;
    int j = ID%4;

    vec3 H[4];
    for (int l=0; l<4; l++) {
        H[l] = vec3(0,0,0);
        for (int k=0; k<4; k++) {
            float c = Q[i][k];
            // XXX: fix this in patchMeshFactory.
//            H[l] += c*iInput[l*4 + k].v.position.xyz;
            H[l] += c*iInput[l + k*4].v.position.xyz;
        }
    }

    vec3 pos = vec3(0,0,0);
    for (int k=0; k<4; k++) {
        pos += Q[j][k]*H[k];
    }

    oOutput[ID].v.position = vec4(pos, 1.0);

    int patchLevel = texelFetchBuffer(g_patchLevelBuffer, gl_PrimitiveID + LevelBase).x;
    // +0.5 to avoid interpolation error of integer value
    oOutput[ID].v.patchCoord = vec4(0, 0,
                                   patchLevel+0.5,
                                   gl_PrimitiveID+LevelBase+0.5);

    OSD_COMPUTE_PTEX_COORD_TESSCONTROL_SHADER;

    if (ID == 0) {
        OSD_PATCH_CULL(16);

#if OSD_ENABLE_SCREENSPACE_TESSELLATION
        gl_TessLevelOuter[0] =
            TessAdaptive(iInput[5].v.position.xyz, iInput[9].v.position.xyz, patchLevel);
        gl_TessLevelOuter[1] =
            TessAdaptive(iInput[5].v.position.xyz, iInput[6].v.position.xyz, patchLevel);
        gl_TessLevelOuter[2] =
            TessAdaptive(iInput[6].v.position.xyz, iInput[10].v.position.xyz, patchLevel);
        gl_TessLevelOuter[3] =
            TessAdaptive(iInput[9].v.position.xyz, iInput[10].v.position.xyz, patchLevel);
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
// Patches.TessEvalRegular
//----------------------------------------------------------
#ifdef PATCH_TESS_EVAL_REGULAR_SHADER

layout(quads) in;
layout(equal_spacing) in;

in block {
    ControlVertex v;
} iInput[];

out block {
    OutputVertex v;
} oOutput;

void main()
{
    float u = gl_TessCoord.x,
          v = gl_TessCoord.y;

    vec3 WorldPos, Tangent, BiTangent;
    vec3 cp[16];
    for(int i = 0; i < 16; ++i) cp[i] = iInput[i].v.position.xyz;
    EvalBSpline(gl_TessCoord.xy, cp, WorldPos, Tangent, BiTangent);

    vec3 normal = normalize(cross(Tangent, BiTangent));

    oOutput.v.position = vec4(WorldPos, 1.0f);
    oOutput.v.normal = normal;
    oOutput.v.tangent = Tangent;

    oOutput.v.patchCoord = iInput[0].v.patchCoord;
    oOutput.v.patchCoord.xy = vec2(u, v);

    OSD_COMPUTE_PTEX_COORD_TESSEVAL_SHADER;

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
} oOutput;

void main() {
    gl_Position = ModelViewProjectionMatrix * position;
    oOutput.v.color = color;
}

#endif

//----------------------------------------------------------
// Patches.FragmentColor
//----------------------------------------------------------
#ifdef FRAGMENT_SHADER

in block {
    OutputVertex v;
} iInput;

void main() {
    gl_FragColor = iInput.v.color;
}
#endif
