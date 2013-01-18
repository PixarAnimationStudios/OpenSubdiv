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
// Patches.Common
//----------------------------------------------------------

#ifndef OSD_NUM_VARYINGS
#define OSD_NUM_VARYINGS 0
#endif

#define M_PI 3.14159265359f

struct ControlVertex {
    vec4 position;
    centroid vec4 patchCoord; // u, v, level, faceID
    ivec4 ptexInfo;  // U offset, V offset, 2^ptexlevel', rotation
    ivec3 clipFlag;
#if OSD_NUM_VARYINGS > 0
    float varyings[OSD_NUM_VARYINGS];
#endif
};

struct OutputVertex {
    vec4 position;
    vec3 normal;
    vec3 tangent;
    centroid vec4 patchCoord; // u, v, level, faceID
    noperspective vec4 edgeDistance;
#if OSD_NUM_VARYINGS > 0
    float varyings[OSD_NUM_VARYINGS];
#endif
};

struct GregControlVertex {
    vec3 position;
    vec3 hullPosition;
    ivec3 clipFlag;
    int  valence;
    vec3 e0;
    vec3 e1;
    uint zerothNeighbor;
    vec3 org;
#if OSD_MAX_VALENCE > 0
    vec3 r[OSD_MAX_VALENCE];
#endif
};

struct GregEvalVertex {
    vec3 position;
    vec3 Ep;
    vec3 Em;
    vec3 Fp;
    vec3 Fm;
    centroid vec4 patchCoord;
    ivec4 ptexInfo;
#if OSD_NUM_VARYINGS > 0
    float varyings[OSD_NUM_VARYINGS];
#endif
};

layout(std140) uniform Transform {
    mat4 ModelViewMatrix;
    mat4 ProjectionMatrix;
    mat4 ModelViewProjectionMatrix;
    mat4 ModelViewInverseMatrix;
};

layout(std140) uniform Tessellation {
    float TessLevel;
    int GregoryQuadOffsetBase;
    int LevelBase;
};

float GetTessLevel(int patchLevel)
{
#if OSD_ENABLE_SCREENSPACE_TESSELLATION
    return TessLevel;
#else
    return TessLevel / pow(2, patchLevel-1);
#endif
}

float GetPostProjectionSphereExtent(vec3 center, float diameter)
{
    vec4 p = ProjectionMatrix * vec4(center, 1.0);
    return abs(diameter * ProjectionMatrix[1][1] / p.w);
}

float TessAdaptive(vec3 p0, vec3 p1, int patchLevel)
{
    // Adaptive factor can be any computation that depends only on arg values.
    // Project the diameter of the edge's bounding sphere instead of using the
    // length of the projected edge itself to avoid problems near silhouettes.
    vec3 center = (p0 + p1) / 2.0;
    float diameter = distance(p0, p1);
    return max(1.0, TessLevel * GetPostProjectionSphereExtent(center, diameter));
}

#ifndef OSD_DISPLACEMENT_CALLBACK
#define OSD_DISPLACEMENT_CALLBACK
#endif

#ifdef USE_PTEX_COORD

#define OSD_DECLARE_PTEX_INDICES_BUFFER uniform isamplerBuffer g_ptexIndicesBuffer;

#define OSD_COMPUTE_PTEX_COORD_TESSCONTROL_SHADER                       \
    {                                                                   \
        ivec2 ptexIndex = texelFetchBuffer(g_ptexIndicesBuffer,         \
                                           gl_PrimitiveID + LevelBase).xy; \
        int lv = 1 << (patchLevel - int(ptexIndex.x & 1));              \
        int faceID = ptexIndex.x >> 3;                                  \
        int u = ptexIndex.y >> 16;                                      \
        int v = (ptexIndex.y & 0xffff);                                 \
        int rotation = (ptexIndex.x >> 1) & 0x3;                        \
        output[ID].v.patchCoord.w = faceID+0.5;                         \
        output[ID].v.ptexInfo = ivec4(u, v, lv, rotation);              \
    }

#define OSD_COMPUTE_PTEX_COORD_TESSEVAL_SHADER                          \
    {                                                                   \
        vec2 uv = output.v.patchCoord.xy;                               \
        ivec2 p = input[0].v.ptexInfo.xy;                               \
        int lv = input[0].v.ptexInfo.z;                                 \
        int rot = input[0].v.ptexInfo.w;                                \
        uv.xy = float(rot==0)*uv.xy                                     \
            + float(rot==1)*vec2(1.0-uv.y, uv.x)                        \
            + float(rot==2)*vec2(1.0-uv.x, 1.0-uv.y)                    \
            + float(rot==3)*vec2(uv.y, 1.0-uv.x);                       \
        output.v.patchCoord.xy = (uv * vec2(1.0)/lv) + vec2(p.x, p.y)/lv; \
    }

#define OSD_COMPUTE_PTEX_COMPATIBLE_TANGENT(ROTATE)             \
    {                                                           \
        int rot = (input[0].v.ptexInfo.w + 4 - ROTATE)%4;       \
        if (rot == 1) {                                         \
            output.v.tangent = -normalize(Tangent);             \
        } else if (rot == 2) {                                  \
            output.v.tangent = -normalize(BiTangent);           \
        } else if (rot == 3) {                                  \
            output.v.tangent = normalize(Tangent);              \
        } else {                                                \
            output.v.tangent = normalize(BiTangent);            \
        }                                                       \
    }

#else
#define OSD_DECLARE_PTEX_INDICES_BUFFER
#define OSD_COMPUTE_PTEX_COORD_TESSCONTROL_SHADER
#define OSD_COMPUTE_PTEX_COORD_TESSEVAL_SHADER
#define OSD_COMPUTE_PTEX_COMPATIBLE_TANGENT(ROTATE)
#endif  // USE_PTEX_COORD

#ifdef OSD_ENABLE_PATCH_CULL

#define OSD_PATCH_CULL_COMPUTE_CLIPFLAGS(P)                     \
    vec4 clipPos = ModelViewProjectionMatrix * P;               \
    bvec3 clip0 = lessThan(clipPos.xyz, vec3(clipPos.w));       \
    bvec3 clip1 = greaterThan(clipPos.xyz, -vec3(clipPos.w));   \
    output.v.clipFlag = ivec3(clip0) + 2*ivec3(clip1);          \

#define OSD_PATCH_CULL(N)                            \
    ivec3 clipFlag = ivec3(0);                       \
    for(int i = 0; i < N; ++i) {                     \
        clipFlag |= input[i].v.clipFlag;             \
    }                                                \
    if (clipFlag != ivec3(3) ) {                     \
        gl_TessLevelInner[0] = 0;                    \
        gl_TessLevelInner[1] = 0;                    \
        gl_TessLevelOuter[0] = 0;                    \
        gl_TessLevelOuter[1] = 0;                    \
        gl_TessLevelOuter[2] = 0;                    \
        gl_TessLevelOuter[3] = 0;                    \
        return;                                      \
    }

#else
#define OSD_PATCH_CULL_COMPUTE_CLIPFLAGS(P)
#define OSD_PATCH_CULL(N)
#endif

// ----------------------------------------------------------------------------

//----------------------------------------------------------
// Patches.Coefficients
//----------------------------------------------------------
// Regular
uniform mat4 Q = mat4(
    1.f/6.f, 2.f/3.f, 1.f/6.f, 0.f,
    0.f,     2.f/3.f, 1.f/3.f, 0.f,
    0.f,     1.f/3.f, 2.f/3.f, 0.f,
    0.f,     1.f/6.f, 2.f/3.f, 1.f/6.f
);

// Boundary
uniform mat4x3 B = mat4x3( 
    1.0f,    0.0f,    0.0f,
    2.f/3.f, 1.f/3.f, 0.0f,
    1.f/3.f, 2.f/3.f, 0.0f,
    1.f/6.f, 2.f/3.f, 1.f/6.f
);

// Corner
uniform mat4 R = mat4( 
    1.f/6.f, 2.f/3.f, 1.f/6.f, 0.0f,
    0.0f,    2.f/3.f, 1.f/3.f, 0.0f,
    0.0f,    1.f/3.f, 2.f/3.f, 0.0f,
    0.0f,    0.0f,    1.0f,    0.0f
);

#if OSD_MAX_VALENCE<=10
uniform float ef[7] = {
    0.813008, 0.500000, 0.363636, 0.287505,
    0.238692, 0.204549, 0.179211
};
#else
uniform float ef[27] = {
    0.812816, 0.500000, 0.363644, 0.287514,
    0.238688, 0.204544, 0.179229, 0.159657,
    0.144042, 0.131276, 0.120632, 0.111614,
    0.103872, 0.09715, 0.0912559, 0.0860444,
    0.0814022, 0.0772401, 0.0734867, 0.0700842,
    0.0669851, 0.0641504, 0.0615475, 0.0591488,
    0.0569311, 0.0548745, 0.0529621
};
#endif

float csf(uint n, uint j)
{
    if (j%2 == 0) {
        return cos((2.0f * M_PI * float(float(j-0)/2.0f))/(float(n)+3.0f));
    } else {
        return sin((2.0f * M_PI * float(float(j-1)/2.0f))/(float(n)+3.0f));
    }
}

void
Univar4x4(in float u, out float B[4], out float D[4])
{
    float t = u;
    float s = 1.0f - u;

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

vec4
EvalBSpline(vec2 uv, vec4 cp[16])
{
    float B[4], D[4];

    Univar4x4(uv.x, B, D);
    vec3 BUCP[4], DUCP[4];

    for (int i=0; i<4; ++i) {
        BUCP[i] = vec3(0.0f, 0.0f, 0.0f);
        DUCP[i] = vec3(0.0f, 0.0f, 0.0f);

        for (int j=0; j<4; ++j) {
/*
#if ROTATE == 1
            vec3 A = cp[4*(3-j) + (3-j)].xyz;
#elif ROTATE == 2
            vec3 A = cp[4*i + (3-j)].xyz;
#elif ROTATE == 3
            vec3 A = cp[4*j + i].xyz;
#else
            vec3 A = cp[4*i + j].xyz;
#endif
*/
            vec3 A = cp[4*i + j].xyz;

            BUCP[i] += A * B[j];
            DUCP[i] += A * D[j];
        }
    }
    vec3 val = vec3(0);

    Univar4x4(uv.y, B, D);

    for (int i=0; i<4; ++i) {
        val += B[i] * BUCP[i];
    }

    return vec4(val, 1);
}

void EvalBSpline(vec2 uv, vec3 cp[16],
                 out vec3 position,
                 out vec3 utangent,
                 out vec3 vtangent)
{
    float B[4], D[4];

    Univar4x4(uv.x, B, D);

    vec3 BUCP[4], DUCP[4];

    for (int i=0; i<4; ++i) {
        BUCP[i] = vec3(0);
        DUCP[i] = vec3(0);

        for (int j=0; j<4; ++j) {
#if ROTATE == 1
            vec3 A = cp[4*(3-j) + (3-i)];
#elif ROTATE == 2
            vec3 A = cp[4*i + (3-j)];
#elif ROTATE == 3
            vec3 A = cp[4*j + i];
#else
            vec3 A = cp[4*i + j];
#endif
            BUCP[i] += A * B[j];
            DUCP[i] += A * D[j];
        }
    }

    position = vec3(0);
    utangent = vec3(0);
    vtangent = vec3(0);

    Univar4x4(uv.y, B, D);

    for (int i=0; i<4; ++i) {
        position += B[i] * BUCP[i];
        utangent += B[i] * DUCP[i];
        vtangent += D[i] * BUCP[i];
    }
}

vec4 EvalGregory(vec2 uv, GregEvalVertex ev[4])
{
    float u = uv.x;
    float v = uv.y;
    vec3 p[20];

    p[0] = ev[0].position;
    p[1] = ev[0].Ep;
    p[2] = ev[0].Em;
    p[3] = ev[0].Fp;
    p[4] = ev[0].Fm;

    p[5] = ev[1].position;
    p[6] = ev[1].Ep;
    p[7] = ev[1].Em;
    p[8] = ev[1].Fp;
    p[9] = ev[1].Fm;

    p[10] = ev[2].position;
    p[11] = ev[2].Ep;
    p[12] = ev[2].Em;
    p[13] = ev[2].Fp;
    p[14] = ev[2].Fm;

    p[15] = ev[3].position;
    p[16] = ev[3].Ep;
    p[17] = ev[3].Em;
    p[18] = ev[3].Fp;
    p[19] = ev[3].Fm;

    vec3 q[16];

    float U = 1-u, V=1-v;

    float d11 = u+v; if(u+v==0.0f) d11 = 1.0f;
    float d12 = U+v; if(U+v==0.0f) d12 = 1.0f;
    float d21 = u+V; if(u+V==0.0f) d21 = 1.0f;
    float d22 = U+V; if(U+V==0.0f) d22 = 1.0f;

    q[ 5] = (u*p[3] + v*p[4])/d11;
    q[ 6] = (U*p[9] + v*p[8])/d12;
    q[ 9] = (u*p[19] + V*p[18])/d21;
    q[10] = (U*p[13] + V*p[14])/d22;

    q[ 0] = p[0];
    q[ 1] = p[1];
    q[ 2] = p[7];
    q[ 3] = p[5];
    q[ 4] = p[2];
    q[ 7] = p[6];
    q[ 8] = p[16];
    q[11] = p[12];
    q[12] = p[15];
    q[13] = p[17];
    q[14] = p[11];
    q[15] = p[10];

    float B[4], D[4];

    Univar4x4(u, B, D);
    vec3 BUCP[4], DUCP[4];

    for (int i=0; i<4; ++i) {
        BUCP[i] =  vec3(0, 0, 0);
        DUCP[i] =  vec3(0, 0, 0);

        for (uint j=0; j<4; ++j) {
            // reverse face front
            vec3 A = q[i + 4*j];

            BUCP[i] += A * B[j];
            DUCP[i] += A * D[j];
        }
    }

    vec3 WorldPos  = vec3(0);

    Univar4x4(v, B, D);

    for (uint i=0; i<4; ++i) {
        WorldPos  += B[i] * BUCP[i];
    }
    return vec4(WorldPos, 1);
}
