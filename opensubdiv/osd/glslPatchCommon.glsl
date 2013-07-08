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

//----------------------------------------------------------
// Patches.Common
//----------------------------------------------------------

#ifndef OSD_USER_VARYING_DECLARE
#define OSD_USER_VARYING_DECLARE
// type var;
#endif

#ifndef OSD_USER_VARYING_ATTRIBUTE_DECLARE
#define OSD_USER_VARYING_ATTRIBUTE_DECLARE
// layout(location = loc) in type var;
#endif

#ifndef OSD_USER_VARYING_PER_VERTEX
#define OSD_USER_VARYING_PER_VERTEX()
// output.var = var;
#endif

#ifndef OSD_USER_VARYING_PER_CONTROL_POINT
#define OSD_USER_VARYING_PER_CONTROL_POINT(ID_OUT, ID_IN)
// output[ID_OUT].var = input[ID_IN].var
#endif

#ifndef OSD_USER_VARYING_PER_EVAL_POINT
#define OSD_USER_VARYING_PER_EVAL_POINT(UV, a, b, c, d)
// output.var =
//     mix(mix(input[a].var, input[b].var, UV.x),
//         mix(input[c].var, input[d].var, UV.x), UV.y)
#endif

#ifndef OSD_TRANSITION_ROTATE
#define OSD_TRANSITION_ROTATE 0
#endif

#if defined OSD_PATCH_BOUNDARY
    #define OSD_PATCH_INPUT_SIZE 12
#elif defined OSD_PATCH_CORNER
    #define OSD_PATCH_INPUT_SIZE 9
#else
    #define OSD_PATCH_INPUT_SIZE 16
#endif

#define M_PI 3.14159265359f

struct ControlVertex {
    vec4 position;
    centroid vec4 patchCoord; // u, v, level, faceID
    ivec4 ptexInfo;  // U offset, V offset, 2^ptexlevel', rotation
    ivec3 clipFlag;
};

struct OutputVertex {
    vec4 position;
    vec3 normal;
    vec3 tangent;
    centroid vec4 patchCoord; // u, v, level, faceID
    centroid vec2 tessCoord; // tesscoord.st
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
};

layout(std140) uniform Transform {
    mat4 ModelViewMatrix;
    mat4 ProjectionMatrix;
    mat4 ModelViewProjectionMatrix;
    mat4 ModelViewInverseMatrix;
#ifdef OSD_USER_TRANSFORM_UNIFORMS
    OSD_USER_TRANSFORM_UNIFORMS
#endif
};

layout(std140) uniform Tessellation {
    float TessLevel;
};

//layout(std140) uniform PrimitiveBufferOffset {
uniform    int GregoryQuadOffsetBase;
uniform    int LevelBase;
//};

float GetTessLevel(int patchLevel)
{
#ifdef OSD_ENABLE_SCREENSPACE_TESSELLATION
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

float TessAdaptive(vec3 p0, vec3 p1)
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

// ----------------------------------------------------------------------------
// ptex coordinates
// ----------------------------------------------------------------------------

uniform isamplerBuffer g_ptexIndicesBuffer;

#define GetPatchLevel()                                                 \
        (texelFetch(g_ptexIndicesBuffer, gl_PrimitiveID + LevelBase).y & 0xf)

#define OSD_COMPUTE_PTEX_COORD_TESSCONTROL_SHADER                       \
    {                                                                   \
        ivec2 ptexIndex = texelFetch(g_ptexIndicesBuffer,               \
                                     gl_PrimitiveID + LevelBase).xy;    \
        int faceID = ptexIndex.x;                                       \
        int lv = 1 << ((ptexIndex.y & 0xf) - ((ptexIndex.y >> 4) & 1)); \
        int u = (ptexIndex.y >> 17) & 0x3ff;                            \
        int v = (ptexIndex.y >> 7) & 0x3ff;                             \
        int rotation = (ptexIndex.y >> 5) & 0x3;                        \
        outpt[ID].v.patchCoord.w = faceID+0.5;                          \
        outpt[ID].v.ptexInfo = ivec4(u, v, lv, rotation);               \
    }

#define OSD_COMPUTE_PTEX_COORD_TESSEVAL_SHADER                          \
    {                                                                   \
        vec2 uv = outpt.v.patchCoord.xy;                                \
        ivec2 p = inpt[0].v.ptexInfo.xy;                                \
        int lv = inpt[0].v.ptexInfo.z;                                  \
        int rot = inpt[0].v.ptexInfo.w;                                 \
        outpt.v.tessCoord.xy = uv;                                      \
        uv.xy = float(rot==0)*uv.xy                                     \
            + float(rot==1)*vec2(1.0-uv.y, uv.x)                        \
            + float(rot==2)*vec2(1.0-uv.x, 1.0-uv.y)                    \
            + float(rot==3)*vec2(uv.y, 1.0-uv.x);                       \
        outpt.v.patchCoord.xy = (uv * vec2(1.0)/lv) + vec2(p.x, p.y)/lv; \
    }

#define OSD_COMPUTE_PTEX_COMPATIBLE_TANGENT(ROTATE)             \
    {                                                           \
        int rot = (inpt[0].v.ptexInfo.w + 4 - ROTATE)%4;        \
        if (rot == 1) {                                         \
            outpt.v.tangent = -normalize(BiTangent);            \
        } else if (rot == 2) {                                  \
            outpt.v.tangent = -normalize(Tangent);              \
        } else if (rot == 3) {                                  \
            outpt.v.tangent = normalize(BiTangent);             \
        } else {                                                \
            outpt.v.tangent = normalize(Tangent);               \
        }                                                       \
    }

// ----------------------------------------------------------------------------
// face varyings
// ----------------------------------------------------------------------------

uniform samplerBuffer g_fvarDataBuffer;

#ifndef OSD_FVAR_WIDTH
#define OSD_FVAR_WIDTH 0
#endif

// XXX: quad only for now
#define OSD_COMPUTE_FACE_VARYING_1(result, fvarOffset, tessCoord)       \
    {                                                                   \
        float v[4];                                                     \
        int primOffset = (gl_PrimitiveID + LevelBase) * 4;              \
        for (int i = 0; i < 4; ++i) {                                   \
            int index = (primOffset+i)*OSD_FVAR_WIDTH + fvarOffset;     \
            v[i] = texelFetch(g_fvarDataBuffer, index).s                \
        }                                                               \
        result = mix(mix(v[0], v[1], tessCoord.s),                      \
                     mix(v[3], v[2], tessCoord.s),                      \
                     tessCoord.t);                                      \
    }

#define OSD_COMPUTE_FACE_VARYING_2(result, fvarOffset, tessCoord)       \
    {                                                                   \
        vec2 v[4];                                                      \
        int primOffset = (gl_PrimitiveID + LevelBase) * 4;              \
        for (int i = 0; i < 4; ++i) {                                   \
            int index = (primOffset+i)*OSD_FVAR_WIDTH + fvarOffset;     \
            v[i] = vec2(texelFetch(g_fvarDataBuffer, index).s,          \
                        texelFetch(g_fvarDataBuffer, index + 1).s);     \
        }                                                               \
        result = mix(mix(v[0], v[1], tessCoord.s),                      \
                     mix(v[3], v[2], tessCoord.s),                      \
                     tessCoord.t);                                      \
    }

#define OSD_COMPUTE_FACE_VARYING_3(result, fvarOffset, tessCoord)       \
    {                                                                   \
        vec3 v[4];                                                      \
        int primOffset = (gl_PrimitiveID + LevelBase) * 4;              \
        for (int i = 0; i < 4; ++i) {                                   \
            int index = (primOffset+i)*OSD_FVAR_WIDTH + fvarOffset;     \
            v[i] = vec3(texelFetch(g_fvarDataBuffer, index).s,          \
                        texelFetch(g_fvarDataBuffer, index + 1).s,      \
                        texelFetch(g_fvarDataBuffer, index + 2).s);     \
        }                                                               \
        result = mix(mix(v[0], v[1], tessCoord.s),                      \
                     mix(v[3], v[2], tessCoord.s),                      \
                     tessCoord.t);                                      \
    }

#define OSD_COMPUTE_FACE_VARYING_4(result, fvarOffset, tessCoord)       \
    {                                                                   \
        vec4 v[4];                                                      \
        int primOffset = (gl_PrimitiveID + LevelBase) * 4;              \
        for (int i = 0; i < 4; ++i) {                                   \
            int index = (primOffset+i)*OSD_FVAR_WIDTH + fvarOffset;     \
            v[i] = vec3(texelFetch(g_fvarDataBuffer, index).s,          \
                        texelFetch(g_fvarDataBuffer, index + 1).s,      \
                        texelFetch(g_fvarDataBuffer, index + 2).s,      \
                        texelFetch(g_fvarDataBuffer, index + 3).s);     \
        }                                                               \
        result = mix(mix(v[0], v[1], tessCoord.s),                      \
                     mix(v[3], v[2], tessCoord.s),                      \
                     tessCoord.t);                                      \
    }

// ----------------------------------------------------------------------------
// patch culling
// ----------------------------------------------------------------------------

#ifdef OSD_ENABLE_PATCH_CULL

#define OSD_PATCH_CULL_COMPUTE_CLIPFLAGS(P)                     \
    vec4 clipPos = ModelViewProjectionMatrix * P;               \
    bvec3 clip0 = lessThan(clipPos.xyz, vec3(clipPos.w));       \
    bvec3 clip1 = greaterThan(clipPos.xyz, -vec3(clipPos.w));   \
    outpt.v.clipFlag = ivec3(clip0) + 2*ivec3(clip1);           \

#define OSD_PATCH_CULL(N)                            \
    ivec3 clipFlag = ivec3(0);                       \
    for(int i = 0; i < N; ++i) {                     \
        clipFlag |= inpt[i].v.clipFlag;              \
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
