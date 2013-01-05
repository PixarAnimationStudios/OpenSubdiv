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
// Patches.Prologue
//----------------------------------------------------------

#ifndef OSD_NUM_VARYINGS
#define OSD_NUM_VARYINGS 0
#endif

#define M_PI 3.14159265359f

struct InputVertex {
    float4 position : POSITION;
    float3 normal : NORMAL;
};

struct HullVertex {
    float4 position : POSITION;
    float4 patchCoord : PATCHCOORD; // u, v, level, faceID
    int4 ptexInfo : PTEXINFO;       // u offset, v offset, 2^ptexlevel, rotation
    int3 clipFlag : CLIPFLAG;
#if OSD_NUM_VARYINGS > 0
    float varyings[OSD_NUM_VARYINGS] : VARYING;
#endif
};

struct OutputVertex {
    float4 positionOut : SV_Position;
    float4 position : POSITION1;
    float3 normal : NORMAL;
    float3 tangent : TANGENT;
    float4 patchCoord : PATCHCOORD; // u, v, level, faceID
    noperspective float4 edgeDistance : EDGEDISTANCE;
#if OSD_NUM_VARYINGS > 0
    float varyings[OSD_NUM_VARYINGS] : VARYING;
#endif
};

struct GregHullVertex {
    float3 position : POSITION0;
    float3 hullPosition : HULLPOSITION;
    int3 clipFlag : CLIPFLAG;
    int valence : BLENDINDICE0;
    float3 e0 : POSITION1;
    float3 e1 : POSITION2;
    uint zerothNeighbor : BLENDINDICE1;
    float3 org : POSITION3;
#if OSD_MAX_VALENCE > 0
    float3 r[OSD_MAX_VALENCE] : POSITION4;
#endif
};

struct GregDomainVertex {
    float3 position : POSITION0;
    float3 Ep : POSITION1;
    float3 Em : POSITION2;
    float3 Fp : POSITION3;
    float3 Fm : POSITION4;
    float4 patchCoord: TEXTURE0;
    float4 ptexInfo: TEXTURE1;
#if OSD_NUM_VARYINGS > 0
    float varyings[OSD_NUM_VARYINGS];
#endif
};

struct HS_CONSTANT_FUNC_OUT {
    float tessLevelInner[2] : SV_InsideTessFactor;
    float tessLevelOuter[4] : SV_TessFactor;
};

cbuffer Transform : register( b0 ) {
    float4x4 ModelViewMatrix;
    float4x4 ProjectionMatrix;
    float4x4 ModelViewProjectionMatrix;
};

cbuffer Tessellation : register( b1 ) {
    float TessLevel;
    int GregoryQuadOffsetBase;
    int LevelBase;
};

float GetTessLevel(int patchLevel) {
#if OSD_ENABLE_SCREENSPACE_TESSELLATION
    return TessLevel;
#else
    return TessLevel / pow(2, patchLevel-1);
#endif
}

float GetPostProjectionSphereExtent(float3 center, float diameter)
{
    float4 p = mul(ProjectionMatrix, float4(center, 1.0));
    return abs(diameter * ProjectionMatrix[1][1] / p.w);
}

float TessAdaptive(float3 p0, float3 p1, int patchLevel)
{
    // Adaptive factor can be any computation that depends only on arg values.
    // Project the diameter of the edge's bounding sphere instead of using the
    // length of the projected edge itself to avoid problems near silhouettes.
    float3 center = (p0 + p1) / 2.0;
    float diameter = distance(p0, p1);
    return max(1.0, TessLevel * GetPostProjectionSphereExtent(center, diameter));
}

#ifndef OSD_DISPLACEMENT_CALLBACK
#define OSD_DISPLACEMENT_CALLBACK
#endif

#ifdef USE_PTEX_COORD

#define OSD_DECLARE_PTEX_INDICES_BUFFER Buffer<int2> g_ptexIndicesBuffer : register( t4 );

#define OSD_COMPUTE_PTEX_COORD_HULL_SHADER                              \
    {                                                                   \
        int2 ptexIndex = g_ptexIndicesBuffer[ID + LevelBase].xy;        \
        int lv = 1 << (patchLevel - int(ptexIndex.x & 1));              \
        int faceID = ptexIndex.x >> 3;                                  \
        int u = ptexIndex.y >> 16;                                      \
        int v = (ptexIndex.y & 0xffff);                                 \
        int rotation = (ptexIndex.x >> 1) & 0x3;                        \
        output.patchCoord.w = faceID+0.5;                               \
        output.ptexInfo = int4(u, v, lv, rotation);                     \
    }

#define OSD_COMPUTE_PTEX_COORD_DOMAIN_SHADER                            \
    {                                                                   \
        float2 uv = output.patchCoord.xy;                               \
        int2 p = patch[0].ptexInfo.xy;                                  \
        int lv = patch[0].ptexInfo.z;                                   \
        int rot = patch[0].ptexInfo.w;                                  \
        uv.xy = float(rot==0)*uv.xy                                     \
            + float(rot==1)*float2(1.0-uv.y, uv.x)                      \
            + float(rot==2)*float2(1.0-uv.x, 1.0-uv.y)                  \
            + float(rot==3)*float2(uv.y, 1.0-uv.x);                     \
        output.patchCoord.xy = (uv * float2(1.0,1.0)/lv) + float2(p.x, p.y)/lv; \
    }

#define OSD_COMPUTE_PTEX_COMPATIBLE_TANGENT(ROTATE)             \
    {                                                           \
        int rot = (patch[0].ptexInfo.w + 4 - ROTATE)%4;         \
        if (rot == 1) {                                         \
            output.tangent = -normalize(Tangent);               \
        } else if (rot == 2) {                                  \
            output.tangent = -normalize(BiTangent);             \
        } else if (rot == 3) {                                  \
            output.tangent = normalize(Tangent);                \
        } else {                                                \
            output.tangent = normalize(BiTangent);              \
        }                                                       \
    }

#else
#define OSD_DECLARE_PTEX_INDICES_BUFFER
#define OSD_COMPUTE_PTEX_COORD_HULL_SHADER
#define OSD_COMPUTE_PTEX_COORD_DOMAIN_SHADER
#define OSD_COMPUTE_PTEX_COMPATIBLE_TANGENT(ROTATE)
#endif  // USE_PTEX_COORD

#ifdef OSD_ENABLE_PATCH_CULL

#define OSD_PATCH_CULL_COMPUTE_CLIPFLAGS(P)                     \
    float4 clipPos = mul(ModelViewProjectionMatrix, P);         \
    int3 clip0 = int3(clipPos.x < clipPos.w,                    \
                      clipPos.y < clipPos.w,                    \
                      clipPos.z < clipPos.w);                   \
    int3 clip1 = int3(clipPos.x > -clipPos.w,                   \
                      clipPos.y > -clipPos.w,                   \
                      clipPos.z > -clipPos.w);                  \
    output.clipFlag = int3(clip0) + 2*int3(clip1);              \

#define OSD_PATCH_CULL(N)                          \
    int3 clipFlag = int3(0,0,0);                   \
    for(int i = 0; i < N; ++i) {                   \
        clipFlag |= patch[i].clipFlag;             \
    }                                              \
    if (any(clipFlag != int3(3,3,3))) {            \
        output.tessLevelInner[0] = 0;              \
        output.tessLevelInner[1] = 0;              \
        output.tessLevelOuter[0] = 0;              \
        output.tessLevelOuter[1] = 0;              \
        output.tessLevelOuter[2] = 0;              \
        output.tessLevelOuter[3] = 0;              \
        return output;                             \
    }

#define OSD_PATCH_CULL_TRIANGLE(N)                 \
    int3 clipFlag = int3(0,0,0);                   \
    for(int i = 0; i < N; ++i) {                   \
        clipFlag |= patch[i].clipFlag;             \
    }                                              \
    if (any(clipFlag != int3(3,3,3))) {            \
        output.tessLevelInner    = 0;              \
        output.tessLevelOuter[0] = 0;              \
        output.tessLevelOuter[1] = 0;              \
        output.tessLevelOuter[2] = 0;              \
        return output;                             \
    }

#else
#define OSD_PATCH_CULL_COMPUTE_CLIPFLAGS(P)
#define OSD_PATCH_CULL(N)
#define OSD_PATCH_CULL_TRIANGLE(N)
#endif

