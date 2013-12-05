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
// Patches.Common
//----------------------------------------------------------

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

struct InputVertex {
    float4 position : POSITION;
    float3 normal : NORMAL;
};

struct HullVertex {
    float4 position : POSITION;
    float4 patchCoord : PATCHCOORD; // u, v, level, faceID
    int4 ptexInfo : PTEXINFO;       // u offset, v offset, 2^ptexlevel, rotation
    int3 clipFlag : CLIPFLAG;
};

struct OutputVertex {
    float4 positionOut : SV_Position;
    float4 position : POSITION1;
    float3 normal : NORMAL;
    float3 tangent : TANGENT0;
    float3 bitangent : TANGENT1;
    float4 patchCoord : PATCHCOORD; // u, v, level, faceID
    noperspective float4 edgeDistance : EDGEDISTANCE;
#if defined(OSD_COMPUTE_NORMAL_DERIVATIVES)
    float3 Nu : TANGENT2;
    float3 Nv : TANGENT3;
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
};

struct HS_CONSTANT_FUNC_OUT {
    float tessLevelInner[2] : SV_InsideTessFactor;
    float tessLevelOuter[4] : SV_TessFactor;
};

// osd shaders need following functions defined
float4x4 OsdModelViewMatrix();
float4x4 OsdProjectionMatrix();
float4x4 OsdModelViewProjectionMatrix();
float OsdTessLevel();
int OsdGregoryQuadOffsetBase();
int OsdPrimitiveIdBase();

float GetTessLevel(int patchLevel)
{
#ifdef OSD_ENABLE_SCREENSPACE_TESSELLATION
    return OsdTessLevel();
#else
    return OsdTessLevel() / pow(2, patchLevel-1);
#endif
}

#ifndef GetPrimitiveID
#define GetPrimitiveID(x) (x + OsdPrimitiveIdBase())
#endif

float GetPostProjectionSphereExtent(float3 center, float diameter)
{
    float4 p = mul(OsdProjectionMatrix(), float4(center, 1.0));
    return abs(diameter * OsdProjectionMatrix()[1][1] / p.w);
}

float TessAdaptive(float3 p0, float3 p1)
{
    // Adaptive factor can be any computation that depends only on arg values.
    // Project the diameter of the edge's bounding sphere instead of using the
    // length of the projected edge itself to avoid problems near silhouettes.
    float3 center = (p0 + p1) / 2.0;
    float diameter = distance(p0, p1);
    return max(1.0, OsdTessLevel() * GetPostProjectionSphereExtent(center, diameter));
}

#ifndef OSD_DISPLACEMENT_CALLBACK
#define OSD_DISPLACEMENT_CALLBACK
#endif

Buffer<int2> OsdPatchParamBuffer : register( t3 );

#define GetPatchLevel(primitiveID)                                      \
    (OsdPatchParamBuffer[GetPrimitiveID(primitiveID)].y & 0xf)

#define OSD_COMPUTE_PTEX_COORD_HULL_SHADER                              \
    {                                                                   \
        int2 ptexIndex = OsdPatchParamBuffer[GetPrimitiveID(primitiveID)].xy; \
        int faceID = ptexIndex.x;                                       \
        int lv = 1 << ((ptexIndex.y & 0xf) - ((ptexIndex.y >> 4) & 1)); \
        int u = (ptexIndex.y >> 17) & 0x3ff;                            \
        int v = (ptexIndex.y >> 7) & 0x3ff;                             \
        int rotation = (ptexIndex.y >> 5) & 0x3;                        \
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

#define OSD_COMPUTE_PTEX_COMPATIBLE_TANGENT(ROTATE)                 \
    {                                                               \
        int rot = (patch[0].ptexInfo.w + 4 - ROTATE)%4;             \
        if (rot == 1) {                                             \
            output.tangent = -BiTangent;                            \
            output.bitangent = Tangent;                             \
        } else if (rot == 2) {                                      \
            output.tangent = -Tangent;                              \
            output.bitangent = -BiTangent;                          \
        } else if (rot == 3) {                                      \
            output.tangent = BiTangent;                             \
            output.bitangent = -Tangent;                            \
        } else {                                                    \
            output.tangent = Tangent;                               \
            output.bitangent = BiTangent;                           \
        }                                                           \
    }

#define OSD_COMPUTE_PTEX_COMPATIBLE_DERIVATIVES(ROTATE)             \
    {                                                               \
        int rot = (patch[0].ptexInfo.w + 4 - ROTATE)%4;             \
        if (rot == 1) {                                             \
            output.tangent = -BiTangent;                            \
            output.bitangent = Tangent;                             \
            output.Nu = -Nv;                                        \
            output.Nv = Nv;                                         \
        } else if (rot == 2) {                                      \
            output.tangent = -Tangent;                              \
            output.bitangent = -BiTangent;                          \
            output.Nu = -Nu;                                        \
            output.Nv = -Nv;                                        \
        } else if (rot == 3) {                                      \
            output.tangent = BiTangent;                             \
            output.bitangent = -Tangent;                            \
            output.Nu = Nv;                                         \
            output.Nv = -Nu;                                        \
        } else {                                                    \
            output.tangent = Tangent;                               \
            output.bitangent = BiTangent;                           \
            output.Nu = Nu;                                         \
            output.Nv = Nv;                                         \
        }                                                           \
    }


#ifdef OSD_ENABLE_PATCH_CULL

#define OSD_PATCH_CULL_COMPUTE_CLIPFLAGS(P)                     \
    float4 clipPos = mul(OsdModelViewProjectionMatrix(), P);    \
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

void Univar4x4(in float u, out float B[4], out float D[4])
{
    float t = u;
    float s = 1.0f - u;

    float A0 =     s * s;
    float A1 = 2 * s * t;
    float A2 = t * t;

    B[0] =          s * A0;
    B[1] = t * A0 + s * A1;
    B[2] = t * A1 + s * A2;
    B[3] = t * A2;

    D[0] =    - A0;
    D[1] = A0 - A1;
    D[2] = A1 - A2;
    D[3] = A2;
}

void
Univar4x4(in float u, out float B[4], out float D[4], out float C[4])
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

    A0 =   - s;
    A1 = s - t;
    A2 = t;

    C[0] =    - A0;
    C[1] = A0 - A1;
    C[2] = A1 - A2;
    C[3] = A2;
}
