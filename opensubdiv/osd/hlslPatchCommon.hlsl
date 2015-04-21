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

// XXXdyu-patch-drawing support for fractional spacing
#undef OSD_FRACTIONAL_ODD_SPACING
#undef OSD_FRACTIONAL_EVEN_SPACING

#define OSD_PATCH_INPUT_SIZE 16

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
#if defined OSD_PATCH_ENABLE_SINGLE_CREASE
    float4 P1 : POSITION1;
    float4 P2 : POSITION2;
    float sharpness : BLENDWEIGHT0;
#endif
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
#if defined OSD_PATCH_ENABLE_SINGLE_CREASE
    float sharpness : BLENDWEIGHT0;
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
    float4 tessOuterLo : TRANSITIONLO;
    float4 tessOuterHi : TRANSITIONHI;
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
    return round(max(1.0, OsdTessLevel() * GetPostProjectionSphereExtent(center, diameter)));
}

#ifndef OSD_DISPLACEMENT_CALLBACK
#define OSD_DISPLACEMENT_CALLBACK
#endif

#if defined OSD_PATCH_ENABLE_SINGLE_CREASE
    Buffer<uint3> OsdPatchParamBuffer : register( t3 );
#else
    Buffer<uint2> OsdPatchParamBuffer : register( t3 );
#endif

#define GetPatchParam(primitiveID)                                      \
    (OsdPatchParamBuffer[GetPrimitiveID(primitiveID)].y)

#define GetPatchLevel(primitiveID)                                      \
    (OsdPatchParamBuffer[GetPrimitiveID(primitiveID)].y & 0xf)

#define GetSharpness(primitiveID)                                       \
    (asfloat(OsdPatchParamBuffer[GetPrimitiveID(primitiveID)].z))

#define OSD_COMPUTE_PTEX_COORD_HULL_SHADER                              \
    {                                                                   \
        int2 ptexIndex = OsdPatchParamBuffer[GetPrimitiveID(primitiveID)].xy; \
        int faceID = ptexIndex.x;                                       \
        int lv = 1 << ((ptexIndex.y & 0x7) - ((ptexIndex.y >> 3) & 1)); \
        int u = (ptexIndex.y >> 22) & 0x3ff;                            \
        int v = (ptexIndex.y >> 12) & 0x3ff;                            \
        output.patchCoord.w = faceID+0.5;                               \
        output.ptexInfo = int4(u, v, lv, 0);                            \
    }

#define OSD_COMPUTE_PTEX_COORD_DOMAIN_SHADER                            \
    {                                                                   \
        float2 uv = output.patchCoord.xy;                               \
        int2 p = patch[0].ptexInfo.xy;                                  \
        int lv = patch[0].ptexInfo.z;                                   \
        int rot = patch[0].ptexInfo.w;                                  \
        output.patchCoord.xy = (uv * float2(1.0,1.0)/lv) + float2(p.x, p.y)/lv; \
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

// ----------------------------------------------------------------------------

float3
EvalBezier(float3 cp[16], float2 uv)
{
    float3 BUCP[4] = { float3(0,0,0), float3(0,0,0), float3(0,0,0), float3(0,0,0) };

    float B[4], D[4];

    Univar4x4(uv.x, B, D);
    for (int i=0; i<4; ++i) {
        for (int j=0; j<4; ++j) {
            float3 A = cp[4*i + j];
            BUCP[i] += A * B[j];
        }
    }

    float3 position = float3(0,0,0);

    Univar4x4(uv.y, B, D);
    for (int k=0; k<4; ++k) {
        position  += B[k] * BUCP[k];
    }

    return position;
}

// ----------------------------------------------------------------------------
