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

#define M_PI 3.14159265359f

struct InputVertex {
    float4 position : POSITION;
    float3 normal : NORMAL;
};

struct HullVertex {
    float4 position : POSITION;
    int4 patchCoord : PATCHCOORD; // U offset, V offset, faceLevel, faceId
#ifdef OSD_ENABLE_PATCH_CULL
    int3 clipFlag : CLIPFLAG;
#endif
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
    float3 tangent : TANGENT;
    float3 bitangent : TANGENT1;
    float4 patchCoord : PATCHCOORD; // u, v, faceLevel, faceId
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
#if defined OSD_MAX_VALENCE && OSD_MAX_VALENCE > 0
    float3 r[OSD_MAX_VALENCE] : POSITION4;
#endif
};

struct GregDomainVertex {
    float3 position : POSITION0;
    float3 Ep : POSITION1;
    float3 Em : POSITION2;
    float3 Fp : POSITION3;
    float3 Fm : POSITION4;
    int4   patchCoord: PATCHCOORD;
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

#ifndef OSD_DISPLACEMENT_CALLBACK
#define OSD_DISPLACEMENT_CALLBACK
#endif

// ----------------------------------------------------------------------------
// Patch Parameters
// ----------------------------------------------------------------------------

//
// Each patch has a corresponding patchParam. This is a set of three values
// specifying additional information about the patch:
//
//    faceId    -- topological face identifier (e.g. Ptex FaceId)
//    bitfield  -- refinement-level, non-quad, boundary, transition, uv-offset
//    sharpness -- crease sharpness for single-crease patches
//
// These are stored in OsdPatchParamBuffer indexed by the value returned
// from OsdGetPatchIndex() which is a function of the current PrimitiveID
// along with an optional client provided offset.
//

#if defined OSD_PATCH_ENABLE_SINGLE_CREASE
    Buffer<uint3> OsdPatchParamBuffer : register( t0 );
#else
    Buffer<uint2> OsdPatchParamBuffer : register( t0 );
#endif

int OsdGetPatchIndex(int primitiveId)
{
    return (primitiveId + OsdPrimitiveIdBase());
}

int3 OsdGetPatchParam(int patchIndex)
{
#if defined OSD_PATCH_ENABLE_SINGLE_CREASE
    return OsdPatchParamBuffer[patchIndex].xyz;
#else
    uint2 p = OsdPatchParamBuffer[patchIndex].xy;
    return int3(p.x, p.y, 0);
#endif
}

int OsdGetPatchFaceId(int3 patchParam)
{
    return patchParam.x;
}

int OsdGetPatchFaceLevel(int3 patchParam)
{
    return (1 << ((patchParam.y & 0x7) - ((patchParam.y >> 3) & 1)));
}

int OsdGetPatchRefinementLevel(int3 patchParam)
{
    return (patchParam.y & 0x7);
}

int OsdGetPatchBoundaryMask(int3 patchParam)
{
    return ((patchParam.y >> 4) & 0xf);
}

int OsdGetPatchTransitionMask(int3 patchParam)
{
    return ((patchParam.y >> 8) & 0xf);
}

int2 OsdGetPatchFaceUV(int3 patchParam)
{
    int u = (patchParam.y >> 22) & 0x3ff;
    int v = (patchParam.y >> 12) & 0x3ff;
    return int2(u,v);
}

float OsdGetPatchSharpness(int3 patchParam)
{
    return asfloat(patchParam.z);
}

int4 OsdGetPatchCoord(int3 patchParam)
{
    int faceId = OsdGetPatchFaceId(patchParam);
    int faceLevel = OsdGetPatchFaceLevel(patchParam);
    int2 faceUV = OsdGetPatchFaceUV(patchParam);
    return int4(faceUV.x, faceUV.y, faceLevel, faceId);
}

float4 OsdInterpolatePatchCoord(float2 localUV, int4 perPrimPatchCoord)
{
    int faceId = perPrimPatchCoord.w;
    int faceLevel = perPrimPatchCoord.z;
    float2 faceUV = float2(perPrimPatchCoord.x, perPrimPatchCoord.y);
    float2 uv = localUV/faceLevel + faceUV/faceLevel;
    return float4(uv.x, uv.y, faceLevel+0.5, faceId+0.5);
}

// ----------------------------------------------------------------------------
// patch culling
// ----------------------------------------------------------------------------

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
OsdEvalBezier(float3 cp[16], float2 uv)
{
    float3 BUCP[4] = {
        float3(0,0,0), float3(0,0,0), float3(0,0,0), float3(0,0,0)
    };

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
// Boundary Interpolation
// ----------------------------------------------------------------------------

void
OsdComputeBSplineBoundaryPoints(inout float3 cpt[16], int3 patchParam)
{
    int boundaryMask = OsdGetPatchBoundaryMask(patchParam);

    if ((boundaryMask & 1) != 0) {
        cpt[0] = 2*cpt[4] - cpt[8];
        cpt[1] = 2*cpt[5] - cpt[9];
        cpt[2] = 2*cpt[6] - cpt[10];
        cpt[3] = 2*cpt[7] - cpt[11];
    }
    if ((boundaryMask & 2) != 0) {
        cpt[3] = 2*cpt[2] - cpt[1];
        cpt[7] = 2*cpt[6] - cpt[5];
        cpt[11] = 2*cpt[10] - cpt[9];
        cpt[15] = 2*cpt[14] - cpt[13];
    }
    if ((boundaryMask & 4) != 0) {
        cpt[12] = 2*cpt[8] - cpt[4];
        cpt[13] = 2*cpt[9] - cpt[5];
        cpt[14] = 2*cpt[10] - cpt[6];
        cpt[15] = 2*cpt[11] - cpt[7];
    }
    if ((boundaryMask & 8) != 0) {
        cpt[0] = 2*cpt[1] - cpt[2];
        cpt[4] = 2*cpt[5] - cpt[6];
        cpt[8] = 2*cpt[9] - cpt[10];
        cpt[12] = 2*cpt[13] - cpt[14];
    }
}

// ----------------------------------------------------------------------------
// Tessellation
// ----------------------------------------------------------------------------

//
// Organization of B-spline and Bezier control points.
//
// Each patch is defined by 16 control points (labeled 0-15).
//
// The patch will be evaluated across the domain from (0,0) at
// the lower-left to (1,1) at the upper-right. When computing
// adaptive tessellation metrics, we consider refined vertex-vertex
// and edge-vertex points along the transition edges of the patch
// (labeled vv* and ev* respectively).
//
// The two segments of each transition edge are labeled Lo and Hi,
// with the Lo segment occuring before the Hi segment along the
// transition edge's domain parameterization. These Lo and Hi segment
// tessellation levels determine how domain evaluation coordinates
// are remapped along transition edges. The Hi segment value will
// be zero for a non-transition edge.
//
// (0,1)                                         (1,1)
//
//   vv3                  ev23                   vv2
//        |       Lo3       |       Hi3       |
//      --O-----------O-----+-----O-----------O--
//        | 12        | 13     14 |        15 |
//        |           |           |           |
//        |           |           |           |
//    Hi0 |           |           |           | Hi2
//        |           |           |           |
//        O-----------O-----------O-----------O
//        | 8         | 9      10 |        11 |
//        |           |           |           |
// ev03 --+           |           |           +-- ev12
//        |           |           |           |
//        | 4         | 5       6 |         7 |
//        O-----------O-----------O-----------O
//        |           |           |           |
//    Lo0 |           |           |           | Lo2
//        |           |           |           |
//        |           |           |           |
//        | 0         | 1       2 |         3 |
//      --O-----------O-----+-----O-----------O--
//        |       Lo1       |       Hi1       |
//   vv0                  ev01                   vv1
//
// (0,0)                                         (1,0)
//

float OsdComputePostProjectionSphereExtent(float3 center, float diameter)
{
    float4 p = mul(OsdModelViewProjectionMatrix(), float4(center, 1.0));
    return abs(diameter * OsdModelViewProjectionMatrix()[1][1] / p.w);
}

float OsdComputeTessLevel(float3 p0, float3 p1)
{
    // Adaptive factor can be any computation that depends only on arg values.
    // Project the diameter of the edge's bounding sphere instead of using the
    // length of the projected edge itself to avoid problems near silhouettes.
    float3 center = (p0 + p1) / 2.0;
    float diameter = distance(p0, p1);
    float projLength = OsdComputePostProjectionSphereExtent(center, diameter);
    return round(max(1.0, OsdTessLevel() * projLength));
}

void
OsdGetTessLevelsUniform(int3 patchParam,
                        inout float4 tessOuterLo, inout float4 tessOuterHi)
{
    int refinementLevel = OsdGetPatchRefinementLevel(patchParam);
    float tessLevel = OsdTessLevel() / pow(2, refinementLevel-1);

    tessOuterLo = float4(tessLevel,tessLevel,tessLevel,tessLevel);
    tessOuterHi = float4(0,0,0,0);
}

void
OsdGetTessLevelsRefinedPoints(float3 cp[16], int3 patchParam,
                              inout float4 tessOuterLo, inout float4 tessOuterHi)
{
    // Each edge of a transition patch is adjacent to one or two patches
    // at the next refined level of subdivision. We compute the corresponding
    // vertex-vertex and edge-vertex refined points along the edges of the
    // patch using Catmull-Clark subdivision stencil weights.
    // For simplicity, we let the optimizer discard unused computation.

    float3 vv0 = (cp[0] + cp[2] + cp[8] + cp[10]) * 0.015625 +
                 (cp[1] + cp[4] + cp[6] + cp[9]) * 0.09375 + cp[5] * 0.5625;
    float3 ev01 = (cp[1] + cp[2] + cp[9] + cp[10]) * 0.0625 +
                  (cp[5] + cp[6]) * 0.375;

    float3 vv1 = (cp[1] + cp[3] + cp[9] + cp[11]) * 0.015625 +
                 (cp[2] + cp[5] + cp[7] + cp[10]) * 0.09375 + cp[6] * 0.5625;
    float3 ev12 = (cp[5] + cp[7] + cp[9] + cp[11]) * 0.0625 +
                  (cp[6] + cp[10]) * 0.375;

    float3 vv2 = (cp[5] + cp[7] + cp[13] + cp[15]) * 0.015625 +
                 (cp[6] + cp[9] + cp[11] + cp[14]) * 0.09375 + cp[10] * 0.5625;
    float3 ev23 = (cp[5] + cp[6] + cp[13] + cp[14]) * 0.0625 +
                  (cp[9] + cp[10]) * 0.375;

    float3 vv3 = (cp[4] + cp[6] + cp[12] + cp[14]) * 0.015625 +
                 (cp[5] + cp[8] + cp[10] + cp[13]) * 0.09375 + cp[9] * 0.5625;
    float3 ev03 = (cp[4] + cp[6] + cp[8] + cp[10]) * 0.0625 +
                  (cp[5] + cp[9]) * 0.375;

    tessOuterLo = float4(0,0,0,0);
    tessOuterHi = float4(0,0,0,0);

    int transitionMask = OsdGetPatchTransitionMask(patchParam);

    if ((transitionMask & 8) != 0) {
        tessOuterLo[0] = OsdComputeTessLevel(vv0, ev03);
        tessOuterHi[0] = OsdComputeTessLevel(vv3, ev03);
    } else {
        tessOuterLo[0] = OsdComputeTessLevel(cp[5], cp[9]);
    }
    if ((transitionMask & 1) != 0) {
        tessOuterLo[1] = OsdComputeTessLevel(vv0, ev01);
        tessOuterHi[1] = OsdComputeTessLevel(vv1, ev01);
    } else {
        tessOuterLo[1] = OsdComputeTessLevel(cp[5], cp[6]);
    }
    if ((transitionMask & 2) != 0) {
        tessOuterLo[2] = OsdComputeTessLevel(vv1, ev12);
        tessOuterHi[2] = OsdComputeTessLevel(vv2, ev12);
    } else {
        tessOuterLo[2] = OsdComputeTessLevel(cp[6], cp[10]);
    }
    if ((transitionMask & 4) != 0) {
        tessOuterLo[3] = OsdComputeTessLevel(vv3, ev23);
        tessOuterHi[3] = OsdComputeTessLevel(vv2, ev23);
    } else {
        tessOuterLo[3] = OsdComputeTessLevel(cp[9], cp[10]);
    }
}

void
OsdGetTessLevelsLimitPoints(float3 cpBezier[16], int3 patchParam,
                            inout float4 tessOuterLo, inout float4 tessOuterHi)
{
    // Each edge of a transition patch is adjacent to one or two patches
    // at the next refined level of subdivision. When the patch control
    // points have been converted to the Bezier basis, the control points
    // at the four corners are on the limit surface (since a Bezier patch
    // interpolates its corner control points). We can compute an adaptive
    // tessellation level for transition edges on the limit surface by
    // evaluating a limit position at the mid point of each transition edge.

    tessOuterLo = float4(0,0,0,0);
    tessOuterHi = float4(0,0,0,0);

    int transitionMask = OsdGetPatchTransitionMask(patchParam);

    if ((transitionMask & 8) != 0) {
        float3 ev03 = OsdEvalBezier(cpBezier, float2(0.0, 0.5));
        tessOuterLo[0] = OsdComputeTessLevel(cpBezier[0], ev03);
        tessOuterHi[0] = OsdComputeTessLevel(cpBezier[12], ev03);
    } else {
        tessOuterLo[0] = OsdComputeTessLevel(cpBezier[0], cpBezier[12]);
    }
    if ((transitionMask & 1) != 0) {
        float3 ev01 = OsdEvalBezier(cpBezier, float2(0.5, 0.0));
        tessOuterLo[1] = OsdComputeTessLevel(cpBezier[0], ev01);
        tessOuterHi[1] = OsdComputeTessLevel(cpBezier[3], ev01);
    } else {
        tessOuterLo[1] = OsdComputeTessLevel(cpBezier[0], cpBezier[3]);
    }
    if ((transitionMask & 2) != 0) {
        float3 ev12 = OsdEvalBezier(cpBezier, float2(1.0, 0.5));
        tessOuterLo[2] = OsdComputeTessLevel(cpBezier[3], ev12);
        tessOuterHi[2] = OsdComputeTessLevel(cpBezier[15], ev12);
    } else {
        tessOuterLo[2] = OsdComputeTessLevel(cpBezier[3], cpBezier[15]);
    }
    if ((transitionMask & 4) != 0) {
        float3 ev23 = OsdEvalBezier(cpBezier, float2(0.5, 1.0));
        tessOuterLo[3] = OsdComputeTessLevel(cpBezier[12], ev23);
        tessOuterHi[3] = OsdComputeTessLevel(cpBezier[15], ev23);
    } else {
        tessOuterLo[3] = OsdComputeTessLevel(cpBezier[12], cpBezier[15]);
    }
}

void
OsdGetTessLevels(float3 cp[16], int3 patchParam,
                 inout float4 tessLevelOuter, inout float4 tessLevelInner,
                 inout float4 tessOuterLo, inout float4 tessOuterHi)
{
#if defined OSD_ENABLE_SCREENSPACE_TESSELLATION
    OsdGetTessLevelsLimitPoints(cp, patchParam, tessOuterLo, tessOuterHi);
#elif defined OSD_ENABLE_SCREENSPACE_TESSELLATION_REFINED
    OsdGetTessLevelsRefinedPoints(cp, patchParam, tessOuterLo, tessOuterHi);
#else
    OsdGetTessLevelsUniform(patchParam, tessOuterLo, tessOuterHi);
#endif

    // Outer levels are the sum of the Lo and Hi segments where the Hi
    // segments will have a length of zero for non-transition edges.
    tessLevelOuter = tessOuterLo + tessOuterHi;

    // Inner levels are the average the corresponding outer levels.
    tessLevelInner[0] = (tessLevelOuter[1] + tessLevelOuter[3]) * 0.5;
    tessLevelInner[1] = (tessLevelOuter[0] + tessLevelOuter[2]) * 0.5;
}

void
OsdGetTessLevels(float3 cp0, float3 cp1, float3 cp2, float3 cp3,
                 int3 patchParam,
                 inout float4 tessLevelOuter, inout float4 tessLevelInner)
{
    float4 tessOuterLo = float4(0,0,0,0);
    float4 tessOuterHi = float4(0,0,0,0);

#if defined OSD_ENABLE_SCREENSPACE_TESSELLATION
    tessOuterLo[0] = OsdComputeTessLevel(cp0, cp1);
    tessOuterLo[1] = OsdComputeTessLevel(cp0, cp3);
    tessOuterLo[2] = OsdComputeTessLevel(cp2, cp3);
    tessOuterLo[3] = OsdComputeTessLevel(cp1, cp2);
    tessOuterHi = float4(0,0,0,0);
#else
    OsdGetTessLevelsUniform(patchParam, tessOuterLo, tessOuterHi);
#endif

    // Outer levels are the sum of the Lo and Hi segments where the Hi
    // segments will have a length of zero for non-transition edges.
    tessLevelOuter = tessOuterLo + tessOuterHi;

    // Inner levels are the average the corresponding outer levels.
    tessLevelInner[0] = (tessLevelOuter[1] + tessLevelOuter[3]) * 0.5;
    tessLevelInner[1] = (tessLevelOuter[0] + tessLevelOuter[2]) * 0.5;
}

float
OsdGetTessTransitionSplit(float t, float n0, float n1)
{
    float ti = round(t * (n0 + n1));

    if (ti <= n0) {
        return 0.5 * (ti / n0);
    } else {
        return 0.5 * ((ti - n0) / n1) + 0.5;
    }
}

float2
OsdGetTessParameterization(float2 uv, float4 tessOuterLo, float4 tessOuterHi)
{
    float2 UV = uv;
    if (UV.x == 0 && tessOuterHi[0] > 0) {
        UV.y = OsdGetTessTransitionSplit(UV.y, tessOuterLo[0], tessOuterHi[0]);
    } else
    if (UV.y == 0 && tessOuterHi[1] > 0) {
        UV.x = OsdGetTessTransitionSplit(UV.x, tessOuterLo[1], tessOuterHi[1]);
    } else
    if (UV.x == 1 && tessOuterHi[2] > 0) {
        UV.y = OsdGetTessTransitionSplit(UV.y, tessOuterLo[2], tessOuterHi[2]);
    } else
    if (UV.y == 1 && tessOuterHi[3] > 0) {
        UV.x = OsdGetTessTransitionSplit(UV.x, tessOuterLo[3], tessOuterHi[3]);
    }
    return UV;
}

