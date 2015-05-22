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

//
// typical shader composition ordering (see glDrawRegistry:_CompileShader)
//
//
// - glsl version string  (#version 430)
//
// - common defines       (#define OSD_ENABLE_PATCH_CULL, ...)
// - source defines       (#define VERTEX_SHADER, ...)
//
// - osd headers          (glslPatchCommon: varying structs,
//                         glslPtexCommon: ptex functions)
// - client header        (Osd*Matrix(), displacement callback, ...)
//
// - osd shader source    (glslPatchBSpline, glslPatchGregory, ...)
//     or
//   client shader source (vertex/geometry/fragment shader)
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

// XXXdyu-patch-drawing support for fractional spacing
#undef OSD_FRACTIONAL_ODD_SPACING
#undef OSD_FRACTIONAL_EVEN_SPACING

#define M_PI 3.14159265359f

#if __VERSION__ < 420
    #define centroid
#endif

struct ControlVertex {
    vec4 position;
    ivec4 patchCoord;  // U offset, V offset, faceLevel, faceId
#ifdef OSD_ENABLE_PATCH_CULL
    ivec3 clipFlag;
#endif
};

struct OutputVertex {
    vec4 position;
    vec3 normal;
    vec3 tangent;
    vec3 bitangent;
    centroid vec4 patchCoord; // u, v, faceLevel, faceId
    centroid vec2 tessCoord; // tesscoord.st
#if defined OSD_COMPUTE_NORMAL_DERIVATIVES
    vec3 Nu;
    vec3 Nv;
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
#if defined OSD_MAX_VALENCE && OSD_MAX_VALENCE > 0
    vec3 r[OSD_MAX_VALENCE];
#endif
};

struct GregEvalVertex {
    vec3 position;
    vec3 Ep;
    vec3 Em;
    vec3 Fp;
    vec3 Fm;
    ivec4 patchCoord;
};

// osd shaders need following functions defined
mat4 OsdModelViewMatrix();
mat4 OsdProjectionMatrix();
mat4 OsdModelViewProjectionMatrix();
float OsdTessLevel();
int OsdGregoryQuadOffsetBase();
int OsdPrimitiveIdBase();
int OsdBaseVertex();

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

uniform isamplerBuffer OsdPatchParamBuffer;

int OsdGetPatchIndex(int primitiveId)
{
    return (primitiveId + OsdPrimitiveIdBase());
}

ivec3 OsdGetPatchParam(int patchIndex)
{
    return texelFetch(OsdPatchParamBuffer, patchIndex).xyz;
}

int OsdGetPatchFaceId(ivec3 patchParam)
{
    return patchParam.x;
}

int OsdGetPatchFaceLevel(ivec3 patchParam)
{
    return (1 << ((patchParam.y & 0x7) - ((patchParam.y >> 3) & 1)));
}

int OsdGetPatchRefinementLevel(ivec3 patchParam)
{
    return (patchParam.y & 0x7);
}

int OsdGetPatchBoundaryMask(ivec3 patchParam)
{
    return ((patchParam.y >> 4) & 0xf);
}

int OsdGetPatchTransitionMask(ivec3 patchParam)
{
    return ((patchParam.y >> 8) & 0xf);
}

ivec2 OsdGetPatchFaceUV(ivec3 patchParam)
{
    int u = (patchParam.y >> 22) & 0x3ff;
    int v = (patchParam.y >> 12) & 0x3ff;
    return ivec2(u,v);
}

float OsdGetPatchSharpness(ivec3 patchParam)
{
    return intBitsToFloat(patchParam.z);
}

ivec4 OsdGetPatchCoord(ivec3 patchParam)
{
    int faceId = OsdGetPatchFaceId(patchParam);
    int faceLevel = OsdGetPatchFaceLevel(patchParam);
    ivec2 faceUV = OsdGetPatchFaceUV(patchParam);
    return ivec4(faceUV.x, faceUV.y, faceLevel, faceId);
}

vec4 OsdInterpolatePatchCoord(vec2 localUV, ivec4 perPrimPatchCoord)
{
    int faceId = perPrimPatchCoord.w;
    int faceLevel = perPrimPatchCoord.z;
    vec2 faceUV = vec2(perPrimPatchCoord.x, perPrimPatchCoord.y);
    vec2 uv = localUV/faceLevel + faceUV/faceLevel;
    // add 0.5 to integer values for more robust interpolation
    return vec4(uv.x, uv.y, faceLevel+0.5f, faceId+0.5f);
}

// ----------------------------------------------------------------------------
// face varyings
// ----------------------------------------------------------------------------

uniform samplerBuffer OsdFVarDataBuffer;

#ifndef OSD_FVAR_WIDTH
#define OSD_FVAR_WIDTH 0
#endif

// ------ extract from quads (catmark, bilinear) ---------
// XXX: only linear interpolation is supported

#define OSD_COMPUTE_FACE_VARYING_1(result, fvarOffset, tessCoord)       \
    {                                                                   \
        float v[4];                                                     \
        int primOffset = OsdGetPatchIndex(gl_PrimitiveID) * 4;          \
        for (int i = 0; i < 4; ++i) {                                   \
            int index = (primOffset+i)*OSD_FVAR_WIDTH + fvarOffset;     \
            v[i] = texelFetch(OsdFVarDataBuffer, index).s               \
        }                                                               \
        result = mix(mix(v[0], v[1], tessCoord.s),                      \
                     mix(v[3], v[2], tessCoord.s),                      \
                     tessCoord.t);                                      \
    }

#define OSD_COMPUTE_FACE_VARYING_2(result, fvarOffset, tessCoord)       \
    {                                                                   \
        vec2 v[4];                                                      \
        int primOffset = OsdGetPatchIndex(gl_PrimitiveID) * 4;          \
        for (int i = 0; i < 4; ++i) {                                   \
            int index = (primOffset+i)*OSD_FVAR_WIDTH + fvarOffset;     \
            v[i] = vec2(texelFetch(OsdFVarDataBuffer, index).s,         \
                        texelFetch(OsdFVarDataBuffer, index + 1).s);    \
        }                                                               \
        result = mix(mix(v[0], v[1], tessCoord.s),                      \
                     mix(v[3], v[2], tessCoord.s),                      \
                     tessCoord.t);                                      \
    }

#define OSD_COMPUTE_FACE_VARYING_3(result, fvarOffset, tessCoord)       \
    {                                                                   \
        vec3 v[4];                                                      \
        int primOffset = OsdGetPatchIndex(gl_PrimitiveID) * 4;          \
        for (int i = 0; i < 4; ++i) {                                   \
            int index = (primOffset+i)*OSD_FVAR_WIDTH + fvarOffset;     \
            v[i] = vec3(texelFetch(OsdFVarDataBuffer, index).s,         \
                        texelFetch(OsdFVarDataBuffer, index + 1).s,     \
                        texelFetch(OsdFVarDataBuffer, index + 2).s);    \
        }                                                               \
        result = mix(mix(v[0], v[1], tessCoord.s),                      \
                     mix(v[3], v[2], tessCoord.s),                      \
                     tessCoord.t);                                      \
    }

#define OSD_COMPUTE_FACE_VARYING_4(result, fvarOffset, tessCoord)       \
    {                                                                   \
        vec4 v[4];                                                      \
        int primOffset = OsdGetPatchIndex(gl_PrimitiveID) * 4;          \
        for (int i = 0; i < 4; ++i) {                                   \
            int index = (primOffset+i)*OSD_FVAR_WIDTH + fvarOffset;     \
            v[i] = vec4(texelFetch(OsdFVarDataBuffer, index).s,         \
                        texelFetch(OsdFVarDataBuffer, index + 1).s,     \
                        texelFetch(OsdFVarDataBuffer, index + 2).s,     \
                        texelFetch(OsdFVarDataBuffer, index + 3).s);    \
        }                                                               \
        result = mix(mix(v[0], v[1], tessCoord.s),                      \
                     mix(v[3], v[2], tessCoord.s),                      \
                     tessCoord.t);                                      \
    }

// ------ extract from triangles (loop) ---------
// XXX: no interpolation supproted

#define OSD_COMPUTE_FACE_VARYING_TRI_1(result, fvarOffset, triVert)     \
    {                                                                   \
        int primOffset = OsdGetPatchIndex(gl_PrimitiveID) * 3;          \
        int index = (primOffset+triVert)*OSD_FVAR_WIDTH + fvarOffset;   \
        result = texelFetch(OsdFVarDataBuffer, index).s;                \
    }

#define OSD_COMPUTE_FACE_VARYING_TRI_2(result, fvarOffset, triVert)     \
    {                                                                   \
        int primOffset = OsdGetPatchIndex(gl_PrimitiveID) * 3;          \
        int index = (primOffset+triVert)*OSD_FVAR_WIDTH + fvarOffset;   \
        result = vec2(texelFetch(OsdFVarDataBuffer, index).s,           \
                      texelFetch(OsdFVarDataBuffer, index + 1).s);      \
    }

#define OSD_COMPUTE_FACE_VARYING_TRI_3(result, fvarOffset, triVert)     \
    {                                                                   \
        int primOffset = OsdGetPatchIndex(gl_PrimitiveID) * 3;          \
        int index = (primOffset+triVert)*OSD_FVAR_WIDTH + fvarOffset;   \
        result = vec3(texelFetch(OsdFVarDataBuffer, index).s,           \
                      texelFetch(OsdFVarDataBuffer, index + 1).s,       \
                      texelFetch(OsdFVarDataBuffer, index + 2).s);      \
    }

#define OSD_COMPUTE_FACE_VARYING_TRI_4(result, fvarOffset, triVert)     \
    {                                                                   \
        int primOffset = OsdGetPatchIndex(gl_PrimitiveID) * 3;          \
        int index = (primOffset+triVert)*OSD_FVAR_WIDTH + fvarOffset;   \
        result = vec4(texelFetch(OsdFVarDataBuffer, index).s,           \
                      texelFetch(OsdFVarDataBuffer, index + 1).s,       \
                      texelFetch(OsdFVarDataBuffer, index + 2).s,       \
                      texelFetch(OsdFVarDataBuffer, index + 3).s);      \
    }

// ----------------------------------------------------------------------------
// patch culling
// ----------------------------------------------------------------------------

#ifdef OSD_ENABLE_PATCH_CULL

#define OSD_PATCH_CULL_COMPUTE_CLIPFLAGS(P)                     \
    vec4 clipPos = OsdModelViewProjectionMatrix() * P;          \
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

vec3
OsdEvalBezier(vec3 cp[16], vec2 uv)
{
    vec3 BUCP[4] = vec3[4](vec3(0,0,0), vec3(0,0,0), vec3(0,0,0), vec3(0,0,0));

    float B[4], D[4];

    Univar4x4(uv.x, B, D);
    for (int i=0; i<4; ++i) {
        for (int j=0; j<4; ++j) {
            vec3 A = cp[4*i + j];
            BUCP[i] += A * B[j];
        }
    }

    vec3 position = vec3(0);

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
OsdComputeBSplineBoundaryPoints(inout vec3 cpt[16], ivec3 patchParam)
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

float OsdComputePostProjectionSphereExtent(vec3 center, float diameter)
{
    vec4 p = OsdModelViewProjectionMatrix() * vec4(center, 1.0);
    return abs(diameter * OsdModelViewProjectionMatrix()[1][1] / p.w);
}

float OsdComputeTessLevel(vec3 p0, vec3 p1)
{
    // Adaptive factor can be any computation that depends only on arg values.
    // Project the diameter of the edge's bounding sphere instead of using the
    // length of the projected edge itself to avoid problems near silhouettes.
    vec3 center = (p0 + p1) / 2.0;
    float diameter = distance(p0, p1);
    float projLength = OsdComputePostProjectionSphereExtent(center, diameter);
    return round(max(1.0, OsdTessLevel() * projLength));
}

void
OsdGetTessLevelsUniform(ivec3 patchParam,
                        inout vec4 tessOuterLo, inout vec4 tessOuterHi)
{
    int refinementLevel = OsdGetPatchRefinementLevel(patchParam);
    float tessLevel = OsdTessLevel() / pow(2, refinementLevel-1);

    tessOuterLo = vec4(tessLevel);
    tessOuterHi = vec4(0);
}

void
OsdGetTessLevelsRefinedPoints(vec3 cp[16], ivec3 patchParam,
                              inout vec4 tessOuterLo, inout vec4 tessOuterHi)
{
    // Each edge of a transition patch is adjacent to one or two patches
    // at the next refined level of subdivision. We compute the corresponding
    // vertex-vertex and edge-vertex refined points along the edges of the
    // patch using Catmull-Clark subdivision stencil weights.
    // For simplicity, we let the optimizer discard unused computation.

    vec3 vv0 = (cp[0] + cp[2] + cp[8] + cp[10]) * 0.015625 +
               (cp[1] + cp[4] + cp[6] + cp[9]) * 0.09375 + cp[5] * 0.5625;
    vec3 ev01 = (cp[1] + cp[2] + cp[9] + cp[10]) * 0.0625 +
                (cp[5] + cp[6]) * 0.375;

    vec3 vv1 = (cp[1] + cp[3] + cp[9] + cp[11]) * 0.015625 +
               (cp[2] + cp[5] + cp[7] + cp[10]) * 0.09375 + cp[6] * 0.5625;
    vec3 ev12 = (cp[5] + cp[7] + cp[9] + cp[11]) * 0.0625 +
                (cp[6] + cp[10]) * 0.375;

    vec3 vv2 = (cp[5] + cp[7] + cp[13] + cp[15]) * 0.015625 +
               (cp[6] + cp[9] + cp[11] + cp[14]) * 0.09375 + cp[10] * 0.5625;
    vec3 ev23 = (cp[5] + cp[6] + cp[13] + cp[14]) * 0.0625 +
                (cp[9] + cp[10]) * 0.375;

    vec3 vv3 = (cp[4] + cp[6] + cp[12] + cp[14]) * 0.015625 +
               (cp[5] + cp[8] + cp[10] + cp[13]) * 0.09375 + cp[9] * 0.5625;
    vec3 ev03 = (cp[4] + cp[6] + cp[8] + cp[10]) * 0.0625 +
                (cp[5] + cp[9]) * 0.375;

    tessOuterLo = vec4(0);
    tessOuterHi = vec4(0);

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
OsdGetTessLevelsLimitPoints(vec3 cpBezier[16], ivec3 patchParam,
                            inout vec4 tessOuterLo, inout vec4 tessOuterHi)
{
    // Each edge of a transition patch is adjacent to one or two patches
    // at the next refined level of subdivision. When the patch control
    // points have been converted to the Bezier basis, the control points
    // at the four corners are on the limit surface (since a Bezier patch
    // interpolates its corner control points). We can compute an adaptive
    // tessellation level for transition edges on the limit surface by
    // evaluating a limit position at the mid point of each transition edge.

    tessOuterLo = vec4(0);
    tessOuterHi = vec4(0);

    int transitionMask = OsdGetPatchTransitionMask(patchParam);

    if ((transitionMask & 8) != 0) {
        vec3 ev03 = OsdEvalBezier(cpBezier, vec2(0.0, 0.5));
        tessOuterLo[0] = OsdComputeTessLevel(cpBezier[0], ev03);
        tessOuterHi[0] = OsdComputeTessLevel(cpBezier[12], ev03);
    } else {
        tessOuterLo[0] = OsdComputeTessLevel(cpBezier[0], cpBezier[12]);
    }
    if ((transitionMask & 1) != 0) {
        vec3 ev01 = OsdEvalBezier(cpBezier, vec2(0.5, 0.0));
        tessOuterLo[1] = OsdComputeTessLevel(cpBezier[0], ev01);
        tessOuterHi[1] = OsdComputeTessLevel(cpBezier[3], ev01);
    } else {
        tessOuterLo[1] = OsdComputeTessLevel(cpBezier[0], cpBezier[3]);
    }
    if ((transitionMask & 2) != 0) {
        vec3 ev12 = OsdEvalBezier(cpBezier, vec2(1.0, 0.5));
        tessOuterLo[2] = OsdComputeTessLevel(cpBezier[3], ev12);
        tessOuterHi[2] = OsdComputeTessLevel(cpBezier[15], ev12);
    } else {
        tessOuterLo[2] = OsdComputeTessLevel(cpBezier[3], cpBezier[15]);
    }
    if ((transitionMask & 4) != 0) {
        vec3 ev23 = OsdEvalBezier(cpBezier, vec2(0.5, 1.0));
        tessOuterLo[3] = OsdComputeTessLevel(cpBezier[12], ev23);
        tessOuterHi[3] = OsdComputeTessLevel(cpBezier[15], ev23);
    } else {
        tessOuterLo[3] = OsdComputeTessLevel(cpBezier[12], cpBezier[15]);
    }
}

void
OsdGetTessLevels(vec3 cp[16], ivec3 patchParam,
                 inout vec4 tessLevelOuter, inout vec2 tessLevelInner,
                 inout vec4 tessOuterLo, inout vec4 tessOuterHi)
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
OsdGetTessLevels(vec3 cp0, vec3 cp1, vec3 cp2, vec3 cp3,
                 ivec3 patchParam,
                 inout vec4 tessLevelOuter, inout vec2 tessLevelInner)
{
    vec4 tessOuterLo = vec4(0);
    vec4 tessOuterHi = vec4(0);

#if defined OSD_ENABLE_SCREENSPACE_TESSELLATION
    tessOuterLo[0] = OsdComputeTessLevel(cp0, cp1);
    tessOuterLo[1] = OsdComputeTessLevel(cp0, cp3);
    tessOuterLo[2] = OsdComputeTessLevel(cp2, cp3);
    tessOuterLo[3] = OsdComputeTessLevel(cp1, cp2);
    tessOuterHi = vec4(0);
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

vec2
OsdGetTessParameterization(vec2 uv, vec4 tessOuterLo, vec4 tessOuterHi)
{
    vec2 UV = uv;
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

