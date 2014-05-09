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

#if __VERSION__ < 420
    #define centroid
#endif

struct ControlVertex {
    vec4 position;
    centroid vec4 patchCoord; // u, v, level, faceID
    ivec4 ptexInfo;  // U offset, V offset, 2^ptexlevel', rotation
    ivec3 clipFlag;
};

struct OutputVertex {
    vec4 position;
    vec3 normal;
    centroid vec4 patchCoord; // u, v, level, faceID
    centroid vec2 tessCoord; // tesscoord.st
    vec3 tangent;
    vec3 bitangent;
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

// osd shaders need following functions defined
mat4 OsdModelViewMatrix();
mat4 OsdProjectionMatrix();
mat4 OsdModelViewProjectionMatrix();
float OsdTessLevel();
int OsdGregoryQuadOffsetBase();
int OsdPrimitiveIdBase();
int OsdBaseVertex();

float GetTessLevel(int patchLevel)
{
#ifdef OSD_ENABLE_SCREENSPACE_TESSELLATION
    return OsdTessLevel();
#else
    return OsdTessLevel() / pow(2, patchLevel-1);
#endif
}

#ifndef GetPrimitiveID
#define GetPrimitiveID() (gl_PrimitiveID + OsdPrimitiveIdBase())
#endif

float GetPostProjectionSphereExtent(vec3 center, float diameter)
{
    vec4 p = OsdProjectionMatrix() * vec4(center, 1.0);
    return abs(diameter * OsdProjectionMatrix()[1][1] / p.w);
}

float TessAdaptive(vec3 p0, vec3 p1)
{
    // Adaptive factor can be any computation that depends only on arg values.
    // Project the diameter of the edge's bounding sphere instead of using the
    // length of the projected edge itself to avoid problems near silhouettes.
    vec3 center = (p0 + p1) / 2.0;
    float diameter = distance(p0, p1);
    return max(1.0, OsdTessLevel() * GetPostProjectionSphereExtent(center, diameter));
}

#ifndef OSD_DISPLACEMENT_CALLBACK
#define OSD_DISPLACEMENT_CALLBACK
#endif

// ----------------------------------------------------------------------------
// ptex coordinates
// ----------------------------------------------------------------------------

uniform isamplerBuffer OsdPatchParamBuffer;

#define GetPatchLevel()                                                 \
    (texelFetch(OsdPatchParamBuffer, GetPrimitiveID()).y & 0xf)

#define OSD_COMPUTE_PTEX_COORD_TESSCONTROL_SHADER                       \
    {                                                                   \
        ivec2 ptexIndex = texelFetch(OsdPatchParamBuffer,               \
                                     GetPrimitiveID()).xy;              \
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

#define OSD_COMPUTE_PTEX_COMPATIBLE_TANGENT(ROTATE)                 \
    {                                                               \
        int rot = (inpt[0].v.ptexInfo.w + 4 - ROTATE)%4;            \
        if (rot == 1) {                                             \
            outpt.v.tangent = -BiTangent;                           \
            outpt.v.bitangent = Tangent;                            \
        } else if (rot == 2) {                                      \
            outpt.v.tangent = -Tangent;                             \
            outpt.v.bitangent = -BiTangent;                         \
        } else if (rot == 3) {                                      \
            outpt.v.tangent = BiTangent;                            \
            outpt.v.bitangent = -Tangent;                           \
        } else {                                                    \
            outpt.v.tangent = Tangent;                              \
            outpt.v.bitangent = BiTangent;                          \
        }                                                           \
    }

#define OSD_COMPUTE_PTEX_COMPATIBLE_DERIVATIVES(ROTATE)             \
    {                                                               \
        int rot = (inpt[0].v.ptexInfo.w + 4 - ROTATE)%4;            \
        if (rot == 1) {                                             \
            outpt.v.tangent = -BiTangent;                           \
            outpt.v.bitangent = Tangent;                            \
            outpt.v.Nu = -Nv;                                       \
            outpt.v.Nv = Nv;                                        \
        } else if (rot == 2) {                                      \
            outpt.v.tangent = -Tangent;                             \
            outpt.v.bitangent = -BiTangent;                         \
            outpt.v.Nu = -Nu;                                       \
            outpt.v.Nv = -Nv;                                       \
        } else if (rot == 3) {                                      \
            outpt.v.tangent = BiTangent;                            \
            outpt.v.bitangent = -Tangent;                           \
            outpt.v.Nu = Nv;                                        \
            outpt.v.Nv = -Nu;                                       \
        } else {                                                    \
            outpt.v.tangent = Tangent;                              \
            outpt.v.bitangent = BiTangent;                          \
            outpt.v.Nu = Nu;                                        \
            outpt.v.Nv = Nv;                                        \
        }                                                           \
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
        int primOffset = GetPrimitiveID() * 4;                          \
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
        int primOffset = GetPrimitiveID() * 4;                          \
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
        int primOffset = GetPrimitiveID() * 4;                          \
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
        int primOffset = GetPrimitiveID() * 4;                          \
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
        int primOffset = GetPrimitiveID() * 3;                          \
        int index = (primOffset+triVert)*OSD_FVAR_WIDTH + fvarOffset;   \
        result = texelFetch(OsdFVarDataBuffer, index).s;                \
    }

#define OSD_COMPUTE_FACE_VARYING_TRI_2(result, fvarOffset, triVert)     \
    {                                                                   \
        int primOffset = GetPrimitiveID() * 3;                          \
        int index = (primOffset+triVert)*OSD_FVAR_WIDTH + fvarOffset;   \
        result = vec2(texelFetch(OsdFVarDataBuffer, index).s,           \
                      texelFetch(OsdFVarDataBuffer, index + 1).s);      \
    }

#define OSD_COMPUTE_FACE_VARYING_TRI_3(result, fvarOffset, triVert)     \
    {                                                                   \
        int primOffset = GetPrimitiveID() * 3;                          \
        int index = (primOffset+triVert)*OSD_FVAR_WIDTH + fvarOffset;   \
        result = vec3(texelFetch(OsdFVarDataBuffer, index).s,           \
                      texelFetch(OsdFVarDataBuffer, index + 1).s,       \
                      texelFetch(OsdFVarDataBuffer, index + 2).s);      \
    }

#define OSD_COMPUTE_FACE_VARYING_TRI_4(result, fvarOffset, triVert)     \
    {                                                                   \
        int primOffset = GetPrimitiveID() * 3;                          \
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
