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
#line 25
//--------------------------------------------------------------
// Common
//--------------------------------------------------------------
uniform float displacementScale = 1.0;
uniform float mipmapBias = 0;

struct PtexPacking
{
    int page;
    int nMipmap;
    int uOffset;
    int vOffset;
    int adjSizeDiffs[4];
    int width;
    int height;
};

vec4 GeneratePatchCoord(vec2 localUV, int primitiveID)  // for non-adpative
{
    ivec2 ptexIndex = texelFetch(OsdPatchParamBuffer, primitiveID).xy;
    int faceID = ptexIndex.x;
    int lv = 1 << ((ptexIndex.y & 0xf) - ((ptexIndex.y >> 4) & 1));
    int u = (ptexIndex.y >> 17) & 0x3ff;
    int v = (ptexIndex.y >> 7) & 0x3ff;
    vec2 uv = localUV;
    uv = (uv * vec2(1.0)/lv) + vec2(u, v)/lv;

    return vec4(uv.x, uv.y, lv+0.5, faceID+0.5);
}

PtexPacking getPtexPacking(isamplerBuffer packings, int faceID)
{
    PtexPacking packing;
    packing.page    = texelFetch(packings, faceID*6).x;
    packing.nMipmap = texelFetch(packings, faceID*6+1).x;
    packing.uOffset = texelFetch(packings, faceID*6+2).x;
    packing.vOffset = texelFetch(packings, faceID*6+3).x;
    int wh          = texelFetch(packings, faceID*6+5).x;
    packing.width   = 1 << (wh >> 8);
    packing.height  = 1 << (wh & 0xff);

    int adjSizeDiffs = texelFetch(packings, faceID*6+4).x;
    packing.adjSizeDiffs[0] = (adjSizeDiffs >> 12) & 0xf;
    packing.adjSizeDiffs[1] = (adjSizeDiffs >> 8) & 0xf;
    packing.adjSizeDiffs[2] = (adjSizeDiffs >> 4) & 0xf;
    packing.adjSizeDiffs[3] = (adjSizeDiffs >> 0) & 0xf;

    return packing;
}

int computeMipmapOffsetU(int w, int level)
{
    int width = 1 << w;
    int m = (0x55555555 & (width | (width-1))) << (w&1);
    int x = ~((1 << (w -((level-1)&~1))) - 1);
    return (m & x) + ((level+1)&~1);
}

int computeMipmapOffsetV(int h, int level)
{
    int height = 1 << h;
    int m = (0x55555555 & (height-1)) << ((h+1)&1);;
    int x = ~((1 << (h - (level&~1))) - 1 );
    return (m & x) + (level&~1);
}

PtexPacking getPtexPacking(isamplerBuffer packings, int faceID, int level)
{
    PtexPacking packing;
    packing.page    = texelFetch(packings, faceID*6).x;
    packing.nMipmap = texelFetch(packings, faceID*6+1).x;
    packing.uOffset = texelFetch(packings, faceID*6+2).x;
    packing.vOffset = texelFetch(packings, faceID*6+3).x;
    int sizeDiffs   = texelFetch(packings, faceID*6+4).x;
    int wh          = texelFetch(packings, faceID*6+5).x;
    int w = wh >> 8;
    int h = wh & 0xff;

    // clamp max level
    level = min(level, packing.nMipmap);

#if 0
    packing.width = 1 << w;
    packing.height = 1 << h;
    // offset mipmap location (slow!)
    for (int i = 1; i <= level; ++i) {
        packing.uOffset += (packing.width+2) * (i&1);
        packing.vOffset += (packing.height+2) * (1-i&1);
        packing.width /= 2;
        packing.height /= 2;
    }
#else
    packing.uOffset += computeMipmapOffsetU(w, level);
    packing.vOffset += computeMipmapOffsetV(h, level);
    packing.width = 1 << (w-level);
    packing.height = 1 << (h-level);
#endif
    return packing;
}

vec4 PTexLookupNearest(vec4 patchCoord,
                       sampler2DArray data,
                       isamplerBuffer packings)
{
    vec2 uv = patchCoord.xy;
    int faceID = int(patchCoord.w);
    PtexPacking ppack = getPtexPacking(packings, faceID);
    vec2 coords = vec2(uv.x * ppack.width + ppack.uOffset,
                       uv.y * ppack.height + ppack.vOffset);
    return texelFetch(data, ivec3(int(coords.x), int(coords.y), ppack.page), 0);
}

vec4 PTexLookupNearest(vec4 patchCoord,
                       int level,
                       sampler2DArray data,
                       isamplerBuffer packings)
{
    vec2 uv = patchCoord.xy;
    int faceID = int(patchCoord.w);
    PtexPacking ppack = getPtexPacking(packings, faceID, level);
    vec2 coords = vec2(uv.x * ppack.width + ppack.uOffset,
                       uv.y * ppack.height + ppack.vOffset);
    return texelFetch(data, ivec3(int(coords.x), int(coords.y), ppack.page), 0);
}

vec4 PTexLookupFast(vec4 patchCoord,
                    sampler2DArray data,
                    isamplerBuffer packings)
{
    vec2 uv = patchCoord.xy;
    int faceID = int(patchCoord.w);
    PtexPacking ppack = getPtexPacking(packings, faceID);

    ivec3 size = textureSize(data, 0);
    vec2 coords = vec2((uv.x * ppack.width + ppack.uOffset)/size.x,
                       (uv.y * ppack.height + ppack.vOffset)/size.y);
    return texture(data, vec3(coords.x, coords.y, ppack.page));
}

vec4 PTexLookup(vec4 patchCoord,
                int level,
                sampler2DArray data,
                isamplerBuffer packings)
{
    vec2 uv = patchCoord.xy;
    int faceID = int(patchCoord.w);
    PtexPacking ppack = getPtexPacking(packings, faceID, level);

    vec2 coords = vec2(uv.x * ppack.width + ppack.uOffset,
                       uv.y * ppack.height + ppack.vOffset);

    coords -= vec2(0.5, 0.5);

    int c0X = int(floor(coords.x));
    int c1X = int(ceil(coords.x));
    int c0Y = int(floor(coords.y));
    int c1Y = int(ceil(coords.y));

    float t = coords.x - float(c0X);
    float s = coords.y - float(c0Y);

    vec4 d0 = texelFetch(data, ivec3(c0X, c0Y, ppack.page), 0);
    vec4 d1 = texelFetch(data, ivec3(c0X, c1Y, ppack.page), 0);
    vec4 d2 = texelFetch(data, ivec3(c1X, c0Y, ppack.page), 0);
    vec4 d3 = texelFetch(data, ivec3(c1X, c1Y, ppack.page), 0);

    vec4 result = (1-t) * ((1-s)*d0 + s*d1) + t * ((1-s)*d2 + s*d3);

    return result;
}

// quadratic

void EvalQuadraticBSpline(float u, out float B[3], out float BU[3])
{
    B[0] = 0.5 * (u*u - 2.0*u + 1);
    B[1] = 0.5 + u - u*u;
    B[2] = 0.5 * u*u;

    BU[0] = u - 1.0;
    BU[1] = 1 - 2 * u;
    BU[2] = u;
}

vec4 PTexLookupQuadratic(out vec4 du,
                         out vec4 dv,
                         vec4 patchCoord,
                         int level,
                         sampler2DArray data,
                         isamplerBuffer packings)
{
    vec2 uv = patchCoord.xy;
    int faceID = int(patchCoord.w);
    PtexPacking ppack = getPtexPacking(packings, faceID, level);

    vec2 coords = vec2(uv.x * ppack.width + ppack.uOffset,
                       uv.y * ppack.height + ppack.vOffset);

    coords -= vec2(0.5, 0.5);

    int cX = int(round(coords.x));
    int cY = int(round(coords.y));

    float x = 0.5 - (float(cX) - coords.x);
    float y = 0.5 - (float(cY) - coords.y);

    // ---------------------------

    vec4 d[9];
    d[0] = texelFetch(data, ivec3(cX-1, cY-1, ppack.page), 0);
    d[1] = texelFetch(data, ivec3(cX-1, cY-0, ppack.page), 0);
    d[2] = texelFetch(data, ivec3(cX-1, cY+1, ppack.page), 0);
    d[3] = texelFetch(data, ivec3(cX-0, cY-1, ppack.page), 0);
    d[4] = texelFetch(data, ivec3(cX-0, cY-0, ppack.page), 0);
    d[5] = texelFetch(data, ivec3(cX-0, cY+1, ppack.page), 0);
    d[6] = texelFetch(data, ivec3(cX+1, cY-1, ppack.page), 0);
    d[7] = texelFetch(data, ivec3(cX+1, cY-0, ppack.page), 0);
    d[8] = texelFetch(data, ivec3(cX+1, cY+1, ppack.page), 0);

    float B[3], D[3];
    vec4 BUCP[3], DUCP[3];
    EvalQuadraticBSpline(y, B, D);

    for (int i = 0; i < 3; ++i) {
        BUCP[i] = vec4(0);
        DUCP[i] = vec4(0);
        for (int j = 0; j < 3; j++) {
            vec4 A = d[i*3+j];
            BUCP[i] += A * B[j];
            DUCP[i] += A * D[j];
        }
    }

    EvalQuadraticBSpline(x, B, D);

    vec4 result = vec4(0);
    du = vec4(0);
    dv = vec4(0);
    for (int i = 0; i < 3; ++i) {
        result += B[i] * BUCP[i];
        du += D[i] * BUCP[i];
        dv += B[i] * DUCP[i];
    }

    du *= ppack.width;
    dv *= ppack.height;

    return result;
}

vec4 PTexMipmapLookup(vec4 patchCoord,
                      float level,
                      sampler2DArray data,
                      isamplerBuffer packings)
{
#if defined(SEAMLESS_MIPMAP)
    // diff level
    int faceID = int(patchCoord.w);
    vec2 uv = patchCoord.xy;
    PtexPacking packing = getPtexPacking(packings, faceID);
    level += mix(mix(packing.adjSizeDiffs[0], packing.adjSizeDiffs[1], uv.x),
                 mix(packing.adjSizeDiffs[3], packing.adjSizeDiffs[2], uv.x),
                 uv.y);
#endif

    int levelm = int(floor(level));
    int levelp = int(ceil(level));
    float t = level - float(levelm);

    vec4 result = (1-t) * PTexLookup(patchCoord, levelm, data, packings)
        + t * PTexLookup(patchCoord, levelp, data, packings);
    return result;
}


vec4 PTexMipmapLookupQuadratic(out vec4 du,
                               out vec4 dv,
                               vec4 patchCoord,
                               float level,
                               sampler2DArray data,
                               isamplerBuffer packings)
{
#if defined(SEAMLESS_MIPMAP)
    // diff level
    int faceID = int(patchCoord.w);
    vec2 uv = patchCoord.xy;
    PtexPacking packing = getPtexPacking(packings, faceID);
    level += mix(mix(packing.adjSizeDiffs[0], packing.adjSizeDiffs[1], uv.x),
                 mix(packing.adjSizeDiffs[3], packing.adjSizeDiffs[2], uv.x),
                 uv.y);
#endif

    int levelm = int(floor(level));
    int levelp = int(ceil(level));
    float t = level - float(levelm);

    vec4 du0, du1, dv0, dv1;
    vec4 r0 = PTexLookupQuadratic(du0, dv0, patchCoord, levelm, data, packings);
    vec4 r1 = PTexLookupQuadratic(du1, dv1, patchCoord, levelp, data, packings);

    vec4 result = mix(r0, r1, t);
    du = mix(du0, du1, t);
    dv = mix(dv0, dv1, t);

    return result;
}

vec4 PTexMipmapLookupQuadratic(vec4 patchCoord,
                               float level,
                               sampler2DArray data,
                               isamplerBuffer packings)
{
    vec4 du, dv;
    return PTexMipmapLookupQuadratic(du, dv, patchCoord, level, data, packings);
}


vec4 PTexLookup(vec4 patchCoord,
                sampler2DArray data,
                isamplerBuffer packings)
{
    return PTexMipmapLookup(patchCoord, mipmapBias, data, packings);
}




#if    defined(DISPLACEMENT_HW_BILINEAR)        \
    || defined(DISPLACEMENT_BILINEAR)           \
    || defined(DISPLACEMENT_BIQUADRATIC)        \
    || defined(NORMAL_HW_SCREENSPACE)           \
    || defined(NORMAL_SCREENSPACE)              \
    || defined(NORMAL_BIQUADRATIC)              \
    || defined(NORMAL_BIQUADRATIC_WG)
uniform sampler2DArray textureDisplace_Data;
uniform isamplerBuffer textureDisplace_Packing;
#endif

#if defined(DISPLACEMENT_HW_BILINEAR) || defined(DISPLACEMENT_BILINEAR) || defined(DISPLACEMENT_BIQUADRATIC)

#undef OSD_DISPLACEMENT_CALLBACK
#define OSD_DISPLACEMENT_CALLBACK              \
    outpt.v.position =                         \
        displacement(outpt.v.position,         \
                     outpt.v.normal,           \
                     outpt.v.patchCoord);

vec4 displacement(vec4 position, vec3 normal, vec4 patchCoord)
{
#if defined(DISPLACEMENT_HW_BILINEAR)
    float disp = PTexLookupFast(patchCoord,
                                textureDisplace_Data,
                                textureDisplace_Packing).x;
#elif defined(DISPLACEMENT_BILINEAR)
    float disp = PTexMipmapLookup(patchCoord, mipmapBias,
                                  textureDisplace_Data,
                                  textureDisplace_Packing).x;
#elif defined(DISPLACEMENT_BIQUADRATIC)
    float disp = PTexMipmapLookupQuadratic(patchCoord, mipmapBias,
                                           textureDisplace_Data,
                                           textureDisplace_Packing).x;
#endif
    return position + vec4(disp * normal, 0) * displacementScale;
}
#endif

//--------------------------------------------------------------
// Vertex Shader
//--------------------------------------------------------------
#ifdef VERTEX_SHADER

layout (location=0) in vec4 position;
layout (location=1) in vec3 normal;

out block {
    OutputVertex v;
} outpt;

void main()
{
    outpt.v.position = ModelViewMatrix * position;
    outpt.v.normal = (ModelViewMatrix * vec4(normal, 0)).xyz;
}

#endif

//--------------------------------------------------------------
// Geometry Shader
//--------------------------------------------------------------
#ifdef GEOMETRY_SHADER

#ifdef PRIM_QUAD

    layout(lines_adjacency) in;

    layout(triangle_strip, max_vertices = 4) out;

    #define EDGE_VERTS 4

#endif // PRIM_QUAD

#ifdef  PRIM_TRI

    layout(triangles) in;

    layout(triangle_strip, max_vertices = 3) out;

    #define EDGE_VERTS 3

#endif // PRIM_TRI

#ifdef  PRIM_LINE
    layout(lines) in;

    layout(line_strip, max_vertices = 2) out;

    #define EDGE_VERTS 2

#endif // PRIM_LINE

in block {
    OutputVertex v;
} inpt[EDGE_VERTS];

out block {
    OutputVertex v;
    noperspective out vec4 edgeDistance;
} outpt;

// --------------------------------------

void emit(int index, vec4 position, vec3 normal, vec4 patchCoord)
{
    outpt.v.position = position;
    outpt.v.patchCoord = patchCoord;
    outpt.v.normal = normal;
    outpt.v.tangent = inpt[index].v.tangent;
    outpt.v.bitangent = inpt[index].v.bitangent;

#if defined(NORMAL_BIQUADRATIC_WG)
    outpt.v.Nu = inpt[index].v.Nu;
    outpt.v.Nv = inpt[index].v.Nv;
#endif

    gl_Position = ProjectionMatrix * outpt.v.position;
    EmitVertex();
}

const float VIEWPORT_SCALE = 1024.0; // XXXdyu

float edgeDistance(vec4 p, vec4 p0, vec4 p1)
{
    return VIEWPORT_SCALE *
        abs((p.x - p0.x) * (p1.y - p0.y) -
            (p.y - p0.y) * (p1.x - p0.x)) / length(p1.xy - p0.xy);
}

#if defined(PRIM_TRI) || defined(PRIM_QUAD)
void emit(int index, vec4 position, vec3 normal, vec4 patchCoord, vec4 edgeVerts[EDGE_VERTS])
{
    outpt.edgeDistance[0] =
        edgeDistance(edgeVerts[index], edgeVerts[0], edgeVerts[1]);
    outpt.edgeDistance[1] =
        edgeDistance(edgeVerts[index], edgeVerts[1], edgeVerts[2]);
#ifdef PRIM_TRI
    outpt.edgeDistance[2] =
        edgeDistance(edgeVerts[index], edgeVerts[2], edgeVerts[0]);
#endif
#ifdef PRIM_QUAD
    outpt.edgeDistance[2] =
        edgeDistance(edgeVerts[index], edgeVerts[2], edgeVerts[3]);
    outpt.edgeDistance[3] =
        edgeDistance(edgeVerts[index], edgeVerts[3], edgeVerts[0]);
#endif

    emit(index, position, normal, patchCoord);
}
#endif

// --------------------------------------

void main()
{
    gl_PrimitiveID = gl_PrimitiveIDIn;

#ifdef PRIM_QUAD
    vec4 patchCoord[4];
    vec4 position[4];
    vec3 normal[4];

    // need to generate patch coord for non-patch quads
    patchCoord[0] = GeneratePatchCoord(vec2(0, 0), gl_PrimitiveID);
    patchCoord[1] = GeneratePatchCoord(vec2(1, 0), gl_PrimitiveID);
    patchCoord[2] = GeneratePatchCoord(vec2(1, 1), gl_PrimitiveID);
    patchCoord[3] = GeneratePatchCoord(vec2(0, 1), gl_PrimitiveID);

#if defined(DISPLACEMENT_HW_BILINEAR) || defined(DISPLACEMENT_BILINEAR) || defined(DISPLACEMENT_BIQUADRATIC)
    position[0] = displacement(inpt[0].v.position, inpt[0].v.normal, patchCoord[0]);
    position[1] = displacement(inpt[1].v.position, inpt[1].v.normal, patchCoord[1]);
    position[2] = displacement(inpt[2].v.position, inpt[2].v.normal, patchCoord[2]);
    position[3] = displacement(inpt[3].v.position, inpt[3].v.normal, patchCoord[3]);
#else
    position[0] = inpt[0].v.position;
    position[1] = inpt[1].v.position;
    position[2] = inpt[2].v.position;
    position[3] = inpt[3].v.position;
#endif

#ifdef NORMAL_FACET
    // XXX: need to use vec C to get triangle normal.
    vec3 A = (position[0] - position[1]).xyz;
    vec3 B = (position[3] - position[1]).xyz;
    vec3 C = (position[2] - position[1]).xyz;
    normal[0] = normalize(cross(B, A));
    normal[1] = normal[0];
    normal[2] = normal[0];
    normal[3] = normal[0];
#else
    normal[0] = inpt[0].v.normal;
    normal[1] = inpt[1].v.normal;
    normal[2] = inpt[2].v.normal;
    normal[3] = inpt[3].v.normal;
#endif

#if defined(GEOMETRY_OUT_WIRE) || defined(GEOMETRY_OUT_LINE)
    vec4 edgeVerts[EDGE_VERTS];
    edgeVerts[0] = ProjectionMatrix * inpt[0].v.position;
    edgeVerts[1] = ProjectionMatrix * inpt[1].v.position;
    edgeVerts[2] = ProjectionMatrix * inpt[2].v.position;
    edgeVerts[3] = ProjectionMatrix * inpt[3].v.position;

    edgeVerts[0].xy /= edgeVerts[0].w;
    edgeVerts[1].xy /= edgeVerts[1].w;
    edgeVerts[2].xy /= edgeVerts[2].w;
    edgeVerts[3].xy /= edgeVerts[3].w;

    emit(0, position[0], normal[0], patchCoord[0], edgeVerts);
    emit(1, position[1], normal[1], patchCoord[1], edgeVerts);
    emit(3, position[3], normal[3], patchCoord[3], edgeVerts);
    emit(2, position[2], normal[2], patchCoord[2], edgeVerts);
#else
    outpt.edgeDistance[0] = 0;
    outpt.edgeDistance[1] = 0;
    outpt.edgeDistance[2] = 0;
    outpt.edgeDistance[3] = 0;
    emit(0, position[0], normal[0], patchCoord[0]);
    emit(1, position[1], normal[1], patchCoord[1]);
    emit(3, position[3], normal[3], patchCoord[3]);
    emit(2, position[2], normal[2], patchCoord[2]);
#endif
#endif // PRIM_QUAD

#ifdef PRIM_TRI
    vec4 position[3];
    vec4 patchCoord[3];
    vec3 normal[3];

    // patch coords are computed in tessellation shader
    patchCoord[0] = inpt[0].v.patchCoord;
    patchCoord[1] = inpt[1].v.patchCoord;
    patchCoord[2] = inpt[2].v.patchCoord;

    position[0] = inpt[0].v.position;
    position[1] = inpt[1].v.position;
    position[2] = inpt[2].v.position;

#ifdef NORMAL_FACET
    // emit flat normals for displaced surface
    vec3 A = (position[0] - position[1]).xyz;
    vec3 B = (position[2] - position[1]).xyz;
    normal[0] = normalize(cross(B, A));
    normal[1] = normal[0];
    normal[2] = normal[0];
#else
    normal[0] = inpt[0].v.normal;
    normal[1] = inpt[1].v.normal;
    normal[2] = inpt[2].v.normal;
#endif

#if defined(GEOMETRY_OUT_WIRE) || defined(GEOMETRY_OUT_LINE)
    vec4 edgeVerts[EDGE_VERTS];
    edgeVerts[0] = ProjectionMatrix * inpt[0].v.position;
    edgeVerts[1] = ProjectionMatrix * inpt[1].v.position;
    edgeVerts[2] = ProjectionMatrix * inpt[2].v.position;

    edgeVerts[0].xy /= edgeVerts[0].w;
    edgeVerts[1].xy /= edgeVerts[1].w;
    edgeVerts[2].xy /= edgeVerts[2].w;

    emit(0, position[0], normal[0], patchCoord[0], edgeVerts);
    emit(1, position[1], normal[1], patchCoord[1], edgeVerts);
    emit(2, position[2], normal[2], patchCoord[2], edgeVerts);
#else
    emit(0, position[0], normal[0], patchCoord[0]);
    emit(1, position[1], normal[1], patchCoord[1]);
    emit(2, position[2], normal[2], patchCoord[2]);
#endif
#endif // PRIM_TRI

#ifdef PRIM_LINE
    emit(0, inpt[0].v.position, inpt[0].v.normal, vec4(0));
    emit(1, inpt[1].v.position, inpt[1].v.normal, vec4(0));
#endif

    EndPrimitive();
}

#endif
//--------------------------------------------------------------
// Fragment Shader
//--------------------------------------------------------------
#ifdef FRAGMENT_SHADER

in block {
    OutputVertex v;
    noperspective in vec4 edgeDistance;
} inpt;

out vec4 outColor;

#if defined(COLOR_PTEX_NEAREST) || defined(COLOR_PTEX_HW_BILINEAR) || defined(COLOR_PTEX_BILINEAR) || defined(COLOR_PTEX_BIQUADRATIC)
uniform sampler2DArray textureImage_Data;
uniform isamplerBuffer textureImage_Packing;
#endif

#ifdef USE_PTEX_OCCLUSION
uniform sampler2DArray textureOcclusion_Data;
uniform isamplerBuffer textureOcclusion_Packing;
#endif

#ifdef USE_PTEX_SPECULAR
uniform sampler2DArray textureSpecular_Data;
uniform isamplerBuffer textureSpecular_Packing;
#endif

#define NUM_LIGHTS 2

struct LightSource {
    vec4 position;
    vec4 ambient;
    vec4 diffuse;
    vec4 specular;
};

layout(std140) uniform Lighting {
    LightSource lightSource[NUM_LIGHTS];
};

#if defined COLOR_PATCHTYPE
uniform vec4 overrideColor;
#endif

#if defined(NORMAL_HW_SCREENSPACE) || defined(NORMAL_SCREENSPACE)

vec3
perturbNormalFromDisplacement(vec3 position, vec3 normal, vec4 patchCoord)
{
    // by Morten S. Mikkelsen
    // http://jbit.net/~sparky/sfgrad_bump/mm_sfgrad_bump.pdf
    // slightly modified for ptex guttering

    vec3 vSigmaS = dFdx(position);
    vec3 vSigmaT = dFdy(position);
    vec3 vN = normal;
    vec3 vR1 = cross(vSigmaT, vN);
    vec3 vR2 = cross(vN, vSigmaS);
    float fDet = dot(vSigmaS, vR1);
#if 0
    // not work well with ptex
    float dBs = dFdx(disp);
    float dBt = dFdy(disp);
#else
    vec2 texDx = dFdx(patchCoord.xy);
    vec2 texDy = dFdy(patchCoord.xy);
    
    // limit forward differencing to the width of ptex gutter
    const float resolution = 128.0;
    float d = min(1, (0.5/resolution)/max(length(texDx), length(texDy)));
    
    vec4 STll = patchCoord;
    vec4 STlr = patchCoord + d * vec4(texDx.x, texDx.y, 0, 0);
    vec4 STul = patchCoord + d * vec4(texDy.x, texDy.y, 0, 0);
#if defined NORMAL_HW_SCREENSPACE
    float Hll = PTexLookupFast(STll, textureDisplace_Data, textureDisplace_Packing).x * displacementScale;
    float Hlr = PTexLookupFast(STlr, textureDisplace_Data, textureDisplace_Packing).x * displacementScale;
    float Hul = PTexLookupFast(STul, textureDisplace_Data, textureDisplace_Packing).x * displacementScale;
#elif defined NORMAL_SCREENSPACE
    float Hll = PTexMipmapLookup(STll, mipmapBias, textureDisplace_Data, textureDisplace_Packing).x * displacementScale;
    float Hlr = PTexMipmapLookup(STlr, mipmapBias, textureDisplace_Data, textureDisplace_Packing).x * displacementScale;
    float Hul = PTexMipmapLookup(STul, mipmapBias, textureDisplace_Data, textureDisplace_Packing).x * displacementScale;
#endif
    float dBs = (Hlr - Hll)/d;
    float dBt = (Hul - Hll)/d;
#endif
    
    vec3 vSurfGrad = sign(fDet) * (dBs * vR1 + dBt * vR2);
    return normalize(abs(fDet) * vN - vSurfGrad);
}
#endif // NORMAL_SCREENSPACE

uniform sampler2D diffuseEnvironmentMap;
uniform sampler2D specularEnvironmentMap;

vec4 getEnvironmentHDR(sampler2D sampler, vec3 dir)
{
    dir = (ModelViewInverseMatrix * vec4(dir, 0)).xyz;
    vec2 uv = vec2((atan(dir.x,dir.z)/3.1415926535897+1)*0.5, (1-dir.y)*0.5);
    vec4 tex = texture(sampler, uv);
    tex = vec4(pow(tex.xyz, vec3(0.4545)), 1);
    return tex;
}

vec4
lighting(vec4 texColor, vec3 Peye, vec3 Neye, float occ)
{
    vec4 color = vec4(0);
    vec3 n = Neye;

    for (int i = 0; i < NUM_LIGHTS; ++i) {

        vec4 Plight = lightSource[i].position;
        vec3 l = (Plight.w == 0.0)
                    ? normalize(Plight.xyz) : normalize(Plight.xyz - Peye);

        vec3 h = normalize(l + vec3(0,0,1));    // directional viewer

        float d = max(0.0, dot(n, l));
        float s = pow(max(0.0, dot(n, h)), 64.0f);

        color += (1.0-occ) * ((lightSource[i].ambient +
                               d * lightSource[i].diffuse) * texColor +
                              s * lightSource[i].specular);
    }

    color.a = 1;

    return color;
}

vec4
edgeColor(vec4 Cfill, vec4 edgeDistance)
{
#if defined(GEOMETRY_OUT_WIRE) || defined(GEOMETRY_OUT_LINE)
#ifdef PRIM_TRI
    float d =
        min(inpt.edgeDistance[0], min(inpt.edgeDistance[1], inpt.edgeDistance[2]));
#endif
#ifdef PRIM_QUAD
    float d =
        min(min(inpt.edgeDistance[0], inpt.edgeDistance[1]),
            min(inpt.edgeDistance[2], inpt.edgeDistance[3]));
#endif
#ifdef PRIM_LINE
    float d = 0;
#endif
    vec4 Cedge = vec4(1.0, 1.0, 0.0, 1.0);
    float p = exp2(-2 * d * d);

#if defined(GEOMETRY_OUT_WIRE)
    if (p < 0.25) discard;
#endif

    Cfill.rgb = mix(Cfill.rgb, Cedge.rgb, p);
#endif
    return Cfill;
}

#if defined(PRIM_QUAD) || defined(PRIM_TRI)
void
main()
{
    // ------------ normal ---------------

#if defined(NORMAL_HW_SCREENSPACE) || defined(NORMAL_SCREENSPACE)
    vec3 normal = perturbNormalFromDisplacement(inpt.v.position.xyz,
                                                inpt.v.normal,
                                                inpt.v.patchCoord);
#elif defined(NORMAL_BIQUADRATIC) || defined(NORMAL_BIQUADRATIC_WG)
    vec4 du, dv;
    vec4 disp = PTexMipmapLookupQuadratic(du, dv, inpt.v.patchCoord,
                                          mipmapBias,
                                          textureDisplace_Data,
                                          textureDisplace_Packing);

    disp *= displacementScale;
    du *= displacementScale;
    dv *= displacementScale;

    vec3 n = normalize(cross(inpt.v.tangent, inpt.v.bitangent));
    vec3 tangent = inpt.v.tangent + n * du.x;
    vec3 bitangent = inpt.v.bitangent + n * dv.x;

#if defined(NORMAL_BIQUADRATIC_WG)
    tangent += inpt.v.Nu * disp.x;
    bitangent += inpt.v.Nv * disp.x;
#endif

    vec3 normal = normalize(cross(tangent, bitangent));
#else
    vec3 normal = inpt.v.normal;
#endif

    // ------------ color ---------------

#if defined COLOR_PTEX_NEAREST
    vec4 texColor = PTexLookupNearest(inpt.v.patchCoord,
                                      textureImage_Data,
                                      textureImage_Packing);
#elif defined COLOR_PTEX_HW_BILINEAR
    vec4 texColor = PTexLookupFast(inpt.v.patchCoord,
                                   textureImage_Data,
                                   textureImage_Packing);
#elif defined COLOR_PTEX_BILINEAR
    vec4 texColor = PTexMipmapLookup(inpt.v.patchCoord, mipmapBias,
                                     textureImage_Data,
                                     textureImage_Packing);
#elif defined COLOR_PTEX_BIQUADRATIC
    vec4 texColor = PTexMipmapLookupQuadratic(inpt.v.patchCoord, mipmapBias,
                                              textureImage_Data,
                                              textureImage_Packing);
#elif defined COLOR_PATCHTYPE
    vec4 texColor = edgeColor(lighting(overrideColor, inpt.v.position.xyz, normal, 0), inpt.edgeDistance);
    outColor = texColor;
    return;
#elif defined COLOR_PATCHCOORD
    vec4 texColor = edgeColor(lighting(inpt.v.patchCoord, inpt.v.position.xyz, normal, 0), inpt.edgeDistance);
    outColor = texColor;
    return;
#elif defined COLOR_NORMAL
    vec4 texColor = edgeColor(vec4(normal, 1), inpt.edgeDistance);
    outColor = texColor;
    return;
#else // COLOR_NONE
    vec4 texColor = vec4(0.5);
#endif

    // gamma correct?
    // texColor = vec4(pow(texColor.xyz, vec3(0.4545)), 1);

    // ------------ occlusion ---------------

#ifdef USE_PTEX_OCCLUSION
    float occ = PTexLookup(inpt.v.patchCoord,
                           textureOcclusion_Data,
                           textureOcclusion_Packing).x;
#else
    float occ = 0.0;
#endif

    // ------------ specular ---------------

#ifdef USE_PTEX_SPECULAR
    float specular = PTexLookup(inpt.v.patchCoord,
                                textureSpecular_Data,
                                textureSpecular_Packing).x;
#else
    float specular = 1.0;
#endif

    // ------------ lighting ---------------

#ifdef USE_IBL
    vec4 a = vec4(0, 0, 0, 1); //ambientColor;
    vec4 d = getEnvironmentHDR(diffuseEnvironmentMap, normal) * 1.4;
    vec3 eye = normalize(inpt.v.position.xyz - vec3(0,0,0));
    vec3 reflect = reflect(eye, normal);
    vec4 s = getEnvironmentHDR(specularEnvironmentMap, reflect);
    const float fresnelBias = 0;
    const float fresnelScale = 1.0;
    const float fresnelPower = 4.0;
    float fresnel = fresnelBias + fresnelScale * pow(1.0+dot(normal,eye), fresnelPower);

    a *= (1.0-occ);
    d *= (1.0-occ);
    s *= min(specular, (1.0-occ)) * fresnel;

    vec4 Cf = (a + d) * texColor + s * 0.5;
#else
    vec4 Cf = lighting(texColor, inpt.v.position.xyz, normal, occ);
#endif

    // ------------ wireframe ---------------

    outColor = edgeColor(Cf, inpt.edgeDistance);
}
#endif //PRIM_TRI || PRIM_QUAD

#if defined(PRIM_LINE)
void
main()
{
    outColor = vec4(0, 1, 0, 1);
}
#endif

#endif
