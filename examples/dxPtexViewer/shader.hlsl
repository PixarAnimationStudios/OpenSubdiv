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
#line 24
struct OutputPointVertex {
    float4 positionOut : SV_Position;
};

cbuffer Config : register( b3 ) {
    float displacementScale;
    float mipmapBias;
};

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

PtexPacking getPtexPacking(Buffer<int> packings, int faceID)
{
    PtexPacking packing;
    packing.page    = packings[faceID*6+0].x;
    packing.nMipmap = packings[faceID*6+1].x;
    packing.uOffset = packings[faceID*6+2].x;
    packing.vOffset = packings[faceID*6+3].x;
    int wh          = packings[faceID*6+5].x;
    packing.width   = 1 << (wh >> 8);
    packing.height  = 1 << (wh & 0xff);

    int adjSizeDiffs = packings[faceID*6+4].x;
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

PtexPacking getPtexPacking(Buffer<int> packings, int faceID, int level)
{
    PtexPacking packing;
    packing.page    = packings[faceID*6+0].x;
    packing.nMipmap = packings[faceID*6+1].x;
    packing.uOffset = packings[faceID*6+2].x;
    packing.vOffset = packings[faceID*6+3].x;
    int wh          = packings[faceID*6+5].x;
    int w = wh >> 8;
    int h = wh & 0xff;

    // clamp max level
    level = min(level, packing.nMipmap);

    packing.uOffset += computeMipmapOffsetU(w, level);
    packing.vOffset += computeMipmapOffsetV(h, level);
    packing.width = 1 << (w-level);
    packing.height = 1 << (h-level);

    return packing;

}

float4 PTexLookupNearest(float4 patchCoord,
                         Texture2DArray data,
                         Buffer<int> packings)
{
    float2 uv = patchCoord.xy;
    int faceID = patchCoord.w;
    PtexPacking ppack = getPtexPacking(packings, faceID);
    float2 coords = float2(uv.x * ppack.width + ppack.uOffset,
                           uv.y * ppack.height + ppack.vOffset);
    return data[int3(int(coords.x), int(coords.y), ppack.page)];
}

float4 PTexLookup(float4 patchCoord,
                  int level,
                  Texture2DArray data,
                  Buffer<int> packings)
{
    float2 uv = patchCoord.xy;
    int faceID = int(patchCoord.w);
    PtexPacking ppack = getPtexPacking(packings, faceID, level);

    float2 coords = float2(uv.x * ppack.width + ppack.uOffset,
                           uv.y * ppack.height + ppack.vOffset);

    coords -= float2(0.5, 0.5);

    int c0X = int(floor(coords.x));
    int c1X = int(ceil(coords.x));
    int c0Y = int(floor(coords.y));
    int c1Y = int(ceil(coords.y));

    float t = coords.x - float(c0X);
    float s = coords.y - float(c0Y);

    float4 d0 = data[int3(c0X, c0Y, ppack.page)];
    float4 d1 = data[int3(c0X, c1Y, ppack.page)];
    float4 d2 = data[int3(c1X, c0Y, ppack.page)];
    float4 d3 = data[int3(c1X, c1Y, ppack.page)];

    float4 result = (1-t) * ((1-s)*d0 + s*d1) + t * ((1-s)*d2 + s*d3);

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

float4 PTexLookupQuadratic(out float4 du,
                           out float4 dv,
                           float4 patchCoord,
                           int level,
                           Texture2DArray data,
                           Buffer<int> packings)
{
    float2 uv = patchCoord.xy;
    int faceID = int(patchCoord.w);
    PtexPacking ppack = getPtexPacking(packings, faceID, level);

    float2 coords = float2(uv.x * ppack.width + ppack.uOffset,
                           uv.y * ppack.height + ppack.vOffset);

    coords -= float2(0.5, 0.5);

    int cX = int(round(coords.x));
    int cY = int(round(coords.y));

    float x = 0.5 - (float(cX) - coords.x);
    float y = 0.5 - (float(cY) - coords.y);

    // ---------------------------

    float4 d[9];
    d[0] = data[int3(cX-1, cY-1, ppack.page)];
    d[1] = data[int3(cX-1, cY-0, ppack.page)];
    d[2] = data[int3(cX-1, cY+1, ppack.page)];
    d[3] = data[int3(cX-0, cY-1, ppack.page)];
    d[4] = data[int3(cX-0, cY-0, ppack.page)];
    d[5] = data[int3(cX-0, cY+1, ppack.page)];
    d[6] = data[int3(cX+1, cY-1, ppack.page)];
    d[7] = data[int3(cX+1, cY-0, ppack.page)];
    d[8] = data[int3(cX+1, cY+1, ppack.page)];

    float B[3], D[3];
    float4 BUCP[3], DUCP[3];
    EvalQuadraticBSpline(y, B, D);

    for (int i = 0; i < 3; ++i) {
        BUCP[i] = float4(0, 0, 0, 0);
        DUCP[i] = float4(0, 0, 0, 0);
        for (int j = 0; j < 3; j++) {
            float4 A = d[i*3+j];
            BUCP[i] += A * B[j];
            DUCP[i] += A * D[j];
        }
    }

    EvalQuadraticBSpline(x, B, D);

    float4 result = float4(0, 0, 0, 0);
    du = float4(0, 0, 0, 0);
    dv = float4(0, 0, 0, 0);
    for (int i = 0; i < 3; ++i) {
        result += B[i] * BUCP[i];
        du += D[i] * BUCP[i];
        dv += B[i] * DUCP[i];
    }

    du *= ppack.width;
    dv *= ppack.height;

    return result;
}

float4 PTexMipmapLookup(float4 patchCoord,
                        float level,
                        Texture2DArray data,
                        Buffer<int> packings)
{
#if defined(SEAMLESS_MIPMAP)
    // diff level
    int faceID = int(patchCoord.w);
    float2 uv = patchCoord.xy;
    PtexPacking packing = getPtexPacking(packings, faceID);
    level += lerp(lerp(packing.adjSizeDiffs[0], packing.adjSizeDiffs[1], uv.x),
                  lerp(packing.adjSizeDiffs[3], packing.adjSizeDiffs[2], uv.x),
                  uv.y);
#endif

    int levelm = int(floor(level));
    int levelp = int(ceil(level));
    float t = level - float(levelm);

    float4 result = (1-t) * PTexLookup(patchCoord, levelm, data, packings)
        + t * PTexLookup(patchCoord, levelp, data, packings);
    return result;
}

float4 PTexMipmapLookupQuadratic(out float4 du,
                                 out float4 dv,
                                 float4 patchCoord,
                                 float level,
                                 Texture2DArray data,
                                 Buffer<int> packings)
{
#if defined(SEAMLESS_MIPMAP)
    // diff level
    int faceID = int(patchCoord.w);
    float2 uv = patchCoord.xy;
    PtexPacking packing = getPtexPacking(packings, faceID);
    level += lerp(lerp(packing.adjSizeDiffs[0], packing.adjSizeDiffs[1], uv.x),
                  lerp(packing.adjSizeDiffs[3], packing.adjSizeDiffs[2], uv.x),
                  uv.y);
#endif

    int levelm = int(floor(level));
    int levelp = int(ceil(level));
    float t = level - float(levelm);

    float4 du0, du1, dv0, dv1;
    float4 r0 = PTexLookupQuadratic(du0, dv0, patchCoord, levelm, data, packings);
    float4 r1 = PTexLookupQuadratic(du1, dv1, patchCoord, levelp, data, packings);

    float4 result = lerp(r0, r1, t);
    du = lerp(du0, du1, t);
    dv = lerp(dv0, dv1, t);

    return result;
}

float4 PTexMipmapLookupQuadratic(float4 patchCoord,
                                 float level,
                                 Texture2DArray data,
                                 Buffer<int> packings)
{
    float4 du, dv;
    return PTexMipmapLookupQuadratic(du, dv, patchCoord, level, data, packings);
}

// ---------------------------------------------------------------------------

#if    defined(DISPLACEMENT_HW_BILINEAR)        \
    || defined(DISPLACEMENT_BILINEAR)           \
    || defined(DISPLACEMENT_BIQUADRATIC)        \
    || defined(NORMAL_HW_SCREENSPACE)           \
    || defined(NORMAL_SCREENSPACE)              \
    || defined(NORMAL_BIQUADRATIC)              \
    || defined(NORMAL_BIQUADRATIC_WG)

Texture2DArray textureDisplace_Data : register(t6);
Buffer<int> textureDisplace_Packing : register(t7);
#endif

#if defined(DISPLACEMENT_HW_BILINEAR) \
    || defined(DISPLACEMENT_BILINEAR) \
    || defined(DISPLACEMENT_BIQUADRATIC)

#undef OSD_DISPLACEMENT_CALLBACK
#define OSD_DISPLACEMENT_CALLBACK              \
    output.position =                          \
        displacement(output.position,          \
                     output.normal,            \
                     output.patchCoord);

float4 displacement(float4 position, float3 normal, float4 patchCoord)
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
    return position + float4(disp*normal, 0) * displacementScale;
}
#endif



// ---------------------------------------------------------------------------
//  Vertex Shader
// ---------------------------------------------------------------------------

void vs_main( in InputVertex input,
              out OutputVertex output )
{
    output.positionOut = mul(ModelViewProjectionMatrix, input.position);
    output.position = mul(ModelViewMatrix, input.position);
    output.normal = mul(ModelViewMatrix,float4(input.normal, 0)).xyz;
}

// ---------------------------------------------------------------------------
//  Geometry Shader
// ---------------------------------------------------------------------------

OutputVertex
outputVertex(OutputVertex input, float3 normal)
{
    OutputVertex v = input;
    v.normal = normal;
    return v;
}

#if defined(GEOMETRY_OUT_WIRE) || defined(GEOMETRY_OUT_LINE)
#ifdef PRIM_TRI
    #define EDGE_VERTS 3
#endif
#ifdef PRIM_QUAD
    #define EDGE_VERTS 4
#endif

static float VIEWPORT_SCALE = 1024.0; // XXXdyu

float edgeDistance(float2 p, float2 p0, float2 p1)
{
    return VIEWPORT_SCALE *
        abs((p.x - p0.x) * (p1.y - p0.y) -
            (p.y - p0.y) * (p1.x - p0.x)) / length(p1.xy - p0.xy);
}

OutputVertex
outputWireVertex(OutputVertex input, float3 normal,
                 int index, float2 edgeVerts[EDGE_VERTS])
{
    OutputVertex v = input;
    v.normal = normal;

    v.edgeDistance[0] =
        edgeDistance(edgeVerts[index], edgeVerts[0], edgeVerts[1]);
    v.edgeDistance[1] =
        edgeDistance(edgeVerts[index], edgeVerts[1], edgeVerts[2]);
#ifdef PRIM_TRI
    v.edgeDistance[2] =
        edgeDistance(edgeVerts[index], edgeVerts[2], edgeVerts[0]);
#endif
#ifdef PRIM_QUAD
    v.edgeDistance[2] =
        edgeDistance(edgeVerts[index], edgeVerts[2], edgeVerts[3]);
    v.edgeDistance[3] =
        edgeDistance(edgeVerts[index], edgeVerts[3], edgeVerts[0]);
#endif

    return v;
}
#endif

#ifdef PRIM_QUAD
[maxvertexcount(6)]
void gs_main( lineadj OutputVertex input[4],
              inout TriangleStream<OutputVertex> triStream )
{
    float3 A = (input[0].position - input[1].position).xyz;
    float3 B = (input[3].position - input[1].position).xyz;
    float3 C = (input[2].position - input[1].position).xyz;

    float3 n0 = normalize(cross(B, A));

    triStream.Append(outputVertex(input[0], n0));
    triStream.Append(outputVertex(input[1], n0));
    triStream.Append(outputVertex(input[3], n0));
    triStream.RestartStrip();
    triStream.Append(outputVertex(input[3], n0));
    triStream.Append(outputVertex(input[1], n0));
    triStream.Append(outputVertex(input[2], n0));
    triStream.RestartStrip();
}
#else // PRIM_TRI
[maxvertexcount(3)]
void gs_main( triangle OutputVertex input[3],
              inout TriangleStream<OutputVertex> triStream )
{
    float4 position[3];
    float4 patchCoord[3];
    float3 normal[3];

    // patch coords are computed in tessellation shader
    patchCoord[0] = input[0].patchCoord;
    patchCoord[1] = input[1].patchCoord;
    patchCoord[2] = input[2].patchCoord;

    position[0] = input[0].position;
    position[1] = input[1].position;
    position[2] = input[2].position;

#ifdef NORMAL_FACET
    // emit flat normals for displaced surface
    float3 A = (position[0] - position[1]).xyz;
    float3 B = (position[2] - position[1]).xyz;
    normal[0]= normalize(cross(B, A));
    normal[1] = normal[0];
    normal[2] = normal[0];
#else
    normal[0] = input[0].normal;
    normal[1] = input[1].normal;
    normal[2] = input[2].normal;
#endif

#if defined(GEOMETRY_OUT_WIRE) || defined(GEOMETRY_OUT_LINE)
    float2 edgeVerts[3];
    edgeVerts[0] = input[0].positionOut.xy / input[0].positionOut.w;
    edgeVerts[1] = input[1].positionOut.xy / input[1].positionOut.w;
    edgeVerts[2] = input[2].positionOut.xy / input[2].positionOut.w;

    triStream.Append(outputWireVertex(input[0], normal[0], 0, edgeVerts));
    triStream.Append(outputWireVertex(input[1], normal[1], 1, edgeVerts));
    triStream.Append(outputWireVertex(input[2], normal[2], 2, edgeVerts));
#else
    triStream.Append(outputVertex(input[0], normal[0]));
    triStream.Append(outputVertex(input[1], normal[1]));
    triStream.Append(outputVertex(input[2], normal[2]));
#endif
}

#endif


// ---------------------------------------------------------------------------
//  Lighting
// ---------------------------------------------------------------------------

#define NUM_LIGHTS 2

struct LightSource {
    float4 position;
    float4 ambient;
    float4 diffuse;
    float4 specular;
};

cbuffer Lighting : register( b2 ) {
    LightSource lightSource[NUM_LIGHTS];
};

float4
lighting(float4 texColor, float3 Peye, float3 Neye, float occ)
{
    float4 color = float4(0.0, 0.0, 0.0, 0.0);
    float3 n = Neye;

    for (int i = 0; i < NUM_LIGHTS; ++i) {

        float4 Plight = lightSource[i].position;
        float3 l = (Plight.w == 0.0)
                    ? normalize(Plight.xyz) : normalize(Plight.xyz - Peye);

        float3 h = normalize(l + float3(0,0,1));    // directional viewer

        float d = max(0.0, dot(n, l));
        float s = pow(max(0.0, dot(n, h)), 64.0f);

        color += (1.0 - occ) * ((lightSource[i].ambient +
                                 d * lightSource[i].diffuse) * texColor +
                                s * lightSource[i].specular);
    }

    color.a = 1.0;
    return color;
}

// ---------------------------------------------------------------------------
//  Pixel Shader
// ---------------------------------------------------------------------------

float4
edgeColor(float4 Cfill, float4 edgeDistance)
{
#if defined(GEOMETRY_OUT_WIRE) || defined(GEOMETRY_OUT_LINE)
#ifdef PRIM_TRI
    float d =
        min(edgeDistance[0], min(edgeDistance[1], edgeDistance[2]));
#endif
#ifdef PRIM_QUAD
    float d =
        min(min(edgeDistance[0], edgeDistance[1]),
            min(edgeDistance[2], edgeDistance[3]));
#endif
    float4 Cedge = float4(1.0, 1.0, 0.0, 1.0);
    float p = exp2(-2 * d * d);

#if defined(GEOMETRY_OUT_WIRE)
    if (p < 0.25) discard;
#endif

    Cfill.rgb = lerp(Cfill.rgb, Cedge.rgb, p);
#endif
    return Cfill;
}

// ---------------------------------------------------------------------------
//  Pixel Shader
// ---------------------------------------------------------------------------

#if defined(COLOR_PTEX_NEAREST) ||     \
    defined(COLOR_PTEX_HW_BILINEAR) || \
    defined(COLOR_PTEX_BILINEAR) ||    \
    defined(COLOR_PTEX_BIQUADRATIC)
Texture2DArray textureImage_Data : register(t4);
Buffer<int> textureImage_Packing : register(t5);
#endif

#ifdef USE_PTEX_OCCLUSION
Texture2DArray textureOcclusion_Data : register(t8);
Buffer<int> textureOcclusion_Packing : register(t9);
#endif

#ifdef USE_PTEX_SPECULAR
Texture2DArray textureSpecular_Data : register(t10);
Buffer<int> textureSpecular_Packing : register(t11);
#endif

void
ps_main(in OutputVertex input,
        out float4 outColor : SV_Target )
{
    // ------------ normal ---------------
#if defined(NORMAL_HW_SCREENSPACE) || defined(NORMAL_SCREENSPACE)
    float3 normal = perturbNormalFromDisplacement(input.position.xyz,
                                                  input.normal,
                                                  input.patchCoord);
#elif defined(NORMAL_BIQUADRATIC) || defined(NORMAL_BIQUADRATIC_WG)
    float4 du, dv;
    float4 disp = PTexMipmapLookupQuadratic(du, dv, input.patchCoord,
                                            mipmapBias,
                                            textureDisplace_Data,
                                            textureDisplace_Packing);

    disp *= displacementScale;
    du *= displacementScale;
    dv *= displacementScale;

    float3 n = normalize(cross(input.tangent, input.bitangent));
    float3 tangent = input.tangent + n * du.x;
    float3 bitangent = input.bitangent + n * dv.x;

#if defined(NORMAL_BIQUADRATIC_WG)
    tangent += input.Nu * disp.x;
    bitangent += input.Nv * disp.x;
#endif

    float3 normal = normalize(cross(tangent, bitangent));
#else
    float3 normal = input.normal;
#endif

    // ------------ color ---------------
#if defined(COLOR_PTEX_NEAREST)
    float4 texColor = PTexLookupNearest(input.patchCoord,
                                        textureImage_Data,
                                        textureImage_Packing);
#elif defined(COLOR_PTEX_HW_BILINEAR)
    float4 texColor = PTexLookupFast(input.patchCoord,
                                   textureImage_Data,
                                   textureImage_Packing);
#elif defined(COLOR_PTEX_BILINEAR)
    float4 texColor = PTexMipmapLookup(input.patchCoord, mipmapBias,
                                     textureImage_Data,
                                     textureImage_Packing);
#elif defined(COLOR_PTEX_BIQUADRATIC)
    float4 texColor = PTexMipmapLookupQuadratic(input.patchCoord, mipmapBias,
                                              textureImage_Data,
                                              textureImage_Packing);
#elif defined(COLOR_PATCHTYPE)
    float4 texColor = edgeColor(lighting(overrideColor, input.position.xyz, normal, 0),
                                input.edgeDistance);
    outColor = texColor;
    return;
#elif defined(COLOR_PATCHCOORD)
    float4 texColor = edgeColor(lighting(input.patchCoord, input.position.xyz, normal, 0),
                                input.edgeDistance);
    outColor = texColor;
    return;
#elif defined(COLOR_NORMAL)
    float4 texColor = edgeColor(float4(normal.x, normal.y, normal.z, 1),
                                input.edgeDistance);
    outColor = texColor;
    return;
#else // COLOR_NONE
    float4 texColor = float4(0.5, 0.5, 0.5, 1);
#endif

    // ------------ occlusion ---------------

#ifdef USE_PTEX_OCCLUSION
    float occ = PTexLookup(input.patchCoord,
                           textureOcclusion_Data,
                           textureOcclusion_Packing).x;
#else
    float occ = 0.0;
#endif

    // ------------ specular ---------------

#ifdef USE_PTEX_SPECULAR
    float specular = PTexLookup(input.patchCoord,
                                textureSpecular_Data,
                                textureSpecular_Packing).x;
#else
    float specular = 1.0;
#endif
    // ------------ lighting ---------------
    float4 Cf = lighting(texColor, input.position.xyz, normal, occ);

    // ------------ wireframe ---------------
    outColor = edgeColor(Cf, input.edgeDistance);
}
