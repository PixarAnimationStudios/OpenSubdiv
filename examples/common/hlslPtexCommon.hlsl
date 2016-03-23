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
// Ptex.Common
//----------------------------------------------------------

struct PtexPacking {
    int page;
    int nMipmap;
    int uOffset;
    int vOffset;
    int adjSizeDiffs[4];
    int width;
    int height;
};

PtexPacking getPtexPacking(Buffer<uint> packings, int faceID)
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

PtexPacking getPtexPacking(Buffer<uint> packings, int faceID, int level)
{
    PtexPacking packing;
    packing.page    = packings[faceID*6+0].x;
    packing.nMipmap = packings[faceID*6+1].x;
    packing.uOffset = packings[faceID*6+2].x;
    packing.vOffset = packings[faceID*6+3].x;
    //int sizeDiffs   = packings[faceID*6+4].x;
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

void evalQuadraticBSpline(float u, out float B[3], out float BU[3])
{
    B[0] = 0.5 * (u*u - 2.0*u + 1);
    B[1] = 0.5 + u - u*u;
    B[2] = 0.5 * u*u;

    BU[0] = u - 1.0;
    BU[1] = 1 - 2 * u;
    BU[2] = u;
}

// ----------------------------------------------------------------------------
// Non-Mipmap Lookups
// ----------------------------------------------------------------------------

float4 PtexLookupNearest(float4 patchCoord,
                         Texture2DArray data,
                         Buffer<uint> packings)
{
    float2 uv = clamp(patchCoord.xy, float2(0,0), float2(1,1));
    int faceID = int(patchCoord.w);
    PtexPacking ppack = getPtexPacking(packings, faceID);
    float2 coords = float2(uv.x * ppack.width + ppack.uOffset,
                           uv.y * ppack.height + ppack.vOffset);
    return data[int3(int(coords.x), int(coords.y), ppack.page)];
}

float4 PtexLookupNearest(float4 patchCoord,
                         int level,
                         Texture2DArray data,
                         Buffer<uint> packings)
{
    float2 uv = clamp(patchCoord.xy, float2(0,0), float2(1,1));
    int faceID = int(patchCoord.w);
    PtexPacking ppack = getPtexPacking(packings, faceID, level);
    float2 coords = float2(uv.x * ppack.width + ppack.uOffset,
                           uv.y * ppack.height + ppack.vOffset);
    return data[int3(int(coords.x), int(coords.y), ppack.page)];
}

float4 PtexLookup(float4 patchCoord,
                  int level,
                  Texture2DArray data,
                  Buffer<uint> packings)
{
    float2 uv = clamp(patchCoord.xy, float2(0,0), float2(1,1));
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

float4 PtexLookupQuadratic(out float4 du,
                           out float4 dv,
                           float4 patchCoord,
                           int level,
                           Texture2DArray data,
                           Buffer<uint> packings)
{
    float2 uv = clamp(patchCoord.xy, float2(0,0), float2(1,1));
    int faceID = int(patchCoord.w);
    PtexPacking ppack = getPtexPacking(packings, faceID, level);

    float2 coords = float2(uv.x * ppack.width + ppack.uOffset,
                           uv.y * ppack.height + ppack.vOffset);

    coords -= float2(0.5, 0.5);

    int cX = int(round(coords.x));
    int cY = int(round(coords.y));

    float x = 0.5 - (float(cX) - coords.x);
    float y = 0.5 - (float(cY) - coords.y);

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
    float4 BUCP[3] = {float4(0,0,0,0), float4(0,0,0,0), float4(0,0,0,0)},
           DUCP[3] = {float4(0,0,0,0), float4(0,0,0,0), float4(0,0,0,0)};

    evalQuadraticBSpline(y, B, D);

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; j++) {
            float4 A = d[i*3+j];
            BUCP[i] += A * B[j];
            DUCP[i] += A * D[j];
        }
    }

    evalQuadraticBSpline(x, B, D);

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

// ----------------------------------------------------------------------------
// MipMap Lookups
// ----------------------------------------------------------------------------

float4 PtexMipmapLookupNearest(float4 patchCoord,
                               int level,
                               Texture2DArray data,
                               Buffer<uint> packings)
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

    float4 result = (1-t) * PtexLookupNearest(patchCoord, levelm, data, packings)
        + t * PtexLookupNearest(patchCoord, levelp, data, packings);
    return result;
}

float4 PtexMipmapLookup(float4 patchCoord,
                        float level,
                        Texture2DArray data,
                        Buffer<uint> packings)
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

    float4 result = (1-t) * PtexLookup(patchCoord, levelm, data, packings)
        + t * PtexLookup(patchCoord, levelp, data, packings);
    return result;
}

float4 PtexMipmapLookupQuadratic(out float4 du,
                                 out float4 dv,
                                 float4 patchCoord,
                                 float level,
                                 Texture2DArray data,
                                 Buffer<uint> packings)
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
    float4 r0 = PtexLookupQuadratic(du0, dv0, patchCoord, levelm, data, packings);
    float4 r1 = PtexLookupQuadratic(du1, dv1, patchCoord, levelp, data, packings);

    float4 result = lerp(r0, r1, t);
    du = lerp(du0, du1, t);
    dv = lerp(dv0, dv1, t);

    return result;
}

float4 PtexMipmapLookupQuadratic(float4 patchCoord,
                                 float level,
                                 Texture2DArray data,
                                 Buffer<uint> packings)
{
    float4 du, dv;
    return PtexMipmapLookupQuadratic(du, dv, patchCoord, level, data, packings);
}

