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

struct Vertex {
    float v[LENGTH];
};

static void clear(struct Vertex *vertex) {
    for (int i = 0; i < LENGTH; i++) {
        vertex->v[i] = 0.0f;
    }
}

static void addWithWeight(struct Vertex *dst,
                          __global float *srcOrigin,
                          int index, float weight) {

    __global float *src = srcOrigin + index * SRC_STRIDE;
    for (int i = 0; i < LENGTH; ++i) {
        dst->v[i] += src[i] * weight;
    }
}

static void writeVertex(__global float *dstOrigin,
                        int index,
                        struct Vertex *src) {

    __global float *dst = dstOrigin + index * DST_STRIDE;
    for (int i = 0; i < LENGTH; ++i) {
        dst[i] = src->v[i];
    }
}
static void writeVertexStride(__global float *dstOrigin,
                              int index,
                              struct Vertex *src,
                              int stride) {

    __global float *dst = dstOrigin + index * stride;
    for (int i = 0; i < LENGTH; ++i) {
        dst[i] = src->v[i];
    }
}


__kernel void computeStencils(
    __global float * src, int srcOffset,
    __global float * dst, int dstOffset,
    __global int * sizes,
    __global int * offsets,
    __global int * indices,
    __global float * weights,
    int batchStart, int batchEnd) {

    int current = get_global_id(0) + batchStart;

    if (current>=batchEnd) {
        return;
    }

    struct Vertex v;
    clear(&v);

    int size = sizes[current],
        offset = offsets[current];

    src += srcOffset;
    dst += dstOffset;

    for (int i=0; i<size; ++i) {
        addWithWeight(&v, src, indices[offset+i], weights[offset+i]);
    }

    writeVertex(dst, current, &v);
}

__kernel void computeStencilsDerivatives(
    __global float * src, int srcOffset,
    __global float * dst, int dstOffset,
    __global float * du,  int duOffset, int duStride,
    __global float * dv,  int dvOffset, int dvStride,
    __global float * duu, int duuOffset, int duuStride,
    __global float * duv, int duvOffset, int duvStride,
    __global float * dvv, int dvvOffset, int dvvStride,
    __global int * sizes,
    __global int * offsets,
    __global int * indices,
    __global float * weights,
    __global float * duWeights,
    __global float * dvWeights,
    __global float * duuWeights,
    __global float * duvWeights,
    __global float * dvvWeights,
    int batchStart, int batchEnd) {

    int current = get_global_id(0) + batchStart;

    if (current>=batchEnd) {
        return;
    }

    struct Vertex v, vdu, vdv, vduu, vduv, vdvv;
    clear(&v);
    clear(&vdu);
    clear(&vdv);
    clear(&vduu);
    clear(&vduv);
    clear(&vdvv);

    int size = sizes[current],
        offset = offsets[current];

    if (src) src += srcOffset;
    if (dst) dst += dstOffset;
    if (du)  du  += duOffset;
    if (dv)  dv  += dvOffset;
    if (duu) duu += duuOffset;
    if (duv) duv += duvOffset;
    if (dvv) dvv += dvvOffset;

    for (int i=0; i<size; ++i) {
        int ofs = offset + i;
        int vid = indices[ofs];
        if (weights)   addWithWeight(  &v, src, vid,   weights[ofs]);
        if (duWeights) addWithWeight(&vdu, src, vid, duWeights[ofs]);
        if (dvWeights) addWithWeight(&vdv, src, vid, dvWeights[ofs]);
        if (duuWeights) addWithWeight(&vduu, src, vid, duuWeights[ofs]);
        if (duvWeights) addWithWeight(&vduv, src, vid, duvWeights[ofs]);
        if (dvvWeights) addWithWeight(&vdvv, src, vid, dvvWeights[ofs]);
    }

    if (dst) writeVertex      (dst, current, &v);
    if (du)  writeVertexStride(du,  current, &vdu, duStride);
    if (dv)  writeVertexStride(dv,  current, &vdv, dvStride);
    if (duu) writeVertexStride(duu, current, &vduu, duuStride);
    if (duv) writeVertexStride(duv, current, &vduv, duvStride);
    if (dvv) writeVertexStride(dvv, current, &vdvv, dvvStride);
}

// ---------------------------------------------------------------------------

struct PatchArray {
    int patchType;
    int numPatches;
    int indexBase;        // an offset within the index buffer
    int primitiveIdBase;  // an offset within the patch param buffer
};

struct PatchCoord {
   int arrayIndex;
   int patchIndex;
   int vertIndex;
   float s;
   float t;
};

struct PatchParam {
    uint field0;
    uint field1;
    float sharpness;
};

static int getDepth(uint patchBits) {
    return (patchBits & 0xf);
}

static float getParamFraction(uint patchBits) {
    bool nonQuadRoot = (patchBits >> 4) & 0x1;
    int depth = getDepth(patchBits);
    if (nonQuadRoot) {
        return 1.0f / (float)( 1 << (depth-1) );
    } else {
        return 1.0f / (float)( 1 << depth );
    }
}

static void normalizePatchCoord(uint patchBits, float *uv) {
    float frac = getParamFraction(patchBits);

    int iu = (patchBits >> 22) & 0x3ff;
    int iv = (patchBits >> 12) & 0x3ff;

    // top left corner
    float pu = (float)iu*frac;
    float pv = (float)iv*frac;

    // normalize u,v coordinates
    uv[0] = (uv[0] - pu) / frac;
    uv[1] = (uv[1] - pv) / frac;
}

static bool isRegular(uint patchBits) {
    return ((patchBits >> 5) & 0x1) != 0;
}

static int getNumControlVertices(int patchType) {
    return (patchType == 3) ? 4 :
           (patchType == 6) ? 16 :
           (patchType == 9) ? 20 : 0;
}

__kernel void computePatches(__global float *src, int srcOffset,
                             __global float *dst, int dstOffset,
                             __global float *du,  int duOffset, int duStride,
                             __global float *dv,  int dvOffset, int dvStride,
                             __global float *duu, int duuOffset, int duuStride,
                             __global float *duv, int duvOffset, int duvStride,
                             __global float *dvv, int dvvOffset, int dvvStride,
                             __global struct PatchCoord *patchCoords,
                             __global struct PatchArray *patchArrayBuffer,
                             __global int *patchIndexBuffer,
                             __global struct PatchParam *patchParamBuffer) {
    int current = get_global_id(0);

    if (src) src += srcOffset;
    if (dst) dst += dstOffset;
    if (du)  du  += duOffset;
    if (dv)  dv  += dvOffset;
    if (duu) duu += duuOffset;
    if (duv) duv += duvOffset;
    if (dvv) dvv += dvvOffset;

    struct PatchCoord coord = patchCoords[current];
    struct PatchArray array = patchArrayBuffer[coord.arrayIndex];

    uint patchBits = patchParamBuffer[coord.patchIndex].field1;
    int patchType = isRegular(patchBits) ? 6 : array.patchType;

    float uv[2] = {coord.s, coord.t};
    normalizePatchCoord(patchBits, uv);
    float dScale = (float)(1 << getDepth(patchBits));
    int boundary = (patchBits >> 8) & 0xf;

    float wP[20], wDs[20], wDt[20], wDss[20], wDst[20], wDtt[20];

    int numControlVertices = 0;
    if (patchType == 3) {
        OsdGetBilinearPatchWeights(uv[0], uv[1], dScale,
            wP, wDs, wDt, wDss, wDst, wDtt);
        numControlVertices = 4;
    } else if (patchType == 6) {
        OsdGetBSplinePatchWeights(uv[0], uv[1], dScale, boundary,
            wP, wDs, wDt, wDss, wDst, wDtt);
        numControlVertices = 16;
    } else if (patchType == 9) {
        OsdGetGregoryPatchWeights(uv[0], uv[1], dScale,
            wP, wDs, wDt, wDss, wDst, wDtt);
        numControlVertices = 20;
    }

    int indexStride = getNumControlVertices(array.patchType);
    int indexBase = array.indexBase + indexStride *
            (coord.patchIndex - array.primitiveIdBase);

    struct Vertex v;
    clear(&v);
    for (int i = 0; i < numControlVertices; ++i) {
        int index = patchIndexBuffer[indexBase + i];
        addWithWeight(&v, src, index, wP[i]);
    }
    writeVertex(dst, current, &v);

    if (du) {
        struct Vertex vdu;
        clear(&vdu);
        for (int i = 0; i < numControlVertices; ++i) {
            int index = patchIndexBuffer[indexBase + i];
            addWithWeight(&vdu, src, index, wDs[i]);
        }
        writeVertexStride(du, current, &vdu, duStride);
    }
    if (dv) {
        struct Vertex vdv;
        clear(&vdv);
        for (int i = 0; i < numControlVertices; ++i) {
            int index = patchIndexBuffer[indexBase + i];
            addWithWeight(&vdv, src, index, wDt[i]);
        }
        writeVertexStride(dv, current, &vdv, dvStride);
    }
    if (duu) {
        struct Vertex vduu;
        clear(&vduu);
        for (int i = 0; i < numControlVertices; ++i) {
            int index = patchIndexBuffer[indexBase + i];
            addWithWeight(&vduu, src, index, wDss[i]);
        }
        writeVertexStride(duu, current, &vduu, duuStride);
    }
    if (duv) {
        struct Vertex vduv;
        clear(&vduv);
        for (int i = 0; i < numControlVertices; ++i) {
            int index = patchIndexBuffer[indexBase + i];
            addWithWeight(&vduv, src, index, wDst[i]);
        }
        writeVertexStride(duv, current, &vduv, duvStride);
    }
    if (dvv) {
        struct Vertex vdvv;
        clear(&vdvv);
        for (int i = 0; i < numControlVertices; ++i) {
            int index = patchIndexBuffer[indexBase + i];
            addWithWeight(&vdvv, src, index, wDtt[i]);
        }
        writeVertexStride(dvv, current, &vdvv, dvvStride);
    }
}
