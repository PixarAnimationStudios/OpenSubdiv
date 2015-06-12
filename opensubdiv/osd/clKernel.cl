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
    __global int * sizes,
    __global int * offsets,
    __global int * indices,
    __global float * weights,
    __global float * duWeights,
    __global float * dvWeights,
    int batchStart, int batchEnd) {

    int current = get_global_id(0) + batchStart;

    if (current>=batchEnd) {
        return;
    }

    struct Vertex v, vdu, vdv;
    clear(&v);
    clear(&vdu);
    clear(&vdv);

    int size = sizes[current],
        offset = offsets[current];

    if (src) src += srcOffset;
    if (dst) dst += dstOffset;
    if (du)  du  += duOffset;
    if (dv)  dv  += dvOffset;

    for (int i=0; i<size; ++i) {
        int ofs = offset + i;
        int vid = indices[ofs];
        if (weights)   addWithWeight(  &v, src, vid,   weights[ofs]);
        if (duWeights) addWithWeight(&vdu, src, vid, duWeights[ofs]);
        if (dvWeights) addWithWeight(&vdv, src, vid, dvWeights[ofs]);
    }

    if (dst) writeVertex      (dst, current, &v);
    if (du)  writeVertexStride(du,  current, &vdu, duStride);
    if (dv)  writeVertexStride(dv,  current, &vdv, dvStride);
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

static void getBSplineWeights(float t, float *point, float *deriv) {
    // The four uniform cubic B-Spline basis functions evaluated at t:
    float one6th = 1.0f / 6.0f;

    float t2 = t * t;
    float t3 = t * t2;

    point[0] = one6th * (1.0f - 3.0f*(t -      t2) -      t3);
    point[1] = one6th * (4.0f           - 6.0f*t2  + 3.0f*t3);
    point[2] = one6th * (1.0f + 3.0f*(t +      t2  -      t3));
    point[3] = one6th * (                                 t3);

    // Derivatives of the above four basis functions at t:
    deriv[0] = -0.5f*t2 +      t - 0.5f;
    deriv[1] =  1.5f*t2 - 2.0f*t;
    deriv[2] = -1.5f*t2 +      t + 0.5f;
    deriv[3] =  0.5f*t2;
}

static void adjustBoundaryWeights(uint bits, float *sWeights, float *tWeights) {
    int boundary = ((bits >> 8) & 0xf);

    if (boundary & 1) {
        tWeights[2] -= tWeights[0];
        tWeights[1] += 2*tWeights[0];
        tWeights[0] = 0;
    }
    if (boundary & 2) {
        sWeights[1] -= sWeights[3];
        sWeights[2] += 2*sWeights[3];
        sWeights[3] = 0;
    }
    if (boundary & 4) {
        tWeights[1] -= tWeights[3];
        tWeights[2] += 2*tWeights[3];
        tWeights[3] = 0;
    }
    if (boundary & 8) {
        sWeights[2] -= sWeights[0];
        sWeights[1] += 2*sWeights[0];
        sWeights[0] = 0;
    }
}

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

__kernel void computePatches(__global float *src, int srcOffset,
                             __global float *dst, int dstOffset,
                             __global float *du,  int duOffset, int duStride,
                             __global float *dv,  int dvOffset, int dvStride,
                             __global struct PatchCoord *patchCoords,
                             __global struct PatchArray *patchArrayBuffer,
                             __global int *patchIndexBuffer,
                             __global struct PatchParam *patchParamBuffer) {
    int current = get_global_id(0);

    if (src) src += srcOffset;
    if (dst) dst += dstOffset;
    if (du)  du += duOffset;
    if (dv)  dv += dvOffset;

    struct PatchCoord coord = patchCoords[current];
    struct PatchArray array = patchArrayBuffer[coord.arrayIndex];

    int patchType = 6; // array.patchType XXX: REGULAR only for now.
    int numControlVertices = 16;
    uint patchBits = patchParamBuffer[coord.patchIndex].field1;

    float uv[2] = {coord.s, coord.t};
    normalizePatchCoord(patchBits, uv);
    float dScale = (float)(1 << getDepth(patchBits));

    float wP[20], wDs[20], wDt[20];
    if (patchType == 6) {  // REGULAR
        float sWeights[4], tWeights[4], dsWeights[4], dtWeights[4];
        getBSplineWeights(uv[0], sWeights, dsWeights);
        getBSplineWeights(uv[1], tWeights, dtWeights);

        adjustBoundaryWeights(patchBits, sWeights, tWeights);
        adjustBoundaryWeights(patchBits, dsWeights, dtWeights);

        for (int k = 0; k < 4; ++k) {
            for (int l = 0; l < 4; ++l) {
                wP[4*k+l]  = sWeights[l]  * tWeights[k];
                wDs[4*k+l] = dsWeights[l] * tWeights[k]  * dScale;
                wDt[4*k+l] = sWeights[l]  * dtWeights[k] * dScale;
            }
        }
    } else {
        // TODO: GREGORY BASIS
    }

    int indexBase = array.indexBase + coord.vertIndex;

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

}
