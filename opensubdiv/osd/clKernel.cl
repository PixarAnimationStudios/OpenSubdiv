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


__kernel void computeStencils(__global float * src,
                              int srcOffset,
                              __global float * dst,
                              int dstOffset,
                              __global int * sizes,
                              __global int * offsets,
                              __global int * indices,
                              __global float * weights,
                              int batchStart,
                              int batchEnd) {

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
    int faceIndex;
    uint patchBits;
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

__kernel void computePatches(__global float *src, int srcOffset,
                             __global float *dst, int dstOffset,
//                             __global float *du,  int duOffset, int duStride,
//                             __global float *dv,  int dvOffset, int dvStride,
                             int numPatchCoords,
                             __global struct PatchCoord *patchCoords,
                             __global struct PatchArray *patchArrayBuffer,
                             __global int *patchIndexBuffer,
                             __global struct PatchParam *patchParamBuffer) {
    int current = get_global_id(0);

    if (current > numPatchCoords) return;

    src += srcOffset;
    dst += dstOffset;
    // du += duOffset;
    // dv += dvOffset;

    struct PatchCoord coord = patchCoords[current];
    int patchIndex = coord.patchIndex;
//    struct PatchArray array = patchArrayBuffer[coord.arrayIndex];
    struct PatchArray array = patchArrayBuffer[0];

    int patchType = 6; // array.x XXX: REGULAR only for now.
    int numControlVertices = 16;

    uint patchBits = patchParamBuffer[patchIndex].patchBits;
//    vec2 uv = normalizePatchCoord(patchBits, vec2(coord.s, coord.t));
    float dScale = 1.0f;//float(1 << getDepth(patchBits));

    float uv[2] = {coord.s, coord.t};

    float wP[20], wDs[20], wDt[20];
    if (patchType == 6) {  // REGULAR
        float sWeights[4], tWeights[4], dsWeights[4], dtWeights[4];
        getBSplineWeights(uv[0], sWeights, dsWeights);
        getBSplineWeights(uv[1], tWeights, dtWeights);

//        adjustBoundaryWeights(patchBits, sWeights, tWeights);
//        adjustBoundaryWeights(patchBits, dsWeights, dtWeights);

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

    struct Vertex v;
    clear(&v);

#if 1
    // debug
    v.v[0] = uv[0];
    v.v[1] = uv[1];
    v.v[2] = patchIndexBuffer[current] * 0.1;
    writeVertex(dst, current, &v);
    return;
#endif

    int indexBase = array.indexBase + coord.vertIndex;
    for (int i = 0; i < numControlVertices; ++i) {
        int index = patchIndexBuffer[indexBase + i];
        if (index < 0) index = 0;
        addWithWeight(&v, src, index, wP[i]);
    }
    writeVertex(dst, current, &v);
}
