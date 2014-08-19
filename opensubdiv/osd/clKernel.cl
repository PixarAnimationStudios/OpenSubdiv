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

#ifndef M_PI
    #define M_PI 3.14159265358979323846  // fix for OSX 10.8 (M_PI is in the Khronos standard...)
#endif

struct Vertex
{
    float v[VERTEX_STRIDE];
};

struct Varying
{
    float v[VARYING_STRIDE];
};

static void clearVertex(struct Vertex *vertex) {

    for (int i = 0; i < NUM_VERTEX_ELEMENTS; i++) {
        vertex->v[i] = 0;
    }
}
static void clearVarying(struct Varying *varying) {

    for (int i = 0; i < NUM_VARYING_ELEMENTS; i++) {
        varying->v[i] = 0;
    }
}

static void addWithWeight(struct Vertex *dst,
                          __global float *srcOrigin,
                          int index, float weight) {

    __global float *src = srcOrigin + index * VERTEX_STRIDE;
    for (int i = 0; i < NUM_VERTEX_ELEMENTS; ++i) {
        dst->v[i] += src[i] * weight;
    }
}

static void addVaryingWithWeight(struct Varying *dst,
                                 __global float *srcOrigin,
                                 int index, float weight) {

    __global float *src = srcOrigin + index * VARYING_STRIDE;
    for (int i = 0; i < NUM_VARYING_ELEMENTS; ++i) {
        dst->v[i] += src[i] * weight;
    }
}

static void writeVertex(__global float *dstOrigin,
                        int index,
                        struct Vertex *src) {

    __global float *dst = dstOrigin + index * VERTEX_STRIDE;
    for (int i = 0; i < NUM_VERTEX_ELEMENTS; ++i) {
        dst[i] = src->v[i];
    }
}

static void writeVarying(__global float *dstOrigin,
                         int index,
                         struct Varying *src) {

    __global float *dst = dstOrigin + index * VARYING_STRIDE;
    for (int i = 0; i < NUM_VARYING_ELEMENTS; ++i) {
        dst[i] = src->v[i];
    }
}

__kernel void computeBilinearEdge(__global float *vertex,
                                  __global float *varying,
                                  __global int *E_IT,
                                  int vertexOffset, int varyingOffset,
                                  int offset, int tableOffset,
                                  int start, int end) {

    int i = start + get_global_id(0) + tableOffset;
    int vid = start + get_global_id(0) + offset;
    int eidx0 = E_IT[2*i+0];
    int eidx1 = E_IT[2*i+1];
    vertex += vertexOffset;
    varying += (varying ? varyingOffset :0);

    struct Vertex dst;
    struct Varying dstVarying;
    clearVertex(&dst);
    clearVarying(&dstVarying);

    addWithWeight(&dst, vertex, eidx0, 0.5f);
    addWithWeight(&dst, vertex, eidx1, 0.5f);

    writeVertex(vertex, vid, &dst);

    if (varying) {
        addVaryingWithWeight(&dstVarying, varying, eidx0, 0.5f);
        addVaryingWithWeight(&dstVarying, varying, eidx1, 0.5f);
        writeVarying(varying, vid, &dstVarying);
    }
}

__kernel void computeBilinearVertex(__global float *vertex,
                                    __global float *varying,
                                    __global int *V_ITa,
                                    int vertexOffset, int varyingOffset,
                                    int offset, int tableOffset,
                                    int start, int end) {

    int i = start + get_global_id(0) + tableOffset;
    int vid = start + get_global_id(0) + offset;
    vertex += vertexOffset;
    varying += (varying ? varyingOffset :0);

    int p = V_ITa[i];

    struct Vertex dst;
    clearVertex(&dst);
    addWithWeight(&dst, vertex, p, 1.0f);

    writeVertex(vertex, vid, &dst);

    if (varying) {
        struct Varying dstVarying;
        clearVarying(&dstVarying);
        addVaryingWithWeight(&dstVarying, varying, p, 1.0f);
        writeVarying(varying, vid, &dstVarying);
    }
}

// ---------------------------------------------------------------------------

__kernel void computeFace(__global float *vertex,
                          __global float *varying,
                          __global int *F_IT,
                          __global int *F_ITa,
                          int vertexOffset, int varyingOffset,
                          int offset, int tableOffset,
                          int start, int end) {

    int i = start + get_global_id(0) + tableOffset;
    int vid = start + get_global_id(0) + offset;
    int h = F_ITa[2*i];
    int n = F_ITa[2*i+1];
    vertex += vertexOffset;
    varying += (varying ? varyingOffset :0);

    float weight = 1.0f/n;

    struct Vertex dst;
    struct Varying dstVarying;
    clearVertex(&dst);
    clearVarying(&dstVarying);
    for (int j=0; j<n; ++j) {
        int index = F_IT[h+j];
        addWithWeight(&dst, vertex, index, weight);
        if (varying) {
            addVaryingWithWeight(&dstVarying, varying, index, weight);
        }
    }
    writeVertex(vertex, vid, &dst);
    if (varying) writeVarying(varying, vid, &dstVarying);
}

__kernel void computeQuadFace(__global float *vertex,
                              __global float *varying,
                              __global int *F_IT,
                              int vertexOffset, int varyingOffset,
                              int offset, int tableOffset,
                              int start, int end) {

    int i = start + get_global_id(0);
    int vid = start + get_global_id(0) + offset;
    vertex += vertexOffset;
    varying += (varying ? varyingOffset :0);

    struct Vertex dst;
    struct Varying dstVarying;
    clearVertex(&dst);
    clearVarying(&dstVarying);

    int fidx0 = F_IT[tableOffset + 4 * i + 0];
    int fidx1 = F_IT[tableOffset + 4 * i + 1];
    int fidx2 = F_IT[tableOffset + 4 * i + 2];
    int fidx3 = F_IT[tableOffset + 4 * i + 3];

    addWithWeight(&dst, vertex, fidx0, 0.25f);
    addWithWeight(&dst, vertex, fidx1, 0.25f);
    addWithWeight(&dst, vertex, fidx2, 0.25f);
    addWithWeight(&dst, vertex, fidx3, 0.25f);

    if (varying) {
        addVaryingWithWeight(&dstVarying, varying, fidx0, 0.25f);
        addVaryingWithWeight(&dstVarying, varying, fidx1, 0.25f);
        addVaryingWithWeight(&dstVarying, varying, fidx2, 0.25f);
        addVaryingWithWeight(&dstVarying, varying, fidx3, 0.25f);
    }

    writeVertex(vertex, vid, &dst);
    if (varying) writeVarying(varying, vid, &dstVarying);
}

__kernel void computeTriQuadFace(__global float *vertex,
                                 __global float *varying,
                                 __global int *F_IT,
                                 int vertexOffset, int varyingOffset,
                                 int offset, int tableOffset,
                                 int start, int end) {

    int i = start + get_global_id(0);
    int vid = start + get_global_id(0) + offset;
    vertex += vertexOffset;
    varying += (varying ? varyingOffset :0);

    struct Vertex dst;
    struct Varying dstVarying;
    clearVertex(&dst);
    clearVarying(&dstVarying);

    int fidx0 = F_IT[tableOffset + 4 * i + 0];
    int fidx1 = F_IT[tableOffset + 4 * i + 1];
    int fidx2 = F_IT[tableOffset + 4 * i + 2];
    int fidx3 = F_IT[tableOffset + 4 * i + 3];
    bool triangle = (fidx2 == fidx3);
    float weight = triangle ? 1.0f / 3.0f : 1.0f / 4.0f;

    addWithWeight(&dst, vertex, fidx0, weight);
    addWithWeight(&dst, vertex, fidx1, weight);
    addWithWeight(&dst, vertex, fidx2, weight);
    if (!triangle)
        addWithWeight(&dst, vertex, fidx3, weight);

    if (varying) {
        addVaryingWithWeight(&dstVarying, varying, fidx0, weight);
        addVaryingWithWeight(&dstVarying, varying, fidx1, weight);
        addVaryingWithWeight(&dstVarying, varying, fidx2, weight);
        if (!triangle)
            addVaryingWithWeight(&dstVarying, varying, fidx3, weight);
    }

    writeVertex(vertex, vid, &dst);
    if (varying) writeVarying(varying, vid, &dstVarying);
}

__kernel void computeEdge(__global float *vertex,
                          __global float *varying,
                          __global int *E_IT,
                          __global float *E_W,
                          int vertexOffset, int varyingOffset,
                          int offset, int tableOffset,
                          int start, int end) {

    int i = start + get_global_id(0) + tableOffset;
    int vid = start + get_global_id(0) + offset;
    int eidx0 = E_IT[4*i+0];
    int eidx1 = E_IT[4*i+1];
    int eidx2 = E_IT[4*i+2];
    int eidx3 = E_IT[4*i+3];
    vertex += vertexOffset;
    varying += (varying ? varyingOffset :0);

    float vertWeight = E_W[i*2+0];

    // Fully sharp edge : vertWeight = 0.5f;
    struct Vertex dst;
    struct Varying dstVarying;
    clearVertex(&dst);
    clearVarying(&dstVarying);

    addWithWeight(&dst, vertex, eidx0, vertWeight);
    addWithWeight(&dst, vertex, eidx1, vertWeight);

    if (eidx2 > -1) {
        float faceWeight = E_W[i*2+1];

        addWithWeight(&dst, vertex, eidx2, faceWeight);
        addWithWeight(&dst, vertex, eidx3, faceWeight);
    }

    writeVertex(vertex, vid, &dst);

    if (varying) {
        addVaryingWithWeight(&dstVarying, varying, eidx0, 0.5f);
        addVaryingWithWeight(&dstVarying, varying, eidx1, 0.5f);
        writeVarying(varying, vid, &dstVarying);
    }
}

__kernel void computeRestrictedEdge(__global float *vertex,
                                    __global float *varying,
                                    __global int *E_IT,
                                    int vertexOffset, int varyingOffset,
                                    int offset, int tableOffset,
                                    int start, int end) {

    int i = start + get_global_id(0) + tableOffset;
    int vid = start + get_global_id(0) + offset;
    int eidx0 = E_IT[4*i+0];
    int eidx1 = E_IT[4*i+1];
    int eidx2 = E_IT[4*i+2];
    int eidx3 = E_IT[4*i+3];
    vertex += vertexOffset;
    varying += (varying ? varyingOffset :0);

    struct Vertex dst;
    struct Varying dstVarying;
    clearVertex(&dst);
    clearVarying(&dstVarying);

    addWithWeight(&dst, vertex, eidx0, 0.25f);
    addWithWeight(&dst, vertex, eidx1, 0.25f);
    addWithWeight(&dst, vertex, eidx2, 0.25f);
    addWithWeight(&dst, vertex, eidx3, 0.25f);

    writeVertex(vertex, vid, &dst);

    if (varying) {
        addVaryingWithWeight(&dstVarying, varying, eidx0, 0.5f);
        addVaryingWithWeight(&dstVarying, varying, eidx1, 0.5f);
        writeVarying(varying, vid, &dstVarying);
    }
}

__kernel void computeVertexA(__global float *vertex,
                             __global float *varying,
                             __global int *V_ITa,
                             __global float *V_W,
                             int vertexOffset, int varyingOffset,
                             int offset, int tableOffset,
                             int start, int end, int pass) {

    int i = start + get_global_id(0) + tableOffset;
    int vid = start + get_global_id(0) + offset;
    int n     = V_ITa[5*i+1];
    int p     = V_ITa[5*i+2];
    int eidx0 = V_ITa[5*i+3];
    int eidx1 = V_ITa[5*i+4];
    vertex += vertexOffset;
    varying += (varying ? varyingOffset :0);

    float weight = (pass==1) ? V_W[i] : 1.0f - V_W[i];

    // In the case of fractional weight, the weight must be inverted since
    // the value is shared with the k_Smooth kernel (statistically the
    // k_Smooth kernel runs much more often than this one)
    if (weight>0.0f && weight<1.0f && n > 0)
        weight=1.0f-weight;

    struct Vertex dst;
    clearVertex(&dst);
    if (pass)
        addWithWeight(&dst, vertex, vid, 1.0f); // copy previous result

    if (eidx0==-1 || (pass==0 && (n==-1)) ) {
        addWithWeight(&dst, vertex, p, weight);
    } else {
        addWithWeight(&dst, vertex, p, weight * 0.75f);
        addWithWeight(&dst, vertex, eidx0, weight * 0.125f);
        addWithWeight(&dst, vertex, eidx1, weight * 0.125f);
    }
    writeVertex(vertex, vid, &dst);

    if (! pass && varying) {
        struct Varying dstVarying;
        clearVarying(&dstVarying);
        addVaryingWithWeight(&dstVarying, varying, p, 1.0f);
        writeVarying(varying, vid, &dstVarying);
    }
}

__kernel void computeVertexB(__global float *vertex,
                             __global float *varying,
                             __global int *V_ITa,
                             __global int *V_IT,
                             __global float *V_W,
                             int vertexOffset, int varyingOffset,
                             int offset, int tableOffset,
                             int start, int end) {

    int i = start + get_global_id(0) + tableOffset;
    int vid = start + get_global_id(0) + offset;
    int h = V_ITa[5*i];
    int n = V_ITa[5*i+1];
    int p = V_ITa[5*i+2];
    vertex += vertexOffset;
    varying += (varying ? varyingOffset :0);

    float weight = V_W[i];
    float wp = 1.0f/(float)(n*n);
    float wv = (n-2.0f) * n * wp;

    struct Vertex dst;
    clearVertex(&dst);

    addWithWeight(&dst, vertex, p, weight * wv);

    for (int j = 0; j < n; ++j) {
        addWithWeight(&dst, vertex, V_IT[h+j*2], weight * wp);
        addWithWeight(&dst, vertex, V_IT[h+j*2+1], weight * wp);
    }
    writeVertex(vertex, vid, &dst);

    if (varying) {
        struct Varying dstVarying;
        clearVarying(&dstVarying);
        addVaryingWithWeight(&dstVarying, varying, p, 1.0f);
        writeVarying(varying, vid, &dstVarying);
    }
}

__kernel void computeRestrictedVertexA(__global float *vertex,
                                       __global float *varying,
                                       __global int *V_ITa,
                                       int vertexOffset, int varyingOffset,
                                       int offset, int tableOffset,
                                       int start, int end) {

    int i = start + get_global_id(0) + tableOffset;
    int vid = start + get_global_id(0) + offset;
    int p     = V_ITa[5*i+2];
    int eidx0 = V_ITa[5*i+3];
    int eidx1 = V_ITa[5*i+4];
    vertex += vertexOffset;
    varying += (varying ? varyingOffset :0);

    struct Vertex dst;
    clearVertex(&dst);
    addWithWeight(&dst, vertex, p, 0.75f);
    addWithWeight(&dst, vertex, eidx0, 0.125f);
    addWithWeight(&dst, vertex, eidx1, 0.125f);
    writeVertex(vertex, vid, &dst);

    if (varying) {
        struct Varying dstVarying;
        clearVarying(&dstVarying);
        addVaryingWithWeight(&dstVarying, varying, p, 1.0f);
        writeVarying(varying, vid, &dstVarying);
    }
}

__kernel void computeRestrictedVertexB1(__global float *vertex,
                                        __global float *varying,
                                        __global int *V_ITa,
                                        __global int *V_IT,
                                        int vertexOffset, int varyingOffset,
                                        int offset, int tableOffset,
                                        int start, int end) {

    int i = start + get_global_id(0) + tableOffset;
    int vid = start + get_global_id(0) + offset;
    int h = V_ITa[5*i];
    int p = V_ITa[5*i+2];
    vertex += vertexOffset;
    varying += (varying ? varyingOffset :0);

    struct Vertex dst;
    clearVertex(&dst);

    addWithWeight(&dst, vertex, p, 0.5f);

    for (int j = 0; j < 8; ++j, ++h) {
        addWithWeight(&dst, vertex, V_IT[h], 0.0625f);
    }
    writeVertex(vertex, vid, &dst);

    if (varying) {
        struct Varying dstVarying;
        clearVarying(&dstVarying);
        addVaryingWithWeight(&dstVarying, varying, p, 1.0f);
        writeVarying(varying, vid, &dstVarying);
    }
}

__kernel void computeRestrictedVertexB2(__global float *vertex,
                                        __global float *varying,
                                        __global int *V_ITa,
                                        __global int *V_IT,
                                        int vertexOffset, int varyingOffset,
                                        int offset, int tableOffset,
                                        int start, int end) {

    int i = start + get_global_id(0) + tableOffset;
    int vid = start + get_global_id(0) + offset;
    int h = V_ITa[5*i];
    int n = V_ITa[5*i+1];
    int p = V_ITa[5*i+2];
    vertex += vertexOffset;
    varying += (varying ? varyingOffset :0);

    float wp = 1.0f/(float)(n*n);
    float wv = (n-2.0f) * n * wp;

    struct Vertex dst;
    clearVertex(&dst);

    addWithWeight(&dst, vertex, p, wv);

    for (int j = 0; j < n; ++j) {
        addWithWeight(&dst, vertex, V_IT[h+j*2], wp);
        addWithWeight(&dst, vertex, V_IT[h+j*2+1], wp);
    }
    writeVertex(vertex, vid, &dst);

    if (varying) {
        struct Varying dstVarying;
        clearVarying(&dstVarying);
        addVaryingWithWeight(&dstVarying, varying, p, 1.0f);
        writeVarying(varying, vid, &dstVarying);
    }
}

__kernel void computeLoopVertexB(__global float *vertex,
                                 __global float *varying,
                                 __global int *V_ITa,
                                 __global int *V_IT,
                                 __global float *V_W,
                                 int vertexOffset, int varyingOffset,
                                 int offset, int tableOffset,
                                 int start, int end) {

    int i = start + get_global_id(0) + tableOffset;
    int vid = start + get_global_id(0) + offset;
    int h = V_ITa[5*i];
    int n = V_ITa[5*i+1];
    int p = V_ITa[5*i+2];
    vertex += vertexOffset;
    varying += (varying ? varyingOffset :0);

    float weight = V_W[i];
    float wp = 1.0f/(float)(n);
    float beta = 0.25f * cos((float)(M_PI) * 2.0f * wp) + 0.375f;
    beta = beta * beta;
    beta = (0.625f - beta) * wp;

    struct Vertex dst;
    clearVertex(&dst);
    addWithWeight(&dst, vertex, p, weight * (1.0f - (beta * n)));

    for (int j = 0; j < n; ++j) {
        addWithWeight(&dst, vertex, V_IT[h+j], weight * beta);
    }
    writeVertex(vertex, vid, &dst);

    if (varying) {
        struct Varying dstVarying;
        clearVarying(&dstVarying);
        addVaryingWithWeight(&dstVarying, varying, p, 1.0f);
        writeVarying(varying, vid, &dstVarying);
    }
}

__kernel void editVertexAdd(__global float *vertex,
                            __global int *editIndices,
                            __global float *editValues,
                            int vertexOffset,
                            int primVarOffset,
                            int primVarWidth,
                            int offset, int tableOffset,
                            int start, int end) {

    int i = start + get_global_id(0) + tableOffset;
    int v = editIndices[i];
    int eid = start + get_global_id(0);
    vertex += vertexOffset;
    vertex += v * VERTEX_STRIDE + primVarOffset;

    for (int j = 0; j < primVarWidth; ++j) {
        vertex[j] += editValues[eid*primVarWidth + j];
    }
}
