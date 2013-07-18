//
//     Copyright 2013 Pixar
//
//     Licensed under the Apache License, Version 2.0 (the "License");
//     you may not use this file except in compliance with the License
//     and the following modification to it: Section 6 Trademarks.
//     deleted and replaced with:
//
//     6. Trademarks. This License does not grant permission to use the
//     trade names, trademarks, service marks, or product names of the
//     Licensor and its affiliates, except as required for reproducing
//     the content of the NOTICE file.
//
//     You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
//     Unless required by applicable law or agreed to in writing,
//     software distributed under the License is distributed on an
//     "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
//     either express or implied.  See the License for the specific
//     language governing permissions and limitations under the
//     License.
//

#ifndef M_PI
    #define M_PI 3.14159265358979323846  // fix for OSX 10.8 (M_PI is in the Khronos standard...)
#endif

struct Vertex
{
    float v[NUM_VERTEX_ELEMENTS];
};

struct Varying
{
    float v[NUM_VARYING_ELEMENTS];
};

__global void clearVertex(struct Vertex *vertex) {

    for (int i = 0; i < NUM_VERTEX_ELEMENTS; i++) {
        vertex->v[i] = 0;
    }
}
__global void clearVarying(struct Varying *varying) {

    for (int i = 0; i < NUM_VARYING_ELEMENTS; i++) {
        varying->v[i] = 0;
    }
}

__global void addWithWeight(struct Vertex *dst, __global struct Vertex *src, float weight) {

    for (int i = 0; i < NUM_VERTEX_ELEMENTS; i++) {
        dst->v[i] += src->v[i] * weight;
    }
}

__global void addVaryingWithWeight(struct Varying *dst, __global struct Varying *src, float weight) {

    for (int i = 0; i < NUM_VARYING_ELEMENTS; i++) {
        dst->v[i] += src->v[i] * weight;
    }
}

__kernel void computeBilinearEdge(__global struct Vertex *vertex,
                                  __global struct Varying *varying,
                                  __global int *E_IT,
                                  int vertexOffset, int tableOffset,
                                  int start, int end) {

    int i = start + get_global_id(0) + tableOffset;
    int vid = start + get_global_id(0) + vertexOffset;
    int eidx0 = E_IT[2*i+0];
    int eidx1 = E_IT[2*i+1];

    struct Vertex dst;
    struct Varying dstVarying;
    clearVertex(&dst);
    clearVarying(&dstVarying);

    addWithWeight(&dst, &vertex[eidx0], 0.5f);
    addWithWeight(&dst, &vertex[eidx1], 0.5f);

    vertex[vid] = dst;

    if (varying) {
        addVaryingWithWeight(&dstVarying, &varying[eidx0], 0.5f);
        addVaryingWithWeight(&dstVarying, &varying[eidx1], 0.5f);
        varying[vid] = dstVarying;
    }
}

__kernel void computeBilinearVertex(__global struct Vertex *vertex,
                                    __global struct Varying *varying,
                                    __global int *V_ITa,
                                    int vertexOffset, int tableOffset,
                                    int start, int end) {

    int i = start + get_global_id(0) + tableOffset;
    int vid = start + get_global_id(0) + vertexOffset;

    int p = V_ITa[i];

    struct Vertex dst;
    clearVertex(&dst);
    addWithWeight(&dst, &vertex[p], 1.0f);

    vertex[vid] = dst;

    if (varying) {
        struct Varying dstVarying;
        clearVarying(&dstVarying);
        addVaryingWithWeight(&dstVarying, &varying[p], 1.0f);
        varying[vid] = dstVarying;
    }
}

// ----------------------------------------------------------------------------------------

__kernel void computeFace(__global struct Vertex *vertex,
                          __global struct Varying *varying,
                          __global int *F_IT,
                          __global int *F_ITa,
                          int vertexOffset, int tableOffset,
                          int start, int end) {

    int i = start + get_global_id(0) + tableOffset;
    int vid = start + get_global_id(0) + vertexOffset;
    int h = F_ITa[2*i];
    int n = F_ITa[2*i+1];

    float weight = 1.0f/n;

    struct Vertex dst;
    struct Varying dstVarying;
    clearVertex(&dst);
    clearVarying(&dstVarying);
    for (int j=0; j<n; ++j) {
        int index = F_IT[h+j];
        addWithWeight(&dst, &vertex[index], weight);
        if(varying) addVaryingWithWeight(&dstVarying, &varying[index], weight);
    }
    vertex[vid] = dst;
    if (varying) varying[vid] = dstVarying;
}

__kernel void computeEdge(__global struct Vertex *vertex,
                          __global struct Varying *varying,
                          __global int *E_IT,
                          __global float *E_W,
                          int vertexOffset, int tableOffset,
                          int start, int end) {

    int i = start + get_global_id(0) + tableOffset;
    int vid = start + get_global_id(0) + vertexOffset;
    int eidx0 = E_IT[4*i+0];
    int eidx1 = E_IT[4*i+1];
    int eidx2 = E_IT[4*i+2];
    int eidx3 = E_IT[4*i+3];

    float vertWeight = E_W[i*2+0];

    // Fully sharp edge : vertWeight = 0.5f;
    struct Vertex dst;
    struct Varying dstVarying;
    clearVertex(&dst);
    clearVarying(&dstVarying);

    addWithWeight(&dst, &vertex[eidx0], vertWeight);
    addWithWeight(&dst, &vertex[eidx1], vertWeight);

    if (eidx2 > -1) {
        float faceWeight = E_W[i*2+1];

        addWithWeight(&dst, &vertex[eidx2], faceWeight);
        addWithWeight(&dst, &vertex[eidx3], faceWeight);
    }

    vertex[vid] = dst;

    if (varying) {
        addVaryingWithWeight(&dstVarying, &varying[eidx0], 0.5f);
        addVaryingWithWeight(&dstVarying, &varying[eidx1], 0.5f);
        varying[vid] = dstVarying;
    }
}

__kernel void computeVertexA(__global struct Vertex *vertex,
                             __global struct Varying *varying,
                             __global int *V_ITa,
                             __global float *V_W,
                             int vertexOffset, int tableOffset,
                             int start, int end, int pass) {

    int i = start + get_global_id(0) + tableOffset;
    int vid = start + get_global_id(0) + vertexOffset;
    int n     = V_ITa[5*i+1];
    int p     = V_ITa[5*i+2];
    int eidx0 = V_ITa[5*i+3];
    int eidx1 = V_ITa[5*i+4];

    float weight = (pass==1) ? V_W[i] : 1.0f - V_W[i];

    // In the case of fractional weight, the weight must be inverted since
    // the value is shared with the k_Smooth kernel (statistically the
    // k_Smooth kernel runs much more often than this one)
    if (weight>0.0f && weight<1.0f && n > 0)
        weight=1.0f-weight;

    struct Vertex dst;
    if (! pass)
        clearVertex(&dst);
    else
        dst = vertex[vid];

    if (eidx0==-1 || (pass==0 && (n==-1)) ) {
        addWithWeight(&dst, &vertex[p], weight);
    } else {
        addWithWeight(&dst, &vertex[p], weight * 0.75f);
        addWithWeight(&dst, &vertex[eidx0], weight * 0.125f);
        addWithWeight(&dst, &vertex[eidx1], weight * 0.125f);
    }
    vertex[vid] = dst;

    if (! pass && varying) {
        struct Varying dstVarying;
        clearVarying(&dstVarying);
        addVaryingWithWeight(&dstVarying, &varying[p], 1.0f);
        varying[vid] = dstVarying;
    }
}

__kernel void computeVertexB(__global struct Vertex *vertex,
                             __global struct Varying *varying,
                             __global int *V_ITa,
                             __global int *V_IT,
                             __global float *V_W,
                             int vertexOffset, int tableOffset,
                             int start, int end) {

    int i = start + get_global_id(0) + tableOffset;
    int vid = start + get_global_id(0) + vertexOffset;
    int h = V_ITa[5*i];
    int n = V_ITa[5*i+1];
    int p = V_ITa[5*i+2];

    float weight = V_W[i];
    float wp = 1.0f/(float)(n*n);
    float wv = (n-2.0f) * n * wp;

    struct Vertex dst;
    clearVertex(&dst);

    addWithWeight(&dst, &vertex[p], weight * wv);

    for (int j = 0; j < n; ++j) {
        addWithWeight(&dst, &vertex[V_IT[h+j*2]], weight * wp);
        addWithWeight(&dst, &vertex[V_IT[h+j*2+1]], weight * wp);
    }
    vertex[vid] = dst;

    if (varying) {
        struct Varying dstVarying;
        clearVarying(&dstVarying);
        addVaryingWithWeight(&dstVarying, &varying[p], 1.0f);
        varying[vid] = dstVarying;
    }
}

__kernel void computeLoopVertexB(__global struct Vertex *vertex,
                                 __global struct Varying *varying,
                                 __global int *V_ITa,
                                 __global int *V_IT,
                                 __global float *V_W,
                                 int vertexOffset, int tableOffset,
                                 int start, int end) {

    int i = start + get_global_id(0) + tableOffset;
    int vid = start + get_global_id(0) + vertexOffset;
    int h = V_ITa[5*i];
    int n = V_ITa[5*i+1];
    int p = V_ITa[5*i+2];

    float weight = V_W[i];
    float wp = 1.0f/(float)(n);
    float beta = 0.25f * cos((float)(M_PI) * 2.0f * wp) + 0.375f;
    beta = beta * beta;
    beta = (0.625f - beta) * wp;

    struct Vertex dst;
    clearVertex(&dst);
    addWithWeight(&dst, &vertex[p], weight * (1.0f - (beta * n)));

    for (int j = 0; j < n; ++j) {
        addWithWeight(&dst, &vertex[V_IT[h+j]], weight * beta);
    }
    vertex[vid] = dst;

    if (varying) {
        struct Varying dstVarying;
        clearVarying(&dstVarying);
        addVaryingWithWeight(&dstVarying, &varying[p], 1.0f);
        varying[vid] = dstVarying;
    }
}

__kernel void editVertexAdd(__global struct Vertex *vertex,
                            __global int *editIndices,
                            __global float *editValues,
                            int primVarOffset,
                            int primVarWidth,
                            int vertexOffset, int tableOffset,
                            int start, int end) {

    int i = start + get_global_id(0) + tableOffset;
    int v = editIndices[i];
    int eid = start + get_global_id(0);
    struct Vertex dst = vertex[v];

    for (int j = 0; j < primVarWidth; ++j) {
        dst.v[j+primVarOffset] += editValues[eid*primVarWidth + j];
    }
    vertex[v] = dst;
}
