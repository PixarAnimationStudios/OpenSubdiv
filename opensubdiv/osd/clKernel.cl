//
//     Copyright (C) Pixar. All rights reserved.
//
//     This license governs use of the accompanying software. If you
//     use the software, you accept this license. If you do not accept
//     the license, do not use the software.
//
//     1. Definitions
//     The terms "reproduce," "reproduction," "derivative works," and
//     "distribution" have the same meaning here as under U.S.
//     copyright law.  A "contribution" is the original software, or
//     any additions or changes to the software.
//     A "contributor" is any person or entity that distributes its
//     contribution under this license.
//     "Licensed patents" are a contributor's patent claims that read
//     directly on its contribution.
//
//     2. Grant of Rights
//     (A) Copyright Grant- Subject to the terms of this license,
//     including the license conditions and limitations in section 3,
//     each contributor grants you a non-exclusive, worldwide,
//     royalty-free copyright license to reproduce its contribution,
//     prepare derivative works of its contribution, and distribute
//     its contribution or any derivative works that you create.
//     (B) Patent Grant- Subject to the terms of this license,
//     including the license conditions and limitations in section 3,
//     each contributor grants you a non-exclusive, worldwide,
//     royalty-free license under its licensed patents to make, have
//     made, use, sell, offer for sale, import, and/or otherwise
//     dispose of its contribution in the software or derivative works
//     of the contribution in the software.
//
//     3. Conditions and Limitations
//     (A) No Trademark License- This license does not grant you
//     rights to use any contributor's name, logo, or trademarks.
//     (B) If you bring a patent claim against any contributor over
//     patents that you claim are infringed by the software, your
//     patent license from such contributor to the software ends
//     automatically.
//     (C) If you distribute any portion of the software, you must
//     retain all copyright, patent, trademark, and attribution
//     notices that are present in the software.
//     (D) If you distribute any portion of the software in source
//     code form, you may do so only under this license by including a
//     complete copy of this license with your distribution. If you
//     distribute any portion of the software in compiled or object
//     code form, you may only do so under a license that complies
//     with this license.
//     (E) The software is licensed "as-is." You bear the risk of
//     using it. The contributors give no express warranties,
//     guarantees or conditions. You may have additional consumer
//     rights under your local laws which this license cannot change.
//     To the extent permitted under your local laws, the contributors
//     exclude the implied warranties of merchantability, fitness for
//     a particular purpose and non-infringement.
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
