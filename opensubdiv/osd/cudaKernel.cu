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

#include <assert.h>

template<int N> struct DeviceVertex
{
    float v[N];

    __device__ void addWithWeight(const DeviceVertex<N> *src, float weight) {
#pragma unroll
        for(int i = 0; i < N; ++i){
            v[i] += src->v[i] * weight;
        }
    }

    __device__ void clear() {
#pragma unroll
        for(int i = 0; i < N; ++i){
            v[i] = 0.0f;
        }
    }
};

// Specialize DeviceVarying for N=0 to avoid compile error:
// "flexible array member in otherwise empty struct"
template<> struct DeviceVertex<0>
{
    __device__ void addWithWeight(const DeviceVertex<0> *src, float weight) {
    }
    __device__ void clear() {
    }
};

struct DeviceTable
{
    void **tables;
    int *F0_IT;
    int *F0_ITa;
    int *E0_IT;
    int *V0_IT;
    int *V0_ITa;
    float *E0_S;
    float *V0_S;
};

__device__ void clear(float *dst, int count)
{
    for(int i = 0; i < count; ++i) dst[i] = 0;
}

__device__ void addWithWeight(float *dst, float *src, float weight, int count)
{
    for(int i = 0; i < count; ++i) dst[i] += src[i] * weight;
}

template <int NUM_VERTEX_ELEMENTS, int NUM_VARYING_ELEMENTS> __global__ void
computeFace(float *fVertex, float *fVaryings, int *F0_IT, int *F0_ITa, int offset, int tableOffset, int start, int end)
{
    DeviceVertex<NUM_VERTEX_ELEMENTS> *vertex = (DeviceVertex<NUM_VERTEX_ELEMENTS>*)fVertex;
    DeviceVertex<NUM_VARYING_ELEMENTS> *varyings = (DeviceVertex<NUM_VARYING_ELEMENTS>*)fVaryings;
    for (int i = start + tableOffset + threadIdx.x + blockIdx.x*blockDim.x;
         i < end + tableOffset;
         i += blockDim.x * gridDim.x) {

        int h = F0_ITa[2*i];
        int n = F0_ITa[2*i+1];
        float weight = 1.0f/n;

        DeviceVertex<NUM_VERTEX_ELEMENTS> dst;
        dst.clear();

        if(NUM_VARYING_ELEMENTS > 0){
            DeviceVertex<NUM_VARYING_ELEMENTS> dstVarying;
            dstVarying.clear();

            for(int j=0; j<n; ++j){
                int index = F0_IT[h+j];
                dst.addWithWeight(&vertex[index], weight);
                dstVarying.addWithWeight(&varyings[index], weight);
            }
            vertex[offset + i - tableOffset] = dst;
            varyings[offset + i - tableOffset] = dstVarying;
        }else{
            for(int j=0; j<n; ++j){
                int index = F0_IT[h+j];
                dst.addWithWeight(&vertex[index], weight);
            }
            vertex[offset + i - tableOffset] = dst;
        }
    }
}

__global__ void
computeFace(float *fVertex, float *fVarying,
            int vertexLength, int vertexStride,
            int varyingLength, int varyingStride,
            int *F0_IT, int *F0_ITa, int offset, int tableOffset, int start, int end)
{
    for (int i = start + tableOffset +threadIdx.x + blockIdx.x*blockDim.x;
        i < end + tableOffset;
        i += blockDim.x * gridDim.x){

        int h = F0_ITa[2*i];
        int n = F0_ITa[2*i+1];
        float weight = 1.0f/n;

        // XXX: can we use local stack like alloca?
        float *dstVertex = fVertex + (i+offset-tableOffset)*vertexStride;
        clear(dstVertex, vertexLength);
        float *dstVarying = fVarying + (i+offset-tableOffset)*varyingStride;
        clear(dstVarying, varyingLength);

        for(int j=0; j<n; ++j){
            int index = F0_IT[h+j];
            addWithWeight(dstVertex, fVertex + index*vertexStride, weight, vertexLength);
            addWithWeight(dstVarying, fVarying + index*varyingStride, weight, varyingLength);
        }
    }
}

template <int NUM_VERTEX_ELEMENTS, int NUM_VARYING_ELEMENTS> __global__ void
computeQuadFace(float *fVertex, float *fVaryings, int *F0_IT, int offset, int tableOffset, int start, int end)
{
    DeviceVertex<NUM_VERTEX_ELEMENTS> *vertex = (DeviceVertex<NUM_VERTEX_ELEMENTS>*)fVertex;
    DeviceVertex<NUM_VARYING_ELEMENTS> *varyings = (DeviceVertex<NUM_VARYING_ELEMENTS>*)fVaryings;
    for (int i = start + threadIdx.x + blockIdx.x*blockDim.x;
         i < end;
         i += blockDim.x * gridDim.x) {

        int fidx0 = F0_IT[tableOffset + 4 * i + 0];
        int fidx1 = F0_IT[tableOffset + 4 * i + 1];
        int fidx2 = F0_IT[tableOffset + 4 * i + 2];
        int fidx3 = F0_IT[tableOffset + 4 * i + 3];

        DeviceVertex<NUM_VERTEX_ELEMENTS> dst;
        dst.clear();

        if(NUM_VARYING_ELEMENTS > 0){
            DeviceVertex<NUM_VARYING_ELEMENTS> dstVarying;
            dstVarying.clear();

            dst.addWithWeight(&vertex[fidx0], 0.25f);
            dst.addWithWeight(&vertex[fidx1], 0.25f);
            dst.addWithWeight(&vertex[fidx2], 0.25f);
            dst.addWithWeight(&vertex[fidx3], 0.25f);
            dstVarying.addWithWeight(&varyings[fidx0], 0.25f);
            dstVarying.addWithWeight(&varyings[fidx1], 0.25f);
            dstVarying.addWithWeight(&varyings[fidx2], 0.25f);
            dstVarying.addWithWeight(&varyings[fidx3], 0.25f);
            vertex[offset + i] = dst;
            varyings[offset + i] = dstVarying;
        }else{
            dst.addWithWeight(&vertex[fidx0], 0.25f);
            dst.addWithWeight(&vertex[fidx1], 0.25f);
            dst.addWithWeight(&vertex[fidx2], 0.25f);
            dst.addWithWeight(&vertex[fidx3], 0.25f);
            vertex[offset + i] = dst;
        }
    }
}

__global__ void
computeQuadFace(float *fVertex, float *fVarying,
                int vertexLength, int vertexStride,
                int varyingLength, int varyingStride,
                int *F0_IT, int offset, int tableOffset, int start, int end)
{
    for (int i = start +threadIdx.x + blockIdx.x*blockDim.x;
        i < end;
        i += blockDim.x * gridDim.x){

        int fidx0 = F0_IT[tableOffset + 4 * i + 0];
        int fidx1 = F0_IT[tableOffset + 4 * i + 1];
        int fidx2 = F0_IT[tableOffset + 4 * i + 2];
        int fidx3 = F0_IT[tableOffset + 4 * i + 3];

        // XXX: can we use local stack like alloca?
        float *dstVertex = fVertex + (i+offset)*vertexStride;
        clear(dstVertex, vertexLength);
        float *dstVarying = fVarying + (i+offset)*varyingStride;
        clear(dstVarying, varyingLength);

        addWithWeight(dstVertex, fVertex + fidx0*vertexStride, 0.25f, vertexLength);
        addWithWeight(dstVertex, fVertex + fidx1*vertexStride, 0.25f, vertexLength);
        addWithWeight(dstVertex, fVertex + fidx2*vertexStride, 0.25f, vertexLength);
        addWithWeight(dstVertex, fVertex + fidx3*vertexStride, 0.25f, vertexLength);
        addWithWeight(dstVarying, fVarying + fidx0*varyingStride, 0.25f, varyingLength);
        addWithWeight(dstVarying, fVarying + fidx1*varyingStride, 0.25f, varyingLength);
        addWithWeight(dstVarying, fVarying + fidx2*varyingStride, 0.25f, varyingLength);
        addWithWeight(dstVarying, fVarying + fidx3*varyingStride, 0.25f, varyingLength);
    }
}

template <int NUM_VERTEX_ELEMENTS, int NUM_VARYING_ELEMENTS> __global__ void
computeTriQuadFace(float *fVertex, float *fVaryings, int *F0_IT, int offset, int tableOffset, int start, int end)
{
    DeviceVertex<NUM_VERTEX_ELEMENTS> *vertex = (DeviceVertex<NUM_VERTEX_ELEMENTS>*)fVertex;
    DeviceVertex<NUM_VARYING_ELEMENTS> *varyings = (DeviceVertex<NUM_VARYING_ELEMENTS>*)fVaryings;
    for (int i = start + threadIdx.x + blockIdx.x*blockDim.x;
         i < end;
         i += blockDim.x * gridDim.x) {

        int fidx0 = F0_IT[tableOffset + 4 * i + 0];
        int fidx1 = F0_IT[tableOffset + 4 * i + 1];
        int fidx2 = F0_IT[tableOffset + 4 * i + 2];
        int fidx3 = F0_IT[tableOffset + 4 * i + 3];

        bool triangle = (fidx2 == fidx3);
        float weight = triangle ? 1.0f / 3.0f : 1.0f / 4.0f;

        DeviceVertex<NUM_VERTEX_ELEMENTS> dst;
        dst.clear();

        if(NUM_VARYING_ELEMENTS > 0){
            DeviceVertex<NUM_VARYING_ELEMENTS> dstVarying;
            dstVarying.clear();

            dst.addWithWeight(&vertex[fidx0], weight);
            dst.addWithWeight(&vertex[fidx1], weight);
            dst.addWithWeight(&vertex[fidx2], weight);
            dstVarying.addWithWeight(&varyings[fidx0], weight);
            dstVarying.addWithWeight(&varyings[fidx1], weight);
            dstVarying.addWithWeight(&varyings[fidx2], weight);
            if (!triangle) {
                dst.addWithWeight(&vertex[fidx3], weight);
                dstVarying.addWithWeight(&varyings[fidx3], 0.25f);
            }
            vertex[offset + i] = dst;
            varyings[offset + i] = dstVarying;
        }else{
            dst.addWithWeight(&vertex[fidx0], weight);
            dst.addWithWeight(&vertex[fidx1], weight);
            dst.addWithWeight(&vertex[fidx2], weight);
            if (!triangle)
                dst.addWithWeight(&vertex[fidx3], weight);
            vertex[offset + i] = dst;
        }
    }
}

__global__ void
computeTriQuadFace(float *fVertex, float *fVarying,
                   int vertexLength, int vertexStride,
                   int varyingLength, int varyingStride,
                   int *F0_IT, int offset, int tableOffset, int start, int end)
{
    for (int i = start +threadIdx.x + blockIdx.x*blockDim.x;
        i < end;
        i += blockDim.x * gridDim.x){

        int fidx0 = F0_IT[tableOffset + 4 * i + 0];
        int fidx1 = F0_IT[tableOffset + 4 * i + 1];
        int fidx2 = F0_IT[tableOffset + 4 * i + 2];
        int fidx3 = F0_IT[tableOffset + 4 * i + 3];

        bool triangle = (fidx2 == fidx3);
        float weight = triangle ? 1.0f / 3.0f : 1.0f / 4.0f;

        // XXX: can we use local stack like alloca?
        float *dstVertex = fVertex + (i+offset)*vertexStride;
        clear(dstVertex, vertexLength);
        float *dstVarying = fVarying + (i+offset)*varyingStride;
        clear(dstVarying, varyingLength);

        addWithWeight(dstVertex, fVertex + fidx0*vertexStride, weight, vertexLength);
        addWithWeight(dstVertex, fVertex + fidx1*vertexStride, weight, vertexLength);
        addWithWeight(dstVertex, fVertex + fidx2*vertexStride, weight, vertexLength);
        addWithWeight(dstVarying, fVarying + fidx0*varyingStride, weight, varyingLength);
        addWithWeight(dstVarying, fVarying + fidx1*varyingStride, weight, varyingLength);
        addWithWeight(dstVarying, fVarying + fidx2*varyingStride, weight, varyingLength);
        if (!triangle) {
            addWithWeight(dstVertex, fVertex + fidx3*vertexStride, weight, vertexLength);
            addWithWeight(dstVarying, fVarying + fidx3*varyingStride, weight, varyingLength);
        }
    }
}

template <int NUM_VERTEX_ELEMENTS, int NUM_VARYING_ELEMENTS> __global__ void
computeEdge(float *fVertex, float *fVaryings, int *E0_IT, float *E0_S, int offset, int tableOffset, int start, int end)
{
    DeviceVertex<NUM_VERTEX_ELEMENTS> *vertex = (DeviceVertex<NUM_VERTEX_ELEMENTS>*)fVertex;
    DeviceVertex<NUM_VARYING_ELEMENTS> *varyings = (DeviceVertex<NUM_VARYING_ELEMENTS>*)fVaryings;

    for (int i = start + tableOffset + threadIdx.x + blockIdx.x*blockDim.x;
         i < end + tableOffset;
         i+= blockDim.x * gridDim.x){

        int eidx0 = E0_IT[4*i+0];
        int eidx1 = E0_IT[4*i+1];
        int eidx2 = E0_IT[4*i+2];
        int eidx3 = E0_IT[4*i+3];

        float vertWeight = E0_S[i*2+0];

        // Fully sharp edge : vertWeight = 0.5f;
        DeviceVertex<NUM_VERTEX_ELEMENTS> dst;
        dst.clear();

        dst.addWithWeight(&vertex[eidx0], vertWeight);
        dst.addWithWeight(&vertex[eidx1], vertWeight);

        if(eidx2 > -1){
            float faceWeight = E0_S[i*2+1];

            dst.addWithWeight(&vertex[eidx2], faceWeight);
            dst.addWithWeight(&vertex[eidx3], faceWeight);
        }
        vertex[offset+i-tableOffset] = dst;

        if(NUM_VARYING_ELEMENTS > 0){
            DeviceVertex<NUM_VARYING_ELEMENTS> dstVarying;
            dstVarying.clear();
            dstVarying.addWithWeight(&varyings[eidx0], 0.5f);
            dstVarying.addWithWeight(&varyings[eidx1], 0.5f);
            varyings[offset+i-tableOffset] = dstVarying;
        }
    }
}

__global__ void
computeEdge(float *fVertex, float *fVarying,
            int vertexLength, int vertexStride,
            int varyingLength, int varyingStride,
            int *E0_IT, float *E0_S, int offset, int tableOffset, int start, int end)
{
    for (int i = start + tableOffset + threadIdx.x + blockIdx.x*blockDim.x;
         i < end + tableOffset;i+= blockDim.x * gridDim.x) {

        int eidx0 = E0_IT[4*i+0];
        int eidx1 = E0_IT[4*i+1];
        int eidx2 = E0_IT[4*i+2];
        int eidx3 = E0_IT[4*i+3];

        float vertWeight = E0_S[i*2+0];

        // Fully sharp edge : vertWeight = 0.5f;
        float *dstVertex = fVertex + (i+offset-tableOffset)*vertexStride;
        clear(dstVertex, vertexLength);

        addWithWeight(dstVertex, fVertex + eidx0*vertexStride, vertWeight, vertexLength);
        addWithWeight(dstVertex, fVertex + eidx1*vertexStride, vertWeight, vertexLength);

        if(eidx2 > -1){
            float faceWeight = E0_S[i*2+1];

            addWithWeight(dstVertex, fVertex + eidx2*vertexStride, faceWeight, vertexLength);
            addWithWeight(dstVertex, fVertex + eidx3*vertexStride, faceWeight, vertexLength);
        }

        if (varyingLength > 0){
            float *dstVarying = fVarying + (i+offset-tableOffset)*varyingStride;
            clear(dstVarying, varyingLength);

            addWithWeight(dstVarying, fVarying + eidx0*varyingStride, 0.5f, varyingLength);
            addWithWeight(dstVarying, fVarying + eidx1*varyingStride, 0.5f, varyingLength);
        }
    }
}

template <int NUM_VERTEX_ELEMENTS, int NUM_VARYING_ELEMENTS> __global__ void
computeRestrictedEdge(float *fVertex, float *fVaryings, int *E0_IT, int offset, int tableOffset, int start, int end)
{
    DeviceVertex<NUM_VERTEX_ELEMENTS> *vertex = (DeviceVertex<NUM_VERTEX_ELEMENTS>*)fVertex;
    DeviceVertex<NUM_VARYING_ELEMENTS> *varyings = (DeviceVertex<NUM_VARYING_ELEMENTS>*)fVaryings;

    for (int i = start + tableOffset + threadIdx.x + blockIdx.x*blockDim.x;
         i < end + tableOffset;
         i+= blockDim.x * gridDim.x){

        int eidx0 = E0_IT[4*i+0];
        int eidx1 = E0_IT[4*i+1];
        int eidx2 = E0_IT[4*i+2];
        int eidx3 = E0_IT[4*i+3];

        DeviceVertex<NUM_VERTEX_ELEMENTS> dst;
        dst.clear();
        dst.addWithWeight(&vertex[eidx0], 0.25f);
        dst.addWithWeight(&vertex[eidx1], 0.25f);
        dst.addWithWeight(&vertex[eidx2], 0.25f);
        dst.addWithWeight(&vertex[eidx3], 0.25f);
        vertex[offset+i-tableOffset] = dst;

        if(NUM_VARYING_ELEMENTS > 0){
            DeviceVertex<NUM_VARYING_ELEMENTS> dstVarying;
            dstVarying.clear();
            dstVarying.addWithWeight(&varyings[eidx0], 0.5f);
            dstVarying.addWithWeight(&varyings[eidx1], 0.5f);
            varyings[offset+i-tableOffset] = dstVarying;
        }
    }
}

__global__ void
computeRestrictedEdge(float *fVertex, float *fVarying,
                      int vertexLength, int vertexStride,
                      int varyingLength, int varyingStride,
                      int *E0_IT, int offset, int tableOffset, int start, int end)
{
    for (int i = start + tableOffset + threadIdx.x + blockIdx.x*blockDim.x;
         i < end + tableOffset;i+= blockDim.x * gridDim.x) {

        int eidx0 = E0_IT[4*i+0];
        int eidx1 = E0_IT[4*i+1];
        int eidx2 = E0_IT[4*i+2];
        int eidx3 = E0_IT[4*i+3];

        float *dstVertex = fVertex + (i+offset-tableOffset)*vertexStride;
        clear(dstVertex, vertexLength);

        addWithWeight(dstVertex, fVertex + eidx0*vertexStride, 0.25f, vertexLength);
        addWithWeight(dstVertex, fVertex + eidx1*vertexStride, 0.25f, vertexLength);
        addWithWeight(dstVertex, fVertex + eidx2*vertexStride, 0.25f, vertexLength);
        addWithWeight(dstVertex, fVertex + eidx3*vertexStride, 0.25f, vertexLength);

        if (varyingLength > 0){
            float *dstVarying = fVarying + (i+offset-tableOffset)*varyingStride;
            clear(dstVarying, varyingLength);

            addWithWeight(dstVarying, fVarying + eidx0*varyingStride, 0.5f, varyingLength);
            addWithWeight(dstVarying, fVarying + eidx1*varyingStride, 0.5f, varyingLength);
        }
    }
}

template <int NUM_VERTEX_ELEMENTS, int NUM_VARYING_ELEMENTS> __global__ void
computeVertexA(float *fVertex, float *fVaryings, int *V0_ITa, float *V0_S, int offset, int tableOffset, int start, int end, int pass)
{
    DeviceVertex<NUM_VERTEX_ELEMENTS> *vertex = (DeviceVertex<NUM_VERTEX_ELEMENTS>*)fVertex;
    DeviceVertex<NUM_VARYING_ELEMENTS> *varyings = (DeviceVertex<NUM_VARYING_ELEMENTS>*)fVaryings;
    for (int i = start + tableOffset + threadIdx.x + blockIdx.x*blockDim.x;
         i < end+tableOffset;
         i += blockDim.x * gridDim.x) {

        int n     = V0_ITa[5*i+1];
        int p     = V0_ITa[5*i+2];
        int eidx0 = V0_ITa[5*i+3];
        int eidx1 = V0_ITa[5*i+4];

        float weight = (pass==1) ? V0_S[i] : 1.0f - V0_S[i];

        // In the case of fractional weight, the weight must be inverted since
        // the value is shared with the k_Smooth kernel (statistically the
        // k_Smooth kernel runs much more often than this one)
        if (weight>0.0f && weight<1.0f && n > 0)
            weight=1.0f-weight;

        DeviceVertex<NUM_VERTEX_ELEMENTS> dst;
        if (not pass) {
            dst.clear();
        } else {
            dst = vertex[i+offset-tableOffset];
        }

        if (eidx0==-1 || (pass==0 && (n==-1)) ) {
            dst.addWithWeight(&vertex[p], weight);
        } else {
            dst.addWithWeight(&vertex[p], weight * 0.75f);
            dst.addWithWeight(&vertex[eidx0], weight * 0.125f);
            dst.addWithWeight(&vertex[eidx1], weight * 0.125f);
        }
        vertex[i+offset-tableOffset] = dst;

        if(NUM_VARYING_ELEMENTS > 0){
            if(not pass){
                DeviceVertex<NUM_VARYING_ELEMENTS> dstVarying;
                dstVarying.clear();
                dstVarying.addWithWeight(&varyings[p], 1.0f);
                varyings[i+offset-tableOffset] = dstVarying;
            }
        }
    }
}

__global__ void
computeVertexA(float *fVertex, float *fVaryings,
               int vertexLength, int vertexStride,
               int varyingLength, int varyingStride,
               int *V0_ITa, float *V0_S, int offset, int tableOffset, int start, int end, int pass)
{
    for (int i = start + tableOffset + threadIdx.x + blockIdx.x*blockDim.x;
         i < end + tableOffset;
         i += blockDim.x * gridDim.x){

        int n     = V0_ITa[5*i+1];
        int p     = V0_ITa[5*i+2];
        int eidx0 = V0_ITa[5*i+3];
        int eidx1 = V0_ITa[5*i+4];

        float weight = (pass==1) ? V0_S[i] : 1.0f - V0_S[i];

        // In the case of fractional weight, the weight must be inverted since
        // the value is shared with the k_Smooth kernel (statistically the
        // k_Smooth kernel runs much more often than this one)
        if (weight>0.0f && weight<1.0f && n > 0)
            weight=1.0f-weight;

        float *dstVertex = fVertex + (i+offset-tableOffset)*vertexStride;
        if (not pass) {
            clear(dstVertex, vertexLength);
        }

        if (eidx0==-1 || (pass==0 && (n==-1)) ) {
            addWithWeight(dstVertex, fVertex + p*vertexStride, weight, vertexLength);
        } else {
            addWithWeight(dstVertex, fVertex + p*vertexStride, weight*0.75f, vertexLength);
            addWithWeight(dstVertex, fVertex + eidx0*vertexStride, weight*0.125f, vertexLength);
            addWithWeight(dstVertex, fVertex + eidx1*vertexStride, weight*0.125f, vertexLength);
        }

        if(varyingLength > 0){
            if(not pass){
                float *dstVarying = fVaryings + (i+offset-tableOffset)*varyingStride;
                clear(dstVarying, varyingLength);
                addWithWeight(dstVarying, fVaryings + p*varyingStride, 1.0f, varyingLength);
            }
        }
    }

}


//texture <int, 1> texV0_IT;

template <int NUM_VERTEX_ELEMENTS, int NUM_VARYING_ELEMENTS> __global__ void
computeVertexB(float *fVertex, float *fVaryings,
               const int *V0_ITa, const int *V0_IT, const float *V0_S, int offset, int tableOffset, int start, int end)
{
    DeviceVertex<NUM_VERTEX_ELEMENTS> *vertex = (DeviceVertex<NUM_VERTEX_ELEMENTS>*)fVertex;
    DeviceVertex<NUM_VARYING_ELEMENTS> *varyings = (DeviceVertex<NUM_VARYING_ELEMENTS>*)fVaryings;
    for (int i = start + tableOffset + threadIdx.x + blockIdx.x*blockDim.x;
         i < end + tableOffset;
         i += blockDim.x * gridDim.x) {

        int h = V0_ITa[5*i];
        int n = V0_ITa[5*i+1];
        int p = V0_ITa[5*i+2];

        float weight = V0_S[i];
        float wp = 1.0f/float(n*n);
        float wv = (n-2.0f) * n * wp;

        DeviceVertex<NUM_VERTEX_ELEMENTS> dst;
        dst.clear();
        dst.addWithWeight(&vertex[p], weight * wv);

        for (int j = 0; j < n; ++j) {
            dst.addWithWeight(&vertex[V0_IT[h+j*2]], weight * wp);
            dst.addWithWeight(&vertex[V0_IT[h+j*2+1]], weight * wp);
//            int idx0 = tex1Dfetch(texV0_IT, h+j*2);
//            int idx1 = tex1Dfetch(texV0_IT, h+j*2+1);
//            dst.addWithWeight(&vertex[idx0], weight * wp);
//            dst.addWithWeight(&vertex[idx1], weight * wp);
        }
        vertex[i+offset-tableOffset] = dst;

        if(NUM_VARYING_ELEMENTS > 0){
            DeviceVertex<NUM_VARYING_ELEMENTS> dstVarying;
            dstVarying.clear();
            dstVarying.addWithWeight(&varyings[p], 1.0f);
            varyings[i+offset-tableOffset] = dstVarying;
        }
    }
}

__global__ void
computeVertexB(float *fVertex, float *fVarying,
               int vertexLength, int vertexStride,
               int varyingLength, int varyingStride,
               const int *V0_ITa, const int *V0_IT, const float *V0_S, int offset, int tableOffset, int start, int end)
{
    for (int i = start + tableOffset + threadIdx.x + blockIdx.x*blockDim.x;
         i < end + tableOffset;
         i += blockDim.x * gridDim.x) {

        int h = V0_ITa[5*i];
        int n = V0_ITa[5*i+1];
        int p = V0_ITa[5*i+2];

        float weight = V0_S[i];
        float wp = 1.0f/float(n*n);
        float wv = (n-2.0f) * n * wp;

        float *dstVertex = fVertex + (i+offset-tableOffset)*vertexStride;
        clear(dstVertex, vertexLength);
        addWithWeight(dstVertex, fVertex + p*vertexStride, weight*wv, vertexLength);

        for (int j = 0; j < n; ++j) {
            addWithWeight(dstVertex, fVertex + V0_IT[h+j*2]*vertexStride, weight*wp, vertexLength);
            addWithWeight(dstVertex, fVertex + V0_IT[h+j*2+1]*vertexStride, weight*wp, vertexLength);
        }

        if (varyingLength > 0) {
            float *dstVarying = fVarying + (i+offset-tableOffset)*varyingStride;
            clear(dstVarying, varyingLength);
            addWithWeight(dstVarying, fVarying + p*varyingStride, 1.0f, varyingLength);
        }
    }
}

template <int NUM_VERTEX_ELEMENTS, int NUM_VARYING_ELEMENTS> __global__ void
computeRestrictedVertexA(float *fVertex, float *fVaryings, int *V0_ITa, int offset, int tableOffset, int start, int end)
{
    DeviceVertex<NUM_VERTEX_ELEMENTS> *vertex = (DeviceVertex<NUM_VERTEX_ELEMENTS>*)fVertex;
    DeviceVertex<NUM_VARYING_ELEMENTS> *varyings = (DeviceVertex<NUM_VARYING_ELEMENTS>*)fVaryings;
    for (int i = start + tableOffset + threadIdx.x + blockIdx.x*blockDim.x;
         i < end+tableOffset;
         i += blockDim.x * gridDim.x) {

        int p     = V0_ITa[5*i+2];
        int eidx0 = V0_ITa[5*i+3];
        int eidx1 = V0_ITa[5*i+4];

        DeviceVertex<NUM_VERTEX_ELEMENTS> dst;
        dst.clear();

        dst.addWithWeight(&vertex[p], 0.75f);
        dst.addWithWeight(&vertex[eidx0], 0.125f);
        dst.addWithWeight(&vertex[eidx1], 0.125f);
        vertex[i+offset-tableOffset] = dst;

        if(NUM_VARYING_ELEMENTS > 0){
            DeviceVertex<NUM_VARYING_ELEMENTS> dstVarying;
            dstVarying.clear();
            dstVarying.addWithWeight(&varyings[p], 1.0f);
            varyings[i+offset-tableOffset] = dstVarying;
        }
    }
}

__global__ void
computeRestrictedVertexA(float *fVertex, float *fVaryings,
                         int vertexLength, int vertexStride,
                         int varyingLength, int varyingStride,
                         int *V0_ITa, int offset, int tableOffset, int start, int end)
{
    for (int i = start + tableOffset + threadIdx.x + blockIdx.x*blockDim.x;
         i < end + tableOffset;
         i += blockDim.x * gridDim.x){

        int p     = V0_ITa[5*i+2];
        int eidx0 = V0_ITa[5*i+3];
        int eidx1 = V0_ITa[5*i+4];

        float *dstVertex = fVertex + (i+offset-tableOffset)*vertexStride;
        clear(dstVertex, vertexLength);

        addWithWeight(dstVertex, fVertex + p*vertexStride, 0.75f, vertexLength);
        addWithWeight(dstVertex, fVertex + eidx0*vertexStride, 0.125f, vertexLength);
        addWithWeight(dstVertex, fVertex + eidx1*vertexStride, 0.125f, vertexLength);

        if(varyingLength > 0){
            float *dstVarying = fVaryings + (i+offset-tableOffset)*varyingStride;
            clear(dstVarying, varyingLength);
            addWithWeight(dstVarying, fVaryings + p*varyingStride, 1.0f, varyingLength);
        }
    }
}

template <int NUM_VERTEX_ELEMENTS, int NUM_VARYING_ELEMENTS> __global__ void
computeRestrictedVertexB1(float *fVertex, float *fVaryings,
                          const int *V0_ITa, const int *V0_IT, int offset, int tableOffset, int start, int end)
{
    DeviceVertex<NUM_VERTEX_ELEMENTS> *vertex = (DeviceVertex<NUM_VERTEX_ELEMENTS>*)fVertex;
    DeviceVertex<NUM_VARYING_ELEMENTS> *varyings = (DeviceVertex<NUM_VARYING_ELEMENTS>*)fVaryings;
    for (int i = start + tableOffset + threadIdx.x + blockIdx.x*blockDim.x;
         i < end + tableOffset;
         i += blockDim.x * gridDim.x) {

        int h = V0_ITa[5*i];
        int p = V0_ITa[5*i+2];

        DeviceVertex<NUM_VERTEX_ELEMENTS> dst;
        dst.clear();
        dst.addWithWeight(&vertex[p], 0.5f);

        for (int j = 0; j < 8; ++j)
            dst.addWithWeight(&vertex[V0_IT[h+j]], 0.0625f);
        vertex[i+offset-tableOffset] = dst;

        if(NUM_VARYING_ELEMENTS > 0){
            DeviceVertex<NUM_VARYING_ELEMENTS> dstVarying;
            dstVarying.clear();
            dstVarying.addWithWeight(&varyings[p], 1.0f);
            varyings[i+offset-tableOffset] = dstVarying;
        }
    }
}

__global__ void
computeRestrictedVertexB1(float *fVertex, float *fVarying,
                          int vertexLength, int vertexStride,
                          int varyingLength, int varyingStride,
                          const int *V0_ITa, const int *V0_IT, int offset, int tableOffset, int start, int end)
{
    for (int i = start + tableOffset + threadIdx.x + blockIdx.x*blockDim.x;
         i < end + tableOffset;
         i += blockDim.x * gridDim.x) {

        int h = V0_ITa[5*i];
        int p = V0_ITa[5*i+2];

        float *dstVertex = fVertex + (i+offset-tableOffset)*vertexStride;
        clear(dstVertex, vertexLength);
        addWithWeight(dstVertex, fVertex + p*vertexStride, 0.5f, vertexLength);

        for (int j = 0; j < 8; ++j)
            addWithWeight(dstVertex, fVertex + V0_IT[h+j]*vertexStride, 0.0625f, vertexLength);

        if (varyingLength > 0) {
            float *dstVarying = fVarying + (i+offset-tableOffset)*varyingStride;
            clear(dstVarying, varyingLength);
            addWithWeight(dstVarying, fVarying + p*varyingStride, 1.0f, varyingLength);
        }
    }
}

template <int NUM_VERTEX_ELEMENTS, int NUM_VARYING_ELEMENTS> __global__ void
computeRestrictedVertexB2(float *fVertex, float *fVaryings,
                          const int *V0_ITa, const int *V0_IT, int offset, int tableOffset, int start, int end)
{
    DeviceVertex<NUM_VERTEX_ELEMENTS> *vertex = (DeviceVertex<NUM_VERTEX_ELEMENTS>*)fVertex;
    DeviceVertex<NUM_VARYING_ELEMENTS> *varyings = (DeviceVertex<NUM_VARYING_ELEMENTS>*)fVaryings;
    for (int i = start + tableOffset + threadIdx.x + blockIdx.x*blockDim.x;
         i < end + tableOffset;
         i += blockDim.x * gridDim.x) {

        int h = V0_ITa[5*i];
        int n = V0_ITa[5*i+1];
        int p = V0_ITa[5*i+2];

        float wp = 1.0f/float(n*n);
        float wv = (n-2.0f) * n * wp;

        DeviceVertex<NUM_VERTEX_ELEMENTS> dst;
        dst.clear();
        dst.addWithWeight(&vertex[p], wv);

        for (int j = 0; j < n; ++j) {
            dst.addWithWeight(&vertex[V0_IT[h+j*2]], wp);
            dst.addWithWeight(&vertex[V0_IT[h+j*2+1]], wp);
        }
        vertex[i+offset-tableOffset] = dst;

        if(NUM_VARYING_ELEMENTS > 0){
            DeviceVertex<NUM_VARYING_ELEMENTS> dstVarying;
            dstVarying.clear();
            dstVarying.addWithWeight(&varyings[p], 1.0f);
            varyings[i+offset-tableOffset] = dstVarying;
        }
    }
}

__global__ void
computeRestrictedVertexB2(float *fVertex, float *fVarying,
                          int vertexLength, int vertexStride,
                          int varyingLength, int varyingStride,
                          const int *V0_ITa, const int *V0_IT, int offset, int tableOffset, int start, int end)
{
    for (int i = start + tableOffset + threadIdx.x + blockIdx.x*blockDim.x;
         i < end + tableOffset;
         i += blockDim.x * gridDim.x) {

        int h = V0_ITa[5*i];
        int n = V0_ITa[5*i+1];
        int p = V0_ITa[5*i+2];

        float wp = 1.0f/float(n*n);
        float wv = (n-2.0f) * n * wp;

        float *dstVertex = fVertex + (i+offset-tableOffset)*vertexStride;
        clear(dstVertex, vertexLength);
        addWithWeight(dstVertex, fVertex + p*vertexStride, wv, vertexLength);

        for (int j = 0; j < n; ++j) {
            addWithWeight(dstVertex, fVertex + V0_IT[h+j*2]*vertexStride, wp, vertexLength);
            addWithWeight(dstVertex, fVertex + V0_IT[h+j*2+1]*vertexStride, wp, vertexLength);
        }

        if (varyingLength > 0) {
            float *dstVarying = fVarying + (i+offset-tableOffset)*varyingStride;
            clear(dstVarying, varyingLength);
            addWithWeight(dstVarying, fVarying + p*varyingStride, 1.0f, varyingLength);
        }
    }
}

// --------------------------------------------------------------------------------------------

template <int NUM_VERTEX_ELEMENTS, int NUM_VARYING_ELEMENTS> __global__ void
computeLoopVertexB(float *fVertex, float *fVaryings, int *V0_ITa, int *V0_IT, float *V0_S, int offset, int tableOffset, int start, int end)
{
    DeviceVertex<NUM_VERTEX_ELEMENTS> *vertex = (DeviceVertex<NUM_VERTEX_ELEMENTS>*)fVertex;
    DeviceVertex<NUM_VARYING_ELEMENTS> *varyings = (DeviceVertex<NUM_VARYING_ELEMENTS>*)fVaryings;
    for (int i = start + tableOffset + threadIdx.x + blockIdx.x*blockDim.x;
         i < end + tableOffset;
         i += blockDim.x * gridDim.x) {

        int h = V0_ITa[5*i];
        int n = V0_ITa[5*i+1];
        int p = V0_ITa[5*i+2];

        float weight = V0_S[i];
        float wp = 1.0f/float(n);
        float beta = 0.25f * __cosf(float(M_PI) * 2.0f * wp) + 0.375f;
        beta = beta * beta;
        beta = (0.625f - beta) * wp;

        DeviceVertex<NUM_VERTEX_ELEMENTS> dst;
        dst.clear();

        dst.addWithWeight(&vertex[p], weight * (1.0f - (beta * n)));

        for (int j = 0; j < n; ++j) {
            dst.addWithWeight(&vertex[V0_IT[h+j]], weight * beta);
        }
        vertex[i+offset-tableOffset] = dst;

        if (NUM_VARYING_ELEMENTS > 0) {
            DeviceVertex<NUM_VARYING_ELEMENTS> dstVarying;
            dstVarying.clear();
            dstVarying.addWithWeight(&varyings[p], 1.0f);
            varyings[i+offset-tableOffset] = dstVarying;
        }
    }
}

__global__ void
computeLoopVertexB(float *fVertex, float *fVarying,
                   int vertexLength, int vertexStride,
                   int varyingLength, int varyingStride,
                   const int *V0_ITa, const int *V0_IT, const float *V0_S, int offset, int tableOffset, int start, int end)
{
    for (int i = start + tableOffset + threadIdx.x + blockIdx.x*blockDim.x;
         i < end + tableOffset;
         i += blockDim.x * gridDim.x) {

        int h = V0_ITa[5*i];
        int n = V0_ITa[5*i+1];
        int p = V0_ITa[5*i+2];

        float weight = V0_S[i];
        float wp = 1.0f/float(n);
        float beta = 0.25f * __cosf(float(M_PI) * 2.0f * wp) + 0.375f;
        beta = beta * beta;
        beta = (0.625f - beta) * wp;

        float *dstVertex = fVertex + (i+offset-tableOffset)*vertexStride;
        clear(dstVertex, vertexLength);
        addWithWeight(dstVertex, fVertex + p*vertexStride, weight*(1.0f-(beta*n)), vertexLength);

        for (int j = 0; j < n; ++j) {
            addWithWeight(dstVertex, fVertex + V0_IT[h+j]*vertexStride, weight*beta, vertexLength);
        }

        if (varyingLength > 0) {
            float *dstVarying = fVarying + (i+offset-tableOffset)*varyingStride;
            clear(dstVarying, varyingLength);
            addWithWeight(dstVarying, fVarying + p*varyingStride, 1.0f, varyingLength);
        }
    }
}

// --------------------------------------------------------------------------------------------

template <int NUM_VERTEX_ELEMENTS, int NUM_VARYING_ELEMENTS> __global__ void
computeBilinearEdge(float *fVertex, float *fVaryings, int *E0_IT, int offset, int tableOffset, int start, int end)
{
    DeviceVertex<NUM_VERTEX_ELEMENTS> *vertex = (DeviceVertex<NUM_VERTEX_ELEMENTS>*)fVertex;
    DeviceVertex<NUM_VARYING_ELEMENTS> *varyings = (DeviceVertex<NUM_VARYING_ELEMENTS>*)fVaryings;
    for (int i = start + tableOffset + threadIdx.x + blockIdx.x*blockDim.x;
         i < end + tableOffset;
         i+= blockDim.x * gridDim.x) {

        int eidx0 = E0_IT[2*i+0];
        int eidx1 = E0_IT[2*i+1];

        DeviceVertex<NUM_VERTEX_ELEMENTS> dst;
        dst.clear();

        dst.addWithWeight(&vertex[eidx0], 0.5f);
        dst.addWithWeight(&vertex[eidx1], 0.5f);

        vertex[offset+i-tableOffset] = dst;

        if (NUM_VARYING_ELEMENTS > 0) {
            DeviceVertex<NUM_VARYING_ELEMENTS> dstVarying;
            dstVarying.clear();
            dstVarying.addWithWeight(&varyings[eidx0], 0.5f);
            dstVarying.addWithWeight(&varyings[eidx1], 0.5f);
            varyings[offset+i-tableOffset] = dstVarying;
        }
    }
}

__global__ void
computeBilinearEdge(float *fVertex, float *fVarying,
                    int vertexLength, int vertexStride,
                    int varyingLength, int varyingStride,
                    int *E0_IT, int offset, int tableOffset, int start, int end)
{
    for (int i = start + tableOffset + threadIdx.x + blockIdx.x*blockDim.x;
         i < end + tableOffset;
         i+= blockDim.x * gridDim.x) {

        int eidx0 = E0_IT[2*i+0];
        int eidx1 = E0_IT[2*i+1];

        float *dstVertex = fVertex + (i+offset-tableOffset)*vertexStride;
        clear(dstVertex, vertexLength);

        addWithWeight(dstVertex, fVertex + eidx0*vertexStride, 0.5f, vertexLength);
        addWithWeight(dstVertex, fVertex + eidx1*vertexStride, 0.5f, vertexLength);

        if (varyingLength > 0) {
            float *dstVarying = fVarying + (i+offset-tableOffset)*varyingStride;
            clear(dstVarying, varyingLength);

            addWithWeight(dstVarying, fVarying + eidx0*varyingStride, 0.5f, varyingLength);
            addWithWeight(dstVarying, fVarying + eidx1*varyingStride, 0.5f, varyingLength);
        }
    }
}

template <int NUM_VERTEX_ELEMENTS, int NUM_VARYING_ELEMENTS> __global__ void
computeBilinearVertex(float *fVertex, float *fVaryings, int *V0_ITa, int offset, int tableOffset, int start, int end)
{
    DeviceVertex<NUM_VERTEX_ELEMENTS> *vertex = (DeviceVertex<NUM_VERTEX_ELEMENTS>*)fVertex;
    DeviceVertex<NUM_VARYING_ELEMENTS> *varyings = (DeviceVertex<NUM_VARYING_ELEMENTS>*)fVaryings;
    for (int i = start + tableOffset + threadIdx.x + blockIdx.x*blockDim.x;
         i < end + tableOffset;
         i += blockDim.x * gridDim.x) {

        int p = V0_ITa[i];

        DeviceVertex<NUM_VERTEX_ELEMENTS> dst;
        dst.clear();

        dst.addWithWeight(&vertex[p], 1.0f);
        vertex[i+offset-tableOffset] = dst;

        if (NUM_VARYING_ELEMENTS > 0) {
            DeviceVertex<NUM_VARYING_ELEMENTS> dstVarying;
            dstVarying.clear();
            dstVarying.addWithWeight(&varyings[p], 1.0f);
            varyings[i+offset-tableOffset] = dstVarying;
        }
    }
}

__global__ void
computeBilinearVertex(float *fVertex, float *fVarying,
                      int vertexLength, int vertexStride,
                      int varyingLength, int varyingStride,
                      const int *V0_ITa, int offset, int tableOffset, int start, int end)
{
    for (int i = start + tableOffset + threadIdx.x + blockIdx.x*blockDim.x;
         i < end + tableOffset;
         i += blockDim.x * gridDim.x) {

        int p = V0_ITa[i];

        float *dstVertex = fVertex + (i+offset-tableOffset)*vertexStride;
        clear(dstVertex, vertexLength);
        addWithWeight(dstVertex, fVertex + p*vertexStride, 1.0f, vertexLength);

        if (varyingLength > 0) {
            float *dstVarying = fVarying + (i+offset-tableOffset)*varyingStride;
            clear(dstVarying, varyingLength);
            addWithWeight(dstVarying, fVarying + p*varyingStride, 1.0f, varyingLength);
        }
    }
}

// --------------------------------------------------------------------------------------------

__global__ void
editVertexAdd(float *fVertex, int vertexLength, int vertexStride,
              int primVarOffset, int primVarWidth,
              int vertexOffset, int tableOffset, int start, int end,
              const int *editIndices, const float *editValues)
{
    for (int i = start + tableOffset + threadIdx.x + blockIdx.x*blockDim.x;
         i < end + tableOffset;
         i += blockDim.x * gridDim.x) {

        float *dstVertex = fVertex + (editIndices[i] + vertexOffset) * vertexStride + primVarOffset;

        for(int j = 0; j < primVarWidth; j++) {
            *dstVertex++ += editValues[i*primVarWidth + j];
        }
    }
}

// --------------------------------------------------------------------------------------------

#include "../version.h"

// XXX: this macro usage is tentative. Since cuda kernel can't be dynamically configured,
// still trying to find better way to have optimized kernel..

#define OPT_KERNEL(NUM_VERTEX_ELEMENTS, NUM_VARYING_ELEMENTS, KERNEL, X, Y, ARG) \
    if(vertexLength == NUM_VERTEX_ELEMENTS &&                           \
       varyingLength == NUM_VARYING_ELEMENTS &&                         \
       vertexStride == vertexLength &&                                  \
       varyingStride == varyingLength)                                  \
    { KERNEL<NUM_VERTEX_ELEMENTS, NUM_VARYING_ELEMENTS><<<X,Y>>>ARG;    \
        return;  }

extern "C" {

void OsdCudaComputeFace(float *vertex, float *varying,
                        int vertexLength, int vertexStride,
                        int varyingLength, int varyingStride,
                        int *F_IT, int *F_ITa, int offset, int tableOffset, int start, int end)
{
    //computeFace<3, 0><<<512,32>>>(vertex, varying, F_IT, F_ITa, offset, start, end);
    OPT_KERNEL(0, 0, computeFace, 512, 32, (vertex, varying, F_IT, F_ITa, offset, tableOffset, start, end));
    OPT_KERNEL(0, 3, computeFace, 512, 32, (vertex, varying, F_IT, F_ITa, offset, tableOffset, start, end));
    OPT_KERNEL(3, 0, computeFace, 512, 32, (vertex, varying, F_IT, F_ITa, offset, tableOffset, start, end));
    OPT_KERNEL(3, 3, computeFace, 512, 32, (vertex, varying, F_IT, F_ITa, offset, tableOffset, start, end));

    // fallback kernel (slow)
    computeFace<<<512, 32>>>(vertex, varying,
                             vertexLength, vertexStride, varyingLength, varyingStride,
                             F_IT, F_ITa, offset, tableOffset, start, end);
}

void OsdCudaComputeQuadFace(float *vertex, float *varying,
                            int vertexLength, int vertexStride,
                            int varyingLength, int varyingStride,
                            int *F_IT, int offset, int tableOffset, int start, int end)
{
    //computeQuadFace<3, 0><<<512,32>>>(vertex, varying, F_IT, offset, start, end);
    OPT_KERNEL(0, 0, computeQuadFace, 512, 32, (vertex, varying, F_IT, offset, tableOffset, start, end));
    OPT_KERNEL(0, 3, computeQuadFace, 512, 32, (vertex, varying, F_IT, offset, tableOffset, start, end));
    OPT_KERNEL(3, 0, computeQuadFace, 512, 32, (vertex, varying, F_IT, offset, tableOffset, start, end));
    OPT_KERNEL(3, 3, computeQuadFace, 512, 32, (vertex, varying, F_IT, offset, tableOffset, start, end));

    // fallback kernel (slow)
    computeQuadFace<<<512, 32>>>(vertex, varying,
                                 vertexLength, vertexStride, varyingLength, varyingStride,
                                 F_IT, offset, tableOffset, start, end);
}

void OsdCudaComputeTriQuadFace(float *vertex, float *varying,
                               int vertexLength, int vertexStride,
                               int varyingLength, int varyingStride,
                               int *F_IT, int offset, int tableOffset, int start, int end)
{
    //computeTriQuadFace<3, 0><<<512,32>>>(vertex, varying, F_IT, offset, start, end);
    OPT_KERNEL(0, 0, computeTriQuadFace, 512, 32, (vertex, varying, F_IT, offset, tableOffset, start, end));
    OPT_KERNEL(0, 3, computeTriQuadFace, 512, 32, (vertex, varying, F_IT, offset, tableOffset, start, end));
    OPT_KERNEL(3, 0, computeTriQuadFace, 512, 32, (vertex, varying, F_IT, offset, tableOffset, start, end));
    OPT_KERNEL(3, 3, computeTriQuadFace, 512, 32, (vertex, varying, F_IT, offset, tableOffset, start, end));

    // fallback kernel (slow)
    computeTriQuadFace<<<512, 32>>>(vertex, varying,
                                    vertexLength, vertexStride, varyingLength, varyingStride,
                                    F_IT, offset, tableOffset, start, end);
}


void OsdCudaComputeEdge(float *vertex, float *varying,
                        int vertexLength, int vertexStride,
                        int varyingLength, int varyingStride,
                        int *E_IT, float *E_W, int offset, int tableOffset, int start, int end)
{
    //computeEdge<0, 3><<<512,32>>>(vertex, varying, E_IT, E_W, offset, start, end);
    OPT_KERNEL(0, 0, computeEdge, 512, 32, (vertex, varying, E_IT, E_W, offset, tableOffset, start, end));
    OPT_KERNEL(0, 3, computeEdge, 512, 32, (vertex, varying, E_IT, E_W, offset, tableOffset, start, end));
    OPT_KERNEL(3, 0, computeEdge, 512, 32, (vertex, varying, E_IT, E_W, offset, tableOffset, start, end));
    OPT_KERNEL(3, 3, computeEdge, 512, 32, (vertex, varying, E_IT, E_W, offset, tableOffset, start, end));

    computeEdge<<<512, 32>>>(vertex, varying,
                             vertexLength, vertexStride, varyingLength, varyingStride,
                             E_IT, E_W, offset, tableOffset, start, end);
}

void OsdCudaComputeRestrictedEdge(float *vertex, float *varying,
                                  int vertexLength, int vertexStride,
                                  int varyingLength, int varyingStride,
                                  int *E_IT, int offset, int tableOffset, int start, int end)
{
    //computeRestrictedEdge<0, 3><<<512,32>>>(vertex, varying, E_IT, offset, start, end);
    OPT_KERNEL(0, 0, computeRestrictedEdge, 512, 32, (vertex, varying, E_IT, offset, tableOffset, start, end));
    OPT_KERNEL(0, 3, computeRestrictedEdge, 512, 32, (vertex, varying, E_IT, offset, tableOffset, start, end));
    OPT_KERNEL(3, 0, computeRestrictedEdge, 512, 32, (vertex, varying, E_IT, offset, tableOffset, start, end));
    OPT_KERNEL(3, 3, computeRestrictedEdge, 512, 32, (vertex, varying, E_IT, offset, tableOffset, start, end));

    computeRestrictedEdge<<<512, 32>>>(vertex, varying,
                                       vertexLength, vertexStride, varyingLength, varyingStride,
                                       E_IT, offset, tableOffset, start, end);
}

void OsdCudaComputeVertexA(float *vertex, float *varying,
                           int vertexLength, int vertexStride,
                           int varyingLength, int varyingStride,
                           int *V_ITa, float *V_W, int offset, int tableOffset, int start, int end, int pass)
{
//    computeVertexA<0, 3><<<512,32>>>(vertex, varying, V_ITa, V_W, offset, start, end, pass);
    OPT_KERNEL(0, 0, computeVertexA, 512, 32, (vertex, varying, V_ITa, V_W, offset, tableOffset, start, end, pass));
    OPT_KERNEL(0, 3, computeVertexA, 512, 32, (vertex, varying, V_ITa, V_W, offset, tableOffset, start, end, pass));
    OPT_KERNEL(3, 0, computeVertexA, 512, 32, (vertex, varying, V_ITa, V_W, offset, tableOffset, start, end, pass));
    OPT_KERNEL(3, 3, computeVertexA, 512, 32, (vertex, varying, V_ITa, V_W, offset, tableOffset, start, end, pass));

    computeVertexA<<<512, 32>>>(vertex, varying,
                                vertexLength, vertexStride, varyingLength, varyingStride,
                                V_ITa, V_W, offset, tableOffset, start, end, pass);
}

void OsdCudaComputeVertexB(float *vertex, float *varying,
                           int vertexLength, int vertexStride,
                           int varyingLength, int varyingStride,
                           int *V_ITa, int *V_IT, float *V_W, int offset, int tableOffset, int start, int end)
{
//    computeVertexB<0, 3><<<512,32>>>(vertex, varying, V_ITa, V_IT, V_W, offset, start, end);
    OPT_KERNEL(0, 0, computeVertexB, 512, 32, (vertex, varying, V_ITa, V_IT, V_W, offset, tableOffset, start, end));
    OPT_KERNEL(0, 3, computeVertexB, 512, 32, (vertex, varying, V_ITa, V_IT, V_W, offset, tableOffset, start, end));
    OPT_KERNEL(3, 0, computeVertexB, 512, 32, (vertex, varying, V_ITa, V_IT, V_W, offset, tableOffset, start, end));
    OPT_KERNEL(3, 3, computeVertexB, 512, 32, (vertex, varying, V_ITa, V_IT, V_W, offset, tableOffset, start, end));

    computeVertexB<<<512, 32>>>(vertex, varying,
                                vertexLength, vertexStride, varyingLength, varyingStride,
                                V_ITa, V_IT, V_W, offset, tableOffset, start, end);
}

void OsdCudaComputeRestrictedVertexA(float *vertex, float *varying,
                                     int vertexLength, int vertexStride,
                                     int varyingLength, int varyingStride,
                                     int *V_ITa, int offset, int tableOffset, int start, int end)
{
//    computeRestrictedVertexA<0, 3><<<512,32>>>(vertex, varying, V_ITa, offset, start, end);
    OPT_KERNEL(0, 0, computeRestrictedVertexA, 512, 32, (vertex, varying, V_ITa, offset, tableOffset, start, end));
    OPT_KERNEL(0, 3, computeRestrictedVertexA, 512, 32, (vertex, varying, V_ITa, offset, tableOffset, start, end));
    OPT_KERNEL(3, 0, computeRestrictedVertexA, 512, 32, (vertex, varying, V_ITa, offset, tableOffset, start, end));
    OPT_KERNEL(3, 3, computeRestrictedVertexA, 512, 32, (vertex, varying, V_ITa, offset, tableOffset, start, end));

    computeRestrictedVertexA<<<512, 32>>>(vertex, varying,
                                          vertexLength, vertexStride, varyingLength, varyingStride,
                                          V_ITa, offset, tableOffset, start, end);
}

void OsdCudaComputeRestrictedVertexB1(float *vertex, float *varying,
                                      int vertexLength, int vertexStride,
                                      int varyingLength, int varyingStride,
                                      int *V_ITa, int *V_IT, int offset, int tableOffset, int start, int end)
{
//    computeRestrictedVertexB1<0, 3><<<512,32>>>(vertex, varying, V_ITa, V_IT, offset, start, end);
    OPT_KERNEL(0, 0, computeRestrictedVertexB1, 512, 32, (vertex, varying, V_ITa, V_IT, offset, tableOffset, start, end));
    OPT_KERNEL(0, 3, computeRestrictedVertexB1, 512, 32, (vertex, varying, V_ITa, V_IT, offset, tableOffset, start, end));
    OPT_KERNEL(3, 0, computeRestrictedVertexB1, 512, 32, (vertex, varying, V_ITa, V_IT, offset, tableOffset, start, end));
    OPT_KERNEL(3, 3, computeRestrictedVertexB1, 512, 32, (vertex, varying, V_ITa, V_IT, offset, tableOffset, start, end));

    computeRestrictedVertexB1 <<<512, 32>>>(vertex, varying,
                                            vertexLength, vertexStride, varyingLength, varyingStride,
                                            V_ITa, V_IT, offset, tableOffset, start, end);
}

void OsdCudaComputeRestrictedVertexB2(float *vertex, float *varying,
                                      int vertexLength, int vertexStride,
                                      int varyingLength, int varyingStride,
                                      int *V_ITa, int *V_IT, int offset, int tableOffset, int start, int end)
{
//    computeRestrictedVertexB2<0, 3><<<512,32>>>(vertex, varying, V_ITa, V_IT, offset, start, end);
    OPT_KERNEL(0, 0, computeRestrictedVertexB2, 512, 32, (vertex, varying, V_ITa, V_IT, offset, tableOffset, start, end));
    OPT_KERNEL(0, 3, computeRestrictedVertexB2, 512, 32, (vertex, varying, V_ITa, V_IT, offset, tableOffset, start, end));
    OPT_KERNEL(3, 0, computeRestrictedVertexB2, 512, 32, (vertex, varying, V_ITa, V_IT, offset, tableOffset, start, end));
    OPT_KERNEL(3, 3, computeRestrictedVertexB2, 512, 32, (vertex, varying, V_ITa, V_IT, offset, tableOffset, start, end));

    computeRestrictedVertexB2 <<<512, 32>>>(vertex, varying,
                                            vertexLength, vertexStride, varyingLength, varyingStride,
                                            V_ITa, V_IT, offset, tableOffset, start, end);
}

void OsdCudaComputeLoopVertexB(float *vertex, float *varying,
                               int vertexLength, int vertexStride,
                               int varyingLength, int varyingStride,
                               int *V_ITa, int *V_IT, float *V_W, int offset, int tableOffset, int start, int end)
{
//    computeLoopVertexB<0, 3><<<512,32>>>(vertex, varying, V_ITa, V_IT, V_W, offset, start, end);
    OPT_KERNEL(0, 0, computeLoopVertexB, 512, 32, (vertex, varying, V_ITa, V_IT, V_W, offset, tableOffset, start, end));
    OPT_KERNEL(0, 3, computeLoopVertexB, 512, 32, (vertex, varying, V_ITa, V_IT, V_W, offset, tableOffset, start, end));
    OPT_KERNEL(3, 0, computeLoopVertexB, 512, 32, (vertex, varying, V_ITa, V_IT, V_W, offset, tableOffset, start, end));
    OPT_KERNEL(3, 3, computeLoopVertexB, 512, 32, (vertex, varying, V_ITa, V_IT, V_W, offset, tableOffset, start, end));

    computeLoopVertexB<<<512, 32>>>(vertex, varying,
                                    vertexLength, vertexStride, varyingLength, varyingStride,
                                    V_ITa, V_IT, V_W, offset, tableOffset, start, end);
}

void OsdCudaComputeBilinearEdge(float *vertex, float *varying,
                                int vertexLength, int vertexStride,
                                int varyingLength, int varyingStride,
                                int *E_IT, int offset, int tableOffset, int start, int end)
{
    //computeBilinearEdge<0, 3><<<512,32>>>(vertex, varying, E_IT, offset, start, end);
    OPT_KERNEL(0, 0, computeBilinearEdge, 512, 32, (vertex, varying, E_IT, offset, tableOffset, start, end));
    OPT_KERNEL(0, 3, computeBilinearEdge, 512, 32, (vertex, varying, E_IT, offset, tableOffset, start, end));
    OPT_KERNEL(3, 0, computeBilinearEdge, 512, 32, (vertex, varying, E_IT, offset, tableOffset, start, end));
    OPT_KERNEL(3, 3, computeBilinearEdge, 512, 32, (vertex, varying, E_IT, offset, tableOffset, start, end));

    computeBilinearEdge<<<512, 32>>>(vertex, varying,
                                     vertexLength, vertexStride, varyingLength, varyingStride,
                                     E_IT, offset, tableOffset, start, end);
}

void OsdCudaComputeBilinearVertex(float *vertex, float *varying,
                                  int vertexLength, int vertexStride,
                                  int varyingLength, int varyingStride,
                                  int *V_ITa, int offset, int tableOffset, int start, int end)
{
//    computeBilinearVertex<0, 3><<<512,32>>>(vertex, varying, V_ITa, offset, start, end);
    OPT_KERNEL(0, 0, computeBilinearVertex, 512, 32, (vertex, varying, V_ITa, offset, tableOffset, start, end));
    OPT_KERNEL(0, 3, computeBilinearVertex, 512, 32, (vertex, varying, V_ITa, offset, tableOffset, start, end));
    OPT_KERNEL(3, 0, computeBilinearVertex, 512, 32, (vertex, varying, V_ITa, offset, tableOffset, start, end));
    OPT_KERNEL(3, 3, computeBilinearVertex, 512, 32, (vertex, varying, V_ITa, offset, tableOffset, start, end));

    computeBilinearVertex<<<512, 32>>>(vertex, varying,
                                       vertexLength, vertexStride, varyingLength, varyingStride,
                                       V_ITa, offset, tableOffset, start, end);
}

void OsdCudaEditVertexAdd(float *vertex, int vertexLength, int vertexStride,
                          int primVarOffset, int primVarWidth,
                          int vertexOffset, int tableOffset,
                          int start, int end, int *editIndices, float *editValues)
{
    editVertexAdd<<<512, 32>>>(vertex, vertexLength, vertexStride, primVarOffset, primVarWidth,
                               vertexOffset, tableOffset, start, end,
                               editIndices, editValues);
}

}  /* extern "C" */
