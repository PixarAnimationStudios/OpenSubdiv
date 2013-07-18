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

#include <assert.h>

template<int N> struct DeviceVertex
{
    float pos[3];
    float userVertexData[N];

    __device__ void addWithWeight(const DeviceVertex<N> *src, float weight) {
        pos[0] += src->pos[0] * weight;
        pos[1] += src->pos[1] * weight;
        pos[2] += src->pos[2] * weight;

        for(int i = 0; i < N; ++i){
            userVertexData[i] += src->userVertexData[i] * weight;
        }
    }
    __device__ void clear() {
        pos[0] = pos[1] = pos[2] = 0.0f;
        for(int i = 0; i < N; ++i){
            userVertexData[i] = 0.0f;
        }
    }
};

template<int N> struct DeviceVarying
{
    float v[N];

    __device__ void addVaryingWithWeight(const DeviceVarying<N> *src, float weight) {
        for(int i = 0; i < N; ++i){
            v[i] += src->v[i] * weight;
        }
    }
    __device__ void clear() {
        for(int i = 0; i < N; ++i){
            v[i] = 0.0f;
        }
    }
};

// Specialize DeviceVarying for N=0 to avoid compile error:
// "flexible array member in otherwise empty struct"
template<> struct DeviceVarying<0>
{
    __device__ void addVaryingWithWeight(const DeviceVarying<0> *src, float weight) {
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

__device__ void addVaryingWithWeight(float *dst, float *src, float weight, int count)
{
    for(int i = 0; i < count; ++i) dst[i] += src[i] * weight;
}

template <int NUM_USER_VERTEX_ELEMENTS, int NUM_VARYING_ELEMENTS> __global__ void
computeFace(float *fVertex, float *fVaryings, int *F0_IT, int *F0_ITa, int offset, int tableOffset, int start, int end)
{
    DeviceVertex<NUM_USER_VERTEX_ELEMENTS> *vertex = (DeviceVertex<NUM_USER_VERTEX_ELEMENTS>*)fVertex;
    DeviceVarying<NUM_VARYING_ELEMENTS> *varyings = (DeviceVarying<NUM_VARYING_ELEMENTS>*)fVaryings;
    for(int i = start + tableOffset + threadIdx.x + blockIdx.x*blockDim.x; i < end + tableOffset; i += blockDim.x * gridDim.x){
        int h = F0_ITa[2*i];
        int n = F0_ITa[2*i+1];
        float weight = 1.0f/n;

        DeviceVertex<NUM_USER_VERTEX_ELEMENTS> dst;
        dst.clear();

        if(NUM_VARYING_ELEMENTS > 0){
            DeviceVarying<NUM_VARYING_ELEMENTS> dstVarying;
            dstVarying.clear();

            for(int j=0; j<n; ++j){
                int index = F0_IT[h+j];
                dst.addWithWeight(&vertex[index], weight);
                dstVarying.addVaryingWithWeight(&varyings[index], weight);
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
computeFace(float *fVertex, int numVertexElements, float *fVaryings, int numVaryingElements,
            int *F0_IT, int *F0_ITa, int offset, int tableOffset, int start, int end)
{
    for(int i = start + tableOffset +threadIdx.x + blockIdx.x*blockDim.x; i < end + tableOffset; i += blockDim.x * gridDim.x){
        int h = F0_ITa[2*i];
        int n = F0_ITa[2*i+1];
        float weight = 1.0f/n;

        // XXX: can we use local stack like alloca?
        float *dstVertex = fVertex + (i+offset-tableOffset)*numVertexElements;
        clear(dstVertex, numVertexElements);
        float *dstVarying = fVaryings + (i+offset-tableOffset)*numVaryingElements;
        clear(dstVarying, numVaryingElements);

        for(int j=0; j<n; ++j){
            int index = F0_IT[h+j];
            addWithWeight(dstVertex, fVertex + index*numVertexElements, weight, numVertexElements);
            addVaryingWithWeight(dstVarying, fVaryings + index*numVaryingElements, weight, numVaryingElements);
        }
    }
}

template <int NUM_USER_VERTEX_ELEMENTS, int NUM_VARYING_ELEMENTS> __global__ void
computeEdge(float *fVertex, float *fVaryings, int *E0_IT, float *E0_S, int offset, int tableOffset, int start, int end)
{
    DeviceVertex<NUM_USER_VERTEX_ELEMENTS> *vertex = (DeviceVertex<NUM_USER_VERTEX_ELEMENTS>*)fVertex;
    DeviceVarying<NUM_VARYING_ELEMENTS> *varyings = (DeviceVarying<NUM_VARYING_ELEMENTS>*)fVaryings;
    for(int i = start + tableOffset + threadIdx.x + blockIdx.x*blockDim.x; i < end + tableOffset; i+= blockDim.x * gridDim.x){
        int eidx0 = E0_IT[4*i+0];
        int eidx1 = E0_IT[4*i+1];
        int eidx2 = E0_IT[4*i+2];
        int eidx3 = E0_IT[4*i+3];

        float vertWeight = E0_S[i*2+0];

        // Fully sharp edge : vertWeight = 0.5f;
        DeviceVertex<NUM_USER_VERTEX_ELEMENTS> dst;
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
            DeviceVarying<NUM_VARYING_ELEMENTS> dstVarying;
            dstVarying.clear();
            dstVarying.addVaryingWithWeight(&varyings[eidx0], 0.5f);
            dstVarying.addVaryingWithWeight(&varyings[eidx1], 0.5f);
            varyings[offset+i-tableOffset] = dstVarying;
        }
    }
}

__global__ void
computeEdge(float *fVertex, int numVertexElements, float *fVarying, int numVaryingElements,
            int *E0_IT, float *E0_S, int offset, int tableOffset, int start, int end)
{
    for(int i = start + tableOffset + threadIdx.x + blockIdx.x*blockDim.x; i < end + tableOffset; i+= blockDim.x * gridDim.x){
        int eidx0 = E0_IT[4*i+0];
        int eidx1 = E0_IT[4*i+1];
        int eidx2 = E0_IT[4*i+2];
        int eidx3 = E0_IT[4*i+3];

        float vertWeight = E0_S[i*2+0];

        // Fully sharp edge : vertWeight = 0.5f;
        float *dstVertex = fVertex + (i+offset-tableOffset)*numVertexElements;
        clear(dstVertex, numVertexElements);

        addWithWeight(dstVertex, fVertex + eidx0*numVertexElements, vertWeight, numVertexElements);
        addWithWeight(dstVertex, fVertex + eidx1*numVertexElements, vertWeight, numVertexElements);

        if(eidx2 > -1){
            float faceWeight = E0_S[i*2+1];

            addWithWeight(dstVertex, fVertex + eidx2*numVertexElements, faceWeight, numVertexElements);
            addWithWeight(dstVertex, fVertex + eidx3*numVertexElements, faceWeight, numVertexElements);
        }

        if(numVaryingElements > 0){
            float *dstVarying = fVarying + (i+offset-tableOffset)*numVaryingElements;
            clear(dstVarying, numVaryingElements);

            addVaryingWithWeight(dstVarying, fVarying + eidx0*numVaryingElements, 0.5f, numVaryingElements);
            addVaryingWithWeight(dstVarying, fVarying + eidx1*numVaryingElements, 0.5f, numVaryingElements);
        }
    }
}

template <int NUM_USER_VERTEX_ELEMENTS, int NUM_VARYING_ELEMENTS> __global__ void
computeVertexA(float *fVertex, float *fVaryings, int *V0_ITa, float *V0_S, int offset, int tableOffset, int start, int end, int pass)
{
    DeviceVertex<NUM_USER_VERTEX_ELEMENTS> *vertex = (DeviceVertex<NUM_USER_VERTEX_ELEMENTS>*)fVertex;
    DeviceVarying<NUM_VARYING_ELEMENTS> *varyings = (DeviceVarying<NUM_VARYING_ELEMENTS>*)fVaryings;
    for(int i = start + tableOffset + threadIdx.x + blockIdx.x*blockDim.x; i < end+tableOffset; i += blockDim.x * gridDim.x){
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

        DeviceVertex<NUM_USER_VERTEX_ELEMENTS> dst;
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
                DeviceVarying<NUM_VARYING_ELEMENTS> dstVarying;
                dstVarying.clear();
                dstVarying.addVaryingWithWeight(&varyings[p], 1.0f);
                varyings[i+offset-tableOffset] = dstVarying;
            }
        }
    }
}

__global__ void
computeVertexA(float *fVertex, int numVertexElements, float *fVaryings, int numVaryingElements,
               int *V0_ITa, float *V0_S, int offset, int tableOffset, int start, int end, int pass)
{
    for(int i = start + tableOffset + threadIdx.x + blockIdx.x*blockDim.x; i < end + tableOffset; i += blockDim.x * gridDim.x){
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

        float *dstVertex = fVertex + (i+offset-tableOffset)*numVertexElements;
        if (not pass) {
            clear(dstVertex, numVertexElements);
        }

        if (eidx0==-1 || (pass==0 && (n==-1)) ) {
            addWithWeight(dstVertex, fVertex + p*numVertexElements, weight, numVertexElements);
        } else {
            addWithWeight(dstVertex, fVertex + p*numVertexElements, weight*0.75f, numVertexElements);
            addWithWeight(dstVertex, fVertex + eidx0*numVertexElements, weight*0.125f, numVertexElements);
            addWithWeight(dstVertex, fVertex + eidx1*numVertexElements, weight*0.125f, numVertexElements);
        }

        if(numVaryingElements > 0){
            if(not pass){
                float *dstVarying = fVaryings + (i+offset-tableOffset)*numVaryingElements;
                clear(dstVarying, numVaryingElements);
                addVaryingWithWeight(dstVarying, fVaryings + p*numVaryingElements, 1.0f, numVaryingElements);
            }
        }
    }

}


//texture <int, 1> texV0_IT;

template <int NUM_USER_VERTEX_ELEMENTS, int NUM_VARYING_ELEMENTS> __global__ void
computeVertexB(float *fVertex, float *fVaryings,
               const int *V0_ITa, const int *V0_IT, const float *V0_S, int offset, int tableOffset, int start, int end)
{
    DeviceVertex<NUM_USER_VERTEX_ELEMENTS> *vertex = (DeviceVertex<NUM_USER_VERTEX_ELEMENTS>*)fVertex;
    DeviceVarying<NUM_VARYING_ELEMENTS> *varyings = (DeviceVarying<NUM_VARYING_ELEMENTS>*)fVaryings;
    for(int i = start + tableOffset + threadIdx.x + blockIdx.x*blockDim.x; i < end + tableOffset; i += blockDim.x * gridDim.x){
        int h = V0_ITa[5*i];
        int n = V0_ITa[5*i+1];
        int p = V0_ITa[5*i+2];

        float weight = V0_S[i];
        float wp = 1.0f/float(n*n);
        float wv = (n-2.0f) * n * wp;

        DeviceVertex<NUM_USER_VERTEX_ELEMENTS> dst;
        dst.clear();
        dst.addWithWeight(&vertex[p], weight * wv);

        for(int j = 0; j < n; ++j){
            dst.addWithWeight(&vertex[V0_IT[h+j*2]], weight * wp);
            dst.addWithWeight(&vertex[V0_IT[h+j*2+1]], weight * wp);
//            int idx0 = tex1Dfetch(texV0_IT, h+j*2);
//            int idx1 = tex1Dfetch(texV0_IT, h+j*2+1);
//            dst.addWithWeight(&vertex[idx0], weight * wp);
//            dst.addWithWeight(&vertex[idx1], weight * wp);
        }
        vertex[i+offset-tableOffset] = dst;

        if(NUM_VARYING_ELEMENTS > 0){
            DeviceVarying<NUM_VARYING_ELEMENTS> dstVarying;
            dstVarying.clear();
            dstVarying.addVaryingWithWeight(&varyings[p], 1.0f);
            varyings[i+offset-tableOffset] = dstVarying;
        }
    }
}

__global__ void
computeVertexB(float *fVertex, int numVertexElements, float *fVaryings, int numVaryingElements,
               const int *V0_ITa, const int *V0_IT, const float *V0_S, int offset, int tableOffset, int start, int end)
{
    for(int i = start + tableOffset + threadIdx.x + blockIdx.x*blockDim.x; i < end + tableOffset; i += blockDim.x * gridDim.x){
        int h = V0_ITa[5*i];
        int n = V0_ITa[5*i+1];
        int p = V0_ITa[5*i+2];

        float weight = V0_S[i];
        float wp = 1.0f/float(n*n);
        float wv = (n-2.0f) * n * wp;

        float *dstVertex = fVertex + (i+offset-tableOffset)*numVertexElements;
        clear(dstVertex, numVertexElements);
        addWithWeight(dstVertex, fVertex + p*numVertexElements, weight*wv, numVertexElements);

        for(int j = 0; j < n; ++j){
            addWithWeight(dstVertex, fVertex + V0_IT[h+j*2]*numVertexElements, weight*wp, numVertexElements);
            addWithWeight(dstVertex, fVertex + V0_IT[h+j*2+1]*numVertexElements, weight*wp, numVertexElements);
        }

        if(numVaryingElements > 0){
            float *dstVarying = fVaryings + (i+offset-tableOffset)*numVaryingElements;
            clear(dstVarying, numVaryingElements);
            addVaryingWithWeight(dstVarying, fVaryings + p*numVaryingElements, 1.0f, numVaryingElements);
        }
    }
}


// --------------------------------------------------------------------------------------------

template <int NUM_USER_VERTEX_ELEMENTS, int NUM_VARYING_ELEMENTS> __global__ void
computeLoopVertexB(float *fVertex, float *fVaryings, int *V0_ITa, int *V0_IT, float *V0_S, int offset, int tableOffset, int start, int end)
{
    DeviceVertex<NUM_USER_VERTEX_ELEMENTS> *vertex = (DeviceVertex<NUM_USER_VERTEX_ELEMENTS>*)fVertex;
    DeviceVarying<NUM_VARYING_ELEMENTS> *varyings = (DeviceVarying<NUM_VARYING_ELEMENTS>*)fVaryings;
    for(int i = start + tableOffset + threadIdx.x + blockIdx.x*blockDim.x; i < end + tableOffset; i += blockDim.x * gridDim.x){
        int h = V0_ITa[5*i];
        int n = V0_ITa[5*i+1];
        int p = V0_ITa[5*i+2];

        float weight = V0_S[i];
        float wp = 1.0f/float(n);
        float beta = 0.25f * __cosf(float(M_PI) * 2.0f * wp) + 0.375f;
        beta = beta * beta;
        beta = (0.625f - beta) * wp;

        DeviceVertex<NUM_USER_VERTEX_ELEMENTS> dst;
        dst.clear();

        dst.addWithWeight(&vertex[p], weight * (1.0f - (beta * n)));

        for(int j = 0; j < n; ++j){
            dst.addWithWeight(&vertex[V0_IT[h+j]], weight * beta);
        }
        vertex[i+offset-tableOffset] = dst;

        if(NUM_VARYING_ELEMENTS > 0){
            DeviceVarying<NUM_VARYING_ELEMENTS> dstVarying;
            dstVarying.clear();
            dstVarying.addVaryingWithWeight(&varyings[p], 1.0f);
            varyings[i+offset-tableOffset] = dstVarying;
        }
    }
}

__global__ void
computeLoopVertexB(float *fVertex, int numVertexElements, float *fVaryings, int numVaryingElements,
                   const int *V0_ITa, const int *V0_IT, const float *V0_S, int offset, int tableOffset, int start, int end)
{
    for(int i = start + tableOffset + threadIdx.x + blockIdx.x*blockDim.x; i < end + tableOffset; i += blockDim.x * gridDim.x){
        int h = V0_ITa[5*i];
        int n = V0_ITa[5*i+1];
        int p = V0_ITa[5*i+2];

        float weight = V0_S[i];
        float wp = 1.0f/float(n);
        float beta = 0.25f * __cosf(float(M_PI) * 2.0f * wp) + 0.375f;
        beta = beta * beta;
        beta = (0.625f - beta) * wp;

        float *dstVertex = fVertex + (i+offset-tableOffset)*numVertexElements;
        clear(dstVertex, numVertexElements);
        addWithWeight(dstVertex, fVertex + p*numVertexElements, weight*(1.0f-(beta*n)), numVertexElements);

        for(int j = 0; j < n; ++j){
            addWithWeight(dstVertex, fVertex + V0_IT[h+j]*numVertexElements, weight*beta, numVertexElements);
        }

        if(numVaryingElements > 0){
            float *dstVarying = fVaryings + (i+offset-tableOffset)*numVaryingElements;
            clear(dstVarying, numVaryingElements);
            addVaryingWithWeight(dstVarying, fVaryings + p*numVaryingElements, 1.0f, numVaryingElements);
        }
    }
}

// --------------------------------------------------------------------------------------------

template <int NUM_USER_VERTEX_ELEMENTS, int NUM_VARYING_ELEMENTS> __global__ void
computeBilinearEdge(float *fVertex, float *fVaryings, int *E0_IT, int offset, int tableOffset, int start, int end)
{
    DeviceVertex<NUM_USER_VERTEX_ELEMENTS> *vertex = (DeviceVertex<NUM_USER_VERTEX_ELEMENTS>*)fVertex;
    DeviceVarying<NUM_VARYING_ELEMENTS> *varyings = (DeviceVarying<NUM_VARYING_ELEMENTS>*)fVaryings;
    for(int i = start + tableOffset + threadIdx.x + blockIdx.x*blockDim.x; i < end + tableOffset; i+= blockDim.x * gridDim.x){
        int eidx0 = E0_IT[2*i+0];
        int eidx1 = E0_IT[2*i+1];

        DeviceVertex<NUM_USER_VERTEX_ELEMENTS> dst;
        dst.clear();

        dst.addWithWeight(&vertex[eidx0], 0.5f);
        dst.addWithWeight(&vertex[eidx1], 0.5f);

        vertex[offset+i-tableOffset] = dst;

        if(NUM_VARYING_ELEMENTS > 0){
            DeviceVarying<NUM_VARYING_ELEMENTS> dstVarying;
            dstVarying.clear();
            dstVarying.addVaryingWithWeight(&varyings[eidx0], 0.5f);
            dstVarying.addVaryingWithWeight(&varyings[eidx1], 0.5f);
            varyings[offset+i-tableOffset] = dstVarying;
        }
    }
}

__global__ void
computeBilinearEdge(float *fVertex, int numVertexElements, float *fVarying, int numVaryingElements,
                    int *E0_IT, int offset, int tableOffset, int start, int end)
{
    for(int i = start + tableOffset + threadIdx.x + blockIdx.x*blockDim.x; i < end + tableOffset; i+= blockDim.x * gridDim.x){
        int eidx0 = E0_IT[2*i+0];
        int eidx1 = E0_IT[2*i+1];

        float *dstVertex = fVertex + (i+offset-tableOffset)*numVertexElements;
        clear(dstVertex, numVertexElements);

        addWithWeight(dstVertex, fVertex + eidx0*numVertexElements, 0.5f, numVertexElements);
        addWithWeight(dstVertex, fVertex + eidx1*numVertexElements, 0.5f, numVertexElements);

        if(numVaryingElements > 0){
            float *dstVarying = fVarying + (i+offset-tableOffset)*numVaryingElements;
            clear(dstVarying, numVaryingElements);

            addVaryingWithWeight(dstVarying, fVarying + eidx0*numVaryingElements, 0.5f, numVaryingElements);
            addVaryingWithWeight(dstVarying, fVarying + eidx1*numVaryingElements, 0.5f, numVaryingElements);
        }
    }
}

template <int NUM_USER_VERTEX_ELEMENTS, int NUM_VARYING_ELEMENTS> __global__ void
computeBilinearVertex(float *fVertex, float *fVaryings, int *V0_ITa, int offset, int tableOffset, int start, int end)
{
    DeviceVertex<NUM_USER_VERTEX_ELEMENTS> *vertex = (DeviceVertex<NUM_USER_VERTEX_ELEMENTS>*)fVertex;
    DeviceVarying<NUM_VARYING_ELEMENTS> *varyings = (DeviceVarying<NUM_VARYING_ELEMENTS>*)fVaryings;
    for(int i = start + tableOffset + threadIdx.x + blockIdx.x*blockDim.x; i < end + tableOffset; i += blockDim.x * gridDim.x){
        int p = V0_ITa[i];

        DeviceVertex<NUM_USER_VERTEX_ELEMENTS> dst;
        dst.clear();

        dst.addWithWeight(&vertex[p], 1.0f);
        vertex[i+offset-tableOffset] = dst;

        if(NUM_VARYING_ELEMENTS > 0){
            DeviceVarying<NUM_VARYING_ELEMENTS> dstVarying;
            dstVarying.clear();
            dstVarying.addVaryingWithWeight(&varyings[p], 1.0f);
            varyings[i+offset-tableOffset] = dstVarying;
        }
    }
}

__global__ void
computeBilinearVertex(float *fVertex, int numVertexElements, float *fVaryings, int numVaryingElements,
                      const int *V0_ITa, int offset, int tableOffset, int start, int end)
{
    for(int i = start + tableOffset + threadIdx.x + blockIdx.x*blockDim.x; i < end + tableOffset; i += blockDim.x * gridDim.x){
        int p = V0_ITa[i];

        float *dstVertex = fVertex + (i+offset-tableOffset)*numVertexElements;
        clear(dstVertex, numVertexElements);
        addWithWeight(dstVertex, fVertex + p*numVertexElements, 1.0f, numVertexElements);

        if(numVaryingElements > 0){
            float *dstVarying = fVaryings + (i+offset-tableOffset)*numVaryingElements;
            clear(dstVarying, numVaryingElements);
            addVaryingWithWeight(dstVarying, fVaryings + p*numVaryingElements, 1.0f, numVaryingElements);
        }
    }
}

// --------------------------------------------------------------------------------------------

__global__ void
editVertexAdd(float *fVertex, int numVertexElements, int primVarOffset, int primVarWidth,
              int vertexOffset, int tableOffset, int start, int end,
              const int *editIndices, const float *editValues)
{
    for(int i = start + tableOffset + threadIdx.x + blockIdx.x*blockDim.x;
        i < end + tableOffset;
        i += blockDim.x * gridDim.x) {

        float *dstVertex = fVertex + (editIndices[i] + vertexOffset) * numVertexElements + primVarOffset;

        for(int j = 0; j < primVarWidth; j++) {
            *dstVertex++ += editValues[i*primVarWidth + j];
        }
    }
}

// --------------------------------------------------------------------------------------------

#include "../version.h"

// XXX: this macro usage is tentative. Since cuda kernel can't be dynamically configured,
// still trying to find better way to have optimized kernel..

#define OPT_KERNEL(NUM_USER_VERTEX_ELEMENTS, NUM_VARYING_ELEMENTS, KERNEL, X, Y, ARG) \
    if(numUserVertexElements == NUM_USER_VERTEX_ELEMENTS && \
       numVaryingElements == NUM_VARYING_ELEMENTS) \
       { KERNEL<NUM_USER_VERTEX_ELEMENTS, NUM_VARYING_ELEMENTS><<<X,Y>>>ARG; \
         return;  }

extern "C" {

void OsdCudaComputeFace(float *vertex, float *varying,
                        int numUserVertexElements, int numVaryingElements,
                        int *F_IT, int *F_ITa, int offset, int tableOffset, int start, int end)
{
    //computeFace<3, 0><<<512,32>>>(vertex, varying, F_IT, F_ITa, offset, start, end);
    OPT_KERNEL(0, 0, computeFace, 512, 32, (vertex, varying, F_IT, F_ITa, offset, tableOffset, start, end));
    OPT_KERNEL(0, 3, computeFace, 512, 32, (vertex, varying, F_IT, F_ITa, offset, tableOffset, start, end));
    OPT_KERNEL(3, 0, computeFace, 512, 32, (vertex, varying, F_IT, F_ITa, offset, tableOffset, start, end));
    OPT_KERNEL(3, 3, computeFace, 512, 32, (vertex, varying, F_IT, F_ITa, offset, tableOffset, start, end));

    // fallback kernel (slow)
    computeFace<<<512, 32>>>(vertex, 3+numUserVertexElements, varying, numVaryingElements,
                             F_IT, F_ITa, offset, tableOffset, start, end);
}

void OsdCudaComputeEdge(float *vertex, float *varying,
                        int numUserVertexElements, int numVaryingElements,
                        int *E_IT, float *E_W, int offset, int tableOffset, int start, int end)
{
    //computeEdge<0, 3><<<512,32>>>(vertex, varying, E_IT, E_W, offset, start, end);
    OPT_KERNEL(0, 0, computeEdge, 512, 32, (vertex, varying, E_IT, E_W, offset, tableOffset, start, end));
    OPT_KERNEL(0, 3, computeEdge, 512, 32, (vertex, varying, E_IT, E_W, offset, tableOffset, start, end));
    OPT_KERNEL(3, 0, computeEdge, 512, 32, (vertex, varying, E_IT, E_W, offset, tableOffset, start, end));
    OPT_KERNEL(3, 3, computeEdge, 512, 32, (vertex, varying, E_IT, E_W, offset, tableOffset, start, end));

    computeEdge<<<512, 32>>>(vertex, 3+numUserVertexElements, varying, numVaryingElements,
                             E_IT, E_W, offset, tableOffset, start, end);
}

void OsdCudaComputeVertexA(float *vertex, float *varying,
                           int numUserVertexElements, int numVaryingElements,
                           int *V_ITa, float *V_W, int offset, int tableOffset, int start, int end, int pass)
{
//    computeVertexA<0, 3><<<512,32>>>(vertex, varying, V_ITa, V_W, offset, start, end, pass);
    OPT_KERNEL(0, 0, computeVertexA, 512, 32, (vertex, varying, V_ITa, V_W, offset, tableOffset, start, end, pass));
    OPT_KERNEL(0, 3, computeVertexA, 512, 32, (vertex, varying, V_ITa, V_W, offset, tableOffset, start, end, pass));
    OPT_KERNEL(3, 0, computeVertexA, 512, 32, (vertex, varying, V_ITa, V_W, offset, tableOffset, start, end, pass));
    OPT_KERNEL(3, 3, computeVertexA, 512, 32, (vertex, varying, V_ITa, V_W, offset, tableOffset, start, end, pass));

    computeVertexA<<<512, 32>>>(vertex, 3+numUserVertexElements, varying, numVaryingElements,
                                V_ITa, V_W, offset, tableOffset, start, end, pass);
}

void OsdCudaComputeVertexB(float *vertex, float *varying,
                           int numUserVertexElements, int numVaryingElements,
                           int *V_ITa, int *V_IT, float *V_W, int offset, int tableOffset, int start, int end)
{
//    computeVertexB<0, 3><<<512,32>>>(vertex, varying, V_ITa, V_IT, V_W, offset, start, end);
    OPT_KERNEL(0, 0, computeVertexB, 512, 32, (vertex, varying, V_ITa, V_IT, V_W, offset, tableOffset, start, end));
    OPT_KERNEL(0, 3, computeVertexB, 512, 32, (vertex, varying, V_ITa, V_IT, V_W, offset, tableOffset, start, end));
    OPT_KERNEL(3, 0, computeVertexB, 512, 32, (vertex, varying, V_ITa, V_IT, V_W, offset, tableOffset, start, end));
    OPT_KERNEL(3, 3, computeVertexB, 512, 32, (vertex, varying, V_ITa, V_IT, V_W, offset, tableOffset, start, end));

    computeVertexB<<<512, 32>>>(vertex, 3+numUserVertexElements, varying, numVaryingElements,
                                V_ITa, V_IT, V_W, offset, tableOffset, start, end);
}

void OsdCudaComputeLoopVertexB(float *vertex, float *varying,
                               int numUserVertexElements, int numVaryingElements,
                               int *V_ITa, int *V_IT, float *V_W, int offset, int tableOffset, int start, int end)
{
//    computeLoopVertexB<0, 3><<<512,32>>>(vertex, varying, V_ITa, V_IT, V_W, offset, start, end);
    OPT_KERNEL(0, 0, computeLoopVertexB, 512, 32, (vertex, varying, V_ITa, V_IT, V_W, offset, tableOffset, start, end));
    OPT_KERNEL(0, 3, computeLoopVertexB, 512, 32, (vertex, varying, V_ITa, V_IT, V_W, offset, tableOffset, start, end));
    OPT_KERNEL(3, 0, computeLoopVertexB, 512, 32, (vertex, varying, V_ITa, V_IT, V_W, offset, tableOffset, start, end));
    OPT_KERNEL(3, 3, computeLoopVertexB, 512, 32, (vertex, varying, V_ITa, V_IT, V_W, offset, tableOffset, start, end));

    computeLoopVertexB<<<512, 32>>>(vertex, 3+numUserVertexElements, varying, numVaryingElements,
                                    V_ITa, V_IT, V_W, offset, tableOffset, start, end);
}

void OsdCudaComputeBilinearEdge(float *vertex, float *varying,
                                int numUserVertexElements, int numVaryingElements,
                                int *E_IT, int offset, int tableOffset, int start, int end)
{
    //computeBilinearEdge<0, 3><<<512,32>>>(vertex, varying, E_IT, offset, start, end);
    OPT_KERNEL(0, 0, computeBilinearEdge, 512, 32, (vertex, varying, E_IT, offset, tableOffset, start, end));
    OPT_KERNEL(0, 3, computeBilinearEdge, 512, 32, (vertex, varying, E_IT, offset, tableOffset, start, end));
    OPT_KERNEL(3, 0, computeBilinearEdge, 512, 32, (vertex, varying, E_IT, offset, tableOffset, start, end));
    OPT_KERNEL(3, 3, computeBilinearEdge, 512, 32, (vertex, varying, E_IT, offset, tableOffset, start, end));

    computeBilinearEdge<<<512, 32>>>(vertex, 3+numUserVertexElements, varying, numVaryingElements,
                                     E_IT, offset, tableOffset, start, end);
}

void OsdCudaComputeBilinearVertex(float *vertex, float *varying,
                                  int numUserVertexElements, int numVaryingElements,
                                  int *V_ITa, int offset, int tableOffset, int start, int end)
{
//    computeBilinearVertex<0, 3><<<512,32>>>(vertex, varying, V_ITa, offset, start, end);
    OPT_KERNEL(0, 0, computeBilinearVertex, 512, 32, (vertex, varying, V_ITa, offset, tableOffset, start, end));
    OPT_KERNEL(0, 3, computeBilinearVertex, 512, 32, (vertex, varying, V_ITa, offset, tableOffset, start, end));
    OPT_KERNEL(3, 0, computeBilinearVertex, 512, 32, (vertex, varying, V_ITa, offset, tableOffset, start, end));
    OPT_KERNEL(3, 3, computeBilinearVertex, 512, 32, (vertex, varying, V_ITa, offset, tableOffset, start, end));

    computeBilinearVertex<<<512, 32>>>(vertex, 3+numUserVertexElements, varying, numVaryingElements,
                                       V_ITa, offset, tableOffset, start, end);
}

void OsdCudaEditVertexAdd(float *vertex, int numUserVertexElements,
                          int primVarOffset, int primVarWidth,
                          int vertexOffset, int tableOffset,
                          int start, int end, int *editIndices, float *editValues)
{
    editVertexAdd<<<512, 32>>>(vertex, 3+numUserVertexElements, primVarOffset, primVarWidth,
                               vertexOffset, tableOffset, start, end,
                               editIndices, editValues);
}

}  /* extern "C" */
