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

// -----------------------------------------------------------------------------
template<int N> struct DeviceVertex {

    float v[N];

    __device__ void addWithWeight(DeviceVertex<N> const & src, float weight) {
#pragma unroll
        for(int i = 0; i < N; ++i){
            v[i] += src.v[i] * weight;
        }
    }

    __device__ void clear() {
#pragma unroll
        for(int i = 0; i < N; ++i){
            v[i] = 0.0f;
        }
    }
};

// Specialize DeviceVertex for N=0 to avoid compile error:
// "flexible array member in otherwise empty struct"
template<> struct DeviceVertex<0> {
    __device__ void addWithWeight(DeviceVertex<0> &src, float weight) {}
    __device__ void clear() {}
};

// -----------------------------------------------------------------------------

__device__ void clear(float *dst, int count)
{
    for(int i = 0; i < count; ++i) dst[i] = 0;
}

__device__ void addWithWeight(float *dst, float const *src, float weight, int count)
{
    for(int i = 0; i < count; ++i) dst[i] += src[i] * weight;
}

// -----------------------------------------------------------------------------

template <int NUM_ELEMENTS> __global__ void
computeStencils(float const * cvs, float * vbuffer,
                unsigned char const * sizes,
                int const * offsets,
                int const * indices,
                float const * weights,
                int start, int end) {

    DeviceVertex<NUM_ELEMENTS> const * src =
        (DeviceVertex<NUM_ELEMENTS> const *)cvs;

    DeviceVertex<NUM_ELEMENTS> * verts =
        (DeviceVertex<NUM_ELEMENTS> *)vbuffer;

    int first = start + threadIdx.x + blockIdx.x*blockDim.x;

    for (int i=first; i<end; i += blockDim.x * gridDim.x) {

        int const * lindices = indices + offsets[i];
        float const * lweights = weights + offsets[i];

        DeviceVertex<NUM_ELEMENTS> dst;
        dst.clear();

        for (int j=0; j<sizes[i]; ++j) {
            dst.addWithWeight(src[lindices[j]], lweights[j]);
        }
        verts[i] = dst;
    }
}

__global__ void
computeStencils(float const * cvs, float * dst,
               int length, int stride,
               unsigned char const * sizes,
               int const * offsets,
               int const * indices,
               float const * weights,
               int start, int end) {

    int first = start + threadIdx.x + blockIdx.x*blockDim.x;

    for (int i=first; i<end; i += blockDim.x * gridDim.x) {

        int const * lindices = indices + offsets[i];
        float const * lweights = weights + offsets[i];

        float * dstVert = dst + i*stride;
        clear(dstVert, length);

        for (int j=0; j<sizes[i]; ++j) {

            float const * srcVert = cvs + lindices[j]*stride;

            addWithWeight(dstVert, srcVert, lweights[j], length);
        }
    }
}

// -----------------------------------------------------------------------------

#include "../version.h"

#define OPT_KERNEL(NUM_ELEMENTS, KERNEL, X, Y, ARG) \
    if (length==NUM_ELEMENTS && stride==length) {   \
        KERNEL<NUM_ELEMENTS><<<X,Y>>>ARG;             \
        return;                                     \
    }

extern "C" {

void
CudaComputeStencils(float const *cvs, float * dst,
                    int length, int stride,
                    unsigned char const * sizes,
                    int const * offsets,
                    int const * indices,
                    float const * weights,
                    int start, int end)
{
    assert(cvs and dst and sizes and offsets and indices and weights and (end>=start));

    if (length==0 or stride==0) {
        return;
    }

    OPT_KERNEL(3, computeStencils, 512, 32, (cvs, dst, sizes, offsets, indices, weights, start, end));
    OPT_KERNEL(4, computeStencils, 512, 32, (cvs, dst, sizes, offsets, indices, weights, start, end));

    computeStencils <<<512, 32>>>(cvs, dst, length, stride,
        sizes, offsets, indices, weights, start, end);
}

// -----------------------------------------------------------------------------

}  /* extern "C" */
