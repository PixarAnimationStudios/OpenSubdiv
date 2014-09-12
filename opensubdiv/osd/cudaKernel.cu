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

// --------------------------------------------------------------------------------------------

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

#define USE_NVIDIA_OPTIMIZATION
#ifdef USE_NVIDIA_OPTIMIZATION

template< int NUM_ELEMENTS, int NUM_THREADS_PER_BLOCK >
__global__ void computeStencilsNv(float const *__restrict cvs,
                                  float * vbuffer,
                                  unsigned char const *__restrict sizes,
                                  int const *__restrict offsets,
                                  int const *__restrict indices,
                                  float const *__restrict weights,
                                  int start,
                                  int end)
{
  // Shared memory to stage indices/weights.
  __shared__ int   smem_indices_buffer[NUM_THREADS_PER_BLOCK];
  __shared__ float smem_weights_buffer[NUM_THREADS_PER_BLOCK];

  // The size of a single warp.
  const int WARP_SIZE = 32;
  // The number of warps per block.
  const int NUM_WARPS_PER_BLOCK = NUM_THREADS_PER_BLOCK / WARP_SIZE;
  // The number of outputs computed by a single warp.
  const int NUM_OUTPUTS_PER_WARP = WARP_SIZE / NUM_ELEMENTS;
  // The number of outputs computed by a block of threads.
  const int NUM_OUTPUTS_PER_BLOCK = NUM_OUTPUTS_PER_WARP*NUM_WARPS_PER_BLOCK;
  // The number of active threads in a warp.
  const int NUM_ACTIVE_THREADS_PER_WARP = NUM_OUTPUTS_PER_WARP * NUM_ELEMENTS;

  // The number of the warp inside the block.
  const int warpId = threadIdx.x / WARP_SIZE;
  const int laneId = threadIdx.x % WARP_SIZE;

  // We use NUM_ELEMENTS threads per output. Find which output/element a thread works on.
  int outputIdx = warpId*NUM_OUTPUTS_PER_WARP + laneId/NUM_ELEMENTS, elementIdx = laneId%NUM_ELEMENTS;

  // Each output corresponds to a section of shared memory.
  volatile int   *smem_indices = &smem_indices_buffer[warpId*WARP_SIZE + (laneId/NUM_ELEMENTS)*NUM_ELEMENTS];
  volatile float *smem_weights = &smem_weights_buffer[warpId*WARP_SIZE + (laneId/NUM_ELEMENTS)*NUM_ELEMENTS];

  // Disable threads that have nothing to do inside the warp.
  int i = end;
  if( laneId < NUM_ACTIVE_THREADS_PER_WARP )
    i = start + blockIdx.x*NUM_OUTPUTS_PER_BLOCK + outputIdx;

  // Iterate over the vertices.
  for( ; i < end ; i += gridDim.x*NUM_OUTPUTS_PER_BLOCK )
  {
    // Each thread computes an element of the final vertex.
    float x = 0.f;

    // Load the offset and the size for each vertex. We have NUM_THREADS_PER_VERTEX threads loading the same value.
    const int offset_i = offsets[i], size_i = sizes[i];

    // Iterate over the stencil.
    for( int j = offset_i, j_end = offset_i+size_i ; j < j_end ; )
    {
      int j_it = j + elementIdx;

      // Load some indices and some weights. The transaction is coalesced.
      smem_indices[elementIdx] = j_it < j_end ? indices[j_it] : 0;
      smem_weights[elementIdx] = j_it < j_end ? weights[j_it] : 0.f;

      // Thread now collaborates to load the vertices.
      #pragma unroll
      for( int k = 0 ; k < NUM_ELEMENTS ; ++k, ++j )
        if( j < j_end )
          x += smem_weights[k] * cvs[smem_indices[k]*NUM_ELEMENTS + elementIdx];
    }

    // Store the vertex.
    vbuffer[NUM_ELEMENTS*i + elementIdx] = x;
  }
}

template< int NUM_THREADS_PER_BLOCK >
__global__ void computeStencilsNv_v4(float const *__restrict cvs,
                                     float * vbuffer,
                                     unsigned char const *__restrict sizes,
                                     int const *__restrict offsets,
                                     int const *__restrict indices,
                                     float const *__restrict weights,
                                     int start,
                                     int end)
{
  // Iterate over the vertices.
  for( int i = start + blockIdx.x*NUM_THREADS_PER_BLOCK + threadIdx.x ; i < end ; i += gridDim.x*NUM_THREADS_PER_BLOCK )
  {
    // Each thread computes an element of the final vertex.
    float4 x = make_float4(0.f, 0.f, 0.f, 0.f);

    // Iterate over the stencil.
    for( int j = offsets[i], j_end = offsets[i]+sizes[i] ; j < j_end ; ++j )
    {
      float w = weights[j];
      float4 tmp = reinterpret_cast<const float4 *__restrict>(cvs)[indices[j]];
      x.x += w*tmp.x;
      x.y += w*tmp.y;
      x.z += w*tmp.z;
      x.w += w*tmp.w;
    }

    // Store the vertex.
    reinterpret_cast<float4*>(vbuffer)[i] = x;
  }
}

#endif // USE_NVIDIA_OPTIMIZATION

// -----------------------------------------------------------------------------

#include "../version.h"

#define OPT_KERNEL(NUM_ELEMENTS, KERNEL, X, Y, ARG) \
    if (length==NUM_ELEMENTS && stride==length) {   \
        KERNEL<NUM_ELEMENTS><<<X,Y>>>ARG;             \
        return;                                     \
    }

#ifdef USE_NVIDIA_OPTIMIZATION
#define OPT_KERNEL_NVIDIA(NUM_ELEMENTS, KERNEL, X, Y, ARG) \
    if (length==NUM_ELEMENTS && stride==length) {   \
        int gridDim = min(X, (end-start+Y-1)/Y); \
        KERNEL<NUM_ELEMENTS, Y><<<gridDim, Y>>>ARG; \
        return;                                     \
    }
#endif

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

#ifdef USE_NVIDIA_OPTIMIZATION
    OPT_KERNEL_NVIDIA(3, computeStencilsNv, 2048, 256, (cvs, dst, sizes, offsets, indices, weights, start, end));
    //OPT_KERNEL_NVIDIA(4, computeStencilsNv, 2048, 256, (cvs, dst, sizes, offsets, indices, weights, start, end));
    if( length==4 && stride==length ) {
      int gridDim = min(2048, (end-start+256-1)/256);
      computeStencilsNv_v4<256><<<gridDim, 256>>>(cvs, dst, sizes, offsets, indices, weights, start, end);
      return;
    }
#else
    OPT_KERNEL(3, computeStencils, 512, 32, (cvs, dst, sizes, offsets, indices, weights, start, end));
    OPT_KERNEL(4, computeStencils, 512, 32, (cvs, dst, sizes, offsets, indices, weights, start, end));
#endif

    computeStencils <<<512, 32>>>(cvs, dst, length, stride,
        sizes, offsets, indices, weights, start, end);
}

// -----------------------------------------------------------------------------

}  /* extern "C" */
