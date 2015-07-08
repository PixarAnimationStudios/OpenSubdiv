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
                int const * sizes,
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
                int length,
                int srcStride,
                int dstStride,
                int const * sizes,
                int const * offsets,
                int const * indices,
                float const * weights,
                int start, int end) {

    int first = start + threadIdx.x + blockIdx.x*blockDim.x;

    for (int i=first; i<end; i += blockDim.x * gridDim.x) {

        int const * lindices = indices + offsets[i];
        float const * lweights = weights + offsets[i];

        float * dstVert = dst + i*dstStride;
        clear(dstVert, length);

        for (int j=0; j<sizes[i]; ++j) {

            float const * srcVert = cvs + lindices[j]*srcStride;

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
                                  int const *__restrict sizes,
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
                                     int const *__restrict sizes,
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
      float4 tmp = reinterpret_cast<const float4 *>(cvs)[indices[j]];
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

// Osd::PatchCoord osd/types.h
struct PatchCoord {
    int arrayIndex;
    int patchIndex;
    int vertIndex;
    float s;
    float t;
};
struct PatchArray {
    int patchType;        // Far::PatchDescriptor::Type
    int numPatches;
    int indexBase;        // offset in the index buffer
    int primitiveIdBase;  // offset in the patch param buffer
};
struct PatchParam {
    unsigned int field0;
    unsigned int field1;
    float sharpness;
};

__device__ void
getBSplineWeights(float t, float point[4], float deriv[4]) {
    // The four uniform cubic B-Spline basis functions evaluated at t:
    float const one6th = 1.0f / 6.0f;

    float t2 = t * t;
    float t3 = t * t2;

    point[0] = one6th * (1.0f - 3.0f*(t -      t2) -      t3);
    point[1] = one6th * (4.0f           - 6.0f*t2  + 3.0f*t3);
    point[2] = one6th * (1.0f + 3.0f*(t +      t2  -      t3));
    point[3] = one6th * (                                 t3);

    // Derivatives of the above four basis functions at t:
    if (deriv) {
        deriv[0] = -0.5f*t2 +      t - 0.5f;
        deriv[1] =  1.5f*t2 - 2.0f*t;
        deriv[2] = -1.5f*t2 +      t + 0.5f;
        deriv[3] =  0.5f*t2;
    }
}

__device__ void
adjustBoundaryWeights(unsigned int bits, float sWeights[4], float tWeights[4]) {
    int boundary = ((bits >> 8) & 0xf);  // far/patchParam.h

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

__device__
int getDepth(unsigned int patchBits) {
    return (patchBits & 0xf);
}

__device__
float getParamFraction(unsigned int patchBits) {
    bool nonQuadRoot = (patchBits >> 4) & 0x1;
    int depth = getDepth(patchBits);
    if (nonQuadRoot) {
        return 1.0f / float( 1 << (depth-1) );
    } else {
        return 1.0f / float( 1 << depth );
    }
}

__device__
void normalizePatchCoord(unsigned int patchBits, float *u, float *v) {
    float frac = getParamFraction(patchBits);

    int iu = (patchBits >> 22) & 0x3ff;
    int iv = (patchBits >> 12) & 0x3ff;

    // top left corner
    float pu = (float)iu*frac;
    float pv = (float)iv*frac;

    // normalize u,v coordinates
    *u = (*u - pu) / frac;
    *v = (*v - pv) / frac;
}

__global__ void
computePatches(const float *src, float *dst, float *dstDu, float *dstDv,
               int length, int srcStride, int dstStride, int dstDuStride, int dstDvStride,
               int numPatchCoords, const PatchCoord *patchCoords,
               const PatchArray *patchArrayBuffer,
               const int *patchIndexBuffer,
               const PatchParam *patchParamBuffer) {

    int first = threadIdx.x + blockIdx.x * blockDim.x;

    // PERFORMANCE: not yet optimized

    float wP[20], wDs[20], wDt[20];

    for (int i = first; i < numPatchCoords; i += blockDim.x * gridDim.x) {

        PatchCoord const &coord = patchCoords[i];
        PatchArray const &array = patchArrayBuffer[coord.arrayIndex];

        int patchType = 6; // array.patchType XXX: REGULAR only for now.
        int numControlVertices = 16;
        // note: patchIndex is absolute.
        unsigned int patchBits = patchParamBuffer[coord.patchIndex].field1;

        // normalize
        float s = coord.s;
        float t = coord.t;
        normalizePatchCoord(patchBits, &s, &t);
        float dScale = (float)(1 << getDepth(patchBits));

        if (patchType == 6) {
            float sWeights[4], tWeights[4], dsWeights[4], dtWeights[4];
            getBSplineWeights(s, sWeights, dsWeights);
            getBSplineWeights(t, tWeights, dtWeights);

            // Compute the tensor product weight of the (s,t) basis function
            // corresponding to each control vertex:
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
            // TODO: Gregory Basis.
            continue;
        }
        const int *cvs = patchIndexBuffer + array.indexBase + coord.vertIndex;

        float * dstVert = dst + i * dstStride;
        clear(dstVert, length);
        for (int j = 0; j < numControlVertices; ++j) {
            const float * srcVert = src + cvs[j] * srcStride;
            addWithWeight(dstVert, srcVert, wP[j], length);
        }
        if (dstDu) {
            float *d = dstDu + i * dstDuStride;
            clear(d, length);
            for (int j = 0; j < numControlVertices; ++j) {
                const float * srcVert = src + cvs[j] * srcStride;
                addWithWeight(d, srcVert, wDs[j], length);
            }
        }
        if (dstDv) {
            float *d = dstDv + i * dstDvStride;
            clear(d, length);
            for (int j = 0; j < numControlVertices; ++j) {
                const float * srcVert = src + cvs[j] * srcStride;
                addWithWeight(d, srcVert, wDt[j], length);
            }
        }
    }
}

// -----------------------------------------------------------------------------

#include "../version.h"

#define OPT_KERNEL(NUM_ELEMENTS, KERNEL, X, Y, ARG) \
    if (length==NUM_ELEMENTS && srcStride==length && dstStride==length) {   \
        KERNEL<NUM_ELEMENTS><<<X,Y>>>ARG;             \
        return;                                     \
    }

#ifdef USE_NVIDIA_OPTIMIZATION
#define OPT_KERNEL_NVIDIA(NUM_ELEMENTS, KERNEL, X, Y, ARG) \
    if (length==NUM_ELEMENTS && srcStride==length && dstStride==length) {   \
        int gridDim = min(X, (end-start+Y-1)/Y); \
        KERNEL<NUM_ELEMENTS, Y><<<gridDim, Y>>>ARG; \
        return;                                     \
    }
#endif

extern "C" {

void CudaEvalStencils(
    const float *src, float *dst,
    int length, int srcStride, int dstStride,
    const int * sizes, const int * offsets, const int * indices,
    const float * weights,
    int start, int end) {
    if (length == 0 or srcStride == 0 or dstStride == 0 or (end <= start)) {
        return;
    }

#ifdef USE_NVIDIA_OPTIMIZATION
    OPT_KERNEL_NVIDIA(3, computeStencilsNv, 2048, 256,
                      (src, dst, sizes, offsets, indices, weights, start, end));
    //OPT_KERNEL_NVIDIA(4, computeStencilsNv, 2048, 256,
    //                  (cvs, dst, sizes, offsets, indices, weights, start, end));
    if (length == 4 && srcStride == length && dstStride == length) {
      int gridDim = min(2048, (end-start+256-1)/256);
      computeStencilsNv_v4<256><<<gridDim, 256>>>(
          src, dst, sizes, offsets, indices, weights, start, end);
      return;
    }
#else
    OPT_KERNEL(3, computeStencils, 512, 32,
               (src, dst, sizes, offsets, indices, weights, start, end));
    OPT_KERNEL(4, computeStencils, 512, 32,
               (src, dst, sizes, offsets, indices, weights, start, end));
#endif

    // generic case (slow)
    computeStencils <<<512, 32>>>(
        src, dst, length, srcStride, dstStride,
        sizes, offsets, indices, weights, start, end);
}

// -----------------------------------------------------------------------------

void CudaEvalPatches(
    const float *src, float *dst,
    int length, int srcStride, int dstStride,
    int numPatchCoords, const PatchCoord *patchCoords,
    const PatchArray *patchArrayBuffer,
    const int *patchIndexBuffer,
    const PatchParam *patchParamBuffer) {

    // PERFORMANCE: not optimized at all

    computePatches <<<512, 32>>>(
        src, dst, NULL, NULL, length, srcStride, dstStride, 0, 0,
        numPatchCoords, patchCoords,
        patchArrayBuffer, patchIndexBuffer, patchParamBuffer);
}

void CudaEvalPatchesWithDerivatives(
    const float *src, float *dst, float *dstDu, float *dstDv,
    int length, int srcStride, int dstStride, int dstDuStride, int dstDvStride,
    int numPatchCoords, const PatchCoord *patchCoords,
    const PatchArray *patchArrayBuffer,
    const int *patchIndexBuffer,
    const PatchParam *patchParamBuffer) {

    // PERFORMANCE: not optimized at all

    computePatches <<<512, 32>>>(
        src, dst, dstDu, dstDv, length, srcStride, dstStride, dstDuStride, dstDvStride,
        numPatchCoords, patchCoords,
        patchArrayBuffer, patchIndexBuffer, patchParamBuffer);
}

}  /* extern "C" */
