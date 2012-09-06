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
#include <stdio.h>
#include <assert.h>

#define USE_BLOCK_OPTIM

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

#ifdef USE_BLOCK_OPTIM

template< int NUM_THREADS_PER_VERTEX, int NUM_VARYING_ELEMENTS >
struct Parallel_varying_vertex
{
  // The number of elements of the varying vertex each thread is responsible for.
  static const int NUM_ELEMENTS_PER_THREAD = (NUM_VARYING_ELEMENTS + NUM_THREADS_PER_VERTEX-1) / NUM_THREADS_PER_VERTEX;

  // The elements.
  float m_elts[NUM_ELEMENTS_PER_THREAD];

  // Initialize the vertex.
  __device__ __forceinline__ Parallel_varying_vertex() 
  {
    #pragma unroll
    for( int i = 0 ; i < NUM_ELEMENTS_PER_THREAD ; ++i )
      m_elts[i] = 0.0f;
  }

  // Load a vertex.
  __device__ __forceinline__ void load( const float *ptr, int element )
  {
    #pragma unroll
    for( int i = 0 ; i < NUM_ELEMENTS_PER_THREAD ; ++i )
    {
      const int idx = i*NUM_THREADS_PER_VERTEX + element;
      if( idx < NUM_VARYING_ELEMENTS )
        m_elts[i] = ptr[idx];
    }
  }

  // Store a vertex.
  __device__ __forceinline__ void store( float *ptr, int element )
  {
    #pragma unroll
    for( int i = 0 ; i < NUM_ELEMENTS_PER_THREAD ; ++i )
    {
      const int idx = i*NUM_THREADS_PER_VERTEX + element;
      if( idx < NUM_VARYING_ELEMENTS )
        ptr[idx] = m_elts[i];
    }
  }

  // Add another vertex.
  __device__ __forceinline__ void add( const Parallel_varying_vertex &other, float weight, int element )
  {
    #pragma unroll
    for( int i = 0 ; i < NUM_ELEMENTS_PER_THREAD ; ++i )
      m_elts[i] += other.m_elts[i] * weight;
  }

  // Add another vertex.
  __device__ __forceinline__ void add( const Parallel_varying_vertex &v0, const Parallel_varying_vertex &v1, float weight, int element )
  {
    #pragma unroll
    for( int i = 0 ; i < NUM_ELEMENTS_PER_THREAD ; ++i )
    {
      m_elts[i] += v0.m_elts[i] * weight;
      m_elts[i] += v1.m_elts[i] * weight;
    } 
  }
};

template< int NUM_THREADS_PER_VERTEX >
struct Parallel_varying_vertex<NUM_THREADS_PER_VERTEX, 0>
{
  __device__ __forceinline__ void load( const float *ptr, int element )
  {}

  __device__ __forceinline__ void store( float *ptr, int element )
  {}

  __device__ __forceinline__ void add( const Parallel_varying_vertex &other, float weight, int element )
  {}

  __device__ __forceinline__ void add( const Parallel_varying_vertex &v0, const Parallel_varying_vertex &v1, float weight, int element )
  {}
};

#endif

#ifdef USE_BLOCK_OPTIM

template< int NUM_THREADS_PER_BLOCK, int NUM_USER_VERTEX_ELEMENTS, int NUM_VARYING_ELEMENTS > 
__global__ 
void computeFace( float *fVertex, float *fVaryings, const int *F0_IT, const int2 *F0_ITa, int offset, int start, int end )
{
  // The number of vertex elements.
  const int NUM_VERTEX_ELEMENTS = 3 + NUM_USER_VERTEX_ELEMENTS;
  // The number of threads per vertex.
  const int NUM_THREADS_PER_VERTEX = NUM_VERTEX_ELEMENTS; // Start simple.
  // The number of vertices computed per block.
  const int NUM_VERTICES_PER_BLOCK = NUM_THREADS_PER_BLOCK / NUM_THREADS_PER_VERTEX;
  // The number of vertices per grid.
  const int NUM_VERTICES_PER_GRID = gridDim.x * NUM_VERTICES_PER_BLOCK;
  // The number of active threads per block (if NUM_THREADS_PER_BLOCK % NUM_THREADS_PER_VERTEX != 0).
  const int NUM_ACTIVE_THREADS_PER_BLOCK = NUM_VERTICES_PER_BLOCK * NUM_THREADS_PER_VERTEX;

  // Warp decomposition.
  const int vertex  = threadIdx.x / NUM_THREADS_PER_VERTEX;
  const int element = threadIdx.x % NUM_THREADS_PER_VERTEX;

  // Is the thread active.
  const bool is_active = threadIdx.x < NUM_ACTIVE_THREADS_PER_BLOCK;

  // Loop over the items...
  for( start += blockIdx.x*NUM_VERTICES_PER_BLOCK + vertex ; start < end ; start += NUM_VERTICES_PER_GRID )
  {
    // h/n.
    const int2 hn = is_active ? F0_ITa[start] : make_int2(-1, -1);

    // Compute the weight.
    float weight = 1.0f / (float) hn.y;
    
    // Each thread stores its coordinate of the vertex.
    float my_elt = 0.0f;
    // Varying vertex.
    Parallel_varying_vertex<NUM_THREADS_PER_VERTEX, NUM_VARYING_ELEMENTS> varying;

    // Compute vertices.
    for( int j = 0 ; j < hn.y ; ++j )
    {
      // Index.
      int idx = is_active ? F0_IT[hn.x+j] : 0;

      // Vertex.
      const float other_elt = is_active ? fVertex[idx*NUM_VERTEX_ELEMENTS + element] : 0.0f;
      my_elt += other_elt * weight; 

      // Varying vertex.
      if( NUM_VARYING_ELEMENTS > 0 && is_active )
      {
        Parallel_varying_vertex<NUM_THREADS_PER_VERTEX, NUM_VARYING_ELEMENTS> vtmp;
        vtmp.load( &fVaryings[idx*NUM_VARYING_ELEMENTS], element );
        varying.add( vtmp, weight, element );
      }
    }

    // Output array.
    int dst_offset = offset + start;

    // Store the results.
    if( is_active )
    {
      fVertex[dst_offset*NUM_VERTEX_ELEMENTS + element] = my_elt;
      varying.store( &fVaryings[dst_offset*NUM_VARYING_ELEMENTS], element );
    }
  }
}

#else

template <int NUM_USER_VERTEX_ELEMENTS, int NUM_VARYING_ELEMENTS> __global__ void
computeFace(float *fVertex, float *fVaryings, int *F0_IT, int *F0_ITa, int offset, int start, int end)
{
    DeviceVertex<NUM_USER_VERTEX_ELEMENTS> *vertex = (DeviceVertex<NUM_USER_VERTEX_ELEMENTS>*)fVertex;
    DeviceVarying<NUM_VARYING_ELEMENTS> *varyings = (DeviceVarying<NUM_VARYING_ELEMENTS>*)fVaryings;
    for(int i = start + threadIdx.x + blockIdx.x*blockDim.x; i < end; i += blockDim.x * gridDim.x){
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
            vertex[offset + i] = dst;
            varyings[offset + i] = dstVarying;
        }else{
            for(int j=0; j<n; ++j){
                int index = F0_IT[h+j];
                dst.addWithWeight(&vertex[index], weight);
            }
            vertex[offset + i] = dst;
        }
    }
}

#endif

__global__ void
computeFace(float *fVertex, int numVertexElements, float *fVaryings, int numVaryingElements,
            int *F0_IT, int *F0_ITa, int offset, int start, int end)
{
    for(int i = start + threadIdx.x + blockIdx.x*blockDim.x; i < end; i += blockDim.x * gridDim.x){
        int h = F0_ITa[2*i];
        int n = F0_ITa[2*i+1];
        float weight = 1.0f/n;

        // XXX: can we use local stack like alloca?
        float *dstVertex = fVertex + (i+offset)*numVertexElements;
        clear(dstVertex, numVertexElements);
        float *dstVarying = fVaryings + (i+offset)*numVaryingElements;
        clear(dstVarying, numVaryingElements);

        for(int j=0; j<n; ++j){
            int index = F0_IT[h+j];
            addWithWeight(dstVertex, fVertex + index*numVertexElements, weight, numVertexElements);
            addVaryingWithWeight(dstVarying, fVaryings + index*numVaryingElements, weight, numVaryingElements);
        }
    }
}

#ifdef USE_BLOCK_OPTIM

template< int NUM_THREADS_PER_BLOCK, int NUM_USER_VERTEX_ELEMENTS, int NUM_VARYING_ELEMENTS > 
__global__ 
void computeEdge( float *fVertex, float *fVaryings, const int4 *E0_IT, const float2 *E0_S, int offset, int start, int end )
{
  // The number of vertex elements.
  const int NUM_VERTEX_ELEMENTS = 3 + NUM_USER_VERTEX_ELEMENTS;
  // The number of threads per vertex.
  const int NUM_THREADS_PER_VERTEX = NUM_VERTEX_ELEMENTS; // Start simple.
  // The number of vertices computed per block.
  const int NUM_VERTICES_PER_BLOCK = NUM_THREADS_PER_BLOCK / NUM_THREADS_PER_VERTEX;
  // The number of vertices per grid.
  const int NUM_VERTICES_PER_GRID = gridDim.x * NUM_VERTICES_PER_BLOCK;

  // The number of active threads per block (if NUM_THREADS_PER_BLOCK % NUM_THREADS_PER_VERTEX != 0).
  const int NUM_ACTIVE_THREADS_PER_BLOCK = NUM_VERTICES_PER_BLOCK * NUM_THREADS_PER_VERTEX;

  // Warp decomposition.
  const int vertex  = threadIdx.x / NUM_THREADS_PER_VERTEX;
  const int element = threadIdx.x % NUM_THREADS_PER_VERTEX;

  // Is the thread active.
  const bool is_active = threadIdx.x < NUM_ACTIVE_THREADS_PER_BLOCK;

  // Loop over the items...
  for( start += blockIdx.x*NUM_VERTICES_PER_BLOCK + vertex ; start < end ; start += NUM_VERTICES_PER_GRID )
  {
    // Edge indices.
    const int4 eidx = is_active ? E0_IT[start] : make_int4(-1, -1, -1, -1);

    // Compute the vertex.
    float my_elt = 0.0f;
    // The vertex/face weights.
    float2 w = is_active ? E0_S[start] : make_float2(0.0f, 0.0f);

    // Add vertices.
    float other_elt0 = is_active ? fVertex[eidx.x*NUM_VERTEX_ELEMENTS + element] : 0.0f;
    my_elt += other_elt0 * w.x;
    float other_elt1 = is_active ? fVertex[eidx.y*NUM_VERTEX_ELEMENTS + element] : 0.0f;
    my_elt += other_elt1 * w.x;

    // Face vertices.
    if( is_active && eidx.z > -1 )
    {
      float other_elt2 = fVertex[eidx.z*NUM_VERTEX_ELEMENTS + element];
      my_elt += other_elt2 * w.y;
      float other_elt3 = fVertex[eidx.w*NUM_VERTEX_ELEMENTS + element];
      my_elt += other_elt3 * w.y;
    }

    // Output array.
    int dst_offset = offset + start;

    // Store the results.
    if( is_active )
      fVertex[dst_offset*NUM_VERTEX_ELEMENTS + element] = my_elt;

    // Varying vertices.
    Parallel_varying_vertex<NUM_THREADS_PER_VERTEX, NUM_VARYING_ELEMENTS> v0, v1, v2;
    if( is_active )
    {
      v0.load ( &fVaryings[eidx.x*NUM_VARYING_ELEMENTS], element );
      v1.load ( &fVaryings[eidx.y*NUM_VARYING_ELEMENTS], element );
      v2.add  ( v0, v1, 0.5f, element );
      v2.store( &fVaryings[dst_offset*NUM_VARYING_ELEMENTS], element );
    }
  }
}

#else

template <int NUM_USER_VERTEX_ELEMENTS, int NUM_VARYING_ELEMENTS> __global__ void
computeEdge(float *fVertex, float *fVaryings, int *E0_IT, float *E0_S, int offset, int start, int end)
{
    DeviceVertex<NUM_USER_VERTEX_ELEMENTS> *vertex = (DeviceVertex<NUM_USER_VERTEX_ELEMENTS>*)fVertex;
    DeviceVarying<NUM_VARYING_ELEMENTS> *varyings = (DeviceVarying<NUM_VARYING_ELEMENTS>*)fVaryings;
    for(int i = start + threadIdx.x + blockIdx.x*blockDim.x; i < end; i+= blockDim.x * gridDim.x){
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
        vertex[offset+i] = dst;

        if(NUM_VARYING_ELEMENTS > 0){
            DeviceVarying<NUM_VARYING_ELEMENTS> dstVarying;
            dstVarying.clear();
            dstVarying.addVaryingWithWeight(&varyings[eidx0], 0.5f);
            dstVarying.addVaryingWithWeight(&varyings[eidx1], 0.5f);
            varyings[offset+i] = dstVarying;
        }
    }
}

#endif

__global__ void
computeEdge(float *fVertex, int numVertexElements, float *fVarying, int numVaryingElements,
            int *E0_IT, float *E0_S, int offset, int start, int end)
{
    for(int i = start + threadIdx.x + blockIdx.x*blockDim.x; i < end; i+= blockDim.x * gridDim.x){
        int eidx0 = E0_IT[4*i+0];
        int eidx1 = E0_IT[4*i+1];
        int eidx2 = E0_IT[4*i+2];
        int eidx3 = E0_IT[4*i+3];

        float vertWeight = E0_S[i*2+0];

        // Fully sharp edge : vertWeight = 0.5f;
        float *dstVertex = fVertex + (i+offset)*numVertexElements;
        clear(dstVertex, numVertexElements);

        addWithWeight(dstVertex, fVertex + eidx0*numVertexElements, vertWeight, numVertexElements);
        addWithWeight(dstVertex, fVertex + eidx1*numVertexElements, vertWeight, numVertexElements);

        if(eidx2 > -1){
            float faceWeight = E0_S[i*2+1];

            addWithWeight(dstVertex, fVertex + eidx2*numVertexElements, faceWeight, numVertexElements);
            addWithWeight(dstVertex, fVertex + eidx3*numVertexElements, faceWeight, numVertexElements);
        }

        if(numVaryingElements > 0){
            float *dstVarying = fVarying + i*numVaryingElements;
            clear(dstVarying, numVaryingElements);

            addVaryingWithWeight(dstVarying, fVarying + eidx0*numVaryingElements, 0.5f, numVaryingElements);
            addVaryingWithWeight(dstVarying, fVarying + eidx1*numVaryingElements, 0.5f, numVaryingElements);
        }
    }
}

template <int NUM_USER_VERTEX_ELEMENTS, int NUM_VARYING_ELEMENTS> __global__ void
computeVertexA(float *fVertex, float *fVaryings, int *V0_ITa, float *V0_S, int offset, int start, int end, int pass)
{
    DeviceVertex<NUM_USER_VERTEX_ELEMENTS> *vertex = (DeviceVertex<NUM_USER_VERTEX_ELEMENTS>*)fVertex;
    DeviceVarying<NUM_VARYING_ELEMENTS> *varyings = (DeviceVarying<NUM_VARYING_ELEMENTS>*)fVaryings;
    for(int i = start + threadIdx.x + blockIdx.x*blockDim.x; i < end; i += blockDim.x * gridDim.x){
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
            dst = vertex[i+offset];
        }

        if (eidx0==-1 || (pass==0 && (n==-1)) ) {
            dst.addWithWeight(&vertex[p], weight);
        } else {
            dst.addWithWeight(&vertex[p], weight * 0.75f);
            dst.addWithWeight(&vertex[eidx0], weight * 0.125f);
            dst.addWithWeight(&vertex[eidx1], weight * 0.125f);
        }
        vertex[i+offset] = dst;

        if(NUM_VARYING_ELEMENTS > 0){
            if(not pass){
                DeviceVarying<NUM_VARYING_ELEMENTS> dstVarying;
                dstVarying.clear();
                dstVarying.addVaryingWithWeight(&varyings[p], 1.0f);
                varyings[i+offset] = dstVarying;
            }
        }
    }
}

__global__ void
computeVertexA(float *fVertex, int numVertexElements, float *fVaryings, int numVaryingElements,
               int *V0_ITa, float *V0_S, int offset, int start, int end, int pass)
{
    for(int i = start + threadIdx.x + blockIdx.x*blockDim.x; i < end; i += blockDim.x * gridDim.x){
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

        float *dstVertex = fVertex + (i+offset)*numVertexElements;
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
                float *dstVarying = fVaryings + i*numVaryingElements;
                clear(dstVarying, numVaryingElements);
                addVaryingWithWeight(dstVarying, fVaryings + p*numVaryingElements, 1.0f, numVaryingElements);
            }
        }
    }

}


//texture <int, 1> texV0_IT;

#ifdef USE_BLOCK_OPTIM

template< int NUM_THREADS_PER_BLOCK, int NUM_USER_VERTEX_ELEMENTS, int NUM_VARYING_ELEMENTS > 
__global__ 
void computeVertexB( float *fVertex, float *fVaryings, const int *V0_ITa, const int *V0_IT, const float *V0_S, int offset, int start, int end )
{
  // The number of vertex elements.
  const int NUM_VERTEX_ELEMENTS = 3 + NUM_USER_VERTEX_ELEMENTS;
  // The number of threads per vertex.
  const int NUM_THREADS_PER_VERTEX = NUM_VERTEX_ELEMENTS; // Start simple.
  // The number of vertices computed per block.
  const int NUM_VERTICES_PER_BLOCK = NUM_THREADS_PER_BLOCK / NUM_THREADS_PER_VERTEX;
  // The number of vertices per grid.
  const int NUM_VERTICES_PER_GRID = gridDim.x * NUM_VERTICES_PER_BLOCK;

  // The number of active threads per block (if NUM_THREADS_PER_BLOCK % NUM_THREADS_PER_VERTEX != 0).
  const int NUM_ACTIVE_THREADS_PER_BLOCK = NUM_VERTICES_PER_BLOCK * NUM_THREADS_PER_VERTEX;

  // Use SMEM to load indices.
  __shared__ int smem[NUM_THREADS_PER_BLOCK];

  // Warp decomposition.
  const int vertex  = threadIdx.x / NUM_THREADS_PER_VERTEX;
  const int element = threadIdx.x % NUM_THREADS_PER_VERTEX;

  // Is the thread active.
  const bool is_active = threadIdx.x < NUM_ACTIVE_THREADS_PER_BLOCK;

  // Shared memory for the vertex.
  int *vertex_smem = &smem[vertex*NUM_THREADS_PER_VERTEX];

  // Loop over the items...
  for( start += blockIdx.x*NUM_VERTICES_PER_BLOCK + vertex ; start < end ; start += NUM_VERTICES_PER_GRID )
  {
    // Load info. Better coalescing.
#if 0
    const int h = is_active ? V0_ITa[5*start + 0] : 0;
    const int n = is_active ? V0_ITa[5*start + 1] : 0;
    const int p = is_active ? V0_ITa[5*start + 2] : 0;
#else
    if( is_active && element < 3 )
      smem[threadIdx.x] = V0_ITa[5*start + element];
    __syncthreads();

    // h/n/p.
    const int h = is_active ? vertex_smem[0] : -1;
    const int n = is_active ? vertex_smem[1] : -1;
    const int p = is_active ? vertex_smem[2] : -1;
#endif

    // The weight.
    float weight = V0_S[start];
    // Compute the weight.
    float inv_n = 1.0f / (float) n;
    // Compute factors.
    float weight_wp = weight * inv_n * inv_n;
    float weight_wv = weight -  2.0f * inv_n;

    // Each thread stores its coordinate of the vertex.
    float my_elt = is_active ? fVertex[p*NUM_VERTEX_ELEMENTS + element] * weight_wv : 0.0f;

    #pragma unroll
    for( int j = 0 ; j < n ; ++j )
    {
      // TODO: make sure h is always even!!! Otherwise use #if 0:
#if 1
      // Load indices.
      int2 idx = is_active ? reinterpret_cast<const int2 *>( &V0_IT[h] )[j] : make_int2(0, 0);
#else
      int2 idx = make_int2(0, 0);
      if( is_active )
      {
        idx.x = V0_IT[h + 2*j + 0];
        idx.y = V0_IT[h + 2*j + 1];
      }
#endif
      
      // Load vertices and update the coordinates.
      const float other_elt0 = is_active ? fVertex[idx.x*NUM_VERTEX_ELEMENTS + element] : 0.0f;
      my_elt += other_elt0 * weight_wp; 
     
      const float other_elt1 = is_active ? fVertex[idx.y*NUM_VERTEX_ELEMENTS + element] : 0.0f;
      my_elt += other_elt1 * weight_wp; 
    }

    // Output array.
    int dst_offset = offset + start;

    // Store the results.
    if( is_active )
      fVertex[dst_offset*NUM_VERTEX_ELEMENTS + element] = my_elt;

    // Varying vertices.
    Parallel_varying_vertex<NUM_THREADS_PER_VERTEX, NUM_VARYING_ELEMENTS> v;
    if( is_active )
    {
      v.load( &fVaryings[p*NUM_VARYING_ELEMENTS], element );
      v.store( &fVaryings[dst_offset*NUM_VARYING_ELEMENTS], element );
    }
    __syncthreads();
  }
}

#else

template <int NUM_USER_VERTEX_ELEMENTS, int NUM_VARYING_ELEMENTS> __global__ void
computeVertexB(float *fVertex, float *fVaryings,
                    const int *V0_ITa, const int *V0_IT, const float *V0_S, int offset, int start, int end)
{
    DeviceVertex<NUM_USER_VERTEX_ELEMENTS> *vertex = (DeviceVertex<NUM_USER_VERTEX_ELEMENTS>*)fVertex;
    DeviceVarying<NUM_VARYING_ELEMENTS> *varyings = (DeviceVarying<NUM_VARYING_ELEMENTS>*)fVaryings;
    for(int i = start + threadIdx.x + blockIdx.x*blockDim.x; i < end; i += blockDim.x * gridDim.x){
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
        vertex[i+offset] = dst;

        if(NUM_VARYING_ELEMENTS > 0){
            DeviceVarying<NUM_VARYING_ELEMENTS> dstVarying;
            dstVarying.clear();
            dstVarying.addVaryingWithWeight(&varyings[p], 1.0f);
            varyings[i+offset] = dstVarying;
        }
    }
}
#endif

__global__ void
computeVertexB(float *fVertex, int numVertexElements, float *fVaryings, int numVaryingElements,
               const int *V0_ITa, const int *V0_IT, const float *V0_S, int offset, int start, int end)
{
    for(int i = start + threadIdx.x + blockIdx.x*blockDim.x; i < end; i += blockDim.x * gridDim.x){
        int h = V0_ITa[5*i];
        int n = V0_ITa[5*i+1];
        int p = V0_ITa[5*i+2];

        float weight = V0_S[i];
        float wp = 1.0f/float(n*n);
        float wv = (n-2.0f) * n * wp;

        float *dstVertex = fVertex + (i+offset)*numVertexElements;
        clear(dstVertex, numVertexElements);
        addWithWeight(dstVertex, fVertex + p*numVertexElements, weight*wv, numVertexElements);

        for(int j = 0; j < n; ++j){
            addWithWeight(dstVertex, fVertex + V0_IT[h+j*2]*numVertexElements, weight*wp, numVertexElements);
            addWithWeight(dstVertex, fVertex + V0_IT[h+j*2+1]*numVertexElements, weight*wp, numVertexElements);
        }

        if(numVaryingElements > 0){
            float *dstVarying = fVaryings + i*numVaryingElements;
            clear(dstVarying, numVaryingElements);
            addVaryingWithWeight(dstVarying, fVaryings + p*numVaryingElements, 1.0f, numVaryingElements);
        }
    }
}


// --------------------------------------------------------------------------------------------

template <int NUM_USER_VERTEX_ELEMENTS, int NUM_VARYING_ELEMENTS> __global__ void
computeLoopVertexB(float *fVertex, float *fVaryings, int *V0_ITa, int *V0_IT, float *V0_S, int offset, int start, int end)
{
    DeviceVertex<NUM_USER_VERTEX_ELEMENTS> *vertex = (DeviceVertex<NUM_USER_VERTEX_ELEMENTS>*)fVertex;
    DeviceVarying<NUM_VARYING_ELEMENTS> *varyings = (DeviceVarying<NUM_VARYING_ELEMENTS>*)fVaryings;
    for(int i = start + threadIdx.x + blockIdx.x*blockDim.x; i < end; i += blockDim.x * gridDim.x){
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
        vertex[i+offset] = dst;

        if(NUM_VARYING_ELEMENTS > 0){
            DeviceVarying<NUM_VARYING_ELEMENTS> dstVarying;
            dstVarying.clear();
            dstVarying.addVaryingWithWeight(&varyings[p], 1.0f);
            varyings[i+offset] = dstVarying;
        }
    }
}

__global__ void
computeLoopVertexB(float *fVertex, int numVertexElements, float *fVaryings, int numVaryingElements,
               const int *V0_ITa, const int *V0_IT, const float *V0_S, int offset, int start, int end)
{
    for(int i = start + threadIdx.x + blockIdx.x*blockDim.x; i < end; i += blockDim.x * gridDim.x){
        int h = V0_ITa[5*i];
        int n = V0_ITa[5*i+1];
        int p = V0_ITa[5*i+2];

        float weight = V0_S[i];
        float wp = 1.0f/float(n);
        float beta = 0.25f * __cosf(float(M_PI) * 2.0f * wp) + 0.375f;
        beta = beta * beta;
        beta = (0.625f - beta) * wp;

        float *dstVertex = fVertex + (i+offset)*numVertexElements;
        clear(dstVertex, numVertexElements);
        addWithWeight(dstVertex, fVertex + p*numVertexElements, weight*(1.0f-(beta*n)), numVertexElements);

        for(int j = 0; j < n; ++j){
            addWithWeight(dstVertex, fVertex + V0_IT[h+j]*numVertexElements, weight*beta, numVertexElements);
        }

        if(numVaryingElements > 0){
            float *dstVarying = fVaryings + i*numVaryingElements;
            clear(dstVarying, numVaryingElements);
            addVaryingWithWeight(dstVarying, fVaryings + p*numVaryingElements, 1.0f, numVaryingElements);
        }
    }
}

// --------------------------------------------------------------------------------------------

template <int NUM_USER_VERTEX_ELEMENTS, int NUM_VARYING_ELEMENTS> __global__ void
computeBilinearEdge(float *fVertex, float *fVaryings, int *E0_IT, int offset, int start, int end)
{
    DeviceVertex<NUM_USER_VERTEX_ELEMENTS> *vertex = (DeviceVertex<NUM_USER_VERTEX_ELEMENTS>*)fVertex;
    DeviceVarying<NUM_VARYING_ELEMENTS> *varyings = (DeviceVarying<NUM_VARYING_ELEMENTS>*)fVaryings;
    for(int i = start + threadIdx.x + blockIdx.x*blockDim.x; i < end; i+= blockDim.x * gridDim.x){
        int eidx0 = E0_IT[2*i+0];
        int eidx1 = E0_IT[2*i+1];

        DeviceVertex<NUM_USER_VERTEX_ELEMENTS> dst;
        dst.clear();

        dst.addWithWeight(&vertex[eidx0], 0.5f);
        dst.addWithWeight(&vertex[eidx1], 0.5f);

        vertex[offset+i] = dst;

        if(NUM_VARYING_ELEMENTS > 0){
            DeviceVarying<NUM_VARYING_ELEMENTS> dstVarying;
            dstVarying.clear();
            dstVarying.addVaryingWithWeight(&varyings[eidx0], 0.5f);
            dstVarying.addVaryingWithWeight(&varyings[eidx1], 0.5f);
            varyings[offset+i] = dstVarying;
        }
    }
}

__global__ void
computeBilinearEdge(float *fVertex, int numVertexElements, float *fVarying, int numVaryingElements,
                    int *E0_IT, int offset, int start, int end)
{
    for(int i = start + threadIdx.x + blockIdx.x*blockDim.x; i < end; i+= blockDim.x * gridDim.x){
        int eidx0 = E0_IT[2*i+0];
        int eidx1 = E0_IT[2*i+1];

        float *dstVertex = fVertex + (i+offset)*numVertexElements;
        clear(dstVertex, numVertexElements);

        addWithWeight(dstVertex, fVertex + eidx0*numVertexElements, 0.5f, numVertexElements);
        addWithWeight(dstVertex, fVertex + eidx1*numVertexElements, 0.5f, numVertexElements);

        if(numVaryingElements > 0){
            float *dstVarying = fVarying + i*numVaryingElements;
            clear(dstVarying, numVaryingElements);

            addVaryingWithWeight(dstVarying, fVarying + eidx0*numVaryingElements, 0.5f, numVaryingElements);
            addVaryingWithWeight(dstVarying, fVarying + eidx1*numVaryingElements, 0.5f, numVaryingElements);
        }
    }
}

template <int NUM_USER_VERTEX_ELEMENTS, int NUM_VARYING_ELEMENTS> __global__ void
computeBilinearVertex(float *fVertex, float *fVaryings, int *V0_ITa, int offset, int start, int end)
{
    DeviceVertex<NUM_USER_VERTEX_ELEMENTS> *vertex = (DeviceVertex<NUM_USER_VERTEX_ELEMENTS>*)fVertex;
    DeviceVarying<NUM_VARYING_ELEMENTS> *varyings = (DeviceVarying<NUM_VARYING_ELEMENTS>*)fVaryings;
    for(int i = start + threadIdx.x + blockIdx.x*blockDim.x; i < end; i += blockDim.x * gridDim.x){
        int p = V0_ITa[i];

        DeviceVertex<NUM_USER_VERTEX_ELEMENTS> dst;
        dst.clear();

        dst.addWithWeight(&vertex[p], 1.0f);
        vertex[i+offset] = dst;

        if(NUM_VARYING_ELEMENTS > 0){
            DeviceVarying<NUM_VARYING_ELEMENTS> dstVarying;
            dstVarying.clear();
            dstVarying.addVaryingWithWeight(&varyings[p], 1.0f);
            varyings[i+offset] = dstVarying;
        }
    }
}

__global__ void
computeBilinearVertex(float *fVertex, int numVertexElements, float *fVaryings, int numVaryingElements,
               const int *V0_ITa, int offset, int start, int end)
{
    for(int i = start + threadIdx.x + blockIdx.x*blockDim.x; i < end; i += blockDim.x * gridDim.x){
        int p = V0_ITa[i];

        float *dstVertex = fVertex + (i+offset)*numVertexElements;
        clear(dstVertex, numVertexElements);
        addWithWeight(dstVertex, fVertex + p*numVertexElements, 1.0f, numVertexElements);

        if(numVaryingElements > 0){
            float *dstVarying = fVaryings + i*numVaryingElements;
            clear(dstVarying, numVaryingElements);
            addVaryingWithWeight(dstVarying, fVaryings + p*numVaryingElements, 1.0f, numVaryingElements);
        }
    }
}

// --------------------------------------------------------------------------------------------

__global__ void
editVertexAdd(float *fVertex, int numVertexElements, int primVarOffset, int primVarWidth,
              int numVertices, const int *editIndices, const float *editValues)
{
    for(int i = threadIdx.x + blockIdx.x*blockDim.x; i < numVertices; i += blockDim.x * gridDim.x) {
        float *dstVertex = fVertex + editIndices[i] * numVertexElements + primVarOffset;

        for(int j = 0; j < primVarWidth; j++) {
            *dstVertex++ += editValues[j];
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

#if defined USE_BLOCK_OPTIM
#define OPT_KERNEL_0(NUM_USER_VERTEX_ELEMENTS, NUM_VARYING_ELEMENTS, KERNEL, X, Y, ARG) \
    if(numUserVertexElements == NUM_USER_VERTEX_ELEMENTS && \
       numVaryingElements == NUM_VARYING_ELEMENTS) \
       { KERNEL<Y, NUM_USER_VERTEX_ELEMENTS, NUM_VARYING_ELEMENTS><<<X,Y>>>ARG; \
         return;  }
#endif

extern "C" {

void OsdCudaComputeFace(float *vertex, float *varying,
                        int numUserVertexElements, int numVaryingElements,
                        int *F_IT, int *F_ITa, int offset, int start, int end)
{
#if defined USE_BLOCK_OPTIM
    OPT_KERNEL_0(0, 0, computeFace, 2048, 128, (vertex, varying, F_IT, (int2*) F_ITa, offset, start, end));
    OPT_KERNEL_0(0, 3, computeFace, 2048, 128, (vertex, varying, F_IT, (int2*) F_ITa, offset, start, end));
    OPT_KERNEL_0(3, 0, computeFace, 2048, 128, (vertex, varying, F_IT, (int2*) F_ITa, offset, start, end));
    OPT_KERNEL_0(3, 3, computeFace, 2048, 128, (vertex, varying, F_IT, (int2*) F_ITa, offset, start, end));
#else
    //computeFace<3, 0><<<512,32>>>(vertex, varying, F_IT, F_ITa, offset, start, end);
    OPT_KERNEL(0, 0, computeFace, 512, 32, (vertex, varying, F_IT, F_ITa, offset, start, end));
    OPT_KERNEL(0, 3, computeFace, 512, 32, (vertex, varying, F_IT, F_ITa, offset, start, end));
    OPT_KERNEL(3, 0, computeFace, 512, 32, (vertex, varying, F_IT, F_ITa, offset, start, end));
    OPT_KERNEL(3, 3, computeFace, 512, 32, (vertex, varying, F_IT, F_ITa, offset, start, end));
#endif

    // fallback kernel (slow)
    computeFace<<<512, 32>>>(vertex, 3+numUserVertexElements, varying, numVaryingElements,
                             F_IT, F_ITa, offset, start, end);
}

void OsdCudaComputeEdge(float *vertex, float *varying,
                        int numUserVertexElements, int numVaryingElements,
                        int *E_IT, float *E_W, int offset, int start, int end)
{
 #if defined USE_BLOCK_OPTIM
    OPT_KERNEL_0(0, 0, computeEdge, 2048, 256, (vertex, varying, (int4*)E_IT, (float2*)E_W, offset, start, end));
    OPT_KERNEL_0(0, 3, computeEdge, 2048, 256, (vertex, varying, (int4*)E_IT, (float2*)E_W, offset, start, end));
    OPT_KERNEL_0(3, 0, computeEdge, 2048, 256, (vertex, varying, (int4*)E_IT, (float2*)E_W, offset, start, end));
    OPT_KERNEL_0(3, 3, computeEdge, 2048, 256, (vertex, varying, (int4*)E_IT, (float2*)E_W, offset, start, end));
#else
    //computeEdge<0, 3><<<512,32>>>(vertex, varying, E_IT, E_W, offset, start, end);
    OPT_KERNEL(0, 0, computeEdge, 512, 32, (vertex, varying, E_IT, E_W, offset, start, end));
    OPT_KERNEL(0, 3, computeEdge, 512, 32, (vertex, varying, E_IT, E_W, offset, start, end));
    OPT_KERNEL(3, 0, computeEdge, 512, 32, (vertex, varying, E_IT, E_W, offset, start, end));
    OPT_KERNEL(3, 3, computeEdge, 512, 32, (vertex, varying, E_IT, E_W, offset, start, end));
#endif

    computeEdge<<<512, 32>>>(vertex, 3+numUserVertexElements, varying, numVaryingElements,
                             E_IT, E_W, offset, start, end);
}

void OsdCudaComputeVertexA(float *vertex, float *varying,
                           int numUserVertexElements, int numVaryingElements,
                           int *V_ITa, float *V_W, int offset, int start, int end, int pass)
{
//    computeVertexA<0, 3><<<512,32>>>(vertex, varying, V_ITa, V_W, offset, start, end, pass);
    OPT_KERNEL(0, 0, computeVertexA, 512, 32, (vertex, varying, V_ITa, V_W, offset, start, end, pass));
    OPT_KERNEL(0, 3, computeVertexA, 512, 32, (vertex, varying, V_ITa, V_W, offset, start, end, pass));
    OPT_KERNEL(3, 0, computeVertexA, 512, 32, (vertex, varying, V_ITa, V_W, offset, start, end, pass));
    OPT_KERNEL(3, 3, computeVertexA, 512, 32, (vertex, varying, V_ITa, V_W, offset, start, end, pass));

    computeVertexA<<<512, 32>>>(vertex, 3+numUserVertexElements, varying, numVaryingElements,
                                V_ITa, V_W, offset, start, end, pass);
}

void OsdCudaComputeVertexB(float *vertex, float *varying,
                           int numUserVertexElements, int numVaryingElements,
                           int *V_ITa, int *V_IT, float *V_W, int offset, int start, int end)
{
#if defined USE_BLOCK_OPTIM
    OPT_KERNEL_0(0, 0, computeVertexB, 2048, 128, (vertex, varying, V_ITa, V_IT, V_W, offset, start, end));
    OPT_KERNEL_0(0, 3, computeVertexB, 2048, 128, (vertex, varying, V_ITa, V_IT, V_W, offset, start, end));
    OPT_KERNEL_0(3, 0, computeVertexB, 2048, 128, (vertex, varying, V_ITa, V_IT, V_W, offset, start, end));
    OPT_KERNEL_0(3, 3, computeVertexB, 2048, 128, (vertex, varying, V_ITa, V_IT, V_W, offset, start, end));

    computeVertexB<<<512, 128>>>(vertex, 3+numUserVertexElements, varying, numVaryingElements,
                                V_ITa, V_IT, V_W, offset, start, end);
#else
//    computeVertexB<0, 3><<<512,32>>>(vertex, varying, V_ITa, V_IT, V_W, offset, start, end);
    OPT_KERNEL(0, 0, computeVertexB, 512, 32, (vertex, varying, V_ITa, V_IT, V_W, offset, start, end));
    OPT_KERNEL(0, 3, computeVertexB, 512, 32, (vertex, varying, V_ITa, V_IT, V_W, offset, start, end));
    OPT_KERNEL(3, 0, computeVertexB, 512, 32, (vertex, varying, V_ITa, V_IT, V_W, offset, start, end));
    OPT_KERNEL(3, 3, computeVertexB, 512, 32, (vertex, varying, V_ITa, V_IT, V_W, offset, start, end));

    computeVertexB<<<512, 32>>>(vertex, 3+numUserVertexElements, varying, numVaryingElements,
                                V_ITa, V_IT, V_W, offset, start, end);
#endif
}

void OsdCudaComputeLoopVertexB(float *vertex, float *varying,
                               int numUserVertexElements, int numVaryingElements,
                               int *V_ITa, int *V_IT, float *V_W, int offset, int start, int end)
{
//    computeLoopVertexB<0, 3><<<512,32>>>(vertex, varying, V_ITa, V_IT, V_W, offset, start, end);
    OPT_KERNEL(0, 0, computeLoopVertexB, 512, 32, (vertex, varying, V_ITa, V_IT, V_W, offset, start, end));
    OPT_KERNEL(0, 3, computeLoopVertexB, 512, 32, (vertex, varying, V_ITa, V_IT, V_W, offset, start, end));
    OPT_KERNEL(3, 0, computeLoopVertexB, 512, 32, (vertex, varying, V_ITa, V_IT, V_W, offset, start, end));
    OPT_KERNEL(3, 3, computeLoopVertexB, 512, 32, (vertex, varying, V_ITa, V_IT, V_W, offset, start, end));

    computeLoopVertexB<<<512, 32>>>(vertex, 3+numUserVertexElements, varying, numVaryingElements,
                                    V_ITa, V_IT, V_W, offset, start, end);
}

void OsdCudaComputeBilinearEdge(float *vertex, float *varying,
                                int numUserVertexElements, int numVaryingElements,
                                int *E_IT, int offset, int start, int end)
{
    //computeBilinearEdge<0, 3><<<512,32>>>(vertex, varying, E_IT, offset, start, end);
    OPT_KERNEL(0, 0, computeBilinearEdge, 512, 32, (vertex, varying, E_IT, offset, start, end));
    OPT_KERNEL(0, 3, computeBilinearEdge, 512, 32, (vertex, varying, E_IT, offset, start, end));
    OPT_KERNEL(3, 0, computeBilinearEdge, 512, 32, (vertex, varying, E_IT, offset, start, end));
    OPT_KERNEL(3, 3, computeBilinearEdge, 512, 32, (vertex, varying, E_IT, offset, start, end));

    computeBilinearEdge<<<512, 32>>>(vertex, 3+numUserVertexElements, varying, numVaryingElements,
                                     E_IT, offset, start, end);
}

void OsdCudaComputeBilinearVertex(float *vertex, float *varying,
                                  int numUserVertexElements, int numVaryingElements,
                                  int *V_ITa, int offset, int start, int end)
{
//    computeBilinearVertex<0, 3><<<512,32>>>(vertex, varying, V_ITa, offset, start, end);
    OPT_KERNEL(0, 0, computeBilinearVertex, 512, 32, (vertex, varying, V_ITa, offset, start, end));
    OPT_KERNEL(0, 3, computeBilinearVertex, 512, 32, (vertex, varying, V_ITa, offset, start, end));
    OPT_KERNEL(3, 0, computeBilinearVertex, 512, 32, (vertex, varying, V_ITa, offset, start, end));
    OPT_KERNEL(3, 3, computeBilinearVertex, 512, 32, (vertex, varying, V_ITa, offset, start, end));

    computeBilinearVertex<<<512, 32>>>(vertex, 3+numUserVertexElements, varying, numVaryingElements,
                                       V_ITa, offset, start, end);
}

void OsdCudaEditVertexAdd(float *vertex, int numUserVertexElements,
                          int primVarOffset, int primVarWidth, int numVertices, int *editIndices, float *editValues)
{
    editVertexAdd<<<512, 32>>>(vertex, 3+numUserVertexElements, primVarOffset, primVarWidth,
                               numVertices, editIndices, editValues);
}

}
