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
#version 400

subroutine void computeKernelType();
subroutine uniform computeKernelType computeKernel;

uniform isamplerBuffer _F0_IT;
uniform isamplerBuffer _F0_ITa;
uniform isamplerBuffer _E0_IT;
uniform isamplerBuffer _V0_IT;
uniform isamplerBuffer _V0_ITa;
uniform samplerBuffer _E0_S;
uniform samplerBuffer _V0_S;

uniform bool vertexPass;
uniform int indexOffset = 0;    // index offset for the level
uniform int indexStart = 0;      // start index for given batch

uniform int F_IT_ofs;
uniform int F_ITa_ofs;
uniform int E_IT_ofs;
uniform int V_IT_ofs;
uniform int V_ITa_ofs;
uniform int E_W_ofs;
uniform int V_W_ofs;

/*
 +-----+---------------------------------+-----
   n-1 |   Level n   |<batch range>|     |  n+1
 +-----+---------------------------------+-----
       ^             ^
  indexOffset        |
                 indexStart
*/

//--------------------------------------------------------------------------------

struct Vertex
{
    vec3 position;
    vec3 normal;
#if NUM_VARYING > 0
    float varyingData[NUM_VARYING];  // XXX: should use vec4 and packing
#endif
};

uniform samplerBuffer vertex;        // vec3[2]  (position, normal)
#if NUM_VARYING > 0
uniform samplerBuffer varyingData;   // float[NUM_VARYING]
#endif

out vec3 outPosition;
out vec3 outNormal;
#if NUM_VARYING > 0
out float outVaryingData[NUM_VARYING];  // output feedback (mapped as a subrange of vertices)
#endif
//out vec3 outVaryingData;      // output feedback (mapped as a subrange of vertices)

void clear(out Vertex v)
{
    v.position = vec3(0);
    v.normal = vec3(0);
#if NUM_VARYING > 0
    for(int i = 0; i < NUM_VARYING; i++){
        v.varyingData[i] = 0;
    }
#endif
}

Vertex readVertex(int index)
{
    // XXX: should be split into two parts for addWithWeight and addVaryingWithWeight
    Vertex v;

    // unpacking
    v.position = texelFetch(vertex, index*2).xyz;
    v.normal = texelFetch(vertex, index*2+1).xyz;
#if NUM_VARYING > 0
    int stride = NUM_VARYING;
    for(int i = 0; i < NUM_VARYING; i++){
        v.varyingData[i] = texelFetch(varyingData, index*stride+i).x;
    }
#endif
    return v;
}

void writeVertex(Vertex v)
{
    // packing
    outPosition = v.position;
    outNormal = normalize(v.normal);
#if NUM_VARYING > 0
    for(int i = 0; i < NUM_VARYING; i++){
        outVaryingData[i] = v.varyingData[i];
    }
#endif
}

void addWithWeight(inout Vertex v, Vertex src, float weight)
{
    v.position += weight * src.position;
    v.normal += weight * src.normal;
}

void addVaryingWithWeight(inout Vertex v, Vertex src, float weight)
{
#if NUM_VARYING > 0
    for(int i = 0; i < NUM_VARYING; i++){
        v.varyingData[i] += weight * src.varyingData[i];
    }
#endif
}

//--------------------------------------------------------------------------------
// Face-vertices compute Kernel
subroutine(computeKernelType)
void catmarkComputeFace()
{
    int i = gl_VertexID + indexStart;
    int h = texelFetch(_F0_ITa, F_ITa_ofs+2*i).x;
    int n = texelFetch(_F0_ITa, F_ITa_ofs+2*i+1).x;

    float weight = 1.0/n;

    Vertex dst;
    clear(dst);
    for(int j=0; j<n; ++j){
        int index = texelFetch(_F0_IT, F_IT_ofs+h+j).x;
        addWithWeight(dst, readVertex(index), weight);
        addVaryingWithWeight(dst, readVertex(index), weight);
    }
    writeVertex(dst);
}

// Edge-vertices compute Kernel
subroutine(computeKernelType)
void catmarkComputeEdge()
{
    int i = gl_VertexID + indexStart;

    Vertex dst;
    clear(dst);

#ifdef OPT_E0_IT_VEC4
    ivec4 eidx = texelFetch(_E0_IT, E_IT_ofs/4+i);
#else
    int eidx0 = texelFetch(_E0_IT, E_IT_ofs+4*i+0).x;
    int eidx1 = texelFetch(_E0_IT, E_IT_ofs+4*i+1).x;
    int eidx2 = texelFetch(_E0_IT, E_IT_ofs+4*i+2).x;
    int eidx3 = texelFetch(_E0_IT, E_IT_ofs+4*i+3).x;
    ivec4 eidx = ivec4(eidx0, eidx1, eidx2, eidx3);
#endif

#ifdef OPT_E0_S_VEC2
    vec2 weight = texelFetch(_E0_S, E_W_ofs/2+i).xy;
    float vertWeight = weight.x;
#else
    float vertWeight = texelFetch(_E0_S, E_W_ofs+i*2+0).x;
#endif

    // Fully sharp edge : vertWeight = 0.5f;
    addWithWeight(dst, readVertex(eidx.x), vertWeight);
    addWithWeight(dst, readVertex(eidx.y), vertWeight);

    if(eidx.z != -1){
#ifdef OPT_E0_S_VEC2
        float faceWeight = weight.y;
#else
        float faceWeight = texelFetch(_E0_S, E_W_ofs+i*2+1).x;
#endif

        addWithWeight(dst, readVertex(eidx.z), faceWeight);
        addWithWeight(dst, readVertex(eidx.w), faceWeight);
    }

    addVaryingWithWeight(dst, readVertex(eidx.x), 0.5f);
    addVaryingWithWeight(dst, readVertex(eidx.y), 0.5f);

    writeVertex(dst);
}

// Edge-vertices compute Kernel (bilinear scheme)
subroutine(computeKernelType)
void bilinearComputeEdge()
{
    int i = gl_VertexID + indexStart;

    Vertex dst;
    clear(dst);

#ifdef OPT_E0_IT_VEC4
    ivec2 eidx = texelFetch(_E0_IT, E_IT_ofs/2+i).xy;
#else
    ivec2 eidx = ivec2(texelFetch(_E0_IT, E_IT_ofs+2*i+0).x,
                       texelFetch(_E0_IT, E_IT_ofs+2*i+1).x);
#endif

    addWithWeight(dst, readVertex(eidx.x), 0.5f);
    addWithWeight(dst, readVertex(eidx.y), 0.5f);

    addVaryingWithWeight(dst, readVertex(eidx.x), 0.5f);
    addVaryingWithWeight(dst, readVertex(eidx.y), 0.5f);

    writeVertex(dst);
}

// Vertex-vertices compute Kernel (bilinear scheme)
subroutine(computeKernelType)
void bilinearComputeVertex()
{
    int i = gl_VertexID + indexStart;

    Vertex dst;
    clear(dst);

    int p = texelFetch(_V0_ITa, V_ITa_ofs+i).x;

    addWithWeight(dst, readVertex(p), 1.0f);

    addVaryingWithWeight(dst, readVertex(p), 1.0f);

    writeVertex(dst);
}

// Vertex-vertices compute Kernels 'A' / k_Crease and k_Corner rules
subroutine(computeKernelType)
void catmarkComputeVertexA()
{
    int i = gl_VertexID + indexStart;

    int n     = texelFetch(_V0_ITa, V_ITa_ofs+5*i+1).x;
    int p     = texelFetch(_V0_ITa, V_ITa_ofs+5*i+2).x;
    int eidx0 = texelFetch(_V0_ITa, V_ITa_ofs+5*i+3).x;
    int eidx1 = texelFetch(_V0_ITa, V_ITa_ofs+5*i+4).x;

    float weight = vertexPass
        ? texelFetch(_V0_S, V_W_ofs+i).x
        : 1.0 - texelFetch(_V0_S, V_W_ofs+i).x;

    // In the case of fractional weight, the weight must be inverted since
    // the value is shared with the k_Smooth kernel (statistically the
    // k_Smooth kernel runs much more often than this one)
    if (weight>0.0 && weight<1.0 && n > 0)
        weight=1.0-weight;

    Vertex dst;
    if(! vertexPass)
        clear(dst);
    else
        dst = readVertex(i + indexOffset);

    if (eidx0==-1 || (vertexPass==false && (n==-1)) ) {
        addWithWeight(dst, readVertex(p), weight);
    } else {
        addWithWeight(dst, readVertex(p), weight * 0.75f);
        addWithWeight(dst, readVertex(eidx0), weight * 0.125f);
        addWithWeight(dst, readVertex(eidx1), weight * 0.125f);
    }
    if(! vertexPass)
        addVaryingWithWeight(dst, readVertex(p), 1);

    writeVertex(dst);
}

// Vertex-vertices compute Kernels 'B' / k_Dart and k_Smooth rules
subroutine(computeKernelType)
void catmarkComputeVertexB()
{
    int i = gl_VertexID + indexStart;

    int h = texelFetch(_V0_ITa, V_ITa_ofs+5*i).x;
#ifdef OPT_CATMARK_V_IT_VEC2
    int h2 = h/2;
#endif
    int n = texelFetch(_V0_ITa, V_ITa_ofs+5*i+1).x;
    int p = texelFetch(_V0_ITa, V_ITa_ofs+5*i+2).x;

    float weight = texelFetch(_V0_S, V_W_ofs+i).x;
    float wp = 1.0/float(n*n);
    float wv = (n-2.0) * n * wp;

    Vertex dst;
    clear(dst);

    addWithWeight(dst, readVertex(p), weight * wv);

    for(int j = 0; j < n; ++j){
#ifdef OPT_CATMARK_V_IT_VEC2
        ivec2 v0it = texelFetch(_V0_IT, V_IT_ofs/2+h2+j).xy;
        addWithWeight(dst, readVertex(v0it.x), weight * wp);
        addWithWeight(dst, readVertex(v0it.y), weight * wp);
#else
        addWithWeight(dst, readVertex(texelFetch(_V0_IT, V_IT_ofs+h+j*2).x), weight * wp);
        addWithWeight(dst, readVertex(texelFetch(_V0_IT, V_IT_ofs+h+j*2+1).x), weight * wp);
#endif
    }
    addVaryingWithWeight(dst, readVertex(p), 1);
    writeVertex(dst);
}

// Vertex-vertices compute Kernels 'B' / k_Dart and k_Smooth rules
subroutine(computeKernelType)
void loopComputeVertexB()
{
    float PI = 3.14159265358979323846264;
    int i = gl_VertexID + indexStart;

    int h = texelFetch(_V0_ITa, V_ITa_ofs+5*i).x;
    int n = texelFetch(_V0_ITa, V_ITa_ofs+5*i+1).x;
    int p = texelFetch(_V0_ITa, V_ITa_ofs+5*i+2).x;

    float weight = texelFetch(_V0_S, V_W_ofs+i).x;
    float wp = 1.0/n;
    float beta = 0.25 * cos(PI*2.0f*wp)+0.375f;
    beta = beta * beta;
    beta = (0.625f-beta)*wp;

    Vertex dst;
    clear(dst);

    addWithWeight(dst, readVertex(p), weight * (1.0-(beta*n)));

    for(int j = 0; j < n; ++j){
        addWithWeight(dst, readVertex(texelFetch(_V0_IT, V_IT_ofs+h+j).x), weight * beta);
    }
    addVaryingWithWeight(dst, readVertex(p), 1);
    writeVertex(dst);
}

void main()
{
    // call subroutine
    computeKernel();
}
