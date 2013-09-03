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

#version 420

subroutine void computeKernelType();
subroutine uniform computeKernelType computeKernel;

uniform isamplerBuffer _F0_IT;
uniform isamplerBuffer _F0_ITa;
uniform isamplerBuffer _E0_IT;
uniform isamplerBuffer _V0_IT;
uniform isamplerBuffer _V0_ITa;
uniform samplerBuffer _E0_S;
uniform samplerBuffer _V0_S;
uniform isamplerBuffer _editIndices;
uniform samplerBuffer _editValues;
layout(size1x32) uniform imageBuffer _vertexBufferImage;

uniform int vertexOffset = 0;   // vertex index offset for the batch
uniform int tableOffset = 0;    // offset of subdivision table
uniform int indexStart = 0;     // start index relative to tableOffset
uniform bool vertexPass;

/*
 +-----+---------------------------------+-----
   n-1 |   Level n   |<batch range>|     |  n+1
 +-----+---------------------------------+-----
       ^             ^
  vertexOffset       |
                 indexStart
*/

//--------------------------------------------------------------------------------

struct Vertex
{
#if NUM_VERTEX_ELEMENTS > 0
    float vertexData[NUM_VERTEX_ELEMENTS];
#endif
#if NUM_VARYING_ELEMENTS > 0
    float varyingData[NUM_VARYING_ELEMENTS];
#endif
};

#if NUM_VERTEX_ELEMENTS > 0
uniform samplerBuffer vertexData;    // float[NUM_VERTEX_ELEMENTS]
#endif
#if NUM_VARYING_ELEMENTS > 0
uniform samplerBuffer varyingData;   // float[NUM_VARYING_ELEMENTS]
#endif

// output feedback (mapped as a subrange of vertices)
#if NUM_VERTEX_ELEMENTS > 0
out float outVertexData[NUM_VERTEX_ELEMENTS];
#endif
#if NUM_VARYING_ELEMENTS > 0
out float outVaryingData[NUM_VARYING_ELEMENTS];
#endif

void clear(out Vertex v)
{
#if NUM_VERTEX_ELEMENTS > 0
    for (int i = 0; i < NUM_VERTEX_ELEMENTS; i++) {
        v.vertexData[i] = 0;
    }
#endif
#if NUM_VARYING_ELEMENTS > 0
    for(int i = 0; i < NUM_VARYING_ELEMENTS; i++){
        v.varyingData[i] = 0;
    }
#endif
}

Vertex readVertex(int index)
{
    // XXX: should be split into two parts for addWithWeight and addVaryingWithWeight
    Vertex v;

    // unpacking
#if NUM_VERTEX_ELEMENTS > 0
    for(int i = 0; i < NUM_VERTEX_ELEMENTS; i++) {
        v.vertexData[i] = texelFetch(vertexData, index*NUM_VERTEX_ELEMENTS+i).x;
    }
#endif
#if NUM_VARYING_ELEMENTS > 0
    for(int i = 0; i < NUM_VARYING_ELEMENTS; i++){
        v.varyingData[i] = texelFetch(varyingData, index*NUM_VARYING_ELEMENTS+i).x;
    }
#endif
    return v;
}

void writeVertex(Vertex v)
{
    // packing
#if NUM_VERTEX_ELEMENTS > 0
    for(int i = 0; i < NUM_VERTEX_ELEMENTS; i++) {
        outVertexData[i] = v.vertexData[i];
    }
#endif
#if NUM_VARYING_ELEMENTS > 0
    for(int i = 0; i < NUM_VARYING_ELEMENTS; i++){
        outVaryingData[i] = v.varyingData[i];
    }
#endif
}

void writeVertexByImageStore(Vertex v, int index)
{
#if NUM_VERTEX_ELEMENTS > 0
    int p = index * NUM_VERTEX_ELEMENTS;
    for(int i = 0; i < NUM_VERTEX_ELEMENTS; i++) {
        imageStore(_vertexBufferImage, p+i, vec4(v.vertexData[i], 0, 0, 0));
    }
#endif
}

void addWithWeight(inout Vertex v, Vertex src, float weight)
{
#if NUM_VERTEX_ELEMENTS > 0
    for(int i = 0; i < NUM_VERTEX_ELEMENTS; i++) {
        v.vertexData[i] += weight * src.vertexData[i];
    }
#endif
}

void addVaryingWithWeight(inout Vertex v, Vertex src, float weight)
{
#if NUM_VARYING_ELEMENTS > 0
    for(int i = 0; i < NUM_VARYING_ELEMENTS; i++){
        v.varyingData[i] += weight * src.varyingData[i];
    }
#endif
}

//--------------------------------------------------------------------------------
// Face-vertices compute Kernel
subroutine(computeKernelType)
void catmarkComputeFace()
{
    int i = gl_VertexID + indexStart + tableOffset;
    int h = texelFetch(_F0_ITa, 2*i).x;
    int n = texelFetch(_F0_ITa, 2*i+1).x;

    float weight = 1.0/n;

    Vertex dst;
    clear(dst);
    for(int j=0; j<n; ++j){
        int index = texelFetch(_F0_IT, h+j).x;
        addWithWeight(dst, readVertex(index), weight);
        addVaryingWithWeight(dst, readVertex(index), weight);
    }
    writeVertex(dst);
}

// Edge-vertices compute Kernel
subroutine(computeKernelType)
void catmarkComputeEdge()
{
    int i = gl_VertexID + indexStart + tableOffset;

    Vertex dst;
    clear(dst);

#ifdef OPT_E0_IT_VEC4
    ivec4 eidx = texelFetch(_E0_IT, i);
#else
    int eidx0 = texelFetch(_E0_IT, 4*i+0).x;
    int eidx1 = texelFetch(_E0_IT, 4*i+1).x;
    int eidx2 = texelFetch(_E0_IT, 4*i+2).x;
    int eidx3 = texelFetch(_E0_IT, 4*i+3).x;
    ivec4 eidx = ivec4(eidx0, eidx1, eidx2, eidx3);
#endif

#ifdef OPT_E0_S_VEC2
    vec2 weight = texelFetch(_E0_S, i).xy;
    float vertWeight = weight.x;
#else
    float vertWeight = texelFetch(_E0_S, i*2+0).x;
#endif

    // Fully sharp edge : vertWeight = 0.5f;
    addWithWeight(dst, readVertex(eidx.x), vertWeight);
    addWithWeight(dst, readVertex(eidx.y), vertWeight);

    if(eidx.z != -1){
#ifdef OPT_E0_S_VEC2
        float faceWeight = weight.y;
#else
        float faceWeight = texelFetch(_E0_S, i*2+1).x;
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
    int i = gl_VertexID + indexStart + tableOffset;

    Vertex dst;
    clear(dst);

#ifdef OPT_E0_IT_VEC4
    ivec2 eidx = texelFetch(_E0_IT, i).xy;
#else
    ivec2 eidx = ivec2(texelFetch(_E0_IT, 2*i+0).x,
                       texelFetch(_E0_IT, 2*i+1).x);
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
    int i = gl_VertexID + indexStart + tableOffset;

    Vertex dst;
    clear(dst);

    int p = texelFetch(_V0_ITa, i).x;

    addWithWeight(dst, readVertex(p), 1.0f);

    addVaryingWithWeight(dst, readVertex(p), 1.0f);

    writeVertex(dst);
}

// Vertex-vertices compute Kernels 'A' / k_Crease and k_Corner rules
subroutine(computeKernelType)
void catmarkComputeVertexA()
{
    int i = gl_VertexID + indexStart + tableOffset;
    int vid = gl_VertexID + indexStart + vertexOffset;

    int n     = texelFetch(_V0_ITa, 5*i+1).x;
    int p     = texelFetch(_V0_ITa, 5*i+2).x;
    int eidx0 = texelFetch(_V0_ITa, 5*i+3).x;
    int eidx1 = texelFetch(_V0_ITa, 5*i+4).x;

    float weight = vertexPass
        ? texelFetch(_V0_S, i).x
        : 1.0 - texelFetch(_V0_S, i).x;

    // In the case of fractional weight, the weight must be inverted since
    // the value is shared with the k_Smooth kernel (statistically the
    // k_Smooth kernel runs much more often than this one)
    if (weight>0.0 && weight<1.0 && n > 0)
        weight=1.0-weight;

    Vertex dst;
    if(! vertexPass)
        clear(dst);
    else
        dst = readVertex(vid);

    if (eidx0==-1 || (vertexPass==false && (n==-1)) ) {
        addWithWeight(dst, readVertex(p), weight);
    } else {
        addWithWeight(dst, readVertex(p), weight * 0.75f);
        addWithWeight(dst, readVertex(eidx0), weight * 0.125f);
        addWithWeight(dst, readVertex(eidx1), weight * 0.125f);
    }
    if(! vertexPass)
        addVaryingWithWeight(dst, readVertex(p), 1.0f);

    writeVertex(dst);
}

// Vertex-vertices compute Kernels 'B' / k_Dart and k_Smooth rules
subroutine(computeKernelType)
void catmarkComputeVertexB()
{
    int i = gl_VertexID + indexStart + tableOffset;

    int h = texelFetch(_V0_ITa, 5*i).x;
#ifdef OPT_CATMARK_V_IT_VEC2
    int h2 = h/2;
#endif
    int n = texelFetch(_V0_ITa, 5*i+1).x;
    int p = texelFetch(_V0_ITa, 5*i+2).x;

    float weight = texelFetch(_V0_S, i).x;
    float wp = 1.0/float(n*n);
    float wv = (n-2.0) * n * wp;

    Vertex dst;
    clear(dst);

    addWithWeight(dst, readVertex(p), weight * wv);

    for(int j = 0; j < n; ++j){
#ifdef OPT_CATMARK_V_IT_VEC2
        ivec2 v0it = texelFetch(_V0_IT, h2+j).xy;
        addWithWeight(dst, readVertex(v0it.x), weight * wp);
        addWithWeight(dst, readVertex(v0it.y), weight * wp);
#else
        addWithWeight(dst, readVertex(texelFetch(_V0_IT, h+j*2).x), weight * wp);
        addWithWeight(dst, readVertex(texelFetch(_V0_IT, h+j*2+1).x), weight * wp);
#endif
    }
    addVaryingWithWeight(dst, readVertex(p), 1.0f);
    writeVertex(dst);
}

// Vertex-vertices compute Kernels 'B' / k_Dart and k_Smooth rules
subroutine(computeKernelType)
void loopComputeVertexB()
{
    float PI = 3.14159265358979323846264;
    int i = gl_VertexID + indexStart + tableOffset;

    int h = texelFetch(_V0_ITa, 5*i).x;
    int n = texelFetch(_V0_ITa, 5*i+1).x;
    int p = texelFetch(_V0_ITa, 5*i+2).x;

    float weight = texelFetch(_V0_S, i).x;
    float wp = 1.0/n;
    float beta = 0.25 * cos(PI*2.0f*wp)+0.375f;
    beta = beta * beta;
    beta = (0.625f-beta)*wp;

    Vertex dst;
    clear(dst);

    addWithWeight(dst, readVertex(p), weight * (1.0-(beta*n)));

    for(int j = 0; j < n; ++j){
        addWithWeight(dst, readVertex(texelFetch(_V0_IT, h+j).x), weight * beta);
    }
    addVaryingWithWeight(dst, readVertex(p), 1.0f);
    writeVertex(dst);
}

// vertex edit kernel
uniform int editPrimVarOffset;
uniform int editPrimVarWidth;

subroutine(computeKernelType)
void editAdd()
{
    int i = gl_VertexID + indexStart + tableOffset;

    int v = texelFetch(_editIndices, i).x;
    Vertex dst = readVertex(v + vertexOffset);

    // this is tricky. _editValues array contains editPrimVarWidth count of values.
    // i.e. if the vertex edit is just for pos Y, editPrimVarOffset = 1 and
    // editPrimVarWidth = 1, then _editValues will be an one element array.
    // below loops iterate over every elements regardless editing values to be applied or not,
    // so we need to make out-of-range edits ineffective.

#if NUM_VERTEX_ELEMENTS > 0
    for (int j = 0; j < NUM_VERTEX_ELEMENTS; ++j) {
        int index = min(j-editPrimVarOffset, editPrimVarWidth-1);
        float editValue = texelFetch(_editValues, i*editPrimVarOffset + index).x;
        editValue *= float(j >= editPrimVarOffset);
        editValue *= float(j < (editPrimVarWidth + editPrimVarOffset));
        dst.vertexData[j] += editValue;
    }
#endif
    writeVertexByImageStore(dst, v + vertexOffset);
}


void main()
{
    // call subroutine
    computeKernel();
}
