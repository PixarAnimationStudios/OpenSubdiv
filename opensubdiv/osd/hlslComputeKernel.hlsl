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

interface IComputeKernel {
    void runKernel( uint3 ID );
};
IComputeKernel kernel;

cbuffer KernelCB : register( b0 ) {
    int indexOffset;    // index offset for the level
    int indexStart;     // start index for given batch
    int indexEnd;       // end index for given batch

    bool vertexPass;
    int F_IT_ofs;
    int F_ITa_ofs;
    int E_IT_ofs;
    int V_IT_ofs;
    int V_ITa_ofs;
    int E_W_ofs;
    int V_W_ofs;

    int editIndices_ofs;
    int editValues_ofs;
    int editPrimVarOffset;
    int editPrimVarWidth;
    int editNumVertices;
};

/*
 +-----+---------------------------------+-----
   n-1 |   Level n   |<batch range>|     |  n+1
 +-----+---------------------------------+-----
       ^             ^             ^
  indexOffset        |             |
                 indexStart     indexEnd
*/

RWBuffer<float> vertexBuffer  : register( u0 );
RWBuffer<float> varyingBuffer : register( u1 );
Buffer<int> _F_IT             : register( t2 );
Buffer<int> _F_ITa            : register( t3 );
Buffer<int> _E_IT             : register( t4 );
Buffer<int> _V_IT             : register( t5 );
Buffer<int> _V_ITa            : register( t6 );
Buffer<float> _E_W            : register( t7 );
Buffer<float> _V_W            : register( t8 );
Buffer<int> _editIndices      : register( t9 );
Buffer<float> _editValues     : register( t10 );

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

void clear(out Vertex v)
{
#if NUM_VERTEX_ELEMENTS > 0
    for(int i = 0; i < NUM_VERTEX_ELEMENTS; i++) {
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
    Vertex v;

#if NUM_VERTEX_ELEMENTS > 0
    for (int i = 0; i < NUM_VERTEX_ELEMENTS; i++) {
        v.vertexData[i] = vertexBuffer[index*NUM_VERTEX_ELEMENTS+i];
    }
#endif
#if NUM_VARYING_ELEMENTS > 0
    for (int i = 0; i < NUM_VARYING_ELEMENTS; i++) {
        v.varyingData[i] = varyingBuffer[index*NUM_VARYING_ELEMENTS+i];
    }
#endif
    return v;
}

void writeVertex(int index, Vertex v)
{
#if NUM_VERTEX_ELEMENTS > 0
    for (int i = 0; i < NUM_VERTEX_ELEMENTS; i++) {
        vertexBuffer[index*NUM_VERTEX_ELEMENTS+i] = v.vertexData[i];
    }
#endif
#if NUM_VARYING_ELEMENTS > 0
    for (int i = 0; i < NUM_VARYING_ELEMENTS; i++) {
        varyingBuffer[index*NUM_VARYING_ELEMENTS+i] = v.varyingData[i];
    }
#endif
}

void addWithWeight(inout Vertex v, Vertex src, float weight)
{
#if NUM_VERTEX_ELEMENTS > 0
    for (int i = 0; i < NUM_VERTEX_ELEMENTS; i++) {
        v.vertexData[i] += weight * src.vertexData[i];
    }
#endif
}

void addVaryingWithWeight(inout Vertex v, Vertex src, float weight)
{
#if NUM_VARYING_ELEMENTS > 0
    for (int i = 0; i < NUM_VARYING_ELEMENTS; i++) {
        v.varyingData[i] += weight * src.varyingData[i];
    }
#endif
}

//--------------------------------------------------------------------------------
// Face-vertices compute Kernel
class CatmarkComputeFace : IComputeKernel {
int placeholder;
void runKernel( uint3 ID )
{
    int i = int(ID.x) + indexStart;
    if (i >= indexEnd) return;

    int h = _F_ITa[F_ITa_ofs+2*i];
    int n = _F_ITa[F_ITa_ofs+2*i+1];

    float weight = 1.0/n;

    Vertex dst;
    clear(dst);
    for(int j=0; j<n; ++j){
        int index = _F_IT[F_IT_ofs+h+j];
        addWithWeight(dst, readVertex(index), weight);
        addVaryingWithWeight(dst, readVertex(index), weight);
    }
    writeVertex(i + indexOffset, dst);
}
};

// Edge-vertices compute Kernel
class CatmarkComputeEdge : IComputeKernel {
int placeholder;
void runKernel( uint3 ID )
{
    int i = int(ID.x) + indexStart;
    if (i >= indexEnd) return;

    Vertex dst;
    clear(dst);

    int eidx0 = _E_IT[E_IT_ofs+4*i+0];
    int eidx1 = _E_IT[E_IT_ofs+4*i+1];
    int eidx2 = _E_IT[E_IT_ofs+4*i+2];
    int eidx3 = _E_IT[E_IT_ofs+4*i+3];
    int4 eidx = int4(eidx0, eidx1, eidx2, eidx3);

    float vertWeight = _E_W[E_W_ofs+i*2+0];

    // Fully sharp edge : vertWeight = 0.5f;
    addWithWeight(dst, readVertex(eidx.x), vertWeight);
    addWithWeight(dst, readVertex(eidx.y), vertWeight);

    if(eidx.z != -1){
        float faceWeight = _E_W[E_W_ofs+i*2+1];

        addWithWeight(dst, readVertex(eidx.z), faceWeight);
        addWithWeight(dst, readVertex(eidx.w), faceWeight);
    }

    addVaryingWithWeight(dst, readVertex(eidx.x), 0.5f);
    addVaryingWithWeight(dst, readVertex(eidx.y), 0.5f);

    writeVertex(i + indexOffset, dst);
}
};

// Edge-vertices compute Kernel (bilinear scheme)
class BilinearComputeEdge : IComputeKernel {
int placeholder;
void runKernel( uint3 ID )
{
    int i = int(ID.x) + indexStart;
    if (i >= indexEnd) return;

    Vertex dst;
    clear(dst);

    int2 eidx = int2(_E_IT[E_IT_ofs+2*i+0],
                     _E_IT[E_IT_ofs+2*i+1]);

    addWithWeight(dst, readVertex(eidx.x), 0.5f);
    addWithWeight(dst, readVertex(eidx.y), 0.5f);

    addVaryingWithWeight(dst, readVertex(eidx.x), 0.5f);
    addVaryingWithWeight(dst, readVertex(eidx.y), 0.5f);

    writeVertex(i + indexOffset, dst);
}
};

// Vertex-vertices compute Kernel (bilinear scheme)
class BilinearComputeVertex : IComputeKernel {
int placeholder;
void runKernel( uint3 ID )
{
    int i = int(ID.x) + indexStart;
    if (i >= indexEnd) return;

    Vertex dst;
    clear(dst);

    int p = _V_ITa[V_ITa_ofs+i];

    addWithWeight(dst, readVertex(p), 1.0f);

    addVaryingWithWeight(dst, readVertex(p), 1.0f);

    writeVertex(i + indexOffset, dst);
}
};

// Vertex-vertices compute Kernels 'A' / k_Crease and k_Corner rules
class CatmarkComputeVertexA : IComputeKernel {
int placeholder;
void runKernel( uint3 ID )
{
    int i = int(ID.x) + indexStart;
    if (i >= indexEnd) return;

    int n     = _V_ITa[V_ITa_ofs+5*i+1];
    int p     = _V_ITa[V_ITa_ofs+5*i+2];
    int eidx0 = _V_ITa[V_ITa_ofs+5*i+3];
    int eidx1 = _V_ITa[V_ITa_ofs+5*i+4];

    float weight = vertexPass
        ? _V_W[V_W_ofs+i]
        : 1.0 - _V_W[V_W_ofs+i];

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

    writeVertex(i + indexOffset, dst);
}
};

// Vertex-vertices compute Kernels 'B' / k_Dart and k_Smooth rules
class CatmarkComputeVertexB : IComputeKernel {
int placeholder;
void runKernel( uint3 ID )
{
    int i = int(ID.x) + indexStart;
    if (i >= indexEnd) return;

    int h = _V_ITa[V_ITa_ofs+5*i];
    int n = _V_ITa[V_ITa_ofs+5*i+1];
    int p = _V_ITa[V_ITa_ofs+5*i+2];

    float weight = _V_W[V_W_ofs+i];
    float wp = 1.0/float(n*n);
    float wv = (n-2.0) * n * wp;

    Vertex dst;
    clear(dst);

    addWithWeight(dst, readVertex(p), weight * wv);

    for(int j = 0; j < n; ++j){
        addWithWeight(dst, readVertex(_V_IT[V_IT_ofs+h+j*2]), weight * wp);
        addWithWeight(dst, readVertex(_V_IT[V_IT_ofs+h+j*2+1]), weight * wp);
    }
    addVaryingWithWeight(dst, readVertex(p), 1);
    writeVertex(i + indexOffset, dst);
}
};

// Vertex-vertices compute Kernels 'B' / k_Dart and k_Smooth rules
class LoopComputeVertexB : IComputeKernel {
int placeholder;
void runKernel( uint3 ID )
{
    float PI = 3.14159265358979323846264;
    int i = int(ID.x) + indexStart;
    if (i >= indexEnd) return;

    int h = _V_ITa[V_ITa_ofs+5*i];
    int n = _V_ITa[V_ITa_ofs+5*i+1];
    int p = _V_ITa[V_ITa_ofs+5*i+2];

    float weight = _V_W[V_W_ofs+i];
    float wp = 1.0/n;
    float beta = 0.25 * cos(PI*2.0f*wp)+0.375f;
    beta = beta * beta;
    beta = (0.625f-beta)*wp;

    Vertex dst;
    clear(dst);

    addWithWeight(dst, readVertex(p), weight * (1.0-(beta*n)));

    for(int j = 0; j < n; ++j){
        addWithWeight(dst, readVertex(_V_IT[V_IT_ofs+h+j]), weight * beta);
    }
    addVaryingWithWeight(dst, readVertex(p), 1);
    writeVertex(i + indexOffset, dst);
}
};

class EditAdd : IComputeKernel {
int placeholder;
void runKernel( uint3 ID )
{
    int i = int(ID.x);
    if (i >= editNumVertices) return;

    int v = _editIndices[editIndices_ofs+i];
    Vertex dst = readVertex(v);

    // seemingly we can't iterate dynamically over vertexData[n]
    // due to mysterious glsl runtime limitation...?
    for (int j = 0; j < NUM_VERTEX_ELEMENTS; ++j) {
        float editValue = _editValues[editValues_ofs+min(j, editPrimVarWidth)];
        editValue *= float(j >= editPrimVarOffset);
        editValue *= float(j < (editPrimVarWidth + editPrimVarOffset));
        dst.vertexData[j] += editValue;
    }
    writeVertex(v, dst);
}
};

CatmarkComputeFace catmarkComputeFace;
CatmarkComputeEdge catmarkComputeEdge;
BilinearComputeEdge bilinearComputeEdge;
BilinearComputeVertex bilinearComputeVertex;
CatmarkComputeVertexA catmarkComputeVertexA;
CatmarkComputeVertexB catmarkComputeVertexB;
LoopComputeVertexB loopComputeVertexB;
EditAdd editAdd;

[numthreads(WORK_GROUP_SIZE, 1, 1)]
void cs_main( uint3 ID : SV_DispatchThreadID )
{
    // call kernel
    kernel.runKernel(ID);
}
