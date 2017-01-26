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

//------------------------------------------------------------------------------


layout(local_size_x=WORK_GROUP_SIZE, local_size_y=1, local_size_z=1) in;
layout(std430) buffer;

// source and destination buffers

uniform int srcOffset = 0;
uniform int dstOffset = 0;
layout(binding=0) buffer src_buffer      { float    srcVertexBuffer[]; };
layout(binding=1) buffer dst_buffer      { float    dstVertexBuffer[]; };

// derivative buffers (if needed)

#if defined(OPENSUBDIV_GLSL_COMPUTE_USE_1ST_DERIVATIVES)
uniform ivec3 duDesc;
uniform ivec3 dvDesc;
layout(binding=2) buffer du_buffer   { float duBuffer[]; };
layout(binding=3) buffer dv_buffer   { float dvBuffer[]; };
#endif

#if defined(OPENSUBDIV_GLSL_COMPUTE_USE_2ND_DERIVATIVES)
uniform ivec3 duuDesc;
uniform ivec3 duvDesc;
uniform ivec3 dvvDesc;
layout(binding=10) buffer duu_buffer   { float duuBuffer[]; };
layout(binding=11) buffer duv_buffer   { float duvBuffer[]; };
layout(binding=12) buffer dvv_buffer   { float dvvBuffer[]; };
#endif

// stencil buffers

#if defined(OPENSUBDIV_GLSL_COMPUTE_KERNEL_EVAL_STENCILS)

uniform int batchStart = 0;
uniform int batchEnd = 0;
layout(binding=4) buffer stencilSizes    { int      _sizes[];   };
layout(binding=5) buffer stencilOffsets  { int      _offsets[]; };
layout(binding=6) buffer stencilIndices  { int      _indices[]; };
layout(binding=7) buffer stencilWeights  { float    _weights[]; };

#if defined(OPENSUBDIV_GLSL_COMPUTE_USE_1ST_DERIVATIVES)
layout(binding=8) buffer stencilDuWeights { float  _duWeights[]; };
layout(binding=9) buffer stencilDvWeights { float  _dvWeights[]; };
#endif

#if defined(OPENSUBDIV_GLSL_COMPUTE_USE_2ND_DERIVATIVES)
layout(binding=13) buffer stencilDuuWeights { float  _duuWeights[]; };
layout(binding=14) buffer stencilDuvWeights { float  _duvWeights[]; };
layout(binding=15) buffer stencilDvvWeights { float  _dvvWeights[]; };
#endif

#endif

// patch buffers

#if defined(OPENSUBDIV_GLSL_COMPUTE_KERNEL_EVAL_PATCHES)

struct PatchCoord {
   int arrayIndex;
   int patchIndex;
   int vertIndex;
   float s;
   float t;
};
struct PatchParam {
    uint field0;
    uint field1;
    float sharpness;
};
uniform ivec4 patchArray[2];
layout(binding=4) buffer patchCoord_buffer { PatchCoord patchCoords[]; };
layout(binding=5) buffer patchIndex_buffer { int patchIndexBuffer[]; };
layout(binding=6) buffer patchParam_buffer { PatchParam patchParamBuffer[]; };

#endif

//------------------------------------------------------------------------------

struct Vertex {
    float vertexData[LENGTH];
};

void clear(out Vertex v) {
    for (int i = 0; i < LENGTH; ++i) {
        v.vertexData[i] = 0;
    }
}

Vertex readVertex(int index) {
    Vertex v;
    int vertexIndex = srcOffset + index * SRC_STRIDE;
    for (int i = 0; i < LENGTH; ++i) {
        v.vertexData[i] = srcVertexBuffer[vertexIndex + i];
    }
    return v;
}

void writeVertex(int index, Vertex v) {
    int vertexIndex = dstOffset + index * DST_STRIDE;
    for (int i = 0; i < LENGTH; ++i) {
        dstVertexBuffer[vertexIndex + i] = v.vertexData[i];
    }
}

void addWithWeight(inout Vertex v, const Vertex src, float weight) {
    for (int i = 0; i < LENGTH; ++i) {
        v.vertexData[i] += weight * src.vertexData[i];
    }
}

#if defined(OPENSUBDIV_GLSL_COMPUTE_USE_1ST_DERIVATIVES)
void writeDu(int index, Vertex du) {
    int duIndex = duDesc.x + index * duDesc.z;
    for (int i = 0; i < LENGTH; ++i) {
        duBuffer[duIndex + i] = du.vertexData[i];
    }
}

void writeDv(int index, Vertex dv) {
    int dvIndex = dvDesc.x + index * dvDesc.z;
    for (int i = 0; i < LENGTH; ++i) {
        dvBuffer[dvIndex + i] = dv.vertexData[i];
    }
}
#endif

#if defined(OPENSUBDIV_GLSL_COMPUTE_USE_2ND_DERIVATIVES)
void writeDuu(int index, Vertex duu) {
    int duuIndex = duuDesc.x + index * duuDesc.z;
    for (int i = 0; i < LENGTH; ++i) {
        duuBuffer[duuIndex + i] = duu.vertexData[i];
    }
}

void writeDuv(int index, Vertex duv) {
    int duvIndex = duvDesc.x + index * duvDesc.z;
    for (int i = 0; i < LENGTH; ++i) {
        duvBuffer[duvIndex + i] = duv.vertexData[i];
    }
}

void writeDvv(int index, Vertex dvv) {
    int dvvIndex = dvvDesc.x + index * dvvDesc.z;
    for (int i = 0; i < LENGTH; ++i) {
        dvvBuffer[dvvIndex + i] = dvv.vertexData[i];
    }
}
#endif

//------------------------------------------------------------------------------
#if defined(OPENSUBDIV_GLSL_COMPUTE_KERNEL_EVAL_STENCILS)

void main() {

    int current = int(gl_GlobalInvocationID.x) + batchStart;

    if (current>=batchEnd) {
        return;
    }

    Vertex dst;
    clear(dst);

    int offset = _offsets[current],
        size   = _sizes[current];

    for (int stencil = 0; stencil < size; ++stencil) {
        int vindex = offset + stencil;
        addWithWeight(
            dst, readVertex(_indices[vindex]), _weights[vindex]);
    }

    writeVertex(current, dst);

#if defined(OPENSUBDIV_GLSL_COMPUTE_USE_1ST_DERIVATIVES)
    Vertex du, dv;
    clear(du);
    clear(dv);
    for (int i=0; i<size; ++i) {
        // expects the compiler optimizes readVertex out here.
        Vertex src = readVertex(_indices[offset+i]);
        addWithWeight(du, src, _duWeights[offset+i]);
        addWithWeight(dv, src, _dvWeights[offset+i]);
    }

    if (duDesc.y > 0) { // length
        writeDu(current, du);
    }
    if (dvDesc.y > 0) {
        writeDv(current, dv);
    }
#endif
#if defined(OPENSUBDIV_GLSL_COMPUTE_USE_2ND_DERIVATIVES)
    Vertex duu, duv, dvv;
    clear(duu);
    clear(duv);
    clear(dvv);
    for (int i=0; i<size; ++i) {
        // expects the compiler optimizes readVertex out here.
        Vertex src = readVertex(_indices[offset+i]);
        addWithWeight(duu, src, _duuWeights[offset+i]);
        addWithWeight(duv, src, _duvWeights[offset+i]);
        addWithWeight(dvv, src, _dvvWeights[offset+i]);
    }

    if (duuDesc.y > 0) { // length
        writeDuu(current, duu);
    }
    if (duvDesc.y > 0) {
        writeDuv(current, duv);
    }
    if (dvvDesc.y > 0) {
        writeDvv(current, dvv);
    }
#endif
}

#endif

//------------------------------------------------------------------------------
#if defined(OPENSUBDIV_GLSL_COMPUTE_KERNEL_EVAL_PATCHES)

// PERFORMANCE: stride could be constant, but not as significant as length

//struct PatchArray {
//    int patchType;
//    int numPatches;
//    int indexBase;        // an offset within the index buffer
//    int primitiveIdBase;  // an offset within the patch param buffer
//};
// # of patcharrays is 1 or 2.

uint getDepth(uint patchBits) {
    return (patchBits & 0xf);
}

float getParamFraction(uint patchBits) {
    uint nonQuadRoot = (patchBits >> 4) & 0x1;
    uint depth = getDepth(patchBits);
    if (nonQuadRoot == 1) {
        return 1.0f / float( 1 << (depth-1) );
    } else {
        return 1.0f / float( 1 << depth );
    }
}

vec2 normalizePatchCoord(uint patchBits, vec2 uv) {
    float frac = getParamFraction(patchBits);

    uint iu = (patchBits >> 22) & 0x3ff;
    uint iv = (patchBits >> 12) & 0x3ff;

    // top left corner
    float pu = float(iu*frac);
    float pv = float(iv*frac);

    // normalize u,v coordinates
    return vec2((uv.x - pu) / frac, (uv.y - pv) / frac);
}

bool isRegular(uint patchBits) {
    return (((patchBits >> 5) & 0x1u) != 0);
}

int getNumControlVertices(int patchType) {
    return (patchType == 3) ? 4 :
           (patchType == 6) ? 16 :
           (patchType == 9) ? 20 : 0;
}

void main() {

    int current = int(gl_GlobalInvocationID.x);

    PatchCoord coord = patchCoords[current];
    int patchIndex = coord.patchIndex;

    ivec4 array = patchArray[coord.arrayIndex];

    uint patchBits = patchParamBuffer[patchIndex].field1;
    int patchType = isRegular(patchBits) ? 6 : array.x;

    vec2 uv = normalizePatchCoord(patchBits, vec2(coord.s, coord.t));
    float dScale = float(1 << getDepth(patchBits));
    int boundary = int((patchBits >> 8) & 0xfU);

    float wP[20], wDs[20], wDt[20], wDss[20], wDst[20], wDtt[20];

    int numControlVertices = 0;
    if (patchType == 3) {
        float wP4[4], wDs4[4], wDt4[4], wDss4[4], wDst4[4], wDtt4[4];
        OsdGetBilinearPatchWeights(uv.s, uv.t, dScale, wP4, wDs4, wDt4, wDss4, wDst4, wDtt4);
        numControlVertices = 4;
        for (int i=0; i<numControlVertices; ++i) {
            wP[i] = wP4[i];
            wDs[i] = wDs4[i];
            wDt[i] = wDt4[i];
            wDss[i] = wDss4[i];
            wDst[i] = wDst4[i];
            wDtt[i] = wDtt4[i];
        }
    } else if (patchType == 6) {
        float wP16[16], wDs16[16], wDt16[16], wDss16[16], wDst16[16], wDtt16[16];
        OsdGetBSplinePatchWeights(uv.s, uv.t, dScale, boundary, wP16, wDs16, wDt16, wDss16, wDst16, wDtt16);
        numControlVertices = 16;
        for (int i=0; i<numControlVertices; ++i) {
            wP[i] = wP16[i];
            wDs[i] = wDs16[i];
            wDt[i] = wDt16[i];
            wDss[i] = wDss16[i];
            wDst[i] = wDst16[i];
            wDtt[i] = wDtt16[i];
        }
    } else if (patchType == 9) {
        OsdGetGregoryPatchWeights(uv.s, uv.t, dScale, wP, wDs, wDt, wDss, wDst, wDtt);
        numControlVertices = 20;
    }

    Vertex dst, du, dv, duu, duv, dvv;
    clear(dst);
    clear(du);
    clear(dv);
    clear(duu);
    clear(duv);
    clear(dvv);

    int indexStride = getNumControlVertices(array.x);
    int indexBase = array.z + indexStride * (patchIndex - array.w);

    for (int cv = 0; cv < numControlVertices; ++cv) {
        int index = patchIndexBuffer[indexBase + cv];
        addWithWeight(dst, readVertex(index), wP[cv]);
        addWithWeight(du, readVertex(index), wDs[cv]);
        addWithWeight(dv, readVertex(index), wDt[cv]);
        addWithWeight(duu, readVertex(index), wDss[cv]);
        addWithWeight(duv, readVertex(index), wDst[cv]);
        addWithWeight(dvv, readVertex(index), wDtt[cv]);
    }
    writeVertex(current, dst);

#if defined(OPENSUBDIV_GLSL_COMPUTE_USE_1ST_DERIVATIVES)
    if (duDesc.y > 0) { // length
        writeDu(current, du);
    }
    if (dvDesc.y > 0) {
        writeDv(current, dv);
    }
#endif
#if defined(OPENSUBDIV_GLSL_COMPUTE_USE_2ND_DERIVATIVES)
    if (duuDesc.y > 0) { // length
        writeDuu(current, duu);
    }
    if (duvDesc.y > 0) { // length
        writeDuv(current, duv);
    }
    if (dvvDesc.y > 0) {
        writeDvv(current, dvv);
    }
#endif
}

#endif
