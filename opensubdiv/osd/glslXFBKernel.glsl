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

uniform samplerBuffer vertexBuffer;
uniform int srcOffset = 0;
out float outVertexBuffer[LENGTH];

//------------------------------------------------------------------------------

struct Vertex {
    float vertexData[LENGTH];
};

void clear(out Vertex v) {
    for (int i = 0; i < LENGTH; i++) {
        v.vertexData[i] = 0;
    }
}

void addWithWeight(inout Vertex v, Vertex src, float weight) {
    for(int j = 0; j < LENGTH; j++) {
        v.vertexData[j] += weight * src.vertexData[j];
    }
}

Vertex readVertex(int index) {
    Vertex v;
    int vertexIndex = srcOffset + index * SRC_STRIDE;
    for(int j = 0; j < LENGTH; j++) {
        v.vertexData[j] = texelFetch(vertexBuffer, vertexIndex+j).x;
    }
    return v;
}

void writeVertex(Vertex v) {
    for(int i = 0; i < LENGTH; i++) {
        outVertexBuffer[i] = v.vertexData[i];
    }
}

//------------------------------------------------------------------------------

#if defined(OPENSUBDIV_GLSL_XFB_USE_1ST_DERIVATIVES) && \
    defined(OPENSUBDIV_GLSL_XFB_INTERLEAVED_1ST_DERIVATIVE_BUFFERS)
out float outDeriv1Buffer[2*LENGTH];

void writeDu(Vertex v) {
    for(int i = 0; i < LENGTH; i++) {
        outDeriv1Buffer[i] = v.vertexData[i];
    }
}

void writeDv(Vertex v) {
    for(int i = 0; i < LENGTH; i++) {
        outDeriv1Buffer[i+LENGTH] = v.vertexData[i];
    }
}
#elif defined(OPENSUBDIV_GLSL_XFB_USE_1ST_DERIVATIVES)
out float outDuBuffer[LENGTH];
out float outDvBuffer[LENGTH];

void writeDu(Vertex v) {
    for(int i = 0; i < LENGTH; i++) {
        outDuBuffer[i] = v.vertexData[i];
    }
}

void writeDv(Vertex v) {
    for(int i = 0; i < LENGTH; i++) {
        outDvBuffer[i] = v.vertexData[i];
    }
}
#endif

#if defined(OPENSUBDIV_GLSL_XFB_USE_2ND_DERIVATIVES) && \
    defined(OPENSUBDIV_GLSL_XFB_INTERLEAVED_2ND_DERIVATIVE_BUFFERS)
out float outDeriv2Buffer[3*LENGTH];

void writeDuu(Vertex v) {
    for(int i = 0; i < LENGTH; i++) {
        outDeriv2Buffer[i] = v.vertexData[i];
    }
}

void writeDuv(Vertex v) {
    for(int i = 0; i < LENGTH; i++) {
        outDeriv2Buffer[i+LENGTH] = v.vertexData[i];
    }
}

void writeDvv(Vertex v) {
    for(int i = 0; i < LENGTH; i++) {
        outDeriv2Buffer[i+2*LENGTH] = v.vertexData[i];
    }
}
#elif defined(OPENSUBDIV_GLSL_XFB_USE_2ND_DERIVATIVES)
out float outDuuBuffer[LENGTH];
out float outDuvBuffer[LENGTH];
out float outDvvBuffer[LENGTH];

void writeDuu(Vertex v) {
    for(int i = 0; i < LENGTH; i++) {
        outDuuBuffer[i] = v.vertexData[i];
    }
}

void writeDuv(Vertex v) {
    for(int i = 0; i < LENGTH; i++) {
        outDuvBuffer[i] = v.vertexData[i];
    }
}

void writeDvv(Vertex v) {
    for(int i = 0; i < LENGTH; i++) {
        outDvvBuffer[i] = v.vertexData[i];
    }
}
#endif

//------------------------------------------------------------------------------

#if defined(OPENSUBDIV_GLSL_XFB_KERNEL_EVAL_STENCILS)

uniform usamplerBuffer sizes;
uniform isamplerBuffer offsets;
uniform isamplerBuffer indices;
uniform samplerBuffer  weights;

#if defined(OPENSUBDIV_GLSL_XFB_USE_1ST_DERIVATIVES)
uniform samplerBuffer  duWeights;
uniform samplerBuffer  dvWeights;
#endif

#if defined(OPENSUBDIV_GLSL_XFB_USE_2ND_DERIVATIVES)
uniform samplerBuffer  duuWeights;
uniform samplerBuffer  duvWeights;
uniform samplerBuffer  dvvWeights;
#endif

uniform int batchStart = 0;
uniform int batchEnd = 0;

void main() {
    int current = gl_VertexID + batchStart;

    if (current>=batchEnd) {
        return;
    }

    Vertex dst, du, dv, duu, duv, dvv;
    clear(dst);
    clear(du);
    clear(dv);
    clear(duu);
    clear(duv);
    clear(dvv);

    int offset = texelFetch(offsets, current).x;
    uint size = texelFetch(sizes, current).x;

    for (int stencil=0; stencil<size; ++stencil) {
        int index = texelFetch(indices, offset+stencil).x;
        float weight = texelFetch(weights, offset+stencil).x;
        addWithWeight(dst, readVertex( index ), weight);

#if defined(OPENSUBDIV_GLSL_XFB_USE_1ST_DERIVATIVES)
        float duWeight = texelFetch(duWeights, offset+stencil).x;
        float dvWeight = texelFetch(dvWeights, offset+stencil).x;
        addWithWeight(du,  readVertex(index), duWeight);
        addWithWeight(dv,  readVertex(index), dvWeight);
#endif
#if defined(OPENSUBDIV_GLSL_XFB_USE_2ND_DERIVATIVES)
        float duuWeight = texelFetch(duuWeights, offset+stencil).x;
        float duvWeight = texelFetch(duvWeights, offset+stencil).x;
        float dvvWeight = texelFetch(dvvWeights, offset+stencil).x;
        addWithWeight(duu,  readVertex(index), duuWeight);
        addWithWeight(duv,  readVertex(index), duvWeight);
        addWithWeight(dvv,  readVertex(index), dvvWeight);
#endif
    }
    writeVertex(dst);

#if defined(OPENSUBDIV_GLSL_XFB_USE_1ST_DERIVATIVES)
    writeDu(du);
    writeDv(dv);
#endif
#if defined(OPENSUBDIV_GLSL_XFB_USE_2ND_DERIVATIVES)
    writeDuu(duu);
    writeDuv(duv);
    writeDvv(dvv);
#endif
}

#endif

//------------------------------------------------------------------------------

#if defined(OPENSUBDIV_GLSL_XFB_KERNEL_EVAL_PATCHES)

layout (location = 0) in ivec3 patchHandles;
layout (location = 1) in vec2  patchCoords;

//struct PatchArray {
//    int patchType;
//    int numPatches;
//    int indexBase;        // an offset within the index buffer
//    int primitiveIdBase;  // an offset within the patch param buffer
//};
// # of patcharrays is 1 or 2.

uniform ivec4 patchArray[2];
uniform isamplerBuffer patchParamBuffer;
uniform isamplerBuffer patchIndexBuffer;

uint getDepth(uint patchBits) {
    return (patchBits & 0xfU);
}

float getParamFraction(uint patchBits) {
    uint nonQuadRoot = (patchBits >> 4) & 0x1U;
    uint depth = getDepth(patchBits);
    if (nonQuadRoot == 1) {
        return 1.0f / float( 1 << (depth-1) );
    } else {
        return 1.0f / float( 1 << depth );
    }
}

vec2 normalizePatchCoord(uint patchBits, vec2 uv) {
    float frac = getParamFraction(patchBits);

    uint iu = (patchBits >> 22) & 0x3ffU;
    uint iv = (patchBits >> 12) & 0x3ffU;

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
    int current = gl_VertexID;

    ivec3 handle = patchHandles;
    int patchIndex = handle.y;

    vec2 coord = patchCoords;
    ivec4 array = patchArray[handle.x];

    uint patchBits = texelFetch(patchParamBuffer, patchIndex).y;
    int patchType = isRegular(patchBits) ? 6 : array.x;

    // normalize
    coord = normalizePatchCoord(patchBits, coord);
    float dScale = float(1 << getDepth(patchBits));
    int boundary = int((patchBits >> 8) & 0xfU);

    float wP[20], wDs[20], wDt[20], wDss[20], wDst[20], wDtt[20];

    int numControlVertices = 0;
    if (patchType == 3) {
        float wP4[4], wDs4[4], wDt4[4], wDss4[4], wDst4[4], wDtt4[4];
        OsdGetBilinearPatchWeights(coord.s, coord.t, dScale, wP4,
                                   wDs4, wDt4, wDss4, wDst4, wDtt4);
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
        OsdGetBSplinePatchWeights(coord.s, coord.t, dScale, boundary, wP16,
                                  wDs16, wDt16, wDss16, wDst16, wDtt16);
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
        OsdGetGregoryPatchWeights(coord.s, coord.t, dScale, wP,
                                  wDs, wDt, wDss, wDst, wDtt);
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
        int index = texelFetch(patchIndexBuffer, indexBase + cv).x;
        addWithWeight(dst, readVertex(index), wP[cv]);
        addWithWeight(du,  readVertex(index), wDs[cv]);
        addWithWeight(dv,  readVertex(index), wDt[cv]);
        addWithWeight(duu, readVertex(index), wDss[cv]);
        addWithWeight(duv, readVertex(index), wDst[cv]);
        addWithWeight(dvv, readVertex(index), wDtt[cv]);
    }

    writeVertex(dst);

#if defined(OPENSUBDIV_GLSL_XFB_USE_1ST_DERIVATIVES)
    writeDu(du);
    writeDv(dv);
#endif
#if defined(OPENSUBDIV_GLSL_XFB_USE_2ND_DERIVATIVES)
    writeDuu(duu);
    writeDuv(duv);
    writeDvv(dvv);
#endif
}

#endif

