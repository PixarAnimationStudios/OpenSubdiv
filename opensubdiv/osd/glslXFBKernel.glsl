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

#if defined(OPENSUBDIV_GLSL_XFB_USE_DERIVATIVES)
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

//------------------------------------------------------------------------------

#if defined(OPENSUBDIV_GLSL_XFB_KERNEL_EVAL_STENCILS)

uniform usamplerBuffer sizes;
uniform isamplerBuffer offsets;
uniform isamplerBuffer indices;
uniform samplerBuffer  weights;

#if defined(OPENSUBDIV_GLSL_XFB_USE_DERIVATIVES)
uniform samplerBuffer  duWeights;
uniform samplerBuffer  dvWeights;
#endif

uniform int batchStart = 0;
uniform int batchEnd = 0;

void main() {
    int current = gl_VertexID + batchStart;

    if (current>=batchEnd) {
        return;
    }

    Vertex dst, du, dv;
    clear(dst);
    clear(du);
    clear(dv);

    int offset = texelFetch(offsets, current).x;
    uint size = texelFetch(sizes, current).x;

    for (int stencil=0; stencil<size; ++stencil) {
        int index = texelFetch(indices, offset+stencil).x;
        float weight = texelFetch(weights, offset+stencil).x;
        addWithWeight(dst, readVertex( index ), weight);

#if defined(OPENSUBDIV_GLSL_XFB_USE_DERIVATIVES)
        float duWeight = texelFetch(duWeights, offset+stencil).x;
        float dvWeight = texelFetch(dvWeights, offset+stencil).x;
        addWithWeight(du,  readVertex(index), duWeight);
        addWithWeight(dv,  readVertex(index), dvWeight);
#endif
    }
    writeVertex(dst);

#if defined(OPENSUBDIV_GLSL_XFB_USE_DERIVATIVES)
    writeDu(du);
    writeDv(dv);
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

void getBSplineWeights(float t, inout vec4 point, inout vec4 deriv) {
    // The four uniform cubic B-Spline basis functions evaluated at t:
    float one6th = 1.0f / 6.0f;

    float t2 = t * t;
    float t3 = t * t2;

    point.x = one6th * (1.0f - 3.0f*(t -      t2) -      t3);
    point.y = one6th * (4.0f           - 6.0f*t2  + 3.0f*t3);
    point.z = one6th * (1.0f + 3.0f*(t +      t2  -      t3));
    point.w = one6th * (                                 t3);

    // Derivatives of the above four basis functions at t:
    deriv.x = -0.5f*t2 +      t - 0.5f;
    deriv.y =  1.5f*t2 - 2.0f*t;
    deriv.z = -1.5f*t2 +      t + 0.5f;
    deriv.w =  0.5f*t2;
}

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

void adjustBoundaryWeights(uint bits, inout vec4 sWeights, inout vec4 tWeights) {
    uint boundary = ((bits >> 8) & 0xfU);

    if ((boundary & 1U) != 0) {
        tWeights[2] -= tWeights[0];
        tWeights[1] += 2*tWeights[0];
        tWeights[0] = 0;
    }
    if ((boundary & 2U) != 0) {
        sWeights[1] -= sWeights[3];
        sWeights[2] += 2*sWeights[3];
        sWeights[3] = 0;
    }
    if ((boundary & 4U) != 0) {
        tWeights[1] -= tWeights[3];
        tWeights[2] += 2*tWeights[3];
        tWeights[3] = 0;
    }
    if ((boundary & 8U) != 0) {
        sWeights[2] -= sWeights[0];
        sWeights[1] += 2*sWeights[0];
        sWeights[0] = 0;
    }
}

void main() {
    int current = gl_VertexID;

    ivec3 handle = patchHandles;
    int patchIndex = handle.y;

    vec2 coord = patchCoords;
    ivec4 array = patchArray[handle.x];
    int patchType = array.x;
    int numControlVertices = 16;

    uint patchBits = texelFetch(patchParamBuffer, patchIndex).y;

    // normalize
    coord = normalizePatchCoord(patchBits, coord);
    float dScale = float(1 << getDepth(patchBits));

    // if regular
    float wP[20], wDs[20], wDt[20];
    {
        vec4 sWeights, tWeights, dsWeights, dtWeights;
        getBSplineWeights(coord.s, sWeights, dsWeights);
        getBSplineWeights(coord.t, tWeights, dtWeights);

        adjustBoundaryWeights(patchBits, sWeights, tWeights);
        adjustBoundaryWeights(patchBits, dsWeights, dtWeights);

        for (int k = 0; k < 4; ++k) {
            for (int l = 0; l < 4; ++l) {
                wP[4*k+l]  = sWeights[l]  * tWeights[k];
                wDs[4*k+l] = dsWeights[l] * tWeights[k]  * dScale;
                wDt[4*k+l] = sWeights[l]  * dtWeights[k] * dScale;
            }
        }
    }

    Vertex dst, du, dv;
    clear(dst);
    clear(du);
    clear(dv);

    int indexBase = array.z + handle.z;
    for (int cv = 0; cv < numControlVertices; ++cv) {
        int index = texelFetch(patchIndexBuffer, indexBase + cv).x;
        addWithWeight(dst, readVertex(index), wP[cv]);
        addWithWeight(du,  readVertex(index), wDs[cv]);
        addWithWeight(dv,  readVertex(index), wDt[cv]);
    }

    writeVertex(dst);

#if defined(OPENSUBDIV_GLSL_XFB_USE_DERIVATIVES)
    writeDu(du);
    writeDv(dv);
#endif

}

#endif

