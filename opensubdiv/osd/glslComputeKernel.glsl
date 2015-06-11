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

// source and destination buffers

uniform int srcOffset = 0;
uniform int dstOffset = 0;
layout(binding=0) buffer src_buffer      { float    srcVertexBuffer[]; };
layout(binding=1) buffer dst_buffer      { float    dstVertexBuffer[]; };

// derivative buffers (if needed)

#if defined(OPENSUBDIV_GLSL_COMPUTE_USE_DERIVATIVES)
uniform ivec3 duDesc;
uniform ivec3 dvDesc;
layout(binding=2) buffer du_buffer   { float duBuffer[]; };
layout(binding=3) buffer dv_buffer   { float dvBuffer[]; };
#endif

// stencil buffers

#if defined(OPENSUBDIV_GLSL_COMPUTE_KERNEL_EVAL_STENCILS)

uniform int batchStart = 0;
uniform int batchEnd = 0;
layout(binding=4) buffer stencilSizes    { int      _sizes[];   };
layout(binding=5) buffer stencilOffsets  { int      _offsets[]; };
layout(binding=6) buffer stencilIndices  { int      _indices[]; };
layout(binding=7) buffer stencilWeights  { float    _weights[]; };

#if defined(OPENSUBDIV_GLSL_COMPUTE_USE_DERIVATIVES)
layout(binding=8) buffer stencilDuWeights { float  _duWeights[]; };
layout(binding=9) buffer stencilDvWeights { float  _dvWeights[]; };
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

#if defined(OPENSUBDIV_GLSL_COMPUTE_USE_DERIVATIVES)
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

#if defined(OPENSUBDIV_GLSL_COMPUTE_USE_DERIVATIVES)
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

void adjustBoundaryWeights(uint bits, inout vec4 sWeights, inout vec4 tWeights) {
    uint boundary = ((bits >> 8) & 0xf);

    if ((boundary & 1) != 0) {
        tWeights[2] -= tWeights[0];
        tWeights[1] += 2*tWeights[0];
        tWeights[0] = 0;
    }
    if ((boundary & 2) != 0) {
        sWeights[1] -= sWeights[3];
        sWeights[2] += 2*sWeights[3];
        sWeights[3] = 0;
    }
    if ((boundary & 4) != 0) {
        tWeights[1] -= tWeights[3];
        tWeights[2] += 2*tWeights[3];
        tWeights[3] = 0;
    }
    if ((boundary & 8) != 0) {
        sWeights[2] -= sWeights[0];
        sWeights[1] += 2*sWeights[0];
        sWeights[0] = 0;
    }
}

void main() {

    int current = int(gl_GlobalInvocationID.x);

    PatchCoord coord = patchCoords[current];
    int patchIndex = coord.patchIndex;

    ivec4 array = patchArray[coord.arrayIndex];
    int patchType = 6; // array.x XXX: REGULAR only for now.
    int numControlVertices = 16;

    uint patchBits = patchParamBuffer[patchIndex].field1;
    vec2 uv = normalizePatchCoord(patchBits, vec2(coord.s, coord.t));
    float dScale = float(1 << getDepth(patchBits));

    float wP[20], wDs[20], wDt[20];
    if (patchType == 6) {  // REGULAR
        vec4 sWeights, tWeights, dsWeights, dtWeights;
        getBSplineWeights(uv.x, sWeights, dsWeights);
        getBSplineWeights(uv.y, tWeights, dtWeights);

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
        // TODO: GREGORY BASIS
    }

    Vertex dst, du, dv;
    clear(dst);
    clear(du);
    clear(dv);

    int indexBase = array.z + coord.vertIndex;
    for (int cv = 0; cv < numControlVertices; ++cv) {
        int index = patchIndexBuffer[indexBase + cv];
        addWithWeight(dst, readVertex(index), wP[cv]);
        addWithWeight(du, readVertex(index), wDs[cv]);
        addWithWeight(dv, readVertex(index), wDt[cv]);
    }
    writeVertex(current, dst);

#if defined(OPENSUBDIV_GLSL_COMPUTE_USE_DERIVATIVES)
    if (duDesc.y > 0) { // length
        writeDu(current, du);
    }
    if (dvDesc.y > 0) {
        writeDv(current, dv);
    }
#endif
}

#endif
