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

#include "../osd/cpuKernel.h"
#include "../osd/vertexDescriptor.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

static inline void
clear(float *dst, OsdVertexBufferDescriptor const &desc) {

    memset(dst, 0, desc.length*sizeof(float));
}

static inline void
addWithWeight(float *dst, const float *srcOrigin, int srcIndex, float weight,
              OsdVertexBufferDescriptor const &desc) {

    if (srcOrigin && dst) {
        const float *src = srcOrigin + srcIndex * desc.stride;
        for (int k = 0; k < desc.length; ++k) {
            dst[k] += src[k] * weight;
        }
    }
}

static inline void
copy(float *dstOrigin, const float *src, int dstIndex,
     OsdVertexBufferDescriptor const &desc) {

    if (dstOrigin && src) {
        float *dst = dstOrigin + dstIndex * desc.stride;
        memcpy(dst, src, desc.length*sizeof(float));
    }
}

void OsdCpuComputeFace(
    float * vertex, float * varying,
    OsdVertexBufferDescriptor const &vertexDesc,
    OsdVertexBufferDescriptor const &varyingDesc,
    const int *F_IT, const int *F_ITa, int vertexOffset, int tableOffset,
    int start, int end) {
    if(vertexDesc == OsdVertexBufferDescriptor(0, 4, 4) && varying == NULL) {
        ComputeFaceKernel<4>
            (vertex, F_IT, F_ITa, vertexOffset, tableOffset, start,  end);
    } else if(vertexDesc == OsdVertexBufferDescriptor(0, 8, 8) && varying == NULL) {
        ComputeFaceKernel<8>
            (vertex, F_IT, F_ITa, vertexOffset, tableOffset, start,  end);
    }
    else {
        float *vertexResults = (float*)alloca(vertexDesc.length * sizeof(float));
        float *varyingResults = (float*)alloca(varyingDesc.length * sizeof(float));

        for (int i = start + tableOffset; i < end + tableOffset; i++) {
            int h = F_ITa[2*i];
            int n = F_ITa[2*i+1];

            float weight = 1.0f/n;
            int dstIndex = i + vertexOffset - tableOffset;

            // clear
            clear(vertexResults, vertexDesc);
            clear(varyingResults, varyingDesc);

            // accum
            for (int j = 0; j < n; ++j) {
                int index = F_IT[h+j];
                addWithWeight(vertexResults, vertex, index, weight, vertexDesc);
                addWithWeight(varyingResults, varying, index, weight, varyingDesc);
            }

            // write results
            copy(vertex, vertexResults, dstIndex, vertexDesc);
            copy(varying, varyingResults, dstIndex, varyingDesc);
        }
    }
}

void OsdCpuComputeQuadFace(
    float * vertex, float * varying,
    OsdVertexBufferDescriptor const &vertexDesc,
    OsdVertexBufferDescriptor const &varyingDesc,
    const int *F_IT, int vertexOffset, int tableOffset,
    int start, int end) {

    float *vertexResults = (float*)alloca(vertexDesc.length * sizeof(float));
    float *varyingResults = (float*)alloca(varyingDesc.length * sizeof(float));

    for (int i = start; i < end; i++) {
        int fidx0 = F_IT[tableOffset + 4 * i + 0];
        int fidx1 = F_IT[tableOffset + 4 * i + 1];
        int fidx2 = F_IT[tableOffset + 4 * i + 2];
        int fidx3 = F_IT[tableOffset + 4 * i + 3];

        int dstIndex = i + vertexOffset;

        // clear
        clear(vertexResults, vertexDesc);
        clear(varyingResults, varyingDesc);

        // accum
        addWithWeight(vertexResults, vertex, fidx0, 0.25f, vertexDesc);
        addWithWeight(vertexResults, vertex, fidx1, 0.25f, vertexDesc);
        addWithWeight(vertexResults, vertex, fidx2, 0.25f, vertexDesc);
        addWithWeight(vertexResults, vertex, fidx3, 0.25f, vertexDesc);
        addWithWeight(varyingResults, varying, fidx0, 0.25f, varyingDesc);
        addWithWeight(varyingResults, varying, fidx1, 0.25f, varyingDesc);
        addWithWeight(varyingResults, varying, fidx2, 0.25f, varyingDesc);
        addWithWeight(varyingResults, varying, fidx3, 0.25f, varyingDesc);

        // write results
        copy(vertex, vertexResults, dstIndex, vertexDesc);
        copy(varying, varyingResults, dstIndex, varyingDesc);
    }
}

void OsdCpuComputeTriQuadFace(
    float * vertex, float * varying,
    OsdVertexBufferDescriptor const &vertexDesc,
    OsdVertexBufferDescriptor const &varyingDesc,
    const int *F_IT, int vertexOffset, int tableOffset,
    int start, int end) {

    float *vertexResults = (float*)alloca(vertexDesc.length * sizeof(float));
    float *varyingResults = (float*)alloca(varyingDesc.length * sizeof(float));

    for (int i = start; i < end; i++) {
        int fidx0 = F_IT[tableOffset + 4 * i + 0];
        int fidx1 = F_IT[tableOffset + 4 * i + 1];
        int fidx2 = F_IT[tableOffset + 4 * i + 2];
        int fidx3 = F_IT[tableOffset + 4 * i + 3];
        bool triangle = (fidx2 == fidx3);
        float weight = (triangle ? 1.0f / 3.0f : 1.0f / 4.0f);

        int dstIndex = i + vertexOffset;

        // clear
        clear(vertexResults, vertexDesc);
        clear(varyingResults, varyingDesc);

        // accum
        addWithWeight(vertexResults, vertex, fidx0, weight, vertexDesc);
        addWithWeight(vertexResults, vertex, fidx1, weight, vertexDesc);
        addWithWeight(vertexResults, vertex, fidx2, weight, vertexDesc);
        addWithWeight(varyingResults, varying, fidx0, weight, varyingDesc);
        addWithWeight(varyingResults, varying, fidx1, weight, varyingDesc);
        addWithWeight(varyingResults, varying, fidx2, weight, varyingDesc);
        if (!triangle) {
            addWithWeight(vertexResults, vertex, fidx3, weight, vertexDesc);
            addWithWeight(varyingResults, varying, fidx3, weight, varyingDesc);
        }

        // write results
        copy(vertex, vertexResults, dstIndex, vertexDesc);
        copy(varying, varyingResults, dstIndex, varyingDesc);
    }
}

void OsdCpuComputeEdge(
    float *vertex, float *varying,
    OsdVertexBufferDescriptor const &vertexDesc,
    OsdVertexBufferDescriptor const &varyingDesc,
    const int *E_IT, const float *E_W, int vertexOffset, int tableOffset,
    int start, int end) {
    if(vertexDesc == OsdVertexBufferDescriptor(0, 4, 4) && varying == NULL) {
        ComputeEdgeKernel<4>(vertex, E_IT, E_W, vertexOffset, tableOffset,
                             start, end);
    }
    else if(vertexDesc == OsdVertexBufferDescriptor(0, 8, 8) && varying == NULL) {
        ComputeEdgeKernel<8>(vertex, E_IT, E_W, vertexOffset, tableOffset,
                             start, end);
    }
    else {
        float *vertexResults = (float*)alloca(vertexDesc.length * sizeof(float));
        float *varyingResults = (float*)alloca(varyingDesc.length * sizeof(float));

        for (int i = start + tableOffset; i < end + tableOffset; i++) {
            int eidx0 = E_IT[4*i+0];
            int eidx1 = E_IT[4*i+1];
            int eidx2 = E_IT[4*i+2];
            int eidx3 = E_IT[4*i+3];

            float vertWeight = E_W[i*2+0];

            int dstIndex = i + vertexOffset - tableOffset;
            clear(vertexResults, vertexDesc);
            clear(varyingResults, varyingDesc);

            addWithWeight(vertexResults, vertex, eidx0, vertWeight, vertexDesc);
            addWithWeight(vertexResults, vertex, eidx1, vertWeight, vertexDesc);

            if (eidx2 != -1) {
                float faceWeight = E_W[i*2+1];

                addWithWeight(vertexResults, vertex, eidx2, faceWeight, vertexDesc);
                addWithWeight(vertexResults, vertex, eidx3, faceWeight, vertexDesc);
            }

            addWithWeight(varyingResults, varying, eidx0, 0.5f, varyingDesc);
            addWithWeight(varyingResults, varying, eidx1, 0.5f, varyingDesc);

            copy(vertex, vertexResults, dstIndex, vertexDesc);
            copy(varying, varyingResults, dstIndex, varyingDesc);
        }
    }
}

void OsdCpuComputeRestrictedEdge(
    float *vertex, float *varying,
    OsdVertexBufferDescriptor const &vertexDesc,
    OsdVertexBufferDescriptor const &varyingDesc,
    const int *E_IT, int vertexOffset, int tableOffset,
    int start, int end) {

    float *vertexResults = (float*)alloca(vertexDesc.length * sizeof(float));
    float *varyingResults = (float*)alloca(varyingDesc.length * sizeof(float));

    for (int i = start + tableOffset; i < end + tableOffset; i++) {
        int eidx0 = E_IT[4*i+0];
        int eidx1 = E_IT[4*i+1];
        int eidx2 = E_IT[4*i+2];
        int eidx3 = E_IT[4*i+3];

        int dstIndex = i + vertexOffset - tableOffset;
        clear(vertexResults, vertexDesc);
        clear(varyingResults, varyingDesc);

        addWithWeight(vertexResults, vertex, eidx0, 0.25f, vertexDesc);
        addWithWeight(vertexResults, vertex, eidx1, 0.25f, vertexDesc);
        addWithWeight(vertexResults, vertex, eidx2, 0.25f, vertexDesc);
        addWithWeight(vertexResults, vertex, eidx3, 0.25f, vertexDesc);

        addWithWeight(varyingResults, varying, eidx0, 0.5f, varyingDesc);
        addWithWeight(varyingResults, varying, eidx1, 0.5f, varyingDesc);

        copy(vertex, vertexResults, dstIndex, vertexDesc);
        copy(varying, varyingResults, dstIndex, varyingDesc);
    }
}

void OsdCpuComputeVertexA(
    float *vertex, float *varying,
    OsdVertexBufferDescriptor const &vertexDesc,
    OsdVertexBufferDescriptor const &varyingDesc,
    const int *V_ITa, const float *V_W, int vertexOffset, int tableOffset,
    int start, int end, int pass) {
    if(vertexDesc == OsdVertexBufferDescriptor(0, 4, 4) && varying == NULL) {
        ComputeVertexAKernel<4>(vertex, V_ITa, V_W, vertexOffset, tableOffset,
                             start, end, pass);
    }
    else if(vertexDesc == OsdVertexBufferDescriptor(0, 8, 8) && varying == NULL) {
        ComputeVertexAKernel<8>(vertex, V_ITa, V_W, vertexOffset, tableOffset,
                             start, end, pass);
    }
    else {
        float *vertexResults = (float*)alloca(vertexDesc.length * sizeof(float));
        float *varyingResults = (float*)alloca(varyingDesc.length * sizeof(float));

        for (int i = start + tableOffset; i < end + tableOffset; i++) {
            int n     = V_ITa[5*i+1];
            int p     = V_ITa[5*i+2];
            int eidx0 = V_ITa[5*i+3];
            int eidx1 = V_ITa[5*i+4];

            float weight = (pass == 1) ? V_W[i] : 1.0f - V_W[i];

            // In the case of fractional weight, the weight must be inverted since
            // the value is shared with the k_Smooth kernel (statistically the
            // k_Smooth kernel runs much more often than this one)
            if (weight > 0.0f && weight < 1.0f && n > 0)
                weight = 1.0f - weight;

            int dstIndex = i + vertexOffset - tableOffset;

            clear(vertexResults, vertexDesc);
            clear(varyingResults, varyingDesc);
            if (pass) {
                // copy previous results
                addWithWeight(vertexResults, vertex, dstIndex, 1.0f, vertexDesc);
            }

            if (eidx0 == -1 || (pass == 0 && (n == -1))) {
                addWithWeight(vertexResults, vertex, p, weight, vertexDesc);
            } else {
                addWithWeight(vertexResults, vertex, p, weight * 0.75f, vertexDesc);
                addWithWeight(vertexResults, vertex, eidx0, weight * 0.125f, vertexDesc);
                addWithWeight(vertexResults, vertex, eidx1, weight * 0.125f, vertexDesc);
            }

            copy(vertex, vertexResults, dstIndex, vertexDesc);
            if (not pass) {
                addWithWeight(varyingResults, varying, p, 1.0f, varyingDesc);
                copy(varying, varyingResults, dstIndex, varyingDesc);
            }
        }
    }
}

void OsdCpuComputeVertexB(
    float *vertex, float *varying,
    OsdVertexBufferDescriptor const &vertexDesc,
    OsdVertexBufferDescriptor const &varyingDesc,
    const int *V_ITa, const int *V_IT, const float *V_W,
    int vertexOffset, int tableOffset, int start, int end) {
    if(vertexDesc == OsdVertexBufferDescriptor(0, 4, 4) && varying == NULL) {
        ComputeVertexBKernel<4>(vertex, V_ITa, V_IT, V_W,
            vertexOffset, tableOffset, start, end);
    }
    else if(vertexDesc == OsdVertexBufferDescriptor(0, 8, 8) && varying == NULL) {
        ComputeVertexBKernel<8>(vertex, V_ITa, V_IT, V_W,
            vertexOffset, tableOffset, start, end);
    }
    else {
        float *vertexResults = (float*)alloca(vertexDesc.length * sizeof(float));
        float *varyingResults = (float*)alloca(varyingDesc.length * sizeof(float));

        for (int i = start + tableOffset; i < end + tableOffset; i++) {
            int h = V_ITa[5*i];
            int n = V_ITa[5*i+1];
            int p = V_ITa[5*i+2];

            float weight = V_W[i];
            float wp = 1.0f/static_cast<float>(n*n);
            float wv = (n-2.0f) * n * wp;

            int dstIndex = i + vertexOffset - tableOffset;
            clear(vertexResults, vertexDesc);
            clear(varyingResults, varyingDesc);

            addWithWeight(vertexResults, vertex, p, weight * wv, vertexDesc);

            for (int j = 0; j < n; ++j) {
                addWithWeight(vertexResults, vertex, V_IT[h+j*2], weight * wp, vertexDesc);
                addWithWeight(vertexResults, vertex, V_IT[h+j*2+1], weight * wp, vertexDesc);
            }
            addWithWeight(varyingResults, varying, p, 1.0f, varyingDesc);

            copy(vertex, vertexResults, dstIndex, vertexDesc);
            copy(varying, varyingResults, dstIndex, varyingDesc);
        }
    }
}

void OsdCpuComputeRestrictedVertexB1(
    float *vertex, float *varying,
    OsdVertexBufferDescriptor const &vertexDesc,
    OsdVertexBufferDescriptor const &varyingDesc,
    const int *V_ITa, const int *V_IT,
    int vertexOffset, int tableOffset, int start, int end) {

    float *vertexResults = (float*)alloca(vertexDesc.length * sizeof(float));
    float *varyingResults = (float*)alloca(varyingDesc.length * sizeof(float));

    for (int i = start + tableOffset; i < end + tableOffset; i++) {
        int h = V_ITa[5*i];
        int p = V_ITa[5*i+2];

        int dstIndex = i + vertexOffset - tableOffset;
        clear(vertexResults, vertexDesc);
        clear(varyingResults, varyingDesc);

        addWithWeight(vertexResults, vertex, p, 0.5f, vertexDesc);

        for (int j = 0; j < 8; ++j, ++h)
            addWithWeight(vertexResults, vertex, V_IT[h], 0.0625f, vertexDesc);

        addWithWeight(varyingResults, varying, p, 1.0f, varyingDesc);

        copy(vertex, vertexResults, dstIndex, vertexDesc);
        copy(varying, varyingResults, dstIndex, varyingDesc);
    }
}

void OsdCpuComputeRestrictedVertexB2(
    float *vertex, float *varying,
    OsdVertexBufferDescriptor const &vertexDesc,
    OsdVertexBufferDescriptor const &varyingDesc,
    const int *V_ITa, const int *V_IT,
    int vertexOffset, int tableOffset, int start, int end) {

    float *vertexResults = (float*)alloca(vertexDesc.length * sizeof(float));
    float *varyingResults = (float*)alloca(varyingDesc.length * sizeof(float));

    for (int i = start + tableOffset; i < end + tableOffset; i++) {
        int h = V_ITa[5*i];
        int n = V_ITa[5*i+1];
        int p = V_ITa[5*i+2];

        float wp = 1.0f/static_cast<float>(n*n);
        float wv = (n-2.0f) * n * wp;

        int dstIndex = i + vertexOffset - tableOffset;
        clear(vertexResults, vertexDesc);
        clear(varyingResults, varyingDesc);

        addWithWeight(vertexResults, vertex, p, wv, vertexDesc);

        for (int j = 0; j < n; ++j) {
            addWithWeight(vertexResults, vertex, V_IT[h+j*2], wp, vertexDesc);
            addWithWeight(vertexResults, vertex, V_IT[h+j*2+1], wp, vertexDesc);
        }
        addWithWeight(varyingResults, varying, p, 1.0f, varyingDesc);

        copy(vertex, vertexResults, dstIndex, vertexDesc);
        copy(varying, varyingResults, dstIndex, varyingDesc);
    }
}

void OsdCpuComputeRestrictedVertexA(
    float *vertex, float *varying,
    OsdVertexBufferDescriptor const &vertexDesc,
    OsdVertexBufferDescriptor const &varyingDesc,
    const int *V_ITa,
    int vertexOffset, int tableOffset, int start, int end) {

    float *vertexResults = (float*)alloca(vertexDesc.length * sizeof(float));
    float *varyingResults = (float*)alloca(varyingDesc.length * sizeof(float));

    for (int i = start + tableOffset; i < end + tableOffset; i++) {
        int p     = V_ITa[5*i+2];
        int eidx0 = V_ITa[5*i+3];
        int eidx1 = V_ITa[5*i+4];

        int dstIndex = i + vertexOffset - tableOffset;
        clear(vertexResults, vertexDesc);
        clear(varyingResults, varyingDesc);

        addWithWeight(vertexResults, vertex, p, 0.75f, vertexDesc);
        addWithWeight(vertexResults, vertex, eidx0, 0.125f, vertexDesc);
        addWithWeight(vertexResults, vertex, eidx1, 0.125f, vertexDesc);
        addWithWeight(varyingResults, varying, p, 1.0f, varyingDesc);

        copy(vertex, vertexResults, dstIndex, vertexDesc);
        copy(varying, varyingResults, dstIndex, varyingDesc);
    }
}

void OsdCpuComputeLoopVertexB(
    float *vertex, float *varying,
    OsdVertexBufferDescriptor const &vertexDesc,
    OsdVertexBufferDescriptor const &varyingDesc,
    const int *V_ITa, const int *V_IT, const float *V_W,
    int vertexOffset, int tableOffset, int start, int end) {
    if(vertexDesc == OsdVertexBufferDescriptor(0, 4, 4) && varying == NULL) {
        ComputeLoopVertexBKernel<4>(vertex, V_ITa, V_IT, V_W, vertexOffset, 
                              tableOffset, start, end);
    }
    else if(vertexDesc == OsdVertexBufferDescriptor(0, 8, 8) && varying == NULL) {
        ComputeLoopVertexBKernel<8>(vertex, V_ITa, V_IT, V_W, vertexOffset, 
                              tableOffset, start, end);    
    }    
    else {
        float *vertexResults = (float*)alloca(vertexDesc.length * sizeof(float));
        float *varyingResults = (float*)alloca(varyingDesc.length * sizeof(float));

        for (int i = start + tableOffset; i < end + tableOffset; i++) {
            int h = V_ITa[5*i];
            int n = V_ITa[5*i+1];
            int p = V_ITa[5*i+2];

            float weight = V_W[i];
            float wp = 1.0f/static_cast<float>(n);
            float beta = 0.25f * cosf(static_cast<float>(M_PI) * 2.0f * wp) + 0.375f;
            beta = beta * beta;
            beta = (0.625f - beta) * wp;

            int dstIndex = i + vertexOffset - tableOffset;
            clear(vertexResults, vertexDesc);
            clear(varyingResults, varyingDesc);

            addWithWeight(vertexResults, vertex, p, weight * (1.0f - (beta * n)), vertexDesc);

            for (int j = 0; j < n; ++j)
                addWithWeight(vertexResults, vertex, V_IT[h+j], weight * beta, vertexDesc);

            addWithWeight(varyingResults, varying, p, 1.0f, varyingDesc);

            copy(vertex, vertexResults, dstIndex, vertexDesc);
            copy(varying, varyingResults, dstIndex, varyingDesc);
        }
    }
}

void OsdCpuComputeBilinearEdge(
    float *vertex, float *varying,
    OsdVertexBufferDescriptor const &vertexDesc,
    OsdVertexBufferDescriptor const &varyingDesc,
    const int *E_IT, int vertexOffset, int tableOffset, int start, int end) {
    if(vertexDesc == OsdVertexBufferDescriptor(0, 4, 4) && varying == NULL) {
        ComputeBilinearEdgeKernel<4>(vertex, E_IT, vertexOffset, tableOffset, 
                                     start, end);
    }
    else if(vertexDesc == OsdVertexBufferDescriptor(0, 8, 8) && varying == NULL) {
        ComputeBilinearEdgeKernel<8>(vertex, E_IT, vertexOffset, tableOffset, 
                                     start, end);      
    }
    else {
        float *vertexResults = (float*)alloca(vertexDesc.length * sizeof(float));
        float *varyingResults = (float*)alloca(varyingDesc.length * sizeof(float));

        for (int i = start + tableOffset; i < end + tableOffset; i++) {
            int eidx0 = E_IT[2*i+0];
            int eidx1 = E_IT[2*i+1];

            int dstIndex = i + vertexOffset - tableOffset;
            clear(vertexResults, vertexDesc);
            clear(varyingResults, varyingDesc);

            addWithWeight(vertexResults, vertex, eidx0, 0.5f, vertexDesc);
            addWithWeight(vertexResults, vertex, eidx1, 0.5f, vertexDesc);

            addWithWeight(varyingResults, varying, eidx0, 0.5f, varyingDesc);
            addWithWeight(varyingResults, varying, eidx1, 0.5f, varyingDesc);

            copy(vertex, vertexResults, dstIndex, vertexDesc);
            copy(varying, varyingResults, dstIndex, varyingDesc);
        }
    }
}

void OsdCpuComputeBilinearVertex(
    float *vertex, float *varying,
    OsdVertexBufferDescriptor const &vertexDesc,
    OsdVertexBufferDescriptor const &varyingDesc,
    const int *V_ITa, int vertexOffset, int tableOffset, int start, int end) {

    float *src, *des;
    for (int i = start + tableOffset; i < end + tableOffset; i++) {
        int p = V_ITa[i];

        int dstIndex = i + vertexOffset - tableOffset;
        if (vertex) {
            src = vertex + p        * vertexDesc.stride;
            des = vertex + dstIndex * vertexDesc.stride;
            memcpy(des, src, sizeof(float)*vertexDesc.length);
        }
        if (varying) {
            src = varying + p        * varyingDesc.stride;
            des = varying + dstIndex * varyingDesc.stride;
            memcpy(des, src, sizeof(float)*varyingDesc.length);
        }
    }
}

void OsdCpuEditVertexAdd(
    float *vertex,
    OsdVertexBufferDescriptor const &vertexDesc,
    int primVarOffset, int primVarWidth, int vertexOffset, int tableOffset,
    int start, int end,
    const unsigned int *editIndices, const float *editValues) {

    for (int i = start+tableOffset; i < end+tableOffset; i++) {

        if (vertex) {
            int editIndex = editIndices[i] + vertexOffset;
            float *dst = vertex + editIndex * vertexDesc.stride + primVarOffset;

            for (int j = 0; j < primVarWidth; ++j) {
                dst[j] += editValues[j];
            }
        }
    }
}

void OsdCpuEditVertexSet(
    float *vertex,
    OsdVertexBufferDescriptor const &vertexDesc,
    int primVarOffset, int primVarWidth, int vertexOffset, int tableOffset,
    int start, int end,
    const unsigned int *editIndices, const float *editValues) {

    for (int i = start+tableOffset; i < end+tableOffset; i++) {

        if (vertex) {
            int editIndex = editIndices[i] + vertexOffset;
            float *dst = vertex + editIndex * vertexDesc.stride + primVarOffset;

            for (int j = 0; j < primVarWidth; ++j) {
                dst[j] = editValues[j];
            }
        }
    }
}

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
