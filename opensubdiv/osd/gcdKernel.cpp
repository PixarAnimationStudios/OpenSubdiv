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

#include "../osd/gcdKernel.h"
#include "../osd/cpuKernel.h"
#include "../osd/vertexDescriptor.h"

#include <math.h>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

const int GCD_WORK_STRIDE = 32;

static inline void
clear(float *origin, int index, OsdVertexBufferDescriptor const &desc) {

    if (origin) {
        float *dst = origin + index * desc.stride + desc.offset;
        memset(dst, 0, desc.length * sizeof(float));
    }
}

static inline void
addWithWeight(float *origin, int dstIndex, int srcIndex,
              float weight, OsdVertexBufferDescriptor const &desc) {

    if (origin) {
        const float *src = origin + srcIndex * desc.stride + desc.offset;
        float *dst = origin + dstIndex * desc.stride + desc.offset;
        for (int k = 0; k < desc.length; ++k) {
            dst[k] += src[k] * weight;
        }
    }
}

void OsdGcdComputeFace(
    float * vertex, float * varying,
    OsdVertexBufferDescriptor const &vertexDesc,
    OsdVertexBufferDescriptor const &varyingDesc,
    const int *F_IT, const int *F_ITa,
    int vertexOffset, int tableOffset, int start, int end,
    dispatch_queue_t gcdq) {

    const int workSize = end-start;
    dispatch_apply(workSize/GCD_WORK_STRIDE, gcdq, ^(size_t blockIdx){
        const int start_i = start + blockIdx*GCD_WORK_STRIDE;
        const int end_i = start_i + GCD_WORK_STRIDE;
        OsdCpuComputeFace(vertex, varying, vertexDesc, varyingDesc,
                          F_IT, F_ITa,
                          vertexOffset, tableOffset, start_i, end_i);
    });
    const int start_e = end - workSize%GCD_WORK_STRIDE;
    const int end_e = end;
    if (start_e < end_e)
        OsdCpuComputeFace(vertex, varying, vertexDesc, varyingDesc,
                          F_IT, F_ITa,
                          vertexOffset, tableOffset, start_e, end_e);
}

void OsdGcdComputeQuadFace(
    float * vertex, float * varying,
    OsdVertexBufferDescriptor const &vertexDesc,
    OsdVertexBufferDescriptor const &varyingDesc,
    const int *F_IT,
    int vertexOffset, int tableOffset, int start, int end,
    dispatch_queue_t gcdq) {

    const int workSize = end-start;
    dispatch_apply(workSize/GCD_WORK_STRIDE, gcdq, ^(size_t blockIdx){
        const int start_i = start + blockIdx*GCD_WORK_STRIDE;
        const int end_i = start_i + GCD_WORK_STRIDE;
        OsdCpuComputeQuadFace(vertex, varying, vertexDesc, varyingDesc,
                              F_IT,
                              vertexOffset, tableOffset, start_i, end_i);
    });
    const int start_e = end - workSize%GCD_WORK_STRIDE;
    const int end_e = end;
    if (start_e < end_e)
        OsdCpuComputeQuadFace(vertex, varying, vertexDesc, varyingDesc,
                              F_IT,
                              vertexOffset, tableOffset, start_e, end_e);
}

void OsdGcdComputeTriQuadFace(
    float * vertex, float * varying,
    OsdVertexBufferDescriptor const &vertexDesc,
    OsdVertexBufferDescriptor const &varyingDesc,
    const int *F_IT,
    int vertexOffset, int tableOffset, int start, int end,
    dispatch_queue_t gcdq) {

    const int workSize = end-start;
    dispatch_apply(workSize/GCD_WORK_STRIDE, gcdq, ^(size_t blockIdx){
        const int start_i = start + blockIdx*GCD_WORK_STRIDE;
        const int end_i = start_i + GCD_WORK_STRIDE;
        OsdCpuComputeTriQuadFace(vertex, varying, vertexDesc, varyingDesc,
                                 F_IT,
                                 vertexOffset, tableOffset, start_i, end_i);
    });
    const int start_e = end - workSize%GCD_WORK_STRIDE;
    const int end_e = end;
    if (start_e < end_e)
        OsdCpuComputeTriQuadFace(vertex, varying, vertexDesc, varyingDesc,
                                 F_IT,
                                 vertexOffset, tableOffset, start_e, end_e);
}

void OsdGcdComputeEdge(
    float * vertex, float * varying,
    OsdVertexBufferDescriptor const &vertexDesc,
    OsdVertexBufferDescriptor const &varyingDesc,
    const int *E_IT, const float *E_W,
    int vertexOffset, int tableOffset, int start, int end,
    dispatch_queue_t gcdq) {

    const int workSize = end-start;
    dispatch_apply(workSize/GCD_WORK_STRIDE, gcdq, ^(size_t blockIdx){
        const int start_i = start + blockIdx*GCD_WORK_STRIDE;
        const int end_i = start_i + GCD_WORK_STRIDE;
        OsdCpuComputeEdge(vertex, varying, vertexDesc, varyingDesc,
                          E_IT, E_W,
                          vertexOffset, tableOffset, start_i, end_i);
    });
    const int start_e = end - workSize%GCD_WORK_STRIDE;
    const int end_e = end;
    if (start_e < end_e)
        OsdCpuComputeEdge(vertex, varying, vertexDesc, varyingDesc,
                          E_IT, E_W,
                          vertexOffset, tableOffset, start_e, end_e);
}

void OsdGcdComputeRestrictedEdge(
    float * vertex, float * varying,
    OsdVertexBufferDescriptor const &vertexDesc,
    OsdVertexBufferDescriptor const &varyingDesc,
    const int *E_IT,    int vertexOffset, int tableOffset, int start, int end,
    dispatch_queue_t gcdq) {

    const int workSize = end-start;
    dispatch_apply(workSize/GCD_WORK_STRIDE, gcdq, ^(size_t blockIdx){
        const int start_i = start + blockIdx*GCD_WORK_STRIDE;
        const int end_i = start_i + GCD_WORK_STRIDE;
        OsdCpuComputeRestrictedEdge(vertex, varying, vertexDesc, varyingDesc,
                                    E_IT,
                                    vertexOffset, tableOffset, start_i, end_i);
    });
    const int start_e = end - workSize%GCD_WORK_STRIDE;
    const int end_e = end;
    if (start_e < end_e)
        OsdCpuComputeRestrictedEdge(vertex, varying, vertexDesc, varyingDesc,
                                    E_IT,
                                    vertexOffset, tableOffset, start_e, end_e);
}

void OsdGcdComputeVertexA(
    float * vertex, float * varying,
    OsdVertexBufferDescriptor const &vertexDesc,
    OsdVertexBufferDescriptor const &varyingDesc,
    const int *V_ITa, const float *V_W,
    int vertexOffset, int tableOffset, int start, int end, int pass,
    dispatch_queue_t gcdq) {

    const int workSize = end-start;
    dispatch_apply(workSize/GCD_WORK_STRIDE, gcdq, ^(size_t blockIdx){
        const int start_i = start + blockIdx*GCD_WORK_STRIDE;
        const int end_i = start_i + GCD_WORK_STRIDE;
        OsdCpuComputeVertexA(vertex, varying, vertexDesc, varyingDesc,
                             V_ITa, V_W,
                             vertexOffset, tableOffset, start_i, end_i, pass);
    });
    const int start_e = end - workSize%GCD_WORK_STRIDE;
    const int end_e = end;
    if (start_e < end_e)
        OsdCpuComputeVertexA(vertex, varying, vertexDesc, varyingDesc,
                             V_ITa, V_W,
                             vertexOffset, tableOffset, start_e, end_e, pass);
}

void OsdGcdComputeVertexB(
    float * vertex, float * varying,
    OsdVertexBufferDescriptor const &vertexDesc,
    OsdVertexBufferDescriptor const &varyingDesc,
    const int *V_ITa, const int *V_IT, const float *V_W,
    int vertexOffset, int tableOffset, int start, int end,
    dispatch_queue_t gcdq) {

    const int workSize = end-start;
    dispatch_apply(workSize/GCD_WORK_STRIDE, gcdq, ^(size_t blockIdx){
        const int start_i = start + blockIdx*GCD_WORK_STRIDE;
        const int end_i = start_i + GCD_WORK_STRIDE;
        OsdCpuComputeVertexB(vertex, varying, vertexDesc, varyingDesc,
                             V_ITa, V_IT, V_W,
                             vertexOffset, tableOffset, start_i, end_i);
    });
    const int start_e = end - workSize%GCD_WORK_STRIDE;
    const int end_e = end;
    if (start_e < end_e)
        OsdCpuComputeVertexB(vertex, varying, vertexDesc, varyingDesc,
                             V_ITa, V_IT, V_W,
                             vertexOffset, tableOffset, start_e, end_e);
}

void OsdGcdComputeRestrictedVertexA(
    float * vertex, float * varying,
    OsdVertexBufferDescriptor const &vertexDesc,
    OsdVertexBufferDescriptor const &varyingDesc,
    const int *V_ITa,
    int vertexOffset, int tableOffset, int start, int end,
    dispatch_queue_t gcdq) {

    const int workSize = end-start;
    dispatch_apply(workSize/GCD_WORK_STRIDE, gcdq, ^(size_t blockIdx){
        const int start_i = start + blockIdx*GCD_WORK_STRIDE;
        const int end_i = start_i + GCD_WORK_STRIDE;
        OsdCpuComputeRestrictedVertexA(vertex, varying, vertexDesc, varyingDesc,
                                       V_ITa,
                                       vertexOffset, tableOffset, start_i, end_i);
    });
    const int start_e = end - workSize%GCD_WORK_STRIDE;
    const int end_e = end;
    if (start_e < end_e)
        OsdCpuComputeRestrictedVertexA(vertex, varying, vertexDesc, varyingDesc,
                                       V_ITa,
                                       vertexOffset, tableOffset, start_e, end_e);
}

void OsdGcdComputeRestrictedVertexB1(
    float * vertex, float * varying,
    OsdVertexBufferDescriptor const &vertexDesc,
    OsdVertexBufferDescriptor const &varyingDesc,
    const int *V_ITa, const int *V_IT,
    int vertexOffset, int tableOffset, int start, int end,
    dispatch_queue_t gcdq) {

    const int workSize = end-start;
    dispatch_apply(workSize/GCD_WORK_STRIDE, gcdq, ^(size_t blockIdx){
        const int start_i = start + blockIdx*GCD_WORK_STRIDE;
        const int end_i = start_i + GCD_WORK_STRIDE;
        OsdCpuComputeRestrictedVertexB1(vertex, varying, vertexDesc, varyingDesc,
                                        V_ITa, V_IT,
                                        vertexOffset, tableOffset, start_i, end_i);
    });
    const int start_e = end - workSize%GCD_WORK_STRIDE;
    const int end_e = end;
    if (start_e < end_e)
        OsdCpuComputeRestrictedVertexB1(vertex, varying, vertexDesc, varyingDesc,
                                        V_ITa, V_IT,
                                        vertexOffset, tableOffset, start_e, end_e);
}

void OsdGcdComputeRestrictedVertexB2(
    float * vertex, float * varying,
    OsdVertexBufferDescriptor const &vertexDesc,
    OsdVertexBufferDescriptor const &varyingDesc,
    const int *V_ITa, const int *V_IT,
    int vertexOffset, int tableOffset, int start, int end,
    dispatch_queue_t gcdq) {

    const int workSize = end-start;
    dispatch_apply(workSize/GCD_WORK_STRIDE, gcdq, ^(size_t blockIdx){
        const int start_i = start + blockIdx*GCD_WORK_STRIDE;
        const int end_i = start_i + GCD_WORK_STRIDE;
        OsdCpuComputeRestrictedVertexB2(vertex, varying, vertexDesc, varyingDesc,
                                        V_ITa, V_IT,
                                        vertexOffset, tableOffset, start_i, end_i);
    });
    const int start_e = end - workSize%GCD_WORK_STRIDE;
    const int end_e = end;
    if (start_e < end_e)
        OsdCpuComputeRestrictedVertexB2(vertex, varying, vertexDesc, varyingDesc,
                                        V_ITa, V_IT,
                                        vertexOffset, tableOffset, start_e, end_e);
}

void OsdGcdComputeLoopVertexB(
    float * vertex, float * varying,
    OsdVertexBufferDescriptor const &vertexDesc,
    OsdVertexBufferDescriptor const &varyingDesc,
    const int *V_ITa, const int *V_IT, const float *V_W,
    int vertexOffset, int tableOffset, int start, int end,
    dispatch_queue_t gcdq) {

    dispatch_apply(end-start, gcdq, ^(size_t blockIdx){
        int i = start+blockIdx+tableOffset;
        int h = V_ITa[5*i];
        int n = V_ITa[5*i+1];
        int p = V_ITa[5*i+2];

        float weight = V_W[i];
        float wp = 1.0f/static_cast<float>(n);
        float beta = 0.25f * cosf(static_cast<float>(M_PI) * 2.0f * wp) + 0.375f;
        beta = beta * beta;
        beta = (0.625f - beta) * wp;

        int dstIndex = vertexOffset + i - tableOffset;
        clear(vertex, dstIndex, vertexDesc);
        clear(varying, dstIndex, varyingDesc);

        addWithWeight(vertex, dstIndex, p, weight * (1.0f - (beta * n)), vertexDesc);

        for (int j = 0; j < n; ++j)
            addWithWeight(vertex, dstIndex, V_IT[h+j], weight * beta, vertexDesc);

        addWithWeight(varying, dstIndex, p, 1.0f, varyingDesc);
    });
}

void OsdGcdComputeBilinearEdge(
    float * vertex, float * varying,
    OsdVertexBufferDescriptor const &vertexDesc,
    OsdVertexBufferDescriptor const &varyingDesc,
    const int *E_IT,
    int vertexOffset, int tableOffset, int start, int end,
    dispatch_queue_t gcdq) {

    dispatch_apply(end-start, gcdq, ^(size_t blockIdx){
        int i = start+blockIdx+tableOffset;
        int eidx0 = E_IT[2*i+0];
        int eidx1 = E_IT[2*i+1];

        int dstIndex = vertexOffset + i - tableOffset;
        clear(vertex, dstIndex, vertexDesc);
        clear(varying, dstIndex, varyingDesc);

        addWithWeight(vertex, dstIndex, eidx0, 0.5f, vertexDesc);
        addWithWeight(vertex, dstIndex, eidx1, 0.5f, vertexDesc);

        addWithWeight(varying, dstIndex, eidx0, 0.5f, varyingDesc);
        addWithWeight(varying, dstIndex, eidx1, 0.5f, varyingDesc);
    });
}

void OsdGcdComputeBilinearVertex(
    float * vertex, float * varying,
    OsdVertexBufferDescriptor const &vertexDesc,
    OsdVertexBufferDescriptor const &varyingDesc,
    const int *V_ITa,
    int vertexOffset, int tableOffset, int start, int end,
    dispatch_queue_t gcdq) {

    dispatch_apply(end-start, gcdq, ^(size_t blockIdx){
        int i = start+blockIdx+tableOffset;
        int p = V_ITa[i];

        int dstIndex = vertexOffset + i - tableOffset;
        clear(vertex, dstIndex, vertexDesc);
        clear(varying, dstIndex, varyingDesc);

        addWithWeight(vertex, dstIndex, p, 1.0f, vertexDesc);
        addWithWeight(varying, dstIndex, p, 1.0f, varyingDesc);
    });
}

void OsdGcdEditVertexAdd(
    float * vertex,
    OsdVertexBufferDescriptor const &vertexDesc,
    int primVarOffset, int /*primVarWidth*/,
    int vertexOffset, int tableOffset,
    int start, int end,
    const unsigned int *editIndices, const float *editValues,
    dispatch_queue_t gcdq) {

    int vertexCount = end - start;
    dispatch_apply(vertexCount, gcdq, ^(size_t blockIdx){
        int i = start + blockIdx + tableOffset;

        if (vertex) {
            int editIndex = editIndices[i] + vertexOffset;
            float *dst = vertex + editIndex * vertexDesc.stride
                + vertexDesc.offset + primVarOffset;

            dst[i] += editValues[i];
        }
    });
}

void OsdGcdEditVertexSet(
    float * vertex,
    OsdVertexBufferDescriptor const &vertexDesc,
    int primVarOffset, int /*primVarWidth*/,
    int vertexOffset, int tableOffset,
    int start, int end,
    const unsigned int *editIndices, const float *editValues,
    dispatch_queue_t gcdq) {

    int vertexCount = end - start;
    dispatch_apply(vertexCount, gcdq, ^(size_t blockIdx){
        int i = start + blockIdx + tableOffset;

        if (vertex) {
            int editIndex = editIndices[i] + vertexOffset;
            float *dst = vertex + editIndex * vertexDesc.stride
                + vertexDesc.offset + primVarOffset;

            dst[i] = editValues[i];
        }
    });
}


}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
