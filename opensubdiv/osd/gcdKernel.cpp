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

#include "../osd/gcdKernel.h"
#include "../osd/cpuKernel.h"
#include "../osd/vertexDescriptor.h"

#include <math.h>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

const int GCD_WORK_STRIDE = 32;


void OsdGcdComputeFace(
    OsdVertexDescriptor const &vdesc, float * vertex, float * varying,
    const int *F_IT, const int *F_ITa,
    int vertexOffset, int tableOffset, int start, int end,
    dispatch_queue_t gcdq) {

    const int workSize = end-start;
    dispatch_apply(workSize/GCD_WORK_STRIDE, gcdq, ^(size_t blockIdx){
        const int start_i = start + blockIdx*GCD_WORK_STRIDE;
        const int end_i = start_i + GCD_WORK_STRIDE;
        OsdCpuComputeFace(vdesc, vertex, varying, F_IT, F_ITa,
                          vertexOffset, tableOffset, start_i, end_i);
    });
    const int start_e = end - workSize%GCD_WORK_STRIDE;
    const int end_e = end;
    if (start_e < end_e)
        OsdCpuComputeFace(vdesc, vertex, varying, F_IT, F_ITa,
                          vertexOffset, tableOffset, start_e, end_e);
}

void OsdGcdComputeEdge(
    OsdVertexDescriptor const &vdesc, float * vertex, float * varying,
    const int *E_IT, const float *E_W,
    int vertexOffset, int tableOffset, int start, int end,
    dispatch_queue_t gcdq) {

    const int workSize = end-start;
    dispatch_apply(workSize/GCD_WORK_STRIDE, gcdq, ^(size_t blockIdx){
        const int start_i = start + blockIdx*GCD_WORK_STRIDE;
        const int end_i = start_i + GCD_WORK_STRIDE;
        OsdCpuComputeEdge(vdesc, vertex, varying, E_IT, E_W,
                          vertexOffset, tableOffset, start_i, end_i);
    });
    const int start_e = end - workSize%GCD_WORK_STRIDE;
    const int end_e = end;
    if (start_e < end_e)
        OsdCpuComputeEdge(vdesc, vertex, varying, E_IT, E_W,
                          vertexOffset, tableOffset, start_e, end_e);
}

void OsdGcdComputeVertexA(
    OsdVertexDescriptor const &vdesc, float * vertex, float * varying,
    const int *V_ITa, const float *V_W,
    int vertexOffset, int tableOffset, int start, int end, int pass,
    dispatch_queue_t gcdq) {

    const int workSize = end-start;
    dispatch_apply(workSize/GCD_WORK_STRIDE, gcdq, ^(size_t blockIdx){
        const int start_i = start + blockIdx*GCD_WORK_STRIDE;
        const int end_i = start_i + GCD_WORK_STRIDE;
        OsdCpuComputeVertexA(vdesc, vertex, varying, V_ITa, V_W,
                             vertexOffset, tableOffset, start_i, end_i, pass);
    });
    const int start_e = end - workSize%GCD_WORK_STRIDE;
    const int end_e = end;
    if (start_e < end_e)
        OsdCpuComputeVertexA(vdesc, vertex, varying, V_ITa, V_W,
                             vertexOffset, tableOffset, start_e, end_e, pass);
}

void OsdGcdComputeVertexB(
    OsdVertexDescriptor const &vdesc, float * vertex, float * varying,
    const int *V_ITa, const int *V_IT, const float *V_W,
    int vertexOffset, int tableOffset, int start, int end,
    dispatch_queue_t gcdq) {

    const int workSize = end-start;
    dispatch_apply(workSize/GCD_WORK_STRIDE, gcdq, ^(size_t blockIdx){
        const int start_i = start + blockIdx*GCD_WORK_STRIDE;
        const int end_i = start_i + GCD_WORK_STRIDE;
        OsdCpuComputeVertexB(vdesc, vertex, varying, V_ITa, V_IT, V_W,
                             vertexOffset, tableOffset, start_i, end_i);
    });
    const int start_e = end - workSize%GCD_WORK_STRIDE;
    const int end_e = end;
    if (start_e < end_e)
        OsdCpuComputeVertexB(vdesc, vertex, varying, V_ITa, V_IT, V_W,
                             vertexOffset, tableOffset, start_e, end_e);
}

void OsdGcdComputeLoopVertexB(
    OsdVertexDescriptor const &vdesc, float * vertex, float * varying,
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
        vdesc.Clear(vertex, varying, dstIndex);

        vdesc.AddWithWeight(vertex, dstIndex, p, weight * (1.0f - (beta * n)));

        for (int j = 0; j < n; ++j)
            vdesc.AddWithWeight(vertex, dstIndex, V_IT[h+j], weight * beta);

        vdesc.AddVaryingWithWeight(varying, dstIndex, p, 1.0f);
    });
}

void OsdGcdComputeBilinearEdge(
    OsdVertexDescriptor const &vdesc, float * vertex, float * varying,
    const int *E_IT,
    int vertexOffset, int tableOffset, int start, int end,
    dispatch_queue_t gcdq) {

    dispatch_apply(end-start, gcdq, ^(size_t blockIdx){
        int i = start+blockIdx+tableOffset;
        int eidx0 = E_IT[2*i+0];
        int eidx1 = E_IT[2*i+1];

        int dstIndex = vertexOffset + i - tableOffset;
        vdesc.Clear(vertex, varying, dstIndex);

        vdesc.AddWithWeight(vertex, dstIndex, eidx0, 0.5f);
        vdesc.AddWithWeight(vertex, dstIndex, eidx1, 0.5f);

        vdesc.AddVaryingWithWeight(varying, dstIndex, eidx0, 0.5f);
        vdesc.AddVaryingWithWeight(varying, dstIndex, eidx1, 0.5f);
    });
}

void OsdGcdComputeBilinearVertex(
    OsdVertexDescriptor const &vdesc, float * vertex, float * varying,
    const int *V_ITa,
    int vertexOffset, int tableOffset, int start, int end,
    dispatch_queue_t gcdq) {

    dispatch_apply(end-start, gcdq, ^(size_t blockIdx){
        int i = start+blockIdx+tableOffset;
        int p = V_ITa[i];

        int dstIndex = vertexOffset + i - tableOffset;
        vdesc.Clear(vertex, varying, dstIndex);

        vdesc.AddWithWeight(vertex, dstIndex, p, 1.0f);
        vdesc.AddVaryingWithWeight(varying, dstIndex, p, 1.0f);
    });
}

void OsdGcdEditVertexAdd(
    OsdVertexDescriptor const &vdesc, float * vertex,
    int primVarOffset, int primVarWidth,
    int vertexOffset, int tableOffset,
    int start, int end,
    const unsigned int *editIndices, const float *editValues,
    dispatch_queue_t gcdq) {

    int vertexCount = end - start;
    dispatch_apply(vertexCount, gcdq, ^(size_t blockIdx){
        int i = start + blockIdx + tableOffset;
        vdesc.ApplyVertexEditAdd(vertex, primVarOffset, primVarWidth,
                                  editIndices[i] + vertexOffset,
                                  &editValues[i*primVarWidth]);
    });
}

void OsdGcdEditVertexSet(
    OsdVertexDescriptor const &vdesc, float * vertex,
    int primVarOffset, int primVarWidth,
    int vertexOffset, int tableOffset,
    int start, int end,
    const unsigned int *editIndices, const float *editValues,
    dispatch_queue_t gcdq) {

    int vertexCount = end - start;
    dispatch_apply(vertexCount, gcdq, ^(size_t blockIdx){
        int i = start + blockIdx + tableOffset;
        vdesc.ApplyVertexEditSet(vertex, primVarOffset, primVarWidth,
                                  editIndices[i] + vertexOffset,
                                  &editValues[i*primVarWidth]);
    });
}


}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
