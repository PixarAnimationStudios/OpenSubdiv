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

#include "../osd/gcdKernel.h"
#include "../osd/cpuKernel.h"
#include "../osd/vertexDescriptor.h"

#include <math.h>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

const int GCD_WORK_STRIDE = 32;


void OsdGcdComputeFace(
    const OsdVertexDescriptor *vdesc, float * vertex, float * varying,
    const int *F_IT, const int *F_ITa, int offset, int start, int end,
    dispatch_queue_t gcdq) {

    const int workSize = end-start;
    dispatch_apply(workSize/GCD_WORK_STRIDE, gcdq, ^(size_t blockIdx){
        const int start_i = start + blockIdx*GCD_WORK_STRIDE;
        const int end_i = start_i + GCD_WORK_STRIDE;
        OsdCpuComputeFace(vdesc, vertex, varying, F_IT, F_ITa, offset,
            start_i, end_i);
    });
    const int start_e = end - workSize%GCD_WORK_STRIDE;
    const int end_e = end;
    if (start_e < end_e)
        OsdCpuComputeFace(vdesc, vertex, varying, F_IT, F_ITa, offset,
            start_e, end_e);
}

void OsdGcdComputeEdge(
    const OsdVertexDescriptor *vdesc, float *vertex, float *varying,
    const int *E_IT, const float *E_W, int offset, int start, int end,
    dispatch_queue_t gcdq) {

    const int workSize = end-start;
    dispatch_apply(workSize/GCD_WORK_STRIDE, gcdq, ^(size_t blockIdx){
        const int start_i = start + blockIdx*GCD_WORK_STRIDE;
        const int end_i = start_i + GCD_WORK_STRIDE;
        OsdCpuComputeEdge(vdesc, vertex, varying, E_IT, E_W, offset,
            start_i, end_i);
    });
    const int start_e = end - workSize%GCD_WORK_STRIDE;
    const int end_e = end;
    if (start_e < end_e)
        OsdCpuComputeEdge(vdesc, vertex, varying, E_IT, E_W, offset,
            start_e, end_e);
}

void OsdGcdComputeVertexA(
    const OsdVertexDescriptor *vdesc, float *vertex, float *varying,
    const int *V_ITa, const float *V_W,
    int offset, int start, int end, int pass,
    dispatch_queue_t gcdq) {

    const int workSize = end-start;
    dispatch_apply(workSize/GCD_WORK_STRIDE, gcdq, ^(size_t blockIdx){
        const int start_i = start + blockIdx*GCD_WORK_STRIDE;
        const int end_i = start_i + GCD_WORK_STRIDE;
        OsdCpuComputeVertexA(vdesc, vertex, varying, V_ITa, V_W, offset,
            start_i, end_i, pass);
    });
    const int start_e = end - workSize%GCD_WORK_STRIDE;
    const int end_e = end;
    if (start_e < end_e)
        OsdCpuComputeVertexA(vdesc, vertex, varying, V_ITa, V_W, offset,
            start_e, end_e, pass);
}

void OsdGcdComputeVertexB(
    const OsdVertexDescriptor *vdesc, float *vertex, float *varying,
    const int *V_ITa, const int *V_IT, const float *V_W,
    int offset, int start, int end,
    dispatch_queue_t gcdq) {

    const int workSize = end-start;
    dispatch_apply(workSize/GCD_WORK_STRIDE, gcdq, ^(size_t blockIdx){
        const int start_i = start + blockIdx*GCD_WORK_STRIDE;
        const int end_i = start_i + GCD_WORK_STRIDE;
        OsdCpuComputeVertexB(vdesc, vertex, varying, V_ITa, V_IT, V_W, offset,
            start_i, end_i);
    });
    const int start_e = end - workSize%GCD_WORK_STRIDE;
    const int end_e = end;
    if (start_e < end_e)
        OsdCpuComputeVertexB(vdesc, vertex, varying, V_ITa, V_IT, V_W, offset,
            start_e, end_e);
}

void OsdGcdComputeLoopVertexB(
    const OsdVertexDescriptor *vdesc, float *vertex, float *varying,
    const int *V_ITa, const int *V_IT, const float *V_W,
    int offset, int start, int end,
    dispatch_queue_t gcdq) {

    dispatch_apply(end-start, gcdq, ^(size_t blockIdx){
        int i = start+blockIdx;
        int h = V_ITa[5*i];
        int n = V_ITa[5*i+1];
        int p = V_ITa[5*i+2];

        float weight = V_W[i];
        float wp = 1.0f/static_cast<float>(n);
        float beta = 0.25f * cosf(static_cast<float>(M_PI) * 2.0f * wp) + 0.375f;
        beta = beta * beta;
        beta = (0.625f - beta) * wp;

        int dstIndex = offset + i;
        vdesc->Clear(vertex, varying, dstIndex);

        vdesc->AddWithWeight(vertex, dstIndex, p, weight * (1.0f - (beta * n)));

        for (int j = 0; j < n; ++j)
            vdesc->AddWithWeight(vertex, dstIndex, V_IT[h+j], weight * beta);

        vdesc->AddVaryingWithWeight(varying, dstIndex, p, 1.0f);
    });
}

void OsdGcdComputeBilinearEdge(
    const OsdVertexDescriptor *vdesc, float *vertex, float *varying,
    const int *E_IT, int offset, int start, int end,
    dispatch_queue_t gcdq) {

    dispatch_apply(end-start, gcdq, ^(size_t blockIdx){
        int i = start+blockIdx;
        int eidx0 = E_IT[2*i+0];
        int eidx1 = E_IT[2*i+1];

        int dstIndex = offset + i;
        vdesc->Clear(vertex, varying, dstIndex);

        vdesc->AddWithWeight(vertex, dstIndex, eidx0, 0.5f);
        vdesc->AddWithWeight(vertex, dstIndex, eidx1, 0.5f);

        vdesc->AddVaryingWithWeight(varying, dstIndex, eidx0, 0.5f);
        vdesc->AddVaryingWithWeight(varying, dstIndex, eidx1, 0.5f);
    });
}

void OsdGcdComputeBilinearVertex(
    const OsdVertexDescriptor *vdesc, float *vertex, float *varying,
    const int *V_ITa, int offset, int start, int end,
    dispatch_queue_t gcdq) {

    dispatch_apply(end-start, gcdq, ^(size_t blockIdx){
        int i = start+blockIdx;
        int p = V_ITa[i];

        int dstIndex = offset + i;
        vdesc->Clear(vertex, varying, dstIndex);

        vdesc->AddWithWeight(vertex, dstIndex, p, 1.0f);
        vdesc->AddVaryingWithWeight(varying, dstIndex, p, 1.0f);
    });
}

void OsdGcdEditVertexAdd(
    const OsdVertexDescriptor *vdesc, float *vertex,
    int primVarOffset, int primVarWidth, int vertexCount,
    const int *editIndices, const float *editValues,
    dispatch_queue_t gcdq) {

    dispatch_apply(vertexCount, gcdq, ^(size_t blockIdx){
        int i = blockIdx;
        vdesc->ApplyVertexEditAdd(vertex, primVarOffset, primVarWidth,
                                  editIndices[i], &editValues[i*primVarWidth]);
    });
}

void OsdGcdEditVertexSet(
    const OsdVertexDescriptor *vdesc, float *vertex,
    int primVarOffset, int primVarWidth, int vertexCount,
    const int *editIndices, const float *editValues,
    dispatch_queue_t gcdq) {

    dispatch_apply(vertexCount, gcdq, ^(size_t blockIdx){
        int i = blockIdx;
        vdesc->ApplyVertexEditSet(vertex, primVarOffset, primVarWidth,
                                  editIndices[i], &editValues[i*primVarWidth]);
    });
}


}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
