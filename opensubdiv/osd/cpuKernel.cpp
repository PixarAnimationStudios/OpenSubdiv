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

#include "../osd/cpuKernel.h"
#include "../osd/vertexDescriptor.h"

#include <math.h>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

void OsdCpuComputeFace(
    OsdVertexDescriptor const &vdesc, float * vertex, float * varying,
    const int *F_IT, const int *F_ITa, int vertexOffset, int tableOffset,
    int start, int end) {

    for (int i = start + tableOffset; i < end + tableOffset; i++) {
        int h = F_ITa[2*i];
        int n = F_ITa[2*i+1];

        float weight = 1.0f/n;

        // XXX: should use local vertex struct variable instead of
        // accumulating directly into global memory.
        int dstIndex = i + vertexOffset - tableOffset;
        vdesc.Clear(vertex, varying, dstIndex);

        for (int j = 0; j < n; ++j) {
            int index = F_IT[h+j];
            vdesc.AddWithWeight(vertex, dstIndex, index, weight);
            vdesc.AddVaryingWithWeight(varying, dstIndex, index, weight);
        }
    }

}

void OsdCpuComputeEdge(
    OsdVertexDescriptor const &vdesc, float *vertex, float *varying,
    const int *E_IT, const float *E_W, int vertexOffset, int tableOffset,
    int start, int end) {

    for (int i = start + tableOffset; i < end + tableOffset; i++) {
        int eidx0 = E_IT[4*i+0];
        int eidx1 = E_IT[4*i+1];
        int eidx2 = E_IT[4*i+2];
        int eidx3 = E_IT[4*i+3];

        float vertWeight = E_W[i*2+0];

        int dstIndex = i + vertexOffset - tableOffset;
        vdesc.Clear(vertex, varying, dstIndex);

        vdesc.AddWithWeight(vertex, dstIndex, eidx0, vertWeight);
        vdesc.AddWithWeight(vertex, dstIndex, eidx1, vertWeight);

        if (eidx2 != -1) {
            float faceWeight = E_W[i*2+1];

            vdesc.AddWithWeight(vertex, dstIndex, eidx2, faceWeight);
            vdesc.AddWithWeight(vertex, dstIndex, eidx3, faceWeight);
        }

        vdesc.AddVaryingWithWeight(varying, dstIndex, eidx0, 0.5f);
        vdesc.AddVaryingWithWeight(varying, dstIndex, eidx1, 0.5f);
    }
}

void OsdCpuComputeVertexA(
    OsdVertexDescriptor const &vdesc, float *vertex, float *varying,
    const int *V_ITa, const float *V_W, int vertexOffset, int tableOffset,
    int start, int end, int pass) {

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

        if (not pass)
            vdesc.Clear(vertex, varying, dstIndex);

        if (eidx0 == -1 || (pass == 0 && (n == -1))) {
            vdesc.AddWithWeight(vertex, dstIndex, p, weight);
        } else {
            vdesc.AddWithWeight(vertex, dstIndex, p, weight * 0.75f);
            vdesc.AddWithWeight(vertex, dstIndex, eidx0, weight * 0.125f);
            vdesc.AddWithWeight(vertex, dstIndex, eidx1, weight * 0.125f);
        }

        if (not pass)
            vdesc.AddVaryingWithWeight(varying, dstIndex, p, 1.0f);
    }
}

void OsdCpuComputeVertexB(
    OsdVertexDescriptor const &vdesc, float *vertex, float *varying,
    const int *V_ITa, const int *V_IT, const float *V_W,
    int vertexOffset, int tableOffset, int start, int end) {

    for (int i = start + tableOffset; i < end + tableOffset; i++) {
        int h = V_ITa[5*i];
        int n = V_ITa[5*i+1];
        int p = V_ITa[5*i+2];

        float weight = V_W[i];
        float wp = 1.0f/static_cast<float>(n*n);
        float wv = (n-2.0f) * n * wp;

        int dstIndex = i + vertexOffset - tableOffset;
        vdesc.Clear(vertex, varying, dstIndex);

        vdesc.AddWithWeight(vertex, dstIndex, p, weight * wv);

        for (int j = 0; j < n; ++j) {
            vdesc.AddWithWeight(vertex, dstIndex, V_IT[h+j*2], weight * wp);
            vdesc.AddWithWeight(vertex, dstIndex, V_IT[h+j*2+1], weight * wp);
        }
        vdesc.AddVaryingWithWeight(varying, dstIndex, p, 1.0f);
    }
}

void OsdCpuComputeLoopVertexB(
    OsdVertexDescriptor const &vdesc, float *vertex, float *varying,
    const int *V_ITa, const int *V_IT, const float *V_W,
    int vertexOffset, int tableOffset, int start, int end) {

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
        vdesc.Clear(vertex, varying, dstIndex);

        vdesc.AddWithWeight(vertex, dstIndex, p, weight * (1.0f - (beta * n)));

        for (int j = 0; j < n; ++j)
            vdesc.AddWithWeight(vertex, dstIndex, V_IT[h+j], weight * beta);

        vdesc.AddVaryingWithWeight(varying, dstIndex, p, 1.0f);
    }
}

void OsdCpuComputeBilinearEdge(
    OsdVertexDescriptor const &vdesc, float *vertex, float *varying,
    const int *E_IT, int vertexOffset, int tableOffset, int start, int end) {

    for (int i = start + tableOffset; i < end + tableOffset; i++) {
        int eidx0 = E_IT[2*i+0];
        int eidx1 = E_IT[2*i+1];

        int dstIndex = i + vertexOffset - tableOffset;
        vdesc.Clear(vertex, varying, dstIndex);

        vdesc.AddWithWeight(vertex, dstIndex, eidx0, 0.5f);
        vdesc.AddWithWeight(vertex, dstIndex, eidx1, 0.5f);

        vdesc.AddVaryingWithWeight(varying, dstIndex, eidx0, 0.5f);
        vdesc.AddVaryingWithWeight(varying, dstIndex, eidx1, 0.5f);
    }
}

void OsdCpuComputeBilinearVertex(
    OsdVertexDescriptor const &vdesc, float *vertex, float *varying,
    const int *V_ITa, int vertexOffset, int tableOffset, int start, int end) {

    for (int i = start + tableOffset; i < end + tableOffset; i++) {
        int p = V_ITa[i];

        int dstIndex = i + vertexOffset - tableOffset;
        vdesc.Clear(vertex, varying, dstIndex);

        vdesc.AddWithWeight(vertex, dstIndex, p, 1.0f);
        vdesc.AddVaryingWithWeight(varying, dstIndex, p, 1.0f);
    }
}

void OsdCpuEditVertexAdd(
    OsdVertexDescriptor const &vdesc, float *vertex,
    int primVarOffset, int primVarWidth, int vertexOffset, int tableOffset,
    int start, int end,
    const unsigned int *editIndices, const float *editValues) {

    for (int i = start+tableOffset; i < end+tableOffset; i++) {
        vdesc.ApplyVertexEditAdd(vertex,
                                 primVarOffset,
                                 primVarWidth,
                                 editIndices[i] + vertexOffset,
                                 &editValues[i*primVarWidth]);
    }
}

void OsdCpuEditVertexSet(
    OsdVertexDescriptor const &vdesc, float *vertex,
    int primVarOffset, int primVarWidth, int vertexOffset, int tableOffset,
    int start, int end,
    const unsigned int *editIndices, const float *editValues) {

    for (int i = start+tableOffset; i < end+tableOffset; i++) {
        vdesc.ApplyVertexEditSet(vertex,
                                 primVarOffset,
                                 primVarWidth,
                                 editIndices[i] + vertexOffset,
                                 &editValues[i*primVarWidth]);
    }
}

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
