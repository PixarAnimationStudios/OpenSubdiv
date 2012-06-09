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
#include <math.h>

#include "../osd/cpuKernel.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

extern void computeFace( const VertexDescriptor *vdesc, float * vertex, float * varying, const int *F_IT, const int *F_ITa, int offset, int start, int end) {
    
    int ve = vdesc->numVertexElements;
    int vev = vdesc->numVaryingElements;
    
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = start; i < end; i++) {
        int h = F_ITa[2*i];
        int n = F_ITa[2*i+1];

        float weight = 1.0f/n;

        // XXX: should use local vertex struct variable instead of accumulating directly into global memory.
        float *dst = &vertex[(offset + i)*ve];
        float *dstVarying = &varying[(offset + i)*vev];
        vdesc->Clear(dst, dstVarying);

        for (int j=0; j<n; ++j) {
            int index = F_IT[h+j];
            vdesc->AddWithWeight(dst, &vertex[index*ve], weight);
            vdesc->AddVaryingWithWeight(dstVarying, &varying[index*vev], weight);
        }
    }
}

extern void computeEdge( const VertexDescriptor *vdesc, float *vertex, float *varying, const int *E_IT, const float *E_W, int offset, int start, int end) {

    int ve = vdesc->numVertexElements;
    int vev = vdesc->numVaryingElements;

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = start; i < end; i++) {
        int eidx0 = E_IT[4*i+0];
        int eidx1 = E_IT[4*i+1];
        int eidx2 = E_IT[4*i+2]; 
        int eidx3 = E_IT[4*i+3];
        
        float vertWeight = E_W[i*2+0];
        
        float *dst = &vertex[(offset+i)*ve];
        float *dstVarying = &varying[(offset+i)*vev];
        vdesc->Clear(dst, dstVarying);
        
        vdesc->AddWithWeight(dst, &vertex[eidx0*ve], vertWeight);
        vdesc->AddWithWeight(dst, &vertex[eidx1*ve], vertWeight);
        
        if (eidx2 != -1) {
            float faceWeight = E_W[i*2+1];
            
            vdesc->AddWithWeight(dst, &vertex[eidx2*ve], faceWeight);
            vdesc->AddWithWeight(dst, &vertex[eidx3*ve], faceWeight);
        }

        vdesc->AddVaryingWithWeight(dstVarying, &varying[eidx0*vev], 0.5f);
        vdesc->AddVaryingWithWeight(dstVarying, &varying[eidx1*vev], 0.5f);
    }
}

extern void computeVertexA(const VertexDescriptor *vdesc, float *vertex, float *varying, const int *V_ITa, const float *V_W, int offset, int start, int end, int pass) {

    int ve = vdesc->numVertexElements;
    int vev = vdesc->numVaryingElements;

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = start; i < end; i++) {
        int n     = V_ITa[5*i+1];
        int p     = V_ITa[5*i+2];
        int eidx0 = V_ITa[5*i+3];
        int eidx1 = V_ITa[5*i+4];
        
        float weight = (pass==1) ? V_W[i] : 1.0f - V_W[i];
        
        // In the case of fractional weight, the weight must be inverted since 
        // the value is shared with the k_Smooth kernel (statistically the 
        // k_Smooth kernel runs much more often than this one)
        if (weight>0.0f && weight<1.0f && n > 0)
            weight=1.0f-weight;
        
        float *dst = &vertex[(offset+i)*ve];
        float *dstVarying = &varying[(offset+i)*vev];
        if(not pass)
            vdesc->Clear(dst, dstVarying);
        
        if (eidx0==-1 || (pass==0 && (n==-1)) ) {
            vdesc->AddWithWeight(dst, &vertex[p*ve], weight);
        } else {
            vdesc->AddWithWeight(dst, &vertex[p*ve], weight * 0.75f);
            vdesc->AddWithWeight(dst, &vertex[eidx0*ve], weight * 0.125f);
            vdesc->AddWithWeight(dst, &vertex[eidx1*ve], weight * 0.125f);
        }

        if (not pass)
            vdesc->AddVaryingWithWeight(dstVarying, &varying[p*vev], 1.0);
    }
}

extern void computeVertexB(const VertexDescriptor *vdesc, float *vertex, float *varying, const int *V_ITa, const int *V_IT, const float *V_W, int offset, int start, int end) {

    int ve = vdesc->numVertexElements;
    int vev = vdesc->numVaryingElements;

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = start; i < end; i++) {
        int h = V_ITa[5*i];
        int n = V_ITa[5*i+1];
        int p = V_ITa[5*i+2];
        
        float weight = V_W[i];
        float wp = 1.0f/float(n*n);
        float wv = (n-2.0f) * n * wp;
        
        float *dst = &vertex[(offset+i)*ve];
        float *dstVarying = &varying[(offset+i)*vev];
        vdesc->Clear(dst, dstVarying);
        
        vdesc->AddWithWeight(dst, &vertex[p*ve], weight * wv);

        for (int j = 0; j < n; ++j) {
            vdesc->AddWithWeight(dst, &vertex[V_IT[h+j*2]*ve], weight * wp);
            vdesc->AddWithWeight(dst, &vertex[V_IT[h+j*2+1]*ve], weight * wp);
        }
        vdesc->AddVaryingWithWeight(dstVarying, &varying[p*vev], 1.0);
    }
}

extern void computeLoopVertexB(const VertexDescriptor *vdesc, float *vertex, float *varying, const int *V_ITa, const int *V_IT, const float *V_W, int offset, int start, int end) {

    int ve = vdesc->numVertexElements;
    int vev = vdesc->numVaryingElements;

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = start; i < end; i++) {
        int h = V_ITa[5*i];
        int n = V_ITa[5*i+1];
        int p = V_ITa[5*i+2];
        
        float weight = V_W[i];
        float wp = 1.0f/float(n);
        float beta = 0.25f * cosf(float(M_PI) * 2.0f * wp) + 0.375f;
        beta = beta * beta;
        beta = (0.625f - beta) * wp;
        
        float *dst = &vertex[(offset+i)*ve];
        float *dstVarying = &varying[(offset+i)*vev];
        vdesc->Clear(dst, dstVarying);
        
        vdesc->AddWithWeight(dst, &vertex[p*ve], weight * (1.0f - (beta * n)));

        for (int j = 0; j < n; ++j)
            vdesc->AddWithWeight(dst, &vertex[V_IT[h+j]*ve], weight * beta);

        vdesc->AddVaryingWithWeight(dstVarying, &varying[p*vev], 1.0f);
    }
}

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv
