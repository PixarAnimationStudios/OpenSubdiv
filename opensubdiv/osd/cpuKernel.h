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
#ifndef OSD_CPU_KERNEL_H
#define OSD_CPU_KERNEL_H

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

struct VertexDescriptor {

    VertexDescriptor(int numVertexElem, int numVaryingElem) 
        : numVertexElements(numVertexElem), numVaryingElements(numVaryingElem) { }

    void Clear(float *vertex, float *varying) const {
        for (int i = 0; i < numVertexElements; ++i)
            vertex[i] = 0.0f;

        for (int i = 0; i < numVaryingElements; ++i)
            varying[i] = 0.0f;
    }
    void AddWithWeight(float *vertex, const float *src, float weight) const {
        for (int i = 0; i < numVertexElements; ++i)
            vertex[i] += src[i] * weight;
    }
    void AddVaryingWithWeight(float *varying, const float *src, float weight) const {
        for (int i = 0; i < numVaryingElements; ++i)
            varying[i] += src[i] * weight;
    }
    
    int numVertexElements;
    int numVaryingElements;
};

extern "C" {

void computeFace(const VertexDescriptor *vdesc, float * vertex, float * varying, const int *F_IT, const int *F_ITa, int offset, int start, int end);

void computeEdge(const VertexDescriptor *vdesc, float *vertex, float * varying, const int *E_IT, const float *E_W, int offset, int start, int end);

void computeVertexA(const VertexDescriptor *vdesc, float *vertex, float * varying, const int *V_ITa, const float *V_W, int offset, int start, int end, int pass);

void computeVertexB(const VertexDescriptor *vdesc, float *vertex, float * varying, const int *V_ITa, const int *V_IT, const float *V_W, int offset, int start, int end);

void computeLoopVertexB(const VertexDescriptor *vdesc, float *vertex, float * varying, const int *V_ITa, const int *V_IT, const float *V_W, int offset, int start, int end);

}

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* OSD_CPU_KERNEL_H */
