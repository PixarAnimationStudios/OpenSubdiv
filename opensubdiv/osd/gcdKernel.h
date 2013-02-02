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
#ifndef OSD_GCD_KERNEL_H
#define OSD_GCD_KERNEL_H

#include <dispatch/dispatch.h>
#include "../version.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

struct OsdVertexDescriptor;

void OsdGcdComputeFace(const OsdVertexDescriptor *vdesc,
                       float * vertex, float * varying,
                       const int *F_IT, const int *F_ITa,
                       int offset, int start, int end,
                       dispatch_queue_t gcdq);

void OsdGcdComputeEdge(const OsdVertexDescriptor *vdesc,
                       float *vertex, float * varying,
                       const int *E_IT, const float *E_ITa,
                       int offset, int start, int end,
                       dispatch_queue_t gcdq);

void OsdGcdComputeVertexA(const OsdVertexDescriptor *vdesc,
                          float *vertex, float * varying,
                          const int *V_ITa, const float *V_IT,
                          int offset, int start, int end, int pass,
                          dispatch_queue_t gcdq);

void OsdGcdComputeVertexB(const OsdVertexDescriptor *vdesc,
                          float *vertex, float * varying,
                          const int *V_ITa, const int *V_IT, const float *V_W,
                          int offset, int start, int end,
                          dispatch_queue_t gcdq);

void OsdGcdComputeLoopVertexB(const OsdVertexDescriptor *vdesc,
                              float *vertex, float * varying,
                              const int *V_ITa, const int *V_IT,
                              const float *V_W,
                              int offset, int start, int end,
                              dispatch_queue_t gcdq);

void OsdGcdComputeBilinearEdge(const OsdVertexDescriptor *vdesc,
                               float *vertex, float * varying,
                               const int *E_IT,
                               int offset, int start, int end,
                               dispatch_queue_t gcdq);

void OsdGcdComputeBilinearVertex(const OsdVertexDescriptor *vdesc,
                                 float *vertex, float * varying,
                                 const int *V_ITa,
                                 int offset, int start, int end,
                                 dispatch_queue_t gcdq);

void OsdGcdEditVertexAdd(const OsdVertexDescriptor *vdesc, float *vertex,
                         int primVarOffset, int primVarWidth, int count,
                         const int *editIndices, const float *editValues,
                         dispatch_queue_t gcdq);

void OsdGcdEditVertexSet(const OsdVertexDescriptor *vdesc, float *vertex,
                         int primVarOffset, int primVarWidth, int count,
                         const int *editIndices, const float *editValues,
                         dispatch_queue_t gcdq);

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OSD_GCD_KERNEL_H
