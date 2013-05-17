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
#ifndef OSD_OMP_KERNEL_H
#define OSD_OMP_KERNEL_H

#include "../version.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

struct OsdVertexDescriptor;

void OsdOmpComputeFace(OsdVertexDescriptor const &vdesc,
                       float * vertex, float * varying,
                       const int *F_IT, const int *F_ITa,
                       int vertexOffset, int tableOffset,
                       int start, int end);

void OsdOmpComputeEdge(OsdVertexDescriptor const &vdesc,
                       float *vertex, float * varying,
                       const int *E_IT, const float *E_ITa,
                       int vertexOffset, int tableOffset,
                       int start, int end);

void OsdOmpComputeVertexA(OsdVertexDescriptor const &vdesc,
                          float *vertex, float * varying,
                          const int *V_ITa, const float *V_IT,
                          int vertexOffset, int tableOffset,
                          int start, int end, int pass);

void OsdOmpComputeVertexB(OsdVertexDescriptor const &vdesc,
                          float *vertex, float * varying,
                          const int *V_ITa, const int *V_IT, const float *V_W,
                          int vertexOffset, int tableOffset,
                          int start, int end);

void OsdOmpComputeLoopVertexB(OsdVertexDescriptor const &vdesc,
                              float *vertex, float * varying,
                              const int *V_ITa, const int *V_IT,
                              const float *V_W,
                              int vertexOffset, int tableOffset,
                              int start, int end);

void OsdOmpComputeBilinearEdge(OsdVertexDescriptor const &vdesc,
                               float *vertex, float * varying,
                               const int *E_IT,
                               int vertexOffset, int tableOffset,
                               int start, int end);

void OsdOmpComputeBilinearVertex(OsdVertexDescriptor const &vdesc,
                                 float *vertex, float * varying,
                                 const int *V_ITa,
                                 int vertexOffset, int tableOffset,
                                 int start, int end);

void OsdOmpEditVertexAdd(OsdVertexDescriptor const &vdesc, float *vertex,
                         int primVarOffset, int primVarWidth,
                         int vertexOffset, int tableOffset,
                         int start, int end,
                         const unsigned int *editIndices,
                         const float *editValues);

void OsdOmpEditVertexSet(OsdVertexDescriptor const &vdesc, float *vertex,
                         int primVarOffset, int primVarWidth,
                         int vertexOffset, int tableOffset,
                         int start, int end,
                         const unsigned int *editIndices,
                         const float *editValues);

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OSD_OMP_KERNEL_H
