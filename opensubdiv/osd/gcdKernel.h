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

#ifndef OSD_GCD_KERNEL_H
#define OSD_GCD_KERNEL_H

#include "../version.h"

#include <dispatch/dispatch.h>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

struct OsdVertexBufferDescriptor;

void OsdGcdComputeFace(float * vertex, float * varying,
                       OsdVertexBufferDescriptor const &vertexDesc,
                       OsdVertexBufferDescriptor const &varyingDesc,
                       const int *F_IT, const int *F_ITa,
                       int vertexOffset, int tableOffset,
                       int start, int end,
                       dispatch_queue_t gcdq);

void OsdGcdComputeQuadFace(float * vertex, float * varying,
                           OsdVertexBufferDescriptor const &vertexDesc,
                           OsdVertexBufferDescriptor const &varyingDesc,
                           const int *F_IT,
                           int vertexOffset, int tableOffset,
                           int start, int end,
                           dispatch_queue_t gcdq);

void OsdGcdComputeTriQuadFace(float * vertex, float * varying,
                              OsdVertexBufferDescriptor const &vertexDesc,
                              OsdVertexBufferDescriptor const &varyingDesc,
                              const int *F_IT,
                              int vertexOffset, int tableOffset,
                              int start, int end,
                              dispatch_queue_t gcdq);

void OsdGcdComputeEdge(float *vertex, float * varying,
                       OsdVertexBufferDescriptor const &vertexDesc,
                       OsdVertexBufferDescriptor const &varyingDesc,
                       const int *E_IT, const float *E_ITa,
                       int vertexOffset, int tableOffset,
                       int start, int end,
                       dispatch_queue_t gcdq);

void OsdGcdComputeRestrictedEdge(float *vertex, float * varying,
                                 OsdVertexBufferDescriptor const &vertexDesc,
                                 OsdVertexBufferDescriptor const &varyingDesc,
                                 const int *E_IT,
                                 int vertexOffset, int tableOffset,
                                 int start, int end,
                                 dispatch_queue_t gcdq);

void OsdGcdComputeVertexA(float *vertex, float * varying,
                          OsdVertexBufferDescriptor const &vertexDesc,
                          OsdVertexBufferDescriptor const &varyingDesc,
                          const int *V_ITa, const float *V_IT,
                          int vertexOffset, int tableOffset,
                          int start, int end, int pass,
                          dispatch_queue_t gcdq);

void OsdGcdComputeVertexB(float *vertex, float * varying,
                          OsdVertexBufferDescriptor const &vertexDesc,
                          OsdVertexBufferDescriptor const &varyingDesc,
                          const int *V_ITa, const int *V_IT, const float *V_W,
                          int vertexOffset, int tableOffset,
                          int start, int end,
                          dispatch_queue_t gcdq);

void OsdGcdComputeRestrictedVertexA(float *vertex, float * varying,
                                    OsdVertexBufferDescriptor const &vertexDesc,
                                    OsdVertexBufferDescriptor const &varyingDesc,
                                    const int *V_ITa,
                                    int vertexOffset, int tableOffset,
                                    int start, int end,
                                    dispatch_queue_t gcdq);

void OsdGcdComputeRestrictedVertexB1(float *vertex, float * varying,
                                     OsdVertexBufferDescriptor const &vertexDesc,
                                     OsdVertexBufferDescriptor const &varyingDesc,
                                     const int *V_ITa, const int *V_IT,
                                     int vertexOffset, int tableOffset,
                                     int start, int end,
                                     dispatch_queue_t gcdq);

void OsdGcdComputeRestrictedVertexB2(float *vertex, float * varying,
                                     OsdVertexBufferDescriptor const &vertexDesc,
                                     OsdVertexBufferDescriptor const &varyingDesc,
                                     const int *V_ITa, const int *V_IT,
                                     int vertexOffset, int tableOffset,
                                     int start, int end,
                                     dispatch_queue_t gcdq);

void OsdGcdComputeLoopVertexB(float *vertex, float * varying,
                              OsdVertexBufferDescriptor const &vertexDesc,
                              OsdVertexBufferDescriptor const &varyingDesc,
                              const int *V_ITa, const int *V_IT,
                              const float *V_W,
                              int vertexOffset, int tableOffset,
                              int start, int end,
                              dispatch_queue_t gcdq);

void OsdGcdComputeBilinearEdge(float *vertex, float * varying,
                               OsdVertexBufferDescriptor const &vertexDesc,
                               OsdVertexBufferDescriptor const &varyingDesc,
                               const int *E_IT,
                               int vertexOffset, int tableOffset,
                               int start, int end,
                               dispatch_queue_t gcdq);

void OsdGcdComputeBilinearVertex(float *vertex, float * varying,
                                 OsdVertexBufferDescriptor const &vertexDesc,
                                 OsdVertexBufferDescriptor const &varyingDesc,
                                 const int *V_ITa,
                                 int vertexOffset, int tableOffset,
                                 int start, int end,
                                 dispatch_queue_t gcdq);

void OsdGcdEditVertexAdd(float *vertex,
                         OsdVertexBufferDescriptor const &vertexDesc,
                         int primVarOffset, int primVarWidth,
                         int vertexOffset, int tableOffset,
                         int start, int end,
                         const unsigned int *editIndices, const float *editValues,
                         dispatch_queue_t gcdq);

void OsdGcdEditVertexSet(float *vertex,
                         OsdVertexBufferDescriptor const &vertexDesc,
                         int primVarOffset, int primVarWidth,
                         int vertexOffset, int tableOffset,
                         int start, int end,
                         const unsigned int *editIndices, const float *editValues,
                         dispatch_queue_t gcdq);

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OSD_GCD_KERNEL_H
