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

#ifndef OSD_TBB_KERNEL_H
#define OSD_TBB_KERNEL_H

#include "../version.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

struct OsdVertexBufferDescriptor;

void OsdTbbComputeFace(float * vertex, float * varying,
                       OsdVertexBufferDescriptor const &vertexDesc,
                       OsdVertexBufferDescriptor const &varyingDesc,
                       int const *F_IT, int const *F_ITa,
                       int vertexOffset, int tableOffset,
                       int start, int end);

void OsdTbbComputeQuadFace(float * vertex, float * varying,
                           OsdVertexBufferDescriptor const &vertexDesc,
                           OsdVertexBufferDescriptor const &varyingDesc,
                           int const *F_IT,
                           int vertexOffset, int tableOffset,
                           int start, int end);

void OsdTbbComputeTriQuadFace(float * vertex, float * varying,
                              OsdVertexBufferDescriptor const &vertexDesc,
                              OsdVertexBufferDescriptor const &varyingDesc,
                              int const *F_IT,
                              int vertexOffset, int tableOffset,
                              int start, int end);

void OsdTbbComputeEdge(float *vertex, float * varying,
                       OsdVertexBufferDescriptor const &vertexDesc,
                       OsdVertexBufferDescriptor const &varyingDesc,
                       int const *E_IT, float const *E_W,
                       int vertexOffset, int tableOffset,
                       int start, int end);

void OsdTbbComputeRestrictedEdge(float *vertex, float * varying,
                                 OsdVertexBufferDescriptor const &vertexDesc,
                                 OsdVertexBufferDescriptor const &varyingDesc,
                                 int const *E_IT,
                                 int vertexOffset, int tableOffset,
                                 int start, int end);

void OsdTbbComputeVertexA(float *vertex, float * varying,
                          OsdVertexBufferDescriptor const &vertexDesc,
                          OsdVertexBufferDescriptor const &varyingDesc,
                          int const *V_ITa, float const *V_W,
                          int vertexOffset, int tableOffset,
                          int start, int end, int pass);

void OsdTbbComputeVertexB(float *vertex, float * varying,
                          OsdVertexBufferDescriptor const &vertexDesc,
                          OsdVertexBufferDescriptor const &varyingDesc,
                          int const *V_ITa, int const *V_IT, float const *V_W,
                          int vertexOffset, int tableOffset,
                          int start, int end);

void OsdTbbComputeRestrictedVertexA(float *vertex, float * varying,
                                    OsdVertexBufferDescriptor const &vertexDesc,
                                    OsdVertexBufferDescriptor const &varyingDesc,
                                    int const *V_ITa,
                                    int vertexOffset, int tableOffset,
                                    int start, int end);

void OsdTbbComputeRestrictedVertexB1(float *vertex, float * varying,
                                     OsdVertexBufferDescriptor const &vertexDesc,
                                     OsdVertexBufferDescriptor const &varyingDesc,
                                     int const *V_ITa, int const *V_IT,
                                     int vertexOffset, int tableOffset,
                                     int start, int end);

void OsdTbbComputeRestrictedVertexB2(float *vertex, float * varying,
                                     OsdVertexBufferDescriptor const &vertexDesc,
                                     OsdVertexBufferDescriptor const &varyingDesc,
                                     int const *V_ITa, int const *V_IT,
                                     int vertexOffset, int tableOffset,
                                     int start, int end);

void OsdTbbComputeLoopVertexB(float *vertex, float * varying,
                              OsdVertexBufferDescriptor const &vertexDesc,
                              OsdVertexBufferDescriptor const &varyingDesc,
                              int const *V_ITa, int const *V_IT,
                              float const *V_W,
                              int vertexOffset, int tableOffset,
                              int start, int end);

void OsdTbbComputeBilinearEdge(float *vertex, float * varying,
                               OsdVertexBufferDescriptor const &vertexDesc,
                               OsdVertexBufferDescriptor const &varyingDesc,
                               int const *E_IT,
                               int vertexOffset, int tableOffset,
                               int start, int end);

void OsdTbbComputeBilinearVertex(float *vertex, float * varying,
                                 OsdVertexBufferDescriptor const &vertexDesc,
                                 OsdVertexBufferDescriptor const &varyingDesc,
                                 int const *V_ITa,
                                 int vertexOffset, int tableOffset,
                                 int start, int end);

void OsdTbbEditVertexAdd(float *vertex,
                         OsdVertexBufferDescriptor const &vertexDesc,
                         int primVarOffset, int primVarWidth,
                         int vertexOffset, int tableOffset,
                         int start, int end,
                         unsigned int const *editIndices,
                         float const *editValues);

void OsdTbbEditVertexSet(float *vertex,
                         OsdVertexBufferDescriptor const &vertexDesc,
                         int primVarOffset, int primVarWidth,
                         int vertexOffset, int tableOffset,
                         int start, int end,
                         unsigned int const *editIndices,
                         float const *editValues);

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OSD_TBB_KERNEL_H
