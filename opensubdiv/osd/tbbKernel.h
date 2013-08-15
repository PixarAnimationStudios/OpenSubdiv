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

#ifndef OSD_TBB_KERNEL_H
#define OSD_TBB_KERNEL_H

#include "../version.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

struct OsdVertexDescriptor;

void OsdTbbComputeFace(OsdVertexDescriptor const &vdesc,
                       float * vertex, float * varying,
                       int const *F_IT, int const *F_ITa,
                       int vertexOffset, int tableOffset,
                       int start, int end);

void OsdTbbComputeEdge(OsdVertexDescriptor const &vdesc,
                       float *vertex, float * varying,
                       int const *E_IT, float const *E_ITa,
                       int vertexOffset, int tableOffset,
                       int start, int end);

void OsdTbbComputeVertexA(OsdVertexDescriptor const &vdesc,
                          float *vertex, float * varying,
                          int const *V_ITa, float const *V_IT,
                          int vertexOffset, int tableOffset,
                          int start, int end, int pass);

void OsdTbbComputeVertexB(OsdVertexDescriptor const &vdesc,
                          float *vertex, float * varying,
                          int const *V_ITa, int const *V_IT, float const *V_W,
                          int vertexOffset, int tableOffset,
                          int start, int end);

void OsdTbbComputeLoopVertexB(OsdVertexDescriptor const &vdesc,
                              float *vertex, float * varying,
                              int const *V_ITa, int const *V_IT,
                              float const *V_W,
                              int vertexOffset, int tableOffset,
                              int start, int end);

void OsdTbbComputeBilinearEdge(OsdVertexDescriptor const &vdesc,
                               float *vertex, float * varying,
                               int const *E_IT,
                               int vertexOffset, int tableOffset,
                               int start, int end);

void OsdTbbComputeBilinearVertex(OsdVertexDescriptor const &vdesc,
                                 float *vertex, float * varying,
                                 int const *V_ITa,
                                 int vertexOffset, int tableOffset,
                                 int start, int end);

void OsdTbbEditVertexAdd(OsdVertexDescriptor const &vdesc, float *vertex,
                         int primVarOffset, int primVarWidth,
                         int vertexOffset, int tableOffset,
                         int start, int end,
                         unsigned int const *editIndices,
                         float const *editValues);

void OsdTbbEditVertexSet(OsdVertexDescriptor const &vdesc, float *vertex,
                         int primVarOffset, int primVarWidth,
                         int vertexOffset, int tableOffset,
                         int start, int end,
                         unsigned int const *editIndices,
                         float const *editValues);

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OSD_TBB_KERNEL_H
