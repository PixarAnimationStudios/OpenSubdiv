// Copyright 2014 Google Inc. All rights reserved.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef OSD_NEON_KERNEL_H
#define OSD_NEON_KERNEL_H

#include "../version.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

void OsdNeonComputeQuadFace(float *vertex, const int *F_IT, int vertexOffset,
    int tableOffset, int batchSize);

void OsdNeonComputeTriQuadFace(float *vertex, const int *F_IT, int vertexOffset,
    int tableOffset, int batchSize);

void OsdNeonComputeRestrictedEdge(float *vertex, const int *E_IT,
    int vertexOffset, int tableOffset, int batchSize);

void OsdNeonComputeRestrictedVertexB1(float *vertex, const int *V_ITa,
    const int *V_IT, int vertexOffset, int tableOffset, int start, int end);

void OsdNeonComputeRestrictedVertexB2(float *vertex, const int *V_ITa,
    const int *V_IT, int vertexOffset, int tableOffset, int start, int end);

void OsdNeonComputeRestrictedVertexA(float *vertex, const int *V_ITa,
    int vertexOffset, int tableOffset, int start, int end);

}  // end namespace OPENSUBDIV_VERSION

using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif
