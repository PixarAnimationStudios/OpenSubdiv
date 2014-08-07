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

#include "../far/dispatcher.h"
#include "../far/subdivisionTables.h"
#include "../osd/cpuKernel.h"
#include "../osd/neonComputeContext.h"
#include "../osd/vertexDescriptor.h"
#include "../osd/error.h"

#include <algorithm>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

OsdNeonComputeContext::OsdNeonComputeContext(
    FarSubdivisionTables const *subdivisionTables,
    FarVertexEditTables const *vertexEditTables)
    : OsdCpuComputeContext(subdivisionTables, vertexEditTables)
{
    // Calculate the maximum vertex valence.
    _maxVertexValence = 0;
    const std::vector<int>& V_ITa = subdivisionTables->Get_V_ITa();
    for (int i = 0; i < (int)V_ITa.size(); i += 5) {
        int vertexValence = V_ITa[i + 1];
        _maxVertexValence = std::max(_maxVertexValence, vertexValence);
    }
}

OsdNeonComputeContext::~OsdNeonComputeContext()
{
}

OsdNeonComputeContext * OsdNeonComputeContext::Create(
    FarSubdivisionTables const *subdivisionTables,
    FarVertexEditTables const *vertexEditTables)
{
    return new OsdNeonComputeContext(subdivisionTables, vertexEditTables);
}

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
