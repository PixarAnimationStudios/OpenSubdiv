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
	
#ifndef OSD_NEON_COMPUTE_CONTEXT_H
#define OSD_NEON_COMPUTE_CONTEXT_H

#include "../version.h"

#include "../far/subdivisionTables.h"
#include "../far/vertexEditTables.h"
#include "../osd/cpuComputeContext.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

class OsdNeonComputeContext : public OsdCpuComputeContext
{
public:
    /// Creates an OsdNeonComputeContext instance
    ///
    /// @param subdivisionTables the FarSubdivisionTables used for this context
    ///
    /// @param vertexEditTables the FarVertexEditTables used for this context
    ///
    static OsdNeonComputeContext * Create(
        FarSubdivisionTables const *subdivisionTables,
        FarVertexEditTables const *vertexEditTables);

    /// Destructor
    virtual ~OsdNeonComputeContext();

    /// Returns the maximum vertex valence
    int GetMaxVertexValence() const
    {
        return _maxVertexValence;
    }

protected:
    explicit OsdNeonComputeContext(
        FarSubdivisionTables const *subdivisionTables,
        FarVertexEditTables const *vertexEditTables);

private:
    int _maxVertexValence;
};

}  // end namespace OPENSUBDIV_VERSION

using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif
