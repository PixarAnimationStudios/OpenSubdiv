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

#include "../far/stencilTables.h"

#include "../osd/cpuComputeContext.h"
#include "../osd/cpuKernel.h"
#include "../osd/error.h"

#include <cstring>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

// ----------------------------------------------------------------------------

CpuComputeContext::CpuComputeContext(
    Far::StencilTables const * vertexStencilTables,
        Far::StencilTables const * varyingStencilTables) :
            _vertexStencilTables(0), _varyingStencilTables(0) {

    // XXXX manuelk we do not own the tables, so use copy-constructor for now
    //              smart pointers eventually
    if (vertexStencilTables) {
        _vertexStencilTables = new Far::StencilTables(*vertexStencilTables);
    }

    if (varyingStencilTables) {
        _varyingStencilTables = new Far::StencilTables(*varyingStencilTables);
    }
}

// ----------------------------------------------------------------------------

CpuComputeContext::~CpuComputeContext() { 

    delete _vertexStencilTables;
    delete _varyingStencilTables;
}

// ----------------------------------------------------------------------------

CpuComputeContext *
CpuComputeContext::Create(
    Far::StencilTables const * vertexStencilTables,
        Far::StencilTables const * varyingStencilTables) {

    return new CpuComputeContext(vertexStencilTables, varyingStencilTables);
}

}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
