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

#ifndef FAR_CPU_EVALSTENCILS_CONTEXT_H
#define FAR_CPU_EVALSTENCILS_CONTEXT_H

#include "../version.h"

#include "../far/stencilTables.h"

#include "../osd/vertexDescriptor.h"
#include "../osd/nonCopyable.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

///
/// \brief CPU stencils evaluation context
///
///
class CpuEvalStencilsContext : private NonCopyable<CpuEvalStencilsContext> {

public:
    /// \brief Creates an CpuEvalStencilsContext instance
    ///
    /// @param stencils  a pointer to the Far::StencilTables
    ///
    static CpuEvalStencilsContext * Create(Far::StencilTables const *stencils);

    /// \brief Returns the Far::StencilTables applied
    Far::StencilTables const * GetStencilTables() const {
        return _stencils;
    }

protected:

    CpuEvalStencilsContext(Far::StencilTables const *stencils);

private:

    Far::StencilTables const * _stencils;
};

} // end namespace Osd

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif // FAR_CPU_EVALSTENCILS_CONTEXT_H
