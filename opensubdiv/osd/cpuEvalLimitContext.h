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

#ifndef OSD_CPU_EVAL_LIMIT_CONTEXT_H
#define OSD_CPU_EVAL_LIMIT_CONTEXT_H

#include "../version.h"

#include "../osd/evalLimitContext.h"
#include "../far/patchTables.h"
#include "../far/patchMap.h"

#include <map>
#include <stdio.h>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

class CpuEvalLimitContext : public EvalLimitContext {
public:

    /// \brief Factory
    /// Returns an EvalLimitContext from the given far patch tables.
    /// Note : the patchtables is expected to be feature-adaptive and have ptex
    ///        coordinates tables.
    ///
    /// @param patchTables  a pointer to an initialized Far::PatchTables
    ///
    static CpuEvalLimitContext * Create(Far::PatchTables const &patchTables);

    Far::PatchTables const & GetPatchTables() const {
        return _patchTables;
    }

    Far::PatchMap const & GetPatchMap() const {
        return _patchMap;
    }

protected:

    explicit CpuEvalLimitContext(Far::PatchTables const & patchTables);

private:

    Far::PatchTables const _patchTables; // Patch topology data
    Far::PatchMap const _patchMap;       // Patch search accelerator
};


} // end namespace Osd

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* OSD_CPU_EVAL_LIMIT_CONTEXT_H */
