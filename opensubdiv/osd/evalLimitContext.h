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

#ifndef OSD_EVAL_LIMIT_CONTEXT_H
#define OSD_EVAL_LIMIT_CONTEXT_H

#include "../version.h"

#include "../far/patchTables.h"

#include "../osd/nonCopyable.h"
#include "../osd/vertex.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {


/// \brief Coordinates set on a limit surface
///
struct LimitLocation {

    LimitLocation() { }

    /// \brief Constructor
    ///
    /// @param f Ptex face id
    ///
    /// @param x parametric location on face
    ///
    /// @param y parametric location on face
    ///
    LimitLocation(int f, float x, float y) : ptexIndex(f), s(x), t(y) { }

    int ptexIndex;      ///< ptex face index

    float s, t;         ///< parametric location on face
};

class LimitLocationsArray {

public:

    /// \brief Constructor
    LimitLocationsArray() : ptexIndex(-1), numLocations(0), s(0), t(0) { }

    int ptexIndex,      ///< ptex face index
        numLocations;   ///< number of (u,v) coordinates in the array

    float const * s,    ///< array of u coordinates
                * t;    ///< array of v coordinates
};


/// \brief LimitEval Context
///
/// A stub class to derive LimitEval context classes.
///
class EvalLimitContext : private NonCopyable<EvalLimitContext> {

public:
    /// \brief Destructor.
    virtual ~EvalLimitContext();

protected:
    explicit EvalLimitContext(Far::PatchTables const & patchTables);

private:
    bool _adaptive;
};

} // end namespace Osd

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* OSD_EVAL_LIMIT_CONTEXT_H */
