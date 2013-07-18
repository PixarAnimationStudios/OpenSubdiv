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
#ifndef OSD_EVAL_LIMIT_CONTEXT_H
#define OSD_EVAL_LIMIT_CONTEXT_H

#include "../version.h"

#include "../far/mesh.h"

#include "../osd/nonCopyable.h"
#include "../osd/vertex.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {


/// \brief Coordinates set on a limit surface
///
class OsdEvalCoords {

public:

    OsdEvalCoords() { }

    /// \brief Constructor
    ///
    /// @param f Ptex face id
    ///
    /// @param x parametric location on face
    ///
    /// @param y parametric location on face
    ///
    OsdEvalCoords(int f, float x, float y) : face(f), u(x), v(y) { }
    
    unsigned int face; //  Ptex face ID
    float u,v;         // local face (u,v)
};


/// \brief LimitEval Context
///
/// A stub class to derive LimitEval context classes.
///
class OsdEvalLimitContext : OsdNonCopyable<OsdEvalLimitContext> {

public:
    /// \brief Destructor.
    virtual ~OsdEvalLimitContext();

protected:
    explicit OsdEvalLimitContext(FarMesh<OsdVertex> const * farmesh);

private:
    bool _adaptive;
};

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* OSD_EVAL_LIMIT_CONTEXT_H */
