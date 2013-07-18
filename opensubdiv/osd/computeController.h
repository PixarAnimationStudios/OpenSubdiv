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
#ifndef OSD_COMPUTE_CONTROLLER_H
#define OSD_COMPUTE_CONTROLLER_H

#include "../version.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

/*!
  \page sequence_page API sequence diagrams

  This section describes the typical sequence of initialization and drawing
  animated prims using OpenSubdiv.

  \section init_sec Initialize

  \image html OsdCreateSequence.png

  \section draw_sec Refine and Draw

  \image html OsdRefineDrawSequence.png

 */

// XXX: do we really need this base class?
class OsdComputeController {
public:
    virtual ~OsdComputeController() {}

protected:
    OsdComputeController() {}
};

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OSD_COMPUTE_CONTROLLER_H
