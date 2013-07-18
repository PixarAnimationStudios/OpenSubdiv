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
#ifndef OSD_VERTEX_H
#define OSD_VERTEX_H

#include "../version.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

template <class T> class HbrVertexEdit;
template <class T> class HbrMovingVertexEdit;
class FarVertexEdit;

//!
/*! 
 */
class OsdVertex {
public:
    OsdVertex() {}

    OsdVertex(int index) {}

    OsdVertex(OsdVertex const & src) {}

    void AddWithWeight(OsdVertex const & i, float weight, void * = 0) {}

    void AddVaryingWithWeight(const OsdVertex & i, float weight, void * = 0) {}

    void Clear(void * = 0) {}

    void ApplyVertexEdit(HbrVertexEdit<OsdVertex> const &) { }

    void ApplyVertexEdit(FarVertexEdit const &) { }

    void ApplyMovingVertexEdit(HbrMovingVertexEdit<OsdVertex> const &) { }
};

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OSD_VERTEX_H
