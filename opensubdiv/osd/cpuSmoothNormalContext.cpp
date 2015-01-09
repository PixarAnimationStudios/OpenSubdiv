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

#include "../osd/cpuSmoothNormalContext.h"
#include "../far/topologyRefiner.h"

#include <string.h>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

CpuSmoothNormalContext::CpuSmoothNormalContext(
    Far::TopologyRefiner const & refiner, int level, bool resetMemory) :
        _numVertices(0), _resetMemory(resetMemory) {

    int nfaces = refiner.GetNumFaces(level),
        nverts = nfaces * 4;

    _faceVerts.resize(nverts);
    Far::Index * dest = &_faceVerts[0];

    for (int face=0; face<nfaces; ++face, dest+=4) {
        Far::ConstIndexArray fverts = refiner.GetFaceVertices(level, face);
        memcpy(dest, fverts.begin(), 4 * sizeof(Far::Index));
    }
}

CpuSmoothNormalContext *
CpuSmoothNormalContext::Create(Far::TopologyRefiner const & refiner,
    int level, bool resetMemory) {

    assert((not refiner.IsUniform()) and
           (refiner.GetMaxLevel()>0) and
           (level>0) and (level<refiner.GetMaxLevel()));
    return new CpuSmoothNormalContext(refiner, level, resetMemory);
}

}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
