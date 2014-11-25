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

#include "../osd/cpuSmoothNormalController.h"

#include <math.h>
#include <string.h>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

inline void
cross(float *n, const float *p0, const float *p1, const float *p2) {

    float a[3] = { p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2] };
    float b[3] = { p2[0]-p0[0], p2[1]-p0[1], p2[2]-p0[2] };
    n[0] = a[1]*b[2]-a[2]*b[1];
    n[1] = a[2]*b[0]-a[0]*b[2];
    n[2] = a[0]*b[1]-a[1]*b[0];

    float rn = 1.0f/sqrtf(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
    n[0] *= rn;
    n[1] *= rn;
    n[2] *= rn;
}

void CpuSmoothNormalController::_smootheNormals(
    CpuSmoothNormalContext * context) {

    VertexBufferDescriptor const & iDesc = context->GetInputVertexDescriptor(),
                                    & oDesc = context->GetOutputVertexDescriptor();

    assert(iDesc.length==3 and oDesc.length==3);

    float const * iBuffer = context->GetCurrentInputVertexBuffer() + iDesc.offset;
    float * oBuffer = context->GetCurrentOutputVertexBuffer() + oDesc.offset;

    int nfaces = context->GetNumFaces(),
        nverts = context->GetNumVertices();

    // if necessary, reset all normal values to 0
    if (context->GetResetMemory()) {
        if (oDesc.length == oDesc.stride) {
            memset(oBuffer, 0, nverts*oDesc.length*sizeof(float));
        } else {
            float * ptr = oBuffer;
            for (int j=0; j<nverts; ++j, ptr += oDesc.stride) {
                memset(ptr, 0, oDesc.length*sizeof(float));
            }
        }
    }

    Far::Index const * fverts = context->GetFaceVertices();
    for (int face=0; face<nfaces; ++face, fverts+=4) {

        float const * p0 = iBuffer + fverts[0]*iDesc.stride,
                    * p1 = iBuffer + fverts[1]*iDesc.stride,
                    * p2 = iBuffer + fverts[2]*iDesc.stride;

        // compute face normal
        float n[3];
        cross( n, p0, p1, p2 );

        // add normal to all vertices of the face
        for (int i=0; i<4; ++i) {

            float * dst = oBuffer + fverts[i]*oDesc.stride;

            dst[0] += n[0];
            dst[1] += n[1];
            dst[2] += n[2];
        }
    }
}

CpuSmoothNormalController::CpuSmoothNormalController() {
}

CpuSmoothNormalController::~CpuSmoothNormalController() {
}

void
CpuSmoothNormalController::Synchronize() {
}

}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
