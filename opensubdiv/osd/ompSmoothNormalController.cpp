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

#include "../osd/ompSmoothNormalController.h"

#ifdef OPENSUBDIV_HAS_OPENMP
    #include <omp.h>
#endif

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

void OmpSmoothNormalController::_smootheNormals(
    CpuSmoothNormalContext * context) {

    VertexBufferDescriptor const & iDesc = context->GetInputVertexDescriptor(),
                                    & oDesc = context->GetOutputVertexDescriptor();

    assert(iDesc.length==3 and oDesc.length==3);

    float const * iBuffer = context->GetCurrentInputVertexBuffer() + iDesc.offset;
    float * oBuffer = context->GetCurrentOutputVertexBuffer() + oDesc.offset;

    std::vector<unsigned int> const & verts = context->GetControlVertices();

    Far::PatchTables::PatchArrayVector const & parrays = context->GetPatchArrayVector();

    if (verts.empty() or parrays.empty() or (not iBuffer) or (not oBuffer)) {
        return;
    }

    for (int i=0; i<(int)parrays.size(); ++i) {

        Far::PatchTables::PatchArray const & pa = parrays[i];

        Far::PatchTables::Type type = pa.GetDescriptor().GetType();


        if (type==Far::PatchTables::QUADS or type==Far::PatchTables::TRIANGLES) {

            int nv = Far::PatchTables::Descriptor::GetNumControlVertices(type);

            // if necessary, reset all normal values to 0
            if (context->GetResetMemory()) {
#pragma omp parallel for
                for (int j=0; j<context->GetNumVertices(); ++j) {
                    float * ptr = oBuffer + j * oDesc.stride;
                    memset(ptr, 0, oDesc.length*sizeof(float));
                }
            }


#pragma omp parallel for
            for (int j=0; j<(int)pa.GetNumPatches(); ++j) {

                int idx = pa.GetVertIndex() + j*nv;

                float const * p0 = iBuffer + verts[idx+0]*iDesc.stride,
                            * p1 = iBuffer + verts[idx+1]*iDesc.stride,
                            * p2 = iBuffer + verts[idx+2]*iDesc.stride;

                // compute face normal
                float n[3];
                cross( n, p0, p1, p2 );

                // add normal to all vertices of the face
                for (int k=0; k<nv; ++k) {

                    float * dst = oBuffer + verts[idx+k]*oDesc.stride;

                    dst[0] += n[0];
                    dst[1] += n[1];
                    dst[2] += n[2];
                }
            }
        }
    }

}

OmpSmoothNormalController::OmpSmoothNormalController() {
}

OmpSmoothNormalController::~OmpSmoothNormalController() {
}

void
OmpSmoothNormalController::Synchronize() {
}

} // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
