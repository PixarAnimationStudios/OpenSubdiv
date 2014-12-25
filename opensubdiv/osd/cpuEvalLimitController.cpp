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

#include "../osd/cpuEvalLimitController.h"
#include "../osd/cpuEvalLimitContext.h"
#include "../osd/cpuEvalLimitKernel.h"
#include "../far/patchTables.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

CpuEvalLimitController::CpuEvalLimitController() {
}

CpuEvalLimitController::~CpuEvalLimitController() {
}

// normalize & rotate (u,v) to the sub-patch
inline void
computeSubPatchCoords(Far::PatchParam pparam, float & u, float & v ) {
    pparam.bitField.Normalize(u, v);
    pparam.bitField.Rotate(u, v);
}

// Vertex interpolation of a sample at the limit
int
CpuEvalLimitController::EvalLimitSample( LimitLocation const & coord,
                                         CpuEvalLimitContext * context,
                                         VertexBufferDescriptor const & outDesc,
                                         float * outQ,
                                         float * outDQU,
                                         float * outDQV ) const {
    typedef Far::PatchDescriptor Desc;

    float s=coord.s,
          t=coord.t;

    Far::PatchMap::Handle const * handle = context->GetPatchMap().FindPatch( coord.ptexIndex, s, t );
    if (not handle) {
        return 0;  // no handle if there is a hole or 'coord' is incorrect
    }

    VertexData const & vertexData = _currentBindState.vertexData;

    if (vertexData.in) {

        Far::PatchTables const & ptables = context->GetPatchTables();

        Far::PatchParam pparam = ptables.GetPatchParam(*handle);
        pparam.bitField.Normalize(s, t);

        Far::ConstIndexArray cvs = ptables.GetPatchVertices(*handle);

        Far::PatchDescriptor desc = ptables.GetPatchDescriptor(*handle);
        switch (desc.GetType()) {
            case Desc::REGULAR  : evalBSpline( pparam.bitField, s, t, cvs.begin(),
                                               vertexData.inDesc,
                                               vertexData.in,
                                               outDesc,
                                               outQ, outDQU, outDQV );
                                  break;
            case Desc::BOUNDARY : evalBoundary( pparam.bitField, s, t, cvs.begin(),
                                                vertexData.inDesc,
                                                vertexData.in,
                                                outDesc,
                                                outQ, outDQU, outDQV );
                                  break;
            case Desc::CORNER   : evalCorner( pparam.bitField, s, t, cvs.begin(),
                                              vertexData.inDesc,
                                              vertexData.in,
                                              outDesc,
                                              outQ, outDQU, outDQV );
                                  break;
            case Desc::GREGORY  : evalGregory( pparam.bitField, t, s, cvs.begin(),
                                               &ptables.GetVertexValenceTable()[0],
                                               ptables.GetPatchQuadOffsets(*handle).begin(),
                                               ptables.GetMaxValence(),
                                               vertexData.inDesc,
                                               vertexData.in,
                                               outDesc,
                                               outQ, outDQU, outDQV );
                                  break;
            case Desc::GREGORY_BOUNDARY : evalGregoryBoundary( pparam.bitField, t, s, cvs.begin(),
                                                               &ptables.GetVertexValenceTable()[0],
                                                               ptables.GetPatchQuadOffsets(*handle).begin(),
                                                               ptables.GetMaxValence(),
                                                               vertexData.inDesc,
                                                               vertexData.in,
                                                               outDesc,
                                                               outQ, outDQU, outDQV );
                                          break;
            case Desc::GREGORY_BASIS : {
                                           Far::StencilTables const * stencils =
                                               ptables.GetEndCapStencilTables();
                                           assert(stencils and stencils->GetNumStencils()>0);
                                           evalGregoryBasis( pparam.bitField, s, t,
                                                             *stencils,
                                                             ptables.GetEndCapStencilIndex(*handle),
                                                             vertexData.inDesc,
                                                             vertexData.in,
                                                             vertexData.outDesc,
                                                             outQ, outDQU, outDQV );
                                       } break;
            default:
                assert(0);
        }
    }
    assert(0);
    return 1;
}

// Vertex interpolation of samples at the limit
int
CpuEvalLimitController::_EvalLimitSample( LimitLocation const & coords,
                                          CpuEvalLimitContext * context,
                                          unsigned int index ) const {
    typedef Far::PatchDescriptor Desc;

    float s=coords.s,
          t=coords.t;

    Far::PatchMap::Handle const * handle = context->GetPatchMap().FindPatch( coords.ptexIndex, s, t );
    if (not handle) {
        return 0;  // no handle if there is a hole or 'coord' is incorrect
    }

    VertexData const & vertexData = _currentBindState.vertexData;

    Far::PatchTables const & ptables = context->GetPatchTables();

    Far::PatchParam pparam = ptables.GetPatchParam(*handle);
    pparam.bitField.Normalize(s, t);

    Far::PatchDescriptor desc = ptables.GetPatchDescriptor(*handle);

    Far::ConstIndexArray cvs = ptables.GetPatchVertices(*handle);

    if (vertexData.in) {

        int offset = vertexData.outDesc.stride * index,
            doffset = vertexData.outDesc.length * index;

        if (vertexData.out) {

            // note : don't apply outDesc.offset here, it's done inside patch
            // evaluation
            float * out   = vertexData.out+offset,
                  * outDu = vertexData.outDu ? vertexData.outDu+doffset : 0,
                  * outDv = vertexData.outDv ? vertexData.outDv+doffset : 0;

            switch (desc.GetType()) {
                case Desc::REGULAR  : evalBSpline( pparam.bitField, s, t, cvs.begin(),
                                                   vertexData.inDesc,
                                                   vertexData.in,
                                                   vertexData.outDesc,
                                                   out, outDu, outDv );
                                      break;
                case Desc::BOUNDARY : evalBoundary( pparam.bitField, s, t, cvs.begin(),
                                                    vertexData.inDesc,
                                                    vertexData.in,
                                                    vertexData.outDesc,
                                                    out, outDu, outDv );
                                      break;
                case Desc::CORNER   : evalCorner( pparam.bitField, s, t, cvs.begin(),
                                                  vertexData.inDesc,
                                                  vertexData.in,
                                                  vertexData.outDesc,
                                                  out, outDu, outDv );
                                      break;
                case Desc::GREGORY  : evalGregory( pparam.bitField, t, s, cvs.begin(),
                                                   &ptables.GetVertexValenceTable()[0],
                                                   ptables.GetPatchQuadOffsets(*handle).begin(),
                                                   ptables.GetMaxValence(),
                                                   vertexData.inDesc,
                                                   vertexData.in,
                                                   vertexData.outDesc,
                                                   out, outDu, outDv );
                                      break;
                case Desc::GREGORY_BOUNDARY : evalGregoryBoundary( pparam.bitField, t, s, cvs.begin(),
                                                                   &ptables.GetVertexValenceTable()[0],
                                                                   ptables.GetPatchQuadOffsets(*handle).begin(),
                                                                   ptables.GetMaxValence(),
                                                                   vertexData.inDesc,
                                                                   vertexData.in,
                                                                   vertexData.outDesc,
                                                                   out, outDu, outDv );
                                              break;
                case Desc::GREGORY_BASIS : {
                                               Far::StencilTables const * stencils =
                                                   ptables.GetEndCapStencilTables();
                                               assert(stencils and stencils->GetNumStencils()>0);
                                               evalGregoryBasis( pparam.bitField, s, t,
                                                                 *stencils,
                                                                 ptables.GetEndCapStencilIndex(*handle),
                                                                 vertexData.inDesc,
                                                                 vertexData.in,
                                                                 vertexData.outDesc,
                                                                 out, outDu, outDv );
                                           } break;
                default:
                    assert(0);
            }
        }
    }

    pparam.bitField.Rotate(s, t);

    VaryingData const & varyingData = _currentBindState.varyingData;

    if (varyingData.in and varyingData.out) {

        static int const zeroRings[6][4] = { {5, 6,10, 9},   // regular
                                             {1, 2, 6, 5},   // boundary / single-crease
                                             {1, 2, 5, 4},   // corner
                                             {0, 1, 2, 3} }; // no permutation

        int const * permute = 0;
        switch (desc.GetType()) {
            case Desc::REGULAR          : permute = zeroRings[0]; break;
            case Desc::SINGLE_CREASE    :
            case Desc::BOUNDARY         : permute = zeroRings[1]; break;
            case Desc::CORNER           : permute = zeroRings[2]; break;
            case Desc::GREGORY          :
            case Desc::GREGORY_BOUNDARY :
            case Desc::GREGORY_BASIS    : permute = zeroRings[3]; break;
            default:
                assert(0);
        };

        int offset = varyingData.outDesc.stride * index;

        Far::Index zeroRing[4] = { cvs[permute[0]],
                                   cvs[permute[1]],
                                   cvs[permute[2]],
                                   cvs[permute[3]]  };

        evalBilinear( t, s, zeroRing,
                      varyingData.inDesc,
                      varyingData.in,
                      varyingData.outDesc,
                      varyingData.out+offset);

    }

    // Note : currently we only support bilinear boundary interpolation rules
    // for limit face-varying data.

    FacevaryingData const & facevaryingData = _currentBindState.facevaryingData;

    if (facevaryingData.in and facevaryingData.out) {

            int offset = facevaryingData.outDesc.stride * index;

            static int const zeroRing[4] = {0,1,2,3};

            // XXXX manuelk this assumes FVar data is ordered with 4 CVs / patch :
            //              bi-cubic FVar interpolation will require proper topology
            //              accessors in Far::PatchTables and this code will change
            evalBilinear( s, t, zeroRing,
                          facevaryingData.inDesc,
                          &facevaryingData.in[handle->patchIndex*4*facevaryingData.outDesc.stride],
                          facevaryingData.outDesc,
                          facevaryingData.out+offset);

    }
    return 1;
}

}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
