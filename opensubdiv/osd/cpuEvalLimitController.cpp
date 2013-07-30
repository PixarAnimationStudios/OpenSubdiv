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

#include "../osd/cpuEvalLimitController.h"
#include "../osd/cpuEvalLimitKernel.h"
#include "../far/patchTables.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

OsdCpuEvalLimitController::OsdCpuEvalLimitController() {
}

OsdCpuEvalLimitController::~OsdCpuEvalLimitController() {
}

int 
OsdCpuEvalLimitController::_EvalLimitSample( OpenSubdiv::OsdEvalCoords const & coords, 
                                             OsdCpuEvalLimitContext * context,
                                             unsigned int index ) {
    float u=coords.u,
          v=coords.v;
          
    FarPatchMap::Handle const * handle = context->GetPatchMap().FindPatch( coords.face, u, v );

    // the map may not be able to return a handle if there is a hole or the face
    // index is incorrect
    if (not handle)
        return 0;

    FarPatchParam::BitField bits = context->GetPatchBitFields()[ handle->patchIdx ];
    
    bits.Normalize( u, v );

    bits.Rotate( u, v );

    FarPatchTables::PatchArray const & parray = context->GetPatchArrayVector()[ handle->patchArrayIdx ];
    
    unsigned int const * cvs = &context->GetControlVertices()[ parray.GetVertIndex() + handle->vertexOffset ];
    
    OsdCpuEvalLimitContext::VertexData & vertexData = context->GetVertexData();

    if (vertexData.IsBound()) {
    
        int offset = vertexData.outDesc.stride * index;
        
        // Based on patch type - go execute interpolation
        switch( parray.GetDescriptor().GetType() ) {

            case FarPatchTables::REGULAR  : if (vertexData.IsBound()) {
                                                evalBSpline( v, u, cvs,
                                                             vertexData.inDesc,
                                                             vertexData.in.GetData(),
                                                             vertexData.outDesc,
                                                             vertexData.out.GetData()+offset, 
                                                             vertexData.outDu.GetData()+offset, 
                                                             vertexData.outDv.GetData()+offset );
                                            } break;

            case FarPatchTables::BOUNDARY : if (vertexData.IsBound()) {
                                                evalBoundary( v, u, cvs,
                                                              vertexData.inDesc,
                                                              vertexData.in.GetData(),
                                                              vertexData.outDesc,
                                                              vertexData.out.GetData()+offset, 
                                                              vertexData.outDu.GetData()+offset, 
                                                              vertexData.outDv.GetData()+offset );
                                            } break;

            case FarPatchTables::CORNER   : if (vertexData.IsBound()) {
                                                evalCorner( v, u, cvs,
                                                            vertexData.inDesc,
                                                            vertexData.in.GetData(),
                                                            vertexData.outDesc,
                                                            vertexData.out.GetData()+offset,
                                                            vertexData.outDu.GetData()+offset,
                                                            vertexData.outDv.GetData()+offset );
                                            } break;


            case FarPatchTables::GREGORY  : if (vertexData.IsBound()) {
                                                evalGregory( v, u, cvs,
                                                             &context->GetVertexValenceTable()[0],
                                                             &context->GetQuadOffsetTable()[ parray.GetQuadOffsetIndex() + handle->vertexOffset ],  
                                                             context->GetMaxValence(),
                                                             vertexData.inDesc,
                                                             vertexData.in.GetData(),
                                                             vertexData.outDesc,
                                                             vertexData.out.GetData()+offset, 
                                                             vertexData.outDu.GetData()+offset, 
                                                             vertexData.outDv.GetData()+offset );
                                            } break;

            case FarPatchTables::GREGORY_BOUNDARY :
                                            if (vertexData.IsBound()) {
                                                evalGregoryBoundary(v, u, cvs,
                                                                    &context->GetVertexValenceTable()[0],
                                                                    &context->GetQuadOffsetTable()[ parray.GetQuadOffsetIndex() + handle->vertexOffset ],
                                                                    context->GetMaxValence(),
                                                                    vertexData.inDesc,
                                                                    vertexData.in.GetData(),
                                                                    vertexData.outDesc,
                                                                    vertexData.out.GetData()+offset,
                                                                    vertexData.outDu.GetData()+offset,
                                                                    vertexData.outDv.GetData()+offset );
                                            } break;

            default:
                assert(0);
        }
    }
    
    OsdCpuEvalLimitContext::VaryingData & varyingData = context->GetVaryingData();

    if (varyingData.IsBound()) {

        static int indices[5][4] = { {5, 6,10, 9},  // regular
                                     {1, 2, 6, 5},  // boundary
                                     {1, 2, 5, 4},  // corner
                                     {0, 1, 2, 3},  // gregory
                                     {0, 1, 2, 3} };// gregory boundary

        int type = (int)(parray.GetDescriptor().GetType() - FarPatchTables::REGULAR);

        int offset = varyingData.outDesc.stride * index;

        unsigned int zeroRing[4] = { cvs[indices[type][0]],
                                     cvs[indices[type][1]],  
                                     cvs[indices[type][2]],  
                                     cvs[indices[type][3]]  };

        evalBilinear( v, u, zeroRing,
                      varyingData.inDesc,
                      varyingData.in.GetData(),
                      varyingData.outDesc,
                      varyingData.out.GetData()+offset);

    }
    
    // Note : currently we only support bilinear boundary interpolation rules
    // for face-varying data. Although Hbr supports 3 additional smooth rule
    // sets, the feature-adaptive patch interpolation code currently does not
    // support them, and neither does this EvalContext.
    OsdCpuEvalLimitContext::FaceVaryingData & faceVaryingData = context->GetFaceVaryingData();
    if (faceVaryingData.IsBound()) {

        FarPatchTables::FVarDataTable const & fvarData = context->GetFVarData();

        if (not fvarData.empty()) {

            int offset = faceVaryingData.outDesc.stride * index;

            static unsigned int zeroRing[4] = {0,1,2,3};

            evalBilinear( v, u, zeroRing,
                          faceVaryingData.inDesc,
                          &fvarData[ handle->patchIdx * 4 * context->GetFVarWidth() ],
                          faceVaryingData.outDesc,
                          faceVaryingData.out.GetData()+offset);
        }
    }
    
    return 1;
}

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
