//
//     Copyright (C) Pixar. All rights reserved.
//
//     This license governs use of the accompanying software. If you
//     use the software, you accept this license. If you do not accept
//     the license, do not use the software.
//
//     1. Definitions
//     The terms "reproduce," "reproduction," "derivative works," and
//     "distribution" have the same meaning here as under U.S.
//     copyright law.  A "contribution" is the original software, or
//     any additions or changes to the software.
//     A "contributor" is any person or entity that distributes its
//     contribution under this license.
//     "Licensed patents" are a contributor's patent claims that read
//     directly on its contribution.
//
//     2. Grant of Rights
//     (A) Copyright Grant- Subject to the terms of this license,
//     including the license conditions and limitations in section 3,
//     each contributor grants you a non-exclusive, worldwide,
//     royalty-free copyright license to reproduce its contribution,
//     prepare derivative works of its contribution, and distribute
//     its contribution or any derivative works that you create.
//     (B) Patent Grant- Subject to the terms of this license,
//     including the license conditions and limitations in section 3,
//     each contributor grants you a non-exclusive, worldwide,
//     royalty-free license under its licensed patents to make, have
//     made, use, sell, offer for sale, import, and/or otherwise
//     dispose of its contribution in the software or derivative works
//     of the contribution in the software.
//
//     3. Conditions and Limitations
//     (A) No Trademark License- This license does not grant you
//     rights to use any contributor's name, logo, or trademarks.
//     (B) If you bring a patent claim against any contributor over
//     patents that you claim are infringed by the software, your
//     patent license from such contributor to the software ends
//     automatically.
//     (C) If you distribute any portion of the software, you must
//     retain all copyright, patent, trademark, and attribution
//     notices that are present in the software.
//     (D) If you distribute any portion of the software in source
//     code form, you may do so only under this license by including a
//     complete copy of this license with your distribution. If you
//     distribute any portion of the software in compiled or object
//     code form, you may only do so under a license that complies
//     with this license.
//     (E) The software is licensed "as-is." You bear the risk of
//     using it. The contributors give no express warranties,
//     guarantees or conditions. You may have additional consumer
//     rights under your local laws which this license cannot change.
//     To the extent permitted under your local laws, the contributors
//     exclude the implied warranties of merchantability, fitness for
//     a particular purpose and non-infringement.
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
