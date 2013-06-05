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
                                             OsdCpuEvalLimitContext const *context,
                                             unsigned int index ) {
    
    int npatches=0; 
    FarPatchTables::PatchHandle const * patchHandles;
    
    // Get the list of all children patches of the face described in coords
    if (not context->GetPatchesMap()->GetChildPatchesHandles(coords.face, &npatches, &patchHandles))
        return 0;
    
    OsdCpuEvalLimitContext::EvalData const & data = context->GetVertexData();
    
    for (int i=0; i<npatches; ++i) {
    
        FarPatchTables::PatchHandle const & handle = patchHandles[i];

        FarPatchParam::BitField bits = context->GetPatchBitFields()[ handle.serialIndex ];
        assert( handle.array < context->GetPatchArrayVector().size() );

        float u=coords.u,
              v=coords.v;
        
        // check if the patch contains the point      
        if (not bits.Normalize(u,v))
            continue;

        assert( (u>=0.0f) and (u<=1.0f) and (v>=0.0f) and (v<=1.0f) );

        bits.Rotate(u, v);

        FarPatchTables::PatchArray const & parray = context->GetPatchArrayVector()[ handle.array ];

        unsigned int const * cvs = &context->GetControlVertices()[ parray.GetVertIndex() + handle.vertexOffset ];

        OsdCpuEvalLimitContext::EvalData const & vertexData  = context->GetVertexData(), 
                                               & varyingData = context->GetVaryingData();

        // Position lookup pointers at the indexed vertex
        float const * inQ = vertexData.GetInputData();
        float * outQ = const_cast<float *>(vertexData.GetOutputData(index));
        float * outdQu = const_cast<float *>(vertexData.GetOutputDU(index));
        float * outdQv = const_cast<float *>(vertexData.GetOutputDV(index));

        // Based on patch type - go execute interpolation
        switch( parray.GetDescriptor().GetType() ) {

            case FarPatchTables::REGULAR  : if (vertexData.IsBound()) {
                                                evalBSpline( v, u, cvs,
                                                             data.GetInputDesc(),
                                                             inQ,
                                                             data.GetOutputDesc(),
                                                             outQ, outdQu, outdQv );
                                            } break;
            
            case FarPatchTables::BOUNDARY : if (vertexData.IsBound()) {
                                                evalBoundary( v, u, cvs,
                                                              data.GetInputDesc(),
                                                              inQ,
                                                              data.GetOutputDesc(),
                                                              outQ, outdQu, outdQv );
                                            } break;
            
            case FarPatchTables::CORNER   : if (vertexData.IsBound()) {
                                                evalCorner( v, u, cvs,
                                                            data.GetInputDesc(),
                                                            inQ,
                                                            data.GetOutputDesc(),
                                                            outQ, outdQu, outdQv);
                                            } break;
 
            
            case FarPatchTables::GREGORY  : if (vertexData.IsBound()) {
                                                evalGregory( v, u, cvs,
                                                             context->GetVertexValenceBuffer(),
                                                             context->GetQuadOffsetBuffer() + parray.GetQuadOffsetIndex() + handle.vertexOffset,  
                                                             context->GetMaxValence(),
                                                             data.GetInputDesc(),
                                                             inQ,
                                                             data.GetOutputDesc(),
                                                             outQ, outdQu, outdQv);
                                            } break;

            case FarPatchTables::GREGORY_BOUNDARY :
                                            if (vertexData.IsBound()) {
                                                evalGregoryBoundary(v, u, cvs,
                                                                    context->GetVertexValenceBuffer(),
                                                                    context->GetQuadOffsetBuffer() + parray.GetQuadOffsetIndex() + handle.vertexOffset,
                                                                    context->GetMaxValence(),
                                                                    data.GetInputDesc(),
                                                                    inQ,
                                                                    data.GetOutputDesc(),
                                                                    outQ, outdQu, outdQv);
                                            } break;

            default:
                assert(0);
        }
        if (varyingData.IsBound()) {
        
            static int indices[5][4] = { {5, 6,10, 9},  // regular
                                         {1, 2, 6, 5},  // boundary
                                         {1, 2, 5, 4},  // corner
                                         {0, 1, 2, 3},  // gregory
                                         {0, 1, 2, 3} };// gregory boundary
 
            int type = (int)(parray.GetDescriptor().GetType() - FarPatchTables::REGULAR);
            
            unsigned int zeroRing[4] = { cvs[indices[type][0]],
                                         cvs[indices[type][1]],  
                                         cvs[indices[type][2]],  
                                         cvs[indices[type][3]]  };
        
            evalVarying( v, u, zeroRing,
                         varyingData.GetInputDesc(),
                         varyingData.GetInputData(),
                         varyingData.GetOutputDesc(),
                         const_cast<float *>(varyingData.GetOutputData(index)) );

        }
        return 1;
    }
    
    return 0;
}

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
