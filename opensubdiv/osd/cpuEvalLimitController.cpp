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
    
    // Position lookup pointers at the indexed vertex
    float const * inQ = context->GetInputVertexData();
    float * outQ = context->GetOutputVertexData() + index * context->GetOutputDesc().stride;
    float * outdQu = context->GetOutputVertexDataUDerivative() + index * context->GetOutputDesc().stride;
    float * outdQv = context->GetOutputVertexDataVDerivative() + index * context->GetOutputDesc().stride;

    for (int i=0; i<npatches; ++i) {
    
        FarPatchTables::PatchHandle const & handle = patchHandles[i];

        FarPatchParam::BitField bits = context->GetPatchBitFields()[ handle.serialIndex ];

        float frac = 1.0f / float( 1 << bits.GetDepth() );
        
        // Are the coordinates within the parametric space covered by the patch ?
        float pu = (float)bits.GetU()*frac;
        if ( (coords.u < pu) or (coords.u > pu+frac) )
            continue;
        
        float pv = (float)bits.GetV()*frac;
        if ( (coords.v < pv) or (coords.v > pv+frac) )
            continue;

        assert( handle.array < context->GetPatchArrayVector().size() );

        FarPatchTables::PatchArray const & parray = context->GetPatchArrayVector()[ handle.array ];

        unsigned int const * cvs = &context->GetControlVertices()[ parray.GetVertIndex() + handle.vertexOffset ];

        // normalize u,v coordinates
        float u = (coords.u - pu) / frac,
              v = (coords.v - pv) / frac;

        assert( (u>=0.0f) and (u<=1.0f) and (v>=0.0f) and (v<=1.0f) );

        typedef FarPatchTables::Descriptor PD;

        // Rotate u,v to compensate for transition pattern orientation
        if ( parray.GetDescriptor().GetPattern()!=FarPatchTables::NON_TRANSITION ) {
            switch( bits.GetRotation() ) {
                 case 0 : break;
                 case 1 : { float tmp=v; v=1.0f-u; u=tmp; } break;
                 case 2 : { u=1.0f-u; v=1.0f-v; } break;
                 case 3 : { float tmp=u; u=1.0f-v; v=tmp; } break;
                 default:
                     assert(0);
            }
        }

        // Based on patch type - go execute interpolation
        switch( parray.GetDescriptor().GetType() ) {

            case FarPatchTables::REGULAR  : { 
                                              evalBSpline( v, u, cvs,
                                                           context->GetInputDesc(),
                                                           inQ,
                                                           context->GetOutputDesc(),
                                                           outQ, outdQu, outdQv); 
                                              return 1; }
            
            case FarPatchTables::BOUNDARY : { return 0; }
            
            case FarPatchTables::CORNER   : { return 0; }
 
            
            case FarPatchTables::GREGORY  : { /*evalGregory(v, u, cvs,
                                                            unsigned int const  * quadOffsetBuffer,
                                                            int maxValence,
                                                            unsigned int const * vertexIndices,
                                                            OsdVertexBufferDescriptor const & inDesc,
                                                            float const * inQ,
                                                            OsdVertexBufferDescriptor const & outDesc,
                                                            float * outQ,
                                                            float * outDQU,
                                                            float * outDQV );*/
                                              return 0; }

            case FarPatchTables::GREGORY_BOUNDARY : { return 0; }

            default: 
                assert(0);
        }        
    }
    return 0;
}

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
