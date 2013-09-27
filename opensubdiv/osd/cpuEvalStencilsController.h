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

#ifndef FAR_CPU_EVALSTENCILS_CONTROLLER_H
#define FAR_CPU_EVALSTENCILS_CONTROLLER_H

#include "../version.h"

#include "../osd/cpuEvalStencilsContext.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

///
/// \brief CPU stencils evaluation controller
///
/// OsdCpuStencilsController is a compute controller class to launch
/// single threaded CPU stencil evalution kernels.
///
/// Controller entities execute requests from Context instances that they share
/// common interfaces with. Controllers are attached to discrete compute devices
/// and share the devices resources with Context entities.
///
class OsdCpuEvalStencilsController {
public:

    /// Constructor.
    OsdCpuEvalStencilsController();

    /// Destructor.
    ~OsdCpuEvalStencilsController();
    
    
    /// \brief Applies stencil weights to the control vertex data
    ///
    /// Applies the stencil weights to the control vertex data to evaluate the
    /// interpolated limit positions at the parametric locations of the stencils
    ///
    /// @param context          the OsdCpuEvalStencilsContext with the stencil weights
    ///
    /// @param controlDataDesc  vertex buffer descriptor for the control vertex data 
    ///
    /// @param controlVertices  vertex buffer with the control vertices data
    ///
    /// @param outputDataDesc   vertex buffer descriptor for the output vertex data
    ///
    /// @param outputData       vertex buffer where the vertex data will be output
    ///
    template<class CONTROL_BUFFER, class OUTPUT_BUFFER>
    int UpdateValues( OsdCpuEvalStencilsContext * context,
                      OsdVertexBufferDescriptor const & controlDataDesc, CONTROL_BUFFER *controlVertices,
                      OsdVertexBufferDescriptor const & outputDataDesc, OUTPUT_BUFFER *outputData ) {
                       
        if (not context->GetStencilTables()->GetNumStencils())
            return 0;

        context->BindControlData( controlDataDesc, controlVertices );

        context->BindOutputData( outputDataDesc, outputData );
        
        int n = _UpdateValues( context );
        
        context->Unbind();
        
        return n;
    }

    /// \brief Applies derivative stencil weights to the control vertex data
    ///
    /// Computes the U and V derivative stencils to the control vertex data at
    /// the parametric locations contained in each stencil
    ///
    /// @param context          the OsdCpuEvalStencilsContext with the stencil weights
    ///
    /// @param controlDataDesc  vertex buffer descriptor for the control vertex data 
    ///
    /// @param controlVertices  vertex buffer with the control vertices data
    ///
    /// @param outputDuDesc     vertex buffer descriptor for the U derivative output data
    ///
    /// @param outputDuData     output vertex buffer for the U derivative data
    ///
    /// @param outputDvDesc     vertex buffer descriptor for the V deriv output data
    ///
    /// @param outputDvData     output vertex buffer for the V derivative data
    ///
    template<class CONTROL_BUFFER, class OUTPUT_BUFFER>
    int UpdateDerivs( OsdCpuEvalStencilsContext * context,
                      OsdVertexBufferDescriptor const & controlDataDesc, CONTROL_BUFFER *controlVertices,
                      OsdVertexBufferDescriptor const & outputDuDesc, OUTPUT_BUFFER *outputDuData, 
                      OsdVertexBufferDescriptor const & outputDvDesc, OUTPUT_BUFFER *outputDvData ) {
                       
        if (not context->GetStencilTables()->GetNumStencils())
            return 0;

        context->BindControlData( controlDataDesc, controlVertices );

        context->BindOutputDerivData( outputDuDesc, outputDuData, outputDvDesc, outputDvData );
        
        int n = _UpdateDerivs( context );
        
        context->Unbind();
        
        return n;
    }
    
    /// Waits until all running subdivision kernels finish.
    void Synchronize();

private:

    int _UpdateValues( OsdCpuEvalStencilsContext * context );
    int _UpdateDerivs( OsdCpuEvalStencilsContext * context );

};

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif // FAR_CPU_EVALSTENCILS_CONTROLLER_H
