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

        bindControlData( controlDataDesc, controlVertices );

        bindOutputData( outputDataDesc, outputData );
        
        int n = _UpdateValues( context );
        
        unbind();
        
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

        bindControlData( controlDataDesc, controlVertices );

        bindOutputDerivData( outputDuDesc, outputDuData, outputDvDesc, outputDvData );
        
        int n = _UpdateDerivs( context );
        
        unbind();
        
        return n;
    }
    
    /// Waits until all running subdivision kernels finish.
    void Synchronize();

protected:

    /// \brief Binds control vertex data buffer
    template<class VERTEX_BUFFER>
    void bindControlData(OsdVertexBufferDescriptor const & controlDataDesc, VERTEX_BUFFER *controlData ) {

        _currentBindState._controlData = controlData ? controlData->BindCpuBuffer() : 0;
        _currentBindState._controlDataDesc = controlDataDesc;

    }

    /// \brief Binds output vertex data buffer
    template<class VERTEX_BUFFER>
    void bindOutputData( OsdVertexBufferDescriptor const & outputDataDesc, VERTEX_BUFFER *outputData ) {

        _currentBindState._outputData = outputData ? outputData->BindCpuBuffer() : 0;
        _currentBindState._outputDataDesc = outputDataDesc;
    }
    
    /// \brief Binds output derivative vertex data buffer
    template<class VERTEX_BUFFER>
    void bindOutputDerivData( OsdVertexBufferDescriptor const & outputDuDesc, VERTEX_BUFFER *outputDu, 
                              OsdVertexBufferDescriptor const & outputDvDesc, VERTEX_BUFFER *outputDv ) {
                              
        _currentBindState._outputUDeriv = outputDu ? outputDu ->BindCpuBuffer() : 0;
        _currentBindState._outputVDeriv = outputDv ? outputDv->BindCpuBuffer() : 0;
        _currentBindState._outputDuDesc = outputDuDesc;
        _currentBindState._outputDvDesc = outputDvDesc;
    }

    /// \brief Unbinds any previously bound vertex and varying data buffers.
    void unbind() {
       _currentBindState._controlData = 0;
       _currentBindState._controlDataDesc.Reset();

       _currentBindState._outputData = 0;
       _currentBindState._outputDataDesc.Reset();
      
       _currentBindState._outputUDeriv = 0;
       _currentBindState._outputDuDesc.Reset();
       
       _currentBindState._outputVDeriv = 0;
       _currentBindState._outputDvDesc.Reset();
    }

private:

    int _UpdateValues( OsdCpuEvalStencilsContext * context );
    int _UpdateDerivs( OsdCpuEvalStencilsContext * context );

    // Bind state is a transitional state during refinement.
    // It doesn't take an ownership of vertex buffers.
    struct BindState {

        // transient mesh data
        OsdVertexBufferDescriptor _controlDataDesc,
                                  _outputDataDesc,
                                  _outputDuDesc,
                                  _outputDvDesc;

        float * _controlData,
              * _outputData,
              * _outputUDeriv,
              * _outputVDeriv;
    };
    
    BindState _currentBindState;
};

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif // FAR_CPU_EVALSTENCILS_CONTROLLER_H
