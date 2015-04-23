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

#ifndef FAR_OMP_EVALSTENCILS_CONTROLLER_H
#define FAR_OMP_EVALSTENCILS_CONTROLLER_H

#include "../version.h"

#include "../osd/cpuEvalStencilsContext.h"

#ifdef OPENSUBDIV_HAS_OPENMP
    #include <omp.h>
#endif

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

///
/// \brief CPU stencils evaluation controller
///
/// CpuStencilsController is a compute controller class to launch
/// single threaded CPU stencil evalution kernels.
///
/// Controller entities execute requests from Context instances that they share
/// common interfaces with. Controllers are attached to discrete compute devices
/// and share the devices resources with Context entities.
///
class OmpEvalStencilsController {
public:

    /// \brief Constructor.
    ///
    /// @param numThreads specifies how many openmp parallel threads to use.
    ///                   -1 attempts to use all available processors.
    ///
    OmpEvalStencilsController(int numThreads=-1);

    /// \brief Destructor.
    ~OmpEvalStencilsController();


    /// \brief Applies stencil weights to the control vertex data
    ///
    /// Applies the stencil weights to the control vertex data to evaluate the
    /// interpolated limit positions at the parametric locations of the stencils
    ///
    /// @param context          the CpuEvalStencilsContext with the stencil weights
    ///
    /// @param controlDataDesc  vertex buffer descriptor for the control vertex data
    ///
    /// @param controlVertices  vertex buffer with the control vertices data
    ///
    /// @param outputDataDesc   vertex buffer descriptor for the output vertex data
    ///
    /// @param outputData       output vertex buffer for the interpolated data
    ///
    template<class CONTROL_BUFFER, class OUTPUT_BUFFER>
    int UpdateValues( CpuEvalStencilsContext * context,
                      VertexBufferDescriptor const & controlDataDesc, CONTROL_BUFFER *controlVertices,
                      VertexBufferDescriptor const & outputDataDesc, OUTPUT_BUFFER *outputData ) {

        if (not context->GetStencilTables()->GetNumStencils())
            return 0;

        omp_set_num_threads(_numThreads);

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
    /// @param context          the CpuEvalStencilsContext with the stencil weights
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
    int UpdateDerivs( CpuEvalStencilsContext * context,
                      VertexBufferDescriptor const & controlDataDesc, CONTROL_BUFFER *controlVertices,
                      VertexBufferDescriptor const & outputDuDesc, OUTPUT_BUFFER *outputDuData,
                      VertexBufferDescriptor const & outputDvDesc, OUTPUT_BUFFER *outputDvData ) {

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
    void bindControlData(VertexBufferDescriptor const & controlDataDesc, VERTEX_BUFFER *controlData ) {

        _currentBindState.controlData = controlData ? controlData->BindCpuBuffer() : 0;
        _currentBindState.controlDataDesc = controlDataDesc;

    }

    /// \brief Binds output vertex data buffer
    template<class VERTEX_BUFFER>
    void bindOutputData( VertexBufferDescriptor const & outputDataDesc, VERTEX_BUFFER *outputData ) {

        _currentBindState.outputData = outputData ? outputData->BindCpuBuffer() : 0;
        _currentBindState.outputDataDesc = outputDataDesc;
    }

    /// \brief Binds output derivative vertex data buffer
    template<class VERTEX_BUFFER>
    void bindOutputDerivData( VertexBufferDescriptor const & outputDuDesc, VERTEX_BUFFER *outputDu,
                              VertexBufferDescriptor const & outputDvDesc, VERTEX_BUFFER *outputDv ) {

        _currentBindState.outputUDeriv = outputDu ? outputDu ->BindCpuBuffer() : 0;
        _currentBindState.outputVDeriv = outputDv ? outputDv->BindCpuBuffer() : 0;
        _currentBindState.outputDuDesc = outputDuDesc;
        _currentBindState.outputDvDesc = outputDvDesc;
    }

    /// \brief Unbinds any previously bound vertex and varying data buffers.
    void unbind() {
        _currentBindState.Reset();
    }

private:

    int _UpdateValues( CpuEvalStencilsContext * context );
    int _UpdateDerivs( CpuEvalStencilsContext * context );

    int _numThreads;

    // Bind state is a transitional state during refinement.
    // It doesn't take an ownership of vertex buffers.
    struct BindState {

        BindState() : controlData(0), outputData(0), outputUDeriv(0), outputVDeriv(0) { }

        void Reset() {
            controlData = outputData = outputUDeriv = outputVDeriv = NULL;
            controlDataDesc.Reset();
            outputDataDesc.Reset();
            outputDuDesc.Reset();
            outputDvDesc.Reset();
        }

        // transient mesh data
        VertexBufferDescriptor controlDataDesc,
                                  outputDataDesc,
                                  outputDuDesc,
                                  outputDvDesc;

        float * controlData,
              * outputData,
              * outputUDeriv,
              * outputVDeriv;
    };

    BindState _currentBindState;
};

} // end namespace Osd

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif // FAR_OMP_EVALSTENCILS_CONTROLLER_H
