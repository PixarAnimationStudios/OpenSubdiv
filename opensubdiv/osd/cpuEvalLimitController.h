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

#ifndef OSD_CPU_EVAL_LIMIT_CONTROLLER_H
#define OSD_CPU_EVAL_LIMIT_CONTROLLER_H

#include "../version.h"

#include "../osd/cpuEvalLimitContext.h"
#include "../osd/vertexDescriptor.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

/// \brief CPU controler for limit surface evaluation.
///
/// A CPU-driven controller that can be called to evaluate samples on the limit
/// surface for a given EvalContext.
///
/// Warning : this eval controller is re-entrant but it breaks the Osd API pattern
/// by requiring client code to bind and unbind the data buffers to the
/// Controller before calling evaluation methods.
///
/// Ex :
/// \code
/// evalCtroller->BindVertexBuffers( ... );
/// evalCtroller->BindVaryingBuffers( ... );
/// evalCtroller->BindFacevaryingBuffers( ... );
///
/// parallel_for( int index=0; i<nsamples; ++index ) {
///    evalCtroller->EvalLimitSample( coord, evalCtxt, index );
/// }
///
/// evalCtroller->Unbind();
/// \endcode
///
class CpuEvalLimitController {

public:
    /// Constructor.
    CpuEvalLimitController();

    /// Destructor.
    ~CpuEvalLimitController();

    /// \brief Binds control vertex data buffer
    ///
    /// @param iDesc   data descriptor shared by all input data buffers
    ///
    /// @param inQ     input vertex data
    ///
    /// @param oDesc   data descriptor shared by all output data buffers
    ///
    /// @param outQ    output vertex data
    ///
    /// @param outdQu  output derivative along "u" of the vertex data (optional)
    ///
    /// @param outdQv  output derivative along "v" of the vertex data (optional)
    ///
    template<class INPUT_BUFFER, class OUTPUT_BUFFER>
    void BindVertexBuffers( VertexBufferDescriptor const & iDesc, INPUT_BUFFER *inQ,
                            VertexBufferDescriptor const & oDesc, OUTPUT_BUFFER *outQ,
                                                                     OUTPUT_BUFFER *outdQu=0,
                                                                     OUTPUT_BUFFER *outdQv=0 ) {
        _currentBindState.vertexData.inDesc = iDesc;
        _currentBindState.vertexData.in = inQ ? inQ->BindCpuBuffer() : 0;

        _currentBindState.vertexData.outDesc = oDesc;
        _currentBindState.vertexData.out = outQ ? outQ->BindCpuBuffer() : 0;
        _currentBindState.vertexData.outDu = outdQu ? outdQu->BindCpuBuffer() : 0;
        _currentBindState.vertexData.outDv = outdQv ? outdQv->BindCpuBuffer() : 0;
    }

    /// \brief Binds the varying-interpolated data streams
    ///
    /// @param iDesc  data descriptor shared by all input data buffers
    ///
    /// @param inQ    input varying data
    ///
    /// @param oDesc  data descriptor shared by all output data buffers
    ///
    /// @param outQ   output varying data
    ///
    template<class INPUT_BUFFER, class OUTPUT_BUFFER>
    void BindVaryingBuffers( VertexBufferDescriptor const & iDesc, INPUT_BUFFER *inQ,
                             VertexBufferDescriptor const & oDesc, OUTPUT_BUFFER *outQ ) {
        _currentBindState.varyingData.inDesc = iDesc;
        _currentBindState.varyingData.in = inQ ? inQ->BindCpuBuffer() : 0;

        _currentBindState.varyingData.outDesc = oDesc;
        _currentBindState.varyingData.out = outQ ? outQ->BindCpuBuffer() : 0;
    }

    /// \brief Binds the face-varying-interpolated data streams
    ///
    /// Note : currently we only support bilinear boundary interpolation rules
    /// for face-varying data. Although Hbr supports 3 addition smooth rule sets,
    /// the feature-adaptive patch interpolation code currently does not support
    /// them, and neither does this EvalContext
    ///
    /// @param iDesc  data descriptor shared by all input data buffers
    ///
    /// @param oDesc  data descriptor shared by all output data buffers
    ///
    /// @param outQ   output face-varying data
    ///
    template<class OUTPUT_BUFFER>
    void BindFacevaryingBuffers( VertexBufferDescriptor const & iDesc,
                                 VertexBufferDescriptor const & oDesc, OUTPUT_BUFFER *outQ ) {
        _currentBindState.facevaryingData.inDesc = iDesc;

        _currentBindState.facevaryingData.outDesc = oDesc;
        _currentBindState.facevaryingData.out = outQ ? outQ->BindCpuBuffer() : 0;
    }

    /// \brief Vertex interpolation of a single sample at the limit
    ///
    /// Evaluates "vertex" interpolation of a single sample on the surface limit.
    ///
    /// This function is re-entrant but does not require binding the
    /// output vertex buffers. Pointers to memory where the data is
    /// output are explicitly passed to the function.
    ///
    /// @param coord    location on the limit surface to be evaluated
    ///
    /// @param context  the EvalLimitContext that the controller will evaluate
    ///
    /// @param outDesc  data descriptor (offset, length, stride)
    ///
    /// @param outQ    output vertex data
    ///
    /// @param outDQU  output derivative along "u" of the vertex data (optional)
    ///
    /// @param outDQV  output derivative along "v" of the vertex data (optional)
    ///
    /// @return 1 if the sample was found
    ///
    int EvalLimitSample( EvalCoords const & coord,
                         CpuEvalLimitContext * context,
                         VertexBufferDescriptor const & outDesc,
                         float * outQ,
                         float * outDQU,
                         float * outDQV ) const;

    /// \brief Vertex interpolation of samples at the limit
    ///
    /// Evaluates "vertex" interpolation of a sample on the surface limit.
    ///
    /// @param coords   location on the limit surface to be evaluated
    ///
    /// @param context  the EvalLimitContext that the controller will evaluate
    ///
    /// @param index    the index of the vertex in the output buffers bound to the
    ///                 context
    ///
    /// @return the number of samples found (0 if the location was tagged as a hole
    ///         or the coordinate was invalid)
    ///
    int EvalLimitSample( EvalCoords const & coords,
                         CpuEvalLimitContext * context,
                         unsigned int index ) const {
        if (not context)
            return 0;

        int n = _EvalLimitSample( coords, context, index );

        return n;
    }

    void Unbind() {
        _currentBindState.Reset();
    }

protected:


    // Vertex interpolated streams
    struct VertexData {

        VertexData() : in(0), out(0), outDu(0), outDv(0) { }


        void Reset() {
            in = out = outDu = outDv = NULL;
            inDesc.Reset();
            outDesc.Reset();
        }

        VertexBufferDescriptor inDesc,
                                  outDesc;
        float * in,
              * out,
              * outDu,
              * outDv;
    };

    // Varying interpolated streams
    struct VaryingData {

        VaryingData() : in(0), out(0) { }


        void Reset() {
            in = out = NULL;
            inDesc.Reset();
            outDesc.Reset();
        }

        VertexBufferDescriptor inDesc,
                                  outDesc;
        float * in,
              * out;
    };

    // Facevarying interpolated streams
    struct FacevaryingData {

        FacevaryingData() : out(0) { }

        void Reset() {
            out = NULL;
            inDesc.Reset();
            outDesc.Reset();
        }

        VertexBufferDescriptor inDesc,
                                  outDesc;
        float * out;
    };


private:

    int _EvalLimitSample( EvalCoords const & coords,
                          CpuEvalLimitContext * context,
                          unsigned int index ) const;

    // Bind state is a transitional state during refinement.
    // It doesn't take an ownership of vertex buffers.
    struct BindState {

        BindState() { }

        void Reset() {
            vertexData.Reset();
            varyingData.Reset();
            facevaryingData.Reset();
        }

        VertexData       vertexData;      // vertex interpolated data descriptor
        VaryingData      varyingData;     // varying interpolated data descriptor
        FacevaryingData  facevaryingData; // face-varying interpolated data descriptor
    };

    BindState _currentBindState;
};

} // end namespace Osd

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* OSD_CPU_EVAL_LIMIT_CONTROLLER_H */
