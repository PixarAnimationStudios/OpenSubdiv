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

#include "../osd/tbbEvalStencilsController.h"

#include <tbb/parallel_for.h>
#include <tbb/task_scheduler_init.h>

#include <cassert>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

#define grain_size  200

TbbEvalStencilsController::TbbEvalStencilsController(int numThreads) {

    _numThreads = numThreads > 0 ? numThreads : tbb::task_scheduler_init::automatic;

   tbb::task_scheduler_init init(numThreads);
}

TbbEvalStencilsController::~TbbEvalStencilsController() {
}


class StencilKernel {

public:
    enum Mode { UNDEFINED, POINT, U_DERIV, V_DERIV };

    StencilKernel( Far::StencilTables const * stencils,
                   VertexBufferDescriptor ctrlDesc,
                   float const * ctrlData ) :
        _stencils(stencils),
        _mode(UNDEFINED),
        _ctrlDesc(ctrlDesc),
        _length(0),
        _outStride(0),
        _outData(0) {

        _ctrlData = ctrlData + ctrlDesc.offset;
    }

    bool SetOutput(Mode mode, VertexBufferDescriptor outDesc, float * outData) {

        if (_ctrlDesc.CanEval(outDesc)) {
            _mode = mode;
            _length = outDesc.length;
            _outStride = outDesc.stride;
            _outData = outData + outDesc.offset;
            return true;
        }
        return false;
    }

    void operator() (tbb::blocked_range<int> const &r) const {

        assert(_stencils and _ctrlData and _length and _outStride and _outData);

        int offset = _stencils->GetOffsets()[r.begin()];

        int const * sizes = &_stencils->GetSizes()[r.begin()],
                  * index = &_stencils->GetControlIndices()[offset];

        float const * weight;

        switch (_mode) {
            case POINT   : weight = &_stencils->GetWeights()[offset]; break;
            case U_DERIV : weight = &_stencils->GetDuWeights()[offset]; break;
            case V_DERIV : weight = &_stencils->GetDvWeights()[offset]; break;
            default:
                return;
        }
        assert( weight);

        float * out = _outData + r.begin() * _outStride;

        for (int i=r.begin(); i<r.end(); ++i, ++sizes) {

            memset( out, 0, _length * sizeof(float) );

            for (int j=0; j<(*sizes); ++j, ++index, ++weight) {

                float const * cv = _ctrlData + (*index)*_ctrlDesc.stride;

                for (int k=0; k<_length; ++k) {
                    out[k] += cv[k] * (*weight);
                }
            }
            out+=_outStride;
        }
    }

private:
    Far::StencilTables const * _stencils;

    Mode _mode;

    VertexBufferDescriptor _ctrlDesc;
    float const * _ctrlData;

    int _length,
        _outStride;

    float * _outData;
};

int
TbbEvalStencilsController::_UpdateValues( CpuEvalStencilsContext * context ) {

    Far::StencilTables const * stencils = context->GetStencilTables();
    if (not stencils)
        return 0;

    int nstencils = stencils->GetNumStencils();
    if (not nstencils)
        return 0;

    StencilKernel kernel( stencils, _currentBindState.controlDataDesc,
                                    _currentBindState.controlData );


    if (not kernel.SetOutput( StencilKernel::POINT,
                              _currentBindState.outputDataDesc,
                              _currentBindState.outputData ))
        return 0;

    tbb::blocked_range<int> range(0, nstencils, grain_size);

    tbb::parallel_for(range, kernel);

    return nstencils;
}

int
TbbEvalStencilsController::_UpdateDerivs( CpuEvalStencilsContext * context ) {

    Far::StencilTables const * stencils = context->GetStencilTables();
    if (not stencils)
        return 0;

    int nstencils = stencils->GetNumStencils();
    if (not nstencils)
        return 0;

    tbb::blocked_range<int> range(0, nstencils, grain_size);

    StencilKernel kernel( stencils, _currentBindState.controlDataDesc,
                                    _currentBindState.controlData );

    if (not kernel.SetOutput( StencilKernel::U_DERIV,
                              _currentBindState.outputDuDesc,
                              _currentBindState.outputUDeriv ) )
        return 0;

    tbb::parallel_for(range, kernel);

    if (not kernel.SetOutput( StencilKernel::V_DERIV,
                              _currentBindState.outputDvDesc,
                              _currentBindState.outputVDeriv ) )
        return 0;

    tbb::parallel_for(range, kernel);

    return nstencils;
}

void
TbbEvalStencilsController::Synchronize() {
}

} // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
