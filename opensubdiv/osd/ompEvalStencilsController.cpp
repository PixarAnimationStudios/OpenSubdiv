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

#include "../osd/ompEvalStencilsController.h"

#include <cassert>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

OsdOmpEvalStencilsController::OsdOmpEvalStencilsController(int numThreads) {

    _numThreads = (numThreads == -1) ? omp_get_num_procs() : numThreads;
}

OsdOmpEvalStencilsController::~OsdOmpEvalStencilsController() {
}

int
OsdOmpEvalStencilsController::_UpdateValues( OsdCpuEvalStencilsContext * context ) {

    int result=0;

    FarStencilTables const * stencils = context->GetStencilTables();

    int nstencils = stencils->GetNumStencils();
    if (not nstencils)
        return result;

    OsdVertexBufferDescriptor ctrlDesc = context->GetControlDataDescriptor(),
                              outDesc = context->GetOutputDataDescriptor();

    // make sure that we have control data to work with
    if (not ctrlDesc.CanEval(outDesc))
        return 0;

    float const * ctrl = context->GetControlData() + ctrlDesc.offset;

    if (not ctrl)
        return result;

#pragma omp parallel for
    for (int i=0; i<nstencils; ++i) {

        int size = stencils->GetSizes()[i],
            offset = stencils->GetOffsets()[i];

        int const * index = &stencils->GetControlIndices().at(offset);

        float const * weight = &stencils->GetWeights().at(offset);

        float * out = context->GetOutputData() + i * outDesc.stride + outDesc.offset;

        memset(out, 0, outDesc.length*sizeof(float));

        for (int j=0; j<size; ++j, ++index, ++weight) {

            float const * cv = ctrl + (*index)*ctrlDesc.stride;

            for (int k=0; k<outDesc.length; ++k) {
                out[k] += cv[k] * (*weight);
            }
        }
    }

    return nstencils;
}

int
OsdOmpEvalStencilsController::_UpdateDerivs( OsdCpuEvalStencilsContext * context ) {

    int result=0;

    FarStencilTables const * stencils = context->GetStencilTables();

    int nstencils = stencils->GetNumStencils();
    if (not nstencils)
        return result;

    OsdVertexBufferDescriptor ctrlDesc = context->GetControlDataDescriptor(),
                              duDesc = context->GetDuDataDescriptor(),
                              dvDesc = context->GetDvDataDescriptor();

    // make sure that we have control data to work with
    if (not (ctrlDesc.CanEval(duDesc) and ctrlDesc.CanEval(dvDesc)))
        return 0;

    float const * ctrl = context->GetControlData() + ctrlDesc.offset;

    if (not ctrl)
        return result;

#pragma omp parallel for
    for (int i=0; i<nstencils; ++i) {

        int size = stencils->GetSizes()[i],
            offset = stencils->GetOffsets()[i];

        int const * index = &stencils->GetControlIndices().at(offset);

        float const * duweight = &stencils->GetDuWeights().at(offset),
                    * dvweight = &stencils->GetDvWeights().at(offset);

        float * du = context->GetOutputUDerivData() + i * duDesc.stride + duDesc.offset,
              * dv = context->GetOutputVDerivData() + i * dvDesc.stride + dvDesc.offset;

        memset(du, 0, duDesc.length*sizeof(float));
        memset(dv, 0, dvDesc.length*sizeof(float));

        for (int j=0; j<size; ++j, ++index, ++duweight, ++dvweight) {

            float const * cv = ctrl + (*index)*ctrlDesc.stride;

            for (int k=0; k<duDesc.length; ++k) {
                du[k] += cv[k] * (*duweight);
                dv[k] += cv[k] * (*dvweight);
            }
        }
    }

    return nstencils;
}

void
OsdOmpEvalStencilsController::Synchronize() {
}


}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
