//
//   Copyright 2015 Pixar
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

#ifndef OPENSUBDIV_OSD_TBB_EVALUATOR_H
#define OPENSUBDIV_OSD_TBB_EVALUATOR_H

#include "../version.h"
#include "../osd/vertexDescriptor.h"

#include <cstddef>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

class TbbEvaluator {
public:
    /// \brief Generic static stencil eval function. This function has a same
    ///        signature as other device kernels have so that it can be called
    ///        transparently from OsdMesh template interface.
    ///
    /// @param srcBuffer      Input primvar buffer.
    ///                       must have BindCpuBuffer() method returning a
    ///                       const float pointer for read
    ///
    /// @param srcDesc        vertex buffer descriptor for the input buffer
    ///
    /// @param dstBuffer      Output primvar buffer
    ///                       must have BindCpuBuffer() method returning a
    ///                       float pointer for write
    ///
    /// @param dstDesc        vertex buffer descriptor for the output buffer
    ///
    /// @param stencilTable   stencil table to be applied.
    ///
    /// @param instance        not used in the tbb kernel
    ///                       (declared as a typed pointer to prevent
    ///                        undesirable template resolution)
    ///
    /// @param deviceContext  not used in the tbb kernel
    ///
    template <typename VERTEX_BUFFER, typename STENCIL_TABLE>
    static bool EvalStencils(VERTEX_BUFFER *srcVertexBuffer,
                        VertexBufferDescriptor const &srcDesc,
                        VERTEX_BUFFER *dstVertexBuffer,
                        VertexBufferDescriptor const &dstDesc,
                        STENCIL_TABLE const *stencilTable,
                        TbbEvaluator const *instance = NULL,
                        void *deviceContext = NULL) {
        (void)instance;   // unused
        (void)deviceContext;  // unused

        return EvalStencils(srcVertexBuffer->BindCpuBuffer(),
                            srcDesc,
                            dstVertexBuffer->BindCpuBuffer(),
                            dstDesc,
                            &stencilTable->GetSizes()[0],
                            &stencilTable->GetOffsets()[0],
                            &stencilTable->GetControlIndices()[0],
                            &stencilTable->GetWeights()[0],
                            /*start = */ 0,
                            /*end   = */ stencilTable->GetNumStencils());
    }

    static bool EvalStencils(const float *src,
                             VertexBufferDescriptor const &srcDesc,
                             float *dst,
                             VertexBufferDescriptor const &dstDesc,
                             const unsigned char *sizes,
                             const int *offsets,
                             const int *indices,
                             const float *weights,
                             int start,
                             int end);

    static void Synchronize(void *deviceContext = NULL);

    static void SetNumThreads(int numThreads);
};


}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv


#endif  // OPENSUBDIV_OSD_TBB_EVALUATOR_H
