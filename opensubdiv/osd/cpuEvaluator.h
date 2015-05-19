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

#pragma once
#ifndef OPENSUBDIV3_OSD_CPU_EVALUATOR_H
#define OPENSUBDIV3_OSD_CPU_EVALUATOR_H

#include "../version.h"

#include <cstddef>
#include <vector>
#include "../osd/vertexDescriptor.h"
#include "../far/patchTables.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

/// \brief Coordinates set on a patch table
///     XXX: this is a temporary structure, exists during Osd refactoring work.
///
struct PatchCoord {
    /// \brief Constructor
    ///
    /// @param p patch handle
    ///
    /// @param s parametric location on the patch
    ///
    /// @param t parametric location on the patch
    ///
    PatchCoord(Far::PatchTables::PatchHandle handle, float s, float t) :
        handle(handle), s(s), t(t) { }

    Far::PatchTables::PatchHandle handle; ///< patch handle
    float s, t;              ///< parametric location on patch
};

typedef std::vector<PatchCoord> PatchCoordArray;


class CpuEvaluator {
public:
    /// \brief Generic static eval stencils function. This function has a same
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
    /// @param instance       not used in the cpu kernel
    ///                       (declared as a typed pointer to prevent
    ///                        undesirable template resolution)
    ///
    /// @param deviceContext  not used in the cpu kernel
    ///
    template <typename SRC_BUFFER, typename DST_BUFFER, typename STENCIL_TABLE>
    static bool EvalStencils(SRC_BUFFER *srcBuffer,
                             VertexBufferDescriptor const &srcDesc,
                             DST_BUFFER *dstBuffer,
                             VertexBufferDescriptor const &dstDesc,
                             STENCIL_TABLE const *stencilTable,
                             const CpuEvaluator *instance = NULL,
                             void * deviceContext = NULL) {
        (void)instance;   // unused
        (void)deviceContext;   // unused

        return EvalStencils(srcBuffer->BindCpuBuffer(),
                            srcDesc,
                            dstBuffer->BindCpuBuffer(),
                            dstDesc,
                            &stencilTable->GetSizes()[0],
                            &stencilTable->GetOffsets()[0],
                            &stencilTable->GetControlIndices()[0],
                            &stencilTable->GetWeights()[0],
                            /*start = */ 0,
                            /*end   = */ stencilTable->GetNumStencils());
    }

    /// stencil evaluate function.
    static bool EvalStencils(const float *src,
                             VertexBufferDescriptor const &srcDesc,
                             float *dst,
                             VertexBufferDescriptor const &dstDesc,
                             const int * sizes,
                             const int * offsets,
                             const int * indices,
                             const float * weights,
                             int start,
                             int end);

    template <typename SRC_BUFFER, typename DST_BUFFER, typename STENCIL_TABLE>
    static bool EvalStencils(SRC_BUFFER *srcBuffer,
                             VertexBufferDescriptor const &srcDesc,
                             DST_BUFFER *dstBuffer,
                             VertexBufferDescriptor const &dstDesc,
                             DST_BUFFER *dstDuBuffer,
                             VertexBufferDescriptor const &dstDuDesc,
                             DST_BUFFER *dstDvBuffer,
                             VertexBufferDescriptor const &dstDvDesc,
                             STENCIL_TABLE const *stencilTable,
                             const CpuEvaluator *evaluator = NULL,
                             void * deviceContext = NULL) {
        (void)evaluator;   // unused
        (void)deviceContext;   // unused

        return EvalStencils(srcBuffer->BindCpuBuffer(),
                            srcDesc,
                            dstBuffer->BindCpuBuffer(),
                            dstDesc,
                            dstDuBuffer->BindCpuBuffer(),
                            dstDuDesc,
                            dstDvBuffer->BindCpuBuffer(),
                            dstDvDesc,
                            &stencilTable->GetSizes()[0],
                            &stencilTable->GetOffsets()[0],
                            &stencilTable->GetControlIndices()[0],
                            &stencilTable->GetWeights()[0],
                            &stencilTable->GetDuWeights()[0],
                            &stencilTable->GetDvWeights()[0],
                            /*start = */ 0,
                            /*end   = */ stencilTable->GetNumStencils());
    }

    static bool EvalStencils(const float *src,
                             VertexBufferDescriptor const &srcDesc,
                             float *dst,
                             VertexBufferDescriptor const &dstDesc,
                             float *dstDu,
                             VertexBufferDescriptor const &dstDuDesc,
                             float *dstDv,
                             VertexBufferDescriptor const &dstDvDesc,
                             const int * sizes,
                             const int * offsets,
                             const int * indices,
                             const float * weights,
                             const float * duWeights,
                             const float * dvWeights,
                             int start,
                             int end);

    /// \brief Generic limit eval function. This function has a same
    ///        signature as other device kernels have so that it can be called
    ///        transparently.
    ///
    ///       XXX: This interface is still work in progress. XXX
    ///
    /// @param srcBuffer        Input primvar buffer.
    ///                         must have BindCpuBuffer() method returning a
    ///                         const float pointer for read
    ///
    /// @param srcDesc          vertex buffer descriptor for the input buffer
    ///
    /// @param dstBuffer        Output primvar buffer
    ///                         must have BindCpuBuffer() method returning a
    ///                         float pointer for write
    ///
    /// @param dstDesc          vertex buffer descriptor for the output buffer
    ///
    /// @param patchCoord       array of locations to be evaluated.
    ///
    /// @param patchTable       Far::PatchTable
    ///
    /// @param instanced        not used in the cpu evaluator
    ///
    /// @param deviceContext    not used in the cpu evaluator
    ///
    template <typename SRC_BUFFER, typename DST_BUFFER>
    static int EvalPatches(SRC_BUFFER *srcBuffer,
                           VertexBufferDescriptor const &srcDesc,
                           DST_BUFFER *dstBuffer,
                           VertexBufferDescriptor const &dstDesc,
                           PatchCoordArray const &patchCoords,
                           Far::PatchTables const *patchTable,
                           CpuEvaluator const *instance,
                           void * deviceContext = NULL) {
        (void)instance;   // unused
        (void)deviceContext;   // unused

        return EvalPatches(srcBuffer->BindCpuBuffer(),
                           srcDesc,
                           dstBuffer->BindCpuBuffer(),
                           dstDesc,
                           patchCoords,
                           patchTable);
    }

    /// \brief limit eval function.
    static int EvalPatches(const float *src,
                           VertexBufferDescriptor const &srcDesc,
                           float *dst,
                           VertexBufferDescriptor const &dstDesc,
                           PatchCoordArray const &patchCoords,
                           Far::PatchTables const *patchTable);

    /// \brief synchronize all asynchronous computation invoked on this device.
    static void Synchronize(void * /*deviceContext = NULL*/) {
        // nothing.
    }
};


}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv


#endif  // OPENSUBDIV3_OSD_CPU_EVALUATOR_H
