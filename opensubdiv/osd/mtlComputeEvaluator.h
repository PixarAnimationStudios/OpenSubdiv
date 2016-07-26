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

#ifndef OPENSUBDIV3_OSD_MTL_COMPUTE_EVALUATOR_H
#define OPENSUBDIV3_OSD_MTL_COMPUTE_EVALUATOR_H

#include "../version.h"
#include "../osd/types.h"
#include "../osd/bufferDescriptor.h"
#include "../osd/mtlCommon.h"

@protocol MTLDevice;
@protocol MTLBuffer;
@protocol MTLLibrary;
@protocol MTLComputePipelineState;

namespace OpenSubdiv
{
    namespace OPENSUBDIV_VERSION
    {
        namespace Far
        {
            class StencilTable;
            class PatchTable;
            class LimitStencilTable;
        }

        namespace Osd
        {
            class MTLStencilTable
            {
            public:
                template<typename STENCIL_TABLE, typename DEVICE_CONTEXT>
                static MTLStencilTable* Create(STENCIL_TABLE* stencilTable,
                                               DEVICE_CONTEXT context)
                {
                    return new MTLStencilTable(stencilTable, context);
                }


                MTLStencilTable(Far::StencilTable const* stencilTable, MTLContext* context);
                MTLStencilTable(Far::LimitStencilTable const* stencilTable, MTLContext* context);
                ~MTLStencilTable();


                id<MTLBuffer> GetSizesBuffer() const { return _sizesBuffer; }
                id<MTLBuffer> GetOffsetsBuffer() const { return _offsetsBuffer; }
                id<MTLBuffer> GetIndicesBuffer() const  { return _indicesBuffer; }
                id<MTLBuffer> GetWeightsBuffer() const { return _weightsBuffer; }
                id<MTLBuffer> GetDuWeightsBuffer() const { return _duWeightsBuffer; }
                id<MTLBuffer> GetDvWeightsBuffer() const { return _dvWeightsBuffer; }

                int GetNumStencils() const  { return _numStencils; }

            private:
                id<MTLBuffer> _sizesBuffer;
                id<MTLBuffer> _offsetsBuffer;
                id<MTLBuffer> _indicesBuffer;
                id<MTLBuffer> _weightsBuffer;
                id<MTLBuffer> _duWeightsBuffer;
                id<MTLBuffer> _dvWeightsBuffer;

                int _numStencils;
            };

            class MTLComputeEvaluator
            {
            public:
                typedef bool Instantiatable;

                static MTLComputeEvaluator * Create(BufferDescriptor const &srcDesc,
                                                      BufferDescriptor const &dstDesc,
                                                      BufferDescriptor const &duDesc,
                                                      BufferDescriptor const &dvDesc,
                                                      MTLContext* context);

                static MTLComputeEvaluator * Create(BufferDescriptor const &srcDesc,
                                                      BufferDescriptor const &dstDesc,
                                                      BufferDescriptor const &duDesc,
                                                      BufferDescriptor const &dvDesc,
                                                      BufferDescriptor const &duuDesc,
                                                      BufferDescriptor const &duvDesc,
                                                      BufferDescriptor const &dvvDesc,
                                                      MTLContext* context);

                MTLComputeEvaluator();
                ~MTLComputeEvaluator();

                /// ----------------------------------------------------------------------
                ///
                ///   Stencil evaluations with StencilTable
                ///
                /// ----------------------------------------------------------------------

                /// \brief Generic static compute function. This function has a same
                ///        signature as other device kernels have so that it can be called
                ///        transparently from OsdMesh template interface.
                ///
                /// @param srcBuffer      Input primvar buffer.
                ///                       must have BindVBO() method returning an
                ///                       MTLBuffer object of source data
                ///
                /// @param srcDesc        vertex buffer descriptor for the input buffer
                ///
                /// @param dstBuffer      Output primvar buffer
                ///                       must have BindVBO() method returning an
                ///                       MTLBuffer object of destination data
                ///
                /// @param dstDesc        vertex buffer descriptor for the output buffer
                ///
                /// @param stencilTable   stencil table to be applied. The table must have
                ///                       MTLBuffer interfaces.
                ///
                /// @param instance       cached compiled instance. Clients are supposed to
                ///                       pre-compile an instance of this class and provide
                ///                       to this function. If it's null the kernel still
                ///                       compute by instantiating on-demand kernel although
                ///                       it may cause a performance problem.
                ///
                /// @param deviceContext  used to obtain the MTLDevice objcet and command queue 
                ///                       to obtain command buffers from.
                ///
                template <typename SRC_BUFFER, typename DST_BUFFER, typename STENCIL_TABLE>
                static bool EvalStencils(
                                         SRC_BUFFER *srcBuffer, BufferDescriptor const &srcDesc,
                                         DST_BUFFER *dstBuffer, BufferDescriptor const &dstDesc,
                                         STENCIL_TABLE const *stencilTable,
                                         MTLComputeEvaluator const *instance,
                                         MTLContext* context)
                {
                    if (instance)
                    {
                        return instance->EvalStencils(srcBuffer, srcDesc,
                                                      dstBuffer, dstDesc,
                                                      stencilTable,
                                                      context);
                    }
                    else
                    {
                        // Create an instace on demand (slow)
                        instance = Create(srcDesc, dstDesc,
                                          BufferDescriptor(),
                                          BufferDescriptor(),
                                          context);
                        if (instance)
                        {
                            bool r = instance->EvalStencils(srcBuffer, srcDesc,
                                                            dstBuffer, dstDesc,
                                                            stencilTable,
                                                            context);
                            delete instance;
                            return r;
                        }
                        return false;
                    }
                }

                /// \brief Generic static compute function. This function has a same
                ///        signature as other device kernels have so that it can be called
                ///        transparently from OsdMesh template interface.
                ///
                /// @param srcBuffer      Input primvar buffer.
                ///                       must have BindVBO() method returning an
                ///                       MTLBuffer object of source data
                ///
                /// @param srcDesc        vertex buffer descriptor for the input buffer
                ///
                /// @param dstBuffer      Output primvar buffer
                ///                       must have BindVBO() method returning an
                ///                       MTLBuffer object of destination data
                ///
                /// @param dstDesc        vertex buffer descriptor for the dstBuffer
                ///
                /// @param duBuffer       Output U-derivative buffer
                ///                       must have BindVBO() method returning an
                ///                       MTLBuffer object of destination data
                ///
                /// @param duDesc         vertex buffer descriptor for the duBuffer
                ///
                /// @param dvBuffer       Output V-derivative buffer
                ///                       must have BindVBO() method returning an
                ///                       MTLBuffer object of destination data
                ///
                /// @param dvDesc         vertex buffer descriptor for the dvBuffer
                ///
                /// @param stencilTable   stencil table to be applied. The table must have
                ///                       SSBO interfaces.
                ///
                /// @param instance       cached compiled instance. Clients are supposed to
                ///                       pre-compile an instance of this class and provide
                ///                       to this function. If it's null the kernel still
                ///                       compute by instantiating on-demand kernel although
                ///                       it may cause a performance problem.
                ///
                /// @param deviceContext  used to obtain the MTLDevice objcet and command queue 
                ///                       to obtain command buffers from.
                ///
                template <typename SRC_BUFFER, typename DST_BUFFER, typename STENCIL_TABLE>
                static bool EvalStencils(
                                         SRC_BUFFER *srcBuffer, BufferDescriptor const &srcDesc,
                                         DST_BUFFER *dstBuffer, BufferDescriptor const &dstDesc,
                                         DST_BUFFER *duBuffer,  BufferDescriptor const &duDesc,
                                         DST_BUFFER *dvBuffer,  BufferDescriptor const &dvDesc,
                                         STENCIL_TABLE const *stencilTable,
                                         MTLComputeEvaluator const *instance,
                                         MTLContext* context) {

                    if (instance) {
                        return instance->EvalStencils(srcBuffer, srcDesc,
                                                      dstBuffer, dstDesc,
                                                      duBuffer,  duDesc,
                                                      dvBuffer,  dvDesc,
                                                      stencilTable,
                                                      context);
                    } else {
                        // Create a pipeline state on demand (slow)
                        instance = Create(srcDesc, dstDesc, duDesc, dvDesc, context);
                        if (instance) {
                            bool r = instance->EvalStencils(srcBuffer, srcDesc,
                                                            dstBuffer, dstDesc,
                                                            duBuffer,  duDesc,
                                                            dvBuffer,  dvDesc,
                                                            stencilTable,
                                                            context);
                            delete instance;
                            return r;
                        }
                        return false;
                    }
                }

                /// Dispatch the compute pipeline on GPU asynchronously.
                /// returns false if the kernel hasn't been compiled yet.
                template <typename SRC_BUFFER, typename DST_BUFFER, typename STENCIL_TABLE>
                bool EvalStencils(
                                  SRC_BUFFER *srcBuffer, BufferDescriptor const &srcDesc,
                                  DST_BUFFER *dstBuffer, BufferDescriptor const &dstDesc,
                                  STENCIL_TABLE const *stencilTable,
                                  MTLContext* context) const
                {
                    return EvalStencils(srcBuffer->BindMTLBuffer(context), srcDesc,
                                        dstBuffer->BindMTLBuffer(context), dstDesc,
                                        0, BufferDescriptor(),
                                        0, BufferDescriptor(),
                                        stencilTable->GetSizesBuffer(),
                                        stencilTable->GetOffsetsBuffer(),
                                        stencilTable->GetIndicesBuffer(),
                                        stencilTable->GetWeightsBuffer(),
                                        0,
                                        0,
                                        /* start = */ 0,
                                        /* end   = */ stencilTable->GetNumStencils(),
                                        context);
                }

                /// Dispatch the compute pipeline on GPU asynchronously.
                /// returns false if the kernel hasn't been compiled yet.
                template <typename SRC_BUFFER, typename DST_BUFFER, typename STENCIL_TABLE>
                bool EvalStencils(
                                  SRC_BUFFER *srcBuffer, BufferDescriptor const &srcDesc,
                                  DST_BUFFER *dstBuffer, BufferDescriptor const &dstDesc,
                                  DST_BUFFER *duBuffer,  BufferDescriptor const &duDesc,
                                  DST_BUFFER *dvBuffer,  BufferDescriptor const &dvDesc,
                                  STENCIL_TABLE const *stencilTable,
                                  MTLContext* context) const
                {
                    return EvalStencils(srcBuffer->BindVBO(), srcDesc,
                                        dstBuffer->BindVBO(), dstDesc,
                                        duBuffer->BindVBO(),  duDesc,
                                        dvBuffer->BindVBO(),  dvDesc,
                                        stencilTable->GetSizesBuffer(),
                                        stencilTable->GetOffsetsBuffer(),
                                        stencilTable->GetIndicesBuffer(),
                                        stencilTable->GetWeightsBuffer(),
                                        stencilTable->GetDuWeightsBuffer(),
                                        stencilTable->GetDvWeightsBuffer(),
                                        /* start = */ 0,
                                        /* end   = */ stencilTable->GetNumStencils(),
                                        context);
                }

                /// Dispatch the compute pipeline on GPU asynchronously.
                /// returns false if the kernel hasn't been compiled yet.
                bool EvalStencils(id<MTLBuffer> srcBuffer, BufferDescriptor const &srcDesc,
                                  id<MTLBuffer> dstBuffer, BufferDescriptor const &dstDesc,
                                  id<MTLBuffer> duBuffer,  BufferDescriptor const &duDesc,
                                  id<MTLBuffer> dvBuffer,  BufferDescriptor const &dvDesc,
                                  id<MTLBuffer> sizesBuffer,
                                  id<MTLBuffer> offsetsBuffer,
                                  id<MTLBuffer> indicesBuffer,
                                  id<MTLBuffer> weightsBuffer,
                                  id<MTLBuffer> duWeightsBuffer,
                                  id<MTLBuffer> dvWeightsBuffer,
                                  int start,
                                  int end,
                                  MTLContext* context) const;


                /// ----------------------------------------------------------------------
                ///
                ///   Limit evaluations with PatchTable
                ///
                /// ----------------------------------------------------------------------
                ///
                /// \brief Generic limit eval function. This function has a same
                ///        signature as other device kernels have so that it can be called
                ///        in the same way.
                ///
                /// @param srcBuffer      Input primvar buffer.
                ///                       must have BindVBO() method returning an 
                ///                       MTLBuffer object of source data
                ///
                /// @param srcDesc        vertex buffer descriptor for the input buffer
                ///
                /// @param dstBuffer      Output primvar buffer
                ///                       must have BindVBO() method returning an
                ///                       MTLBuffer object of destination data
                ///
                /// @param dstDesc        vertex buffer descriptor for the output buffer
                ///
                /// @param numPatchCoords number of patchCoords.
                ///
                /// @param patchCoords    array of locations to be evaluated.
                ///                       must have BindVBO() method returning an
                ///                       array of PatchCoord struct in VBO.
                ///
                /// @param patchTable     MTLPatchTable or equivalent
                ///
                /// @param instance       cached compiled instance. Clients are supposed to
                ///                       pre-compile an instance of this class and provide
                ///                       to this function. If it's null the kernel still
                ///                       compute by instantiating on-demand kernel although
                ///                       it may cause a performance problem.
                ///
                /// @param deviceContext  used to obtain the MTLDevice objcet and command queue 
                ///                       to obtain command buffers from.
                ///
                template <typename SRC_BUFFER, typename DST_BUFFER,
                typename PATCHCOORD_BUFFER, typename PATCH_TABLE>
                static bool EvalPatches(
                                        SRC_BUFFER *srcBuffer, BufferDescriptor const &srcDesc,
                                        DST_BUFFER *dstBuffer, BufferDescriptor const &dstDesc,
                                        int numPatchCoords,
                                        PATCHCOORD_BUFFER *patchCoords,
                                        PATCH_TABLE *patchTable,
                                        MTLComputeEvaluator const *instance,
                                        MTLContext* context) {

                    if (instance) {
                        return instance->EvalPatches(srcBuffer, srcDesc,
                                                     dstBuffer, dstDesc,
                                                     numPatchCoords, patchCoords,
                                                     patchTable,
                                                     context);
                    } else {
                        // Create an instance on demand (slow)
                        instance = Create(srcDesc, dstDesc,
                                          BufferDescriptor(),
                                          BufferDescriptor(), context );
                        if (instance) {
                            bool r = instance->EvalPatches(srcBuffer, srcDesc,
                                                           dstBuffer, dstDesc,
                                                           numPatchCoords, patchCoords,
                                                           patchTable,
                                                           context);
                            delete instance;
                            return r;
                        }
                        return false;
                    }
                }

                /// \brief Generic limit eval function. This function has a same
                ///        signature as other device kernels have so that it can be called
                ///        in the same way.
                ///
                /// @param srcBuffer      Input primvar buffer.
                ///                       must have BindVBO() method returning an
                ///                       MTLBuffer object of source data
                ///
                /// @param srcDesc        vertex buffer descriptor for the input buffer
                ///
                /// @param dstBuffer      Output primvar buffer
                ///                       must have BindVBO() method returning an
                ///                       MTLBuffer object of destination data
                ///
                /// @param dstDesc        vertex buffer descriptor for the output buffer
                ///
                /// @param duBuffer
                ///
                /// @param duDesc
                ///
                /// @param dvBuffer
                ///
                /// @param dvDesc
                ///
                /// @param numPatchCoords number of patchCoords.
                ///
                /// @param patchCoords    array of locations to be evaluated.
                ///                       must have BindVBO() method returning an
                ///                       array of PatchCoord struct in VBO.
                ///
                /// @param patchTable     MTLPatchTable or equivalent
                ///
                /// @param instance       cached compiled instance. Clients are supposed to
                ///                       pre-compile an instance of this class and provide
                ///                       to this function. If it's null the kernel still
                ///                       compute by instantiating on-demand kernel although
                ///                       it may cause a performance problem.
                ///
                /// @param deviceContext  used to obtain the MTLDevice objcet and command queue 
                ///                       to obtain command buffers from.
                ///
                template <typename SRC_BUFFER, typename DST_BUFFER,
                typename PATCHCOORD_BUFFER, typename PATCH_TABLE>
                static bool EvalPatches(
                                        SRC_BUFFER *srcBuffer, BufferDescriptor const &srcDesc,
                                        DST_BUFFER *dstBuffer, BufferDescriptor const &dstDesc,
                                        DST_BUFFER *duBuffer,  BufferDescriptor const &duDesc,
                                        DST_BUFFER *dvBuffer,  BufferDescriptor const &dvDesc,
                                        int numPatchCoords,
                                        PATCHCOORD_BUFFER *patchCoords,
                                        PATCH_TABLE *patchTable,
                                        MTLComputeEvaluator* instance,
                                        MTLContext* context) {

                    if (instance) {
                        return instance->EvalPatches(srcBuffer, srcDesc,
                                                     dstBuffer, dstDesc,
                                                     duBuffer, duDesc,
                                                     dvBuffer, dvDesc,
                                                     numPatchCoords, patchCoords,
                                                     patchTable,
                                                     context);
                    } else {
                        // Create an instance on demand (slow)
                        instance = Create(srcDesc, dstDesc, duDesc, dvDesc, context);
                        if (instance) {
                            bool r = instance->EvalPatches(srcBuffer, srcDesc,
                                                           dstBuffer, dstDesc,
                                                           duBuffer, duDesc,
                                                           dvBuffer, dvDesc,
                                                           numPatchCoords, patchCoords,
                                                           patchTable,
                                                           context);
                            delete instance;
                            return r;
                        }
                        return false;
                    }
                }

                /// \brief Generic limit eval function. This function has a same
                ///        signature as other device kernels have so that it can be called
                ///        in the same way.
                ///
                /// @param srcBuffer      Input primvar buffer.
                ///                       must have BindVBO() method returning a
                ///                       const float pointer for read
                ///
                /// @param srcDesc        vertex buffer descriptor for the input buffer
                ///
                /// @param dstBuffer      Output primvar buffer
                ///                       must have BindVBOBuffer() method returning a
                ///                       float pointer for write
                ///
                /// @param dstDesc        vertex buffer descriptor for the output buffer
                ///
                /// @param numPatchCoords number of patchCoords.
                ///
                /// @param patchCoords    array of locations to be evaluated.
                ///                       must have BindVBO() method returning an
                ///                       array of PatchCoord struct in VBO.
                ///
                /// @param patchTable     MTLPatchTable or equivalent
                ///
                /// @param deviceContext  used to obtain the MTLDevice objcet and command queue 
                ///                       to obtain command buffers from.
                ///
                template <typename SRC_BUFFER, typename DST_BUFFER,
                typename PATCHCOORD_BUFFER, typename PATCH_TABLE>
                bool EvalPatches(
                                 SRC_BUFFER *srcBuffer, BufferDescriptor const &srcDesc,
                                 DST_BUFFER *dstBuffer, BufferDescriptor const &dstDesc,
                                 int numPatchCoords,
                                 PATCHCOORD_BUFFER *patchCoords,
                                 PATCH_TABLE *patchTable,
                                 MTLContext* context) const {

                    return EvalPatches(srcBuffer->BindVBO(), srcDesc,
                                       dstBuffer->BindVBO(), dstDesc,
                                       0, BufferDescriptor(),
                                       0, BufferDescriptor(),
                                       numPatchCoords,
                                       patchCoords->BindVBO(),
                                       patchTable->GetPatchArrays(),
                                       patchTable->GetPatchIndexBuffer(),
                                       patchTable->GetPatchParamBuffer(),
                                       context);
                }

                /// \brief Generic limit eval function with derivatives. This function has
                ///        a same signature as other device kernels have so that it can be
                ///        called in the same way.
                ///
                /// @param srcBuffer        Input primvar buffer.
                ///                         must have BindVBO() method returning a
                ///                         const float pointer for read
                ///
                /// @param srcDesc          vertex buffer descriptor for the input buffer
                ///
                /// @param dstBuffer        Output primvar buffer
                ///                         must have BindVBO() method returning a
                ///                         float pointer for write
                ///
                /// @param dstDesc          vertex buffer descriptor for the output buffer
                ///
                /// @param duBuffer         Output U-derivatives buffer
                ///                         must have BindVBO() method returning a
                ///                         float pointer for write
                ///
                /// @param duDesc           vertex buffer descriptor for the duBuffer
                ///
                /// @param dvBuffer         Output V-derivatives buffer
                ///                         must have BindVBO() method returning a
                ///                         float pointer for write
                ///
                /// @param dvDesc           vertex buffer descriptor for the dvBuffer
                ///
                /// @param numPatchCoords   number of patchCoords.
                ///
                /// @param patchCoords      array of locations to be evaluated.
                ///
                /// @param patchTable       MTLPatchTable or equivalent
                ///
                /// @param deviceContext  used to obtain the MTLDevice objcet and command queue 
                ///                       to obtain command buffers from.
                ///
                template <typename SRC_BUFFER, typename DST_BUFFER,
                typename PATCHCOORD_BUFFER, typename PATCH_TABLE>
                bool EvalPatches(
                                 SRC_BUFFER *srcBuffer, BufferDescriptor const &srcDesc,
                                 DST_BUFFER *dstBuffer, BufferDescriptor const &dstDesc,
                                 DST_BUFFER *duBuffer,  BufferDescriptor const &duDesc,
                                 DST_BUFFER *dvBuffer,  BufferDescriptor const &dvDesc,
                                 int numPatchCoords,
                                 PATCHCOORD_BUFFER *patchCoords,
                                 PATCH_TABLE *patchTable,
                                 MTLContext* context) const {

                    return EvalPatches(srcBuffer->BindVBO(), srcDesc,
                                       dstBuffer->BindVBO(), dstDesc,
                                       duBuffer->BindVBO(),  duDesc,
                                       dvBuffer->BindVBO(),  dvDesc,
                                       numPatchCoords,
                                       patchCoords->BindVBO(),
                                       patchTable->GetPatchArrays(),
                                       patchTable->GetPatchIndexBuffer(),
                                       patchTable->GetPatchParamBuffer(),
                                       context);
                }

                bool EvalPatches(id<MTLBuffer> srcBuffer, BufferDescriptor const &srcDesc,
                                 id<MTLBuffer> dstBuffer, BufferDescriptor const &dstDesc,
                                 id<MTLBuffer> duBuffer, BufferDescriptor const &duDesc,
                                 id<MTLBuffer> dvBuffer, BufferDescriptor const &dvDesc,
                                 int numPatchCoords,
                                 id<MTLBuffer> patchCoordsBuffer,
                                 const PatchArrayVector &patchArrays,
                                 id<MTLBuffer> patchIndexBuffer,
                                 id<MTLBuffer> patchParamsBuffer,
                                 MTLContext* context) const;


                /// \brief Generic limit eval function. This function has a same
                ///        signature as other device kernels have so that it can be called
                ///        in the same way.
                ///
                /// @param srcBuffer      Input primvar buffer.
                ///                       must have BindVBO() method returning a GL
                ///                       buffer object of source data
                ///
                /// @param srcDesc        vertex buffer descriptor for the input buffer
                ///
                /// @param dstBuffer      Output primvar buffer
                ///                       must have BindVBO() method returning a GL
                ///                       buffer object of destination data
                ///
                /// @param dstDesc        vertex buffer descriptor for the output buffer
                ///
                /// @param numPatchCoords number of patchCoords.
                ///
                /// @param patchCoords    array of locations to be evaluated.
                ///                       must have BindVBO() method returning an
                ///                       array of PatchCoord struct in VBO.
                ///
                /// @param patchTable     MTLPatchTable or equivalent
                ///
                /// @param instance       cached compiled instance. Clients are supposed to
                ///                       pre-compile an instance of this class and provide
                ///                       to this function. If it's null the kernel still
                ///                       compute by instantiating on-demand kernel although
                ///                       it may cause a performance problem.
                ///
                /// @param deviceContext  used to obtain the MTLDevice objcet and command queue 
                ///                       to obtain command buffers from.
                ///
                template <typename SRC_BUFFER, typename DST_BUFFER,
                          typename PATCHCOORD_BUFFER, typename PATCH_TABLE>
                static bool EvalPatchesVarying(
                    SRC_BUFFER *srcBuffer, BufferDescriptor const &srcDesc,
                    DST_BUFFER *dstBuffer, BufferDescriptor const &dstDesc,
                    int numPatchCoords,
                    PATCHCOORD_BUFFER *patchCoords,
                    PATCH_TABLE *patchTable,
                    MTLComputeEvaluator const *instance,
                    MTLContext* deviceContext) {

                    if (instance) {
                        return instance->EvalPatchesVarying(
                                                     srcBuffer, srcDesc,
                                                     dstBuffer, dstDesc,
                                                     numPatchCoords, patchCoords,
                                                     patchTable,
                                                     deviceContext);
                    } else {
                        // Create an instance on demand (slow)
                        instance = Create(srcDesc, dstDesc,
                                          BufferDescriptor(),
                                          BufferDescriptor(),
                                          deviceContext);
                        if (instance) {
                            bool r = instance->EvalPatchesVarying(
                                                           srcBuffer, srcDesc,
                                                           dstBuffer, dstDesc,
                                                           numPatchCoords, patchCoords,
                                                           patchTable,
                                                           deviceContext);
                            delete instance;
                            return r;
                        }
                        return false;
                    }
                }

                /// \brief Generic limit eval function. This function has a same
                ///        signature as other device kernels have so that it can be called
                ///        in the same way.
                ///
                /// @param srcBuffer      Input primvar buffer.
                ///                       must have BindVBO() method returning a
                ///                       const float pointer for read
                ///
                /// @param srcDesc        vertex buffer descriptor for the input buffer
                ///
                /// @param dstBuffer      Output primvar buffer
                ///                       must have BindVBOBuffer() method returning a
                ///                       float pointer for write
                ///
                /// @param dstDesc        vertex buffer descriptor for the output buffer
                ///
                /// @param numPatchCoords number of patchCoords.
                ///
                /// @param patchCoords    array of locations to be evaluated.
                ///                       must have BindVBO() method returning an
                ///                       array of PatchCoord struct in VBO.
                ///
                /// @param patchTable     MTLPatchTable or equivalent
                ///
                /// @param deviceContext  used to obtain the MTLDevice objcet and command queue 
                ///                       to obtain command buffers from.
                ///
                template <typename SRC_BUFFER, typename DST_BUFFER,
                          typename PATCHCOORD_BUFFER, typename PATCH_TABLE>
                bool EvalPatchesVarying(
                    SRC_BUFFER *srcBuffer, BufferDescriptor const &srcDesc,
                    DST_BUFFER *dstBuffer, BufferDescriptor const &dstDesc,
                    int numPatchCoords,
                    PATCHCOORD_BUFFER *patchCoords,
                    PATCH_TABLE *patchTable,
                    MTLContext* deviceContext) const {

                    return EvalPatches(srcBuffer->BindVBO(), srcDesc,
                                       dstBuffer->BindVBO(), dstDesc,
                                       0, BufferDescriptor(),
                                       0, BufferDescriptor(),
                                       numPatchCoords,
                                       patchCoords->BindVBO(),
                                       patchTable->GetVaryingPatchArrays(),
                                       patchTable->GetVaryingPatchIndexBuffer(),
                                       patchTable->GetPatchParamBuffer(),
                                       deviceContext
                                       );
                }

                /// \brief Generic limit eval function. This function has a same
                ///        signature as other device kernels have so that it can be called
                ///        in the same way.
                ///
                /// @param srcBuffer      Input primvar buffer.
                ///                       must have BindVBO() method returning a GL
                ///                       buffer object of source data
                ///
                /// @param srcDesc        vertex buffer descriptor for the input buffer
                ///
                /// @param dstBuffer      Output primvar buffer
                ///                       must have BindVBO() method returning a GL
                ///                       buffer object of destination data
                ///
                /// @param dstDesc        vertex buffer descriptor for the output buffer
                ///
                /// @param numPatchCoords number of patchCoords.
                ///
                /// @param patchCoords    array of locations to be evaluated.
                ///                       must have BindVBO() method returning an
                ///                       array of PatchCoord struct in VBO.
                ///
                /// @param patchTable     MTLPatchTable or equivalent
                ///
                /// @param fvarChannel    face-varying channel
                ///
                /// @param instance       cached compiled instance. Clients are supposed to
                ///                       pre-compile an instance of this class and provide
                ///                       to this function. If it's null the kernel still
                ///                       compute by instantiating on-demand kernel although
                ///                       it may cause a performance problem.
                ///
                /// @param deviceContext  used to obtain the MTLDevice objcet and command queue 
                ///                       to obtain command buffers from.
                ///
                template <typename SRC_BUFFER, typename DST_BUFFER,
                          typename PATCHCOORD_BUFFER, typename PATCH_TABLE>
                static bool EvalPatchesFaceVarying(
                    SRC_BUFFER *srcBuffer, BufferDescriptor const &srcDesc,
                    DST_BUFFER *dstBuffer, BufferDescriptor const &dstDesc,
                    int numPatchCoords,
                    PATCHCOORD_BUFFER *patchCoords,
                    PATCH_TABLE *patchTable,
                    int fvarChannel,
                    MTLComputeEvaluator const *instance,
                    MTLContext* deviceContext) {

                    if (instance) {
                        return instance->EvalPatchesFaceVarying(
                                                     srcBuffer, srcDesc,
                                                     dstBuffer, dstDesc,
                                                     numPatchCoords, patchCoords,
                                                     patchTable, fvarChannel,
                                                     deviceContext);
                    } else {
                        // Create an instance on demand (slow)
                        instance = Create(srcDesc, dstDesc,
                                          BufferDescriptor(),
                                          BufferDescriptor(),
                                          deviceContext);
                        if (instance) {
                            bool r = instance->EvalPatchesFaceVarying(
                                                           srcBuffer, srcDesc,
                                                           dstBuffer, dstDesc,
                                                           numPatchCoords, patchCoords,
                                                           patchTable, fvarChannel,
                                                           deviceContext);
                            delete instance;
                            return r;
                        }
                        return false;
                    }
                }

                /// \brief Generic limit eval function. This function has a same
                ///        signature as other device kernels have so that it can be called
                ///        in the same way.
                ///
                /// @param srcBuffer      Input primvar buffer.
                ///                       must have BindVBO() method returning a
                ///                       const float pointer for read
                ///
                /// @param srcDesc        vertex buffer descriptor for the input buffer
                ///
                /// @param dstBuffer      Output primvar buffer
                ///                       must have BindVBOBuffer() method returning a
                ///                       float pointer for write
                ///
                /// @param dstDesc        vertex buffer descriptor for the output buffer
                ///
                /// @param numPatchCoords number of patchCoords.
                ///
                /// @param patchCoords    array of locations to be evaluated.
                ///                       must have BindVBO() method returning an
                ///                       array of PatchCoord struct in VBO.
                ///
                /// @param patchTable     MTLPatchTable or equivalent
                ///
                /// @param fvarChannel    face-varying channel
                ///
                /// @param deviceContext  used to obtain the MTLDevice objcet and command queue 
                ///                       to obtain command buffers from.
                ///
                template <typename SRC_BUFFER, typename DST_BUFFER,
                          typename PATCHCOORD_BUFFER, typename PATCH_TABLE>
                bool EvalPatchesFaceVarying(
                    SRC_BUFFER *srcBuffer, BufferDescriptor const &srcDesc,
                    DST_BUFFER *dstBuffer, BufferDescriptor const &dstDesc,
                    int numPatchCoords,
                    PATCHCOORD_BUFFER *patchCoords,
                    PATCH_TABLE *patchTable,
                    MTLContext* deviceContext,
                    int fvarChannel = 0
                    ) const {

                    return EvalPatches(srcBuffer->BindVBO(), srcDesc,
                                       dstBuffer->BindVBO(), dstDesc,
                                       0, BufferDescriptor(),
                                       0, BufferDescriptor(),
                                       numPatchCoords,
                                       patchCoords->BindVBO(),
                                       patchTable->GetFVarPatchArrays(fvarChannel),
                                       patchTable->GetFVarPatchIndexBuffer(fvarChannel),
                                       patchTable->GetFVarPatchParamBuffer(fvarChannel),
                                       deviceContext);
                }

                /// Configure compute pipline state. Returns false if it fails to create the pipeline state.
                bool Compile(BufferDescriptor const &srcDesc,
                             BufferDescriptor const &dstDesc,
                             BufferDescriptor const &duDesc,
                             BufferDescriptor const &dvDesc,
                             MTLContext* context);

                /// Wait for the dispatched kernel to finish.
                static void Synchronize(MTLContext* context);

                private:

                id<MTLLibrary> _computeLibrary;
                id<MTLComputePipelineState> _evalStencils;
                id<MTLComputePipelineState> _evalPatches;
                id<MTLBuffer> _parameterBuffer;

                int _workGroupSize;
            };
        } //end namespace Osd
    } //end namespace OPENSUBDIV_VERSION
    using namespace OPENSUBDIV_VERSION;
} //end namespace OpenSubdiv
#endif // OPENSUBDIV3_OSD_MTL_COMPUTE_EVALUATOR_H
