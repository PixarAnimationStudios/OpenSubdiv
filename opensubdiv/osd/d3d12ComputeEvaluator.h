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

#ifndef OPENSUBDIV3_OSD_D3D12_COMPUTE_EVALUATOR_H
#define OPENSUBDIV3_OSD_D3D12_COMPUTE_EVALUATOR_H

#include "../version.h"

struct ID3D12CommandQueue;
struct ID3D12Resource;
struct ID3D12ComputeShader;
struct ID3D12PipelineState;
struct ID3D12RootSignature;
struct ID3D10Blob;
struct ID3D12CommandAllocator;
struct ID3D12Fence;
struct ID3D12DescriptorHeap;
struct D3D12_CPU_DESCRIPTOR_HANDLE;

#include "../osd/bufferDescriptor.h"
#include "d3d12commandqueuecontext.h"
#include "d3d12DeferredDeletionUniquePtr.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Far {
    class StencilTable;
}

namespace Osd {

/// \brief D3D11 stencil table
///
/// This class is a D3D11 Shader Resource View representation of
/// Far::StencilTable.
///
/// D3D12ComputeEvaluator consumes this table to apply stencils
///
class D3D12StencilTable {
public:
    template <typename DEVICE_CONTEXT>
    static D3D12StencilTable *Create(Far::StencilTable const *stencilTable,
                                      DEVICE_CONTEXT context) {
        return new D3D12StencilTable(stencilTable, context->GetDeviceContext());
    }

    static D3D12StencilTable *Create(Far::StencilTable const *stencilTable,
                                     D3D12CommandQueueContext *D3D12CommandQueueContext) {
        return new D3D12StencilTable(stencilTable, D3D12CommandQueueContext);
    }

    D3D12StencilTable(Far::StencilTable const *stencilTable,
                      D3D12CommandQueueContext *D3D12CommandQueueContext);

    ~D3D12StencilTable();

    // interfaces needed for D3D12ComputeEvaluator
    CPUDescriptorHandle GetSizesSRV() const { return _sizes; }
    CPUDescriptorHandle GetOffsetsSRV() const { return _offsets; }
    CPUDescriptorHandle GetIndicesSRV() const { return _indices; }
    CPUDescriptorHandle GetWeightsSRV() const { return _weights; }


    int GetNumStencils() const { return _numStencils; }

private:
    DeferredDeletionUniquePtr<ID3D12Resource> _sizesBuffer;
    DeferredDeletionUniquePtr<ID3D12Resource> _offsetsBuffer;
    DeferredDeletionUniquePtr<ID3D12Resource> _indicesBuffer;
    DeferredDeletionUniquePtr<ID3D12Resource> _weightsBuffer;

    CPUDescriptorHandle _sizes;
    CPUDescriptorHandle _offsets;
    CPUDescriptorHandle _indices;
    CPUDescriptorHandle _weights;

    int _numStencils;
};

// ---------------------------------------------------------------------------

class D3D12ComputeEvaluator {
public:
    typedef bool Instantiatable;
    static D3D12ComputeEvaluator * Create(BufferDescriptor const &srcDesc,
                                          BufferDescriptor const &dstDesc,
                                          BufferDescriptor const &duDesc,
                                          BufferDescriptor const &dvDesc,
                                          D3D12CommandQueueContext *D3D12CommandQueueContext);

    static D3D12ComputeEvaluator * Create(BufferDescriptor const &srcDesc,
                                          BufferDescriptor const &dstDesc,
                                          BufferDescriptor const &duDesc,
                                          BufferDescriptor const &dvDesc,
                                          BufferDescriptor const &duuDesc,
                                          BufferDescriptor const &duvDesc,
                                          BufferDescriptor const &dvvDesc,
                                          D3D12CommandQueueContext *D3D12CommandQueueContext);

    /// Constructor.
    D3D12ComputeEvaluator();

    /// Destructor.
    ~D3D12ComputeEvaluator();

    /// \brief Generic static compute function. This function has a same
    ///        signature as other device kernels have so that it can be called
    ///        transparently from OsdMesh template interface.
    ///
    /// @param srcBuffer      Input primvar buffer.
    ///                       must have BindVBO() method returning a
    ///                       const float pointer for read
    ///
    /// @param srcDesc        vertex buffer descriptor for the input buffer
    ///
    /// @param dstBuffer      Output primvar buffer
    ///                       must have BindVBO() method returning a
    ///                       float pointer for write
    ///
    /// @param dstDesc        vertex buffer descriptor for the output buffer
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
    /// @param deviceContext  ID3D11DeviceContext.
    ///
    template <typename SRC_BUFFER, typename DST_BUFFER, typename STENCIL_TABLE>
    static bool EvalStencils(
        SRC_BUFFER *srcBuffer, BufferDescriptor const &srcDesc,
        DST_BUFFER *dstBuffer, BufferDescriptor const &dstDesc,
        STENCIL_TABLE const *stencilTable,
        D3D12ComputeEvaluator const *instance,
        D3D12CommandQueueContext *D3D12CommandQueueContext) {
        if (instance) {
            return instance->EvalStencils(srcBuffer, srcDesc,
                                          dstBuffer, dstDesc,
                                          stencilTable,
                                          D3D12CommandQueueContext);
        } else {
            // Create an instace on demand (slow)
            instance = Create(srcDesc, dstDesc,
                              BufferDescriptor(),
                              BufferDescriptor(),
                              D3D12CommandQueueContext);
            if (instance) {
                bool r = instance->EvalStencils(srcBuffer, srcDesc,
                                                dstBuffer, dstDesc,
                                                stencilTable,
                                                D3D12CommandQueueContext);
                delete instance;
                return r;
            }
            return false;
        }
    }

    /// Dispatch the DX compute kernel on GPU asynchronously.
    /// returns false if the kernel hasn't been compiled yet.
    template <typename SRC_BUFFER, typename DST_BUFFER, typename STENCIL_TABLE>
    bool EvalStencils(
        SRC_BUFFER *srcBuffer, BufferDescriptor const &srcDesc,
        DST_BUFFER *dstBuffer, BufferDescriptor const &dstDesc,
        STENCIL_TABLE const *stencilTable,
        D3D12CommandQueueContext *D3D12CommandQueueContext) const {
        return EvalStencils(srcBuffer->BindD3D12UAV(D3D12CommandQueueContext), srcDesc,
                            dstBuffer->BindD3D12UAV(D3D12CommandQueueContext), dstDesc,
                            stencilTable->GetSizesSRV(),
                            stencilTable->GetOffsetsSRV(),
                            stencilTable->GetIndicesSRV(),
                            stencilTable->GetWeightsSRV(),
                            /* start = */ 0,
                            /* end   = */ stencilTable->GetNumStencils(),
                            D3D12CommandQueueContext);
    }

    /// Dispatch the DX compute kernel on GPU asynchronously.
    /// returns false if the kernel hasn't been compiled yet.
    bool EvalStencils(CPUDescriptorHandle srcSRV,
                      BufferDescriptor const &srcDesc,
                      CPUDescriptorHandle dstUAV,
                      BufferDescriptor const &dstDesc,
                      CPUDescriptorHandle sizesSRV,
                      CPUDescriptorHandle offsetsSRV,
                      CPUDescriptorHandle indicesSRV,
                      CPUDescriptorHandle weightsSRV,
                      int start,
                      int end,
                      D3D12CommandQueueContext *D3D12CommandQueueContext) const;

    /// Configure DX kernel. Returns false if it fails to compile the kernel.
    bool Compile(BufferDescriptor const &srcDesc,
                 BufferDescriptor const &dstDesc,
                 D3D12CommandQueueContext *D3D12CommandQueueContext);

    /// Wait the dispatched kernel finishes.
    static void Synchronize(D3D12CommandQueueContext *D3D12CommandQueueContext);

private:
    enum
    {
        SingleBufferCSIndex = 0,
        SeparateBufferCSIndex = 1,
        NumberOfCSTypes
    };

    enum
    {
        ViewSlot = 0,
        KernelUniformArgsRootConstantSlot,
        NumSlots
    };

    enum
    {
        SizeSRVDescriptorOffset = 0,
        OffsetSRVDescriptorOffset,
        IndexSRVDescriptorOffset,
        WeightSRVDescriptorOffset,
        SourceUAVDescriptorOffset,
        DestinationUAVDescriptorOffset,
        NumDescriptors,
    };

    DeferredDeletionUniquePtr<ID3D12RootSignature> _rootSignature;
    DeferredDeletionUniquePtr<ID3D12PipelineState> _computePSOs[NumberOfCSTypes];

    mutable CPUDescriptorHandle _descriptorTable[NumDescriptors];
    mutable GPUDescriptorHandle _lastGpuDescriptorTable;
    int _workGroupSize;
};

}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv


#endif  // OPENSUBDIV3_OSD_D3D12_COMPUTE_EVALUATOR_H
