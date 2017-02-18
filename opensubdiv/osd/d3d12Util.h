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

#ifndef OPENSUBDIV3_OSD_D3D12_COMMON_HPP
#define OPENSUBDIV3_OSD_D3D12_COMMON_HPP

#include <d3d11.h>
#include <d3d12.h>
#include "d3dx12.h"
#include <vector>
#include <atlbase.h>

#include "d3d12DeferredDeletionUniquePtr.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {
    static void ThrowFailure(HRESULT hr)
    {
        if (FAILED(hr))
        {
            throw hr;
        }
    }
    
    class ScopedCommandListAllocatorPair : public CommandListAllocatorPair
    {
    public:
        ScopedCommandListAllocatorPair(D3D12CommandQueueContext *context, CommandListAllocatorPair pair) :
            CommandListAllocatorPair(pair),
            _D3D12CommandQueueContext(context)
        {
        }

        ~ScopedCommandListAllocatorPair()
        {
            _D3D12CommandQueueContext->ReleaseCommandListAllocatorPair(*this);
        }
        D3D12CommandQueueContext *_D3D12CommandQueueContext;
    };

    typedef DeferredDeletionUniquePtr<ID3D12Resource> ResourceDeferredDeletionUniquePtr;

    static D3D12_RESOURCE_STATES GetDefaultResourceStateFromHeapType(D3D12_HEAP_TYPE heapType)
    {
        switch (heapType)
        {
        default:
        case D3D12_HEAP_TYPE_UPLOAD:
            return D3D12_RESOURCE_STATE_GENERIC_READ;
        case D3D12_HEAP_TYPE_CUSTOM:
        case D3D12_HEAP_TYPE_DEFAULT:
            return D3D12_RESOURCE_STATE_COMMON;
        case D3D12_HEAP_TYPE_READBACK:
            return D3D12_RESOURCE_STATE_COPY_DEST;
        }
    }

    static void CreateCommittedBuffer(
        unsigned int dataSize,
        D3D12_HEAP_TYPE heapType,
        D3D12_RESOURCE_STATES initialState,
        D3D12CommandQueueContext *D3D12CommandQueueContext,
        ResourceDeferredDeletionUniquePtr &resource,
        D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE,
        D3D12_HEAP_FLAGS heapFlags = D3D12_HEAP_FLAG_NONE)
    {
        ID3D12Device *device = D3D12CommandQueueContext->GetDevice();
        CComPtr<ID3D12Resource> buffer;

        const D3D12_HEAP_PROPERTIES heapProperties = CD3DX12_HEAP_PROPERTIES(heapType);
        const D3D12_RESOURCE_DESC bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(dataSize, flags);

        ThrowFailure(device->CreateCommittedResource(&heapProperties, heapFlags, &bufferDesc, initialState, nullptr, IID_PPV_ARGS(&buffer)));

        resource.AddRefAndAttach(D3D12CommandQueueContext, buffer);
    }

    static void createBuffer(
        unsigned int dataSize,
        D3D12_HEAP_TYPE heapType,
        D3D12CommandQueueContext *D3D12CommandQueueContext,
        ResourceDeferredDeletionUniquePtr &resource)
    {
        CreateCommittedBuffer(dataSize, heapType, GetDefaultResourceStateFromHeapType(heapType), D3D12CommandQueueContext, resource);
    }

    static void createCpuReadableBuffer(
        unsigned int dataSize,
        D3D12CommandQueueContext *D3D12CommandQueueContext,
        ResourceDeferredDeletionUniquePtr &resource)
    {
        createBuffer(dataSize, D3D12_HEAP_TYPE_READBACK, D3D12CommandQueueContext, resource);
    }

    static void createCpuWritableBuffer(
        unsigned int dataSize,
        D3D12CommandQueueContext *D3D12CommandQueueContext,
        ResourceDeferredDeletionUniquePtr &resource)
    {
        createBuffer(dataSize, D3D12_HEAP_TYPE_UPLOAD, D3D12CommandQueueContext, resource);
    }

    static void createDefaultBuffer(
        unsigned int dataSize,
        D3D12CommandQueueContext *D3D12CommandQueueContext,
        ResourceDeferredDeletionUniquePtr &resource)
    {
        createBuffer(dataSize, D3D12_HEAP_TYPE_DEFAULT, D3D12CommandQueueContext, resource);
    }

    static void createBufferWithInitialData(void *pData,
        unsigned int dataSize,
        D3D12CommandQueueContext *D3D12CommandQueueContext,
        ID3D12GraphicsCommandList *pCommandList,
        ResourceDeferredDeletionUniquePtr &resource) {

        ID3D12Device *device = D3D12CommandQueueContext->GetDevice();
        createDefaultBuffer(dataSize, D3D12CommandQueueContext, resource);

        ResourceDeferredDeletionUniquePtr uploadHeap;
        createCpuWritableBuffer(dataSize, D3D12CommandQueueContext, uploadHeap);
        {
            void *pMappedData;
            uploadHeap.Get()->Map(0, nullptr, &pMappedData);
            memcpy(pMappedData, pData, dataSize);

            D3D12_RANGE writtenRange = CD3DX12_RANGE(0, dataSize);
            uploadHeap.Get()->Unmap(0, &writtenRange);
        }
        pCommandList->CopyBufferRegion(resource, 0, uploadHeap.Get(), 0, dataSize);
    }

    template <typename T>
    static void createBufferWithVectorInitialData(std::vector<T> const &src,
        D3D12CommandQueueContext *D3D12CommandQueueContext,
        ID3D12GraphicsCommandList *pCommandList,
        ResourceDeferredDeletionUniquePtr &resource) {
        return createBufferWithInitialData((void*)&src.at(0), (unsigned int)(src.size() * sizeof(T)), D3D12CommandQueueContext, pCommandList, resource);
    }

    static CPUDescriptorHandle AllocateUAV(
        D3D12CommandQueueContext *D3D12CommandQueueContext, 
        ID3D12Resource *resource, 
        DXGI_FORMAT format, 
        SIZE_T numElements)
    {
        CPUDescriptorHandle cpuHandle = D3D12CommandQueueContext->GetDescriptorHeapManager().AllocateDescriptor();

        D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
        uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        uavDesc.Format = format;
        uavDesc.Buffer.NumElements = (unsigned int)numElements;
        D3D12CommandQueueContext->GetDevice()->CreateUnorderedAccessView(resource, nullptr, &uavDesc, D3D12DescriptorHeapManager::ConvertToD3D12CPUHandle(cpuHandle));

        return cpuHandle;
    }

    static CPUDescriptorHandle AllocateSRV(
        D3D12CommandQueueContext *D3D12CommandQueueContext, 
        ID3D12Resource *resource, 
        DXGI_FORMAT format, 
        SIZE_T numElements)
    {
        CPUDescriptorHandle cpuHandle = D3D12CommandQueueContext->GetDescriptorHeapManager().AllocateDescriptor();

        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        srvDesc.Format = format;
        srvDesc.Buffer.NumElements = (unsigned int)numElements;
        D3D12CommandQueueContext->GetDevice()->CreateShaderResourceView(resource, &srvDesc, D3D12DescriptorHeapManager::ConvertToD3D12CPUHandle(cpuHandle));

        return cpuHandle;
    }
}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OPENSUBDIV3_OSD_D3D12_COMMON_HPP
