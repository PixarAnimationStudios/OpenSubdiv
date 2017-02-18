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

#include "d3d12descriptorHeapManager.h"
#include "d3d12Util.h"
#include <d3d12.h>
#include <d3dx12.h>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {
    D3D12DescriptorHeapManager::D3D12DescriptorHeapManager(ID3D12Device *device, unsigned int nodeMask) :
        _device(device), _onlineDescriptorsUsed(0), _descriptorSize(device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV))
    {
        D3D12_DESCRIPTOR_HEAP_DESC heapDesc;
        heapDesc.NodeMask = nodeMask;
        heapDesc.NumDescriptors = cMaxSimulataneousDescriptors;
        heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
        ThrowFailure(device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&_offlineDescriptorHeap)));

        heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        ThrowFailure(device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&_onlineDescriptorHeap)));

        for (unsigned int slot = 0; slot < cMaxSimulataneousDescriptors; slot++)
        {
            _freeSlotsQueue.push(slot);
        }
    }

    CPUDescriptorHandle D3D12DescriptorHeapManager::AllocateDescriptor()
    {
        D3D12_CPU_DESCRIPTOR_HANDLE CPUDescriptorHandle = CD3DX12_CPU_DESCRIPTOR_HANDLE(_offlineDescriptorHeap->GetCPUDescriptorHandleForHeapStart(), _freeSlotsQueue.front(), _descriptorSize);
        _freeSlotsQueue.pop();

        return CPUDescriptorHandle.ptr;
    }

    void D3D12DescriptorHeapManager::ReleaseDescriptor(CPUDescriptorHandle &handle)
    {
        D3D12_CPU_DESCRIPTOR_HANDLE base = _offlineDescriptorHeap->GetCPUDescriptorHandleForHeapStart();
        _freeSlotsQueue.push((unsigned int)((ConvertToD3D12CPUHandle(handle).ptr - base.ptr) / _descriptorSize));
    }

    GPUDescriptorHandle D3D12DescriptorHeapManager::UploadDescriptors(UINT NumViews, CPUDescriptorHandle *pViews)
    {
        D3D12_GPU_DESCRIPTOR_HANDLE destGpuHandle = CD3DX12_GPU_DESCRIPTOR_HANDLE(_onlineDescriptorHeap->GetGPUDescriptorHandleForHeapStart(), _onlineDescriptorsUsed, _descriptorSize);

        if (_onlineDescriptorsUsed + NumViews > cMaxSimulataneousDescriptors)
        {
            // Flush here
            _onlineDescriptorsUsed = 0;
        }

        for (UINT viewIndex = 0; viewIndex < NumViews; viewIndex++)
        {
            D3D12_CPU_DESCRIPTOR_HANDLE destCpuHandle = CD3DX12_CPU_DESCRIPTOR_HANDLE(_onlineDescriptorHeap->GetCPUDescriptorHandleForHeapStart(), _onlineDescriptorsUsed, _descriptorSize);
            _device->CopyDescriptorsSimple(1, destCpuHandle, ConvertToD3D12CPUHandle(pViews[viewIndex]), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            _onlineDescriptorsUsed++;
        }

        return destGpuHandle.ptr;
    }

    D3D12_CPU_DESCRIPTOR_HANDLE D3D12DescriptorHeapManager::ConvertToD3D12CPUHandle(const CPUDescriptorHandle &handle)
    {
        D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle;
        cpuHandle.ptr = handle;
        return cpuHandle;
    }

    D3D12_GPU_DESCRIPTOR_HANDLE D3D12DescriptorHeapManager::ConvertToD3D12GPUHandle(const GPUDescriptorHandle &handle)
    {
        D3D12_GPU_DESCRIPTOR_HANDLE gpuHandle;
        gpuHandle.ptr = handle;
        return gpuHandle;
    }
}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv
