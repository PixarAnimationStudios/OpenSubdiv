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

#ifndef OPENSUBDIV3_OSD_D3D12_DESCRIPTOR_HEAP_MANAGER_H
#define OPENSUBDIV3_OSD_D3D12_DESCRIPTOR_HEAP_MANAGER_H

#include "../version.h"

#include <queue>
#include <atlbase.h>

struct ID3D12Device;
struct D3D12_CPU_DESCRIPTOR_HANDLE;
struct D3D12_GPU_DESCRIPTOR_HANDLE;
struct ID3D12DescriptorHeap;

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

    typedef SIZE_T CPUDescriptorHandle;
    typedef unsigned long long GPUDescriptorHandle;

    class D3D12DescriptorHeapManager
    {
    public:
        D3D12DescriptorHeapManager(ID3D12Device *device, unsigned int nodeMask);

        CPUDescriptorHandle AllocateDescriptor();
        void ReleaseDescriptor(CPUDescriptorHandle &handle);
        GPUDescriptorHandle UploadDescriptors(unsigned int NumViews, CPUDescriptorHandle *pViews);
        ID3D12DescriptorHeap *GetDescriptorHeap() const { return _onlineDescriptorHeap; }

        static D3D12_CPU_DESCRIPTOR_HANDLE ConvertToD3D12CPUHandle(const CPUDescriptorHandle &handle);
        static D3D12_GPU_DESCRIPTOR_HANDLE ConvertToD3D12GPUHandle(const GPUDescriptorHandle &handle);
    private:
        ID3D12Device *_device;

        static const unsigned int cMaxSimulataneousDescriptors = 500;
        CComPtr<ID3D12DescriptorHeap> _offlineDescriptorHeap;
        std::queue<unsigned int> _freeSlotsQueue;
        unsigned int _descriptorSize;

        CComPtr<ID3D12DescriptorHeap> _onlineDescriptorHeap;
        unsigned int _onlineDescriptorsUsed;
    };

}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OPENSUBDIV3_OSD_D3D12_DESCRIPTOR_HEAP_MANAGER_H
