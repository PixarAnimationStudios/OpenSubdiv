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

#include "../osd/d3d12ComputeEvaluator.h"

#include <cassert>
#include <sstream>
#include <string>

#include <d3d12.h>
#include "d3dx12.h"

#include <d3dcompiler.h>


#include "../far/error.h"
#include "../far/stencilTable.h"

#include "d3d12util.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

CommandListAllocatorPairAllocator::CommandListAllocatorPairAllocator(ID3D12Device *device, unsigned int nodeMask) : _device(device), _nodeMask(nodeMask) {}

CommandListAllocatorPair CommandListAllocatorPairAllocator::Allocate() {
    CommandListAllocatorPair pair;
    ThrowFailure(_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&pair._allocator)));
    ThrowFailure(_device->CreateCommandList(_nodeMask, D3D12_COMMAND_LIST_TYPE_DIRECT, pair._allocator, nullptr, IID_PPV_ARGS(&pair._commandList)));

    return pair;
}

void CommandListAllocatorPairAllocator::Free(CommandListAllocatorPair &allocation) {
    allocation._allocator->Release();
    allocation._commandList->Release();
}

void CommandListAllocatorPairAllocator::Reset(CommandListAllocatorPair &allocation) {
    allocation._allocator->Reset();
    allocation._commandList->Reset(allocation._allocator, nullptr);
}

void D3D12DeferredDeletionQueue::Push(unsigned long long fenceValue, ID3D12Object *object)
{
    object->AddRef();
    FenceTrackedObjectQueue<ID3D12Object *>::Push(fenceValue, object);
}

void D3D12DeferredDeletionQueue::DeleteObject(ID3D12Object **object)
{
    (*object)->Release();
}

D3D12CommandQueueContext::D3D12CommandQueueContext(ID3D12CommandQueue *commandQueue, ID3D12Device *device, unsigned int nodeMask, ID3D11DeviceContext *deviceContext, ID3D11On12Device *D3D11on12Device) :
    _queue(commandQueue),
    _device(device),
    _D3D11Context(deviceContext),
    _descriptorHeapManager(device, nodeMask),
    _commandListAllocator(CommandListAllocatorPairAllocator(device, nodeMask)),
    _D3D11on12Device(D3D11on12Device),
    _nodeMask(nodeMask),
    _fenceValue(0)
{
    ThrowFailure(_device->CreateFence(_fenceValue, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&_fence)));
    _fenceValue++;

    _waitEvent = CreateEvent(nullptr, false, false, nullptr);
}

D3D12CommandQueueContext::~D3D12CommandQueueContext()
{
    Syncronize();
    _commandListAllocator.DeleteUnusedObjects(_fence->GetCompletedValue());
    _deferredDeletionQueue.DeleteUnusedObjects(_fence->GetCompletedValue());

    CloseHandle(_waitEvent);
}

CommandListAllocatorPair D3D12CommandQueueContext::GetCommandListAllocatorPair()
{
    return _commandListAllocator.Allocate(_fence->GetCompletedValue());
}

void D3D12CommandQueueContext::ExecuteCommandList(ID3D12CommandList *CommandList)
{
    ID3D12CommandList *CommandLists[] = { CommandList };
    _queue->ExecuteCommandLists(ARRAYSIZE(CommandLists), CommandLists);

    SignalAndIncrementFence();

    NotifyOnCommandListSubmission();
}

void D3D12CommandQueueContext::Syncronize()
{
    unsigned long long currentFence = _fenceValue;
    SignalAndIncrementFence();

    ThrowFailure(_fence->SetEventOnCompletion(currentFence, _waitEvent));
    WaitForSingleObject(_waitEvent, INFINITE);
}

void D3D12CommandQueueContext::SignalAndIncrementFence()
{
    _queue->Signal(_fence, _fenceValue);
    _fenceValue++;
}


void D3D12CommandQueueContext::ReleaseCommandListAllocatorPair(CommandListAllocatorPair &pair)
{
    _commandListAllocator.Release(_fenceValue, &pair);
}

void D3D12CommandQueueContext::DeleteD3D12Object(ID3D12Object *Object)
{
    _deferredDeletionQueue.Push(_fenceValue, Object);
}

void D3D12CommandQueueContext::NotifyOnCommandListSubmission()
{
    // Periodically check on the deletion queue to free up objects
    _deferredDeletionQueue.DeleteUnusedObjects(_fence->GetCompletedValue());
}


D3D12CommandQueueContext *CreateD3D12CommandQueueContext(ID3D12CommandQueue *commandQueue, unsigned int nodeMask, ID3D11DeviceContext *deviceContext, ID3D11On12Device *pD3D11on12Device)
{
    CComPtr<ID3D12Device> device;
    commandQueue->GetDevice(IID_PPV_ARGS(&device));
    return new D3D12CommandQueueContext(commandQueue, device, nodeMask, deviceContext, pD3D11on12Device);
}

void FreeD3D12CommandQueueContext(D3D12CommandQueueContext *D3D12CommandQueueContext)
{
    delete D3D12CommandQueueContext;
}

}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
