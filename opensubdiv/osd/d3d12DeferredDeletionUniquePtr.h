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

#ifndef OPENSUBDIV3_OSD_D3D12_DEFERRED_DELETION_UNIQUE_PTR_H
#define OPENSUBDIV3_OSD_D3D12_DEFERRED_DELETION_UNIQUE_PTR_H

#include "../version.h"

#include "../osd/nonCopyable.h"
#include "../osd/d3d12commandqueuecontext.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

    // Unlike D3D11, D3D12 requires applications to explicitly track when the GPU is finished
    // with an ID3D12Object before destroying it. DeferredDeletionUniquePtr ensures that when the 
    // destructor is called, the D3D12Object will be passed along to a Deferred Deletion Queue 
    // that tracks when the GPU is done with the object
    template<typename Object>
    class DeferredDeletionUniquePtr : NonCopyable<DeferredDeletionUniquePtr<Object>>
    {
    public:
        DeferredDeletionUniquePtr() :
            _D3D12CommandQueueContext(nullptr),
            _object(nullptr)
        {}

        DeferredDeletionUniquePtr(D3D12CommandQueueContext *D3D12CommandQueueContext, Object *object)
        {
            AddRefAndAttach(D3D12CommandQueueContext, object);
        }

        void AddRefAndAttach(D3D12CommandQueueContext *D3D12CommandQueueContext, Object *object)
        {
            ReleaseObjectToDeletionQueueIfExists();

            _D3D12CommandQueueContext = D3D12CommandQueueContext;
            _object = object;
            _object->AddRef();
        }

        ~DeferredDeletionUniquePtr() {
            ReleaseObjectToDeletionQueueIfExists();
        }
        Object *Get() const { return _object; }
        operator Object*() { return _object; }
        Object * const *operator&() const { return &_object; }
        Object *operator->() { return _object; }
    private:
        void ReleaseObjectToDeletionQueueIfExists()
        {
            if (_object)
            {
                _D3D12CommandQueueContext->DeleteD3D12Object(_object);

                _object->Release();
                _object = nullptr;
            }
        }

        D3D12CommandQueueContext *_D3D12CommandQueueContext;
        Object *_object;
    };

}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OPENSUBDIV3_OSD_D3D12_DEFERRED_DELETION_UNIQUE_PTR_H
