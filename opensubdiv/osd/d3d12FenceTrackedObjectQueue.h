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

#ifndef OPENSUBDIV3_OSD_D3D12_FENCE_TRACKED_OBJECT_QUEUE_H
#define OPENSUBDIV3_OSD_D3D12_FENCE_TRACKED_OBJECT_QUEUE_H

#include "../version.h"

#include <queue>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

    // Keeps a queue of objects that get paired with a fence value 
    // that mark the last time the GPU referenced that object.
    // Several helper functions exist to verify if there are object 
    // available that are no longer referenced by the GPU
    template <typename Object>
    class FenceTrackedObjectQueue
    {
    protected:
        virtual void DeleteObject(Object *Object) = 0;

    public:
        void Push(unsigned long long fenceValue, Object object)
        {
            if(!IsEmpty()) assert(GetFrontFenceValue() <= fenceValue);
            
            _Queue.push(ObjectFencePair<Object>(fenceValue, object));
        }

        bool IsEmpty() const
        {
            return _Queue.empty();
        }

        bool IsObjectAvailable(unsigned long long completedFenceValue) const
        {
            return !IsEmpty() && completedFenceValue >= _Queue.front()._fenceValue;
        }

        Object Pop()
        {
            assert(!IsEmpty());

            Object objectToReturn = _Queue.front()._object;
            _Queue.pop();
            return objectToReturn;
        }

        void DeleteUnusedObjects(unsigned long long completedFenceValue)
        {
            while (IsObjectAvailable(completedFenceValue))
            {
                DeleteObject(&_Queue.front()._object);
                _Queue.pop();
            }
        }
    private:
        unsigned long long GetFrontFenceValue()
        {
            assert(!IsEmpty());
            return _Queue.front()._fenceValue;
        }

        template<typename Object>
        class ObjectFencePair
        {
        public:
            ObjectFencePair() : _fenceValue(0) {}

            ObjectFencePair(UINT64 fenceValue, typename Object object) :
                _fenceValue(fenceValue), _object(object) {}

            UINT64 _fenceValue;
            Object _object;
        };

        std::queue<ObjectFencePair<Object>> _Queue;
    };

}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OPENSUBDIV3_OSD_D3D12_FENCE_TRACKED_OBJECT_QUEUE_H
