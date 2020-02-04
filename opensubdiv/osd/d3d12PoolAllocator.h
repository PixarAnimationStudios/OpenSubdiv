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

#ifndef OPENSUBDIV3_OSD_D3D12_POOL_ALLOCATOR_H
#define OPENSUBDIV3_OSD_D3D12_POOL_ALLOCATOR_H

#include "../version.h"

#include <queue>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {
    template<typename AllocationType>
    class IAllocator
    {
    public:
        typedef AllocationType Allocation;
        AllocationType Allocate() { assert(false); return AllocationType(); };
        void Free(AllocationType &allocation) { assert(false); };
        void Reset(AllocationType &allocation) { assert(false); }
    };

    template<typename AllocationType, typename Allocator>
    class D3D12PoolAllocator : public FenceTrackedObjectQueue<AllocationType>
    {
    public:
        D3D12PoolAllocator(Allocator allocator) :
            _allocator(allocator)
        {
        }

        ~D3D12PoolAllocator()
        {
            // It's the callers responsibility to make sure all objects 
            // are trimmed before deleting the allocator
            assert(IsEmpty());
        }

        void DeleteObject(AllocationType *allocation)
        {
            _allocator.Free(*allocation);
        }

        AllocationType Allocate(unsigned long long completeFenceValue)
        {
            if (IsObjectAvailable(completeFenceValue))
            {
                AllocationType allocation = Pop();

                _allocator.Reset(allocation);
                return allocation;
            }
            else
            {
                return _allocator.Allocate();
            }
        }

        void Release(unsigned long long fenceValue, AllocationType *pAllocations, unsigned int numAllocations = 1)
        {
            assert(numAllocations >= 1);

            for (unsigned int i = 0; i < numAllocations; i++)
            {
                Push(fenceValue, pAllocations[i]);
            }
        }

    private:
        Allocator _allocator;
    };
}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OPENSUBDIV3_OSD_D3D12_POOL_ALLOCATOR_H
