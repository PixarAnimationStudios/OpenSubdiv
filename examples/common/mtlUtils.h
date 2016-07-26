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

#ifndef OPENSUBDIV_EXAMPLES_MTL_UTILS_H
#define OPENSUBDIV_EXAMPLES_MTL_UTILS_H

#include <Metal/Metal.h>
#include <cstring>
#include <TargetConditionals.h>
#include <vector>

namespace OpenSubdiv { namespace OPENSUBDIV_VERSION { namespace Osd {
#if TARGET_OS_EMBEDDED
constexpr auto MTLDefaultStorageMode = MTLResourceStorageModeShared;
#else
constexpr auto MTLDefaultStorageMode = MTLResourceStorageModeManaged;
#endif

template<typename T>
id<MTLBuffer> MTLNewBufferFromVector(id<MTLDevice> device, const std::vector<T>& vec) {
    const auto typeSize = sizeof(T);
    const auto bufferSize = typeSize * vec.size();
    return [device newBufferWithBytes:vec.data() length:bufferSize options:MTLDefaultStorageMode];
}

template<typename DataType, size_t RingLength>
class MTLRingBuffer {
    private:
    id<MTLBuffer> _buffers[RingLength];
    uint _bufferIndex;
    
    public:
    template<typename SizedType = DataType>
    MTLRingBuffer(id<MTLDevice> device, size_t count, NSString* label, MTLResourceOptions options = MTLDefaultStorageMode) {
        alloc<SizedType>(device, count, options, label);
    }
    
    MTLRingBuffer() {
        for(uint i = 0; i < RingLength; i++) {
            _buffers[i] = nil;
        }
        _bufferIndex = 0;
    }
    
    void markModified() const {
#if !TARGET_OS_EMBEDDED
        const auto bufferLength = [buffer() length];
        [buffer() didModifyRange: NSMakeRange(0, bufferLength)];
#endif
    }
    
    template<typename SizedType = DataType>
    void alloc(id<MTLDevice> device, size_t count, NSString* label, MTLResourceOptions options = MTLDefaultStorageMode) {
        if(count) {
            for(uint i = 0; i < RingLength; i++) {
                _buffers[i] = [device newBufferWithLength: count * sizeof(SizedType) options:options];
                _buffers[i].label = label;
                _bufferIndex = i;
                
                if((options & MTLResourceStorageModePrivate) == 0)
                {
                    memset(data(), 0, count * sizeof(SizedType));
                    markModified();
                }
            }
        } else {
            for(uint i = 0; i < RingLength; i++) {
                _buffers[i] = nil;
            }
        }
        _bufferIndex = 0;
    }
    
    id<MTLBuffer> buffer() const {
        return _buffers[_bufferIndex];
    }
    
    DataType* data() const {
        return reinterpret_cast<DataType*>([buffer() contents]);
    }
    
    void next() {
        _bufferIndex = (_bufferIndex + 1) % RingLength;
    }
    
    operator id<MTLBuffer>() const {
        return buffer();
    }
    
    DataType* operator->() const {
        return data();
    }
    
    DataType& operator[](int idx) const {
        return (data())[idx];
    }
};

#endif  // OPENSUBDIV_EXAMPLES_MTL_UTILS_H
} } using namespace OPENSUBDIV_VERSION; }
