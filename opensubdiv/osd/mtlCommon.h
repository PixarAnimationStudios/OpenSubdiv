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

#ifndef OPENSUBDIV3_OSD_MTL_COMMON_H
#define OPENSUBDIV3_OSD_MTL_COMMON_H

#include "../version.h"

#include <Metal/Metal.h>
#include <cstddef>
#include <vector>

#define OSD_METAL_DEFERRED 0

@protocol MTLDevice;
@protocol MTLCommandQueue;
@protocol MTLCommandBuffer;
@protocol MTLBuffer;

namespace OpenSubdiv
{
namespace OPENSUBDIV_VERSION
{
namespace Osd
{
class MTLContext
{
public:
	id<MTLDevice> device = nullptr;
	id<MTLCommandQueue> commandQueue = nullptr;

    
#if OSD_METAL_DEFERRED
    /*
      Interface designed to let an application manage how OSD will generate it's data when using Metal.
     The intention is to support applications that may have to make many OSD calls per frame and don't want to
     incur the overhead of scheduling each OSD call as a seperate GPU workload and wait for its completion.
     By overriding the functions below an application can merge all OSD GPU workloads into a single command buffer
     and execute (commit) it at a time of it's choosing. It may also choose to integrate its own workloads into the
     command buffer. If it does that it must ensure that no command encoders are active prior to any OSD call.
      OSD will allocate the MTLBuffers for the output data but it's up to the application to release them as only it
     can say when the data can be freed.
     
      If none of the virtual functions are overridden then the provided versions will drive OSD in normal fashion.
     i.e. Each call to OSD will generate a Metal command buffer and wait for its completion.
     */
    
    virtual id<MTLCommandBuffer> GetCommandBuffer(id<MTLCommandQueue> cmdQueue)   { return [cmdQueue commandBuffer]; };
    virtual void                 CommitCommandBuffer(id<MTLCommandBuffer> cmdBuf) { [cmdBuf commit]; }
    virtual void                 WaitUntilCompleted(id<MTLCommandBuffer> cmdBuf)  { [cmdBuf waitUntilCompleted]; }
#if !__has_feature(objc_arc)
    virtual void                 ReleaseBuffer(id<MTLBuffer> buffer)              { [buffer release]; }
#endif


#endif
    id<MTLBuffer> CreateBuffer(const void* data, const size_t length);

    template <typename T>
    id<MTLBuffer> CreateBuffer(const std::vector<T> &vec)
    {
        return CreateBuffer(vec.data(), sizeof(T) * vec.size());
    }
};

} // end namespace Osd

} // end namespace OPENSUBDIV_VERSION

} // end namespace OpenSubdiv


#endif //OPENSUBDIV3_OSD_MTL_COMMON_H
