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

#include "../osd/mtlCommon.h"

namespace OpenSubdiv
{
namespace OPENSUBDIV_VERSION
{
namespace Osd
{

id<MTLBuffer>
MTLContext::CreateBuffer(const void* data, const size_t length)
{
    if (data == nullptr || length == 0) {
        return nil;
    }
#if TARGET_OS_IOS || TARGET_OS_TV
    return [context->device newBufferWithBytes:vec.data()
                                        length:length
                                       options:MTLResourceOptionCPUCacheModeDefault];
#elif TARGET_OS_OSX
#if !OSD_METAL_DEFERRED
    @autoreleasepool
    {
        auto cmdBuf = [commandQueue commandBuffer];
#else
    {
        auto cmdBuf = GetCommandBuffer(commandQueue);
#endif
        auto blitEncoder = [cmdBuf blitCommandEncoder];

        auto stageBuffer = [device newBufferWithBytes:data
                                               length:length
                                              options:MTLResourceOptionCPUCacheModeDefault];

        auto finalBuffer = [device newBufferWithLength:length
                                               options:MTLResourceStorageModePrivate];

        [blitEncoder copyFromBuffer:stageBuffer
                       sourceOffset:0
                           toBuffer:finalBuffer
                  destinationOffset:0
                               size:length];
        [blitEncoder endEncoding];
#if OSD_METAL_DEFERRED
        CommitCommandBuffer(cmdBuf);
        WaitUntilCompleted(cmdBuf);
#else
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
#endif
        
#if !__has_feature(objc_arc)
        [stageBuffer release];
#endif

        return finalBuffer;
    }
#endif
}


} // end namespace Osd

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

