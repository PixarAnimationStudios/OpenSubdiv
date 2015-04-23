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

#include "../osd/ompKernel.h"
#include "../osd/vertexDescriptor.h"

#include <cassert>
#include <cstdlib>
#include <omp.h>
#include <vector>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

template <class T> T *
elementAtIndex(T * src, int index, VertexBufferDescriptor const &desc) {

    return src + index * desc.stride;
}

static inline void
clear(float *dst, VertexBufferDescriptor const &desc) {

    assert(dst);
    memset(dst, 0, desc.length*sizeof(float));
}

static inline void
addWithWeight(float *dst, const float *src, int srcIndex, float weight,
              VertexBufferDescriptor const &desc) {

    assert(src and dst);
    src = elementAtIndex(src, srcIndex, desc);
    for (int k = 0; k < desc.length; ++k) {
        dst[k] += src[k] * weight;
    }
}

static inline void
copy(float *dst, int dstIndex, const float *src,
     VertexBufferDescriptor const &desc) {

    assert(src and dst);

    dst = elementAtIndex(dst, dstIndex, desc);
    memcpy(dst, src, desc.length*sizeof(float));
}


// XXXX manuelk this should be optimized further by using SIMD - considering
//              OMP is somewhat obsolete - this is probably not worth it.
void
OmpComputeStencils(VertexBufferDescriptor const &vertexDesc,
                      float const * vertexSrc,
                      float * vertexDst,
                      unsigned char const * sizes,
                      int const * offsets,
                      int const * indices,
                      float const * weights,
                      int start, int end) {

    assert(start>=0 and start<end);

    int numThreads = omp_get_max_threads(),
        nstencils = end-start;

    float * result = (float*)alloca(vertexDesc.length*numThreads*sizeof(float));

#pragma omp parallel for
    for (int i=0; i<nstencils; ++i) {

        int index = i + (start>0 ? start : 0); // Stencil index

        // Get thread-local pointers
        int const           * threadIndices = indices + offsets[index];
        float const         * threadWeights = weights + offsets[index];

        int threadId = omp_get_thread_num();

        float * threadResult = result + threadId*vertexDesc.length;

        clear(threadResult, vertexDesc);

        for (int j=0; j<(int)sizes[index]; ++j) {
            addWithWeight(threadResult, vertexSrc,
                threadIndices[j], threadWeights[j], vertexDesc);
        }

        copy(vertexDst, i, threadResult, vertexDesc);
    }

}

} // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
