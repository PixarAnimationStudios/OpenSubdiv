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

#include "../osd/ompEvaluator.h"
#include "../osd/ompKernel.h"
#include <omp.h>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

/* static */
bool
OmpEvaluator::EvalStencils(const float *src,
                           VertexBufferDescriptor const &srcDesc,
                           float *dst,
                           VertexBufferDescriptor const &dstDesc,
                           const unsigned char * sizes,
                           const int * offsets,
                           const int * indices,
                           const float * weights,
                           int start, int end) {
    if (end <= start) return true;

    // we can probably expand cpuKernel.cpp to here.
    OmpEvalStencils(src, srcDesc, dst, dstDesc,
                    sizes, offsets, indices, weights, start, end);

    return true;
}

/* static */
void
OmpEvaluator::Synchronize(void * /*deviceContext*/) {
    // we use "omp parallel for" and it synchronizes by itself
}

/* static */
void
OmpEvaluator::SetNumThreads(int numThreads) {
    omp_set_num_threads(numThreads);
}

}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
