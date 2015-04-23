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

#ifndef FAR_KERNEL_BATCH_H
#define FAR_KERNEL_BATCH_H

#include "../version.h"

#include <vector>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Far {

/// \brief A GP Compute Kernel descriptor.
///
/// Vertex refinement through subdivision schemes requires the successive
/// application of dedicated compute kernels. OpenSubdiv groups these vertices
/// in batches based on their topology in order to minimize the number of kernel
/// switches to process a given primitive.
///
struct KernelBatch {

public:

    enum KernelType {
        KERNEL_UNKNOWN=0,
        KERNEL_STENCIL_TABLE,
        KERNEL_USER_DEFINED
    };

    /// \brief Constructor.
    /// 
    /// @param _kernelType    The type of compute kernel kernel
    ///
    /// @param _level         The level of subdivision of the vertices in the batch
    ///
    /// @param _start         Index of the first vertex in the batch
    ///
    /// @param _end           Index of the last vertex in the batch
    ///
    KernelBatch( int _kernelType, int _level, int _start, int _end ) :
        kernelType(_kernelType), level(_level), start(_start), end(_end) { }

    int kernelType,   // the type of compute kernel kernel
        level,        // the level of subdivision of the vertices in the batch
        start,        // index of the first vertex in the batch
        end;          // index of the last vertex in the batch
};

typedef std::vector<KernelBatch> KernelBatchVector;

} // end namespace Far

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif  /* FAR_KERNEL_BATCH_H */
