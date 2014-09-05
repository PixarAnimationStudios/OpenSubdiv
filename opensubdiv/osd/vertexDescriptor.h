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

#ifndef OSD_CPU_VERTEX_DESCRIPTOR_H
#define OSD_CPU_VERTEX_DESCRIPTOR_H

#include "../version.h"
#include <string.h>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

/// \brief Describes vertex elements in interleaved data buffers
struct VertexBufferDescriptor {

    /// Default Constructor
    VertexBufferDescriptor() : offset(0), length(0), stride(0) { }

    /// Constructor
    VertexBufferDescriptor(int o, int l, int s) : offset(o), length(l), stride(s) { }

    /// True if the descriptor values are internally consistent
    bool IsValid() const {
        return ((length>0) and (offset<stride) and (length<=stride-offset));
    }

    /// True if the 'other' descriptor can be used as a destination for
    /// data evaluations.
    bool CanEval( VertexBufferDescriptor const & other ) const {
        return (IsValid() and
                other.IsValid() and
                (length==other.length));
    }

    /// Resets the descriptor to default
    void Reset() {
        offset = length = stride = 0;
    }

    /// True if the descriptors are identical
    bool operator == ( VertexBufferDescriptor const other ) const {
        return (offset == other.offset and
                length == other.length and
                stride == other.stride);
    }

    int offset;  // offset to desired element data
    int length;  // number or length of the data
    int stride;  // stride to the next element
};

} // end namespace Osd

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* OSD_CPU_VERTEX_DESCRIPTRO_H */
