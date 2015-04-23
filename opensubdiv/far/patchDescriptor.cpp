//
//   Copyright 2014 DreamWorks Animation LLC.
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

#include "../far/patchDescriptor.h"

#include <cassert>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Far {


//
// Lists of patch Descriptors for each subdivision scheme
//

static PatchDescriptorVector const &
getAdaptiveCatmarkDescriptors() {

    static PatchDescriptorVector _descriptors;

    if (_descriptors.empty()) {

        _descriptors.reserve(72);

        // non-transition patches : 7
        for (int i=PatchDescriptor::REGULAR;
            i<=PatchDescriptor::GREGORY_BASIS; ++i) {

            _descriptors.push_back(
                PatchDescriptor(i, PatchDescriptor::NON_TRANSITION, 0));
        }

        // transition patches (1 + 4 * 3) * 5 = 65
        for (int i=PatchDescriptor::PATTERN0; i<=PatchDescriptor::PATTERN4; ++i) {

            _descriptors.push_back(
                PatchDescriptor(PatchDescriptor::REGULAR, i, 0) );

            // 4 rotations for single-crease, boundary and corner patches
            for (int j=0; j<4; ++j) {
                _descriptors.push_back(
                    PatchDescriptor(PatchDescriptor::SINGLE_CREASE, i, j));
            }

            for (int j=0; j<4; ++j) {
                _descriptors.push_back(
                    PatchDescriptor(PatchDescriptor::BOUNDARY, i, j));
            }

            for (int j=0; j<4; ++j) {
                _descriptors.push_back(
                   PatchDescriptor(PatchDescriptor::CORNER, i, j));
            }
        }
    }
    return _descriptors;
}
static PatchDescriptorVector const &
getAdaptiveLoopDescriptors() {

    static PatchDescriptorVector _descriptors;

    if (_descriptors.empty()) {
        _descriptors.reserve(1);
        _descriptors.push_back(
            PatchDescriptor(PatchDescriptor::LOOP, PatchDescriptor::NON_TRANSITION, 0) );
    }
    return _descriptors;
}
PatchDescriptorVector const &
PatchDescriptor::GetAdaptivePatchDescriptors(Sdc::SchemeType type) {

    static PatchDescriptorVector _empty;

    switch (type) {
        case Sdc::SCHEME_BILINEAR : return _empty;
        case Sdc::SCHEME_CATMARK  : return getAdaptiveCatmarkDescriptors();
        case Sdc::SCHEME_LOOP     : return getAdaptiveLoopDescriptors();
        default:
          assert(0);
    }
    return _empty;
}


} // end namespace Far

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv
