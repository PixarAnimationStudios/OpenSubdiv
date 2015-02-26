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
#include <cstdio>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Far {



//
// Lists of valid patch Descriptors for each subdivision scheme
//

ConstPatchDescriptorArray
PatchDescriptor::GetAdaptivePatchDescriptors(Sdc::SchemeType type) {

    static PatchDescriptor _loopDescriptors[] = {
        // XXXX work in progress !
        PatchDescriptor(LOOP, NON_TRANSITION, 0)
    };

    static PatchDescriptor _catmarkDescriptors[] = {

        // non-transition patches : 7
        PatchDescriptor(REGULAR,          NON_TRANSITION, 0),
        PatchDescriptor(SINGLE_CREASE,    NON_TRANSITION, 0),
        PatchDescriptor(BOUNDARY,         NON_TRANSITION, 0),
        PatchDescriptor(CORNER,           NON_TRANSITION, 0),
        PatchDescriptor(GREGORY,          NON_TRANSITION, 0),
        PatchDescriptor(GREGORY_BOUNDARY, NON_TRANSITION, 0),
        PatchDescriptor(GREGORY_BASIS,    NON_TRANSITION, 0),

        // transition pattern 0
        PatchDescriptor(REGULAR,       PATTERN0, 0),
        PatchDescriptor(SINGLE_CREASE, PATTERN0, 0),
        PatchDescriptor(SINGLE_CREASE, PATTERN0, 1),
        PatchDescriptor(SINGLE_CREASE, PATTERN0, 2),
        PatchDescriptor(SINGLE_CREASE, PATTERN0, 3),
        PatchDescriptor(BOUNDARY,      PATTERN0, 0),
        PatchDescriptor(BOUNDARY,      PATTERN0, 1),
        PatchDescriptor(BOUNDARY,      PATTERN0, 2),
        PatchDescriptor(BOUNDARY,      PATTERN0, 3),
        PatchDescriptor(CORNER,        PATTERN0, 0),
        PatchDescriptor(CORNER,        PATTERN0, 1),
        PatchDescriptor(CORNER,        PATTERN0, 2),
        PatchDescriptor(CORNER,        PATTERN0, 3),

        // transition pattern 1
        PatchDescriptor(REGULAR,       PATTERN1, 0),
        PatchDescriptor(SINGLE_CREASE, PATTERN1, 0),
        PatchDescriptor(SINGLE_CREASE, PATTERN1, 1),
        PatchDescriptor(SINGLE_CREASE, PATTERN1, 2),
        PatchDescriptor(SINGLE_CREASE, PATTERN1, 3),
        PatchDescriptor(BOUNDARY,      PATTERN1, 0),
        PatchDescriptor(BOUNDARY,      PATTERN1, 1),
        PatchDescriptor(BOUNDARY,      PATTERN1, 2),
        PatchDescriptor(BOUNDARY,      PATTERN1, 3),
        PatchDescriptor(CORNER,        PATTERN1, 0),
        PatchDescriptor(CORNER,        PATTERN1, 1),
        PatchDescriptor(CORNER,        PATTERN1, 2),
        PatchDescriptor(CORNER,        PATTERN1, 3),

        // transition pattern 2
        PatchDescriptor(REGULAR,       PATTERN2, 0),
        PatchDescriptor(SINGLE_CREASE, PATTERN2, 0),
        PatchDescriptor(SINGLE_CREASE, PATTERN2, 1),
        PatchDescriptor(SINGLE_CREASE, PATTERN2, 2),
        PatchDescriptor(SINGLE_CREASE, PATTERN2, 3),
        PatchDescriptor(BOUNDARY,      PATTERN2, 0),
        PatchDescriptor(BOUNDARY,      PATTERN2, 1),
        PatchDescriptor(BOUNDARY,      PATTERN2, 2),
        PatchDescriptor(BOUNDARY,      PATTERN2, 3),
        PatchDescriptor(CORNER,        PATTERN2, 0),
        PatchDescriptor(CORNER,        PATTERN2, 1),
        PatchDescriptor(CORNER,        PATTERN2, 2),
        PatchDescriptor(CORNER,        PATTERN2, 3),

        // transition pattern 3
        PatchDescriptor(REGULAR,       PATTERN3, 0),
        PatchDescriptor(SINGLE_CREASE, PATTERN3, 0),
        PatchDescriptor(SINGLE_CREASE, PATTERN3, 1),
        PatchDescriptor(SINGLE_CREASE, PATTERN3, 2),
        PatchDescriptor(SINGLE_CREASE, PATTERN3, 3),
        PatchDescriptor(BOUNDARY,      PATTERN3, 0),
        PatchDescriptor(BOUNDARY,      PATTERN3, 1),
        PatchDescriptor(BOUNDARY,      PATTERN3, 2),
        PatchDescriptor(BOUNDARY,      PATTERN3, 3),
        PatchDescriptor(CORNER,        PATTERN3, 0),
        PatchDescriptor(CORNER,        PATTERN3, 1),
        PatchDescriptor(CORNER,        PATTERN3, 2),
        PatchDescriptor(CORNER,        PATTERN3, 3),

        // transition pattern 4
        PatchDescriptor(REGULAR,       PATTERN4, 0),
        PatchDescriptor(SINGLE_CREASE, PATTERN4, 0),
        PatchDescriptor(SINGLE_CREASE, PATTERN4, 1),
        PatchDescriptor(SINGLE_CREASE, PATTERN4, 2),
        PatchDescriptor(SINGLE_CREASE, PATTERN4, 3),
        PatchDescriptor(BOUNDARY,      PATTERN4, 0),
        PatchDescriptor(BOUNDARY,      PATTERN4, 1),
        PatchDescriptor(BOUNDARY,      PATTERN4, 2),
        PatchDescriptor(BOUNDARY,      PATTERN4, 3),
        PatchDescriptor(CORNER,        PATTERN4, 0),
        PatchDescriptor(CORNER,        PATTERN4, 1),
        PatchDescriptor(CORNER,        PATTERN4, 2),
        PatchDescriptor(CORNER,        PATTERN4, 3),
    };

    switch (type) {
        case Sdc::SCHEME_BILINEAR :
            return ConstPatchDescriptorArray(0, 0);
        case Sdc::SCHEME_CATMARK :
            return ConstPatchDescriptorArray(_catmarkDescriptors,
                (int)(sizeof(_catmarkDescriptors)/sizeof(PatchDescriptor)));
        case Sdc::SCHEME_LOOP :
            return ConstPatchDescriptorArray(_loopDescriptors,
                (int)(sizeof(_loopDescriptors)/sizeof(PatchDescriptor)));
        default:
          assert(0);
    }
    return ConstPatchDescriptorArray(0, 0);;
}

void
PatchDescriptor::print() const {
    static char const * types[13] = {
        "NON_PATCH", "POINTS", "LINES", "QUADS", "TRIANGLES", "LOOP", "REGULAR",
            "SINGLE_CREASE", "BOUNDARY", "CORNER", "GREGORY",
                "GREGORY_BOUNDARY", "GREGORY_BASIS" };

    printf("    type %s trans %d rot %d\n",
        types[_type], _pattern, _rotation);
}



} // end namespace Far

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv
