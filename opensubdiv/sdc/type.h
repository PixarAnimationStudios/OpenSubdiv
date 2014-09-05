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
#ifndef SDC_TYPE_H
#define SDC_TYPE_H

#include "../version.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Sdc {

//
//  Enumerated type for all subdivisions schemes supported by OpenSubdiv:
//
//  Questions:
//      In general, scoping of enumeration names is an issue given the lack of nested
//  namespaces.  Originally I gave all other types a qualifying prefix to avoid conflicts
//  but these names didn't seem to warrant it, but I added one later.
//
//  Note there is a similar Scheme enum in FarSubdivisionTables that includes UNKNOWN=0
//  along with the same three constants.
//
enum Type {
    TYPE_BILINEAR,
    TYPE_CATMARK,
    TYPE_LOOP
};


//
//  Traits associated with all types -- these are specialized and instantiated for
//  each of the supported types.
//
//  Traits do not vary with the topology or any options applied to the scheme.  They
//  are intended to help construct more general queries about a subdivision scheme
//  in a context where its details may be less well understood.  They serve little
//  purpose in code specialized to the particular scheme, i.e. in code already
//  specialized for Catmark, the values for these traits for the Catmark scheme are
//  typically known and their usage well understood.
//
//  Question:
//      Do we really need/want these TypeTraits, or will static methods on another
//  class specialized for the type suffice, i.e. Scheme<SCHEME_TYPE>?
//      If yes, there will be little in here other than Sdc::Type, which we may want
//  to merge into <sdc/options.h>.
//
enum Split {
    SPLIT_TO_QUADS,  // used by Catmark and Bilinear
    SPLIT_TO_TRIS,   // used by Loop
    SPLIT_HYBRID     // not currently used (potential future extension)
};

template <Type SCHEME_TYPE>
struct TypeTraits {

    static Type GetType() {
        return SCHEME_TYPE;
    }

    static Split       TopologicalSplitType();
    static int         LocalNeighborhoodSize();
    static int         RegularVertexValence();
    static int         RegularFaceValence();
    static char const* Label();
};


} // end namespace sdc

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;
} // end namespace OpenSubdiv

#endif /* SDC_TYPE_H */
