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
#ifndef SDC_TYPES_H
#define SDC_TYPES_H

#include "../version.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Sdc {

///
///  \brief Enumerated type for all subdivisions schemes supported by OpenSubdiv
///
enum Type {
    TYPE_BILINEAR,
    TYPE_CATMARK,
    TYPE_LOOP
};


///
///  \brief Enumerated type for all face splitting scheme
///
enum Split {
    SPLIT_TO_QUADS,  ///< Used by Catmark and Bilinear
    SPLIT_TO_TRIS,   ///< Used by Loop
    SPLIT_HYBRID     ///< Not currently used (potential future extension)
};

///
///  \brief Traits associated the types of all subdivision schemes -- parameterized by
///  the scheme type.  All traits are also defined on the scheme itself.
///
struct TypeTraits {

    static Type GetType(Type schemeType) { return schemeType; }

    static Split GetTopologicalSplitType(Type schemeType);
    static int   GetRegularFaceSize(Type schemeType);
    static int   GetRegularVertexValence(Type schemeType);
    static int   GetLocalNeighborhoodSize(Type schemeType);

    static char const* GetName(Type schemeType);
};


} // end namespace sdc

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;
} // end namespace OpenSubdiv

#endif /* SDC_TYPE_H */
