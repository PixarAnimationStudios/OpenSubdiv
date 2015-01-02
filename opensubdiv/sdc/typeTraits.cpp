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
#include "../sdc/type.h"

#include "../sdc/bilinearScheme.h"
#include "../sdc/catmarkScheme.h"
#include "../sdc/loopScheme.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Sdc {

struct TypeTraitsEntry {
    char const * _name;

    Split _splitType;
    int   _regularFaceSize;
    int   _regularVertexValence;
    int   _localNeighborhood;
};

static const TypeTraitsEntry typeTraitsTable[3] = {
    { "bilinear", Scheme<TYPE_BILINEAR>::GetTopologicalSplitType(),
                  Scheme<TYPE_BILINEAR>::GetRegularFaceSize(),
                  Scheme<TYPE_BILINEAR>::GetRegularVertexValence(),
                  Scheme<TYPE_BILINEAR>::GetLocalNeighborhoodSize() },
    { "catmark",  Scheme<TYPE_CATMARK>::GetTopologicalSplitType(),
                  Scheme<TYPE_CATMARK>::GetRegularFaceSize(),
                  Scheme<TYPE_CATMARK>::GetRegularVertexValence(),
                  Scheme<TYPE_CATMARK>::GetLocalNeighborhoodSize() },
    { "loop",     Scheme<TYPE_LOOP>::GetTopologicalSplitType(),
                  Scheme<TYPE_LOOP>::GetRegularFaceSize(),
                  Scheme<TYPE_LOOP>::GetRegularVertexValence(),
                  Scheme<TYPE_LOOP>::GetLocalNeighborhoodSize() }
};

//
//  Static methods for TypeTraits:
//
char const*
TypeTraits::GetName(Type schemeType) {

    return typeTraitsTable[schemeType]._name;
}

Split
TypeTraits::GetTopologicalSplitType(Type schemeType) {

    return typeTraitsTable[schemeType]._splitType;
}

int
TypeTraits::GetRegularFaceSize(Type schemeType) {

    return typeTraitsTable[schemeType]._regularFaceSize;
}

int
TypeTraits::GetRegularVertexValence(Type schemeType) {

    return typeTraitsTable[schemeType]._regularVertexValence;
}

int
TypeTraits::GetLocalNeighborhoodSize(Type schemeType) {

    return typeTraitsTable[schemeType]._localNeighborhood;
}

} // end namespace sdc

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv
