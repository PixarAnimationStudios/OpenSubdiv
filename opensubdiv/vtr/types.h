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
#ifndef VTR_TYPES_H
#define VTR_TYPES_H

#include "../version.h"

#include "../vtr/array.h"

#include <vector>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Vtr {

//
//  A few types (and constants) are declared here while Vtr is being
//  developed.  These tend to be used by more than one Vtr class, i.e.
//  both Level and Refinement and are often present in their
//  interfaces.
//
//  Is the sharpness overkill -- perhaps sdc should define this...
//
typedef float Sharpness;

//
//  Indices -- note we can't use sized integer types like uint32_t, etc. as use of
//  stdint is not portable.
//
//  The convention throughout the OpenSubdiv code is to use "int" in most places,
//  with "unsigned int" being limited to a few cases (why?).  So we continue that
//  trend here and use "int" for topological indices (with -1 indicating "invalid")
//  despite the fact that we lose half the range compared to using "uint" (with ~0
//  as invalid).
//
typedef int            Index;      //  Used to index the vectors of components
typedef unsigned char  LocalIndex; //  Used to index one component within another

static const Index  INDEX_INVALID = -1;

inline bool IndexIsValid(Index index) { return (index != INDEX_INVALID); }

//
//  Note for aggregate types the use of "vector" wraps an std:;vector (typically a
//  member variable) and is fully resizable and owns its own storage, whereas "array"
//  is typically used in index a fixed subset of pre-allocated memory:
//
typedef std::vector<Index>  IndexVector;

typedef Array<Index>        IndexArray;
typedef Array<LocalIndex>   LocalIndexArray;


} // end namespace Vtr

} // end namespace OPENSUBDIV_VERSION

using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* VTR_TYPES_H */
