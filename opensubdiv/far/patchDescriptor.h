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

#ifndef OPENSUBDIV3_FAR_PATCH_DESCRIPTOR_H
#define OPENSUBDIV3_FAR_PATCH_DESCRIPTOR_H

#include "../version.h"

#include "../far/types.h"
#include "../sdc/types.h"

#include <vector>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Far {

/// \brief Describes the type of a patch
///
/// Uniquely identifies all the different types of patches
///
/// * Uniformly subdivided meshes contain bilinear patches of either QUADS
///   or TRIANGLES
///
/// * Adaptively subdivided meshes contain bicubic patches of types REGULAR,
///   GREGORY, GREGORY_BOUNDARY, GREGORY_BASIS.
///
/// Bitfield layout :
///
///  Field       | Bits | Content
///  ------------|:----:|------------------------------------------------------
///  _type       | 4    | patch type
///
class PatchDescriptor {

public:

    enum Type {
        NON_PATCH = 0,     ///< undefined

        POINTS,            ///< points (useful for cage drawing)
        LINES,             ///< lines  (useful for cage drawing)

        QUADS,             ///< bilinear quads-only patches
        TRIANGLES,         ///< bilinear triangles-only mesh

        LOOP,              ///< Loop patch

        REGULAR,           ///< feature-adaptive bicubic patches
        GREGORY,
        GREGORY_BOUNDARY,
        GREGORY_BASIS
    };

public:

    /// \brief Default constructor.
    PatchDescriptor() :
        _type(NON_PATCH) { }

    /// \brief Constructor
    PatchDescriptor(int type) :
        _type(type) { }

    /// \brief Copy Constructor
    PatchDescriptor( PatchDescriptor const & d ) :
        _type(d.GetType()) { }

    /// \brief Returns the type of the patch
    Type GetType() const {
        return (Type)_type;
    }

    /// \brief Returns true if the type is an adaptive patch
    static inline bool IsAdaptive(Type type) {
        return (type>=LOOP && type<=GREGORY_BASIS);
    }

    /// \brief Returns true if the type is an adaptive patch
    bool IsAdaptive() const {
        return IsAdaptive( this->GetType() );
    }

    /// \brief Returns the number of control vertices expected for a patch of the
    /// type described
    static inline short GetNumControlVertices( Type t );

    static inline short GetNumFVarControlVertices( Type t );

    /// \brief Returns the number of control vertices expected for a patch of the
    /// type described
    short GetNumControlVertices() const {
        return GetNumControlVertices( this->GetType() );
    }

    /// \brief Returns the number of control vertices expected for a patch of the
    /// type described
    short GetNumFVarControlVertices() const {
        return GetNumFVarControlVertices( this->GetType() );
    }

    /// \brief Number of control vertices of Regular Patches in table.
    static short GetRegularPatchSize() { return 16; }

    /// \brief Number of control vertices of Gregory (and Gregory Boundary) Patches in table.
    static short GetGregoryPatchSize() { return 4; }

    /// \brief Number of control vertices of Gregory patch basis (20)
    static short GetGregoryBasisPatchSize() { return 20; }


    /// \brief Returns a vector of all the legal patch descriptors for the
    ///        given adaptive subdivision scheme
    static Vtr::ConstArray<PatchDescriptor> GetAdaptivePatchDescriptors(Sdc::SchemeType type);

    /// \brief Allows ordering of patches by type
    inline bool operator < ( PatchDescriptor const other ) const;

    /// \brief True if the descriptors are identical
    inline bool operator == ( PatchDescriptor const other ) const;

    // debug helper
    void print() const;

private:
    unsigned int  _type:4;
};

typedef Vtr::ConstArray<PatchDescriptor> ConstPatchDescriptorArray;

// Returns the number of control vertices expected for a patch of this type
inline short
PatchDescriptor::GetNumControlVertices( Type type ) {
    switch (type) {
        case REGULAR           : return GetRegularPatchSize();
        case QUADS             : return 4;
        case GREGORY           :
        case GREGORY_BOUNDARY  : return GetGregoryPatchSize();
        case GREGORY_BASIS     : return GetGregoryBasisPatchSize();
        case TRIANGLES         : return 3;
        case LINES             : return 2;
        case POINTS            : return 1;
        default : return -1;
    }
}

// Returns the number of face-varying control vertices expected for a patch of this type
inline short
PatchDescriptor::GetNumFVarControlVertices( Type type ) {
    switch (type) {
        case REGULAR           : return GetRegularPatchSize();
        case QUADS             : return 4;
        case TRIANGLES         : return 3;
        case LINES             : return 2;
        case POINTS            : return 1;
        case GREGORY_BASIS     : assert(0); return GetGregoryBasisPatchSize();
        case GREGORY           :
        case GREGORY_BOUNDARY  : assert(0); // unsupported types
        default : return -1;
    }
}

// Allows ordering of patches by type
inline bool
PatchDescriptor::operator < ( PatchDescriptor const other ) const {
    return (_type < other._type);
}

// True if the descriptors are identical
inline bool
PatchDescriptor::operator == ( PatchDescriptor const other ) const {
    return _type == other._type;
}



} // end namespace Far

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* OPENSUBDIV3_FAR_PATCH_DESCRIPTOR_H */
