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

#ifndef OPENSUBDIV3_FAR_PATCH_PARAM_H
#define OPENSUBDIV3_FAR_PATCH_PARAM_H

#include "../version.h"

#include "../far/types.h"

#include <cassert>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Far {

/// \brief Local patch parameterization descriptor
///
/// Coarse mesh faces are split into sets of patches in both uniform and feature
/// adaptive modes. In order to maintain local patch parameterization, it is
/// necessary to retain some information, such as level of subdivision, face-
/// winding status... This parameterization is directly applicable to ptex textures,
/// but has to be remapped to a specific layout for uv textures.
///
/// Bitfield layout :
///
///  Field      | Bits | Content
///  -----------|:----:|------------------------------------------------------
///  level      | 3    | the subdivision level of the patch
///  nonquad    | 1    | whether the patch is the child of a non-quad face
///  boundary   | 4    | boundary edge mask encoding
///  transition | 4    | transition edge mask encoding
///  v          | 10   | log2 value of u parameter at first patch corner
///  u          | 10   | log2 value of v parameter at first patch corner
///
/// Note : the bitfield is not expanded in the struct due to differences in how
///        GPU & CPU compilers pack bit-fields and endian-ness.
///
struct PatchParam {
    Index faceIndex:32; // Ptex face index

    struct BitField {
        unsigned int field:32;

        /// \brief Sets the values of the bit fields
        ///
        /// @param u value of the u parameter for the first corner of the face
        /// @param v value of the v parameter for the first corner of the face
        ///
        /// @param rots rotations required to reproduce CCW face-winding
        /// @param depth subdivision level of the patch
        /// @param nonquad true if the root face is not a quad
        ///
        void Set( short u, short v, unsigned char depth, bool nonquad,
                  unsigned short boundary, unsigned short transition );

        /// \brief Returns the log2 value of the u parameter at the top left corner of
        /// the patch
        unsigned short GetU() const { return (unsigned short)((field >> 22) & 0x3ff); }

        /// \brief Returns the log2 value of the v parameter at the top left corner of
        /// the patch
        unsigned short GetV() const { return (unsigned short)((field >> 12) & 0x3ff); }

        /// \brief Returns the transition edge encoding for the patch.
        unsigned short GetTransition() const { return (unsigned short)((field >> 8) & 0xf); }

        /// \brief Returns the boundary edge encoding for the patch.
        unsigned short GetBoundary() const { return (unsigned short)((field >> 4) & 0xf); }

        /// \brief True if the parent coarse face is a non-quad
        bool NonQuadRoot() const { return (field >> 3) & 0x1; }

        /// \brief Returns the fratcion of normalized parametric space covered by the
        /// sub-patch.
        float GetParamFraction() const;

        /// \brief Returns the level of subdivision of the patch
        unsigned char GetDepth() const { return  (unsigned char)(field & 0x7); }

        /// The (u,v) pair is normalized to this sub-parametric space.
        ///
        /// @param u  u parameter
        /// @param v  v parameter
        ///
        void Normalize( float & u, float & v ) const;

        /// \brief Resets the values to 0
        void Clear() { field = 0; }

    } bitField;

    /// \brief Sets the values of the bit fields
    ///
    /// @param faceid ptex face index
    ///
    /// @param u value of the u parameter for the first corner of the face
    /// @param v value of the v parameter for the first corner of the face
    ///
    /// @param rots rotations required to reproduce CCW face-winding
    /// @param depth subdivision level of the patch
    /// @param nonquad true if the root face is not a quad
    ///
    void Set( Index faceid, short u, short v, unsigned char depth, bool nonquad ,
              unsigned short boundary, unsigned short transition );

    /// \brief Resets everything to 0
    void Clear();
};

typedef std::vector<PatchParam> PatchParamTable;

typedef Vtr::Array<PatchParam> PatchParamArray;
typedef Vtr::ConstArray<PatchParam> ConstPatchParamArray;

inline void
PatchParam::BitField::Set( short u, short v, unsigned char depth, bool nonquad,
                           unsigned short boundary, unsigned short transition ) {
    field = (u << 22) |
            (v << 12) |
            (transition << 8) |
            (boundary << 4) |
            ((nonquad ? 1:0) << 3) |
            (nonquad ? depth+1 : depth);
}


inline float
PatchParam::BitField::GetParamFraction( ) const {
    if (NonQuadRoot()) {
        return 1.0f / float( 1 << (GetDepth()-1) );
    } else {
        return 1.0f / float( 1 << GetDepth() );
    }
}

inline void
PatchParam::BitField::Normalize( float & u, float & v ) const {

    float frac = GetParamFraction();

    // top left corner
    float pu = (float)GetU()*frac;
    float pv = (float)GetV()*frac;

    // normalize u,v coordinates
    u = (u - pu) / frac,
    v = (v - pv) / frac;
}

inline void
PatchParam::Set( Index faceid, short u, short v, unsigned char depth, bool nonquad,
                 unsigned short boundary, unsigned short transition ) {
    faceIndex = faceid;
    bitField.Set(u,v,depth,nonquad,boundary,transition);
}

inline void
PatchParam::Clear() {
    faceIndex = 0;
    bitField.Clear();
}

} // end namespace Far

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* OPENSUBDIV3_FAR_PATCH_PARAM */
