//
//     Copyright 2013 Pixar
//
//     Licensed under the Apache License, Version 2.0 (the "License");
//     you may not use this file except in compliance with the License
//     and the following modification to it: Section 6 Trademarks.
//     deleted and replaced with:
//
//     6. Trademarks. This License does not grant permission to use the
//     trade names, trademarks, service marks, or product names of the
//     Licensor and its affiliates, except as required for reproducing
//     the content of the NOTICE file.
//
//     You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
//     Unless required by applicable law or agreed to in writing,
//     software distributed under the License is distributed on an
//     "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
//     either express or implied.  See the License for the specific
//     language governing permissions and limitations under the
//     License.
//

#ifndef FAR_PATCH_PARAM_H
#define FAR_PATCH_PARAM_H

#include "../version.h"

#include <cassert>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

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
///  level      | 4    | the subdivision level of the patch                   
///  nonquad    | 1    | whether the patch is the child of a non-quad face    
///  rotation   | 2    | patch rotations necessary to match CCW face-winding  
///  v          | 10   | log2 value of u parameter at first patch corner      
///  u          | 10   | log2 value of v parameter at first patch corner      
///  reserved1  | 5    | padding                                              
/// 
/// Note : the bitfield is not expanded in the struct due to differences in how
///        GPU & CPU compilers pack bit-fields and endian-ness.
///
struct FarPatchParam {
    unsigned int faceIndex:32; // Ptex face index
    
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
        void Set( short u, short v, unsigned char rots, unsigned char depth, bool nonquad ) {
            field = (u << 17) |
                    (v << 7) |
                    (rots << 5) |
                    ((nonquad ? 1:0) << 4) |
                    (nonquad ? depth+1 : depth);
        }

        /// \brief Returns the log2 value of the u parameter at the top left corner of
        /// the patch
        unsigned short GetU() const { return (field >> 17) & 0x3ff; }

        /// \brief Returns the log2 value of the v parameter at the top left corner of
        /// the patch
        unsigned short GetV() const { return (field >> 7) & 0x3ff; }

        /// \brief Returns the rotation of the patch (the number of CCW parameter winding)
        unsigned char GetRotation() const { return (field >> 5) & 0x3; }

        /// \brief True if the parent coarse face is a non-quad
        bool NonQuadRoot() const { return (field >> 4) & 0x1; }
        
        /// \brief Returns the fratcion of normalized parametric space covered by the 
        /// sub-patch.
        float GetParamFraction() const;

        /// \brief Returns the level of subdivision of the patch 
        unsigned char GetDepth() const { return (field & 0xf); }

        /// The (u,v) pair is normalized to this sub-parametric space. 
        ///
        /// @param u  u parameter
        ///
        /// @param v  v parameter
        ///
        void Normalize( float & u, float & v ) const;
        
        /// \brief Rotate (u,v) pair to compensate for transition pattern and boundary
        /// orientations.
        ///
        /// @param u  u parameter
        ///
        /// @param v  v parameter
        ///
        void Rotate( float & u, float & v ) const;

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
    void Set( unsigned int faceid, short u, short v, unsigned char rots, unsigned char depth, bool nonquad ) {
        faceIndex = faceid;
        bitField.Set(u,v,rots,depth,nonquad);
    }
    
    /// \brief Resets everything to 0
    void Clear() { 
        faceIndex = 0;
        bitField.Clear();
    }
};

inline float 
FarPatchParam::BitField::GetParamFraction( ) const {
    if (NonQuadRoot()) {
        return 1.0f / float( 1 << (GetDepth()-1) );
    } else {
        return 1.0f / float( 1 << GetDepth() );
    }
}

inline void
FarPatchParam::BitField::Normalize( float & u, float & v ) const {

    float frac = GetParamFraction();

    // top left corner
    float pu = (float)GetU()*frac;
    float pv = (float)GetV()*frac;

    // normalize u,v coordinates
    u = (u - pu) / frac,
    v = (v - pv) / frac;
}

inline void 
FarPatchParam::BitField::Rotate( float & u, float & v ) const {
    switch( GetRotation() ) {
         case 0 : break;
         case 1 : { float tmp=v; v=1.0f-u; u=tmp; } break;
         case 2 : { u=1.0f-u; v=1.0f-v; } break;
         case 3 : { float tmp=u; u=1.0f-v; v=tmp; } break;
         default:
             assert(0);
    }
}

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* FAR_PATCH_PARAM */
