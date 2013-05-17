//
//     Copyright (C) Pixar. All rights reserved.
//
//     This license governs use of the accompanying software. If you
//     use the software, you accept this license. If you do not accept
//     the license, do not use the software.
//
//     1. Definitions
//     The terms "reproduce," "reproduction," "derivative works," and
//     "distribution" have the same meaning here as under U.S.
//     copyright law.  A "contribution" is the original software, or
//     any additions or changes to the software.
//     A "contributor" is any person or entity that distributes its
//     contribution under this license.
//     "Licensed patents" are a contributor's patent claims that read
//     directly on its contribution.
//
//     2. Grant of Rights
//     (A) Copyright Grant- Subject to the terms of this license,
//     including the license conditions and limitations in section 3,
//     each contributor grants you a non-exclusive, worldwide,
//     royalty-free copyright license to reproduce its contribution,
//     prepare derivative works of its contribution, and distribute
//     its contribution or any derivative works that you create.
//     (B) Patent Grant- Subject to the terms of this license,
//     including the license conditions and limitations in section 3,
//     each contributor grants you a non-exclusive, worldwide,
//     royalty-free license under its licensed patents to make, have
//     made, use, sell, offer for sale, import, and/or otherwise
//     dispose of its contribution in the software or derivative works
//     of the contribution in the software.
//
//     3. Conditions and Limitations
//     (A) No Trademark License- This license does not grant you
//     rights to use any contributor's name, logo, or trademarks.
//     (B) If you bring a patent claim against any contributor over
//     patents that you claim are infringed by the software, your
//     patent license from such contributor to the software ends
//     automatically.
//     (C) If you distribute any portion of the software, you must
//     retain all copyright, patent, trademark, and attribution
//     notices that are present in the software.
//     (D) If you distribute any portion of the software in source
//     code form, you may do so only under this license by including a
//     complete copy of this license with your distribution. If you
//     distribute any portion of the software in compiled or object
//     code form, you may only do so under a license that complies
//     with this license.
//     (E) The software is licensed "as-is." You bear the risk of
//     using it. The contributors give no express warranties,
//     guarantees or conditions. You may have additional consumer
//     rights under your local laws which this license cannot change.
//     To the extent permitted under your local laws, the contributors
//     exclude the implied warranties of merchantability, fitness for
//     a particular purpose and non-infringement.
//

#ifndef FAR_PATCH_PARAM_H
#define FAR_PATCH_PARAM_H

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
///   level:4      - the subdivision level of the patch
///   nonquad:1;   - whether the patch is the child of a non-quad face
///   rotation:2;  - patch rotations necessary to match CCW face-winding
///   v:10;        - log2 value of u parameter at first patch corner
///   u:10;        - log2 value of v parameter at first patch corner
///   reserved1:5; - padding
///
/// Note : the bitfield is not expanded in the struct due to differences in how
///        GPU & CPU compilers pack bit-fields and endian-ness.
///
struct FarPatchParam {
    unsigned int faceIndex:32; // Ptex face index
    
    struct BitField {
        unsigned int field:32;
        
        /// Sets the values of the bit fields
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

        /// Returns the log2 value of the u parameter at the top left corner of
        /// the patch
        unsigned short GetU() const { return (field >> 17) & 0x3ff; }

        /// Returns the log2 value of the v parameter at the top left corner of
        /// the patch
        unsigned short GetV() const { return (field >> 7) & 0x3ff; }

        /// Returns the rotation of the patch (the number of CCW parameter winding)
        unsigned char GetRotation() const { return (field >> 5) & 0x3; }

        /// True if the parent coarse face is a non-quad
        bool NonQuadRoot() const { return (field >> 4) & 0x1; }

        /// Returns the level of subdivision of the patch 
        unsigned char GetDepth() const { return (field & 0xf); }

        /// Resets the values to 0
        void Clear() { field = 0; }
                
    } bitField;

    /// Sets the values of the bit fields
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
    
    /// Resets everything to 0
    void Clear() { 
        faceIndex = 0;
        bitField.Clear();
    }    
};

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* FAR_PATCH_PARAM */
