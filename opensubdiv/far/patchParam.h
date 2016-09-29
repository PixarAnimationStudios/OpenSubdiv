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

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Far {

/// \brief Patch parameterization
///
/// Topological refinement splits coarse mesh faces into refined faces.
///
/// This patch parameterzation describes the relationship between one
/// of these refined faces and its corresponding coarse face. It is used
/// both for refined faces that are represented as full limit surface
/// parametric patches as well as for refined faces represented as simple
/// triangles or quads. This parameterization is needed to interpolate
/// primvar data across a refined face.
///
/// The U,V and refinement level parameters describe the scale and offset
/// needed to map a location on the patch between levels of refinement.
/// The encoding of these values exploits the quad-tree organization of
/// the faces produced by subdivision. We encode the U,V origin of the
/// patch using two 10-bit integer values and the refinement level as
/// a 4-bit integer. This is sufficient to represent up through 10 levels
/// of refinement.
///
/// Special consideration must be given to the refined faces resulting from
/// irregular coarse faces. We adopt a convention similar to Ptex texture
/// mapping and define the parameterization for these faces in terms of the
/// regular faces resulting from the first topological splitting of the
/// irregular coarse face.
///
/// When computing the basis functions needed to evaluate the limit surface
/// parametric patch representing a refined face, we also need to know which
/// edges of the patch are interpolated boundaries. These edges are encoded
/// as a boundary bitmask identifying the boundary edges of the patch in
/// sequential order starting from the first vertex of the refined face.
///
/// A sparse topological refinement (like feature adaptive refinement) can
/// produce refined faces that are adjacent to faces at the next level of
/// subdivision. We identify these transitional edges with a transition
/// bitmask using the same encoding as the boundary bitmask.
///
/// For triangular subdivision schemes we specify the parameterization using
/// a similar method. Alternate triangles at a given level of refinement
/// are parameterized from their opposite corners and encoded as occupying
/// the opposite diagonal of the quad-tree hierarchy. The third barycentric
/// coordinate is dependent on and can be derived from the other two
/// coordinates. This encoding also takes inspiration from the Ptex
/// texture mapping specification.
///
/// Bitfield layout :
///
///  Field0     | Bits | Content
///  -----------|:----:|------------------------------------------------------
///  faceId     | 28   | the faceId of the patch
///  transition | 4    | transition edge mask encoding
///
///  Field1     | Bits | Content
///  -----------|:----:|------------------------------------------------------
///  level      | 4    | the subdivision level of the patch
///  nonquad    | 1    | whether patch is refined from a non-quad face
///  regular    | 1    | whether patch is regular
///  unused     | 2    | unused
///  boundary   | 4    | boundary edge mask encoding
///  v          | 10   | log2 value of u parameter at first patch corner
///  u          | 10   | log2 value of v parameter at first patch corner
///
/// Note : the bitfield is not expanded in the struct due to differences in how
///        GPU & CPU compilers pack bit-fields and endian-ness.
///
/*!
 \verbatim
 Quad Patch Parameterization

 (0,1)                           (1,1)
   +-------+-------+---------------+
   |       |       |               |
   |   L2  |   L2  |               |
   |0,3    |1,3    |               |
   +-------+-------+       L1      |
   |       |       |               |
   |   L2  |   L2  |               |
   |0,2    |1,2    |1,1            |
   +-------+-------+---------------+
   |               |               |
   |               |               |
   |               |               |
   |       L1      |       L1      |
   |               |               |
   |               |               |
   |0,0            |1,0            |
   +---------------+---------------+
 (0,0)                           (1,0)
 \endverbatim
*/
/*!
 \verbatim
 Triangle Patch Parameterization

 (0,1)                           (1,1)  (0,1,0)
   +-------+-------+---------------+       +
   | \     | \     | \             |       | \
   |L2 \   |L2 \   |   \           |       |   \
   |0,3  \ |1,3  \ |     \         |       | L2  \
   +-------+-------+       \       |       +-------+
   | \     | \     |   L1    \     |       | \  L2 | \
   |L2 \   |L2 \   |           \   |       |   \   |   \
   |0,2  \ |1,2  \ |1,1          \ |       | L2  \ | L2  \
   +-------+-------+---------------+       +-------+-------+
   | \             | \             |       | \             | \
   |   \           |   \           |       |   \           |   \
   |     \         |     \         |       |     \    L1   |     \
   |       \       |       \       |       |       \       |       \
   |   L1    \     |   L1    \     |       |   L1    \     |   L1    \
   |           \   |           \   |       |           \   |           \
   |0,0          \ |1,0          \ |       |             \ |             \
   +---------------+---------------+       +---------------+---------------+
 (0,0)                           (1,0)  (0,0,1)                         (1,0,0)
 \endverbatim
*/

struct PatchParam {
    /// \brief Sets the values of the bit fields
    ///
    /// @param faceid face index
    ///
    /// @param u value of the u parameter for the first corner of the face
    /// @param v value of the v parameter for the first corner of the face
    ///
    /// @param depth subdivision level of the patch
    /// @param nonquad true if the root face is not a quad
    ///
    /// @param boundary 4-bits identifying boundary edges
    /// @param transition 4-bits identifying transition edges
    ///
    /// @param regular whether the patch is regular
    ///
    void Set(Index faceid, short u, short v,
             unsigned short depth, bool nonquad,
             unsigned short boundary, unsigned short transition,
             bool regular = false);

    /// \brief Resets everything to 0
    void Clear() { field0 = field1 = 0; }

    /// \brief Retuns the faceid
    Index GetFaceId() const { return Index(unpack(field0,28,0)); }

    /// \brief Returns the log2 value of the u parameter at
    /// the first corner of the patch
    unsigned short GetU() const { return (unsigned short)unpack(field1,10,22); }

    /// \brief Returns the log2 value of the v parameter at
    /// the first corner of the patch
    unsigned short GetV() const { return (unsigned short)unpack(field1,10,12); }

    /// \brief Returns the transition edge encoding for the patch.
    unsigned short GetTransition() const { return (unsigned short)unpack(field0,4,28); }

    /// \brief Returns the boundary edge encoding for the patch.
    unsigned short GetBoundary() const { return (unsigned short)unpack(field1,4,8); }

    /// \brief True if the parent coarse face is a non-quad
    bool NonQuadRoot() const { return (unpack(field1,1,4) != 0); }

    /// \brief Returns the level of subdivision of the patch
    unsigned short GetDepth() const { return (unsigned short)unpack(field1,4,0); }

    /// \brief Returns the fraction of the coarse face parametric space
    /// covered by this refined face.
    float GetParamFraction() const;

    /// \brief Maps the (u,v) parameterization from coarse to refined
    /// The (u,v) pair is mapped from the base face parameterization to
    /// the refined face parameterization
    ///
    void MapBaseToRefined( float & u, float & v ) const;

    /// \brief Maps the (u,v) parameterization from refined to coarse
    /// The (u,v) pair is mapped from the refined face parameterization to
    /// the base face parameterization
    ///
    void MapRefinedToBase( float & u, float & v ) const;

    /// \brief The (u,v) pair is normalized to this sub-parametric space.
    ///
    /// @param u  u parameter
    /// @param v  v parameter
    ///
    /// @see PatchParam#MapBaseToRefined
    void Normalize( float & u, float & v ) const;

    /// \brief Returns whether the patch is regular
    bool IsRegular() const { return (unpack(field1,1,5) != 0); }

    unsigned int field0:32;
    unsigned int field1:32;

private:
    unsigned int pack(unsigned int value, int width, int offset) const {
        return (unsigned int)((value & ((1<<width)-1)) << offset);
    }

    unsigned int unpack(unsigned int value, int width, int offset) const {
        return (unsigned short)((value >> offset) & ((1<<width)-1));
    }
};

typedef std::vector<PatchParam> PatchParamTable;

typedef Vtr::Array<PatchParam> PatchParamArray;
typedef Vtr::ConstArray<PatchParam> ConstPatchParamArray;

inline void
PatchParam::Set(Index faceid, short u, short v,
                unsigned short depth, bool nonquad,
                unsigned short boundary, unsigned short transition,
                bool regular) {
    field0 = pack(faceid,    28,  0) |
             pack(transition, 4, 28);

    field1 = pack(u,         10, 22) |
             pack(v,         10, 12) |
             pack(boundary,   4,  8) |
             pack(regular,    1,  5) |
             pack(nonquad,    1,  4) |
             pack(depth,      4,  0);
}

inline float
PatchParam::GetParamFraction( ) const {
    if (NonQuadRoot()) {
        return 1.0f / float( 1 << (GetDepth()-1) );
    } else {
        return 1.0f / float( 1 << GetDepth() );
    }
}

inline void
PatchParam::MapBaseToRefined( float & u, float & v ) const {

    float frac = GetParamFraction();

    float pu = (float)GetU()*frac;
    float pv = (float)GetV()*frac;

    u = (u - pu) / frac,
    v = (v - pv) / frac;
}

inline void
PatchParam::MapRefinedToBase( float & u, float & v ) const {

    float frac = GetParamFraction();

    float pu = (float)GetU()*frac;
    float pv = (float)GetV()*frac;

    u = u * frac + pu,
    v = v * frac + pv;
}

inline void
PatchParam::Normalize( float & u, float & v ) const {
    return MapBaseToRefined(u, v);
}

} // end namespace Far

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* OPENSUBDIV3_FAR_PATCH_PARAM */
