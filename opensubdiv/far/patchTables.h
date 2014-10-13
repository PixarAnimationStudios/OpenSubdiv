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

#ifndef FAR_PATCH_TABLES_H
#define FAR_PATCH_TABLES_H

#include "../version.h"

#include "../far/patchParam.h"
#include "../far/patchParam.h"
#include "../far/types.h"

#include "../sdc/type.h"

#include <cstdlib>
#include <cassert>
#include <algorithm>
#include <vector>
#include <map>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Far {

/// \brief Container for patch vertex indices tables
///
/// PatchTables contain the lists of vertices for each patch of an adaptive
/// mesh representation.
///
class PatchTables {

public:

    typedef std::vector<Index>      PTable;
    typedef std::vector<Index>      VertexValenceTable;
    typedef std::vector<Index>      QuadOffsetTable;
    typedef std::vector<PatchParam> PatchParamTable;

    enum Type {
        NON_PATCH = 0,     ///< undefined

        POINTS,            ///< points (useful for cage drawing)
        LINES,             ///< lines  (useful for cage drawing)

        QUADS,             ///< bilinear quads-only patches
        TRIANGLES,         ///< bilinear triangles-only mesh

        LOOP,              ///< Loop patch  (currently unsupported)

        REGULAR,           ///< feature-adaptive bicubic patches
        SINGLE_CREASE,
        BOUNDARY,
        CORNER,
        GREGORY,
        GREGORY_BOUNDARY
    };

    enum TransitionPattern {
        NON_TRANSITION = 0,
        PATTERN0,
        PATTERN1,
        PATTERN2,
        PATTERN3,
        PATTERN4
    };

public:

    /// \brief Describes the type of a patch
    ///
    /// Uniquely identifies all the types of patches in a mesh :
    ///
    /// * Raw polygon meshes are identified as POLYGONS and can contain faces
    ///   with arbitrary number of vertices
    ///
    /// * Uniformly subdivided meshes contain bilinear patches of either QUADS
    ///   or TRIANGLES
    ///
    /// * Adaptively subdivided meshes contain bicubic patches of types REGULAR,
    ///   BOUNDARY, CORNER, GREGORY, GREGORY_BOUNDARY. These bicubic patches are
    ///   also further distinguished by a transition pattern as well as a rotational
    ///   orientation.
    ///
    class Descriptor {

    public:
        /// \brief Default constructor.
        Descriptor() :
            _type(NON_PATCH), _pattern(NON_TRANSITION), _rotation(0) {}

        /// \brief Constructor
        Descriptor(int type, int pattern, unsigned char rotation) :
            _type(type), _pattern(pattern), _rotation(rotation) { }

        /// \brief Copy Constructor
        Descriptor( Descriptor const & d ) :
            _type(d.GetType()), _pattern(d.GetPattern()), _rotation(d.GetRotation()) { }

        /// \brief Returns the type of the patch
        Type GetType() const {
            return (Type)_type;
        }

        /// \brief Returns the transition pattern of the patch if any (5 types)
        TransitionPattern GetPattern() const {
            return (TransitionPattern)_pattern;
        }

        /// \brief Returns the rotation of the patch (4 rotations)
        unsigned char GetRotation() const {
            return (unsigned char)_rotation;
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

        /// \brief Allows ordering of patches by type
        inline bool operator < ( Descriptor const other ) const;

        /// \brief True if the descriptors are identical
        inline bool operator == ( Descriptor const other ) const;

    private:
        friend class PatchTablesFactory;

        unsigned int  _type:4;
        unsigned int  _pattern:3;
        unsigned int  _rotation:2;
    };

    typedef std::vector<Descriptor> DescriptorVector;

    /// \brief Returns a vector of all the legal patch descriptors for the
    ///        given adaptive subdivision scheme
    static DescriptorVector const & GetAdaptiveDescriptors(Sdc::Type type);


public:

    /// \brief Array of patches of the same type
    class PatchArray {

    public:
        /// \brief Constructor.
        ///
        /// @param desc             descriptor information for the patches in
        ///                         the array
        ///
        /// @param vertIndex        absolute index to the first control vertex
        ///                         of the first patch in the PTable
        ///
        /// @param patchIndex       absolute index of the first patch in the
        ///                         array
        ///
        /// @param npatches         number of patches in the array
        ///
        /// @param quadOffsetIndex  absolute index of the first quad offset
        ///                         entry
        ///
        PatchArray( Descriptor desc, Index vertIndex, Index patchIndex,
            Index npatches, Index quadOffsetIndex ) :
                _desc(desc), _range(vertIndex, patchIndex, npatches, quadOffsetIndex) { }

        /// Returns a patch descriptor defining the type of patches in the array
        Descriptor GetDescriptor() const {
            return _desc;
        }

        /// \brief Describes the range of patches in a PatchArray
        struct ArrayRange {

            /// \brief Constructor
            ///
            /// @param vIndex     absolute index to the first control vertex
            ///                   of the first patch in the PTable
            ///
            /// @param pIndex     absolute index of the first patch in the
            ///                   array
            ///
            /// @param npatches   number of patches in the array
            ///
            /// @param qoIndex    absolute index of the first quad offset
            ///                   entry
            ///
            ArrayRange( Index vIndex, Index pIndex, int npatches, Index qoIndex ) :
                npatches(npatches), vertIndex(vIndex), patchIndex(pIndex),
                    quadOffsetIndex(qoIndex) { }

            int npatches;          ///< number of patches in the array

            Index vertIndex,       ///< absolute index to the first control vertex of the first patch in the PTable
                  patchIndex,      ///< absolute index of the first patch in the array
                  quadOffsetIndex; ///< absolute index of the first quad offset entry
        };

        /// \brief Returns a array range struct
        ArrayRange const & GetArrayRange() const {
            return _range;
        }

        /// \brief Returns the index of the first control vertex of the first patch
        /// of this array in the global PTable
        Index GetVertIndex() const {
            return _range.vertIndex;
        }

        /// \brief Returns the global index of the first patch in this array (Used to
        /// access param / fvar table data)
        Index GetPatchIndex() const {
            return _range.patchIndex;
        }

        /// \brief Returns the number of patches in the array
        int GetNumPatches() const {
            return _range.npatches;
        }

        /// \brief Returns the index to the first entry in the QuadOffsetTable
        Index GetQuadOffsetIndex() const {
            return _range.quadOffsetIndex;
        }

    private:
        friend class PatchTablesFactory;

        Descriptor _desc;   // type of patches in the array

        ArrayRange _range;  // index locators in the array
    };

    typedef std::vector<PatchArray> PatchArrayVector;

    /// \brief Handle that can be used as unique patch identifier within PatchTables
    struct PatchHandle {
        Index patchArrayIdx,  ///< OsdPatchArray containing the patch
              patchIdx,       ///< Index of the patch in the OsdPatchArray
              vertexOffset;   ///< Relative offset to the first CV of the patch in the patch array
    };

    /// \brief Get the table of patch control vertices
    PTable const & GetPatchTable() const { return _patches; }

    /// \brief Returns a pointer to the array of patches matching the descriptor
    PatchArray const * GetPatchArray( Descriptor desc ) const {
        return const_cast<PatchTables *>(this)->findPatchArray( desc );
    }

    /// \brief Returns all arrays of patches
    PatchArrayVector const & GetPatchArrayVector() const {
        return _patchArrays;
    }

    /// brief Returns a pointer to the PatchArry of uniformly subdivided faces at 'level'
    ///
    /// @param level  the level of subdivision of the faces (returns the highest
    ///               level by default)
    ///
    /// @return       a pointer to the PatchArray or NULL if the mesh is not uniformly
    ///               subdivided or the level cannot be found.
    ///
    PatchArray const * GetUniformPatchArray(int level=0) const;

    /// \brief Returns a pointer to the vertex indices of uniformly subdivided faces
    ///
    /// In uniform mode the PatchTablesFactory can be set to generate either a
    /// patch array containing the faces at the highest level of subdivision, or
    /// a range of arrays, corresponding to multiple successive levels of subdivision.
    ///
    /// Note : level '0' is not the coarse mesh. Currently there is no path in the
    /// factories to convert the coarse mesh to PatchTables.
    ///
    /// @param level  the level of subdivision of the faces (returns the highest
    ///               level by default)
    ///
    /// @return       a pointer to the first vertex index or NULL if the mesh
    ///               is not uniformly subdivided or the level cannot be found.
    ///
    Index const * GetUniformFaceVertices(int level=0) const;

    /// \brief Returns the number of faces in a uniformly subdivided mesh at a given level
    ///
    /// In uniform mode the PatchTablesFactory can be set to generate either a
    /// patch array containing the faces at the highest level of subdivision, or
    /// a range of arrays, corresponding to multiple successive levels of subdivision.
    ///
    /// Note : level '0' is not the coarse mesh. Currently there is no path in the
    /// factories to convert the coarse mesh to PatchTables.
    ///
    /// @param level  the level of subdivision of the faces (returns the highest
    ///               level by default)
    ///
    /// @return       the number of faces in the mesh given the subdivision level
    ///               or -1 if the mesh is not uniform or the level is incorrect.
    ///
    int GetNumUniformFaces(int level=0) const;

    /// \brief Returns a vertex valence table used by Gregory patches
    VertexValenceTable const & GetVertexValenceTable() const { return _vertexValenceTable; }

    /// \brief Returns a quad offsets table used by Gregory patches
    QuadOffsetTable const & GetQuadOffsetTable() const { return _quadOffsetTable; }

    /// \brief Returns a PatchParamTable for each type of patch
    PatchParamTable const & GetPatchParamTable() const { return _paramTable; }

    /// \brief Returns a sharpness index table for each type of patch (if exists)
    std::vector<int> const &GetSharpnessIndexTable() const { return _sharpnessIndexTable; }

    /// \brief Returns sharpness values (if exists)
    std::vector<float> const &GetSharpnessValues() const { return _sharpnessValues; }

    /// \brief Number of control vertices of Regular Patches in table.
    static short GetRegularPatchSize() { return 16; }

    /// \brief Number of control vertices of Boundary Patches in table.
    static short GetBoundaryPatchSize() { return 12; }

    /// \brief Number of control vertices of Boundary Patches in table.
    static short GetCornerPatchSize() { return 9; }

    /// \brief Number of control vertices of Gregory (and Gregory Boundary) Patches in table.
    static short GetGregoryPatchSize() { return 4; }

    /// \brief Returns the total number of patches stored in the tables
    int GetNumPatchesTotal() const;

    /// \brief Returns the total number of control vertex indices in the tables
    int GetNumControlVerticesTotal() const;

    /// \brief Returns max vertex valence
    int GetMaxValence() const { return _maxValence; }

    /// \brief True if the patches are of feature adaptive types
    bool IsFeatureAdaptive() const;

    /// \brief Returns the total number of vertices in the mesh across across all depths
    int GetNumPtexFaces() const { return _numPtexFaces; }

    /// \brief Face-varying patch vertex indices tables
    ///
    /// FVarPatchTables contain the topology for face-varying primvar data
    /// channels. The patch ordering matches that of PatchTables PatchArrays.
    ///
    class FVarPatchTables {

    public:

        /// \brief Returns the number of face-varying primvar channels
        int GetNumChannels() const {
            return (int)_channels.size();
        }

        /// \brief Returns the face-varying patches vertex indices
        ///
        /// @param channel  Then face-varying primvar channel index
        ///
        std::vector<Index> const & GetPatchVertices(int channel) const {
            return _channels[channel].patchVertIndices;
        }

    private:
        friend class PatchTablesFactory;

        struct Channel {
            friend class PatchTablesFactory;

            std::vector<Index> patchVertIndices; // face-varying vertex indices
        };

        std::vector<Channel> _channels; // face-varying primvar channels
    };

    /// \brief Returns the face-varying patches
    FVarPatchTables const * GetFVarPatchTables() const { return _fvarPatchTables; }

    /// \brief Public constructor
    ///
    /// @param patchArrays       Vector of descriptors and ranges for arrays of patches
    ///
    /// @param patches           Indices of the control vertices of the patches
    ///
    /// @param vertexValences    Vertex valance table
    ///
    /// @param quadOffsets       Quad offset table
    ///
    /// @param fvarPatchTables   Indices of the face-varying control vertices of the patches
    ///
    /// @param patchParams       Local patch parameterization
    ///
    /// @param maxValence        Highest vertex valence allowed in the mesh
    ///
    PatchTables(PatchArrayVector const & patchArrays,
                PTable const & patches,
                VertexValenceTable const * vertexValences,
                QuadOffsetTable const * quadOffsets,
                PatchParamTable const * patchParams,
                FVarPatchTables const * fvarPatchTables,
                int maxValence);

    /// \brief Destructor
    ~PatchTables() { delete _fvarPatchTables; }

public:

    //
    // Interpolation methods
    //

    /// \brief Interpolate the (s,t) parametric location of a *bilinear* patch
    ///
    /// \note This method can only be used on uniform PatchTables of quads (see
    ///       IsFeatureAdaptive() method)
    ///
    /// @param handle  A patch handle indentifying the sub-patch containing the
    ///                (s,t) location
    ///
    /// @param s       Patch coordinate (in coarse face normalized space)
    ///
    /// @param t       Patch coordinate (in coarse face normalized space)
    ///
    /// @param src     Source primvar buffer (control vertices data)
    ///
    /// @param dst     Destination primvar buffer (limit surface data)
    ///
    template <class T, class U> void Interpolate(PatchHandle const & handle,
        float s, float t, T const & src, U * dst) const;

    /// \brief Interpolate the (s,t) parametric location of a bilinear (quad)
    /// patch
    ///
    template <class T, class U> static void
    InterpolateBilinear(Index const * cvs, float s, float t,
        T const & src, U * dst);

    /// \brief Interpolate the (s,t) parametric location of a regular bicubic
    ///        patch
    ///
    /// @param cvs     Array of 16 control vertex indices
    ///
    /// @param Q       Array of 16 bicubic weights for the control vertices
    ///
    /// @param Qd1     Array of 16 bicubic 's' tangent weights for the control
    ///                vertices
    ///
    /// @param Qd2     Array of 16 bicubic 't' tangent weights for the control
    ///                vertices
    ///
    /// @param src     Source primvar buffer (control vertices data)
    ///
    /// @param dst     Destination primvar buffer (limit surface data)
    ///
    template <class T, class U> static void
    InterpolateRegularPatch(Index const * cvs,
        float const * Q, float const *Qd1, float const *Qd2, T const & src, U * dst);

    /// \brief Interpolate the (s,t) parametric location of a boundary bicubic
    ///        patch
    ///
    /// @param cvs     Array of 12 control vertex indices
    ///
    /// @param Q       Array of 12 bicubic weights for the control vertices
    ///
    /// @param Qd1     Array of 12 bicubic 's' tangent weights for the control
    ///                vertices
    ///
    /// @param Qd2     Array of 12 bicubic 't' tangent weights for the control
    ///                vertices
    ///
    /// @param src     Source primvar buffer (control vertices data)
    ///
    /// @param dst     Destination primvar buffer (limit surface data)
    ///
    template <class T, class U> static void
    InterpolateBoundaryPatch(Index const * cvs,
        float const * Q, float const *Qd1, float const *Qd2, T const & src, U * dst);

    /// \brief Interpolate the (s,t) parametric location of a corner bicubic
    ///        patch
    ///
    /// @param cvs     Array of 9 control vertex indices
    ///
    /// @param Q       Array of 9 bicubic weights for the control vertices
    ///
    /// @param Qd1     Array of 9 bicubic 's' tangent weights for the control
    ///                vertices
    ///
    /// @param Qd2     Array of 9 bicubic 't' tangent weights for the control
    ///                vertices
    ///
    /// @param src     Source primvar buffer (control vertices data)
    ///
    /// @param dst     Destination primvar buffer (limit surface data)
    ///
    template <class T, class U> static void
    InterpolateCornerPatch(Index const * cvs,
        float const * Q, float const *Qd1, float const *Qd2, T const & src, U * dst);

    /// \brief Interpolate the (s,t) parametric location of a *bicubic* patch
    ///
    /// \note This method can only be used on feature adaptive PatchTables (ie.
    ///       IsFeatureAdaptive() is false)
    ///
    /// @param handle  A patch handle indentifying the sub-patch containing the
    ///                (s,t) location
    ///
    /// @param s       Patch coordinate (in coarse face normalized space)
    ///
    /// @param t       Patch coordinate (in coarse face normalized space)
    ///
    /// @param src     Source primvar buffer (control vertices data)
    ///
    /// @param dst     Destination primvar buffer (limit surface data)
    ///
    template <class T, class U> void Limit(PatchHandle const & handle,
        float s, float t, T const & src, U * dst) const;

private:

    friend class PatchTablesFactory;

    // Returns bi-cubic interpolation coefficients for a given (u,v) location
    // on a b-spline patch
    static void getBSplineWeightsAtUV(PatchParam::BitField bits, float s, float t,
        float point[16], float deriv1[16], float deriv2[16]);

private:

    // Returns the array of patches of type "desc", or NULL if there aren't any in the primitive
    PatchArray * findPatchArray( Descriptor desc );

    static DescriptorVector const & getBilinearDescriptors();
    static DescriptorVector const & getAdaptiveCatmarkDescriptors();
    static DescriptorVector const & getAdaptiveLoopDescriptors();

    // Private constructor
    PatchTables( int maxvalence ) : _fvarPatchTables(0), _maxValence(maxvalence) { }

    PatchArrayVector     _patchArrays;        // Vector of descriptors for arrays of patches

    PTable               _patches;            // Indices of the control vertices of the patches

    VertexValenceTable   _vertexValenceTable; // Vertex valence table (for Gregory patches)

    QuadOffsetTable      _quadOffsetTable;    // Quad offsets table (for Gregory patches)

    PatchParamTable      _paramTable;         // PatchParam bitfields (one per patch)

    FVarPatchTables const * _fvarPatchTables; // sparse face-varying patch table (one per patch)

    std::vector<int>     _sharpnessIndexTable;// Indices of single-crease sharpness (one per patch)

    std::vector<float>   _sharpnessValues;    // Sharpness values.

    // highest vertex valence allowed in the mesh (used for Gregory
    // vertexValance & quadOffset tables)
    int _maxValence;

    // number of total ptex faces in quads or triangles(loop)
    int _numPtexFaces;

};

// Returns the number of control vertices expected for a patch of this type
inline short
PatchTables::Descriptor::GetNumControlVertices( PatchTables::Type type ) {
    switch (type) {
        case REGULAR           : return PatchTables::GetRegularPatchSize();
        case SINGLE_CREASE     : return PatchTables::GetRegularPatchSize();
        case QUADS             : return 4;
        case GREGORY           :
        case GREGORY_BOUNDARY  : return PatchTables::GetGregoryPatchSize();
        case BOUNDARY          : return PatchTables::GetBoundaryPatchSize();
        case CORNER            : return PatchTables::GetCornerPatchSize();
        case TRIANGLES         : return 3;
        case LINES             : return 2;
        case POINTS            : return 1;
        default : return -1;
    }
}

// Returns the number of face-varying control vertices expected for a patch of this type
inline short
PatchTables::Descriptor::GetNumFVarControlVertices( PatchTables::Type type ) {
    switch (type) {
        case REGULAR           : // We only support bilinear interpolation for now,
        case SINGLE_CREASE     :
        case QUADS             : // so all these patches only carry 4 CVs.
        case GREGORY           :
        case GREGORY_BOUNDARY  :
        case BOUNDARY          :
        case CORNER            : return 4;
        case TRIANGLES         : return 3;
        case LINES             : return 2;
        case POINTS            : return 1;
        default : return -1;
    }
}

// Allows ordering of patches by type
inline bool
PatchTables::Descriptor::operator < ( Descriptor const other ) const {
    return _pattern < other._pattern or ((_pattern == other._pattern) and
          (_type < other._type or ((_type == other._type) and
          (_rotation < other._rotation))));
}

// True if the descriptors are identical
inline bool
PatchTables::Descriptor::operator == ( Descriptor const other ) const {
    return     _pattern == other._pattern    and
                  _type == other._type       and
              _rotation == other._rotation;
}

template <class T, class U>
inline void
PatchTables::InterpolateBilinear(Index const * cvs, float s, float t,
    T const & src, U * dst) {

    float os = 1.0f - s,
          ot = 1.0f - t,
            Q[4] = { os*ot,  s*ot, s*t, os*t },
          dQ1[4] = { t-1.0f,   ot,   t,   -t },
          dQ2[4] = { s-1.0f,   -s,   s,   os };

    for (int k=0; k<4; ++k) {
        dst->AddWithWeight(src[cvs[k]], Q[k], dQ1[k], dQ2[k]);
    }
}


template <class T, class U>
inline void
PatchTables::InterpolateRegularPatch(Index const * cvs,
    float const * Q, float const *Qd1, float const *Qd2,
        T const & src, U * dst) {

    //
    //  v0 -- v1 -- v2 -- v3
    //   |.....|.....|.....|
    //   |.....|.....|.....|
    //  v4 -- v5 -- v6 -- v7
    //   |.....|.....|.....|
    //   |.....|.....|.....|
    //  v8 -- v9 -- v10-- v11
    //   |.....|.....|.....|
    //   |.....|.....|.....|
    //  v12-- v13-- v14-- v15
    //
    for (int k=0; k<16; ++k) {
        dst->AddWithWeight(src[cvs[k]], Q[k], Qd1[k], Qd2[k]);
    }
}

template <class T, class U>
inline void
PatchTables::InterpolateBoundaryPatch(Index const * cvs,
    float const * Q, float const *Qd1, float const *Qd2,
        T const & src, U * dst) {

    // mirror the missing vertices (M)
    //
    //  M0 -- M1 -- M2 -- M3 (corner)
    //   |     |     |     |
    //   |     |     |     |
    //  v0 -- v1 -- v2 -- v3    M : mirrored
    //   |.....|.....|.....|
    //   |.....|.....|.....|
    //  v4 -- v5 -- v6 -- v7    v : original Cv
    //   |.....|.....|.....|
    //   |.....|.....|.....|
    //  v8 -- v9 -- v10-- v11
    //
    for (int k=0; k<4; ++k) { // M0 - M3
        dst->AddWithWeight(src[cvs[k]],    2.0f*Q[k],  2.0f*Qd1[k],  2.0f*Qd2[k]);
        dst->AddWithWeight(src[cvs[k+4]], -1.0f*Q[k], -1.0f*Qd1[k], -1.0f*Qd2[k]);
    }
    for (int k=0; k<12; ++k) {
        dst->AddWithWeight(src[cvs[k]], Q[k+4], Qd1[k+4], Qd2[k+4]);
    }
}

template <class T, class U>
inline void
PatchTables::InterpolateCornerPatch(Index const * cvs,
    float const * Q, float const *Qd1, float const *Qd2,
        T const & src, U * dst) {

    // mirror the missing vertices (M)
    //
    //  M0 -- M1 -- M2 -- M3 (corner)
    //   |     |     |     |
    //   |     |     |     |
    //  v0 -- v1 -- v2 -- M4    M : mirrored
    //   |.....|.....|     |
    //   |.....|.....|     |
    //  v3.--.v4.--.v5 -- M5    v : original Cv
    //   |.....|.....|     |
    //   |.....|.....|     |
    //  v6 -- v7 -- v8 -- M6
    //
    for (int k=0; k<3; ++k) { // M0 - M2
        dst->AddWithWeight(src[cvs[k  ]],  2.0f*Q[k],  2.0f*Qd1[k],  2.0f*Qd2[k]);
        dst->AddWithWeight(src[cvs[k+3]], -1.0f*Q[k], -1.0f*Qd1[k], -1.0f*Qd2[k]);
    }
    for (int k=0; k<3; ++k) { // M4 - M6
        int idx = (k+1)*4 + 3;
        dst->AddWithWeight(src[cvs[k*3+2]],  2.0f*Q[idx],  2.0f*Qd1[idx],  2.0f*Qd2[idx]);
        dst->AddWithWeight(src[cvs[k*3+1]], -1.0f*Q[idx], -1.0f*Qd1[idx], -1.0f*Qd2[idx]);
    }
    // M3 = -2.v1 + 4.v2 + v4 - 2.v5
    dst->AddWithWeight(src[cvs[1]], -2.0f*Q[3], -2.0f*Qd1[3], -2.0f*Qd2[3]);
    dst->AddWithWeight(src[cvs[2]],  4.0f*Q[3],  4.0f*Qd1[3],  4.0f*Qd2[3]);
    dst->AddWithWeight(src[cvs[4]],  1.0f*Q[3],  1.0f*Qd1[3],  1.0f*Qd2[3]);
    dst->AddWithWeight(src[cvs[5]], -2.0f*Q[3], -2.0f*Qd1[3], -2.0f*Qd2[3]);
    for (int y=0; y<3; ++y) { // v0 - v8
        for (int x=0; x<3; ++x) {
            int idx = y*4+x+4;
            dst->AddWithWeight(src[cvs[y*3+x]], Q[idx], Qd1[idx], Qd2[idx]);
        }
    }
}

// Interpolates the limit position of a parametric location on a patch
template <class T, class U>
inline void
PatchTables::Interpolate(PatchHandle const & handle, float s, float t,
    T const & src, U * dst) const {

    assert(dst and (not IsFeatureAdaptive()));

    PatchTables::PatchArray const & parray =
        _patchArrays[handle.patchArrayIdx];

    Index const * cvs =
        &_patches[parray.GetVertIndex() + handle.vertexOffset];

    PatchParam::BitField const & bits =
        _paramTable[handle.patchIdx].bitField;

    bits.Normalize(s,t);

    Type ptype = parray.GetDescriptor().GetType();
    assert(ptype==QUADS);

    dst->Clear();

    InterpolateBilinear(cvs, s, t, src, dst);
}

// Interpolates the limit position of a parametric location on a patch
template <class T, class U>
inline void
PatchTables::Limit(PatchHandle const & handle, float s, float t,
    T const & src, U * dst) const {

    assert(dst and IsFeatureAdaptive());

    PatchTables::PatchArray const & parray =
        _patchArrays[handle.patchArrayIdx];

    Index const * cvs =
        &_patches[parray.GetVertIndex() + handle.vertexOffset];

    PatchParam::BitField const & bits =
        _paramTable[handle.patchIdx].bitField;

    bits.Normalize(s,t);

    Type ptype = parray.GetDescriptor().GetType();

    if (ptype>=REGULAR and ptype<=CORNER) {

        float Q[16], Qd1[16], Qd2[16];

        getBSplineWeightsAtUV(bits, s, t, Q, Qd1, Qd2);

        float scale = float(1 << bits.GetDepth());
        for (int k=0; k<16; ++k) {
            Qd1[k] *= scale;
            Qd2[k] *= scale;
        }

        dst->Clear();

        switch (ptype) {
            case REGULAR:
                InterpolateRegularPatch(cvs, Q, Qd1, Qd2, src, dst);
                break;
            case SINGLE_CREASE:
                // TODO: implement InterpolateSingleCreasePatch().
                //InterpolateRegularPatch(cvs, Q, Qd1, Qd2, src, dst);
                break;
            case BOUNDARY:
                InterpolateBoundaryPatch(cvs, Q, Qd1, Qd2, src, dst);
                break;
            case CORNER:
                InterpolateCornerPatch(cvs, Q, Qd1, Qd2, src, dst);
                break;
            default:
                assert(0);
        }
    } else if (ptype>=GREGORY and ptype<=GREGORY_BOUNDARY) {

    } else {
        assert(0);
    }
}

} // end namespace Far

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* FAR_PATCH_TABLES */
