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
#include "../far/types.h"

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
    typedef std::vector<unsigned int>  PTable;
    typedef std::vector<int>           VertexValenceTable;
    typedef std::vector<unsigned int>  QuadOffsetTable;
    typedef std::vector<PatchParam> PatchParamTable;

    enum Type {
        NON_PATCH = 0,     ///< undefined

        POINTS,            ///< points  (useful for cage drawing)
        LINES,             ///< lines   (useful for cage drawing)

        QUADS,             ///< bilinear quads-only patches
        TRIANGLES,         ///< bilinear triangles-only mesh

        LOOP,              ///< Loop patch  (unsupported)

        REGULAR,           ///< feature-adaptive bicubic patches
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
    /// An iterator class is provided as a convenience to enumerate over the set
    /// of valid feature adaptive patch descriptors.
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

        /// Returns a vector of all the legal patch descriptors
        static inline std::vector<Descriptor> const & GetAllValidDescriptors();

        /// \brief Allows ordering of patches by type
        inline bool operator < ( Descriptor const other ) const;

        /// \brief True if the descriptors are identical
        inline bool operator == ( Descriptor const other ) const;

        /// \brief Descriptor Iterator
        /// Iterates through the patches in the following preset order
        ///
        /// ANY order:
        ///        POINTS
        ///        LINES
        ///        QUADS
        ///        TRIANGLES
        ///        LOOP
        ///
        /// FEATURE_ADAPTIVE_CATMARK order:
        ///
        ///       NON_TRANSITION ( REGULAR
        ///                         BOUNDARY
        ///                         CORNER
        ///                         GREGORY
        ///                         GREGORY_BOUNDARY )
        ///
        ///        PATTERN0 ( REGULAR
        ///                   BOUNDARY ROT0 ROT1 ROT2 ROT3
        ///                   CORNER   ROT0 ROT1 ROT2 ROT3 )
        ///
        ///        PATTERN1 ( REGULAR
        ///                   BOUNDARY ROT0 ROT1 ROT2 ROT3
        ///                   CORNER   ROT0 ROT1 ROT2 ROT3 )
        ///        ...
        ///
        ///        NON_TRANSITION NON_PATCH ROT0 (end)
        ///
        class iterator;

        enum PrimType {
            ANY,
            FEATURE_ADAPTIVE_CATMARK
        };

        /// \brief Returns a patch type iterator
        /// @param type       if type=ANY then the iterater points to type POINTS
        ///                   if type=FEATURE_ADAPTIVE_CATMARK then the iterator
        ///                   points to type NON_TRANSITION REGULAR
        static iterator begin(PrimType type);

        /// \brief Returns an iterator to the end of the list of patch types (NON_PATCH)
        static iterator end();

    private:
        friend class PatchTablesFactory;
        friend class iterator;

        unsigned int  _type:4;
        unsigned int  _pattern:3;
        unsigned int  _rotation:2;
    };




    /// \brief Describes an array of patches of the same type
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
        PatchArray( Descriptor desc, unsigned int vertIndex, unsigned int patchIndex, unsigned int npatches, unsigned int quadOffsetIndex ) :
            _desc(desc), _range(vertIndex, patchIndex, npatches, quadOffsetIndex) { }

        /// Returns a patch descriptor defining the type of patches in the array
        Descriptor GetDescriptor() const {
            return _desc;
        }

        /// \brief Describes the range of patches in a PatchArray
        struct ArrayRange {

            /// \brief Constructor
            ///
            /// @param ivertIndex        absolute index to the first control vertex
            ///                          of the first patch in the PTable
            ///
            /// @param ipatchIndex       absolute index of the first patch in the
            ///                          array
            ///
            /// @param inpatches         number of patches in the array
            ///
            /// @param iquadOffsetIndex  absolute index of the first quad offset
            ///                          entry
            ///
            ArrayRange( unsigned int ivertIndex, unsigned int ipatchIndex, unsigned int inpatches, unsigned int iquadOffsetIndex ) :
                vertIndex(ivertIndex), patchIndex(ipatchIndex), npatches(inpatches), quadOffsetIndex(iquadOffsetIndex) { }

            unsigned int vertIndex,       // absolute index to the first control vertex of the first patch in the PTable
                         patchIndex,      // absolute index of the first patch in the array
                         npatches,        // number of patches in the array
                         quadOffsetIndex; // absolute index of the first quad offset entry
        };

        /// \brief Returns a array range struct
        ArrayRange const & GetArrayRange() const {
            return _range;
        }

        /// \brief Returns the index of the first control vertex of the first patch
        /// of this array in the global PTable
        unsigned int GetVertIndex() const {
            return _range.vertIndex;
        }

        /// \brief Returns the global index of the first patch in this array (Used to
        /// access param / fvar table data)
        unsigned int GetPatchIndex() const {
            return _range.patchIndex;
        }

        /// \brief Returns the number of patches in the array
        unsigned int GetNumPatches() const {
            return _range.npatches;
        }

        /// \brief Returns the index to the first entry in the QuadOffsetTable
        unsigned int GetQuadOffsetIndex() const {
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
        unsigned int patchArrayIdx,  // OsdPatchArray containing the patch
                     patchIdx,       // Absolute index of the patch
                     vertexOffset;   // Offset to the first CV of the patch
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
    PatchArray const * GetPatchArray(int level=0) const;

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
    unsigned int const * GetFaceVertices(int level=0) const;

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
    int GetNumFaces(int level=0) const;

    /// \brief Returns a vertex valence table used by Gregory patches
    VertexValenceTable const & GetVertexValenceTable() const { return _vertexValenceTable; }

    /// \brief Returns a quad offsets table used by Gregory patches
    QuadOffsetTable const & GetQuadOffsetTable() const { return _quadOffsetTable; }

    /// \brief Returns a PatchParamTable for each type of patch
    PatchParamTable const & GetPatchParamTable() const { return _paramTable; }

    /// \brief Ringsize of Regular Patches in table.
    static short GetRegularPatchRingsize() { return 16; }

    /// \brief Ringsize of Boundary Patches in table.
    static short GetBoundaryPatchRingsize() { return 12; }

    /// \brief Ringsize of Boundary Patches in table.
    static short GetCornerPatchRingsize() { return 9; }

    /// \brief Ringsize of Gregory (and Gregory Boundary) Patches in table.
    static short GetGregoryPatchRingsize() { return 4; }

    /// \brief Returns the total number of patches stored in the tables
    int GetNumPatches() const;

    /// \brief Returns the total number of control vertex indices in the tables
    int GetNumControlVertices() const;

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
        std::vector<unsigned int> const & GetPatchVertices(int channel) const {
            return _channels[channel].patchVertIndices;
        }

    private:
        friend class PatchTablesFactory;

        struct Channel {
            friend class PatchTablesFactory;

            std::vector<unsigned int> patchVertIndices; // face-varying vertex indices
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
    InterpolateBilinear(unsigned int const * cvs, float s, float t,
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
    InterpolateRegularPatch(unsigned int const * cvs,
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
    InterpolateBoundaryPatch(unsigned int const * cvs,
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
    InterpolateCornerPatch(unsigned int const * cvs,
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
    inline PatchArray * findPatchArray( Descriptor desc );

    // Private constructor
    PatchTables( int maxvalence ) : _fvarPatchTables(0), _maxValence(maxvalence) { }

    PatchArrayVector     _patchArrays;        // Vector of descriptors for arrays of patches

    PTable               _patches;            // Indices of the control vertices of the patches

    VertexValenceTable   _vertexValenceTable; // vertex valence table (for Gregory patches)

    QuadOffsetTable      _quadOffsetTable;    // quad offsets table (for Gregory patches)

    PatchParamTable      _paramTable;

    FVarPatchTables const * _fvarPatchTables; // sparse face-varying patch table

    // highest vertex valence allowed in the mesh (used for Gregory
    // vertexValance & quadOffset tables)
    int _maxValence;

    // number of total ptex faces in quads or triangles(loop)
    int _numPtexFaces;

};

/// \brief Descriptor iterator class
class PatchTables::Descriptor::iterator {
    public:

        /// Constructor
        iterator() : _pos(-1) {}

        /// Copy Constructor
        iterator(Descriptor desc);

        /// Iteration increment operator
        iterator & operator ++ ();

        /// True of the two descriptors are identical
        bool operator == ( iterator const & other ) const {
            return (_pos==other._pos);
        }

        /// True if the two descriptors are different
        bool operator != ( iterator const & other ) const {
            return not (*this==other);
        }

        /// Dereferencing operator
        Descriptor const * operator -> () const {
            return  getValue();
        }

        /// Dereferencing operator
        Descriptor const & operator * () const {
            return *getValue();
        }

    private:
        inline Descriptor const * getValue() const;

        int _pos;
};

// Iterator constructor
inline PatchTables::Descriptor::iterator::iterator(Descriptor desc) {

    _pos = -1;
    std::vector<Descriptor> const & descs =
        Descriptor::GetAllValidDescriptors();

    for (int i=0; i<(int)descs.size(); ++i) {
        if (descs[i] == desc) {
            _pos = i;
            break;
        }
    }
}

// Iteration increment operator
inline PatchTables::Descriptor::iterator &
PatchTables::Descriptor::iterator::operator ++ () {

    if (++_pos>=(int)Descriptor::GetAllValidDescriptors().size()) {
        _pos = -1;
    }
    return *this;
}

inline PatchTables::Descriptor const *
PatchTables::Descriptor::iterator::getValue() const {

    static Descriptor _nonpatch;

    std::vector<Descriptor> const & descs =
        Descriptor::GetAllValidDescriptors();

    if (_pos>=0 and _pos<(int)descs.size()) {
        return &descs[_pos];
    }

    return &_nonpatch;
}

inline std::vector<PatchTables::Descriptor> const &
PatchTables::Descriptor::GetAllValidDescriptors() {

    static std::vector<Descriptor> _descriptors;

    if (_descriptors.empty()) {
        _descriptors.reserve(55);

        // non-patch primitives
        for (int i=POINTS; i<=LOOP; ++i) {
            _descriptors.push_back( Descriptor(i, NON_TRANSITION, 0) );
        }

        // non-transition patches
        for (int i=REGULAR; i<=GREGORY_BOUNDARY; ++i) {
            _descriptors.push_back( Descriptor(i, NON_TRANSITION, 0) );
        }

        // transition patches
        for (int i=PATTERN0; i<=PATTERN4; ++i) {

            _descriptors.push_back( Descriptor(REGULAR, i, 0) );

            // 4 rotations for boundary & corner patches
            for (int j=0; j<4; ++j) {
                _descriptors.push_back( Descriptor(BOUNDARY, i, j) );
            }

            for (int j=0; j<4; ++j) {
                _descriptors.push_back( Descriptor(CORNER, i, j) );
            }
        }
    }

    return _descriptors;
}

// Returns an iterator to the first type of patch (REGULAR NON_TRANSITION ROT0)
inline PatchTables::Descriptor::iterator
PatchTables::Descriptor::begin(PrimType type) {
    switch (type) {
        case ANY:
            return iterator( Descriptor(POINTS, NON_TRANSITION, 0) );
        case FEATURE_ADAPTIVE_CATMARK:
            return iterator( Descriptor(REGULAR, NON_TRANSITION, 0) );
        default:
            return iterator( Descriptor() );
    }
}

// Returns an iterator to the end of the list of patch types (NON_PATCH)
inline PatchTables::Descriptor::iterator
PatchTables::Descriptor::end() {
    return iterator( Descriptor() );
}

// Constructor
inline
PatchTables::PatchTables(PatchArrayVector const & patchArrays,
                               PTable const & patches,
                               VertexValenceTable const * vertexValences,
                               QuadOffsetTable const * quadOffsets,
                               PatchParamTable const * patchParams,
                               FVarPatchTables const * fvarPatchTables,
                               int maxValence) :
    _patchArrays(patchArrays),
    _patches(patches),
    _fvarPatchTables(fvarPatchTables),
    _maxValence(maxValence),
    _numPtexFaces(0) {

    // copy other tables if exist
    if (vertexValences)
        _vertexValenceTable = *vertexValences;
    if (quadOffsets)
        _quadOffsetTable = *quadOffsets;
    if (patchParams)
        _paramTable = *patchParams;
}

inline bool
PatchTables::IsFeatureAdaptive() const {

    // the vertex valence table is only used by Gregory patches, so the PatchTables
    // contain feature adaptive patches if this is not empty.
    if (not _vertexValenceTable.empty())
        return true;

    PatchArrayVector const & parrays = GetPatchArrayVector();

    // otherwise, we have to check each patch array
    for (int i=0; i<(int)parrays.size(); ++i) {

        if (parrays[i].GetDescriptor().GetType() >= REGULAR and
            parrays[i].GetDescriptor().GetType() <= GREGORY_BOUNDARY)
            return true;

    }
    return false;
}

// Returns the number of control vertices expected for a patch of this type
inline short
PatchTables::Descriptor::GetNumControlVertices( PatchTables::Type type ) {
    switch (type) {
        case REGULAR           : return PatchTables::GetRegularPatchRingsize();
        case QUADS             : return 4;
        case GREGORY           :
        case GREGORY_BOUNDARY  : return PatchTables::GetGregoryPatchRingsize();
        case BOUNDARY          : return PatchTables::GetBoundaryPatchRingsize();
        case CORNER            : return PatchTables::GetCornerPatchRingsize();
        case TRIANGLES         : return 3;
        case LINES             : return 2;
        case POINTS            : return 1;
        default : return -1;
    }
}

// Returns the total number of control vertex indices in the tables
inline int
PatchTables::GetNumControlVertices() const {

    int result=0;
    for (int i=0; i<(int)_patchArrays.size(); ++i) {
        result += _patchArrays[i].GetDescriptor().GetNumControlVertices() *
                  _patchArrays[i].GetNumPatches();
    }

    return result;
}

// Returns the number of face-varying control vertices expected for a patch of this type
inline short
PatchTables::Descriptor::GetNumFVarControlVertices( PatchTables::Type type ) {
    switch (type) {
        case REGULAR           : // We only support bilinear interpolation for now,
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

// Returns a pointer to the PatchArry of uniformly subdivided faces at 'level'
inline PatchTables::PatchArray const *
PatchTables::GetPatchArray(int level) const {

    if (IsFeatureAdaptive())
        return NULL;

    PatchArrayVector const & parrays = GetPatchArrayVector();

    if (parrays.empty())
        return NULL;

    if (level < 1) {
        return &(*parrays.rbegin());
    } else if ((level-1) < (int)parrays.size() ) {
        return &parrays[level-1];
    }

    return NULL;
}

// Returns a pointer to the vertex indices of uniformly subdivided faces
inline unsigned int const *
PatchTables::GetFaceVertices(int level) const {

    PatchArray const * parray = GetPatchArray(level);

    if (parray) {
        return &GetPatchTable()[ parray->GetVertIndex() ];
    }
    return NULL;
}

// Returns the number of faces in a uniformly subdivided mesh at a given level
inline int
PatchTables::GetNumFaces(int level) const {

    PatchArray const * parray = GetPatchArray(level);

    if (parray) {
        return parray->GetNumPatches();
    }
    return -1;
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

// Returns a pointer to the array of patches matching the descriptor
inline PatchTables::PatchArray *
PatchTables::findPatchArray( PatchTables::Descriptor desc ) {

    for (int i=0; i<(int)_patchArrays.size(); ++i) {
        if (_patchArrays[i].GetDescriptor()==desc)
            return &_patchArrays[i];
    }
    return 0;
}

// Returns the total number of patches stored in the tables
inline int
PatchTables::GetNumPatches() const {
    // there is one PatchParam record for each patch in the mesh
    return (int)GetPatchParamTable().size();
}

template <class T, class U>
inline void
PatchTables::InterpolateBilinear(unsigned int const * cvs, float s, float t,
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
PatchTables::InterpolateRegularPatch(unsigned int const * cvs,
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
PatchTables::InterpolateBoundaryPatch(unsigned int const * cvs,
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
PatchTables::InterpolateCornerPatch(unsigned int const * cvs,
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

    unsigned int const * cvs =
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

    unsigned int const * cvs =
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
