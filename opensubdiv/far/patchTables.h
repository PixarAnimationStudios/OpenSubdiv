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

#ifndef FAR_PATCH_TABLES_H
#define FAR_PATCH_TABLES_H

#include "../version.h"

#include "../far/patchParam.h"

#include <cstdlib>
#include <cassert>
#include <vector>
#include <map>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

/// \brief Container for patch vertex indices tables
///
/// FarPatchTables contain the lists of vertices for each patch of an adaptive
/// mesh representation.
///
class FarPatchTables {

public:
    typedef std::vector<unsigned int>  PTable;
    typedef std::vector<int>           VertexValenceTable;
    typedef std::vector<unsigned int>  QuadOffsetTable;
    typedef std::vector<FarPatchParam> PatchParamTable;
    typedef std::vector<float>         FVarDataTable;

    enum Type {
        NON_PATCH = 0,   // undefined
 
        POINTS,          // points  (useful for cage drawing)
        LINES,           // lines   (useful for cage drawing)
 
        QUADS,           // bilinear quads-only patches
        TRIANGLES,       // bilinear triangles-only mesh
 
        LOOP,            // Loop patch  (unsupported)

        REGULAR,         // feature-adaptive bicubic patches
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
        PATTERN4,
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
        /// Default constructor.
        Descriptor() :
            _type(NON_PATCH), _pattern(NON_TRANSITION), _rotation(0) {}
            
        /// Constructor
        Descriptor(int type, int pattern, unsigned char rotation) :
            _type(type), _pattern(pattern), _rotation(rotation) { }

        /// Copy Constructor
        Descriptor( Descriptor const & d ) :
            _type(d.GetType()), _pattern(d.GetPattern()), _rotation(d.GetRotation()) { }
        
        /// Returns the type of the patch
        Type GetType() const {
            return (Type)_type;
        }
        
        /// Returns the transition pattern of the patch if any (5 types)
        TransitionPattern GetPattern() const {
            return (TransitionPattern)_pattern;
        }
        
        /// Returns the rotation of the patch (4 rotations)
        unsigned char GetRotation() const {
            return _rotation;
        }
                
        /// Returns the number of control vertices expected for a patch of the
        /// type described
        static short GetNumControlVertices( Type t );
        
        /// Returns the number of control vertices expected for a patch of the 
        /// type described
        short GetNumControlVertices() const {
            return GetNumControlVertices( this->GetType() );
        }
        
        /// Iterates through the patches in the following preset order
        ///
        /// NON_TRANSITION ( REGULAR 
        ///                  BOUNDARY
        ///                  CORNER
        ///                  GREGORY
        ///                  GREGORY_BOUNDARY )
        ///
        /// PATTERN0 ( REGULAR 
        ///            BOUNDARY ROT0 ROT1 ROT2 ROT3
        ///            CORNER   ROT0 ROT1 ROT2 ROT3 )
        ///
        /// PATTERN1 ( REGULAR 
        ///            BOUNDARY ROT0 ROT1 ROT2 ROT3
        ///            CORNER   ROT0 ROT1 ROT2 ROT3 )
        /// ...           
        ///
        /// NON_TRANSITION NON_PATCH ROT0 (end)
        ///
        Descriptor & operator ++ ();
        
        /// Allows ordering of patches by type
        bool operator < ( Descriptor const other ) const;

        /// True if the descriptors are identical
        bool operator == ( Descriptor const other ) const;
        
        /// Descriptor Iterator 
        class iterator;

        /// Returns an iterator to the first type of patch (REGULAR NON_TRANSITION ROT0)
        static iterator begin() {
            return iterator( Descriptor(REGULAR, NON_TRANSITION, 0) );
        }

        /// Returns an iterator to the end of the list of patch types (NON_PATCH)
        static iterator end() {
            return iterator( Descriptor() );
        }
        
    private:
        template <class T> friend class FarPatchTablesFactory;
        friend class iterator;
        
        unsigned int  _type:4;
        unsigned int  _pattern:3;
        unsigned int  _rotation:2;
    };


    /// \brief Descriptor iterator class 
    class Descriptor::iterator {
        public:
            /// Constructor
            iterator() {}

            /// Copy Constructor
            iterator(Descriptor desc) : pos(desc) { }
            
            /// Iteration increment operator
            iterator & operator ++ () { ++pos; return *this; }
            
            /// True of the two descriptors are identical
            bool operator == ( iterator const & other ) const { return (pos==other.pos); }

            /// True if the two descriptors are different
            bool operator != ( iterator const & other ) const { return not (*this==other); }
            
            /// Dereferencing operator
            Descriptor * operator -> () { return &pos; }
            
            /// Dereferencing operator
            Descriptor & operator * () { return pos; }

        private:
            Descriptor pos;
    };


    /// \brief Describes an array of patches of the same type
    class PatchArray {
    
    public:
        /// Constructor.
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
        
            /// Constructor
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
            ArrayRange( unsigned int vertIndex, unsigned int patchIndex, unsigned int npatches, unsigned int quadOffsetIndex ) :
                vertIndex(vertIndex), patchIndex(patchIndex), npatches(npatches), quadOffsetIndex(quadOffsetIndex) { }

            unsigned int vertIndex,       // absolute index to the first control vertex of the first patch in the PTable
                         patchIndex,      // absolute index of the first patch in the array
                         npatches,        // number of patches in the array
                         quadOffsetIndex; // absolute index of the first quad offset entry
        };

        /// Returns a array range struct
        ArrayRange const & GetArrayRange() const {
            return _range;
        }

        /// Returns the index of the first control vertex of the first patch 
        /// of this array in the global PTable
        unsigned int GetVertIndex() const { 
            return _range.vertIndex;
        }
        
        /// Returns the global index of the first patch in this array (Used to
        /// access param / fvar table data)
        unsigned int GetPatchIndex() const {
            return _range.patchIndex;
        }
        
        /// Returns the number of patches in the array
        unsigned int GetNumPatches() const {
            return _range.npatches;
        }

        unsigned int GetQuadOffsetIndex() const {
            return _range.quadOffsetIndex;
        }
    
    private:
        template <class T> friend class FarPatchTablesFactory;
        
        Descriptor _desc;   // type of patches in the array

        ArrayRange _range;  // index locators in the array
    };
    
    typedef std::vector<PatchArray> PatchArrayVector;


    /// Unique patch identifier within a PatchArrayVector
    struct PatchHandle {
    
        unsigned int array,        // OsdPatchArray containing the patch
                     vertexOffset, // Offset to the first CV of the patch
                     serialIndex;  // Serialized Index of the patch
    };


    /// \brief Maps sub-patches to coarse faces
    class PatchMap {
    
    public:
        // Constructor
        PatchMap( FarPatchTables const & patchTables );
        
        /// \brief Returns the number and list of patch indices for a given face.
        ///
        /// PatchMaps map coarse faces to their childrn feature adaptive patches. 
        /// Coarse faces are indexed using their ptex face ID to resolve parametric
        /// ambiguity on non-quad faces. Note : this "map" is actually a vector, so
        /// queries are O(1) order.
        ///
        /// @param faceid    the face index to search for
        ///
        /// @param npatches  the number of children patches found for the faceid
        ///
        /// @param patches   a set of pointers to the individual patch handles
        ///
        bool GetChildPatchesHandles( int faceid, int * npatches, PatchHandle const ** patches ) const;
        
    private:
        typedef std::multimap<unsigned int, PatchHandle> MultiMap;

        // Patch handle allowing location of individual patch data inside patch
        // arrays or in serialized form
        std::vector<PatchHandle> _handles;
        
        // offset to the first handle of the child patches for each coarse face
        std::vector<unsigned int> _offsets; 
    };

    /// Constructor
    ///
    /// @param patchArrays      Vector of descriptors and ranges for arrays of patches
    ///
    /// @param patches          Indices of the control vertices of the patches
    ///
    /// @param vertexValences   Vertex valance table
    ///
    /// @param quadOffsets      Quad offset table
    ///
    /// @param patchParams      Local patch parameterization
    ///
    /// @param fvarData         Face varying data table
    ///
    /// @param maxValence       Highest vertex valence allowed in the mesh
    ///
    FarPatchTables(PatchArrayVector const & patchArrays,
                   PTable const & patches,
                   VertexValenceTable const * vertexValences,
                   QuadOffsetTable const * quadOffsets,
                   PatchParamTable const * patchParams,
                   FVarDataTable const * fvarData,
                   int maxValence);

    /// Get the table of patch control vertices
    PTable const & GetPatchTable() const { return _patches; }

    /// Returns a pointer to the array of patches matching the descriptor
    PatchArray const * GetPatchArray( Descriptor desc ) const { 
        return const_cast<FarPatchTables *>(this)->findPatchArray( desc ); 
    }

    /// Returns all arrays of patches
    PatchArrayVector const & GetPatchArrayVector() const {
        return _patchArrays;
    }
    
    /// Returns a pointer to the vertex indices of uniformly subdivided faces
    ///
    /// @param level  the level of subdivision of the faces
    ///
    /// @return       a pointer to the first vertex index or NULL if the mesh
    ///               is not uniformly subdivided or the level cannot be found.
    ///
    unsigned int const * GetFaceVertices(int level) const;

    /// Returns the number of faces in a uniformly subdivided mesh at a given level
    ///
    /// @param level  the level of subdivision of the faces
    ///
    /// @return       the number of faces in the mesh given the subdivision level
    ///               or -1 if the mesh is not uniform or the level incorrect.
    ///
    int GetNumFaces(int level) const;
    
    /// Returns a vertex valence table used by Gregory patches
    VertexValenceTable const & GetVertexValenceTable() const { return _vertexValenceTable; }

    /// Returns a quad offsets table used by Gregory patches
    QuadOffsetTable const & GetQuadOffsetTable() const { return _quadOffsetTable; }

    /// Returns a PatchParamTable for each type of patch
    PatchParamTable const & GetPatchParamTable() const { return _paramTable; }

    /// Returns an FVarDataTable for each type of patch
    /// The data is stored as a run of totalFVarWidth floats per-vertex per-face
    /// e.g.: for UV data it has the structure of float[p][4][2] where 
    /// p=primitiveID and totalFVarWidth=2:
    ///      [ [ uv uv uv uv ] [ uv uv uv uv ] [ ... ] ]
    ///            prim 0           prim 1
    FVarDataTable const & GetFVarDataTable() const { return _fvarTable; }

    /// Ringsize of Regular Patches in table.
    static int GetRegularPatchRingsize() { return 16; }

    /// Ringsize of Boundary Patches in table.
    static int GetBoundaryPatchRingsize() { return 12; }

    /// Ringsize of Boundary Patches in table.
    static int GetCornerPatchRingsize() { return 9; }

    /// Ringsize of Gregory (and Gregory Boundary) Patches in table.
    static int GetGregoryPatchRingsize() { return 4; }

    /// Returns the total number of patches stored in the tables
    int GetNumPatches() const;
    
    /// Returns the total number of control vertex indices in the tables
    int GetNumControlVertices() const;

    /// Returns max vertex valence
    int GetMaxValence() const { return _maxValence; }
    
    /// True if the patches are of feature adaptive types
    bool IsFeatureAdaptive() const;
    
private:

    template <class T> friend class FarPatchTablesFactory;
    template <class T, class U> friend class FarMultiMeshFactory;

    // Returns the array of patches of type "desc", or NULL if there aren't any in the primitive
    PatchArray * findPatchArray( Descriptor desc );

    // Private constructor
    FarPatchTables( int maxvalence ) : _maxValence(maxvalence) { }

    
    PatchArrayVector    _patchArrays;        // Vector of descriptors for arrays of patches
    
    PTable              _patches;            // Indices of the control vertices of the patches

    VertexValenceTable  _vertexValenceTable; // vertex valence table (for Gregory patches)
    
    QuadOffsetTable     _quadOffsetTable;    // quad offsets table (for Gregory patches)
    
    PatchParamTable     _paramTable;

    FVarDataTable       _fvarTable;

    // highest vertex valence allowed in the mesh (used for Gregory 
    // vertexValance & quadOffset tables)
    int _maxValence;
};

// Constructor
inline
FarPatchTables::FarPatchTables(PatchArrayVector const & patchArrays,
                               PTable const & patches,
                               VertexValenceTable const * vertexValences,
                               QuadOffsetTable const * quadOffsets,
                               PatchParamTable const * patchParams,
                               FVarDataTable const * fvarData,
                               int maxValence) :
    _patchArrays(patchArrays),
    _patches(patches),
    _maxValence(maxValence) {

    // copy other tables if exist
    if (vertexValences)
        _vertexValenceTable = *vertexValences;
    if (quadOffsets)
        _quadOffsetTable = *quadOffsets;
    if (patchParams)
        _paramTable = *patchParams;
    if (fvarData)
        _fvarTable = *fvarData;
}

inline bool 
FarPatchTables::IsFeatureAdaptive() const { 
    return ((not _vertexValenceTable.empty()) and (not _quadOffsetTable.empty())); 
}
 
// Returns the number of control vertices expected for a patch of this type
inline short 
FarPatchTables::Descriptor::GetNumControlVertices( FarPatchTables::Type type ) {
    switch (type) {
        case REGULAR           : return FarPatchTables::GetRegularPatchRingsize();
        case QUADS             : return 4;
        case GREGORY           :
        case GREGORY_BOUNDARY  : return FarPatchTables::GetGregoryPatchRingsize();
        case BOUNDARY          : return FarPatchTables::GetBoundaryPatchRingsize();
        case CORNER            : return FarPatchTables::GetCornerPatchRingsize();
        case TRIANGLES         : return 3;
        case LINES             : return 2;
        case POINTS            : return 1;
        default : return -1;
    }
}

// Iterates in order through the patch types, patterns and rotation in a preset order
inline FarPatchTables::Descriptor & 
FarPatchTables::Descriptor::operator ++ () {

    if (GetPattern()==NON_TRANSITION) {
        if (GetType()==GREGORY_BOUNDARY) {
            _type=REGULAR;
            ++_pattern;
        } else
            ++_type;
    } else {

        switch (GetType()) {
            case REGULAR  : ++_type; 
                            _rotation=0; 
                            break;

            case BOUNDARY : if (GetRotation()==3) {
                                ++_type; 
                                _rotation=0;
                            } else {
                                ++_rotation;
                            }; break;

            case CORNER   : if (GetRotation()==3) {
                                  if (GetPattern()!=PATTERN4) {
                                      _type=REGULAR;
                                      _rotation=0;
                                      ++_pattern;
                                  } else {
                                      *this = Descriptor();
                                  }
                              } else {
                                  ++_rotation;
                              }; break;
            
            case NON_PATCH : break;
            
            default:
                assert(0);
        }
    }
    return *this;
}

// Constructor
inline
FarPatchTables::PatchMap::PatchMap( FarPatchTables const & patchTables ) {

    // Create a PatchHandle for each patch in the primitive

    int npatches = (int)patchTables.GetNumPatches();
    _handles.reserve(npatches);

    FarPatchTables::PatchArrayVector const & patchArrays =
        patchTables.GetPatchArrayVector();

    FarPatchTables::PatchParamTable const & paramTable =
        patchTables.GetPatchParamTable();
    assert( not paramTable.empty() );

    int nfaces =0;
    MultiMap mmap;

    for (int arrayid = 0; arrayid < (int)patchArrays.size(); ++arrayid) {

        FarPatchTables::PatchArray const & pa = patchArrays[arrayid];

         int ringsize = pa.GetDescriptor().GetNumControlVertices();

         for (unsigned int j=0; j < pa.GetNumPatches(); ++j) {

            int faceId = paramTable[pa.GetPatchIndex()+j].faceIndex;

            PatchHandle handle = { arrayid, j*ringsize, (unsigned int)mmap.size() };

            mmap.insert( std::pair<unsigned int, PatchHandle>(faceId, handle));

            nfaces = std::max(nfaces, faceId);
        }
    }
    ++nfaces;

    _handles.resize( mmap.size() );
    _offsets.reserve( nfaces );
    _offsets.push_back(0);

    // Serialize the multi-map

    unsigned int handlesIdx = 0, faceId=mmap.begin()->first;

    for (MultiMap::const_iterator it=mmap.begin(); it!=mmap.end(); ++it, ++handlesIdx) {

        assert(it->first >= faceId);

        if (it->first != faceId) {

            faceId = it->first;

            // position the offset marker to the new face                    
            _offsets.push_back( handlesIdx );
        }

        // copy the patch id into the table
        _handles[handlesIdx] = it->second;
    }
}

// Returns the number and list of patch indices for a given face.
inline bool 
FarPatchTables::PatchMap::GetChildPatchesHandles( int faceid, int * npatches, PatchHandle const ** patches ) const {

    if (_handles.empty() or _offsets.empty() or (faceid>=(int)_offsets.size()))
        return false;

    *npatches = (faceid==(int)_offsets.size()-1 ? 
        (unsigned int)_handles.size()-1 : _offsets[faceid+1]) - _offsets[faceid] + 1;

    *patches = &_handles[ _offsets[faceid] ];

    return true;
}

// Returns a pointer to the vertex indices of uniformly subdivided faces
inline unsigned int const * 
FarPatchTables::GetFaceVertices(int level) const {

    if (IsFeatureAdaptive())
        return NULL;
    
    PatchArrayVector const & parrays = GetPatchArrayVector();
    
    if ( (level-1) < (int)parrays.size() ) {
        return &GetPatchTable()[ parrays[level-1].GetVertIndex() ];
    }
    
    return NULL;
}

// Returns the number of faces in a uniformly subdivided mesh at a given level
inline int
FarPatchTables::GetNumFaces(int level) const {

    if (IsFeatureAdaptive())
        return -1;
    
    PatchArrayVector const & parrays = GetPatchArrayVector();
    
    if ( (level-1) < (int)parrays.size() ) {
        return parrays[level-1].GetNumPatches();
    }
    
    return -1;
}

// Allows ordering of patches by type
inline bool 
FarPatchTables::Descriptor::operator < ( Descriptor const other ) const {
    return _pattern < other._pattern or ((_pattern == other._pattern) and
          (_type < other._type or ((_type == other._type) and
          (_rotation < other._rotation))));
}

// True if the descriptors are identical
inline bool
FarPatchTables::Descriptor::operator == ( Descriptor const other ) const {
    return     _pattern == other._pattern    and
                  _type == other._type       and
              _rotation == other._rotation;
}

// Returns a pointer to the array of patches matching the descriptor
inline FarPatchTables::PatchArray *
FarPatchTables::findPatchArray( FarPatchTables::Descriptor desc ) {

    for (int i=0; i<(int)_patchArrays.size(); ++i) {
        if (_patchArrays[i].GetDescriptor()==desc)
            return &_patchArrays[i];
    }
    return 0;
}

// Returns the total number of patches stored in the tables
inline int
FarPatchTables::GetNumPatches() const {

    int result=0;
    for (int i=0; i<(int)_patchArrays.size(); ++i) {
        result += _patchArrays[i].GetNumPatches();
    }

    return result;
}

// Returns the total number of control vertex indices in the tables
inline int 
FarPatchTables::GetNumControlVertices() const {

    int result=0;
    for (int i=0; i<(int)_patchArrays.size(); ++i) {
        result += _patchArrays[i].GetDescriptor().GetNumControlVertices() * 
                  _patchArrays[i].GetNumPatches();
    }

    return result;
}



} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* FAR_PATCH_TABLES */
