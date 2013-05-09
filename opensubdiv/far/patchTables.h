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

#include <stdlib.h>
#include <vector>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

/// \brief Flattened ptex coordinates indexing system
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
struct FarPtexCoord {
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
/*
/// \brief Indices for multi-mesh patch arrays
// XXXX manuelk : we should probably derive FarMultiPatchTables for multi-meshes
struct FarPatchCount {
    int nonPatch;                 // reserved for uniform and loop
    int regular;
    int boundary;
    int corner;
    int gregory;
    int boundaryGregory;
    int transitionRegular[5];
    int transitionBoundary[5][4];
    int transitionCorner[5][4];

    /// Constructor.
    FarPatchCount() {
        nonPatch = regular = boundary = corner = gregory = boundaryGregory = 0;
        for (int i = 0; i < 5; ++i) {
            transitionRegular[i] = 0;
            for (int j = 0; j < 4; ++j) {
                transitionBoundary[i][j] = 0;
                transitionCorner[i][j] = 0;
            }
        }
    }

    /// Adds the indices from another patchTable.
    void Append(FarPatchCount const &p) {
        nonPatch += p.nonPatch;
        regular += p.regular;
        boundary += p.boundary;
        corner += p.corner;
        gregory += p.gregory;
        boundaryGregory += p.boundaryGregory;
        for (int i = 0; i < 5; ++i) {
            transitionRegular[i] += p.transitionRegular[i];
            for (int j = 0; j < 4; ++j) {
                transitionBoundary[i][j] += p.transitionBoundary[i][j];
                transitionCorner[i][j] += p.transitionCorner[i][j];
            }
        }
    }    
};

typedef std::vector<FarPatchCount> FarPatchCountVector;
*/
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
    typedef std::vector<FarPtexCoord>  PtexCoordinateTable;
    typedef std::vector<float>         FVarDataTable;

    enum Type {
        NON_PATCH = 0, // undefined
        
        QUADS,         // quads-only mesh
        TRIANGLES,     // triangles-only mesh
        POLYGONS,      // general polygon mesh
 
        LOOP,          // Loop patch  (unsupported)

        REGULAR,
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
    class Descriptor {
    
    public:
        /// Default constructor.
        Descriptor() :
            _type(NON_PATCH), _pattern(NON_TRANSITION), _rotation(0) { }
            
        /// Constructor
        Descriptor(int type, int pattern, unsigned char rotation) :
            _type((Type)type), _pattern((TransitionPattern)pattern), _rotation(rotation) { }

        /// Copy Constructor
        Descriptor( Descriptor const & d ) :
            _type(d.GetType()), _pattern(d.GetPattern()), _rotation(d.GetRotation()) { }
        
        /// Returns the type of the patch
        Type GetType() const {
            return _type;
        }
        
        /// Returns the transition pattern of the patch if any (5 types)
        TransitionPattern GetPattern() const {
            return _pattern;
        }
        
        /// Returns the rotation of the patch (4 rotations)
        unsigned char GetRotation() const {
            return _rotation;
        }
                
        /// Returns the number of control vertices expected for a patch of this type
        static short GetNumControlVertices( Type t );
        
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
        bool operator < ( Descriptor const other );

        /// True if the descriptors are identical
        bool operator == ( Descriptor const other );
        
        /// Descriptor Iterator 
        class iterator;

        static iterator begin() {
            return iterator( Descriptor(REGULAR, NON_TRANSITION, 0) );
        }

        static iterator end() {
            return iterator( Descriptor() );
        }
        
    private:
        template <class T> friend class FarPatchTablesFactory;
        friend class iterator;
        
        Type              _type:4;
        TransitionPattern _pattern:3;
        unsigned char     _rotation:2;
    };

    /// \brief Descriptor iterator class 
    class Descriptor::iterator {
        public:
            iterator() {}

            iterator(Descriptor desc) : pos(desc) { }
            
            iterator & operator ++ () { ++pos; return *this; }
            
            bool operator == ( iterator const & other ) { return (pos==other.pos); }

            bool operator != ( iterator const & other ) { return not (*this==other); }
            
            Descriptor * operator -> () { return &pos; }
            
            Descriptor & operator * () { return pos; }

        private:
            Descriptor pos;
    };

    /// \brief Describes an array of patches of the same type
    class PatchArray {
    
    public:
        PatchArray( Descriptor const & desc, unsigned int vertIndex, unsigned int patchIndex, unsigned int npatches ) :
            _desc(desc), _vertIndex(vertIndex), _patchIndex(patchIndex), _npatches(npatches) { }
    
        Descriptor GetDescriptor() const {
            return _desc;
        }
        
        unsigned int GetVertIndex() const { 
            return _vertIndex;
        }
        
        unsigned int GetPatchIndex() const {
            return _patchIndex;
        }
        
        unsigned int GetNumPatches() const {
            return _npatches;
        }
    
    private:
        template <class T> friend class FarPatchTablesFactory;
        
        Descriptor _desc;
        unsigned int _vertIndex,  // absolute index to the first control vertex of the first patch in the PTable
                     _patchIndex, // absolute index of the first patch in the array
                     _npatches;   // number of patches in the array
    };
    
    typedef std::vector<PatchArray> PatchArrayVector;

    /// Get the table of patch control vertices
    PTable const & GetPatchTable() const { return _patches; }

    /// Returns a pointer to the array of patches matching the descriptor
    PatchArray * GetPatchArray( Descriptor desc ) const { 
        return const_cast<FarPatchTables *>(this)->findPatchArray( desc ); 
    }

    /// Returns a vertex valence table used by Gregory patches
    VertexValenceTable const & GetVertexValenceTable() const { return _vertexValenceTable; }

    /// Returns a quad offsets table used by Gregory patches
    QuadOffsetTable const & GetQuadOffsetTable() const { return _quadOffsetTable; }

    /// Returns a PtexCoordinateTable for each type of patch
    PtexCoordinateTable const & GetPtexCoordinatesTable() const { return _ptexTable; }

    /// Returns an FVarDataTable for each type of patch
    FVarDataTable const & GetFFVarDataTable() const { return _fvarTable; }

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
private:

    template <class T> friend class FarPatchTablesFactory;
    template <class T, class U> friend class FarMultiMeshFactory;

    PatchArray * findPatchArray( Descriptor desc );

    // Private constructor
    FarPatchTables( int maxvalence ) : _maxValence(maxvalence) { }

    // Vector of descriptors for arrays of patches
    PatchArrayVector _patchArrays;


    
    PTable _patches; // Indices of the control vertices of the patches

    VertexValenceTable _vertexValenceTable; // vertex valence table (for Gregory patches)
    
    QuadOffsetTable _quadOffsetTable; // quad offsets table (for Gregory patches)
    
    PtexCoordinateTable _ptexTable;

    FVarDataTable _fvarTable;

    // highest vertex valence allowed in the mesh (used for Gregory 
    // vertexValance & quadOffset talbes)
    int _maxValence;
};


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

// Allows ordering of patches by type
inline bool 
FarPatchTables::Descriptor::operator < ( Descriptor const other ) {
    if (_pattern==NON_TRANSITION) {
        return _type < other._type;
    } else {
        if (_pattern==other._pattern)
            return _rotation < other._rotation;
        else
            return _pattern < other._pattern;
    }
} 

// True if the descriptors are identical
bool 
FarPatchTables::Descriptor::operator == ( Descriptor const other ) {
    return  _pattern == other._pattern and
               _type == other._type    and
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
