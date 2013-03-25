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
///        compilers pack bit-fields and endian-ness.
///
struct FarPtexCoord {
    unsigned int faceIndex:32; // Ptex face index
    unsigned int bitField:32;  // Patch description bits
    
    /// Sets teh values of the bit fields
    ///
    /// @param faceid ptex face index
    ///
    /// @param u value of the u parameter for the first corner of the face
    /// @param v value of the v parameter for the first corner of the face
    ///
    /// @params rots rotations required to reproduce CCW face-winding
    /// @params depth subdivision level of the patch
    /// @params nonquad true if the root face is not a quad
    ///
    void Set( unsigned int faceid, 
              short u, short v,
              unsigned char rots, unsigned char depth, bool nonquad ) {
                    
        faceIndex = faceid;
        bitField = (u << 17) |
                   (v << 7) |
                   (rots << 5) |
                   ((nonquad ? 1:0) << 4) |
                   (nonquad ? depth+1 : depth);
    }
    
    /// Resets the values to 0
    void Clear() {
        faceIndex = bitField = 0;
    }
};

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

/// \brief Container for patch vertex indices tables
///
/// FarPatchTables contain the lists of vertices for each patch of an adaptive
/// mesh representation.
///
class FarPatchTables {

public:
    /// Patch table : (vert indices, patch level) pairs
    typedef std::pair<std::vector<unsigned int>, 
                      std::vector<unsigned char> > PTable; 
                      
    typedef std::vector<int> VertexValenceTable;

    typedef std::vector<unsigned int> QuadOffsetTable;

    typedef std::vector<FarPtexCoord> PtexCoordinateTable;
    
    typedef std::vector<float> FVarDataTable;


    /// Returns a FarTable containing the vertex indices for all the Full Regular patches
    PTable const & GetFullRegularPatches() const { return _full._R_IT; }

    /// Returns a FarTable containing the vertex indices for all the Full Boundary patches
    PTable const & GetFullBoundaryPatches() const { return _full._B_IT; }

    /// Returns a FarTable containing the vertex indices for all the Full Corner patches
    PTable const & GetFullCornerPatches() const { return _full._C_IT; }

    /// Returns a FarTable containing the vertex indices for all the Full Gregory Regular patches
    PTable const & GetFullGregoryPatches() const { return _full._G_IT; }

    /// Returns a FarTable containing the vertex indices for all the Full Gregory Boundary patches
    PTable const & GetFullBoundaryGregoryPatches() const { return _full._G_B_IT; }

    /// Returns a vertex valence table used by Gregory patches
    VertexValenceTable const & GetVertexValenceTable() const { return _vertexValenceTable; }

    /// Returns a quad offsets table used by Gregory patches
    QuadOffsetTable const & GetQuadOffsetTable() const { return _quadOffsetTable; }


    /// Returns a FarTable containing the vertex indices for all the Transition Regular patches
    PTable const & GetTransitionRegularPatches(unsigned char pattern) const { return _transition[pattern]._R_IT; }

    /// Returns a FarTable containing the vertex indices for all the Transition Boundary patches
    PTable const & GetTransitionBoundaryPatches(unsigned char pattern, unsigned char rot) const { return _transition[pattern]._B_IT[rot]; }

    /// Returns a FarTable containing the vertex indices for all the Transition Corner patches
    PTable const & GetTransitionCornerPatches(unsigned char pattern, unsigned char rot) const { return _transition[pattern]._C_IT[rot]; }


    /// Ringsize of Regular Patches in table.
    static int GetRegularPatchRingsize() { return 16; }

    /// Ringsize of Boundary Patches in table.
    static int GetBoundaryPatchRingsize() { return 12; }

    /// Ringsize of Boundary Patches in table.
    static int GetCornerPatchRingsize() { return 9; }

    /// Ringsize of Gregory (and Gregory Boundary) Patches in table.
    static int GetGregoryPatchRingsize() { return 4; }


    /// Returns a PtexCoordinateTable for each type of patch
    PtexCoordinateTable const & GetFullRegularPtexCoordinates() const { return _full._R_PTX; }

    PtexCoordinateTable const & GetFullBoundaryPtexCoordinates() const { return _full._B_PTX; }

    PtexCoordinateTable const & GetFullCornerPtexCoordinates() const { return _full._C_PTX; }

    PtexCoordinateTable const & GetFullGregoryPtexCoordinates() const { return _full._G_PTX; }

    PtexCoordinateTable const & GetFullBoundaryGregoryPtexCoordinates() const { return _full._G_B_PTX; }

    PtexCoordinateTable const & GetTransitionRegularPtexCoordinates(unsigned char pattern) const { return _transition[pattern]._R_PTX; }

    PtexCoordinateTable const & GetTransitionBoundaryPtexCoordinates(unsigned char pattern, unsigned char rot) const { return _transition[pattern]._B_PTX[rot]; }

    PtexCoordinateTable const & GetTransitionCornerPtexCoordinates(unsigned char pattern, unsigned char rot) const { return _transition[pattern]._C_PTX[rot]; }

    /// Returns an FVarDataTable for each type of patch
    FVarDataTable const & GetFullRegularFVarData() const { return _full._R_FVD; }

    FVarDataTable const & GetFullBoundaryFVarData() const { return _full._B_FVD; }

    FVarDataTable const & GetFullCornerFVarData() const { return _full._C_FVD; }

    FVarDataTable const & GetFullGregoryFVarData() const { return _full._G_FVD; }

    FVarDataTable const & GetFullBoundaryGregoryFVarData() const { return _full._G_B_FVD; }

    FVarDataTable const & GetTransitionRegularFVarData(unsigned char pattern) const { return _transition[pattern]._R_FVD; }

    FVarDataTable const & GetTransitionBoundaryFVarData(unsigned char pattern, unsigned char rot) const { return _transition[pattern]._B_FVD[rot]; }

    FVarDataTable const & GetTransitionCornerFVarData(unsigned char pattern, unsigned char rot) const { return _transition[pattern]._C_FVD[rot]; }

    /// Returns the total number of patches stored in the tables
    size_t GetNumPatches() const;
    
    /// Returns the total number of control vertex indices in the tables
    size_t GetNumControlVertices() const;

    /// Returns max vertex valence
    int GetMaxValence() const { return _maxValence; }

    /// Returns PatchCounts
    FarPatchCountVector const & GetPatchCounts() const { return _patchCounts; }

private:

    template <class T> friend class FarPatchTablesFactory;
    template <class T, class U> friend class FarMultiMeshFactory;

    // Private constructor
    FarPatchTables( int maxvalence ) : _maxValence(maxvalence) { }

    // FarTables for full / end patches
    struct Patches {
        PTable _R_IT,   // regular patches vertex indices table
               _B_IT,   // boundary 
               _C_IT,   // corner
               _G_IT,   // gregory 
               _G_B_IT; // gregory 

        PtexCoordinateTable _R_PTX, // regular patches ptex indices table
                            _B_PTX,
                            _C_PTX,
                            _G_PTX,
                            _G_B_PTX;
        
        FVarDataTable       _R_FVD, // regular patches face-varying indices table
                            _B_FVD,
                            _C_FVD,
                            _G_FVD,
                            _G_B_FVD;
    };
    
    // FarTables for transition patches
    struct TPatches {
        PTable _R_IT,    // regular patches
               _B_IT[4], // boundary patches (4 rotations)
               _C_IT[4]; // corner patches (4 rotations)

        PtexCoordinateTable _R_PTX,
                            _B_PTX[4],
                            _C_PTX[4];
               
        FVarDataTable       _R_FVD,
                            _B_FVD[4],
                            _C_FVD[4];
    };

    Patches _full; // full patches tables
    
    TPatches _transition[5]; // transition patches tables

    // XXXX manuelk : Greg. patch tables need to be localized to Gregory CVs only.

    // vertex valence table (for Gregory patches)
    VertexValenceTable _vertexValenceTable; 

    // quad offsets table (for Gregory patches)
    QuadOffsetTable _quadOffsetTable; 

    // highest vertex valence allowed in the mesh (used for Gregory 
    // vertexValance & quadOffset talbes)
    int _maxValence;
    
    // vector of counters for aggregated patch tables used by multi-meshes
    FarPatchCountVector _patchCounts; 
};

// Returns the total number of patches stored in the tables
inline size_t 
FarPatchTables::GetNumPatches() const {

    // We can use directly the size of the levels table ("second") because
    // there is 1 value per patch 
    size_t count = _full._R_IT.second.size()  + 
                   _full._B_IT.second.size() +
                   _full._C_IT.second.size() +
                   _full._G_IT.second.size()  + 
                   _full._G_B_IT.second.size();

    for (int i = 0; i < 5; ++i) {
        count += _transition[i]._R_IT.second.size();
        for (int j = 0; j < 4; ++j) {
            count += _transition[i]._B_IT[j].second.size()+
                     _transition[i]._C_IT[j].second.size();
        }
    }

    return count;
}

// Returns the total number of control vertex indices in the tables
inline size_t 
FarPatchTables::GetNumControlVertices() const {

    // The "first" table of a PTable contains the vertex indices of each
    // patch, so we can directly use those to tally our count.
    size_t count = _full._R_IT.first.size() + 
                   _full._B_IT.first.size() +
                   _full._C_IT.first.size() +
                   _full._G_IT.first.size() + 
                   _full._G_B_IT.first.size();

    for (int i = 0; i < 5; ++i) {
        count += _transition[i]._R_IT.first.size();
        for (int j = 0; j < 4; ++j) {
            count += _transition[i]._B_IT[j].first.size()+
                     _transition[i]._C_IT[j].first.size();
        }
    }

    return count;
}



} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* FAR_PATCH_TABLES */
