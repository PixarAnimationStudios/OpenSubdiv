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

#ifndef FAR_PTACH_TABLES_H
#define FAR_PTACH_TABLES_H

#include "../version.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

/// \brief Container for patch vertex indices tables
///
/// FarPatchTables contain the lists of vertices for each patch of an adaptive
/// mesh representation.
///
class FarPatchTables {

public:
    typedef FarTable<unsigned int> PTable;

    typedef std::vector<int> VertexValenceTable;

    typedef std::vector<unsigned int> QuadOffsetTable;

    typedef std::vector<int> PtexCoordinateTable;
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

    /// Ringsize of Gregory Patches in table.
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

    /// Returns max vertex valence
    int GetMaxValence() const { return _maxValence; }

private:

    template <class T> friend class FarPatchTablesFactory;

    // Private constructor
    FarPatchTables( int maxlevel, int maxvalence ) : _full(maxlevel+1), _maxValence(maxvalence) {
        for (unsigned char i=0; i<5; ++i)
            _transition[i].SetMaxLevel(maxlevel+1);
    }

    // FarTables for full / end patches
    struct Patches {
        PTable _R_IT,   // regular patches
               _B_IT,   // boundary patches
               _C_IT,   // corner patches
               _G_IT,   // gregory patches
               _G_B_IT; // gregory boundary patches

        PtexCoordinateTable _R_PTX,
                            _B_PTX,
                            _C_PTX,
                            _G_PTX,
                            _G_B_PTX;
        
        FVarDataTable       _R_FVD,
                            _B_FVD,
                            _C_FVD,
                            _G_FVD,
                            _G_B_FVD;
        
        Patches(int maxlevel) : _R_IT(maxlevel), 
                                _B_IT(maxlevel),
                                _C_IT(maxlevel),
                                _G_IT(maxlevel),
                                _G_B_IT(maxlevel)
        { }
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
               
        void SetMaxLevel(int maxlevel) {
            _R_IT.SetMaxLevel(maxlevel);
            for (unsigned char i=0; i<4; ++i) {
                _B_IT[i].SetMaxLevel(maxlevel);
                _C_IT[i].SetMaxLevel(maxlevel);                
            }
        }       
    };

    Patches _full;
    
    TPatches _transition[5];

    VertexValenceTable _vertexValenceTable;

    QuadOffsetTable _quadOffsetTable;

    int _maxValence;
};

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* FAR_VERTEX_EDIT_TABLES_H */
