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
#include "../far/patchTablesFactory.h"
#include "../far/gregoryBasis.h"
#include "../far/patchTables.h"
#include "../far/topologyRefiner.h"
#include "../vtr/level.h"
#include "../vtr/refinement.h"

#include <algorithm>
#include <cassert>
#include <cstring>


namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Far {

//
//  A convenience container for the different types of feature adaptive patches
//
template<class TYPE>
struct PatchTypes {

    static const int NUM_TRANSITIONS=6,
                     NUM_ROTATIONS=4;

    TYPE R[NUM_TRANSITIONS],                   // regular patch
         S[NUM_TRANSITIONS][NUM_ROTATIONS],    // single-crease patch
         B[NUM_TRANSITIONS][NUM_ROTATIONS],    // boundary patch (4 rotations)
         C[NUM_TRANSITIONS][NUM_ROTATIONS],    // corner patch (4 rotations)
         G,                                    // gregory patch
         GB,                                   // gregory boundary patch
         GP;                                   // gregory basis patch

    PatchTypes() { std::memset(this, 0, sizeof(PatchTypes<TYPE>)); }

    // Returns the number of patches based on the patch type in the descriptor
    TYPE & getValue( PatchDescriptor desc ) {
        switch (desc.GetType()) {
            case PatchDescriptor::REGULAR          : return R[desc.GetPattern()];
            case PatchDescriptor::SINGLE_CREASE    : return S[desc.GetPattern()][desc.GetRotation()];
            case PatchDescriptor::BOUNDARY         : return B[desc.GetPattern()][desc.GetRotation()];
            case PatchDescriptor::CORNER           : return C[desc.GetPattern()][desc.GetRotation()];
            case PatchDescriptor::GREGORY          : return G;
            case PatchDescriptor::GREGORY_BOUNDARY : return GB;
            case PatchDescriptor::GREGORY_BASIS    : return GP;
            default : assert(0);
        }
        // can't be reached (suppress compiler warning)
        return R[0];
    }

    // Counts the number of arrays required to store each type of patch used
    // in the primitive
    int getNumPatchArrays() const {
        int result=0;
        for (int i=0; i<6; ++i) {

            if (R[i]) ++result;

            for (int j=0; j<4; ++j) {
                if (S[i][j]) ++result;
                if (B[i][j]) ++result;
                if (C[i][j]) ++result;

            }
        }
        if (G) ++result;
        if (GB) ++result;
        if (GP) ++result;
        return result;
    }

    // Returns true if there's any single-crease patch
    bool hasSingleCreasedPatches() const {
        for (int i=0; i<6; ++i) {
            for (int j=0; j<4; ++j) {
                if (S[i][j]) return true;
            }
        }
        return false;
    }
};

typedef PatchTypes<Index *>         PatchCVPointers;
typedef PatchTypes<PatchParam *>    PatchParamPointers;
typedef PatchTypes<Index *>         SharpnessIndexPointers;
typedef PatchTypes<int>             PatchCounters;
typedef PatchTypes<Index **>        PatchFVarPointers;

//
//  A simple struct containing all information gathered about a face that is relevant
//  to constructing a patch for it (some of these enums should probably be defined more
//  as part of PatchTables)
//
//  Like the HbrFace<T>::AdaptiveFlags, this struct aggregates all of the face tags
//  supporting feature adaptive refinement.  For now it is not used elsewhere and can
//  remain local to this implementation, but we may want to move it into a header of
//  its own if it has greater use later.
//
//  Note that several properties being assigned here attempt to do so given a 4-bit
//  mask of properties at the edges or vertices of the quad.  Still not sure exactly
//  what will be done this way, but the goal is to create lookup tables (of size 16
//  for the 4 bits) to quickly determine was is needed, rather than iteration and
//  branching on the edges or vertices.
//
struct PatchFaceTag {
public:
    //  The HBR_ADAPTIVE TransitionType from <hbr/face.h> -- now named to more clearly
    //  reflect the number and orientation of transitional edges.  Note that the values
    //  assigned here need to match the intended purpose to remain consistent with Hbr:
    enum TransitionType {
        NONE          = 0,
        TRANS_ONE     = 1,
        TRANS_TWO_ADJ = 2,
        TRANS_THREE   = 3,
        TRANS_ALL     = 4,
        TRANS_TWO_OPP = 5
    };

public:
    unsigned int   _hasPatch        : 1;
    unsigned int   _isRegular       : 1;
    unsigned int   _isTransitional  : 1;
    unsigned int   _transitionType  : 3;
    unsigned int   _transitionRot   : 2;
    unsigned int   _boundaryIndex   : 2;
    unsigned int   _boundaryCount   : 3;
    unsigned int   _hasBoundaryEdge : 3;
    unsigned int   _isSingleCrease  : 1;

    void clear() { std::memset(this, 0, sizeof(*this)); }

    void assignBoundaryPropertiesFromEdgeMask(int boundaryEdgeMask) {
        //
        //  The number of rotations to apply for boundary or corner patches varies on both
        //  where the boundary/corner occurs and whether boundary or corner -- so using a
        //  4-bit mask should be sufficient to quickly determine all cases:
        //
        //  Note that we currently expect patches with multiple boundaries to have already
        //  been isolated, so asserts are applied for such unexpected cases.
        //
        //  Is the compiler going to build the 16-entry lookup table here, or should we do
        //  it ourselves?
        //
        _hasBoundaryEdge = true;

        switch (boundaryEdgeMask) {
        case 0x0:  _boundaryCount = 0, _boundaryIndex = 0, _hasBoundaryEdge = false;  break;  // no boundaries

        case 0x1:  _boundaryCount = 1, _boundaryIndex = 0;  break;  // boundary edge 0
        case 0x2:  _boundaryCount = 1, _boundaryIndex = 1;  break;  // boundary edge 1
        case 0x3:  _boundaryCount = 2, _boundaryIndex = 1;  break;  // corner/crease vertex 1
        case 0x4:  _boundaryCount = 1, _boundaryIndex = 2;  break;  // boundary edge 2
        case 0x5:  assert(false);                           break;  // N/A - opposite boundary edges
        case 0x6:  _boundaryCount = 2, _boundaryIndex = 2;  break;  // corner/crease vertex 2
        case 0x7:  assert(false);                           break;  // N/A - three boundary edges
        case 0x8:  _boundaryCount = 1, _boundaryIndex = 3;  break;  // boundary edge 3
        case 0x9:  _boundaryCount = 2, _boundaryIndex = 0;  break;  // corner/crease vertex 0
        case 0xa:  assert(false);                           break;  // N/A - opposite boundary edges
        case 0xb:  assert(false);                           break;  // N/A - three boundary edges
        case 0xc:  _boundaryCount = 2, _boundaryIndex = 3;  break;  // corner/crease vertex 3
        case 0xd:  assert(false);                           break;  // N/A - three boundary edges
        case 0xe:  assert(false);                           break;  // N/A - three boundary edges
        case 0xf:  assert(false);                           break;  // N/A - all boundaries
        default:   assert(false);                           break;
        }
    }

    void assignBoundaryPropertiesFromVertexMask(int boundaryVertexMask) {
        //
        //  This is strictly needed for the irregular case when a vertex is a boundary in
        //  the presence of no boundary edges -- an extra-ordinary face with only one corner
        //  on the boundary.
        //
        //  Its unclear at this point if patches with more than one such vertex are supported
        //  (if so, how do we deal with rotations) so for now we only allow one such vertex
        //  and assert for all other cases.
        //
        assert(_hasBoundaryEdge == false);

        switch (boundaryVertexMask) {
        case 0x0:  _boundaryCount = 0;                      break;  // no boundaries
        case 0x1:  _boundaryCount = 1, _boundaryIndex = 0;  break;  // boundary vertex 0
        case 0x2:  _boundaryCount = 1, _boundaryIndex = 1;  break;  // boundary vertex 1
        case 0x3:  assert(false);                           break;
        case 0x4:  _boundaryCount = 1, _boundaryIndex = 2;  break;  // boundary vertex 2
        case 0x5:  assert(false);                           break;
        case 0x6:  assert(false);                           break;
        case 0x7:  assert(false);                           break;
        case 0x8:  _boundaryCount = 1, _boundaryIndex = 3;  break;  // boundary vertex 3
        case 0x9:  assert(false);                           break;
        case 0xa:  assert(false);                           break;
        case 0xb:  assert(false);                           break;
        case 0xc:  assert(false);                           break;
        case 0xd:  assert(false);                           break;
        case 0xe:  assert(false);                           break;
        case 0xf:  assert(false);                           break;
        default:   assert(false);                           break;
        }
    }

    void assignTransitionRotationForCorner(int transitionEdgeMask) {
        //
        //  Corner transition patches have only two interior edges that may be transitional.
        //
        //  Either both are transitional (TRANS_TWO_ADJ) with only a single possible orientation,
        //  or only one is transitional (TRANS_ONE) with two possibilities.  The former case is
        //  trivial.  For the latter, use the known corner index to identify one of the two
        //  possible transition masks and test to determine between the two cases.
        //
        if (_transitionType == TRANS_ONE) {
            int const edgeMaskPerCorner[] = { 4, 8, 1, 2 };

            _transitionRot = 1 + (edgeMaskPerCorner[_boundaryIndex] != transitionEdgeMask);
        } else {
            _transitionRot = 1;
        }
    }

    void assignTransitionRotationForBoundary(int transitionEdgeMask) {
        //
        //  Boundary transition patches have three interior edges that may be transitional.
        //
        //  The case of all three transitional (TRANS_THREE) has only one orientation, while the
        //  case of two opposite transitional edges (TRANS_TWO_OPP) also has only one orientation.
        //  So both of these are trivially handled.
        //
        //  The case of a single transitional edge (TRANS_ONE) or one transitional edge (TRANS_TWO_ADJ)
        //  both have multiple orientations -- three for TRANS_ONE and two for TRANS_TWO_ADJ.  Each is
        //  handled separately:
        //
        if (_transitionType == TRANS_ONE) {
            if (transitionEdgeMask == (1 << ((_boundaryIndex + 2) % 4))) {
                _transitionRot = 2;
            } else if (transitionEdgeMask == (1 << ((_boundaryIndex + 1) % 4))) {
                _transitionRot = 1;
            } else {
                _transitionRot = 3;
            }
            // XXXX manuelk mirror this rotation to match shader idiosyncracies
            _transitionRot = (4-_transitionRot)%4;
        } else if (_transitionType == TRANS_TWO_ADJ) {
            int const edgeMaskPerBoundary[] = { 6, 12, 9, 3 };
            _transitionRot = 1 + (edgeMaskPerBoundary[_boundaryIndex] == transitionEdgeMask);
        } else if (_transitionType == TRANS_THREE) {
            _transitionRot = 0;
        } else {
            _transitionRot = 1;
        }
    }

    void assignTransitionRotationForSingleCrease(int transitionEdgeMask) {
        //
        // Single crease transition patches.
        //
        // rotate edgemask by boundaryIndex to align the creased edge
        //
        transitionEdgeMask = ((transitionEdgeMask >> _boundaryIndex) |
                              (transitionEdgeMask << (4-_boundaryIndex))) % 16;

        /*
           edgemask  type    : rotation to match to shader
           0000  0 : NONE    : 0
           0001  1 : ONE     : 0
           0010  2 : ONE     : 3
           0011  3 : TWO_ADJ : 3
           0100  4 : ONE     : 2
           0101  5 : TWO_OPP : 0
           0110  6 : TWO_ADJ : 2
           0111  7 : THREE   : 1  (needs verify)
           1000  8 : ONE     : 1
           1001  9 : TWO_ADJ : 0
           1010 10 : TWO_OPP : 1
           1011 11 : THREE   : 2  (needs verify)
           1100 12 : TWO_ADJ : 1
           1101 13 : THREE   : 3
           1110 14 : THREE   : 0  (needs verify)
           1111 15 : ALL     : 0
        */
        static int transitionRots[16] = {0, 0, 3, 3, 2, 0, 2, 1, 1, 0,  1,  2,  1,  3,  0,  0 };

        _transitionRot = transitionRots[transitionEdgeMask];
    }

    void assignTransitionPropertiesFromEdgeMask(int transitionEdgeMask) {
        //
        //  Note the transition rotations will be a function of the boundary rotations, and
        //  so boundary rotations/index should have been previously assigned:
        //
        //  As with the boundary rotation case, consider retrieving values from static 16-
        //  entry lookup tables if possible (depending on the function involving boundary
        //  rotations)...
        //
        _isTransitional = (transitionEdgeMask != 0);

        switch (transitionEdgeMask) {
            case 0x0:  _transitionType = NONE;          break;  // no transitions
            case 0x1:  _transitionType = TRANS_ONE;     break;  // single edge 0
            case 0x2:  _transitionType = TRANS_ONE;     break;  // single edge 1
            case 0x3:  _transitionType = TRANS_TWO_ADJ; break;  // two adjacent edges, 0 and 1
            case 0x4:  _transitionType = TRANS_ONE;     break;  // single edge 2
            case 0x5:  _transitionType = TRANS_TWO_OPP; break;  // two opposite edges, 0 and 2
            case 0x6:  _transitionType = TRANS_TWO_ADJ; break;  // two adjacent edges, 1 and 2
            case 0x7:  _transitionType = TRANS_THREE;   break;  // three edges, all but 3
            case 0x8:  _transitionType = TRANS_ONE;     break;  // single edge 3
            case 0x9:  _transitionType = TRANS_TWO_ADJ; break;  // two adjacent edges, 3 and 0
            case 0xa:  _transitionType = TRANS_TWO_OPP; break;  // two opposite edges, 1 and 3
            case 0xb:  _transitionType = TRANS_THREE;   break;  // three edges, all but 2
            case 0xc:  _transitionType = TRANS_TWO_ADJ; break;  // two adjacent edges, 2 and 3
            case 0xd:  _transitionType = TRANS_THREE;   break;  // three edges, all but 1
            case 0xe:  _transitionType = TRANS_THREE;   break;  // three edges, all but 0
            case 0xf:  _transitionType = TRANS_ALL;     break;  // all edges
            default:   assert(false);                   break;
        }

        //  May need another switch/lookup table here or combine it with the above -- the
        //  results below are a function of both transition and boundary properties...
        if (transitionEdgeMask == 0) {
            _transitionRot = 0;
        } else if (_boundaryCount == 0 and _isSingleCrease) {
                assignTransitionRotationForSingleCrease(transitionEdgeMask);
        } else if (_boundaryCount == 0) {
            // XXXX manuelk Rotations are mostly a direct map of the transitionEdgeMask
            //                  Except for:
            //                  - TRANS_TWO_ADJ that has rotation { 1, 2, 0, 3 }
            //                  - TRANS_THREE that has rotation { 3, 2, 1, 0 }
            //                  (matching shader idiosyncracies)
            static unsigned char transitionRots[16] = {0, 0, 1, 1, 2, 0, 2, 3, 3, 0, 1, 2, 3, 1, 0, 0};

            _transitionRot = transitionRots[transitionEdgeMask];
        } else if (_boundaryCount == 1) {
            assignTransitionRotationForBoundary(transitionEdgeMask);
        } else if (_boundaryCount == 2) {
            assignTransitionRotationForCorner(transitionEdgeMask);
        }
    }
};

typedef std::vector<PatchFaceTag> PatchTagVector;


//
//  Trivial anonymous helper functions:
//
namespace {
    inline void
    offsetAndPermuteIndices(Far::Index const indices[], int count,
                            Far::Index offset, int const permutation[],
                            Far::Index result[]) {

        if (permutation) {
            for (int i = 0; i < count; ++i) {
                result[i] = offset + indices[permutation[i]];
            }
        } else if (offset) {
            for (int i = 0; i < count; ++i) {
                result[i] = offset + indices[i];
            }
        } else {
            std::memcpy(result, indices, count * sizeof(Far::Index));
        }
    }
} // namespace anon

//
//  Reserves tables based on the contents of the PatchArrayVector in the PatchTables:
//
void
PatchTablesFactory::allocateTables(PatchTables * tables, int /* nlevels */, bool hasSharpness) {

    int ncvs = 0, npatches = 0;
    for (int i=0; i<tables->GetNumPatchArrays(); ++i) {
        npatches += tables->GetNumPatches(i);
        ncvs += tables->GetNumControlVertices(i);
    }

    if (ncvs==0 or npatches==0)
        return;

    tables->_patchVerts.resize( ncvs );

    tables->_paramTable.resize( npatches );

    if (hasSharpness) {
        tables->_sharpnessIndices.resize( npatches, Vtr::INDEX_INVALID );
    }
}

PatchTables::FVarPatchTables *
PatchTablesFactory::allocateFVarTables( TopologyRefiner const & refiner,
    PatchTables const & tables, Options options ) {

    assert( refiner.GetNumFVarChannels()>0 );

    FVarPatchTables * fvarTables = new FVarPatchTables;

    fvarTables->_channels.resize( refiner.GetNumFVarChannels() );

    if (refiner.IsUniform()) {

        assert( not tables.IsFeatureAdaptive() );

        int maxlevel = refiner.GetMaxLevel();
        for (int channel=0; channel<refiner.GetNumFVarChannels(); ++channel) {

            int nverts = options.generateAllLevels ?
                refiner.GetNumFacesTotal() :
                    refiner.GetNumFaces(maxlevel);

            assert(tables.GetNumPatchArrays()>0);
            nverts *= tables.GetPatchArrayDescriptor(0).GetNumFVarControlVertices();
            if (options.triangulateQuads) {
                nverts *= 2;
            }
            assert(nverts>0);
            fvarTables->_channels[channel].patchVertIndices.resize(nverts);
        }
    } else {

        assert( tables.IsFeatureAdaptive() );

        int nverts=0;
        for (int i=0; i<tables.GetNumPatchArrays(); ++i) {
            nverts += tables.GetNumPatches(i) *
                tables.GetPatchArrayDescriptor(i).GetNumFVarControlVertices();
        }
        assert(nverts>0);

        for (int channel=0; channel<refiner.GetNumFVarChannels(); ++channel) {
            fvarTables->_channels[channel].patchVertIndices.resize(nverts);
        }
    }

    return fvarTables;
}

//
//  Populates the face-varying data buffer 'coord' for the given face, returning
//  a pointer to the next descriptor
//
PatchParam *
PatchTablesFactory::computePatchParam(TopologyRefiner const & refiner,
                                         int depth, Vtr::Index faceIndex, int rotation,
                                         PatchParam *coord) {

    if (coord == NULL) return NULL;

    // Move up the hierarchy accumulating u,v indices to the coarse level:
    int childIndexInParent = 0,
        u = 0,
        v = 0,
        ofs = 1;

    bool nonquad = (refiner.GetFaceVertices(depth, faceIndex).size() != 4);

    for (int i = depth; i > 0; --i) {
        Vtr::Refinement const& refinement  = refiner.getRefinement(i-1);
        Vtr::Level const&      parentLevel = refiner.getLevel(i-1);

        Vtr::Index parentFaceIndex    = refinement.getChildFaceParentFace(faceIndex);
                 childIndexInParent = refinement.getChildFaceInParentFace(faceIndex);

        if (parentLevel.getFaceVertices(parentFaceIndex).size() == 4) {
            switch ( childIndexInParent ) {
                case 0 :                     break;
                case 1 : { u+=ofs;         } break;
                case 2 : { u+=ofs; v+=ofs; } break;
                case 3 : {         v+=ofs; } break;
            }
            ofs = (unsigned short)(ofs << 1);
        } else {
            nonquad = true;
            // If the root face is not a quad, we need to offset the ptex index
            // CCW to match the correct child face
            Vtr::ConstIndexArray children = refinement.getFaceChildFaces(parentFaceIndex);
            for (int j=0; j<children.size(); ++j) {
                if (children[j]==faceIndex) {
                    childIndexInParent = j;
                    break;
                }
            }
        }
        faceIndex = parentFaceIndex;
    }

    Vtr::Index ptexIndex = refiner.GetPtexIndex(faceIndex);
    assert(ptexIndex!=-1);

    if (nonquad) {
        ptexIndex+=childIndexInParent;
        --depth;
    }

    coord->Set(ptexIndex, (short)u, (short)v, (unsigned char) rotation, (unsigned char) depth, nonquad);

    return ++coord;
}

// XXXX manuelk work in progress for end-cap topology gathering
#ifdef ENDCAP_TOPOPOLGY
//
// Populate the topology table used by Gregory-basis patches
//
// Note: 'faceIndex' values are expected to be sorted in ascending order !!!
static int
gatherGregoryBasisTopology(Vtr::Level const& level, Index faceIndex, int numVertices,
    PatchFaceTag const * levelPatchTags,
        bool skip[0], std::vector<Index> & basisIndices, PatchTables::PTable & topology) {

    assert(not topology.empty());
    Index * dest = &topology[basisIndices.size()*20];

    assert(Vtr::INDEX_INVALID==0xFFFFFFFF);
    memset(dest, 0xFF, 20*sizeof(Index));

    IndexArray fedges = level.getFaceEdges(faceIndex);
    assert(fedges.size()==4);

    for (int i=0; i<4; ++i) {
        Index edge = fedges[i],
              adjface = 0;

        { // Gather adjacent faces
            IndexArray adjfaces = level.getEdgeFaces(edge);
            for (int i=0; i<adjfaces.size(); ++i) {
                if (adjfaces[i]==faceIndex) {
                    // XXXX manuelk if 'edge' is non-manifold, arbitrarily pick the
                    // next face in the list of adjacent faces
                    adjface = (adjfaces[(i+1)%adjfaces.size()]);
                    break;
                }
            }
        }
        // We are looking for adjacent faces that:
        // - exist (no boundary)
        // - have alraedy been processed (known CV indices)
        // - are also Gregory basis patches
        if (adjface!=Vtr::INDEX_INVALID and (adjface < faceIndex) and
            (not levelPatchTags[adjface]._isRegular)) {

            IndexArray aedges = level.getFaceEdges(adjface);
            int aedge = aedges.FindIndexIn4Tuple(edge);
            assert(aedge!=Vtr::INDEX_INVALID);

            // Find index of basis in the list of basis already generated
            struct compare {
                static int op(void const * a, void const * b) {
                   return *(Index *)a - *(Index *)b;
                }
            };

            Index * ptr = (Index *)std::bsearch( &adjface, &basisIndices[0],
                basisIndices.size(), sizeof(Index), compare::op);

            int srcBasisIdx = ptr - &basisIndices[0];
            assert(ptr and srcBasisIdx>=0 and srcBasisIdx<(int)basisIndices.size());

            // Copy the indices of CVs from the face on the other side of the
            // shared edge
            static int const gregoryEdgeVerts[4][4] = { { 0,  1,  7,  5},
                                                        { 5,  6, 12, 10},
                                                        {10, 11, 17, 15},
                                                        {15, 16,  2,  0} };
            Index * src = &topology[srcBasisIdx*20];
            for (int j=0; j<4; ++j) {
                dest[i*4+j] = src[gregoryEdgeVerts[aedge][j]];
            }

            skip[i] = true;
        } else {
            skip[i] = false;
        }
    }
    for (int i=0; i<20; ++i) {
        if (dest[i]==Vtr::INDEX_INVALID) {
            dest[i] = numVertices++;
        }
    }
    basisIndices.push_back(faceIndex);
    return numVertices;
}
#endif

//
//  Populate the quad-offsets table used by Gregory patches
//
void
PatchTablesFactory::getQuadOffsets(
    Vtr::Level const& level, Index faceIndex, unsigned int offsets[]) {

    Vtr::ConstIndexArray fVerts = level.getFaceVertices(faceIndex);

    for (int i = 0; i < 4; ++i) {

        Vtr::Index vIndex = fVerts[i];
        Vtr::ConstIndexArray vFaces = level.getVertexFaces(vIndex),
                      vEdges = level.getVertexEdges(vIndex);

        int thisFaceInVFaces = -1;
        for (int j = 0; j < vFaces.size(); ++j) {
            if (faceIndex == vFaces[j]) {
                thisFaceInVFaces = j;
                break;
            }
        }
        assert(thisFaceInVFaces != -1);

        Index vOffsets[2];
        vOffsets[0] = thisFaceInVFaces;
        vOffsets[1] = (thisFaceInVFaces + 1)%vEdges.size();
        // we have to use the number of incident edges to modulo the local index
        // because there could be 2 consecutive edges in the face belonging to
        // the Gregory patch.
        offsets[i] = vOffsets[0] | (vOffsets[1] << 8);
    }
}


//
//  Indexing sharpnesses
//
int
PatchTablesFactory::assignSharpnessIndex( PatchTables *tables, float sharpness ) {

    // linear search
    for (int i=0; i<(int)tables->_sharpnessValues.size(); ++i) {
        if (tables->_sharpnessValues[i] == sharpness) return i;
    }
    tables->_sharpnessValues.push_back(sharpness);
    return (int)tables->_sharpnessValues.size()-1;
}

//
//  We should be able to use a single Create() method for both the adaptive and uniform
//  cases.  In the past, more additional arguments were passed to the uniform version,
//  but that may no longer be necessary (see notes in the uniform version below)...
//
PatchTables *
PatchTablesFactory::Create( TopologyRefiner const & refiner, Options options ) {

    if (refiner.IsUniform()) {
        return createUniform(refiner, options);
    } else {
        return createAdaptive(refiner, options);
    }
}

static void
gatherFVarPatchVertices(TopologyRefiner const & refiner,
    int level, Index faceIndex, int rotation, Index const * levelOffsets, Index ** fptrs) {

    for (int channel=0; channel<refiner.GetNumFVarChannels(); ++channel) {
        ConstIndexArray fverts = refiner.GetFVarFaceValues(level, faceIndex, channel);
        for (int vert=0; vert<fverts.size(); ++vert) {
            fptrs[channel][vert] = levelOffsets[channel] + fverts[(vert+rotation)%4];
        }
        fptrs[channel]+=fverts.size();
    }
}

PatchTables *
PatchTablesFactory::createUniform( TopologyRefiner const & refiner, Options options ) {

    assert(refiner.IsUniform());

    // ensure that triangulateQuads is only set for quadrilateral schemes
    options.triangulateQuads &= (refiner.GetSchemeType()==Sdc::SCHEME_BILINEAR or
                                 refiner.GetSchemeType()==Sdc::SCHEME_CATMARK);

    int maxvalence = refiner.getLevel(0).getMaxValence(),
        maxlevel = refiner.GetMaxLevel(),
        firstlevel = options.generateAllLevels ? 0 : maxlevel,
        nlevels = maxlevel-firstlevel+1;

    PatchDescriptor::Type ptype = PatchDescriptor::NON_PATCH;
    if (options.triangulateQuads) {
        ptype = PatchDescriptor::TRIANGLES;
    } else {
        switch (refiner.GetSchemeType()) {
            case Sdc::SCHEME_BILINEAR :
            case Sdc::SCHEME_CATMARK  : ptype = PatchDescriptor::QUADS; break;
            case Sdc::SCHEME_LOOP     : ptype = PatchDescriptor::TRIANGLES; break;
        }
    }
    assert(ptype!=PatchDescriptor::NON_PATCH);

    //
    //  Create the instance of the tables and allocate and initialize its members.
    //
    PatchTables * tables = new PatchTables(maxvalence);

    tables->_numPtexFaces = refiner.GetNumPtexFaces();

    tables->reservePatchArrays(nlevels);

    PatchDescriptor desc(ptype, PatchDescriptor::NON_TRANSITION, 0);

    // generate patch arrays
    for (int level=firstlevel, poffset=0, voffset=0; level<=maxlevel; ++level) {

        int npatches = refiner.GetNumFaces(level);
        if (refiner.HasHoles()) {
            npatches -= refiner.GetNumHoles(level);
        }
        assert(npatches>=0);

        if (options.triangulateQuads)
            npatches *= 2;

        if (level>=firstlevel) {
            tables->pushPatchArray(desc, npatches, &voffset, &poffset, 0);
        }
    }

    // Allocate various tables
    allocateTables( tables, 0, /*hasSharpness=*/false );

    if (options.generateFVarTables) {
        tables->_fvarPatchTables = allocateFVarTables( refiner, *tables, options );
    }

    //
    //  Now populate the patches:
    //
    Index          * iptr = &tables->_patchVerts[0];
    PatchParam     * pptr = &tables->_paramTable[0];
    Index         ** fptr = 0;

    if (tables->_fvarPatchTables) {
        int nchannels = refiner.GetNumFVarChannels();
        fptr = (Index **)alloca(nchannels*sizeof(Index *));
        for (int channel=0; channel<nchannels; ++channel) {
            fptr[channel] = const_cast<Index *>(
                &tables->_fvarPatchTables->_channels[channel].patchVertIndices[0]);
        }
    }

    Index levelVertOffset = options.generateAllLevels ? 0 : refiner.GetNumVertices(0);

    Index * levelFVarVertOffsets = 0;
    if (tables->_fvarPatchTables) {
         levelFVarVertOffsets = (Index *)alloca(refiner.GetNumFVarChannels()*sizeof(Index));
         memset(levelFVarVertOffsets, 0, refiner.GetNumFVarChannels()*sizeof(Index));
    }

    for (int level=1; level<=maxlevel; ++level) {

        int nfaces = refiner.GetNumFaces(level);
        if (level>=firstlevel) {
            for (int face=0; face<nfaces; ++face) {

                if (refiner.HasHoles() and refiner.IsHole(level, face)) {
                    continue;
                }

                ConstIndexArray fverts = refiner.GetFaceVertices(level, face);

                for (int vert=0; vert<fverts.size(); ++vert) {
                    *iptr++ = levelVertOffset + fverts[vert];
                }

                pptr = computePatchParam(refiner, level, face, /*rot*/0, pptr);

                if (tables->_fvarPatchTables) {
                    gatherFVarPatchVertices(refiner, level, face, 0, levelFVarVertOffsets, fptr);
                }

                if (options.triangulateQuads) {
                    // Triangulate the quadrilateral: {v0,v1,v2,v3} -> {v0,v1,v2},{v3,v0,v2}.
                    *iptr = *(iptr - 4); // copy v0 index
                    ++iptr;
                    *iptr = *(iptr - 3); // copy v2 index
                    ++iptr;

                    *pptr = *(pptr - 1); // copy first patch param
                    ++pptr;

                    if (tables->_fvarPatchTables) {
                        for (int channel=0; channel<refiner.GetNumFVarChannels(); ++channel) {
                            *fptr[channel] = *(fptr[channel]-4); // copy fv0 index
                            ++fptr[channel];
                            *fptr[channel] = *(fptr[channel]-3); // copy fv2 index
                            ++fptr[channel];
                        }
                    }
                }
            }
        }

        if (options.generateAllLevels) {
            levelVertOffset += refiner.GetNumVertices(level);
            if (tables->_fvarPatchTables) {
                int nchannels = refiner.GetNumFVarChannels();
                for (int channel=0; channel<nchannels; ++channel) {
                    levelFVarVertOffsets[channel] += refiner.GetNumFVarValues(level, channel);
                }
            }
        }
    }
    return tables;
}

PatchTables *
PatchTablesFactory::createAdaptive( TopologyRefiner const & refiner, Options options ) {

    assert(not refiner.IsUniform());

    //
    //  First identify the patches -- accumulating the inventory patches for all of the
    //  different types and information about the patch for each face:
    //
    PatchCounters             patchInventory;
    std::vector<PatchFaceTag> patchTags;

    identifyAdaptivePatches(refiner, patchInventory, patchTags, options);

    //
    //  Create the instance of the tables and allocate and initialize its members based on
    //  the inventory of patches determined above:
    //
    int maxValence = refiner.getLevel(0).getMaxValence();

    PatchTables * tables = new PatchTables(maxValence);

    // Populate the patch array descriptors
    tables->reservePatchArrays(patchInventory.getNumPatchArrays());

    // sort through the inventory and push back non-empty patch arrays
    typedef PatchDescriptorVector DescVec;

    DescVec const & descs = PatchDescriptor::GetAdaptivePatchDescriptors(Sdc::SCHEME_CATMARK);

    int voffset=0, poffset=0, qoffset=0;
    for (DescVec::const_iterator it=descs.begin(); it!=descs.end(); ++it) {
        tables->pushPatchArray(*it, patchInventory.getValue(*it), &voffset, &poffset, &qoffset );
    }

    tables->_numPtexFaces = refiner.GetNumPtexFaces();

    // Allocate various tables
    bool hasSharpness = patchInventory.hasSingleCreasedPatches();
    allocateTables( tables, 0, hasSharpness );

    if (options.generateFVarTables) {
        tables->_fvarPatchTables = allocateFVarTables( refiner, *tables, options );
    }

    // Specifics for Gregory patches
    if ((patchInventory.G > 0) or (patchInventory.GB > 0)) {
        tables->_quadOffsetsTable.resize( patchInventory.G*4 + patchInventory.GB*4 );
    }

    //
    //  Now populate the patches:
    //
    populateAdaptivePatches(refiner, patchInventory, patchTags, tables, options);

    return tables;
}

//
//  Identify all patches required for faces at all levels -- accumulating the number of patches
//  for each type, and retaining enough information for the patch for each face to populate it
//  later with no additional analysis.
//
void
PatchTablesFactory::identifyAdaptivePatches( TopologyRefiner const & refiner,
                                             PatchCounters &         patchInventory,
                                             PatchTagVector &        patchTags,
                                             Options                 options ) {
    //
    //  Iterate through the levels of refinement to inspect and tag components with information
    //  relative to patch generation.  We allocate all of the tags locally and use them to
    //  populate the patches once a complete inventory has been taken and all tables appropriately
    //  allocated and initialized:
    //
    //  The first Level may have no Refinement if it is the only level -- similarly the last Level
    //  has no Refinement, so a single level is effectively the last, but with less information
    //  available in some cases, as it was not generated by refinement.
    //
    patchTags.resize(refiner.GetNumFacesTotal());

    PatchFaceTag * levelPatchTags = &patchTags[0];

    for (int i = 0; i < refiner.GetNumLevels(); ++i) {
        Vtr::Level const * level = &refiner.getLevel(i);

        //
        //  Given components at Level[i], we need to be looking at Refinement[i] -- and not
        //  [i-1] -- because the Refinement has transitional information for its parent edges
        //  and faces.  But we also need to be looking at Refinement[i-1] to know about the
        //  ancestry of the components, i.e. are they "complete" wrt their ancestors (if not,
        //  they are supporting components
        //
        //  For components in this level, we want to determine:
        //    - what Edges are "transitional" (already done in Refinement for parent)
        //    - what Faces are "transitional" (already done in Refinement for parent)
        //    - what Faces are "complete" (done for child vertices in Refinement)
        //
        bool isLevelFirst = (i == 0);
        bool isLevelLast  = (i == refiner.GetMaxLevel());

        Vtr::Refinement const * refinePrev = isLevelFirst ? 0 : &refiner.getRefinement(i-1);
        Vtr::Refinement const * refineNext = isLevelLast  ? 0 : &refiner.getRefinement(i);

        Vtr::Refinement::SparseTag const * vtrFaceTags = refineNext ? &refineNext->_parentFaceTag[0] : 0;

        for (int faceIndex = 0; faceIndex < level->getNumFaces(); ++faceIndex) {

            if (level->isHole(faceIndex)) {
                continue;
            }

            Vtr::Refinement::SparseTag vtrFaceTag = vtrFaceTags ? vtrFaceTags[faceIndex] : Vtr::Refinement::SparseTag();
            PatchFaceTag&            patchTag   = levelPatchTags[faceIndex];

            patchTag.clear();
            patchTag._hasPatch = false;

            //
            //  This face does not warrant a patch under the following conditions:
            //
            //      - the face was fully refined into child faces
            //      - the face is not a quad (should have been refined, so assert)
            //      - the face is not "complete"
            //
            //  The first is trivially determined, and the second is really redundant.  The
            //  last -- "incompleteness" -- indicates a face that exists to support the limit
            //  of some neighboring component, and which does not have its own neighborhood
            //  fully defined for its limit.  If any child vertex of a vertex of this face is
            //  "incomplete", the face must be "incomplete" (note all faces in level 0 are
            //  complete and do not warrant closer inspection).
            //
            if (vtrFaceTag._selected) {
                continue;
            }

            Vtr::ConstIndexArray fVerts = level->getFaceVertices(faceIndex);
            assert(fVerts.size() == 4);

            if (!isLevelFirst and (refinePrev->_childVertexTag[fVerts[0]]._incomplete or
                                   refinePrev->_childVertexTag[fVerts[1]]._incomplete or
                                   refinePrev->_childVertexTag[fVerts[2]]._incomplete or
                                   refinePrev->_childVertexTag[fVerts[3]]._incomplete)) {
                continue;
            }

            //
            //  We have a quad that will be represented as a B-spline or Gregory patch.  Use
            //  the "composite" tag for the face that combines tags for all face-verts -- we
            //  can use it to quickly determine if any vertex is irregular or on a boundary.
            //
            //  Inspect the edges for boundaries and transitional edges and pack results into
            //  4-bit masks.  We detect boundary edges rather than vertices as we hope to
            //  replace the mask in future with one for infinitely sharp edges -- allowing
            //  us to detect regular patches and avoid isolation.  We still need to account
            //  for the irregular/xordinary case when a corner vertex is a boundary but there
            //  are no boundary edges.
            //
            //  As for transition detection, assign the transition properties (even if 0) as
            //  their rotations override boundary rotations (when no transition)
            //
            //  NOTE on non-manifold support:
            //      Patches from non-manifold verts are not yet supported -- the extraction
            //  of patch points at corners currently assumes manifold.  Supporting interior
            //  hard edges (below) will allow non-manifold patches with inf sharp boundaries.
            //
            //  NOTE on infinitely sharp (hard) edges:
            //      We should be able to adapt this later to detect hard (inf-sharp) edges
            //  rather than just boundary edges -- there is a similar tag per edge.  That
            //  should allow us to generate regular patches for interior hard features.
            //
            Vtr::Level::VTag compFaceVertTag = level->getFaceCompositeVTag(fVerts);

            //  Patches for non-manifold faces not yet supported (see above note)
            assert(!compFaceVertTag._nonManifold);

            patchTag._hasPatch  = true;
            patchTag._isRegular = not compFaceVertTag._xordinary;

            int boundaryEdgeMask = 0;

            bool hasBoundaryVertex = compFaceVertTag._boundary;

            // single crease patch optimization
            if (options.useSingleCreasePatch and
                not compFaceVertTag._xordinary and not hasBoundaryVertex) {

                Vtr::ConstIndexArray fEdges = level->getFaceEdges(faceIndex);
                Vtr::Level::ETag compFaceETag = level->getFaceCompositeETag(fEdges);

                if (compFaceETag._semiSharp or compFaceETag._infSharp) {
                    float sharpness = 0;
                    int rotation = 0;
                    if (level->isSingleCreasePatch(faceIndex, &sharpness, &rotation)) {

                        // cap sharpness to the max isolation level
                        float cappedSharpness = std::min(sharpness, (float)(options.maxIsolationLevel-i));
                        if (cappedSharpness > 0) {
                            patchTag._isSingleCrease = true;
                            patchTag._boundaryIndex = (rotation + 2) % 4;
                        }
                    }
                }
            }

            if (hasBoundaryVertex) {
                Vtr::ConstIndexArray fEdges = level->getFaceEdges(faceIndex);

                boundaryEdgeMask = ((level->_edgeTags[fEdges[0]]._boundary) << 0) |
                                   ((level->_edgeTags[fEdges[1]]._boundary) << 1) |
                                   ((level->_edgeTags[fEdges[2]]._boundary) << 2) |
                                   ((level->_edgeTags[fEdges[3]]._boundary) << 3);

                if (boundaryEdgeMask) {
                    patchTag.assignBoundaryPropertiesFromEdgeMask(boundaryEdgeMask);
                } else {
                    int boundaryVertMask = ((level->_vertTags[fVerts[0]]._boundary) << 0) |
                                           ((level->_vertTags[fVerts[1]]._boundary) << 1) |
                                           ((level->_vertTags[fVerts[2]]._boundary) << 2) |
                                           ((level->_vertTags[fVerts[3]]._boundary) << 3);

                    patchTag.assignBoundaryPropertiesFromVertexMask(boundaryVertMask);
                }
            }
            patchTag.assignTransitionPropertiesFromEdgeMask(vtrFaceTag._transitional);

            //
            //  This treatment may become optional in future -- consider approximating smooth
            //  corners with regular B-spline patches instead of Gregory.  The smooth corner
            //  must be properly isolated from any other irregular vertices, otherwise the
            //  Gregory patch is necessary.
            //
            bool approxSmoothCornerWithRegularPatch = true;
            if (approxSmoothCornerWithRegularPatch) {
                if (!patchTag._isRegular && (patchTag._boundaryCount == 2)) {
                    //  We may have a sharp corner opposite/adjacent an xordinary vertex --
                    //  need to make sure there is only one xordinary vertex and that it
                    //  is the corner vertex.
                    int xordCorner = 0;
                    int xordCount = 0;
                    if (level->_vertTags[fVerts[0]]._xordinary) { xordCount++; xordCorner = 0; }
                    if (level->_vertTags[fVerts[1]]._xordinary) { xordCount++; xordCorner = 1; }
                    if (level->_vertTags[fVerts[2]]._xordinary) { xordCount++; xordCorner = 2; }
                    if (level->_vertTags[fVerts[3]]._xordinary) { xordCount++; xordCorner = 3; }

                    if (xordCount == 1) {
                        //  The two boundary edges must be either side of the corner vertex:
                        int const expectedCornerEdgeMask[4] = { 8+1, 1+2, 2+4, 4+8 };
                        if (boundaryEdgeMask == expectedCornerEdgeMask[xordCorner]) {
                            patchTag._isRegular = true;
                        }
                    }
                }
            }

            //
            //  Identify and increment counts for regular patches (both non-transitional and
            //  transitional) and extra-ordinary patches (always non-transitional):
            //
            if (patchTag._isRegular) {
                int transIndex = patchTag._transitionType;
                int transRot   = patchTag._transitionRot;

                if (!patchTag._isSingleCrease && patchTag._boundaryCount == 0) {
                    patchInventory.R[transIndex]++;
                } else if (patchTag._isSingleCrease && patchTag._boundaryCount == 0) {
                    patchInventory.S[transIndex][transRot]++;
                } else if (patchTag._boundaryCount == 1) {
                    patchInventory.B[transIndex][transRot]++;
                } else {
                    patchInventory.C[transIndex][transRot]++;
                }
            } else {
                // if end-cap patches use a stencils-driven basis, we don't need
                // to track regular / boundary cases
                if (not options.adaptiveStencilTables) {
                    if (patchTag._boundaryCount == 0) {
                        patchInventory.G++;
                    } else {
                        patchInventory.GB++;
                    }
                } else {
                    patchInventory.GP++;
                }
            }
        }
        levelPatchTags += level->getNumFaces();
    }
}

//
//  Populate all adaptive patches now that the tables to hold data for them have been allocated.
//  We need the inventory (counts per patch type) and the patch tags per face that were previously
//  idenified.
//
void
PatchTablesFactory::populateAdaptivePatches( TopologyRefiner const & refiner,
                                             PatchCounters const &   patchInventory,
                                             PatchTagVector const &  patchTags,
                                             PatchTables *           tables,
                                             Options                 options ) {

    //
    //  Setup convenience pointers at the beginning of each patch array for each
    // table (patches, ptex)
    //
    PatchCVPointers    iptrs;
    PatchParamPointers pptrs;
    PatchFVarPointers  fptrs;
    SharpnessIndexPointers sptrs;

    typedef PatchDescriptorVector DescVec;

    DescVec const & descs = PatchDescriptor::GetAdaptivePatchDescriptors(Sdc::SCHEME_CATMARK);

    for (DescVec::const_iterator it=descs.begin(); it!=descs.end(); ++it) {

        Index arrayIndex = tables->findPatchArray(*it);

        if (arrayIndex==Vtr::INDEX_INVALID) {
            continue;
        }

        iptrs.getValue( *it ) = tables->getPatchArrayVertices(arrayIndex).begin();
        pptrs.getValue( *it ) = tables->getPatchParams(arrayIndex).begin();
        if (patchInventory.hasSingleCreasedPatches()) {
            sptrs.getValue( *it ) = tables->getSharpnessIndices(arrayIndex);
        }

        if (tables->_fvarPatchTables) {
            // XXXX manuelk revisit when implementing bi-cubic fvar interp !!!
            int nchannels = refiner.GetNumFVarChannels();

            Index ** fptr = (Index **)alloca(nchannels*sizeof(Index *));
            for (int channel=0; channel<nchannels; ++channel) {

                fptr[channel] = tables->getFVarVerts(arrayIndex, channel).begin();
            }
            fptrs.getValue( *it ) = fptr;
        }
    }

    unsigned int * quad_G_C0_P = patchInventory.G>0 ? &tables->_quadOffsetsTable[0] : 0,
                 * quad_G_C1_P = patchInventory.GB>0 ? &tables->_quadOffsetsTable[patchInventory.G*4] : 0;

    std::vector<unsigned char> gregoryVertexFlags;

    //
    //  To avoid gathering vertex neighborhoods for all vertices, identify vertices involved in
    //  gregory patches as the faces are traversed, to be gathered later:
    //
    bool hasGregoryPatches = (patchInventory.G > 0) or (patchInventory.GB > 0) or (patchInventory.GP > 0);
    GregoryBasisFactory * gregoryStencilsFactory = 0;
#ifdef ENDCAP_TOPOPOLGY
    int numGregoryBasisVertices=0;
    std::vector<Index> gregoryBasisIndices;
#endif
    if (hasGregoryPatches) {

        StencilTables const * adaptiveStencils = options.adaptiveStencilTables;
        if (adaptiveStencils and patchInventory.GP>0) {

            int maxvalence = refiner.getLevel(0).getMaxValence(),
                npatches = patchInventory.GP;

            gregoryStencilsFactory =
                new GregoryBasisFactory(refiner, *adaptiveStencils, npatches, maxvalence);

#ifdef ENDCAP_TOPOPOLGY
            gregoryBasisIndices.reserve(npatches);
            tables->_endcapTopology.resize(npatches*20);
#endif
        }
        gregoryVertexFlags.resize(refiner.GetNumVerticesTotal(), false);
    }

    //
    //  Now iterate through the faces for all levels and populate the patches:
    //
    int levelFaceOffset = 0;
    int levelVertOffset = 0;
    int * levelFVarVertOffsets = 0;
    if (tables->_fvarPatchTables) {
         levelFVarVertOffsets = (int *)alloca(refiner.GetNumFVarChannels());
         memset(levelFVarVertOffsets, 0, refiner.GetNumFVarChannels()*sizeof(int));
    }

    for (int i = 0; i < refiner.GetNumLevels(); ++i) {
        Vtr::Level const * level = &refiner.getLevel(i);

        const PatchFaceTag * levelPatchTags = &patchTags[levelFaceOffset];

        for (int faceIndex = 0; faceIndex < level->getNumFaces(); ++faceIndex) {

            if (level->isHole(faceIndex)) {
                continue;
            }

            const PatchFaceTag& patchTag = levelPatchTags[faceIndex];
            if (not patchTag._hasPatch) {
                continue;
            }

            if (patchTag._isRegular) {
                Index patchVerts[16];

                int tIndex = patchTag._transitionType;
                int rIndex = patchTag._transitionRot;
                int bIndex = patchTag._boundaryIndex;

                if (!patchTag._isSingleCrease && patchTag._boundaryCount == 0) {
                    int const permuteInterior[16] = { 5, 6, 7, 8, 4, 0, 1, 9, 15, 3, 2, 10, 14, 13, 12, 11 };

                    level->gatherQuadRegularInteriorPatchVertices(faceIndex, patchVerts, rIndex);
                    offsetAndPermuteIndices(patchVerts, 16, levelVertOffset, permuteInterior, iptrs.R[tIndex]);

                    iptrs.R[tIndex] += 16;
                    pptrs.R[tIndex] = computePatchParam(refiner, i, faceIndex, rIndex, pptrs.R[tIndex]);

                    if (tables->_fvarPatchTables) {
                        gatherFVarPatchVertices(refiner, i, faceIndex, rIndex, levelFVarVertOffsets, fptrs.R[tIndex]);
                    }
                } else {
                    //  For the boundary and corner cases, the Hbr code makes some adjustments to the
                    //  rotations here from the way they were defined earlier.  That raises questions
                    //  as to the purpose of the earlier assignments and their naming.  I'd prefer to
                    //  label the sets of rotations for their intended purpose, and to compute and
                    //  assign them earlier for use here with no adjustment.
                    //
                    //  Non-transition case:
                    //      rot = 0;  // outside switch
                    //      f->_adaptiveFlags.brots = (f->_adaptiveFlags.rots + 1) % 4;
                    //  Transition case:
                    //      rot = f->_adaptiveFlags.brots;  //  is this now same as transition rots?
                    //
                    //  Both cases of "rot" above are now handled with the "transition rotation" -- still
                    //  not clear what the purpose of the other is.  Need to look into usage of these
                    //  adaptive-flag rotations in:
                    //      getOneRing, computePatchParam, computeFVarData
                    //  It may be that a separate "face rotation" flag is warranted if we need something
                    //  else dependent on the boundary orientation.
                    //
                    if (patchTag._isSingleCrease && patchTag._boundaryCount==0) {
                        int const permuteInterior[16] = { 5, 6, 7, 8, 4, 0, 1, 9, 15, 3, 2, 10, 14, 13, 12, 11 };
                        level->gatherQuadRegularInteriorPatchVertices(faceIndex, patchVerts, bIndex);
                        offsetAndPermuteIndices(patchVerts, 16, levelVertOffset, permuteInterior, iptrs.S[tIndex][rIndex]);

                        int creaseEdge = (bIndex+2)%4;
                        float sharpness = level->getEdgeSharpness((level->getFaceEdges(faceIndex)[creaseEdge]));
                        sharpness = std::min(sharpness, (float)(options.maxIsolationLevel-i));

                        iptrs.S[tIndex][rIndex] += 16;
                        pptrs.S[tIndex][rIndex] = computePatchParam(refiner, i, faceIndex, bIndex, pptrs.S[tIndex][rIndex]);
                        *sptrs.S[tIndex][rIndex]++ = assignSharpnessIndex(tables, sharpness);

                        if (tables->_fvarPatchTables) {
                            gatherFVarPatchVertices(refiner, i, faceIndex, bIndex, levelFVarVertOffsets, fptrs.S[tIndex][rIndex]);
                        }
                    } else if (patchTag._boundaryCount == 1) {
                        int const permuteBoundary[12] = { 11, 3, 0, 4, 10, 2, 1, 5, 9, 8, 7, 6 };

                        level->gatherQuadRegularBoundaryPatchVertices(faceIndex, patchVerts, bIndex);
                        offsetAndPermuteIndices(patchVerts, 12, levelVertOffset, permuteBoundary, iptrs.B[tIndex][rIndex]);

                        iptrs.B[tIndex][rIndex] += 12;
                        pptrs.B[tIndex][rIndex] = computePatchParam(refiner, i, faceIndex, bIndex, pptrs.B[tIndex][rIndex]);

                        if (tables->_fvarPatchTables) {
                            gatherFVarPatchVertices(refiner, i, faceIndex, bIndex, levelFVarVertOffsets, fptrs.B[tIndex][rIndex]);
                        }
                    } else {
                        int const permuteCorner[9] = { 8, 3, 0, 7, 2, 1, 6, 5, 4 };

                        level->gatherQuadRegularCornerPatchVertices(faceIndex, patchVerts, bIndex);
                        offsetAndPermuteIndices(patchVerts, 9, levelVertOffset, permuteCorner, iptrs.C[tIndex][rIndex]);

                        bIndex = (bIndex+3)%4;

                        iptrs.C[tIndex][rIndex] += 9;
                        pptrs.C[tIndex][rIndex] = computePatchParam(refiner, i, faceIndex, bIndex, pptrs.C[tIndex][rIndex]);

                        if (tables->_fvarPatchTables) {
                            gatherFVarPatchVertices(refiner, i, faceIndex, bIndex, levelFVarVertOffsets, fptrs.C[tIndex][rIndex]);
                        }
                    }
                }
            } else {
                if (gregoryStencilsFactory) {
                    // Gregory basis end-cap (20 CVs - no quad-offsets / valence tables)
                    assert(i==refiner.GetMaxLevel());
                    // Gregory Boundary Patch (4 CVs 0-ring for varying interpolation)
                    Vtr::ConstIndexArray faceVerts = level->getFaceVertices(faceIndex);
                    for (int j = 0; j < 4; ++j) {
                        iptrs.GP[j] = faceVerts[j] + levelVertOffset;
                        gregoryVertexFlags[iptrs.GP[j]] = true;
                    }
                    iptrs.GP += 4;

#ifdef ENDCAP_TOPOPOLGY
                    bool edgeSkip[4];
                    numGregoryBasisVertices = gatherGregoryBasisTopology(*level, faceIndex, numGregoryBasisVertices,
                        levelPatchTags, edgeSkip, gregoryBasisIndices, tables->_endcapTopology);
#endif
                    gregoryStencilsFactory->AddPatchBasis(faceIndex);

                    pptrs.GP = computePatchParam(refiner, i, faceIndex, 0, pptrs.GP);

                    if (tables->_fvarPatchTables) {
                        gatherFVarPatchVertices(refiner, i, faceIndex, 0, levelFVarVertOffsets, fptrs.GP);
                    }
                } else {
                    if (patchTag._boundaryCount == 0) {
                        // Gregory Regular Patch (4 CVs + quad-offsets / valence tables)
                        Vtr::ConstIndexArray faceVerts = level->getFaceVertices(faceIndex);
                        for (int j = 0; j < 4; ++j) {
                            iptrs.G[j] = faceVerts[j] + levelVertOffset;
                            gregoryVertexFlags[iptrs.G[j]] = true;
                        }
                        iptrs.G += 4;

                        getQuadOffsets(*level, faceIndex, quad_G_C0_P);
                        quad_G_C0_P += 4;

                        pptrs.G = computePatchParam(refiner, i, faceIndex, 0, pptrs.G);

                        if (tables->_fvarPatchTables) {
                            gatherFVarPatchVertices(refiner, i, faceIndex, 0, levelFVarVertOffsets, fptrs.G);
                        }
                    } else {
                        // Gregory Boundary Patch (4 CVs + quad-offsets / valence tables)
                        Vtr::ConstIndexArray faceVerts = level->getFaceVertices(faceIndex);
                        for (int j = 0; j < 4; ++j) {
                            iptrs.GB[j] = faceVerts[j] + levelVertOffset;
                            gregoryVertexFlags[iptrs.GB[j]] = true;
                        }
                        iptrs.GB += 4;

                        getQuadOffsets(*level, faceIndex, quad_G_C1_P);
                        quad_G_C1_P += 4;

                        //int bIndex = (patchTag._boundaryIndex+1)%4;

                        pptrs.GB = computePatchParam(refiner, i, faceIndex, 0, pptrs.GB);

                        if (tables->_fvarPatchTables) {
                            gatherFVarPatchVertices(refiner, i, faceIndex, 0, levelFVarVertOffsets, fptrs.GB);
                        }
                    }
                }
            }
        }
        levelFaceOffset += level->getNumFaces();
        levelVertOffset += level->getNumVertices();
        if (tables->_fvarPatchTables) {
            int nchannels = refiner.GetNumFVarChannels();
            for (int channel=0; channel<nchannels; ++channel) {
                levelFVarVertOffsets[channel] += refiner.GetNumFVarValues(i, channel);
            }
        }
    }

    if (gregoryStencilsFactory) {
        tables->_endcapStencilTables =
            gregoryStencilsFactory->CreateStencilTables();
        delete gregoryStencilsFactory;
    }

    //
    //  Now deal with the "vertex valence" table for Gregory patches -- this table contains the one-ring
    //  of vertices around each vertex.  Currently it is extremely wasteful for the following reasons:
    //      - it allocates 2*maxvalence+1 for ALL vertices
    //      - it initializes the one-ring for ALL vertices
    //  We use the full size expected (not sure what else relies on that) but we avoiding initializing
    //  the vast majority of vertices that are not associated with gregory patches -- by having previously
    //  marked those that are associated above and skipping all others.
    //
    if ((patchInventory.G > 0) or (patchInventory.GB > 0)) {
        const int SizePerVertex = 2*tables->_maxValence + 1;

        std::vector<Index> & vTable = tables->_vertexValenceTable;
        vTable.resize(refiner.GetNumVerticesTotal() * SizePerVertex);

        int vOffset = 0;
        int levelLast = refiner.GetMaxLevel();
        for (int i = 0; i <= levelLast; ++i) {

            Vtr::Level const * level = &refiner.getLevel(i);

            if (i == levelLast) {

                int vTableOffset = vOffset * SizePerVertex;

                for (int vIndex = 0; vIndex < level->getNumVertices(); ++vIndex) {
                    int* vTableEntry = &vTable[vTableOffset];

                    //
                    //  If not marked as a vertex of a gregory patch, just set to 0 to ignore.  Otherwise
                    //  gather the one-ring around the vertex and set its resulting size (note the negative
                    //  size used to distinguish between boundary/interior):
                    //
                    //if (!gregoryVertexFlags[vIndex + vOffset]) {
                        vTableEntry[0] = 0;
                    //} else {

                        int * ringDest = vTableEntry + 1,
                              ringSize = level->gatherManifoldVertexRingFromIncidentQuads(vIndex, vOffset, ringDest);

                        if (ringSize & 1) {
                            // boundary vertex : duplicate boundary vertex index
                            // and store negative valence.
                            ringSize++;
                            vTableEntry[ringSize]=vTableEntry[ringSize-1];
                            vTableEntry[0] = -ringSize/2;
                        } else {
                            vTableEntry[0] = ringSize/2;
                        }
                    //}
                    vTableOffset += SizePerVertex;
                }
            }
            vOffset += level->getNumVertices();
        }
    }
}

} // end namespace Far

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv
