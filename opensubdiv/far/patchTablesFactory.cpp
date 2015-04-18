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
#include "../far/topologyRefiner.h"
#include "../vtr/level.h"
#include "../vtr/refinement.h"

#include <algorithm>
#include <cassert>
#include <cstring>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace {
//
//  A convenience container for the different types of feature adaptive patches
//
template <class TYPE>
struct PatchTypes {


    TYPE R,    // regular patch
         B,    // boundary patch (4 rotations)
         C,    // corner patch (4 rotations)
         G,    // gregory patch
         GB,   // gregory boundary patch
         GP;   // gregory basis patch

    PatchTypes() { std::memset(this, 0, sizeof(PatchTypes<TYPE>)); }

    // Returns the number of patches based on the patch type in the descriptor
    TYPE & getValue( Far::PatchDescriptor desc ) {
        switch (desc.GetType()) {
            case Far::PatchDescriptor::REGULAR          : return R;
            case Far::PatchDescriptor::BOUNDARY         : return B;
            case Far::PatchDescriptor::CORNER           : return C;
            case Far::PatchDescriptor::GREGORY          : return G;
            case Far::PatchDescriptor::GREGORY_BOUNDARY : return GB;
            case Far::PatchDescriptor::GREGORY_BASIS    : return GP;
            default : assert(0);
        }
        // can't be reached (suppress compiler warning)
        return R;
    }

    // Counts the number of arrays required to store each type of patch used
    // in the primitive
    int getNumPatchArrays() const {
        int result=0;
        if (R) ++result;
        if (B) ++result;
        if (C) ++result;
        if (G) ++result;
        if (GB) ++result;
        if (GP) ++result;
        return result;
    }
};

typedef PatchTypes<Far::Index *>      PatchCVPointers;
typedef PatchTypes<Far::PatchParam *> PatchParamPointers;
typedef PatchTypes<Far::Index *>      SharpnessIndexPointers;
typedef PatchTypes<Far::Index>        PatchFVarOffsets;
typedef PatchTypes<Far::Index **>     PatchFVarPointers;


} // namespace anon


namespace Far {

void
PatchTablesFactory::PatchFaceTag::clear() {
    std::memset(this, 0, sizeof(*this));
}

void
PatchTablesFactory::PatchFaceTag::assignBoundaryPropertiesFromEdgeMask(int boundaryEdgeMask) {
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
    _boundaryMask = boundaryEdgeMask;

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

void
PatchTablesFactory::PatchFaceTag::assignBoundaryPropertiesFromVertexMask(int boundaryVertexMask) {
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
    _boundaryMask = boundaryVertexMask;

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

//
//  Trivial anonymous helper functions:
//
static inline void
offsetAndPermuteIndices(Far::Index const indices[], int count,
                        Far::Index offset, int const permutation[],
                        Far::Index result[]) {

    if (permutation) {
        for (int i = 0; i < count; ++i) {
            if (permutation[i] < 0) { // XXXdyu-patch-drawing
                result[i] = offset + indices[0]; // XXXdyu-patch-drawing
            } else {
                result[i] = offset + indices[permutation[i]];
            }
        }
    } else if (offset) {
        for (int i = 0; i < count; ++i) {
            result[i] = offset + indices[i];
        }
    } else {
        std::memcpy(result, indices, count * sizeof(Far::Index));
    }
}

//
// Face-varying channel cursor
//
// This cursors allows to iterate over a set of selected face-varying channels.
// If client-code specifies an optional sub-set of the list of channels carried
// by the TopologyRefiner, the cursor can traverse this list and return both its
// current position in the sub-set and the original index of the corresponding
// channel in the TopologyRefiner.
//
class FVarChannelCursor {

public:

    FVarChannelCursor(TopologyRefiner const & refiner,
        PatchTablesFactory::Options options) {
        if (options.generateFVarTables) {
            // If client-code does not select specific channels, default to all
            // the channels in the refiner.
            if (options.numFVarChannels==-1) {
                _numChannels = refiner.GetNumFVarChannels();
                _channelIndices = 0;
            } else {
                assert(options.numFVarChannels<=refiner.GetNumFVarChannels());
                _numChannels = options.numFVarChannels;
                _channelIndices = options.fvarChannelIndices;
            }
        } else {
            _numChannels = 0;
        }
        _currentChannel = this->begin();
    }

    // Increment cursor
    FVarChannelCursor & operator++() {
        ++_currentChannel;
        return *this;
    }

    // Assign a position to a cursor
    FVarChannelCursor & operator = (int currentChannel) {
        _currentChannel = currentChannel;
        return *this;
    }

    // Compare cursor positions
    bool operator != (int pos) {
        return _currentChannel < pos;
    }

    // Return FVar channel index in the TopologyRefiner list
    // XXXX use something better than dereferencing operator maybe ?
    int operator*() {
        assert(_currentChannel<_numChannels);
        // If the cursor is iterating over a sub-set of channels, return the
        // channel index from the sub-set, otherwise use the current cursor
        // position as channel index.
        return _channelIndices ?
            _channelIndices[_currentChannel] : _currentChannel;
    }

    int pos()   { return _currentChannel; }
    int begin() { return 0; }
    int end()   { return _numChannels; }
    int size()  { return _numChannels; }

private:
    int _numChannels,             // total number of channels
        _currentChannel;          // current cursor position
    int const * _channelIndices;  // list of selected channel indices
};

//
// Adaptive Context
//
// Helper class aggregating transient contextual data structures during the
// creation of feature adaptive patch tables. The structure simplifies
// the function prototypes of high-level private methods in the factory.
// This helps keeping the factory class stateless.
//
// Note : struct members are not re-entrant nor are they intended to be !
//
struct PatchTablesFactory::AdaptiveContext {

public:
    AdaptiveContext(TopologyRefiner const & refiner, Options options,
                    EndPatchFactory *endPatchFactory);

    TopologyRefiner const & refiner;

    Options const options;

    // The patch tables being created
    PatchTables * tables;

    EndPatchFactory * endPatchFactory;

public:

    //
    // Vertex
    //

    // Counters accumulating each type of patches during topology traversal
    PatchTypes<int> patchInventory;

    // Bit tags accumulating patch attributes during topology traversal
    PatchTagVector patchTags;

public:

    //
    // Face-varying
    //

    // True if face-varying patches need to be generated for this topology
    bool RequiresFVarPatches() const;

    // A cursor to iterate through the face-varying channels requested
    // by client-code
    FVarChannelCursor fvarChannelCursor;

    // Allocate temporary space to store face-varying values : because we do
    // not know yet the types of each patch, we pre-emptively allocate
    // non-sparse arrays for each channel. Patches are assumed to have a maximum
    // of fvarPatchSize CVs).
    void AllocateFVarPatchValues(int npatches);

    static const int fvarPatchSize = 16;

    // We need temporary storage space to accumulate fvar values as we sort the
    // vertices of the adapative cubic patches. FVar patch types do not match
    // vertex patch types, and unfortunately we cannot generate offsets for a
    // given patch until we have traversed the entire adaptive hierarchy. Instead
    // of incurring another full hierarchy traversal, we store the FVar values
    // in a temporary array with patches of fixed size. Once the values have been
    // populated (in the correct sorted order), we copy them in the final sparse
    // vectors and generate offsets.
    std::vector<std::vector<Index> > fvarPatchValues;
};

// Constructor
PatchTablesFactory::AdaptiveContext::AdaptiveContext(
    TopologyRefiner const & ref, Options opts,
    EndPatchFactory * endPatchFactory) :
    refiner(ref), options(opts), tables(0),
    endPatchFactory(endPatchFactory),
    fvarChannelCursor(ref, opts) {

    fvarPatchValues.resize(fvarChannelCursor.size());
}

void
PatchTablesFactory::AdaptiveContext::AllocateFVarPatchValues(int npatches) {

    FVarChannelCursor & fvc = fvarChannelCursor;
    for (fvc=fvc.begin(); fvc!=fvc.end(); ++fvc) {

        Sdc::Options::FVarLinearInterpolation interpolation =
            refiner.GetFVarLinearInterpolation(*fvc);

        // the LINEAR_ALL rule can populate values immediately (all quads) so
        // we do not need this temporary storage
        if (interpolation != Sdc::Options::FVAR_LINEAR_ALL) {
            fvarPatchValues[fvc.pos()].resize(npatches*fvarPatchSize);
        }
    }
}

bool
PatchTablesFactory::AdaptiveContext::RequiresFVarPatches() const {
    return not fvarPatchValues.empty();
}

//
//  Reserves tables based on the contents of the PatchArrayVector in the PatchTables:
//
void
PatchTablesFactory::allocateVertexTables(PatchTables * tables, int /* nlevels */, bool hasSharpness) {

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

//
//  Allocate face-varying tables
//
void
PatchTablesFactory::allocateFVarChannels(TopologyRefiner const & refiner,
    Options options, int npatches, PatchTables * tables) {

    assert(options.generateFVarTables and
        refiner.GetNumFVarChannels()>0 and npatches>0 and tables);

    // Create a channel cursor to iterate over client-selected channels or
    // default to the channels found in the TopologyRefiner
    FVarChannelCursor fvc(refiner, options);
    if (fvc.size()==0) {
        return;
    }

    tables->allocateFVarPatchChannels(fvc.size());

    // Iterate with the cursor to initialize each channel
    for (fvc=fvc.begin(); fvc!=fvc.end(); ++fvc) {

        Sdc::Options::FVarLinearInterpolation interpolation =
            refiner.GetFVarLinearInterpolation(*fvc);

        tables->setFVarPatchChannelLinearInterpolation(fvc.pos(), interpolation);

        int nverts = 0;
        if (interpolation==Sdc::Options::FVAR_LINEAR_ALL) {

            PatchDescriptor::Type type = options.triangulateQuads ?
                PatchDescriptor::TRIANGLES : PatchDescriptor::QUADS;

            tables->setFVarPatchChannelPatchesType(fvc.pos(), type);

            nverts =
                npatches * PatchDescriptor::GetNumFVarControlVertices(type);

        }
        tables->allocateChannelValues(fvc.pos(), npatches, nverts);
    }
}


// gather face-varying patch points
inline int
PatchTablesFactory::gatherFVarData(AdaptiveContext & context, int level,
    Index faceIndex, Index levelFaceOffset, int rotation,
        Index const * levelFVarVertOffsets, Index fofss, Index ** fptrs) {

    if (not context.RequiresFVarPatches()) {
        return 0;
    }

    TopologyRefiner const & refiner = context.refiner;

    PatchTables * tables = context.tables;

    assert((levelFaceOffset + faceIndex)<(int)context.patchTags.size());
    PatchFaceTag & vertexPatchTag = context.patchTags[levelFaceOffset + faceIndex];

    Index patchVerts[context.fvarPatchSize];

    // Iterate over valid FVar channels (if any)
    FVarChannelCursor & fvc = context.fvarChannelCursor;
    for (fvc=fvc.begin(); fvc!=fvc.end(); ++fvc) {

        Vtr::Level const & vtxLevel = refiner.getLevel(level);
        Vtr::FVarLevel const & fvarLevel = vtxLevel.getFVarLevel(*fvc);

        if (refiner.GetFVarLinearInterpolation(*fvc)!=Sdc::Options::FVAR_LINEAR_ALL) {

            //
            // Bi-cubic patches
            //

            //  If the face-varying topology matches the vertex topology (which should be the
            //  dominant case), we can use the patch tag for the original vertex patch --
            //  quickly check the composite tag for the face-varying values at the corners:
            //
            PatchFaceTag fvarPatchTag = vertexPatchTag;

            ConstIndexArray faceVerts = vtxLevel.getFaceVertices(faceIndex),
                            fvarValues = fvarLevel.getFaceValues(faceIndex);

            Vtr::FVarLevel::ValueTag compFVarTagsForFace =
                fvarLevel.getFaceCompositeValueTag(fvarValues, faceVerts);

            if (compFVarTagsForFace.isMismatch()) {

                //  At least one of the corner vertices has differing topology in FVar space,
                //  so we need to perform similar analysis to what was done to determine the
                //  face's original patch tag to determine the face-varying patch tag here.
                //
                //  Recall how that patch tag is initialized:
                //      - a "composite" (bitwise-OR) tag of the face's VTags is taken
                //      - if determined to be on a boundary, a "boundary mask" is built and
                //        passed to the PatchFaceTag to determine boundary orientation
                //      - when necessary, a "composite" tag for the face's ETags is inspected
                //      - special case for "single-crease patch"
                //      - special case for "approx smooth corner with regular patch"
                //
                //  Note differences here (simplifications):
                //      - we don't need to deal with the single-crease patch case:
                //          - if vertex patch was single crease the mismatching FVar patch
                //            cannot be
                //          - the fvar patch cannot become single-crease patch as only sharp
                //            (discts) edges are introduced, which are now boundary edges
                //      - the "approx smooth corner with regular patch" case was ignored:
                //          - its unclear if it should persist for the vertex patch
                //
                //  As was the case with the vertex patch, since we are creating a patch it
                //  is assumed that all required isolation has occurred.  For example, a
                //  regular patch at level 0 that has a FVar patch with too many boundaries
                //  (or local xordinary vertices) is going to cause trouble here...
                //

                //
                //  Gather the VTags for the four corners of the FVar patch (these are the VTag
                //  of each vertex merged with the FVar tag of its value) while computing the
                //  composite VTag:
                //
                Vtr::Level::VTag fvarVertTags[4];

                Vtr::Level::VTag compFVarVTag =
                            fvarLevel.getFaceCompositeValueAndVTag(fvarValues, faceVerts, fvarVertTags);

                //
                //  Clear/re-initialize the FVar patch tag and compute the appropriate boundary
                //  masks if boundary orientation is necessary:
                //
                fvarPatchTag.clear();
                fvarPatchTag._hasPatch  = true;
                fvarPatchTag._isRegular = not compFVarVTag._xordinary;

                if (compFVarVTag._boundary) {
                    Vtr::Level::ETag fvarEdgeTags[4];

                    ConstIndexArray faceEdges = vtxLevel.getFaceEdges(faceIndex);

                    Vtr::Level::ETag compFVarETag =
                                fvarLevel.getFaceCompositeCombinedEdgeTag(faceEdges, fvarEdgeTags);

                    if (compFVarETag._boundary) {
                        int boundaryEdgeMask = (fvarEdgeTags[0]._boundary << 0) |
                                               (fvarEdgeTags[1]._boundary << 1) |
                                               (fvarEdgeTags[2]._boundary << 2) |
                                               (fvarEdgeTags[3]._boundary << 3);

                        fvarPatchTag.assignBoundaryPropertiesFromEdgeMask(boundaryEdgeMask);
                    } else {
                        int boundaryVertMask = (fvarVertTags[0]._boundary << 0) |
                                               (fvarVertTags[1]._boundary << 1) |
                                               (fvarVertTags[2]._boundary << 2) |
                                               (fvarVertTags[3]._boundary << 3);

                        fvarPatchTag.assignBoundaryPropertiesFromVertexMask(boundaryVertMask);
                    }
                }
            }

            //
            //  Determine and assign the type of the patch
            //
            PatchDescriptor::Type fvarPatchType = PatchDescriptor::REGULAR;
            if (not fvarPatchTag._isRegular) {
                // because we do not want to have to generate vertex-valence
                // & quad-offset tables for each fvar channel, we default to
                // Gregory-basis type patchs only (and use stencils to
                // compute the 20 cvs basis)
                fvarPatchType = context.options.useFVarQuadEndCaps ?
                    PatchDescriptor::QUADS : PatchDescriptor::GREGORY_BASIS;
            } else if (fvarPatchTag._boundaryCount > 1) {
                fvarPatchType = PatchDescriptor::CORNER;
            } else if (fvarPatchTag._boundaryCount == 1) {
                fvarPatchType = PatchDescriptor::BOUNDARY;
            } else if (fvarPatchTag._isSingleCrease) {
                fvarPatchType = PatchDescriptor::REGULAR;
            }

            Vtr::Array<PatchDescriptor::Type> patchTypes =
                tables->getFVarPatchTypes(fvc.pos());
            assert(not patchTypes.empty());
            patchTypes[fofss] = fvarPatchType;


            int const * permutation = 0;

            //  Gather the verts FVar values
            //     XXXX Patch verts should be rotated to match boundary / corner
            //     edges. Transition patterns should not be a concern, however
            //     we need to match parametric space, so this may need to be
            //     revisited...
            int orientationIndex = fvarPatchTag._boundaryIndex;
            if (fvarPatchType == PatchDescriptor::REGULAR) {
                static int const permuteRegular[16] = { 5, 6, 7, 8, 4, 0, 1, 9, 15, 3, 2, 10, 14, 13, 12, 11 };
                vtxLevel.gatherQuadRegularInteriorPatchPoints(faceIndex, patchVerts, orientationIndex, *fvc);
                permutation = permuteRegular;
            } else if (fvarPatchType == PatchDescriptor::CORNER) {
                static int const permuteCorner[9] = { 8, 3, 0, 7, 2, 1, 6, 5, 4 };
                vtxLevel.gatherQuadRegularCornerPatchPoints(faceIndex, patchVerts, orientationIndex, *fvc);
                permutation = permuteCorner;
            } else if (fvarPatchType == PatchDescriptor::BOUNDARY) {
                static int const permuteBoundary[12] = { 11, 3, 0, 4, 10, 2, 1, 5, 9, 8, 7, 6 };
                vtxLevel.gatherQuadRegularBoundaryPatchPoints(faceIndex, patchVerts, orientationIndex, *fvc);
                permutation = permuteBoundary;
            } else if (fvarPatchType == PatchDescriptor::QUADS) {
                vtxLevel.gatherQuadLinearPatchPoints(faceIndex, patchVerts, orientationIndex, *fvc);
                permutation = 0;
            } else if (fvarPatchType == PatchDescriptor::GREGORY_BASIS) {
                // XXXX
                // Gregory basis patch : we need to gather the vertices and
                // generate the stencil. We can use the index in the vertex
                // patch array to index the stencils.
                assert(0);
            } else {
                // note : we do not plan on supporting direct evaluation types
                // of Gregory patches, because they requre extremely inefficient
                // quad-offset and vertex-valence data structures.
                assert(0);
            }

            int nverts = PatchDescriptor::GetNumFVarControlVertices(fvarPatchType);
            assert(nverts <= context.fvarPatchSize);

            offsetAndPermuteIndices(patchVerts, nverts, levelFVarVertOffsets[fvc.pos()],
                permutation, &context.fvarPatchValues[fvc.pos()][fofss*context.fvarPatchSize]);
        } else {

            //
            // Bi-linear patches
            //

            ConstIndexArray fvarValues = fvarLevel.getFaceValues(faceIndex);

            // Store verts values directly in non-sparse context channel arrays
            for (int vert=0; vert<fvarValues.size(); ++vert) {
                fptrs[fvc.pos()][vert] =
                    levelFVarVertOffsets[fvc.pos()] + fvarValues[(vert+rotation)%4];
            }
            fptrs[fvc.pos()]+=fvarValues.size();
        }
    }
    return 1;
}

//
//  Populates the face-varying data buffer 'coord' for the given face, returning
//  a pointer to the next descriptor
//
PatchParam *
PatchTablesFactory::computePatchParam(
        TopologyRefiner const & refiner, int depth, Vtr::Index faceIndex,
        int rotation, int boundaryMask, int transitionMask, PatchParam *coord) {

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

    boundaryMask = ((((boundaryMask << 4) | boundaryMask) >> rotation)) & 0xf;
    transitionMask = ((((transitionMask << 4) | transitionMask) >> rotation)) & 0xf;

    coord->Set(ptexIndex, (short)u, (short)v, (unsigned char) depth, nonquad,
               (unsigned short) boundaryMask, (unsigned short) transitionMask);

    return ++coord;
}


//
//  Indexing sharpnesses
//
inline int
assignSharpnessIndex(float sharpness, std::vector<float> & sharpnessValues) {

    // linear search
    for (int i=0; i<(int)sharpnessValues.size(); ++i) {
        if (sharpnessValues[i] == sharpness) {
            return i;
        }
    }
    sharpnessValues.push_back(sharpness);
    return (int)sharpnessValues.size()-1;
}

//
//  We should be able to use a single Create() method for both the adaptive and uniform
//  cases.  In the past, more additional arguments were passed to the uniform version,
//  but that may no longer be necessary (see notes in the uniform version below)...
//
PatchTables *
PatchTablesFactory::Create( TopologyRefiner const & refiner, Options options,
                            EndPatchFactory *endPatchFactory) {

    if (refiner.IsUniform()) {
        return createUniform(refiner, options);
    } else {
        return createAdaptive(refiner, options, endPatchFactory);
    }
}

PatchTables *
PatchTablesFactory::createUniform(TopologyRefiner const & refiner, Options options) {

    assert(refiner.IsUniform());

    // ensure that triangulateQuads is only set for quadrilateral schemes
    options.triangulateQuads &= (refiner.GetSchemeType()==Sdc::SCHEME_BILINEAR or
                                 refiner.GetSchemeType()==Sdc::SCHEME_CATMARK);

    int maxvalence = refiner.GetMaxValence(),
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

    PatchDescriptor desc(ptype);

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
    allocateVertexTables( tables, 0, /*hasSharpness=*/false );

    bool generateFVarPatches=false;
    FVarChannelCursor fvc(refiner, options);
    if (options.generateFVarTables and fvc.size()>0) {
        int npatches = tables->GetNumPatchesTotal();
        allocateFVarChannels(refiner, options, npatches, tables);
        assert(fvc.size() == tables->GetNumFVarChannels());
    }

    //
    //  Now populate the patches:
    //

    Index          * iptr = &tables->_patchVerts[0];
    PatchParam     * pptr = &tables->_paramTable[0];
    Index         ** fptr = 0;

    Index levelVertOffset = options.generateAllLevels ?
        0 : refiner.GetNumVertices(0);

    Index * levelFVarVertOffsets = 0;
    if (generateFVarPatches) {

        levelFVarVertOffsets = (Index *)alloca(fvc.size()*sizeof(Index));
        memset(levelFVarVertOffsets, 0, fvc.size()*sizeof(Index));

        fptr = (Index **)alloca(fvc.size()*sizeof(Index *));
        for (fvc=fvc.begin(); fvc!=fvc.end(); ++fvc) {
            fptr[fvc.pos()] = tables->getFVarPatchesValues(fvc.pos()).begin();
        }
    }

    for (int level=1; level<=maxlevel; ++level) {

        int nfaces = refiner.GetNumFaces(level);
        if (level>=firstlevel) {
            for (int face=0; face<nfaces; ++face) {

                if (refiner.HasHoles() and refiner.IsFaceHole(level, face)) {
                    continue;
                }

                ConstIndexArray fverts = refiner.GetFaceVertices(level, face);
                for (int vert=0; vert<fverts.size(); ++vert) {
                    *iptr++ = levelVertOffset + fverts[vert];
                }

                pptr = computePatchParam(refiner, level, face, /*rot*/0, /*boundary*/0, /*transition*/0, pptr);

                if (generateFVarPatches) {
                    for (fvc=fvc.begin(); fvc!=fvc.end(); ++fvc) {
                        ConstIndexArray fvalues = refiner.GetFVarFaceValues(level, face, *fvc);
                        for (int vert=0; vert<fvalues.size(); ++vert) {
                            assert((levelVertOffset + fvalues[vert]) < (int)tables->getFVarPatchesValues(fvc.pos()).size());
                            fptr[fvc.pos()][vert] = levelFVarVertOffsets[fvc.pos()] + fvalues[vert];
                        }
                        fptr[fvc.pos()]+=fvalues.size();
                    }
                }

                if (options.triangulateQuads) {
                    // Triangulate the quadrilateral: {v0,v1,v2,v3} -> {v0,v1,v2},{v3,v0,v2}.
                    *iptr = *(iptr - 4); // copy v0 index
                    ++iptr;
                    *iptr = *(iptr - 3); // copy v2 index
                    ++iptr;

                    *pptr = *(pptr - 1); // copy first patch param
                    ++pptr;

                    if (generateFVarPatches) {
                        for (fvc=fvc.begin(); fvc!=fvc.end(); ++fvc) {
                            *fptr[fvc.pos()] = *(fptr[fvc.pos()]-4); // copy fv0 index
                            ++fptr[fvc.pos()];
                            *fptr[fvc.pos()] = *(fptr[fvc.pos()]-3); // copy fv2 index
                            ++fptr[fvc.pos()];
                        }
                    }
                }
            }
        }

        if (options.generateAllLevels) {
            levelVertOffset += refiner.GetNumVertices(level);
            for (fvc=fvc.begin(); fvc!=fvc.end(); ++fvc) {
                levelFVarVertOffsets[fvc.pos()] += refiner.GetNumFVarValues(level, fvc.pos());
            }
        }
    }
    return tables;
}

PatchTables *
PatchTablesFactory::createAdaptive(TopologyRefiner const & refiner, Options options,
                                   EndPatchFactory *endPatchFactory) {

    assert(not refiner.IsUniform());

    AdaptiveContext context(refiner, options, endPatchFactory);

    //
    //  First identify the patches -- accumulating the inventory patches for all of the
    //  different types and information about the patch for each face:
    //

    identifyAdaptivePatches(context);

    //
    //  Create the instance of the tables and allocate and initialize its members based on
    //  the inventory of patches determined above:
    //
    int maxValence = refiner.GetMaxValence();

    context.tables = new PatchTables(maxValence);

    // Populate the patch array descriptors
    context.tables->reservePatchArrays(context.patchInventory.getNumPatchArrays());

    // Sort through the inventory and push back non-empty patch arrays
    ConstPatchDescriptorArray const & descs =
        PatchDescriptor::GetAdaptivePatchDescriptors(Sdc::SCHEME_CATMARK);

    int voffset=0, poffset=0, qoffset=0;
    for (int i=0; i<descs.size(); ++i) {
        PatchDescriptor desc = descs[i];
        context.tables->pushPatchArray(desc,
            context.patchInventory.getValue(desc), &voffset, &poffset, &qoffset );
    }

    context.tables->_numPtexFaces = refiner.GetNumPtexFaces();

    // Allocate various tables
    bool hasSharpness = context.options.useSingleCreasePatch;
    allocateVertexTables(context.tables, 0, hasSharpness);

    if (context.RequiresFVarPatches()) {

        int npatches = context.tables->GetNumPatchesTotal();

        allocateFVarChannels(refiner, options, npatches, context.tables);

        // Reserve temporary non-sparse storage for non-linear fvar channels.
        // FVar Values for these channels are copied into the final
        // FVarPatchChannel after the second traversal happens within the call to
        // populateAdaptivePatches()
        context.AllocateFVarPatchValues(npatches);
    }

    //
    //  Now populate the patches:
    //
    populateAdaptivePatches(context);

    return context.tables;
}

//
//  Identify all patches required for faces at all levels -- accumulating the number of patches
//  for each type, and retaining enough information for the patch for each face to populate it
//  later with no additional analysis.
//
void
PatchTablesFactory::identifyAdaptivePatches(AdaptiveContext & context) {

    TopologyRefiner const & refiner = context.refiner;

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
    context.patchTags.resize(refiner.GetNumFacesTotal());

    PatchFaceTag * levelPatchTags = &context.patchTags[0];

    for (int levelIndex = 0; levelIndex < refiner.GetNumLevels(); ++levelIndex) {
        Vtr::Level const * level = &refiner.getLevel(levelIndex);

        //
        //  Given components at Level[i], we need to be looking at Refinement[i] -- and not
        //  [i-1] -- because the Refinement has transitional information for its parent edges
        //  and faces.
        //
        //  For components in this level, we want to determine:
        //    - what Edges are "transitional" (already done in Refinement for parent)
        //    - what Faces are "transitional" (already done in Refinement for parent)
        //    - what Faces are "complete" (applied to this Level in previous refinement)
        //
        Vtr::Refinement const            * refinement = 0;
        Vtr::Refinement::SparseTag const * refinedFaceTags = 0;

        if (levelIndex < refiner.GetMaxLevel()) {
            refinement      = &refiner.getRefinement(levelIndex);
            refinedFaceTags = &refinement->_parentFaceTag[0];
        }

        for (int faceIndex = 0; faceIndex < level->getNumFaces(); ++faceIndex) {

            PatchFaceTag & patchTag = levelPatchTags[faceIndex];
            patchTag.clear();
            patchTag._hasPatch = false;

            if (level->isFaceHole(faceIndex)) {
                continue;
            }

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
            //  "incomplete" (and all are tagged) the face must be "incomplete", so get the
            //  "composite" tag which combines bits for all vertices:
            //
            Vtr::Refinement::SparseTag refinedFaceTag = refinedFaceTags ?
                refinedFaceTags[faceIndex] : Vtr::Refinement::SparseTag();

            if (refinedFaceTag._selected) {
                continue;
            }

            Vtr::ConstIndexArray fVerts = level->getFaceVertices(faceIndex);
            assert(fVerts.size() == 4);

            Vtr::Level::VTag compFaceVertTag = level->getFaceCompositeVTag(fVerts);
            if (compFaceVertTag._incomplete) {
                continue;
            }

            //
            //  We have a quad that will be represented as a B-spline or Gregory patch.  Use
            //  the "composite" tag again to quickly determine if any vertex is irregular, on
            //  a boundary, non-manifold, etc.
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
            //  NOTE on patches around non-manifold vertices:
            //      In most the use of regular boundary or corner patches is what we want,
            //  but in some, i.e. when a non-manifold vertex is infinitely sharp, using
            //  such patches will create some discontinuities.  At this point non-manifold
            //  support is still evolving and is not strictly defined, so this is left to
            //  a later date to resolve.
            //
            //  NOTE on infinitely sharp (hard) edges:
            //      We should be able to adapt this later to detect hard (inf-sharp) edges
            //  rather than just boundary edges -- there is a similar tag per edge.  That
            //  should allow us to generate regular patches for interior hard features.
            //
            bool hasBoundaryVertex    = compFaceVertTag._boundary;
            bool hasNonManifoldVertex = compFaceVertTag._nonManifold;
            bool hasXOrdinaryVertex   = compFaceVertTag._xordinary;

            patchTag._hasPatch  = true;
            patchTag._isRegular = not hasXOrdinaryVertex or hasNonManifoldVertex;

            // single crease patch optimization
            if (context.options.useSingleCreasePatch and
                not hasXOrdinaryVertex and not hasBoundaryVertex and not hasNonManifoldVertex) {

                Vtr::ConstIndexArray fEdges = level->getFaceEdges(faceIndex);
                Vtr::Level::ETag compFaceETag = level->getFaceCompositeETag(fEdges);

                if (compFaceETag._semiSharp or compFaceETag._infSharp) {
                    float sharpness = 0;
                    int rotation = 0;
                    if (level->isSingleCreasePatch(faceIndex, &sharpness, &rotation)) {

                        // cap sharpness to the max isolation level
                        float cappedSharpness =
                                std::min(sharpness, (float)(context.options.maxIsolationLevel - levelIndex));
                        if (cappedSharpness > 0) {
                            patchTag._isSingleCrease = true;
                            patchTag._boundaryIndex = (rotation + 2) % 4;
                        }
                    }
                }
            }

            //  Identify boundaries for both regular and xordinary patches -- non-manifold
            //  edges and vertices are interpreted as boundaries for regular patches
            if (hasBoundaryVertex or hasNonManifoldVertex) {
                Vtr::ConstIndexArray fEdges = level->getFaceEdges(faceIndex);

                int boundaryEdgeMask = ((level->_edgeTags[fEdges[0]]._boundary) << 0) |
                                       ((level->_edgeTags[fEdges[1]]._boundary) << 1) |
                                       ((level->_edgeTags[fEdges[2]]._boundary) << 2) |
                                       ((level->_edgeTags[fEdges[3]]._boundary) << 3);
                if (hasNonManifoldVertex) {
                    int nonManEdgeMask = ((level->_edgeTags[fEdges[0]]._nonManifold) << 0) |
                                         ((level->_edgeTags[fEdges[1]]._nonManifold) << 1) |
                                         ((level->_edgeTags[fEdges[2]]._nonManifold) << 2) |
                                         ((level->_edgeTags[fEdges[3]]._nonManifold) << 3);
                    boundaryEdgeMask |= nonManEdgeMask;
                }

                if (boundaryEdgeMask) {
                    patchTag.assignBoundaryPropertiesFromEdgeMask(boundaryEdgeMask);
                } else {
                    int boundaryVertMask = ((level->_vertTags[fVerts[0]]._boundary) << 0) |
                                           ((level->_vertTags[fVerts[1]]._boundary) << 1) |
                                           ((level->_vertTags[fVerts[2]]._boundary) << 2) |
                                           ((level->_vertTags[fVerts[3]]._boundary) << 3);

                    if (hasNonManifoldVertex) {
                        int nonManVertMask = ((level->_vertTags[fVerts[0]]._nonManifold) << 0) |
                                             ((level->_vertTags[fVerts[1]]._nonManifold) << 1) |
                                             ((level->_vertTags[fVerts[2]]._nonManifold) << 2) |
                                             ((level->_vertTags[fVerts[3]]._nonManifold) << 3);
                        boundaryVertMask |= nonManVertMask;
                    }
                    patchTag.assignBoundaryPropertiesFromVertexMask(boundaryVertMask);
                }
            }

            //  XXXX (barfowl) -- why are we approximating a smooth x-ordinary corner with
            //  a sharp corner patch?  The boundary/corner points of the regular patch are
            //  not even made colinear to make it smoother.  Something historical here...
            //
            //  So this treatment may become optional in future and is bracketed with a
            //  condition now for that reason.  We approximate x-ordinary smooth corners
            //  with regular B-spline patches instead of using a Gregory patch.  The smooth
            //  corner must be properly isolated from any other irregular vertices (as it
            //  will be at any level > 1) otherwise the Gregory patch is necessary.
            //
            //  This flag to be initialized with a future option... ?
            bool approxSmoothCornerWithRegularPatch = true;

            if (approxSmoothCornerWithRegularPatch) {
                if (!patchTag._isRegular and (patchTag._boundaryCount == 2)) {
                    //  We may have a sharp corner opposite/adjacent an xordinary vertex --
                    //  need to make sure there is only one xordinary vertex and that it
                    //  is the corner vertex.
                    if (levelIndex > 1) {
                        patchTag._isRegular = true;
                    } else {
                        int xordVertex = 0;
                        int xordCount = 0;
                        if (level->_vertTags[fVerts[0]]._xordinary) { xordCount++; xordVertex = 0; }
                        if (level->_vertTags[fVerts[1]]._xordinary) { xordCount++; xordVertex = 1; }
                        if (level->_vertTags[fVerts[2]]._xordinary) { xordCount++; xordVertex = 2; }
                        if (level->_vertTags[fVerts[3]]._xordinary) { xordCount++; xordVertex = 3; }

                        if (xordCount == 1) {
                            //  We require the vertex opposite the xordinary vertex be interior:
                            if (not level->_vertTags[fVerts[(xordVertex + 2) % 4]]._boundary) {
                                patchTag._isRegular = true;
                            }
                        }
                    }
                }
            }

            //
            //  Now that all boundary features have have been identified and tagged, assign
            //  the transition type for the patch before taking inventory.
            //
            //  Identify and increment counts for regular patches (both non-transitional and
            //  transitional) and extra-ordinary patches (always non-transitional):
            //
            patchTag.assignTransitionPropertiesFromEdgeMask(refinedFaceTag._transitional);

            if (patchTag._isRegular) {

                if (patchTag._boundaryCount == 0) {
                    context.patchInventory.R++;
                } else if (patchTag._boundaryCount == 1) {
                    context.patchInventory.R++;
                } else {
                    context.patchInventory.R++;
                }
            } else {
                // endcap process is delegated to the endpatch factory
                if (context.endPatchFactory) {
                    switch(context.endPatchFactory->GetPatchType(patchTag)) {
                    case Far::PatchDescriptor::REGULAR:
                        context.patchInventory.R++; break;
                    case Far::PatchDescriptor::GREGORY:
                        context.patchInventory.G++; break;
                    case Far::PatchDescriptor::GREGORY_BOUNDARY:
                        context.patchInventory.GB++; break;
                    case Far::PatchDescriptor::GREGORY_BASIS:
                        context.patchInventory.GP++; break;
                    default:
                        // error?
                        break;
                    }
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
PatchTablesFactory::populateAdaptivePatches(AdaptiveContext & context) {

    TopologyRefiner const & refiner = context.refiner;

    PatchTables * tables = context.tables;

    //
    //  Setup convenience pointers at the beginning of each patch array for each
    // table (patches, ptex)
    //
    PatchCVPointers    iptrs;
    PatchParamPointers pptrs;
    PatchFVarOffsets   fofss;
    PatchFVarPointers  fptrs;
    SharpnessIndexPointers sptrs;

    ConstPatchDescriptorArray const & descs =
        PatchDescriptor::GetAdaptivePatchDescriptors(Sdc::SCHEME_CATMARK);

    for (int i=0; i<descs.size(); ++i) {

        PatchDescriptor desc = descs[i];

        Index arrayIndex = tables->findPatchArray(desc);

        if (arrayIndex==Vtr::INDEX_INVALID) {
            continue;
        }

        iptrs.getValue(desc) = tables->getPatchArrayVertices(arrayIndex).begin();
        pptrs.getValue(desc) = tables->getPatchParams(arrayIndex).begin();
        if (context.options.useSingleCreasePatch) {
            sptrs.getValue(desc) = tables->getSharpnessIndices(arrayIndex);
        }

        if (context.RequiresFVarPatches()) {

            Index & offsets = fofss.getValue(desc);
            offsets = tables->getPatchIndex(arrayIndex, 0);

            // XXXX manuelk this stuff will go away as we use offsets from FVarPatchChannel
            FVarChannelCursor & fvc = context.fvarChannelCursor;
            assert(fvc.size() == tables->GetNumFVarChannels());

            Index ** fptr = (Index **)alloca(fvc.size()*sizeof(Index *));
            for (fvc=fvc.begin(); fvc!=fvc.end(); ++fvc) {

                Index pidx = tables->getPatchIndex(arrayIndex, 0);
                int ofs = pidx * 4;
                fptr[fvc.pos()] = &tables->getFVarPatchesValues(fvc.pos())[ofs];
            }
            fptrs.getValue(desc) = fptr;
        }
    }

    //
    //  Now iterate through the faces for all levels and populate the patches:
    //
    int levelFaceOffset = 0,
        levelVertOffset = 0;
    int * levelFVarVertOffsets = 0;
    if (context.RequiresFVarPatches()) {
         int nchannels = refiner.GetNumFVarChannels();
         levelFVarVertOffsets = (int *)alloca(nchannels);
         memset(levelFVarVertOffsets, 0, nchannels*sizeof(int));
    }

    for (int i = 0; i < refiner.GetNumLevels(); ++i) {
        Vtr::Level const * level = &refiner.getLevel(i);

        const PatchFaceTag * levelPatchTags = &context.patchTags[levelFaceOffset];

        for (int faceIndex = 0; faceIndex < level->getNumFaces(); ++faceIndex) {

            if (level->isFaceHole(faceIndex)) {
                continue;
            }

            const PatchFaceTag& patchTag = levelPatchTags[faceIndex];
            if (not patchTag._hasPatch) {
                continue;
            }

            if (patchTag._isRegular) {
                Index patchVerts[16];

                int bIndex = patchTag._boundaryIndex;
                int boundaryMask = patchTag._boundaryMask;
                int transitionMask = patchTag._transitionMask;

                if (!patchTag._isSingleCrease and patchTag._boundaryCount == 0) {
                    int const permuteInterior[16] = { 5, 6, 7, 8, 4, 0, 1, 9, 15, 3, 2, 10, 14, 13, 12, 11 };

                    level->gatherQuadRegularInteriorPatchPoints(faceIndex, patchVerts, /*rotation*/0);
                    offsetAndPermuteIndices(patchVerts, 16, levelVertOffset, permuteInterior, iptrs.R);

                    iptrs.R += 16;
                    pptrs.R = computePatchParam(refiner, i, faceIndex, /*rotation*/0, /*boundary*/0, transitionMask, pptrs.R);
                    // XXX: sharpness will be integrated into patch param soon.
                    if (sptrs.R) *sptrs.R++ = assignSharpnessIndex(0, tables->_sharpnessValues);

                    fofss.R += gatherFVarData(context,
                        i, faceIndex, levelFaceOffset, /*rotation*/0, levelFVarVertOffsets, fofss.R, fptrs.R);
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
                    if (patchTag._isSingleCrease and patchTag._boundaryCount==0) {
                        int const permuteInterior[16] = { 5, 6, 7, 8, 4, 0, 1, 9, 15, 3, 2, 10, 14, 13, 12, 11 };
                        level->gatherQuadRegularInteriorPatchPoints(faceIndex, patchVerts, bIndex);
                        offsetAndPermuteIndices(patchVerts, 16, levelVertOffset, permuteInterior, iptrs.R);

                        int creaseEdge = (bIndex+2)%4;
                        float sharpness = level->getEdgeSharpness((level->getFaceEdges(faceIndex)[creaseEdge]));
                        sharpness = std::min(sharpness, (float)(context.options.maxIsolationLevel-i));

                        iptrs.R += 16;
                        pptrs.R = computePatchParam(refiner, i, faceIndex, bIndex, /*boundary*/0, transitionMask, pptrs.R);
                        if (sptrs.R) *sptrs.R++ = assignSharpnessIndex(sharpness, tables->_sharpnessValues);

                        fofss.R += gatherFVarData(context,
                            i, faceIndex, levelFaceOffset, bIndex, levelFVarVertOffsets, fofss.R, fptrs.R);
                    } else if (patchTag._boundaryCount == 1) {
                        int const permuteBoundary[16] = { -1, 4, 5, 6, -1, 0, 1, 7, -1, 3, 2, 8, -1, 11, 10, 9 };

                        level->gatherQuadRegularBoundaryPatchPoints(faceIndex, patchVerts, bIndex);
                        offsetAndPermuteIndices(patchVerts, 16, levelVertOffset, permuteBoundary, iptrs.R);

                        bIndex = (bIndex+1)%4;

                        iptrs.R += 16;
                        pptrs.R = computePatchParam(refiner, i, faceIndex, bIndex, boundaryMask, transitionMask, pptrs.R);

                        if (sptrs.R) *sptrs.R++ = assignSharpnessIndex(0, tables->_sharpnessValues);

                        fofss.R += gatherFVarData(context,
                            i, faceIndex, levelFaceOffset, /*rotation*/0, levelFVarVertOffsets, fofss.R, fptrs.R);
                    } else {
                        int const permuteCorner[16] = { -1, -1, -1, -1, -1, 0, 1, 4, -1, 3, 2, 5, -1, 8, 7, 6 };

                        level->gatherQuadRegularCornerPatchPoints(faceIndex, patchVerts, bIndex);
                        offsetAndPermuteIndices(patchVerts, 16, levelVertOffset, permuteCorner, iptrs.R);

                        iptrs.R += 16;
                        pptrs.R = computePatchParam(refiner, i, faceIndex, bIndex, boundaryMask, transitionMask, pptrs.R);

                        if (sptrs.R) *sptrs.R++ = assignSharpnessIndex(0, tables->_sharpnessValues);

                        fofss.R += gatherFVarData(context,
                            i, faceIndex, levelFaceOffset, /*rotation*/0, levelFVarVertOffsets, fofss.R, fptrs.R);
                    }
                }
            } else {
                // emit end patch. end patch should be in the max level (until we implement DFAS)
                assert(i==refiner.GetMaxLevel());

                // endpatch factory tells vertex indices. The indices are offsetted
                // so that they can directly copied into patcharray.
                if (context.endPatchFactory) {
                    Far::ConstIndexArray cvs =
                        context.endPatchFactory->GetTopology(*level, faceIndex,
                                                             levelPatchTags,
                                                             levelVertOffset);

                    switch(context.endPatchFactory->GetPatchType(patchTag)) {
                    case PatchDescriptor::GREGORY_BASIS:
                    {
                        for (int j = 0; j < cvs.size(); ++j) iptrs.GP[j] = cvs[j];
                        iptrs.GP += cvs.size();
                        pptrs.GP = computePatchParam(
                            refiner, i, faceIndex, 0, /*boundary*/0, /*transition*/0, pptrs.GP);
                        fofss.GP += gatherFVarData(context,
                                                   i, faceIndex, levelFaceOffset,
                                                   0, levelFVarVertOffsets, fofss.GP, fptrs.GP);
                    }
                    break;
                    case PatchDescriptor::GREGORY:
                    {
                        for (int j = 0; j < cvs.size(); ++j) iptrs.G[j] = cvs[j];
                        iptrs.G += cvs.size();
                        pptrs.G = computePatchParam(
                            refiner, i, faceIndex, 0, /*boundary*/0, /*transition*/0, pptrs.G);
                        fofss.G += gatherFVarData(context,
                                                  i, faceIndex, levelFaceOffset,
                                                  0, levelFVarVertOffsets, fofss.G, fptrs.G);
                    }
                    break;
                    case PatchDescriptor::GREGORY_BOUNDARY:
                    {
                        for (int j = 0; j < cvs.size(); ++j) iptrs.GB[j] = cvs[j];
                        iptrs.GB += cvs.size();
                        pptrs.GB = computePatchParam(
                            refiner, i, faceIndex, 0, /*boundary*/0, /*transition*/0, pptrs.GB);
                        fofss.GB += gatherFVarData(context,
                                                   i, faceIndex, levelFaceOffset,
                                                   0, levelFVarVertOffsets, fofss.GB, fptrs.GB);
                    }
                    default:
                        // unknown
                        assert(false);
                    break;
                    }
                }
            }
        }
        levelFaceOffset += level->getNumFaces();
        levelVertOffset += level->getNumVertices();
        if (context.RequiresFVarPatches()) {
            int nchannels = refiner.GetNumFVarChannels();
            for (int channel=0; channel<nchannels; ++channel) {
                levelFVarVertOffsets[channel] += level->getNumFVarValues(channel);
            }
        }
    }

    if (context.RequiresFVarPatches()) {
        // Compress & copy FVar values from context into FVarPatchChannel
        // sparse array, generate offsets

        FVarChannelCursor & fvc = context.fvarChannelCursor;
        for (fvc=fvc.begin(); fvc!=fvc.end(); ++fvc) {

            if (tables->GetFVarChannelLinearInterpolation(fvc.pos())!=Sdc::Options::FVAR_LINEAR_ALL) {
                tables->setBicubicFVarPatchChannelValues(fvc.pos(),
                    context.fvarPatchSize, context.fvarPatchValues[fvc.pos()]);
            }
        }
    }

}

} // end namespace Far

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv
