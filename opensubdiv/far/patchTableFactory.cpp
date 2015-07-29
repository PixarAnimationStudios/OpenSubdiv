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
#include "../far/patchTableFactory.h"
#include "../far/error.h"
#include "../far/ptexIndices.h"
#include "../far/topologyRefiner.h"
#include "../vtr/level.h"
#include "../vtr/fvarLevel.h"
#include "../vtr/refinement.h"
#include "../far/endCapBSplineBasisPatchFactory.h"
#include "../far/endCapGregoryBasisPatchFactory.h"
#include "../far/endCapLegacyGregoryPatchFactory.h"

#include <algorithm>
#include <cassert>
#include <cstring>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace {

//
//  A convenience container for the different types of feature adaptive patches.
//  Each instance associates a value of the template parameter type with each
//  patch type.
//
template <class TYPE>
struct PatchTypes {

    TYPE R,    // regular patch
         G,    // gregory patch
         GB,   // gregory boundary patch
         GP;   // gregory basis patch

    PatchTypes() { std::memset(this, 0, sizeof(PatchTypes<TYPE>)); }

    TYPE & getValue( Far::PatchDescriptor desc ) {
        switch (desc.GetType()) {
            case Far::PatchDescriptor::REGULAR          : return R;
            case Far::PatchDescriptor::GREGORY          : return G;
            case Far::PatchDescriptor::GREGORY_BOUNDARY : return GB;
            case Far::PatchDescriptor::GREGORY_BASIS    : return GP;
            default : assert(0);
        }
        // can't be reached (suppress compiler warning)
        return R;
    }
};

typedef PatchTypes<Far::Index *>      PatchCVPointers;
typedef PatchTypes<Far::PatchParam *> PatchParamPointers;
typedef PatchTypes<Far::Index *>      SharpnessIndexPointers;
typedef PatchTypes<Far::Index>        PatchFVarOffsets;
typedef PatchTypes<Far::Index **>     PatchFVarPointers;

//  Helpers for compiler warnings and floating point equality tests
#ifdef __INTEL_COMPILER
#pragma warning (push)
#pragma warning disable 1572
#endif

inline bool isSharpnessEqual(float s1, float s2) { return (s1 == s2); }

#ifdef __INTEL_COMPILER
#pragma warning (pop)
#endif

} // namespace anon


namespace Far {

void
PatchTableFactory::PatchFaceTag::clear() {
    std::memset(this, 0, sizeof(*this));
}

void
PatchTableFactory::PatchFaceTag::assignBoundaryPropertiesFromEdgeMask(int boundaryEdgeMask) {
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
PatchTableFactory::PatchFaceTag::assignBoundaryPropertiesFromVertexMask(int boundaryVertexMask) {
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

    // The patch vertices for boundary and corner patches
    // are assigned index values even though indices will
    // be undefined along boundary and corner edges.
    // When the resulting patch table is going to be used
    // as indices for drawing, it is convenient for invalid
    // indices to be replaced with known good values, such
    // as the first un-permuted index, which is the index
    // of the first vertex of the patch face.
    Far::Index knownGoodIndex = indices[0];

    if (permutation) {
        for (int i = 0; i < count; ++i) {
            if (permutation[i] < 0) {
                result[i] = offset + knownGoodIndex;
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
                      PatchTableFactory::Options options)
        : _channelIndices(0)
    {
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
    bool operator != (int posArg) {
        return _currentChannel < posArg;
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

    int pos() const   { return _currentChannel; }
    int begin() const { return 0; }
    int end() const   { return _numChannels; }
    int size() const  { return _numChannels; }

private:
    int _numChannels,             // total number of channels
        _currentChannel;          // current cursor position
    int const * _channelIndices;  // list of selected channel indices
};

//
// Adaptive Context
//
// Helper class aggregating transient contextual data structures during the
// creation of feature adaptive patch table. The structure simplifies
// the function prototypes of high-level private methods in the factory.
// This helps keeping the factory class stateless.
//
// Note : struct members are not re-entrant nor are they intended to be !
//
struct PatchTableFactory::AdaptiveContext {

public:
    AdaptiveContext(TopologyRefiner const & refiner, Options options);

    TopologyRefiner const & refiner;

    Options const options;

    // The patch table being created
    PatchTable * table;

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
};

// Constructor
PatchTableFactory::AdaptiveContext::AdaptiveContext(
    TopologyRefiner const & ref, Options opts) :
    refiner(ref), options(opts), table(0),
    fvarChannelCursor(ref, opts) {
}

bool
PatchTableFactory::AdaptiveContext::RequiresFVarPatches() const {
    return (fvarChannelCursor.size() > 0);
}

//
//  Reserves tables based on the contents of the PatchArrayVector in the PatchTable:
//
void
PatchTableFactory::allocateVertexTables(PatchTable * table, int /* nlevels */, bool hasSharpness) {

    int ncvs = 0, npatches = 0;
    for (int i=0; i<table->GetNumPatchArrays(); ++i) {
        npatches += table->GetNumPatches(i);
        ncvs += table->GetNumControlVertices(i);
    }

    if (ncvs==0 or npatches==0)
        return;

    table->_patchVerts.resize( ncvs );

    table->_paramTable.resize( npatches );

    if (hasSharpness) {
        table->_sharpnessIndices.resize( npatches, Vtr::INDEX_INVALID );
    }
}

//
//  Allocate face-varying tables
//
void
PatchTableFactory::allocateFVarChannels(TopologyRefiner const & refiner,
    Options options, int npatches, PatchTable * table) {

    assert(options.generateFVarTables and
        refiner.GetNumFVarChannels()>0 and npatches>0 and table);

    // Create a channel cursor to iterate over client-selected channels or
    // default to the channels found in the TopologyRefiner
    FVarChannelCursor fvc(refiner, options);
    if (fvc.size()==0) {
        return;
    }

    table->allocateFVarPatchChannels(fvc.size());

    // Iterate with the cursor to initialize each channel
    for (fvc=fvc.begin(); fvc!=fvc.end(); ++fvc) {

        Sdc::Options::FVarLinearInterpolation interpolation =
            refiner.GetFVarLinearInterpolation(*fvc);

        table->setFVarPatchChannelLinearInterpolation(interpolation, fvc.pos());

        int nverts = 0;

        PatchDescriptor::Type type = options.triangulateQuads ?
            PatchDescriptor::TRIANGLES : PatchDescriptor::QUADS;

        nverts =
            npatches * PatchDescriptor::GetNumFVarControlVertices(type);

        table->allocateFVarPatchChannelValues(npatches, nverts, fvc.pos());
    }
}


// gather face-varying patch points
int
PatchTableFactory::gatherFVarData(AdaptiveContext & context, int level,
    Index faceIndex, Index levelFaceOffset, int rotation,
        Index const * levelFVarVertOffsets, Index fofss, Index ** fptrs) {

    (void)levelFaceOffset;  // not used
    (void)fofss;  // not used

    if (not context.RequiresFVarPatches()) {
        return 0;
    }

    TopologyRefiner const & refiner = context.refiner;

    // Iterate over valid FVar channels (if any)
    FVarChannelCursor & fvc = context.fvarChannelCursor;
    for (fvc=fvc.begin(); fvc!=fvc.end(); ++fvc) {

        Vtr::internal::Level const & vtxLevel = refiner.getLevel(level);
        Vtr::internal::FVarLevel const & fvarLevel = vtxLevel.getFVarLevel(*fvc);

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
    return 1;
}

//
//  Populates the PatchParam for the given face, returning
//  a pointer to the next entry
//
PatchParam *
PatchTableFactory::computePatchParam(
    TopologyRefiner const & refiner, PtexIndices const &ptexIndices,
    int depth, Vtr::Index faceIndex, int boundaryMask, 
    int transitionMask, PatchParam *param) {

    if (param == NULL) return NULL;

    // Move up the hierarchy accumulating u,v indices to the coarse level:
    int childIndexInParent = 0,
        u = 0,
        v = 0,
        ofs = 1;

    bool nonquad = (refiner.GetLevel(depth).GetFaceVertices(faceIndex).size() != 4);

    for (int i = depth; i > 0; --i) {
        Vtr::internal::Refinement const& refinement  = refiner.getRefinement(i-1);
        Vtr::internal::Level const&      parentLevel = refiner.getLevel(i-1);

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

    Vtr::Index ptexIndex = ptexIndices.GetFaceId(faceIndex);
    assert(ptexIndex!=-1);

    if (nonquad) {
        ptexIndex+=childIndexInParent;
        --depth;
    }

    param->Set(ptexIndex, (short)u, (short)v, (unsigned short) depth, nonquad,
               (unsigned short) boundaryMask, (unsigned short) transitionMask);

    return ++param;
}


//
//  Indexing sharpnesses
//
inline int
assignSharpnessIndex(float sharpness, std::vector<float> & sharpnessValues) {

    // linear search
    for (int i=0; i<(int)sharpnessValues.size(); ++i) {
        if (isSharpnessEqual(sharpnessValues[i], sharpness)) {
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
PatchTable *
PatchTableFactory::Create(TopologyRefiner const & refiner, Options options) {

    if (refiner.IsUniform()) {
        return createUniform(refiner, options);
    } else {
        return createAdaptive(refiner, options);
    }
}

PatchTable *
PatchTableFactory::createUniform(TopologyRefiner const & refiner, Options options) {

    assert(refiner.IsUniform());

    // ensure that triangulateQuads is only set for quadrilateral schemes
    options.triangulateQuads &= (refiner.GetSchemeType()==Sdc::SCHEME_BILINEAR or
                                 refiner.GetSchemeType()==Sdc::SCHEME_CATMARK);

    // level=0 may contain n-gons, which are not supported in PatchTable.
    // even if generateAllLevels = true, we start from level 1.

    int maxvalence = refiner.GetMaxValence(),
        maxlevel = refiner.GetMaxLevel(),
        firstlevel = options.generateAllLevels ? 1 : maxlevel,
        nlevels = maxlevel-firstlevel+1;

    PtexIndices ptexIndices(refiner);

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
    //  Create the instance of the table and allocate and initialize its members.
    //
    PatchTable * table = new PatchTable(maxvalence);

    table->_numPtexFaces = ptexIndices.GetNumFaces();

    table->reservePatchArrays(nlevels);

    PatchDescriptor desc(ptype);

    // generate patch arrays
    for (int level=firstlevel, poffset=0, voffset=0; level<=maxlevel; ++level) {

        TopologyLevel const & refLevel = refiner.GetLevel(level);

        int npatches = refLevel.GetNumFaces();
        if (refiner.HasHoles()) {
            for (int i = npatches - 1; i >= 0; --i) {
                npatches -= refLevel.IsFaceHole(i);
            }
        }
        assert(npatches>=0);

        if (options.triangulateQuads)
            npatches *= 2;

        table->pushPatchArray(desc, npatches, &voffset, &poffset, 0);
    }

    // Allocate various tables
    allocateVertexTables( table, 0, /*hasSharpness=*/false );

    FVarChannelCursor fvc(refiner, options);
    bool generateFVarPatches = (options.generateFVarTables and fvc.size()>0);
    if (generateFVarPatches) {
        int npatches = table->GetNumPatchesTotal();
        allocateFVarChannels(refiner, options, npatches, table);
        assert(fvc.size() == table->GetNumFVarChannels());
    }

    //
    //  Now populate the patches:
    //

    Index          * iptr = &table->_patchVerts[0];
    PatchParam     * pptr = &table->_paramTable[0];
    Index         ** fptr = 0;

    // we always skip level=0 vertices (control cages)
    Index levelVertOffset = refiner.GetLevel(0).GetNumVertices();

    Index * levelFVarVertOffsets = 0;
    if (generateFVarPatches) {

        levelFVarVertOffsets = (Index *)alloca(fvc.size()*sizeof(Index));
        memset(levelFVarVertOffsets, 0, fvc.size()*sizeof(Index));

        fptr = (Index **)alloca(fvc.size()*sizeof(Index *));
        for (fvc=fvc.begin(); fvc!=fvc.end(); ++fvc) {
            fptr[fvc.pos()] = table->getFVarValues(fvc.pos()).begin();
        }
    }

    for (int level=1; level<=maxlevel; ++level) {

        TopologyLevel const & refLevel = refiner.GetLevel(level);

        int nfaces = refLevel.GetNumFaces();
        if (level>=firstlevel) {
            for (int face=0; face<nfaces; ++face) {

                if (refiner.HasHoles() and refLevel.IsFaceHole(face)) {
                    continue;
                }

                ConstIndexArray fverts = refLevel.GetFaceVertices(face);
                for (int vert=0; vert<fverts.size(); ++vert) {
                    *iptr++ = levelVertOffset + fverts[vert];
                }

                pptr = computePatchParam(refiner, ptexIndices, level, face, /*boundary*/0, /*transition*/0, pptr);

                if (generateFVarPatches) {
                    for (fvc=fvc.begin(); fvc!=fvc.end(); ++fvc) {
                        ConstIndexArray fvalues = refLevel.GetFaceFVarValues(face, *fvc);
                        for (int vert=0; vert<fvalues.size(); ++vert) {
                            assert((levelVertOffset + fvalues[vert]) < (int)table->getFVarValues(fvc.pos()).size());
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
            levelVertOffset += refiner.GetLevel(level).GetNumVertices();
            for (fvc=fvc.begin(); fvc!=fvc.end(); ++fvc) {
                levelFVarVertOffsets[fvc.pos()] += refiner.GetLevel(level).GetNumFVarValues(fvc.pos());
            }
        }
    }
    return table;
}

PatchTable *
PatchTableFactory::createAdaptive(TopologyRefiner const & refiner, Options options) {

    assert(not refiner.IsUniform());

    PtexIndices ptexIndices(refiner);

    AdaptiveContext context(refiner, options);

    //
    //  First identify the patches -- accumulating the inventory patches for all of the
    //  different types and information about the patch for each face:
    //
    identifyAdaptivePatches(context);

    //
    //  Create the instance of the table and allocate and initialize its members based on
    //  the inventory of patches determined above:
    //
    int maxValence = refiner.GetMaxValence();

    context.table = new PatchTable(maxValence);

    // Populate the patch array descriptors
    int numPatchArrays = 0;
    if (context.patchInventory.R > 0) ++numPatchArrays;
    if (context.patchInventory.G > 0) ++numPatchArrays;
    if (context.patchInventory.GB > 0) ++numPatchArrays;
    if (context.patchInventory.GP > 0) ++numPatchArrays;

    context.table->reservePatchArrays(numPatchArrays);

    // Sort through the inventory and push back non-empty patch arrays
    ConstPatchDescriptorArray const & descs =
        PatchDescriptor::GetAdaptivePatchDescriptors(Sdc::SCHEME_CATMARK);

    int voffset=0, poffset=0, qoffset=0;
    for (int i=0; i<descs.size(); ++i) {
        PatchDescriptor desc = descs[i];
        context.table->pushPatchArray(desc,
            context.patchInventory.getValue(desc), &voffset, &poffset, &qoffset );
    }

    context.table->_numPtexFaces = ptexIndices.GetNumFaces();

    // Allocate various tables
    bool hasSharpness = context.options.useSingleCreasePatch;
    allocateVertexTables(context.table, 0, hasSharpness);

    if (context.RequiresFVarPatches()) {

        int npatches = context.table->GetNumPatchesTotal();

        allocateFVarChannels(refiner, options, npatches, context.table);
    }

    //
    //  Now populate the patches:
    //
    populateAdaptivePatches(context, ptexIndices);

    return context.table;
}

//
//  Identify all patches required for faces at all levels -- accumulating the number of patches
//  for each type, and retaining enough information for the patch for each face to populate it
//  later with no additional analysis.
//
void
PatchTableFactory::identifyAdaptivePatches(AdaptiveContext & context) {

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
        Vtr::internal::Level const * level = &refiner.getLevel(levelIndex);

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
        Vtr::internal::Refinement const            * refinement = 0;
        Vtr::internal::Refinement::SparseTag const * refinedFaceTags = 0;

        if (levelIndex < refiner.GetMaxLevel()) {
            refinement      = &refiner.getRefinement(levelIndex);
            refinedFaceTags = &refinement->getParentFaceSparseTag(0);
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
            Vtr::internal::Refinement::SparseTag refinedFaceTag = refinedFaceTags ?
                refinedFaceTags[faceIndex] : Vtr::internal::Refinement::SparseTag();

            if (refinedFaceTag._selected) {
                continue;
            }

            Vtr::ConstIndexArray fVerts = level->getFaceVertices(faceIndex);
            assert(fVerts.size() == 4);

            Vtr::internal::Level::VTag compFaceVertTag = level->getFaceCompositeVTag(fVerts);
            if (compFaceVertTag._incomplete) {
                continue;
            }

            //
            //  We have a quad that will be represented as a B-spline or end cap patch.  Use
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
            //  As for transition detection, assign the transition properties (even if 0).
            //
            //  NOTE on patches around non-manifold vertices:
            //      In most cases the use of regular boundary or corner patches is what we want,
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
                Vtr::internal::Level::ETag compFaceETag = level->getFaceCompositeETag(fEdges);

                if (compFaceETag._semiSharp or compFaceETag._infSharp) {
                    float sharpness = 0;
                    int rotation = 0;
                    if (level->isSingleCreasePatch(faceIndex, &sharpness, &rotation)) {

                        // cap sharpness to the max isolation level
                        float cappedSharpness =
                                std::min(sharpness, (float)(context.options.maxIsolationLevel - levelIndex));
                        if (cappedSharpness > 0) {
                            patchTag._isSingleCrease = true;
                            patchTag._boundaryIndex = rotation;
                        }
                    }
                }
            }

            //  Identify boundaries for both regular and xordinary patches -- non-manifold
            //  edges and vertices are interpreted as boundaries for regular patches
            if (hasBoundaryVertex or hasNonManifoldVertex) {
                Vtr::ConstIndexArray fEdges = level->getFaceEdges(faceIndex);

                int boundaryEdgeMask = ((level->getEdgeTag(fEdges[0])._boundary) << 0) |
                                       ((level->getEdgeTag(fEdges[1])._boundary) << 1) |
                                       ((level->getEdgeTag(fEdges[2])._boundary) << 2) |
                                       ((level->getEdgeTag(fEdges[3])._boundary) << 3);
                if (hasNonManifoldVertex) {
                    int nonManEdgeMask = ((level->getEdgeTag(fEdges[0])._nonManifold) << 0) |
                                         ((level->getEdgeTag(fEdges[1])._nonManifold) << 1) |
                                         ((level->getEdgeTag(fEdges[2])._nonManifold) << 2) |
                                         ((level->getEdgeTag(fEdges[3])._nonManifold) << 3);
                    boundaryEdgeMask |= nonManEdgeMask;
                }

                if (boundaryEdgeMask) {
                    patchTag.assignBoundaryPropertiesFromEdgeMask(boundaryEdgeMask);
                } else {
                    int boundaryVertMask = ((level->getVertexTag(fVerts[0])._boundary) << 0) |
                                           ((level->getVertexTag(fVerts[1])._boundary) << 1) |
                                           ((level->getVertexTag(fVerts[2])._boundary) << 2) |
                                           ((level->getVertexTag(fVerts[3])._boundary) << 3);

                    if (hasNonManifoldVertex) {
                        int nonManVertMask = ((level->getVertexTag(fVerts[0])._nonManifold) << 0) |
                                             ((level->getVertexTag(fVerts[1])._nonManifold) << 1) |
                                             ((level->getVertexTag(fVerts[2])._nonManifold) << 2) |
                                             ((level->getVertexTag(fVerts[3])._nonManifold) << 3);
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
                        if (level->getVertexTag(fVerts[0])._xordinary) { xordCount++; xordVertex = 0; }
                        if (level->getVertexTag(fVerts[1])._xordinary) { xordCount++; xordVertex = 1; }
                        if (level->getVertexTag(fVerts[2])._xordinary) { xordCount++; xordVertex = 2; }
                        if (level->getVertexTag(fVerts[3])._xordinary) { xordCount++; xordVertex = 3; }

                        if (xordCount == 1) {
                            //  We require the vertex opposite the xordinary vertex be interior:
                            if (not level->getVertexTag(fVerts[(xordVertex + 2) % 4])._boundary) {
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
                // select endcap patchtype
                switch(context.options.GetEndCapType()) {
                case Options::ENDCAP_GREGORY_BASIS:
                    context.patchInventory.GP++;
                    break;
                case Options::ENDCAP_BSPLINE_BASIS:
                    context.patchInventory.R++;
                    break;
                case Options::ENDCAP_LEGACY_GREGORY:
                    if (patchTag._boundaryCount == 0) {
                        context.patchInventory.G++;
                    } else {
                        context.patchInventory.GB++;
                    }
                    break;
                case Options::ENDCAP_BILINEAR_BASIS:
                    // not implemented yet
                    assert(false);
                    break;
                default:
                    // no endcap
                    break;
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
PatchTableFactory::populateAdaptivePatches(
    AdaptiveContext & context, PtexIndices const & ptexIndices) {

    TopologyRefiner const & refiner = context.refiner;

    PatchTable * table = context.table;

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

        Index arrayIndex = table->findPatchArray(desc);

        if (arrayIndex==Vtr::INDEX_INVALID) {
            continue;
        }

        iptrs.getValue(desc) = table->getPatchArrayVertices(arrayIndex).begin();
        pptrs.getValue(desc) = table->getPatchParams(arrayIndex).begin();
        if (context.options.useSingleCreasePatch) {
            sptrs.getValue(desc) = table->getSharpnessIndices(arrayIndex);
        }

        if (context.RequiresFVarPatches()) {

            Index & offsets = fofss.getValue(desc);
            offsets = table->getPatchIndex(arrayIndex, 0);

            // XXXX manuelk this stuff will go away as we use offsets from FVarPatchChannel
            FVarChannelCursor & fvc = context.fvarChannelCursor;
            assert(fvc.size() == table->GetNumFVarChannels());

            Index ** fptr = (Index **)alloca(fvc.size()*sizeof(Index *));
            for (fvc=fvc.begin(); fvc!=fvc.end(); ++fvc) {

                Index pidx = table->getPatchIndex(arrayIndex, 0);
                int ofs = pidx * 4;
                fptr[fvc.pos()] = &table->getFVarValues(fvc.pos())[ofs];
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
         levelFVarVertOffsets = (int *)alloca(nchannels*sizeof(int));
         memset(levelFVarVertOffsets, 0, nchannels*sizeof(int));
    }

    // endcap factories
    // XXX
    EndCapBSplineBasisPatchFactory *endCapBSpline = NULL;
    EndCapGregoryBasisPatchFactory *endCapGregoryBasis = NULL;
    EndCapLegacyGregoryPatchFactory *endCapLegacyGregory = NULL;

    switch(context.options.GetEndCapType()) {
    case Options::ENDCAP_GREGORY_BASIS:
        endCapGregoryBasis = new EndCapGregoryBasisPatchFactory(
            refiner, context.options.shareEndCapPatchPoints);
        break;
    case Options::ENDCAP_BSPLINE_BASIS:
        endCapBSpline = new EndCapBSplineBasisPatchFactory(refiner);
        break;
    case Options::ENDCAP_LEGACY_GREGORY:
        endCapLegacyGregory = new EndCapLegacyGregoryPatchFactory(refiner);
        break;
    default:
        break;
    }

    for (int i = 0; i < refiner.GetNumLevels(); ++i) {
        Vtr::internal::Level const * level = &refiner.getLevel(i);

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

                int const * permutation = 0;
                // only single-crease patch has a sharpness.
                float sharpness = 0;

                if (patchTag._boundaryCount == 0) {
                    static int const permuteRegular[16] = { 5, 6, 7, 8, 4, 0, 1, 9, 15, 3, 2, 10, 14, 13, 12, 11 };
                    permutation = permuteRegular;

                    if (patchTag._isSingleCrease) {
                        boundaryMask = (1<<bIndex);
                        sharpness = level->getEdgeSharpness((level->getFaceEdges(faceIndex)[bIndex]));
                        sharpness = std::min(sharpness, (float)(context.options.maxIsolationLevel-i));
                    }

                    level->gatherQuadRegularInteriorPatchPoints(faceIndex, patchVerts, 0 /* no rotation*/);
                } else if (patchTag._boundaryCount == 1) {
                    // Expand boundary patch vertices and rotate to restore correct orientation.
                    static int const permuteBoundary[4][16] = {
                        { -1, -1, -1, -1, 11, 3, 0, 4, 10, 2, 1, 5, 9, 8, 7, 6 },
                        { 9, 10, 11, -1, 8, 2, 3, -1, 7, 1, 0, -1, 6, 5, 4, -1 },
                        { 6, 7, 8, 9, 5, 1, 2, 10, 4, 0, 3, 11, -1, -1, -1, -1 },
                        { -1, 4, 5, 6, -1, 0, 1, 7, -1, 3, 2, 8, -1, 11, 10, 9 } };
                    permutation = permuteBoundary[bIndex];
                    level->gatherQuadRegularBoundaryPatchPoints(faceIndex, patchVerts, bIndex);
                } else if (patchTag._boundaryCount == 2) {
                    // Expand corner patch vertices and rotate to restore correct orientation.
                    static int const permuteCorner[4][16] = {
                        { -1, -1, -1, -1, -1, 0, 1, 4, -1, 3, 2, 5, -1, 8, 7, 6 },
                        { -1, -1, -1, -1, 8, 3, 0, -1, 7, 2, 1, -1, 6, 5, 4, -1 },
                        { 6, 7, 8, -1, 5, 2, 3, -1, 4, 1, 0, -1, -1, -1, -1, -1 },
                        { -1, 4, 5, 6, -1, 1, 2, 7, -1, 0, 3, 8, -1, -1, -1, -1 } };
                    permutation = permuteCorner[bIndex];
                    level->gatherQuadRegularCornerPatchPoints(faceIndex, patchVerts, bIndex);
                } else {
                    assert(patchTag._boundaryCount <= 2);
                }

                offsetAndPermuteIndices(patchVerts, 16, levelVertOffset, permutation, iptrs.R);

                iptrs.R += 16;
                pptrs.R = computePatchParam(refiner, ptexIndices, i, faceIndex, boundaryMask, transitionMask, pptrs.R);
                // XXX: sharpness will be integrated into patch param soon.
                if (sptrs.R) *sptrs.R++ = assignSharpnessIndex(sharpness, table->_sharpnessValues);

                fofss.R += gatherFVarData(context,
                                          i, faceIndex, levelFaceOffset, /*rotation*/0, levelFVarVertOffsets, fofss.R, fptrs.R);
            } else {
                // emit end patch. end patch should be in the max level (until we implement DFAS)
                assert(i==refiner.GetMaxLevel());

                // switch endcap patchtype by option
                switch(context.options.GetEndCapType()) {
                case Options::ENDCAP_GREGORY_BASIS:
                {
                    // note: this call will be moved into vtr::level.
                    ConstIndexArray cvs = endCapGregoryBasis->GetPatchPoints(
                        level, faceIndex, levelPatchTags, levelVertOffset);

                    for (int j = 0; j < cvs.size(); ++j) iptrs.GP[j] = cvs[j];
                    iptrs.GP += cvs.size();
                    pptrs.GP = computePatchParam(
                        refiner, ptexIndices, i, faceIndex, /*boundary*/0, /*transition*/0, pptrs.GP);
                    if (sptrs.R) *sptrs.R++ = assignSharpnessIndex(0, table->_sharpnessValues);
                    fofss.GP += gatherFVarData(context,
                                               i, faceIndex, levelFaceOffset,
                                               0, levelFVarVertOffsets, fofss.GP, fptrs.GP);
                    break;
                }
                case Options::ENDCAP_BSPLINE_BASIS:
                {
                    ConstIndexArray cvs = endCapBSpline->GetPatchPoints(
                        level, faceIndex, levelPatchTags, levelVertOffset);

                    for (int j = 0; j < cvs.size(); ++j) iptrs.R[j] = cvs[j];
                    iptrs.R += cvs.size();
                    pptrs.R = computePatchParam(
                        refiner, ptexIndices, i, faceIndex, /*boundary*/0, /*transition*/0, pptrs.R);
                    if (sptrs.R) *sptrs.R++ = assignSharpnessIndex(0, table->_sharpnessValues);
                    fofss.R += gatherFVarData(context,
                                              i, faceIndex, levelFaceOffset,
                                              0, levelFVarVertOffsets, fofss.R, fptrs.R);
                    break;
                }
                case Options::ENDCAP_LEGACY_GREGORY:
                {
                    ConstIndexArray cvs = endCapLegacyGregory->GetPatchPoints(
                        level, faceIndex, levelPatchTags, levelVertOffset);

                    if (patchTag._boundaryCount == 0) {
                        for (int j = 0; j < cvs.size(); ++j) iptrs.G[j] = cvs[j];
                        iptrs.G += cvs.size();
                        pptrs.G = computePatchParam(
                            refiner, ptexIndices, i, faceIndex, /*boundary*/0, /*transition*/0, pptrs.G);
                        if (sptrs.R) *sptrs.R++ = assignSharpnessIndex(0, table->_sharpnessValues);
                        fofss.G += gatherFVarData(context,
                                                  i, faceIndex, levelFaceOffset,
                                                  0, levelFVarVertOffsets, fofss.G, fptrs.G);
                    } else {
                        for (int j = 0; j < cvs.size(); ++j) iptrs.GB[j] = cvs[j];
                        iptrs.GB += cvs.size();
                        pptrs.GB = computePatchParam(
                            refiner, ptexIndices, i, faceIndex, /*boundary*/0, /*transition*/0, pptrs.GB);
                        if (sptrs.R) *sptrs.R++ = assignSharpnessIndex(0, table->_sharpnessValues);
                        fofss.GB += gatherFVarData(context,
                                                   i, faceIndex, levelFaceOffset,
                                                   0, levelFVarVertOffsets, fofss.GB, fptrs.GB);
                    }
                    break;
                }
                case Options::ENDCAP_BILINEAR_BASIS:
                    // not implemented yet
                    assert(false);
                    break;
                default:
                    // no endcap
                    break;
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

    // finalize end patches
    switch(context.options.GetEndCapType()) {
    case Options::ENDCAP_GREGORY_BASIS:
        table->_localPointStencils =
            endCapGregoryBasis->CreateVertexStencilTable();
        table->_localPointVaryingStencils =
            endCapGregoryBasis->CreateVaryingStencilTable();
        delete endCapGregoryBasis;
        break;
    case Options::ENDCAP_BSPLINE_BASIS:
        table->_localPointStencils =
            endCapBSpline->CreateVertexStencilTable();
        table->_localPointVaryingStencils =
            endCapBSpline->CreateVaryingStencilTable();
        delete endCapBSpline;
        break;
    case Options::ENDCAP_LEGACY_GREGORY:
        endCapLegacyGregory->Finalize(
            table->GetMaxValence(),
            &table->_quadOffsetsTable,
            &table->_vertexValenceTable);
        delete endCapLegacyGregory;
        break;
    default:
        break;
    }
}

} // end namespace Far

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv

