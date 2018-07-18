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
#include "../far/patchBuilder.h"
#include "../far/error.h"
#include "../far/ptexIndices.h"
#include "../far/topologyRefiner.h"
#include "../vtr/level.h"
#include "../vtr/fvarLevel.h"
#include "../vtr/refinement.h"
#include "../vtr/stackBuffer.h"

#include <algorithm>
#include <cassert>
#include <cstring>

#include <cstdio>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {
namespace Far {

using Vtr::Array;
using Vtr::ConstArray;
using Vtr::IndexVector;
using Vtr::internal::Level;
using Vtr::internal::StackBuffer;

namespace {
    //
    //  Helpers for compiler warnings and floating point equality tests
    //
#ifdef __INTEL_COMPILER
#pragma warning (push)
#pragma warning disable 1572
#endif
    inline bool isSharpnessEqual(float s1, float s2) { return (s1 == s2); }
#ifdef __INTEL_COMPILER
#pragma warning (pop)
#endif

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

    inline bool
    isBoundaryFace(Level const & level, Index face) {

        return (level.getFaceCompositeVTag(face)._boundary != 0);
    }

    inline void
    offsetIndices(Index indices[], int size, int offset) {

        for (int i = 0; i < size; ++i) {
            indices[i] += offset;
        }
    }
} // namespace anon


//
//  The main PatchTableBuilder class with context
//
//  Helper class aggregating transient contextual data structures during the
//  creation of a patch table.  This helps keeping the factory class stateless.
//
class PatchTableBuilder {
public:
    //
    //  Public interface intended for use by the PatchTableFactory -- all
    //  else is solely for internal use:
    //
    typedef PatchTableFactory::Options Options;

    PatchTableBuilder(TopologyRefiner const & refiner, Options options);
    ~PatchTableBuilder();

    void BuildUniform();
    void BuildAdaptive();

    PatchTable * GetPatchTable() const { return _table; };

private:
    //  Simple struct to store <face,level> pair for a patch:
    struct PatchTuple {
        PatchTuple(Index face, int level) : faceIndex(face), levelIndex(level) { }

        Index faceIndex;
        int   levelIndex;
    };
    typedef std::vector<PatchTuple> PatchTupleVector;

    //  Struct comprising a collection of topological properties for a patch that
    //  may be shared (between vertex and face-varying patches):
    typedef SparseMatrix<float>   Matrix;
    struct PatchInfo {
        PatchInfo() : isRegular(false), isRegSingleCrease(false),
                      regBoundaryMask(0), regSharpness(0.0f),
                      paramBoundaryMask(0) { }

        bool         isRegular;
        bool         isRegSingleCrease;
        int          regBoundaryMask;
        float        regSharpness;
        Level::VSpan irregCornerSpans[4];
        int          paramBoundaryMask;
        Matrix       matrix;
    };

private:
    //
    //  Internal LocalPointHelper class
    //
    //  A LocalPointHelper manages the number, sharing of and StencilTable for
    //  the local points of a patch or one of its face-varying channels.  An
    //  instance of the helper does not know anything about the properties of
    //  the Builder classes that use it.  It can combine local points for any
    //  patch types all in one, so the regular and irregular patch types can
    //  both be local and differ, e.g. Bezier for regular and Gregory for
    //  irregular, and points can be effectively shared, appropriate stencils
    //  created, etc.
    //
    //  While not dependent on a particular patch type, some methods do require
    //  a patch type argument in order to know what can be done with its local
    //  points, e.g. which points can be shared with adjacent patches.
    //
    class LocalPointHelper {
    public:
        struct Options {
            Options() : shareLocalPoints(false),
                        reuseSourcePoints(false),
                        createStencilTable(true) { }

            unsigned int shareLocalPoints   : 1;
            unsigned int reuseSourcePoints  : 1;
            unsigned int createStencilTable : 1;
        };

    public:
        LocalPointHelper(TopologyRefiner const & refiner,
                         Options const & options,
                         int fvarChannel,
                         int numLocalPointsExpected);
        ~LocalPointHelper();

    public:
        int GetNumLocalPoints() const { return _numLocalPoints; }

        int AppendLocalPatchPoints(int levelIndex, Index faceIndex,
                                   Matrix const &        conversionMatrix,
                                   PatchDescriptor::Type patchType,
                                   Index const           sourcePoints[],
                                   int                   sourcePointOffset,
                                   Index                 patchPoints[]);

        StencilTable * AcquireStencilTable() {
            return acquireStencilTable(&_stencilTable);
        }

    private:
        //  Internal methods:
        void appendLocalPointStencil(Matrix const & conversionMatrix,
                                     int            stencilRow,
                                     Index const    sourcePoints[],
                                     int            sourcePointOffset);

        void appendLocalPointStencils(Matrix const & conversionMatrix,
                                      Index const    sourcePoints[],
                                      int            sourcePointOffset);

        //  Methods for local point Varying stencils
        //      XXXX -- hope to get rid of these...
        void appendLocalPointVaryingStencil(int const *  varyingIndices,
                                            int          patchPointIndex,
                                            Index const  sourcePoints[],
                                            int          sourcePointOffset);

        void appendLocalPointVaryingStencils(int const *    varyingIndices,
                                             Matrix const & conversionMatrix,
                                             Index const    sourcePoints[],
                                             int            sourcePointOffset);

        StencilTable * acquireStencilTable(StencilTable **stencilTableMember);

        Index findSharedCornerPoint(int levelIndex, Index valueIndex,
                                    Index newIndex);
        Index findSharedEdgePoint(int levelIndex, Index edgeIndex, int edgeEnd,
                                  Index newIndex);

    private:
        //  Member variables:
        TopologyRefiner const& _refiner;
        Options                _options;

        int _fvarChannel;
        int _numLocalPoints;
        int _localPointOffset;

        std::vector<IndexVector> _sharedCornerPoints;
        std::vector<IndexVector> _sharedEdgePoints;

        StencilTable * _stencilTable;

    //  This was hopefully transitional but will persist -- the should be
    //  no need for Varying local points or stencils associated with them.
    public:
        StencilTable * AcquireStencilTableVarying() {
            return acquireStencilTable(&_stencilTableVarying);
        }

        StencilTable * _stencilTableVarying;
    };

private:
    //
    //  Internal LegacyGregoryHelper class
    //
    //  This local class helps to populate the arrays in the PatchTable
    //  associated with legacy-Gregory patches, i.e. the quad-offset and
    //  vertex-valence tables.  These patches are always associated with
    //  faces at the last level of refinement, so only the face index in
    //  that level is required to identify them.
    //
    class LegacyGregoryHelper {
    public:
        LegacyGregoryHelper(TopologyRefiner const & ref) : _refiner(ref) { }
        ~LegacyGregoryHelper() { }

    public:
        int GetNumBoundaryPatches() const { return (int)_boundaryFaceIndices.size(); }
        int GetNumInteriorPatches() const { return (int)_interiorFaceIndices.size(); }

        void AddPatchFace(int level, Index face);
        void FinalizeQuadOffsets(  PatchTable::QuadOffsetsTable & qTable);
        void FinalizeVertexValence(PatchTable::VertexValenceTable & vTable,
                                   int lastLevelVertOffset);
    private:
        TopologyRefiner const& _refiner;
        std::vector<Index> _interiorFaceIndices;
        std::vector<Index> _boundaryFaceIndices;
    };

private:
    //  Builder methods for internal use:

    //  Simple queries:
    int getRefinerFVarChannel(int fvcInTable) const {
        return (fvcInTable >= 0) ? _fvarChannelIndices[fvcInTable] : -1;
    }

    bool isFVarChannelLinear(int fvcInTable) const {
        if (_options.generateFVarLegacyLinearPatches) return true;
        return (_refiner.GetFVarLinearInterpolation(
            getRefinerFVarChannel(fvcInTable)) == Sdc::Options::FVAR_LINEAR_ALL);
    }

    bool doesFVarTopologyMatch(PatchTuple const & patch, int fvcInTable) {
        return _patchBuilder->DoesFaceVaryingPatchMatch(
            patch.levelIndex, patch.faceIndex,
            getRefinerFVarChannel(fvcInTable));
    }

    //  Methods for identifying and assigning patch-related data:
    void identifyPatchTopology(PatchTuple const & patch, PatchInfo & patchInfo,
                               int fvcInTable = -1);

    int assignPatchPointsAndStencils(PatchTuple const & patch,
                                     PatchInfo const & patchInfo,
                                     Index * patchPoints,
                                     LocalPointHelper & localHelper,
                                     int fvcInTable = -1);

    int assignFacePoints(PatchTuple const & patch,
                         Index * patchPoints,
                         int fvcInTable = -1) const;

    //  High level methods for assembling the table:
    void identifyAdaptivePatches();
    void populateAdaptivePatches();

    void allocateVertexTables();
    void allocateFVarChannels();

    int estimateLocalPointCount(LocalPointHelper::Options const & options,
                                int fvcInTable = -1) const;

private:
    //  Refiner and Options passed on construction:
    TopologyRefiner const & _refiner;
    Options const           _options;

    // Flags indicating the need for processing based on provided options
    unsigned int _requiresLocalPoints          : 1;
    unsigned int _requiresRegularLocalPoints   : 1;
    unsigned int _requiresIrregularLocalPoints : 1;
    unsigned int _requiresSharpnessArray       : 1;
    unsigned int _requiresFVarPatches          : 1;
    unsigned int _requiresVaryingPatches       : 1;
    unsigned int _requiresVaryingLocalPoints   : 1;

    // The PatchTable being constructed and classes to help its construction:
    PatchTable * _table;

    PatchBuilder *    _patchBuilder;
    PtexIndices const _ptexIndices;

    // Vector of tuples for each patch identified during topology traversal
    // and the numbers of irregular and irregular patches identifed:
    PatchTupleVector _patches;

    int _numRegularPatches;
    int _numIrregularPatches;

    // Vectors for remapping indices of vertices and fvar values as well
    // as the fvar channels (when a subset is chosen)
    std::vector<int>                _levelVertOffsets;
    std::vector< std::vector<int> > _levelFVarValueOffsets;
    std::vector<int>                _fvarChannelIndices;

    // State and helpers for legacy features
    bool                  _requiresLegacyGregoryTables;
    LegacyGregoryHelper * _legacyGregoryHelper;
};

// Constructor
PatchTableBuilder::PatchTableBuilder(
    TopologyRefiner const & refiner, Options opts) :
    _refiner(refiner), _options(opts),
    _table(0), _patchBuilder(0), _ptexIndices(refiner),
    _numRegularPatches(0), _numIrregularPatches(0),
    _legacyGregoryHelper(0) {

    if (_options.generateFVarTables) {
        // If client-code does not select specific channels, default to all
        // the channels in the refiner.
        if (_options.numFVarChannels==-1) {
            _fvarChannelIndices.resize(_refiner.GetNumFVarChannels());
            for (int fvc=0;fvc<(int)_fvarChannelIndices.size(); ++fvc) {
                _fvarChannelIndices[fvc] = fvc; // std::iota
            }
        } else {
            _fvarChannelIndices.assign(
                _options.fvarChannelIndices,
                _options.fvarChannelIndices + _options.numFVarChannels);
        }
    }

    //
    //  Will need to translate PatchTableFactory options to the newer set of
    //  PatchBuilder options in the near future.  And the state variables are
    //  a potentially complex combination of options and so are handled after
    //  the PatchBuilder construction (to potentially help) rather than in
    //  the initializer list.
    //
    PatchBuilder::Options patchOptions;

    patchOptions.regBasisType = PatchBuilder::BASIS_REGULAR;
    switch (_options.GetEndCapType()) {
        case Options::ENDCAP_BILINEAR_BASIS:
            patchOptions.irregBasisType = PatchBuilder::BASIS_LINEAR;
            break;
        case Options::ENDCAP_BSPLINE_BASIS:
            patchOptions.irregBasisType = PatchBuilder::BASIS_REGULAR;
            break;
        case Options::ENDCAP_GREGORY_BASIS:
            patchOptions.irregBasisType = PatchBuilder::BASIS_GREGORY;
            break;
        default:
            //  The PatchBuilder will infer if left un-specified
            patchOptions.irregBasisType = PatchBuilder::BASIS_UNSPECIFIED;
            break;
    }
    patchOptions.fillMissingBoundaryPoints   = true;
    patchOptions.approxInfSharpWithSmooth    = !_options.useInfSharpPatch;
    patchOptions.approxSmoothCornerWithSharp =
        _options.generateLegacySharpCornerPatches;

    _patchBuilder = PatchBuilder::Create(_refiner, patchOptions);

    //
    //  Initialize member variables that capture specified options:
    //
    _requiresRegularLocalPoints =
        (patchOptions.regBasisType != PatchBuilder::BASIS_REGULAR);
    _requiresIrregularLocalPoints =
        (_options.GetEndCapType() != Options::ENDCAP_LEGACY_GREGORY);
    _requiresLocalPoints =
        _requiresIrregularLocalPoints || _requiresRegularLocalPoints;

    _requiresSharpnessArray = _options.useSingleCreasePatch;
    _requiresFVarPatches = ! _fvarChannelIndices.empty();

    _requiresVaryingPatches = true;      // eventually to be made public
    _requiresVaryingLocalPoints = true;  // currently fixed

    //
    //  Create and initialize the new PatchTable instance to be assembled:
    //
    _table = new PatchTable(_refiner.GetMaxValence());

    _table->_numPtexFaces = _ptexIndices.GetNumFaces();

    //  State and helper to support LegacyGregory arrays in the PatchTable:
    _requiresLegacyGregoryTables = !_refiner.IsUniform() &&
        (_options.GetEndCapType() == Options::ENDCAP_LEGACY_GREGORY);

    if (_requiresLegacyGregoryTables) {
        _legacyGregoryHelper = new LegacyGregoryHelper(_refiner);
    }

}

PatchTableBuilder::~PatchTableBuilder() {

    delete _patchBuilder;
    delete _legacyGregoryHelper;
}


void
PatchTableBuilder::identifyPatchTopology(PatchTuple const & patch,
        PatchInfo & patchInfo, int fvarInTable) {

    int   patchLevel = patch.levelIndex;
    Index patchFace  = patch.faceIndex;

    int fvarInRefiner = getRefinerFVarChannel(fvarInTable);

    patchInfo.isRegular = _patchBuilder->IsPatchRegular(
        patchLevel, patchFace, fvarInTable);

    if (patchInfo.isRegular) {
        patchInfo.regBoundaryMask = _patchBuilder->GetRegularPatchBoundaryMask(
            patchLevel, patchFace, fvarInRefiner);

        patchInfo.isRegSingleCrease = false;
        patchInfo.regSharpness      = 0.0f;
        patchInfo.paramBoundaryMask = patchInfo.regBoundaryMask;

        //  If converting to another basis, get the change-of-basis matrix:
        if (_requiresRegularLocalPoints) {
            // _patchBuilder->GetRegularConversionMatrix(... patchInfo.matrix);
        }

        //
        //  Test regular interior patches for a single-crease patch when it
        //  was specified.
        //
        //  Note that the PatchTable clamps the sharpness of single-crease
        //  patches to that of the maximimu refinement level, so any single-
        //  crease patches at the last level will be reduced to regular
        //  patches (maintaining continuity with other semi-sharp patches
        //  also reduced to regular).
        //
        if (_requiresSharpnessArray &&
                (patchInfo.regBoundaryMask == 0) && (fvarInRefiner < 0)) {
            if (patchLevel < (int) _options.maxIsolationLevel) {
                PatchBuilder::SingleCreaseInfo creaseInfo;

                if (_patchBuilder->IsRegularSingleCreasePatch(
                        patchLevel, patchFace, creaseInfo)) {
                    creaseInfo.creaseSharpness =
                        std::min(creaseInfo.creaseSharpness,
                             (float)(_options.maxIsolationLevel - patchLevel));

                    patchInfo.isRegSingleCrease = true;
                    patchInfo.regSharpness      = creaseInfo.creaseSharpness;
                    patchInfo.paramBoundaryMask = (1 << creaseInfo.creaseEdgeInFace);
                }
            }
        }
    } else if (_requiresIrregularLocalPoints) {
        _patchBuilder->GetIrregularPatchCornerSpans(
            patchLevel, patchFace, patchInfo.irregCornerSpans, fvarInRefiner);

        _patchBuilder->GetIrregularPatchConversionMatrix(
            patchLevel, patchFace, patchInfo.irregCornerSpans, patchInfo.matrix);

        patchInfo.paramBoundaryMask = 0;
    }
}

int
PatchTableBuilder::assignPatchPointsAndStencils(PatchTuple const & patch,
        PatchInfo const & patchInfo, Index * patchPoints,
        LocalPointHelper & localHelper, int fvarInTable) {

    //
    //  This is where the interesting/complicated new work will take place
    //  when a change-of-basis is determined necessary and previously assigned
    //  to the PatchInfo
    //
    //  No change-of-basis means no local points or stencils associated with
    //  them, which should be trivial but should also only be true in the
    //  regular case.  (It is also the case that no local points will be
    //  generated for irregular patches when the LegacyGregory option is
    //  used -- so that possibility is still accounted for here.)
    //
    //  Regarding the return result, it should just be the size of the patch
    //  associated with the regular/irregular patch type chosen.  This could
    //  be retrieved from the PatchBuilder or PatchDescriptors -- alternatively
    //  a return value could be removed and the client left to increment by
    //  such a fixed step (which is already the case for FVar channels)
    //
    //  The more interesting size here is the number of local points/stencils
    //  added.
    //
    int fvarInRefiner = getRefinerFVarChannel(fvarInTable);

    int sourcePointOffset = (fvarInTable < 0)
                          ? _levelVertOffsets[patch.levelIndex]
                          : _levelFVarValueOffsets[fvarInTable][patch.levelIndex];

    int numPatchPoints = 0;
    if (patchInfo.isRegular) {
        if (!_requiresRegularLocalPoints) {
            numPatchPoints = _patchBuilder->GetRegularPatchPoints(
                    patch.levelIndex, patch.faceIndex,
                    patchInfo.regBoundaryMask, patchPoints, fvarInRefiner);

            //  PatchBuilder set to fill missing boundary points so offset all
            offsetIndices(patchPoints, numPatchPoints, sourcePointOffset);
        } else {
            //
            //  Future support for regular patches converted to another basis.
            //  Note the "source points" are not returned in the same
            //  orientation and there may be fewer than expected number in the
            //  case of boundaries:
            /*
            StackBuffer<Index,64,true> sourcePoints(
                patchInfo.matrix.GetNumColumns());

            _patchBuilder->GetRegularPatchSourcePoints(
                    patch.levelIndex, patch.faceIndex,
                    patchInfo.regBoundaryMask, sourcePoints, fvarInRefiner);

            localHelper.AppendLocalPatchPoints(
                    patch.levelIndex, patch.faceIndex,
                    patchInfo.matrix, _patchBuilder->GetRegularPatchType(),
                    sourcePoints, sourcePointOffset, patchPoints);

            numPatchPoints = patchInfo.matrix.GetNumRows();
            */
        }
    } else if (_requiresIrregularLocalPoints) {
        StackBuffer<Index,64,true> sourcePoints(patchInfo.matrix.GetNumColumns());

        _patchBuilder->GetIrregularPatchSourcePoints(
                patch.levelIndex, patch.faceIndex,
                patchInfo.irregCornerSpans, sourcePoints, fvarInRefiner);

        localHelper.AppendLocalPatchPoints(
                patch.levelIndex, patch.faceIndex,
                patchInfo.matrix, _patchBuilder->GetIrregularPatchType(),
                sourcePoints, sourcePointOffset, patchPoints);

        numPatchPoints = patchInfo.matrix.GetNumRows();
    }
    return numPatchPoints;
}

int
PatchTableBuilder::assignFacePoints(PatchTuple const & patch,
                                    Index * patchPoints,
                                    int fvarInTable) const {

    Level const & level = _refiner.getLevel(patch.levelIndex);

    int facePointOffset = (fvarInTable < 0)
                        ? _levelVertOffsets[patch.levelIndex]
                        : _levelFVarValueOffsets[fvarInTable][patch.levelIndex];

    int fvarInRefiner = getRefinerFVarChannel(fvarInTable);

    ConstIndexArray facePoints = (fvarInRefiner < 0)
                    ? level.getFaceVertices(patch.faceIndex)
                    : level.getFaceFVarValues(patch.faceIndex, fvarInRefiner);

    for (int i = 0; i < facePoints.size(); ++i) {
        patchPoints[i] = facePoints[i] + facePointOffset;
    }
    return facePoints.size();
}

//
//  Reserves tables based on contents of the PatchArrayVector in the PatchTable:
//
void
PatchTableBuilder::allocateVertexTables() {

    int ncvs = 0, npatches = 0;
    for (int i=0; i<_table->GetNumPatchArrays(); ++i) {
        npatches += _table->GetNumPatches(i);
        ncvs += _table->GetNumControlVertices(i);
    }

    if (ncvs==0 || npatches==0)
        return;

    _table->_patchVerts.resize( ncvs );

    _table->_paramTable.resize( npatches );

    if (! _refiner.IsUniform() && _requiresVaryingPatches) {
        _table->allocateVaryingVertices(
            PatchDescriptor(_patchBuilder->GetLinearPatchType()), npatches);
    }

    if (_requiresSharpnessArray) {
        _table->_sharpnessIndices.resize( npatches, Vtr::INDEX_INVALID );
    }
}

//
//  Allocate face-varying tables
//
void
PatchTableBuilder::allocateFVarChannels() {

    int npatches = _table->GetNumPatchesTotal();

    _table->allocateFVarPatchChannels((int)_fvarChannelIndices.size());

    // Initialize each channel
    for (int fvc=0; fvc<(int)_fvarChannelIndices.size(); ++fvc) {
        int refinerChannel = _fvarChannelIndices[fvc];

        Sdc::Options::FVarLinearInterpolation interpolation =
            _refiner.GetFVarLinearInterpolation(refinerChannel);

        _table->setFVarPatchChannelLinearInterpolation(interpolation, fvc);

        PatchDescriptor::Type fvarPatchType = _patchBuilder->GetLinearPatchType();
        if (_refiner.IsUniform()) {
            if (_options.triangulateQuads) {
                fvarPatchType = PatchDescriptor::TRIANGLES;
            }
        } else {
            if (!isFVarChannelLinear(fvc)) {
                fvarPatchType = _patchBuilder->GetIrregularPatchType();
            }
        }
        _table->allocateFVarPatchChannelValues(
                PatchDescriptor(fvarPatchType), npatches, fvc);
    }
}

void
PatchTableBuilder::BuildUniform() {

    assert(_refiner.IsUniform());

    // Default behavior is to include base level vertices in the patch vertices
    // for vertex and varying patches, but not face-varying.  Consider exposing
    // these as public options in future so that clients can create consistent
    // behavior:

    bool includeBaseLevelIndices     = true;
    bool includeBaseLevelFVarIndices = false;

    // ensure that triangulateQuads is only set for quadrilateral schemes
    bool triangulateQuads =
        _options.triangulateQuads && (_patchBuilder->GetRegularFaceSize() == 4);

    // level=0 may contain n-gons, which are not supported in PatchTable.
    // even if generateAllLevels = true, we start from level 1.

    int maxlevel = _refiner.GetMaxLevel(),
        firstlevel = _options.generateAllLevels ? 1 : maxlevel,
        nlevels = maxlevel-firstlevel+1;

    PatchDescriptor::Type ptype = triangulateQuads
                                ? PatchDescriptor::TRIANGLES
                                : _patchBuilder->GetLinearPatchType();

    //
    //  Allocate and initialize the table's members.
    //
    _table->reservePatchArrays(nlevels);

    PatchDescriptor desc(ptype);

    // generate patch arrays
    for (int level=firstlevel, poffset=0, voffset=0; level<=maxlevel; ++level) {

        TopologyLevel const & refLevel = _refiner.GetLevel(level);

        int npatches = refLevel.GetNumFaces();
        if (_refiner.HasHoles()) {
            for (int i = npatches - 1; i >= 0; --i) {
                npatches -= refLevel.IsFaceHole(i);
            }
        }
        assert(npatches>=0);

        if (triangulateQuads)
            npatches *= 2;

        _table->pushPatchArray(desc, npatches, &voffset, &poffset, 0);
    }

    // Allocate various tables
    allocateVertexTables();

    if (_requiresFVarPatches) {
        allocateFVarChannels();
    }

    //
    //  Now populate the patches:
    //

    Index          * iptr = &_table->_patchVerts[0];
    PatchParam     * pptr = &_table->_paramTable[0];
    Index         ** fptr  = 0;
    PatchParam    ** fpptr = 0;

    Index levelVertOffset = includeBaseLevelIndices
                          ? _refiner.GetLevel(0).GetNumVertices() : 0;

    Index * levelFVarVertOffsets = 0;
    if (_requiresFVarPatches) {

        levelFVarVertOffsets =
            (Index *)alloca(_fvarChannelIndices.size()*sizeof(Index));
        memset(levelFVarVertOffsets, 0, _fvarChannelIndices.size()*sizeof(Index));

        fptr = (Index **)alloca(_fvarChannelIndices.size()*sizeof(Index *));
        fpptr = (PatchParam **)alloca(_fvarChannelIndices.size()*sizeof(PatchParam *));
        for (int fvc=0; fvc<(int)_fvarChannelIndices.size(); ++fvc) {
            fptr[fvc] = _table->getFVarValues(fvc).begin();
            fpptr[fvc] = _table->getFVarPatchParams(fvc).begin();

            if (includeBaseLevelFVarIndices) {
                int refinerChannel = _fvarChannelIndices[fvc];
                levelFVarVertOffsets[fvc] =
                    _refiner.GetLevel(0).GetNumFVarValues(refinerChannel);
            }
        }
    }

    for (int level=1; level<=maxlevel; ++level) {

        TopologyLevel const & refLevel = _refiner.GetLevel(level);

        int nfaces = refLevel.GetNumFaces();
        if (level>=firstlevel) {
            for (int face=0; face<nfaces; ++face) {

                if (_refiner.HasHoles() && refLevel.IsFaceHole(face)) {
                    continue;
                }

                ConstIndexArray fverts = refLevel.GetFaceVertices(face);
                for (int vert=0; vert<fverts.size(); ++vert) {
                    *iptr++ = levelVertOffset + fverts[vert];
                }

                PatchParam pparam = _patchBuilder->ComputePatchParam(
                    level, face, _ptexIndices);
                *pptr++ = pparam;

                if (_requiresFVarPatches) {
                    for (int fvc=0; fvc<(int)_fvarChannelIndices.size(); ++fvc) {
                        int refinerChannel = _fvarChannelIndices[fvc];

                        ConstIndexArray fvalues =
                            refLevel.GetFaceFVarValues(face, refinerChannel);
                        for (int vert=0; vert<fvalues.size(); ++vert) {
                            assert((levelFVarVertOffsets[fvc] + fvalues[vert])
                                < (int)_table->getFVarValues(fvc).size());
                            fptr[fvc][vert] =
                                levelFVarVertOffsets[fvc] + fvalues[vert];
                        }
                        fptr[fvc]+=fvalues.size();
                        *fpptr[fvc]++ = pparam;
                    }
                }

                if (triangulateQuads) {
                    // Triangulate the quadrilateral:
                    //     {v0,v1,v2,v3} -> {v0,v1,v2},{v3,v0,v2}.
                    *iptr = *(iptr - 4); // copy v0 index
                    ++iptr;
                    *iptr = *(iptr - 3); // copy v2 index
                    ++iptr;

                    *pptr++ = pparam;

                    if (_requiresFVarPatches) {
                        for (int fvc=0; fvc<(int)_fvarChannelIndices.size(); ++fvc) {
                            *fptr[fvc] = *(fptr[fvc]-4); // copy fv0 index
                            ++fptr[fvc];
                            *fptr[fvc] = *(fptr[fvc]-3); // copy fv2 index
                            ++fptr[fvc];

                            *fpptr[fvc]++ = pparam;
                        }
                    }
                }
            }
        }

        if (_options.generateAllLevels) {
            levelVertOffset += _refiner.GetLevel(level).GetNumVertices();

            if (_requiresFVarPatches) {
                for (int fvc=0; fvc<(int)_fvarChannelIndices.size(); ++fvc) {
                    int refinerChannel = _fvarChannelIndices[fvc];
                    levelFVarVertOffsets[fvc] +=
                        _refiner.GetLevel(level).GetNumFVarValues(refinerChannel);
                }
            }
        }
    }
}

void
PatchTableBuilder::BuildAdaptive() {

    assert(! _refiner.IsUniform());

    identifyAdaptivePatches();
    populateAdaptivePatches();
}

//
//  Identify all patches required for faces at all levels -- appending the
//  <level,face> pairs to identify each patch for later construction, while
//  accumulating the number of regular vs irregular patches to size tables.
//
void
PatchTableBuilder::identifyAdaptivePatches() {

    //
    //  Iterate through the levels of refinement.  Accumulate index offsets
    //  for each level while inspecting faces for patches:
    //
    int reservePatches = _refiner.GetNumFacesTotal();
    _patches.reserve(reservePatches);

    _levelVertOffsets.push_back(0);
    _levelFVarValueOffsets.resize(_fvarChannelIndices.size());
    for (int fvc=0; fvc<(int)_fvarChannelIndices.size(); ++fvc) {
        _levelFVarValueOffsets[fvc].push_back(0);
    }

    for (int levelIndex=0; levelIndex<_refiner.GetNumLevels(); ++levelIndex) {
        Level const & level = _refiner.getLevel(levelIndex);

        _levelVertOffsets.push_back(
            _levelVertOffsets.back() + level.getNumVertices());

        for (int fvc=0; fvc<(int)_fvarChannelIndices.size(); ++fvc) {
            int refinerChannel = _fvarChannelIndices[fvc];
            _levelFVarValueOffsets[fvc].push_back(
                _levelFVarValueOffsets[fvc].back()
                + level.getNumFVarValues(refinerChannel));
        }

        for (int faceIndex = 0; faceIndex < level.getNumFaces(); ++faceIndex) {

            if (_patchBuilder->IsFaceAPatch(levelIndex, faceIndex) &&
                _patchBuilder->IsFaceALeaf(levelIndex, faceIndex)) {

                _patches.push_back(PatchTuple(faceIndex, levelIndex));

                // Count the patches here to simplify subsequent allocation.
                if (_patchBuilder->IsPatchRegular(levelIndex, faceIndex)) {
                    ++_numRegularPatches;
                } else {
                    ++_numIrregularPatches;

                    //  LegacyGregory needs to distinguish boundary vs interior
                    if (_requiresLegacyGregoryTables) {
                        _legacyGregoryHelper->AddPatchFace(levelIndex, faceIndex);
                    }
                }
            }
        }
    }
}

//
//  Populate adaptive patches that we've previously identified.
//
void
PatchTableBuilder::populateAdaptivePatches() {

    // State needed to populate an array in the patch table.
    // Pointers in this structure are initialized after the patch array
    // data buffers have been allocated and are then incremented as we
    // populate data into the patch table. Currently, we'll have at
    // most 3 patch arrays: Regular, Irregular, and IrregularBoundary.
    struct PatchArrayBuilder {
        PatchArrayBuilder()
            : patchType(PatchDescriptor::NON_PATCH), numPatches(0)
            , iptr(NULL), pptr(NULL), sptr(NULL), vptr(NULL) { }

        PatchDescriptor::Type patchType;
        int numPatches;

        Index      *iptr;
        PatchParam *pptr;
        Index      *sptr;
        Index      *vptr;

        StackBuffer<Index*,1>      fptr;   // fvar indices
        StackBuffer<PatchParam*,1> fpptr;  // fvar patch-params

    private:
        // Non-copyable
        PatchArrayBuilder(PatchArrayBuilder const &) {}
        PatchArrayBuilder & operator=(PatchArrayBuilder const &) {return *this;}

    } arrayBuilders[3];

    // Regular patches patches will be packed into the first patch array
    // Irregular patches will be packed into arrays according to optional
    // specification -- sharing the array with regular patches or packed
    // into an array of their own.
    int ARRAY_REGULAR   = 0;
    int ARRAY_IRREGULAR = 1;
    int ARRAY_BOUNDARY  = 2; // only used by LegacyGregory

    arrayBuilders[ARRAY_REGULAR].patchType = _patchBuilder->GetRegularPatchType();
    arrayBuilders[ARRAY_REGULAR].numPatches = _numRegularPatches;

    int numPatchArrays = (_numRegularPatches > 0);
    if (_numIrregularPatches > 0) {
        if (!_requiresLegacyGregoryTables) {
            //
            //  Pack irregular patches into same array as regular or separately:
            //
            if (_patchBuilder->GetRegularPatchType() ==
                    _patchBuilder->GetIrregularPatchType()) {
                ARRAY_IRREGULAR = ARRAY_REGULAR;
                numPatchArrays = 1;  // needed in case no regular patches
            } else {
                ARRAY_IRREGULAR = numPatchArrays;
                numPatchArrays ++;
            }
            arrayBuilders[ARRAY_IRREGULAR].patchType =
                _patchBuilder->GetIrregularPatchType();
            arrayBuilders[ARRAY_IRREGULAR].numPatches += _numIrregularPatches;
        } else {
            //
            // Arrays for Legacy-Gregory tables -- irregular patches are split
            // into two arrays for interior and boundary patches
            //
            ARRAY_IRREGULAR = numPatchArrays;
            arrayBuilders[ARRAY_IRREGULAR].patchType =
                PatchDescriptor::GREGORY;
            arrayBuilders[ARRAY_IRREGULAR].numPatches =
                _legacyGregoryHelper->GetNumInteriorPatches();
            numPatchArrays += (arrayBuilders[ARRAY_IRREGULAR].numPatches > 0);

            ARRAY_BOUNDARY = numPatchArrays;
            arrayBuilders[ARRAY_BOUNDARY].patchType =
                PatchDescriptor::GREGORY_BOUNDARY;
            arrayBuilders[ARRAY_BOUNDARY].numPatches =
                _legacyGregoryHelper->GetNumBoundaryPatches();
            numPatchArrays += (arrayBuilders[ARRAY_BOUNDARY].numPatches > 0);
        }
    }

    // Create patch arrays
    _table->reservePatchArrays(numPatchArrays);

    int voffset=0, poffset=0, qoffset=0;
    for (int arrayIndex=0; arrayIndex<numPatchArrays; ++arrayIndex) {
        PatchArrayBuilder & arrayBuilder = arrayBuilders[arrayIndex];
        _table->pushPatchArray(PatchDescriptor(arrayBuilder.patchType),
            arrayBuilder.numPatches, &voffset, &poffset, &qoffset );
    }

    // Allocate patch array data buffers
    allocateVertexTables();

    if (_requiresFVarPatches) {
        allocateFVarChannels();
    }

    // Initialize pointers used while populating patch array data buffers
    for (int arrayIndex=0; arrayIndex<numPatchArrays; ++arrayIndex) {
        PatchArrayBuilder & arrayBuilder = arrayBuilders[arrayIndex];

        arrayBuilder.iptr = _table->getPatchArrayVertices(arrayIndex).begin();
        arrayBuilder.pptr = _table->getPatchParams(arrayIndex).begin();
        if (_requiresSharpnessArray) {
            arrayBuilder.sptr = _table->getSharpnessIndices(arrayIndex);
        }
        if (_requiresVaryingPatches) {
            arrayBuilder.vptr = _table->getPatchArrayVaryingVertices(arrayIndex).begin();
        }

        if (_requiresFVarPatches) {
            arrayBuilder.fptr.SetSize((int)_fvarChannelIndices.size());
            arrayBuilder.fpptr.SetSize((int)_fvarChannelIndices.size());

            for (int fvc=0; fvc<(int)_fvarChannelIndices.size(); ++fvc) {

                PatchDescriptor desc = _table->GetFVarPatchDescriptor(fvc);

                Index pidx = _table->getPatchIndex(arrayIndex, 0);
                int   ofs  = pidx * desc.GetNumControlVertices();

                arrayBuilder.fptr[fvc] = &_table->getFVarValues(fvc)[ofs];
                arrayBuilder.fpptr[fvc] = &_table->getFVarPatchParams(fvc)[pidx];
            }
        }
    }

    //
    //  Initializing StencilTable and other helpers:
    //
    LocalPointHelper * vertexLocalPointHelper = 0;
    LocalPointHelper * varyingLocalPointHelper = 0;

    StackBuffer<LocalPointHelper*,4> fvarLocalPointHelpers;

    if (_requiresLocalPoints) {
        LocalPointHelper::Options opts;
        opts.createStencilTable = true;
        opts.shareLocalPoints   = _options.shareEndCapPatchPoints;
        opts.reuseSourcePoints  = (_patchBuilder->GetIrregularPatchType() ==
                                   _patchBuilder->GetNativePatchType() );

        vertexLocalPointHelper = new LocalPointHelper(
                _refiner, opts, -1, estimateLocalPointCount(opts, -1));

        if (_requiresFVarPatches) {
            fvarLocalPointHelpers.SetSize((int)_fvarChannelIndices.size());

            for (int fvc = 0; fvc < (int)_fvarChannelIndices.size(); ++fvc) {
                fvarLocalPointHelpers[fvc] = new LocalPointHelper(
                        _refiner, opts, getRefinerFVarChannel(fvc),
                        estimateLocalPointCount(opts, fvc));
            }
        }
    }

    //  Populate patch data buffers
    //
    //  Intentionally declare local vairables to contain patch topology info
    //  outside the loop to avoid repeated memory de-allocation/re-allocation
    //  associated with a change-of-basis.
    PatchInfo patchInfo;
    PatchInfo fvarPatchInfo;

    for (int patchIndex = 0; patchIndex < (int)_patches.size(); ++patchIndex) {

        PatchTuple const & patch = _patches[patchIndex];

        //
        //  Identify and assign points, stencils, sharpness, etc. for this patch:
        //
        identifyPatchTopology(patch, patchInfo);

        PatchArrayBuilder * arrayBuilder = &arrayBuilders[ARRAY_REGULAR];
        if (!patchInfo.isRegular) {
            arrayBuilder = &arrayBuilders[ARRAY_IRREGULAR];
        }

        if (_requiresLegacyGregoryTables && !patchInfo.isRegular) {
            if (isBoundaryFace(
                    _refiner.getLevel(patch.levelIndex), patch.faceIndex)) {
                arrayBuilder = &arrayBuilders[ARRAY_BOUNDARY];
            }
            arrayBuilder->iptr += assignFacePoints(patch, arrayBuilder->iptr);
        } else {
            arrayBuilder->iptr += assignPatchPointsAndStencils(patch,
                        patchInfo, arrayBuilder->iptr, *vertexLocalPointHelper);
        }
        if (_requiresSharpnessArray) {
            *arrayBuilder->sptr++ = assignSharpnessIndex(patchInfo.regSharpness,
                        _table->_sharpnessValues);
        }

        //
        //  Identify the PatchParam -- which may vary slightly for face-varying
        //  channels but will still be used to initialize them when so to avoid
        //  recomputing from scratch
        //
        PatchParam patchParam =
            _patchBuilder->ComputePatchParam(patch.levelIndex, patch.faceIndex,
                  _ptexIndices, patchInfo.paramBoundaryMask,
                  patchInfo.isRegular /* condition to compute transition mask */);
        *arrayBuilder->pptr++ = patchParam;

        //
        //  Assignment of face-varying patches is split between the comon/trivial
        //  case of linear patches and the more complex non-linear cases:
        //
        assert((_fvarChannelIndices.size() > 0) == _requiresFVarPatches);
        for (int fvc = 0; fvc < (int)_fvarChannelIndices.size(); ++fvc) {

            if (isFVarChannelLinear(fvc)) {
                assignFacePoints(patch, arrayBuilder->fptr[fvc], fvc);

                *arrayBuilder->fpptr[fvc] = patchParam;
            } else {
                //
                //  For non-linear patches, reuse patch information when the
                //  topology of the face in face-varying space matches the
                //  original patch:
                //
                bool fvcTopologyMatches = doesFVarTopologyMatch(patch, fvc);

                PatchInfo & fvcPatchInfo = fvcTopologyMatches
                                         ? patchInfo : fvarPatchInfo;

                if (!fvcTopologyMatches) {
                    identifyPatchTopology(patch, fvcPatchInfo, fvc);
                }
                assignPatchPointsAndStencils(patch, fvcPatchInfo,
                        arrayBuilder->fptr[fvc], *fvarLocalPointHelpers[fvc], fvc);

                //  Initialize and assign the face-varying PatchParam:
                PatchParam & fvcPatchParam = *arrayBuilder->fpptr[fvc];
                fvcPatchParam.Set(
                   patchParam.GetFaceId(),
                   patchParam.GetU(), patchParam.GetV(),
                   patchParam.GetDepth(),
                   patchParam.NonQuadRoot(),
                   fvcPatchInfo.paramBoundaryMask,
                   patchParam.GetTransition(),
                   fvcPatchInfo.isRegular);
            }
            arrayBuilder->fpptr[fvc] ++;
            arrayBuilder->fptr[fvc] +=
                _table->GetFVarPatchDescriptor(fvc).GetNumControlVertices();
        }

        if (_requiresVaryingPatches) {
            arrayBuilder->vptr += assignFacePoints(patch, arrayBuilder->vptr);
        }
    }

    //
    //  Finalizing and destroying StencilTable and other helpers:
    //
    if (_requiresLocalPoints) {
        if (_requiresVaryingPatches) {
            //  XXXX (barfowl) - note this is still managed by the vertex helper
            _table->_localPointVaryingStencils =
                vertexLocalPointHelper->AcquireStencilTableVarying();
            delete varyingLocalPointHelper;
        }

        _table->_localPointStencils =
            vertexLocalPointHelper->AcquireStencilTable();
        delete vertexLocalPointHelper;

        if (_requiresFVarPatches) {
            _table->_localPointFaceVaryingStencils.resize(_fvarChannelIndices.size());

            for (int fvc=0; fvc<(int)_fvarChannelIndices.size(); ++fvc) {
                _table->_localPointFaceVaryingStencils[fvc] =
                        fvarLocalPointHelpers[fvc]->AcquireStencilTable();
                delete fvarLocalPointHelpers[fvc];
            }
        }
    }
    if (_requiresLegacyGregoryTables) {
        _legacyGregoryHelper->FinalizeQuadOffsets(_table->_quadOffsetsTable);
        _legacyGregoryHelper->FinalizeVertexValence(_table->_vertexValenceTable,
                                       _levelVertOffsets[_refiner.GetMaxLevel()]);
    }
}

int
PatchTableBuilder::estimateLocalPointCount(
        LocalPointHelper::Options const & options,
        int fvarChannel) const {

    if ((fvarChannel >= 0) && isFVarChannelLinear(fvarChannel)) return 0;

    //
    //  Estimate the local points required for the Vertex topology in all cases
    //  as it may be used in the estimates for face-varying channels:
    //
    int estLocalPoints = 0;

    if (_requiresRegularLocalPoints) {
        //
        //  Either all regular patches are being converted to a non-native
        //  basis, or boundary points are being extrapolated in the regular
        //  basis.  The latter case typically involves a small fraction of
        //  patches and points, so we don't estimate any local points and
        //  leave it up to incremental allocation later to account for them.
        //
        int numPointsPerPatch = PatchDescriptor(
                _patchBuilder->GetRegularPatchType()).GetNumControlVertices();

        if (_patchBuilder->GetRegularPatchType() !=
                _patchBuilder->GetNativePatchType()) {
            estLocalPoints += _numRegularPatches * numPointsPerPatch;
        }
    } 
    if (_requiresIrregularLocalPoints) {
        bool oldEstimate = false;
        int numIrregularPatches = oldEstimate ?
            _refiner.GetLevel(_refiner.GetMaxLevel()).GetNumFaces() :
            _numIrregularPatches;

        //
        //  If converting to a basis other than regular, its difficult to
        //  predict the degree of point-sharing that may occur, and in general,
        //  the maximal benefit is small so we don't attempt to compensate for
        //  it here.  If converting to the same basis as regular, roughly half
        //  of the points of the patch are involved in approximating the
        //  irregularity (and cannot be shared) so don't include the source
        //  points that will be used in such patches.
        //
        int numPointsPerPatch = PatchDescriptor(
            _patchBuilder->GetIrregularPatchType()).GetNumControlVertices();

        if (options.reuseSourcePoints && (_patchBuilder->GetIrregularPatchType() ==
                                          _patchBuilder->GetNativePatchType())) {
            numPointsPerPatch /= 2;
        }
        estLocalPoints += numIrregularPatches * numPointsPerPatch;
    } 

    //
    //  Its difficult to estimate the differences in number of irregularities
    //  between the vertex topology and a face-varying channel without more
    //  detailed inspection.
    //
    //  A much higher number of fvar-values than vertices is an indication that
    //  the number of differences increases, and that generally lowers the
    //  number of irregular patches due to more regular patches on face-varying
    //  boundaries, but not always.  The use of some interpolation types, e.g.
    //  LINEAR_BOUNDARIES, combined with inf-sharp patches can increase the
    //  number of irregularities significantly.
    //
    if ((fvarChannel >= 0) && (_refiner.GetNumLevels() > 1)) {
        //
        //  We're seeing face-varying stencil table sizes about 1/4 the size of
        //  the vertex stencil table for what seem like typical cases...
        //
        //  Identify the ratio of fvar-values to vertices that typically leads
        //  to these reductions and reduce the number of expected local points
        //  proportionally.  Use the number of fvar-values at level 1:  level 0
        //  can be misleading as there can be many FEWER than the number of
        //  vertices, but subsequent levels will have at least one unique
        //  fvar-value per vertex, and later levels will have a much higher
        //  percentage shared as a result of refinement.
        //
        int fvc = getRefinerFVarChannel(fvarChannel);
        if (_refiner.GetLevel(1).GetNumFVarValues(fvc) >
                _refiner.GetLevel(1).GetNumVertices()) {
            //  Scale this based on ratio of fvar-values to vertices...
            float fvarReduction = 0.5;

            estLocalPoints = (int) ((float)estLocalPoints * fvarReduction);
        }
    }
    return estLocalPoints;
}


//
//  Member function definitions for the LocalPointHelper class:
//
PatchTableBuilder::LocalPointHelper::LocalPointHelper(
        TopologyRefiner const & refiner, Options const & options,
        int fvarChannel, int numLocalPointsExpected) :
            _refiner(refiner), _options(options), _fvarChannel(fvarChannel),
            _numLocalPoints(0), _stencilTable(0), _stencilTableVarying(0) {

    _localPointOffset = (_fvarChannel < 0)
                      ? _refiner.GetNumVerticesTotal()
                      : _refiner.GetNumFVarValuesTotal(_fvarChannel);

    if (_options.createStencilTable) {
        _stencilTable = new StencilTable(0);
        if (_stencilTable && (_fvarChannel < 0)) {
            _stencilTableVarying = new StencilTable(0);
        }
    }

    //
    //  Reserving space for the local-point stencils has been a source of
    //  problems in the past so we rely on the PatchTableBuilder to provide
    //  a reasonable estimate to the LocalPointHelper on construction.  For
    //  large meshes a limit on the size initially reserve is capped.
    //
    //  The average number of entries per stencil has been historically set
    //  at 16, which seemed high and was reduced on further investigation.
    //
    if (_stencilTable) {
        //  Historic note:  limits to 100M (=800M bytes) entries for reserved size
        size_t const MaxEntriesToReserve  = 100 * 1024 * 1024;
        size_t const AvgEntriesPerStencil = 9;  // originally 16

        size_t numStencilsExpected       = numLocalPointsExpected;
        size_t numStencilEntriesExpected = numStencilsExpected * AvgEntriesPerStencil;

        size_t numEntriesToReserve = std::min(numStencilEntriesExpected,
                                              MaxEntriesToReserve);
        if (numEntriesToReserve) {
            _stencilTable->reserve((int)numStencilsExpected,
                                   (int)numEntriesToReserve);
            if (_stencilTableVarying) {
                //  Varying stencils have only one entry per point
                _stencilTableVarying->reserve((int)numStencilsExpected,
                                              (int)numStencilsExpected);
            }
        }
    }
}

StencilTable *
PatchTableBuilder::LocalPointHelper::acquireStencilTable(
        StencilTable **stencilTableMember) {

    StencilTable * stencilTable = *stencilTableMember;

    if (stencilTable && (stencilTable->GetNumStencils() > 0)) {
        stencilTable->finalize();
    } else {
        delete stencilTable;
        stencilTable = 0;
    }
    *stencilTableMember = 0;
    return stencilTable;
}

PatchTableBuilder::LocalPointHelper::~LocalPointHelper() {

    delete _stencilTable;
    delete _stencilTableVarying;
}

Index
PatchTableBuilder::LocalPointHelper::findSharedCornerPoint(int levelIndex,
        Index vertIndex, Index localPointIndex) {

    if (_sharedCornerPoints.empty()) {
        _sharedCornerPoints.resize(_refiner.GetNumLevels());
    }

    IndexVector & vertexPoints = _sharedCornerPoints[levelIndex];
    if (vertexPoints.empty()) {
        Level const & level = _refiner.getLevel(levelIndex);

        if (_fvarChannel < 0) {
            vertexPoints.resize(level.getNumVertices(), INDEX_INVALID);
        } else {
            vertexPoints.resize(level.getNumFVarValues(_fvarChannel), INDEX_INVALID);
        }
    }

    Index & assignedIndex = vertexPoints[vertIndex];
    if (!IndexIsValid(assignedIndex)) {
        assignedIndex = localPointIndex;
    }
    return assignedIndex;
}

Index
PatchTableBuilder::LocalPointHelper::findSharedEdgePoint(int levelIndex,
         Index edgeIndex, int edgeEnd, Index localPointIndex) {

    if (_sharedEdgePoints.empty()) {
        _sharedEdgePoints.resize(_refiner.GetNumLevels());
    }

    IndexVector & edgePoints = _sharedEdgePoints[levelIndex];
    if (edgePoints.empty()) {
        edgePoints.resize(
                2 * _refiner.GetLevel(levelIndex).GetNumEdges(), INDEX_INVALID);
    }

    Index & assignedIndex = edgePoints[2 * edgeIndex + edgeEnd];
    if (!IndexIsValid(assignedIndex)) {
        assignedIndex = localPointIndex;
    } else {
    }
    return assignedIndex;
}

void
PatchTableBuilder::LocalPointHelper::appendLocalPointStencil(
    Matrix const & conversionMatrix,
    int            stencilRow,
    Index const    sourcePoints[],
    int            sourcePointOffset) {

    int               stencilSize   = conversionMatrix.GetRowSize(stencilRow);
    ConstArray<int>   matrixColumns = conversionMatrix.GetRowColumns(stencilRow);
    ConstArray<float> matrixWeights = conversionMatrix.GetRowElements(stencilRow);

    _stencilTable->_sizes.push_back(stencilSize);
    for (int i = 0; i < stencilSize; ++i) {
        _stencilTable->_weights.push_back(matrixWeights[i]);
        _stencilTable->_indices.push_back(
                sourcePoints[matrixColumns[i]] + sourcePointOffset);
    }
}

void
PatchTableBuilder::LocalPointHelper::appendLocalPointStencils(
    Matrix const & conversionMatrix,
    Index const    sourcePoints[],
    int            sourcePointOffset) {

    //
    //  Resize the StencilTable members to accomodate all rows and elements from
    //  the given set of points represented by the matrix
    //
    int numNewStencils = conversionMatrix.GetNumRows();
    int numNewElements = conversionMatrix.GetNumElements();

    size_t numOldStencils = _stencilTable->_sizes.size();
    size_t numOldElements = _stencilTable->_indices.size();

    //  Assign the sizes for the new stencils:
    _stencilTable->_sizes.resize(numOldStencils + numNewStencils);

    int * newSizes = &_stencilTable->_sizes[numOldStencils];
    for (int i = 0; i < numNewStencils; ++i) {
        newSizes[i] = conversionMatrix.GetRowSize(i);
    }

    //  Assign remapped indices for the stencils:
    _stencilTable->_indices.resize(numOldElements + numNewElements);

    int const * mtxIndices = &conversionMatrix.GetColumns()[0];
    int *       newIndices = &_stencilTable->_indices[numOldElements];

    for (int i = 0; i < numNewElements; ++i) {
        newIndices[i] = sourcePoints[mtxIndices[i]] + sourcePointOffset;
    }

    //  Copy the stencil weights direct from the matrix elements:
    _stencilTable->_weights.resize(numOldElements + numNewElements);

    float const * mtxWeights = &conversionMatrix.GetElements()[0];
    float *       newWeights = &_stencilTable->_weights[numOldElements];

    std::memcpy(newWeights, mtxWeights, numNewElements * sizeof(float));
}

//
//  XXXX (barfowl) - still unsure of how best to deal with varying stencils for
//  the new local points (or why they are even necessary)...  Assigning any
//  legitmate value seems to suffice (probably computed but no longer accessed).
//  These have historically been assigned using the face-vertex for the corner
//  of the face that is most associated with this patch-point.
//
//  If eventually warranted, consider a lookup table for each patch type that
//  returns the corner index most associated with each patch point.
//
namespace {
    inline int const *
    GetVaryingIndicesPerType(PatchDescriptor::Type type) {

        static int const linearIndices[4] = { 0, 1, 2, 3 };
        static int const bsplineIndices[] =
                { 0, 0, 1, 1, 0, 0, 1, 1, 3, 3, 2, 2, 3, 3, 2, 2 };
        static int const gregoryIndices[] =
                { 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3 };

        if (type == PatchDescriptor::GREGORY_BASIS) {
            return gregoryIndices;
        } else if (type == PatchDescriptor::REGULAR) {
            return bsplineIndices;
        } else if (type == PatchDescriptor::QUADS) {
            return linearIndices;
        }
        return 0;
    }
}

void
PatchTableBuilder::LocalPointHelper::appendLocalPointVaryingStencil(
    int const * varyingIndices, int patchPointIndex,
    Index const sourcePoints[], int sourcePointOffset) {

    Index varyingPoint =
        sourcePoints[varyingIndices[patchPointIndex]] + sourcePointOffset;

    _stencilTableVarying->_sizes.push_back(1);
    _stencilTableVarying->_indices.push_back(varyingPoint);
    _stencilTableVarying->_weights.push_back(1.0f);
}

void
PatchTableBuilder::LocalPointHelper::appendLocalPointVaryingStencils(
    int const *    varyingIndices, Matrix const & conversionMatrix,
    Index const    sourcePoints[], int sourcePointOffset) {

    for (int i = 0; i < conversionMatrix.GetNumRows(); ++i) {
        Index varyingPoint = sourcePoints[varyingIndices[i]] + sourcePointOffset;

        _stencilTableVarying->_sizes.push_back(1);
        _stencilTableVarying->_indices.push_back(varyingPoint);
        _stencilTableVarying->_weights.push_back(1.0f);
    }
}

namespace {
    typedef short ShareBits;

    static ShareBits const SHARE_CORNER     = 0x10;
    static ShareBits const SHARE_EDGE_BEGIN = 0x20;
    static ShareBits const SHARE_EDGE_END   = 0x40;
    static ShareBits const SHARE_EDGE       = SHARE_EDGE_BEGIN | SHARE_EDGE_END;
    static ShareBits const SHARE_INDEX_MASK = 0x0f;

    inline ShareBits const *
    GetShareBitsPerType(PatchDescriptor::Type type) {

        static ShareBits const linearQuadBits[5] = { 0x10, 0x11, 0x12, 0x13,
                                                     SHARE_CORNER };
        static ShareBits const gregoryQuadBits[21] = { 0x10, 0x20, 0x43, 0, 0,
                                                       0x11, 0x21, 0x40, 0, 0,
                                                       0x12, 0x22, 0x41, 0, 0,
                                                       0x13, 0x23, 0x42, 0, 0, 
                                                       SHARE_CORNER | SHARE_EDGE };

        if (type == PatchDescriptor::QUADS) {
            return linearQuadBits;
        } else if (type == PatchDescriptor::GREGORY_BASIS) {
            return gregoryQuadBits;
        }
        return 0;
    }
}

int
PatchTableBuilder::LocalPointHelper::AppendLocalPatchPoints(
    int levelIndex, Index faceIndex,
    Matrix const &        matrix,
    PatchDescriptor::Type patchType,
    Index const           sourcePoints[],
    int                   sourcePointOffset,
    Index                 patchPoints[]) {

    //
    //  If sharing local points, verify the type of this patch supports
    //  sharing and disable it if not:
    //
    int numPatchPoints  = matrix.GetNumRows();

    int firstNewLocalPoint = _localPointOffset + _numLocalPoints;
    int nextNewLocalPoint  = firstNewLocalPoint;

    ShareBits const* shareBitsPerPoint = _options.shareLocalPoints ?
                        GetShareBitsPerType(patchType) : 0;

    bool shareLocalPointsForThisPatch = (shareBitsPerPoint != 0);

    int const * varyingIndices =
        _stencilTableVarying ? GetVaryingIndicesPerType(patchType) : 0;

    //
    //  When point-sharing is not enabled, all patch points are generally
    //  new local points -- the exception to this occurs when "re-using"
    //  source points, i.e. the resulting patch can be a mixture of source
    //  and local points (typically when the irregular patch type is the
    //  same as the regular patch type native to the scheme).
    //
    if (!shareLocalPointsForThisPatch) {
        if (!_options.reuseSourcePoints) {
            appendLocalPointStencils(matrix, sourcePoints, sourcePointOffset);
            if (varyingIndices) {
                appendLocalPointVaryingStencils(
                    varyingIndices, matrix, sourcePoints, sourcePointOffset);
            }
            for (int i = 0; i < numPatchPoints; ++i) {
                patchPoints[i] = nextNewLocalPoint ++;
            }
        } else {
            for (int i = 0; i < numPatchPoints; ++i) {
                if (_options.reuseSourcePoints && (matrix.GetRowSize(i) == 1)) {
                    patchPoints[i] = sourcePoints[matrix.GetRowColumns(i)[0]]
                                   + sourcePointOffset;
                    continue;
                }
                appendLocalPointStencil(
                        matrix, i, sourcePoints, sourcePointOffset);

                if (varyingIndices) {
                    appendLocalPointVaryingStencil(
                        varyingIndices, i, sourcePoints, sourcePointOffset);
                }
                patchPoints[i] = nextNewLocalPoint ++;
            }
        }
    } else {
        //  Gather topology info according to the sharing for this patch type
        //
        Level const & level = _refiner.getLevel(levelIndex);

        ConstIndexArray fCorners;
        ConstIndexArray fVerts;
        ConstIndexArray fEdges;
        bool            fEdgeReversed[4];
        bool            fEdgeBoundary[4];

        ShareBits const shareBitsForType = shareBitsPerPoint[numPatchPoints];
        if (shareBitsForType) {
            if (shareBitsForType & SHARE_CORNER) {
                fCorners = (_fvarChannel < 0) 
                         ? level.getFaceVertices(faceIndex)
                         : level.getFaceFVarValues(faceIndex, _fvarChannel);
            }
            if (shareBitsForType & SHARE_EDGE) {
                fEdges = level.getFaceEdges(faceIndex);
                fVerts = (_fvarChannel < 0) ? fCorners
                                            : level.getFaceVertices(faceIndex);

                Level::ETag fEdgeTags[4];
                level.getFaceETags(faceIndex, fEdgeTags, _fvarChannel);

                for (int i = 0; i < fEdges.size(); ++i) {
                    fEdgeReversed[i] = level.getEdgeVertices(fEdges[i])[0] != fVerts[i];
                    fEdgeBoundary[i] = fEdgeTags[i]._boundary;
                }
            }
        }

        //  Inspect the sharing bits for each point -- if set, see if a local
        //  point for the corresponding vertex/fvar-value or edge was
        //  previously used:
        //
        for (int i = 0; i < numPatchPoints; ++i) {
            if (_options.reuseSourcePoints && (matrix.GetRowSize(i) == 1)) {
                patchPoints[i] = sourcePoints[matrix.GetRowColumns(i)[0]]
                               + sourcePointOffset;
                continue;
            }

            int patchPoint = nextNewLocalPoint;

            if (shareBitsPerPoint[i]) {
                int index = shareBitsPerPoint[i] & SHARE_INDEX_MASK;

                if (shareBitsPerPoint[i] & SHARE_CORNER) {
                    patchPoint = findSharedCornerPoint(
                        levelIndex, fCorners[index], nextNewLocalPoint);
                } else if (!fEdgeBoundary[index]) {
                    int edgeEnd = (((shareBitsPerPoint[i] & SHARE_EDGE_END) > 0) !=
                                   fEdgeReversed[index]);
                    patchPoint = findSharedEdgePoint(
                        levelIndex, fEdges[index], edgeEnd, nextNewLocalPoint);
                }
            }
            if (patchPoint == nextNewLocalPoint) {
                appendLocalPointStencil(matrix, i, sourcePoints, sourcePointOffset);
                if (varyingIndices) {
                    appendLocalPointVaryingStencil(
                        varyingIndices, i, sourcePoints, sourcePointOffset);
                }
                nextNewLocalPoint ++;
            }
            patchPoints[i] = patchPoint;
        }
    }

    int numNewLocalPoints = nextNewLocalPoint - firstNewLocalPoint;
    _numLocalPoints += numNewLocalPoints;
    return numNewLocalPoints;
}


//
//  Member function definitions for the LegacyGregoryHelper class:
//
void
PatchTableBuilder::LegacyGregoryHelper::AddPatchFace(int level, Index face) {

    if (_refiner.getLevel(level).getFaceCompositeVTag(face)._boundary) {
        _boundaryFaceIndices.push_back(face);
    } else {
        _interiorFaceIndices.push_back(face);
    }
}

void
PatchTableBuilder::LegacyGregoryHelper::FinalizeQuadOffsets(
        PatchTable::QuadOffsetsTable & qTable) {

    size_t numInteriorPatches = _interiorFaceIndices.size();
    size_t numBoundaryPatches = _boundaryFaceIndices.size();
    size_t numTotalPatches    = numInteriorPatches + numBoundaryPatches;

    struct QuadOffset {
        static int Assign(Level const& level, Index faceIndex,
                          unsigned int offsets[]) {
            ConstIndexArray fVerts = level.getFaceVertices(faceIndex);

            for (int i = 0; i < 4; ++i) {
                ConstIndexArray vFaces = level.getVertexFaces(fVerts[i]);
                int faceInVFaces = vFaces.FindIndex(faceIndex);

                // we have to use number of incident edges to modulo the local
                // index as there could be 2 consecutive edges in the face
                // belonging to the patch
                int vOffset0 = faceInVFaces;
                int vOffset1 = (faceInVFaces + 1) %
                                    level.getVertexEdges(fVerts[i]).size();
                offsets[i] = vOffset0 | (vOffset1 << 8);
            }
            return 4;
        }
    };

    if (numTotalPatches > 0) {
        qTable.resize(numTotalPatches*4);

        // all patches assumed to be at the last level
        Level const &maxLevel = _refiner.getLevel(_refiner.GetMaxLevel());

        PatchTable::QuadOffsetsTable::value_type *p = &(qTable[0]);
        for (size_t i = 0; i < numInteriorPatches; ++i) {
            p += QuadOffset::Assign(maxLevel, _interiorFaceIndices[i], p);
        }
        for (size_t i = 0; i < numBoundaryPatches; ++i) {
            p += QuadOffset::Assign(maxLevel, _boundaryFaceIndices[i], p);
        }
    }
}

void
PatchTableBuilder::LegacyGregoryHelper::FinalizeVertexValence(
        PatchTable::VertexValenceTable & vTable, int lastLevelOffset) {

    //  Populate the "vertex valence" table for Gregory patches -- this table
    //  contains the one-ring of vertices around each vertex.  Currently it is
    //  extremely wasteful for the following reasons:
    //      - it allocates 2*maxvalence+1 for ALL vertices
    //      - it initializes the one-ring for ALL vertices
    //  We use the full size expected (not sure what else relies on that) but
    //  we avoid initializing the vast majority of vertices that are not
    //  associated with gregory patches -- only populating those in the last
    //  level (an older version attempted to avoid vertices not involved with
    //  Gregory patches)
    //
    int vWidth = 2*_refiner.GetMaxValence() + 1;

    vTable.resize((long)_refiner.GetNumVerticesTotal() * vWidth);

    Level const & lastLevel = _refiner.getLevel(_refiner.GetMaxLevel());

    int * vTableEntry = &vTable[lastLevelOffset * vWidth];

    for (int vIndex = 0; vIndex < lastLevel.getNumVertices(); ++vIndex) {
        //  Gather the one-ring around the vertex and set its resulting size
        //  (note negative size used to distinguish between boundary/interior):
        //
        int *ringDest = vTableEntry + 1;
        int  ringSize =
                lastLevel.gatherQuadRegularRingAroundVertex(vIndex, ringDest);

        for (int j = 0; j < ringSize; ++j) {
            ringDest[j] += lastLevelOffset;
        }
        if (ringSize & 1) {
            // boundary: duplicate end vertex index and store negative valence
            ringSize++;
            vTableEntry[ringSize]=vTableEntry[ringSize-1];
            vTableEntry[0] = -ringSize/2;
        } else {
            vTableEntry[0] = ringSize/2;
        }
        vTableEntry += vWidth;
    }
}


//
//  The sole public PatchTableFactory method to create a PatchTable -- deferring
//  to the PatchTableBuilder implementation
//
PatchTable *
PatchTableFactory::Create(TopologyRefiner const & refiner, Options options) {

    PatchTableBuilder builder(refiner, options);

    if (refiner.IsUniform()) {
        builder.BuildUniform();
    } else {
        builder.BuildAdaptive();
    }
    return builder.GetPatchTable();
}


//
//  Implementation of PatchTableFactory::PatchFaceTag -- unintentionally
//  exposed to the public interface, no longer internally used and so
//  planned for deprecation
//
void
PatchTableFactory::PatchFaceTag::clear() {
    std::memset(this, 0, sizeof(*this));
}

void
PatchTableFactory::PatchFaceTag::assignTransitionPropertiesFromEdgeMask(
        int tMask) {

    _transitionMask = tMask;
}

void
PatchTableFactory::PatchFaceTag::assignBoundaryPropertiesFromEdgeMask(
        int eMask) {

    static int const edgeMaskToCount[16] =
        { 0, 1, 1, 2, 1, -1, 2, -1, 1, 2, -1, -1, 2, -1, -1, -1 };
    static int const edgeMaskToIndex[16] =
        { -1, 0, 1, 1, 2, -1, 2, -1, 3, 0, -1, -1, 3, -1, -1,-1 };

    assert(edgeMaskToCount[eMask] != -1);
    assert(edgeMaskToIndex[eMask] != -1);

    _boundaryMask    = eMask;
    _hasBoundaryEdge = (eMask > 0);

    _boundaryCount = edgeMaskToCount[eMask];
    _boundaryIndex = edgeMaskToIndex[eMask];
}

void
PatchTableFactory::PatchFaceTag::assignBoundaryPropertiesFromVertexMask(
        int vMask) {

    // This is only intended to support the case of a single boundary vertex
    // with no boundary edges, which can only occur with an irregular vertex

    static int const singleBitVertexMaskToCount[16] =
        { 0, 1, 1, -1, 1, -1 , -1, -1, 1, -1 , -1, -1, -1, -1 , -1, -1 };
    static int const singleBitVertexMaskToIndex[16] =
        { 0, 0, 1, -1, 2, -1 , -1, -1, 3, -1 , -1, -1, -1, -1 , -1, -1 };

    assert(_hasBoundaryEdge == false);
    assert(singleBitVertexMaskToCount[vMask] != -1);
    assert(singleBitVertexMaskToIndex[vMask] != -1);

    _boundaryMask = vMask;

    _boundaryCount = singleBitVertexMaskToCount[vMask];
    _boundaryIndex = singleBitVertexMaskToIndex[vMask];
}

} // end namespace Far

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv
