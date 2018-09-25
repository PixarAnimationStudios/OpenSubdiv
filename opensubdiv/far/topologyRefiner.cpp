//
//   Copyright 2014 DreamWorks Animation LLC.
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
#include "../far/topologyRefiner.h"
#include "../far/error.h"
#include "../vtr/fvarLevel.h"
#include "../vtr/sparseSelector.h"
#include "../vtr/quadRefinement.h"
#include "../vtr/triRefinement.h"

#include <cassert>
#include <cstdio>


namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Far {

//
//  Relatively trivial construction/destruction -- the base level (level[0]) needs
//  to be explicitly initialized after construction and refinement then applied
//
TopologyRefiner::TopologyRefiner(Sdc::SchemeType schemeType, Sdc::Options schemeOptions) :
    _subdivType(schemeType),
    _subdivOptions(schemeOptions),
    _isUniform(true),
    _hasHoles(false),
    _maxLevel(0),
    _uniformOptions(0),
    _adaptiveOptions(0),
    _totalVertices(0),
    _totalEdges(0),
    _totalFaces(0),
    _totalFaceVertices(0),
    _maxValence(0),
    _baseLevelOwned(true) {

    //  Need to revisit allocation scheme here -- want to use smart-ptrs for these
    //  but will probably have to settle for explicit new/delete...
    _levels.reserve(10);
    _levels.push_back(new Vtr::internal::Level);
    _farLevels.reserve(10);
    assembleFarLevels();
}

//
//  The copy constructor is protected and used by the factory to create a new instance
//  from only the base level of the given instance -- it does not create a full copy.
//  So members reflecting any refinement are default-initialized while those dependent
//  on the base level are copied or explicitly initialized after its assignment.
//
TopologyRefiner::TopologyRefiner(TopologyRefiner const & source) :
    _subdivType(source._subdivType),
    _subdivOptions(source._subdivOptions),
    _isUniform(true),
    _hasHoles(source._hasHoles),
    _maxLevel(0),
    _uniformOptions(0),
    _adaptiveOptions(0),
    _baseLevelOwned(false) {

    _levels.reserve(10);
    _levels.push_back(source._levels[0]);
    initializeInventory();

    _farLevels.reserve(10);
    assembleFarLevels();
}


TopologyRefiner::~TopologyRefiner() {

    for (int i=0; i<(int)_levels.size(); ++i) {
        if ((i > 0) || _baseLevelOwned) delete _levels[i];
    }

    for (int i=0; i<(int)_refinements.size(); ++i) {
        delete _refinements[i];
    }
}

void
TopologyRefiner::Unrefine() {

    if (_levels.size()) {
        for (int i=1; i<(int)_levels.size(); ++i) {
            delete _levels[i];
        }
        _levels.resize(1);
        initializeInventory();
    }
    for (int i=0; i<(int)_refinements.size(); ++i) {
        delete _refinements[i];
    }
    _refinements.clear();

    assembleFarLevels();
}


//
//  Initializing and updating the component inventory:
//
void
TopologyRefiner::initializeInventory() {

    if (_levels.size()) {
        assert(_levels.size() == 1);

        Vtr::internal::Level const & baseLevel = *_levels[0];

        _totalVertices     = baseLevel.getNumVertices();
        _totalEdges        = baseLevel.getNumEdges();
        _totalFaces        = baseLevel.getNumFaces();
        _totalFaceVertices = baseLevel.getNumFaceVerticesTotal();

        _maxValence = baseLevel.getMaxValence();
    } else {
        _totalVertices     = 0;
        _totalEdges        = 0;
        _totalFaces        = 0;
        _totalFaceVertices = 0;

        _maxValence = 0;
    }
}

void
TopologyRefiner::updateInventory(Vtr::internal::Level const & newLevel) {

    _totalVertices     += newLevel.getNumVertices();
    _totalEdges        += newLevel.getNumEdges();
    _totalFaces        += newLevel.getNumFaces();
    _totalFaceVertices += newLevel.getNumFaceVerticesTotal();

    _maxValence = std::max(_maxValence, newLevel.getMaxValence());
}

void
TopologyRefiner::appendLevel(Vtr::internal::Level & newLevel) {

    _levels.push_back(&newLevel);

    updateInventory(newLevel); 
}

void
TopologyRefiner::appendRefinement(Vtr::internal::Refinement & newRefinement) {

    _refinements.push_back(&newRefinement);
}

void
TopologyRefiner::assembleFarLevels() {

    _farLevels.resize(_levels.size());

    _farLevels[0]._refToParent = 0;
    _farLevels[0]._level       = _levels[0];
    _farLevels[0]._refToChild  = 0;

    int nRefinements = (int)_refinements.size();
    if (nRefinements) {
        _farLevels[0]._refToChild = _refinements[0];

        for (int i = 1; i < nRefinements; ++i) {
            _farLevels[i]._refToParent = _refinements[i - 1];
            _farLevels[i]._level       = _levels[i];
            _farLevels[i]._refToChild  = _refinements[i];;
        }

        _farLevels[nRefinements]._refToParent = _refinements[nRefinements - 1];
        _farLevels[nRefinements]._level       = _levels[nRefinements];
        _farLevels[nRefinements]._refToChild  = 0;
    }
}


//
//  Accessors to the topology information:
//
int
TopologyRefiner::GetNumFVarValuesTotal(int channel) const {
    int sum = 0;
    for (int i = 0; i < (int)_levels.size(); ++i) {
        sum += _levels[i]->getNumFVarValues(channel);
    }
    return sum;
}


//
//  Main refinement method -- allocating and initializing levels and refinements:
//
void
TopologyRefiner::RefineUniform(UniformOptions options) {

    if (_levels[0]->getNumVertices() == 0) {
        Error(FAR_RUNTIME_ERROR,
            "Failure in TopologyRefiner::RefineUniform() -- base level is uninitialized.");
        return;
    }
    if (_refinements.size()) {
        Error(FAR_RUNTIME_ERROR,
            "Failure in TopologyRefiner::RefineUniform() -- previous refinements already applied.");
        return;
    }

    //
    //  Allocate the stack of levels and the refinements between them:
    //
    _uniformOptions = options;

    _isUniform = true;
    _maxLevel = options.refinementLevel;

    Sdc::Split splitType = Sdc::SchemeTypeTraits::GetTopologicalSplitType(_subdivType);

    //
    //  Initialize refinement options for Vtr -- adjusting full-topology for the last level:
    //
    Vtr::internal::Refinement::Options refineOptions;
    refineOptions._sparse         = false;
    refineOptions._faceVertsFirst = options.orderVerticesFromFacesFirst;

    for (int i = 1; i <= (int)options.refinementLevel; ++i) {
        refineOptions._minimalTopology =
            options.fullTopologyInLastLevel ? false : (i == (int)options.refinementLevel);

        Vtr::internal::Level& parentLevel = getLevel(i-1);
        Vtr::internal::Level& childLevel  = *(new Vtr::internal::Level);

        Vtr::internal::Refinement* refinement = 0;
        if (splitType == Sdc::SPLIT_TO_QUADS) {
            refinement = new Vtr::internal::QuadRefinement(parentLevel, childLevel, _subdivOptions);
        } else {
            refinement = new Vtr::internal::TriRefinement(parentLevel, childLevel, _subdivOptions);
        }
        refinement->refine(refineOptions);

        appendLevel(childLevel);
        appendRefinement(*refinement);
    }
    assembleFarLevels();
}

//
//  Internal utility class and function supporting feature adaptive selection of faces...
//
namespace internal {
    //
    //  FeatureMask is a simple set of bits identifying features to be selected during a level of
    //  adaptive refinement.  Adaptive refinement options passed the Refiner are interpreted as a
    //  specific set of features defined here.  Given options to reduce faces generated at deeper
    //  levels, a method to "reduce" the set of features is also provided here.
    //
    //  This class was specifically not nested in TopologyRefiner to allow simple non-class methods
    //  to make use of it in the core selection methods.  Those selection methods were similarly
    //  made non-class methods to ensure they conform to the feature set defined by the FeatureMask
    //  and not some internal class state.
    //
    class FeatureMask {
    public:
        typedef TopologyRefiner::AdaptiveOptions Options;
        typedef unsigned int                     int_type;

        void Clear()         { *((int_type*)this) = 0; }
        bool IsEmpty() const { return *((int_type*)this) == 0; }

        FeatureMask() { Clear(); }
        FeatureMask(Options const & options, Sdc::SchemeType sType) { Clear(); InitializeFeatures(options, sType); }

        //  These are the two primary methods intended for use -- intialization via a set of Options
        //  and reduction of the subsequent feature set (which presumes prior initialization with the
        //  same set as give)
        //
        void InitializeFeatures(Options const & options, Sdc::SchemeType sType);
        void ReduceFeatures(    Options const & options);

    public:
        int_type selectXOrdinaryInterior : 1;
        int_type selectXOrdinaryBoundary : 1;

        int_type selectSemiSharpSingle    : 1;
        int_type selectSemiSharpNonSingle : 1;

        int_type selectInfSharpRegularCrease   : 1;
        int_type selectInfSharpRegularCorner   : 1;
        int_type selectInfSharpIrregularDart   : 1;
        int_type selectInfSharpIrregularCrease : 1;
        int_type selectInfSharpIrregularCorner : 1;

        int_type selectNonManifold  : 1;
        int_type selectFVarFeatures : 1;
    };

    void
    FeatureMask::InitializeFeatures(Options const & options, Sdc::SchemeType subdType) {

        //
        //  Support for the "single-crease patch" case is limited to the subdivision scheme
        //  (currently only Catmull-Clark).  It has historically been applied to both semi-
        //  sharp and inf-sharp creases -- the semi-sharp application is still relevant,
        //  but the inf-sharp has been superceded.
        //
        //  The inf-sharp single-crease case now corresponds to an inf-sharp regular crease
        //  in the interior -- and since such regular creases on the boundary are never
        //  considered for selection (just as interior smoot regular faces are not), this
        //  feature is only relevant for the interior case.  So aside from it being used
        //  when regular inf-sharp features are all selected, it can also be used for the
        //  single-crease case.
        //
        bool useSingleCreasePatch = options.useSingleCreasePatch &&
                                    (Sdc::SchemeTypeTraits::GetRegularFaceSize(subdType) == 4);

        //  Extra-ordinary features (independent of the inf-sharp options):
        selectXOrdinaryInterior = true;
        selectXOrdinaryBoundary = true;

        //  Semi-sharp features -- the regular single crease case and all others:
        selectSemiSharpSingle    = !useSingleCreasePatch;
        selectSemiSharpNonSingle = true;

        //  Inf-sharp features -- boundary extra-ordinary vertices are irreg creases:
        selectInfSharpRegularCrease   = !(options.useInfSharpPatch || useSingleCreasePatch);
        selectInfSharpRegularCorner   = !options.useInfSharpPatch;
        selectInfSharpIrregularDart   = true;
        selectInfSharpIrregularCrease = true;
        selectInfSharpIrregularCorner = true;

        selectNonManifold  = true;
        selectFVarFeatures = options.considerFVarChannels;
    }

    void
    FeatureMask::ReduceFeatures(Options const & options) {

        //  Disable typical xordinary vertices:
        selectXOrdinaryInterior = false;
        selectXOrdinaryBoundary = false;

        //  If minimizing inf-sharp patches, disable all but sharp/corner irregularities
        if (options.useInfSharpPatch) {
            selectInfSharpRegularCrease    = false;
            selectInfSharpRegularCorner    = false;
            selectInfSharpIrregularDart    = false;
            selectInfSharpIrregularCrease  = false;
        }
    }
} // end namespace internal

void
TopologyRefiner::RefineAdaptive(AdaptiveOptions options,
                                ConstIndexArray baseFacesToRefine) {

    if (_levels[0]->getNumVertices() == 0) {
        Error(FAR_RUNTIME_ERROR,
            "Failure in TopologyRefiner::RefineAdaptive() -- base level is uninitialized.");
        return;
    }
    if (_refinements.size()) {
        Error(FAR_RUNTIME_ERROR,
            "Failure in TopologyRefiner::RefineAdaptive() -- previous refinements already applied.");
        return;
    }

    //
    //  Initialize member and local variables from the adaptive options:
    //
    _isUniform = false;
    _adaptiveOptions = options;

    //
    //  Initialize the feature-selection options based on given options -- with two sets
    //  of levels isolating different sets of features, initialize the two feature sets
    //  up front and use the appropriate one for each level:
    //
    int shallowLevel = std::min<int>(options.secondaryLevel, options.isolationLevel);
    int deeperLevel  = options.isolationLevel;

    int potentialMaxLevel = deeperLevel;

    internal::FeatureMask moreFeaturesMask(options, _subdivType);
    internal::FeatureMask lessFeaturesMask = moreFeaturesMask;

    if (shallowLevel < potentialMaxLevel) {
        lessFeaturesMask.ReduceFeatures(options);
    }

    //
    //  Features are not relevant to schemes whose influence does not extend beyond the
    //  face -- only irregular faces matter in such cases so clear all other features.
    //  If face-varying channels are considered, make sure non-linear channels are present
    //  and turn off consideration if none present:
    //
    if (Sdc::SchemeTypeTraits::GetLocalNeighborhoodSize(_subdivType) == 0) {
        moreFeaturesMask.Clear();
        lessFeaturesMask.Clear();
    } else if (moreFeaturesMask.selectFVarFeatures) {
        bool nonLinearChannelsPresent = false;
        for (int channel = 0; channel < _levels[0]->getNumFVarChannels(); ++channel) {
            nonLinearChannelsPresent |= !_levels[0]->getFVarLevel(channel).isLinear();
        }
        if (!nonLinearChannelsPresent) {
            moreFeaturesMask.selectFVarFeatures = false;
            lessFeaturesMask.selectFVarFeatures = false;
        }
    }

    //
    //  Initialize refinement options for Vtr -- full topology is always generated in
    //  the last level as expected usage is for patch retrieval:
    //
    Vtr::internal::Refinement::Options refineOptions;

    refineOptions._sparse          = true;
    refineOptions._minimalTopology = false;
    refineOptions._faceVertsFirst  = options.orderVerticesFromFacesFirst;

    Sdc::Split splitType = Sdc::SchemeTypeTraits::GetTopologicalSplitType(_subdivType);

    for (int i = 1; i <= potentialMaxLevel; ++i) {

        Vtr::internal::Level& parentLevel     = getLevel(i-1);
        Vtr::internal::Level& childLevel      = *(new Vtr::internal::Level);

        Vtr::internal::Refinement* refinement = 0;
        if (splitType == Sdc::SPLIT_TO_QUADS) {
            refinement = new Vtr::internal::QuadRefinement(parentLevel, childLevel, _subdivOptions);
        } else {
            refinement = new Vtr::internal::TriRefinement(parentLevel, childLevel, _subdivOptions);
        }

        //
        //  Initialize a Selector to mark a sparse set of components for refinement -- choose
        //  the feature selection mask appropriate to the level:
        //
        Vtr::internal::SparseSelector selector(*refinement);

        internal::FeatureMask const & levelFeatures = (i <= shallowLevel) ? moreFeaturesMask
                                                                          : lessFeaturesMask;

        if (i == 1) {
            selectFeatureAdaptiveComponents(selector, levelFeatures, baseFacesToRefine);
        } else {
            selectFeatureAdaptiveComponents(selector, levelFeatures, ConstIndexArray());
        }

        if (selector.isSelectionEmpty()) {
            delete refinement;
            delete &childLevel;
            break;
        } else {
            refinement->refine(refineOptions);

            appendLevel(childLevel);
            appendRefinement(*refinement);
        }
    }
    _maxLevel = (unsigned int) _refinements.size();

    assembleFarLevels();
}

//
//  Local utility functions for selecting features in faces for adaptive refinement:
//
namespace {
    //
    //  First are a couple of low-level utility methods to perform the same analysis
    //  at a corner or the entire face for specific detection of inf-sharp or boundary
    //  features.  These are shared between the analysis of the main face and those in
    //  face-varying channels (which only differ from the main face in the presence of
    //  face-varying boundaries).
    //
    //  The first can be applied equally to an individual corner or to the entire face
    //  (using its composite tag).  The second applies to the entire face, making use
    //  of the first, and is the main entry point for dealng with inf-sharp features.
    //
    //  Note we can use the composite tag here even though it arises from all corners
    //  of the face and so does not represent a specific corner.  When at least one
    //  smooth interior vertex exists, it limits the combinations that can exist on the
    //  remaining corners (though quads and tris cannot be treated equally here).
    //
    //  If any inf-sharp features are to be selected, identify them first as irregular
    //  or not, then qualify them more specifically.  (Remember that a regular vertex
    //  may have its neighboring faces partitioned into irregular regions in the
    //  presence of inf-sharp edges.  Similarly an irregular vertex may have its
    //  neighborhood partitioned into regular regions.)
    //
    inline bool
    doesInfSharpVTagHaveFeatures(Vtr::internal::Level::VTag compVTag,
                                 internal::FeatureMask const & featureMask) {

        //  Note that even though the given VTag may represent an individual corner, we
        //  use more general bitwise tests here (particularly the Rule) so that we can
        //  pass in a composite tag for the entire face and have the same tests applied:
        //
        if (compVTag._infIrregular) {
            if (compVTag._rule & Sdc::Crease::RULE_CORNER) {
                return featureMask.selectInfSharpIrregularCorner;
            } else if (compVTag._rule & Sdc::Crease::RULE_CREASE) {
                return compVTag._boundary ? featureMask.selectXOrdinaryBoundary :
                                            featureMask.selectInfSharpIrregularCrease;
            } else if (compVTag._rule & Sdc::Crease::RULE_DART) {
                return featureMask.selectInfSharpIrregularDart;
            }
        } else if (compVTag._boundary) {
            //  Remember that regular boundary features should never be selected, except
            //  for a boundary crease sharpened (and so a Corner) by an interior edge:

            if (compVTag._rule & Sdc::Crease::RULE_CORNER) {
                return compVTag._corner ? false : featureMask.selectInfSharpRegularCorner;
            } else {
                return false;
            }
        } else {
            if (compVTag._rule & Sdc::Crease::RULE_CORNER) {
                return featureMask.selectInfSharpRegularCorner;
            } else {
                return featureMask.selectInfSharpRegularCrease;
            }
        }
        return false;
    }

    inline bool
    doesInfSharpFaceHaveFeatures(Vtr::internal::Level::VTag compVTag,
                                 Vtr::internal::Level::VTag vTags[], int numVerts,
                                 internal::FeatureMask const & featureMask) {
        //
        //  For quads, if at least one smooth corner of a regular face, features
        //  are isolated enough to make use of the composite tag alone.
        //
        //  For tris, the presence of boundaries creates more ambiguity, so we
        //  need to exclude that case and inspect corner features individually.
        //
        bool atLeastOneSmoothCorner = (compVTag._rule & Sdc::Crease::RULE_SMOOTH);
        if (numVerts == 4) {
            if (atLeastOneSmoothCorner) {
                return doesInfSharpVTagHaveFeatures(compVTag, featureMask);
            } else {
                //  Construction of quad patches was originally written to require
                //  isolation of boundary features -- some of the core dependencies
                //  are being removed but other subtle dependencies remain and need
                //  a more concerted effort to deal with.  Once all such dependencies
                //  have been removed, the condition to immediately select the face
                //  here can be removed in favor of closer inspection of each corner.
                //
                bool quadPatchesRequireBoundaryIsolation = true;
                if (quadPatchesRequireBoundaryIsolation) {
                    return true;
                } else {
                    for (int i = 0; i < 4; ++i) {
                        if (!(vTags[i]._rule & Sdc::Crease::RULE_SMOOTH)) {
                            if (doesInfSharpVTagHaveFeatures(vTags[i], featureMask)) {
                                return true;
                            }
                        }
                    }
                    return false;
                }
            }
        } else {
            if (atLeastOneSmoothCorner && !compVTag._boundary) {
                return doesInfSharpVTagHaveFeatures(compVTag, featureMask);
            } else {
                for (int i = 0; i < 3; ++i) {
                    if (!(vTags[i]._rule & Sdc::Crease::RULE_SMOOTH)) {
                        if (doesInfSharpVTagHaveFeatures(vTags[i], featureMask)) {
                            return true;
                        }
                    }
                }
                return false;
            }
        }
    }

    //
    //  This is the core method/function for analyzing a face and deciding whether or not
    //  to included it during feature-adaptive refinement.
    //
    //  Topological analysis of the face exploits tags that are applied to corner vertices
    //  and carried through the refinement hierarchy.  The tags were designed with this
    //  in mind and also to be combined via bitwise-OR to make collective decisions about
    //  the neighborhood of the entire face.
    //
    //  After a few trivial acceptances/rejections, feature detection is divided up into
    //  semi-sharp and inf-sharp cases -- note that both may be present, but semi-sharp
    //  features have an implicit precedence until they decay and so are handled first.
    //  They are also fairly trivial to deal with (most often requiring selection) while
    //  the presence of boundaries and additional options complicates the inf-sharp case.
    //  Since the inf-sharp logic needs to be applied in face-varying cases, it exists in
    //  a separate method.
    //
    //  This was originally written specific to the quad-centric Catmark scheme and was
    //  since generalized to support Loop given the enhanced tagging of components based
    //  on the scheme.  Any enhancements here should be aware of the intended generality.
    //  Ultimately it may not be worth trying to keep this general and we will be better
    //  off specializing it for each scheme.  The fact that this method is intimately tied
    //  to patch generation also begs for it to become part of a class that encompasses
    //  both the feature adaptive tagging and the identification of the intended patches
    //  that result from it.
    //
    bool
    doesFaceHaveFeatures(Vtr::internal::Level const& level, Index face,
                         internal::FeatureMask const & featureMask) {

        if (featureMask.IsEmpty()) return false;

        using Vtr::internal::Level;

        ConstIndexArray fVerts = level.getFaceVertices(face);

        //  Gather and combine the VTags:
        Level::VTag vTags[4];
        level.getFaceVTags(face, vTags);

        //  Incomplete faces (incomplete neighborhood) are never candidates for inclusion:
        Level::VTag compFaceVTag = Level::VTag::BitwiseOr(vTags, fVerts.size());
        if (compFaceVTag._incomplete) {
            return false;
        }

        //  Select non-manifold features if specified, otherwise treat as inf-sharp:
        if (compFaceVTag._nonManifold && featureMask.selectNonManifold) {
            return true;
        }

        //  Select (smooth) xord vertices if specified, boundaries handled with inf-sharp:
        if (compFaceVTag._xordinary && featureMask.selectXOrdinaryInterior) {
            if (compFaceVTag._rule == Sdc::Crease::RULE_SMOOTH) {
                return true;
            } else if (level.getDepth() < 2) {
                for (int i = 0; i < fVerts.size(); ++i) {
                    if (vTags[i]._xordinary && (vTags[i]._rule == Sdc::Crease::RULE_SMOOTH)) {
                        return true;
                    }
                }
            }
        }

        //  If all smooth corners, no remaining features to select (x-ordinary dealt with):
        if (compFaceVTag._rule == Sdc::Crease::RULE_SMOOTH) {
            return false;
        }

        //  Semi-sharp features -- select all immediately or test the single-crease case:
        if (compFaceVTag._semiSharp || compFaceVTag._semiSharpEdges) {
            if (featureMask.selectSemiSharpSingle && featureMask.selectSemiSharpNonSingle) {
                return true;
            } else if (level.isSingleCreasePatch(face)) {
                return featureMask.selectSemiSharpSingle;
            } else {
                return featureMask.selectSemiSharpNonSingle;
            }
        }

        //  Inf-sharp features (including boundaries) -- delegate to shared method:
        if (compFaceVTag._infSharp || compFaceVTag._infSharpEdges) {
            return doesInfSharpFaceHaveFeatures(compFaceVTag, vTags, fVerts.size(), featureMask);
        }
        return false;
    }

    //
    //  Analyzing the face-varying topology for selection is considerably simpler that
    //  for the face and its vertices -- in part due to the fact that these faces lie on
    //  face-varying boundaries, and also due to assumptions about prior inspection:
    //
    //      - it is assumed the face topologgy does not match, so the face must lie on
    //        a FVar boundary, i.e. inf-sharp
    //
    //      - it is assumed the face vertices were already inspected, so cases such as
    //        semi-sharp or smooth interior x-ordinary features have already triggered
    //        selection
    //
    //  That leaves the inspection of inf-sharp features, for the tags from the face
    //  varying channel -- code that is shared with the main face.
    //
    bool
    doesFaceHaveDistinctFaceVaryingFeatures(Vtr::internal::Level const& level, Index face,
                                internal::FeatureMask const & featureMask, int fvarChannel) {

        using Vtr::internal::Level;

        ConstIndexArray fVerts = level.getFaceVertices(face);

        assert(!level.doesFaceFVarTopologyMatch(face, fvarChannel));

        //  We can't use the composite VTag for the face here as it only includes the FVar
        //  values specific to this face.  We need to account for all FVar values around
        //  each corner of the face -- including those in potentially completely disjoint
        //  sets -- to ensure that adjacent faces remain compatibly refined (i.e. differ
        //  by only one level), so we use the composite tags for the corner vertices:
        //
        Level::VTag vTags[4];

        for (int i = 0; i < fVerts.size(); ++i) {
            vTags[i] = level.getVertexCompositeFVarVTag(fVerts[i], fvarChannel);
        }
        Level::VTag compVTag = Level::VTag::BitwiseOr(vTags, fVerts.size());

        //  Select non-manifold features if specified, otherwise treat as inf-sharp:
        if (compVTag._nonManifold && featureMask.selectNonManifold) {
            return true;
        }

        //  Any remaining locally extra-ordinary face-varying boundaries warrant selection:
        if (compVTag._xordinary && featureMask.selectXOrdinaryInterior) {
            return true;
        }

        //  Given faces with differing FVar topology are on boundaries, defer to inf-sharp:
        return doesInfSharpFaceHaveFeatures(compVTag, vTags, fVerts.size(), featureMask);
    }

} // end namespace

//
//   Method for selecting components for sparse refinement based on the feature-adaptive needs
//   of patch generation.
//
//   It assumes we have a freshly initialized SparseSelector (i.e. nothing already selected)
//   and will select all relevant topological features for inclusion in the subsequent sparse
//   refinement.
//
namespace {
    bool
    faceNextToIrregFace(Vtr::internal::Level const& level, Index faceIndex, int regFaceSize) {

        //  Wish there was a better way to determine this -- we must inspect the sizes
        //  of all incident faces of all corners of the face...
        //
        Vtr::ConstIndexArray faceVerts = level.getFaceVertices(faceIndex);

        for (int i = 0; i < faceVerts.size(); ++i) {
            ConstIndexArray vertFaces = level.getVertexFaces(faceVerts[i]);

            for (int j = 0; j < vertFaces.size(); ++j) {
                if (level.getFaceVertices(vertFaces[j]).size() != regFaceSize) {
                    return true;
                }
            }
        }
        return false;
    }
}

void
TopologyRefiner::selectFeatureAdaptiveComponents(Vtr::internal::SparseSelector& selector,
                                                 internal::FeatureMask const & featureMask,
                                                 ConstIndexArray facesToRefine) {

    Vtr::internal::Level const& level = selector.getRefinement().parent();
    int levelDepth = level.getDepth();

    bool selectIrregularFaces = (levelDepth == 0);
    if (featureMask.IsEmpty() && !selectIrregularFaces) return;

    int numFVarChannels = featureMask.selectFVarFeatures ? level.getNumFVarChannels() : 0;
    int regularFaceSize = selector.getRefinement().getRegularFaceSize();
    int neighborhood    = Sdc::SchemeTypeTraits::GetLocalNeighborhoodSize(_subdivType);

    //
    //  Inspect each face and the properties tagged at all of its corners:
    //
    int numFacesToRefine = facesToRefine.size() ? facesToRefine.size() : level.getNumFaces();

    for (int fIndex = 0; fIndex < numFacesToRefine; ++fIndex) {

        Vtr::Index face = facesToRefine.size() ? facesToRefine[fIndex] : (Index) fIndex;

        if (level.isFaceHole(face)) {
            continue;
        }

        //
        //  Testing irregular faces is only necessary at level 0, and potentially warrants
        //  separating out as the caller can detect these.
        //
        if (selectIrregularFaces) {
            Vtr::ConstIndexArray faceVerts = level.getFaceVertices(face);

            if (faceVerts.size() != regularFaceSize) {
                selector.selectFace(face);

                //  For non-linear schemes, if the faces to refine were not explicitly
                //  specified, select all faces adjacent to this irregular face.  (This
                //  is the only place where other faces are selected as a side effect
                //  and somewhat undermines the whole intent of the per-face traversal.)
                //
                if ((neighborhood > 0) && facesToRefine.empty()) {
                    for (int i = 0; i < faceVerts.size(); ++i) {
                        ConstIndexArray fVertFaces = level.getVertexFaces(faceVerts[i]);
                        for (int j = 0; j < fVertFaces.size(); ++j) {
                            selector.selectFace(fVertFaces[j]);
                        }
                    }
                }
                continue;
            } else {
                //  For non-linear schemes, if the faces to refine were explicitly
                //  specified, we can't count on this regular face be selected as
                //  adjacent to an irregular face above, so see if any of its
                //  neighboring faces are irregular and select if so:
                //
                if ((neighborhood > 0) && !facesToRefine.empty()) {
                    if (faceNextToIrregFace(level, face, regularFaceSize)) {
                        selector.selectFace(face);
                        continue;
                    }
                }
            }
        }

        //
        //  Test if the face has any of the specified features present.  If not, and FVar
        //  channels are to be considered, look for features in the FVar channels:
        //
        bool selectFace = doesFaceHaveFeatures(level, face, featureMask);

        if (!selectFace && featureMask.selectFVarFeatures) {
            for (int channel = 0; !selectFace && (channel < numFVarChannels); ++channel) {

                //  Only test the face for this channel if the topology does not match:
                if (!level.doesFaceFVarTopologyMatch(face, channel)) {
                    selectFace = doesFaceHaveDistinctFaceVaryingFeatures(
                                        level, face, featureMask, channel);
                }
            }
        }
        if (selectFace) {
            selector.selectFace(face);
        }
    }
}

} // end namespace Far

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv
