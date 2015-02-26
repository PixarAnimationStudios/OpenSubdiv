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
    _useSingleCreasePatch(false),
    _maxLevel(0) {

    //  Need to revisit allocation scheme here -- want to use smart-ptrs for these
    //  but will probably have to settle for explicit new/delete...
    _levels.reserve(10);
    _levels.push_back(new Vtr::Level);
}

TopologyRefiner::~TopologyRefiner() {

    for (int i=0; i<(int)_levels.size(); ++i) {
        delete _levels[i];
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
    }
    for (int i=0; i<(int)_refinements.size(); ++i) {
        delete _refinements[i];
    }
    _refinements.clear();
}


//
//  Accessors to the topology information:
//
int
TopologyRefiner::GetNumVerticesTotal() const {
    int sum = 0;
    for (int i = 0; i < (int)_levels.size(); ++i) {
        sum += _levels[i]->getNumVertices();
    }
    return sum;
}
int
TopologyRefiner::GetNumEdgesTotal() const {
    int sum = 0;
    for (int i = 0; i < (int)_levels.size(); ++i) {
        sum += _levels[i]->getNumEdges();
    }
    return sum;
}
int
TopologyRefiner::GetNumFacesTotal() const {
    int sum = 0;
    for (int i = 0; i < (int)_levels.size(); ++i) {
        sum += _levels[i]->getNumFaces();
    }
    return sum;
}
int
TopologyRefiner::GetNumFaceVerticesTotal() const {
    int sum = 0;
    for (int i = 0; i < (int)_levels.size(); ++i) {
        sum += _levels[i]->getNumFaceVerticesTotal();
    }
    return sum;
}
int
TopologyRefiner::GetNumFVarValuesTotal(int channel) const {
    int sum = 0;
    for (int i = 0; i < (int)_levels.size(); ++i) {
        sum += _levels[i]->getNumFVarValues(channel);
    }
    return sum;
}

int
TopologyRefiner::GetNumHoles(int level) const {
    int sum = 0;
    Vtr::Level const & lvl = getLevel(level);
    for (Index face = 0; face < lvl.getNumFaces(); ++face) {
        if (lvl.isHole(face)) {
            ++sum;
        }
    }
    return sum;
}

//
//  Ptex information accessors
//
void
TopologyRefiner::initializePtexIndices() const {
    Vtr::Level const & coarseLevel = getLevel(0);
    std::vector<int> & ptexIndices = const_cast<std::vector<int> &>(_ptexIndices);

    int nfaces = coarseLevel.getNumFaces();
    ptexIndices.resize(nfaces+1);
    int ptexID=0;
    int regFaceSize = Sdc::SchemeTypeTraits::GetRegularFaceSize(GetSchemeType());
    for (int i = 0; i < nfaces; ++i) {
        ptexIndices[i] = ptexID;
        Vtr::ConstIndexArray fverts = coarseLevel.getFaceVertices(i);
        ptexID += fverts.size()==regFaceSize ? 1 : fverts.size();
    }
    // last entry contains the number of ptex texture faces
    ptexIndices[nfaces]=ptexID;
}
int
TopologyRefiner::GetNumPtexFaces() const {
    if (_ptexIndices.empty()) {
        initializePtexIndices();
    }
    return _ptexIndices.back();
}
int
TopologyRefiner::GetPtexIndex(Index f) const {
    if (_ptexIndices.empty()) {
        initializePtexIndices();
    }
    assert(f<(int)_ptexIndices.size());
    return _ptexIndices[f];
}

namespace {
    // Returns the face adjacent to 'face' along edge 'edge'
    inline Index
    getAdjacentFace(Vtr::Level const & level, Index edge, Index face) {
        Far::ConstIndexArray adjFaces = level.getEdgeFaces(edge);
        if (adjFaces.size()!=2) {
            return -1;
        }
        return (adjFaces[0]==face) ? adjFaces[1] : adjFaces[0];
    }
}

void
TopologyRefiner::GetPtexAdjacency(int face, int quadrant,
        int adjFaces[4], int adjEdges[4]) const {

    assert(GetSchemeType()==Sdc::SCHEME_CATMARK);

    if (_ptexIndices.empty()) {
        initializePtexIndices();
    }

    Vtr::Level const & level = getLevel(0);

    ConstIndexArray fedges = level.getFaceEdges(face);

    if (fedges.size()==4) {

        // Regular ptex quad face
        for (int i=0; i<4; ++i) {
            int edge = fedges[i];
            Index adjface = getAdjacentFace(level, edge, face);
            if (adjface==-1) {
                adjFaces[i] = -1;  // boundary or non-manifold
                adjEdges[i] = 0;
            } else {

                ConstIndexArray aedges = level.getFaceEdges(adjface);
                if (aedges.size()==4) {
                    adjFaces[i] = _ptexIndices[adjface];
                    adjEdges[i] = aedges.FindIndexIn4Tuple(edge);
                    assert(adjEdges[i]!=-1);
                } else {
                    // neighbor is a sub-face
                    adjFaces[i] = _ptexIndices[adjface] +
                                  (aedges.FindIndex(edge)+1)%aedges.size();
                    adjEdges[i] = 3;
                }
                assert(adjFaces[i]!=-1);
            }
        }
    } else {

        //  Ptex sub-face 'quadrant' (non-quad)
        //
        // Ptex adjacency pattern for non-quads:
        //
        //             v2
        /*             o
        //            / \
        //           /   \
        //          /0   3\
        //         /       \
        //        o_ 1   2 _o
        //       /  -_   _-  \
        //      /  2  -o-  1  \
        //     /3      |      0\
        //    /       1|2       \
        //   /    0    |    3    \
        //  o----------o----------o
        // v0                     v1
        */
        assert(quadrant>=0 and quadrant<fedges.size());

        int nextQuadrant = (quadrant+1) % fedges.size(),
            prevQuadrant = (quadrant+fedges.size()-1) % fedges.size();

        {   // resolve neighbors within the sub-face (edges 1 & 2)
            adjFaces[1] = _ptexIndices[face] + nextQuadrant;
            adjEdges[1] = 2;

            adjFaces[2] = _ptexIndices[face] + prevQuadrant;
            adjEdges[2] = 1;
        }

        {   // resolve neighbor outisde the sub-face (edge 0)
            int edge0 = fedges[quadrant];
            Index adjface0 = getAdjacentFace(level, edge0, face);
            if (adjface0==-1) {
                adjFaces[0] = -1;  // boundary or non-manifold
                adjEdges[0] = 0;
            } else {
                ConstIndexArray afedges = level.getFaceEdges(adjface0);
                if (afedges.size()==4) {
                   adjFaces[0] = _ptexIndices[adjface0];
                   adjEdges[0] = afedges.FindIndexIn4Tuple(edge0);
                } else {
                   int subedge = (afedges.FindIndex(edge0)+1)%afedges.size();
                   adjFaces[0] = _ptexIndices[adjface0] + subedge;
                   adjEdges[0] = 3;
                }
                assert(adjFaces[0]!=-1);
            }

            // resolve neighbor outisde the sub-face (edge 3)
            int edge3 = fedges[prevQuadrant];
            Index adjface3 = getAdjacentFace(level, edge3, face);
            if (adjface3==-1) {
                adjFaces[3]=-1;  // boundary or non-manifold
                adjEdges[3]=0;
            } else {
                ConstIndexArray afedges = level.getFaceEdges(adjface3);
                if (afedges.size()==4) {
                   adjFaces[3] = _ptexIndices[adjface3];
                   adjEdges[3] = afedges.FindIndexIn4Tuple(edge3);
                } else {
                   int subedge = afedges.FindIndex(edge3);
                   adjFaces[3] = _ptexIndices[adjface3] + subedge;
                   adjEdges[3] = 0;
                }
                assert(adjFaces[3]!=-1);
            }
        }
    }
}

//
//  Main refinement method -- allocating and initializing levels and refinements:
//
void
TopologyRefiner::RefineUniform(UniformOptions options) {

    assert(_levels[0]->getNumVertices() > 0);  //  Make sure the base level has been initialized

    //
    //  Allocate the stack of levels and the refinements between them:
    //
    _isUniform = true;
    _maxLevel = options.refinementLevel;

    Sdc::Split splitType = (_subdivType == Sdc::SCHEME_LOOP) ? Sdc::SPLIT_TO_TRIS : Sdc::SPLIT_TO_QUADS;

    //
    //  Initialize refinement options for Vtr -- adjusting full-topology for the last level:
    //
    Vtr::Refinement::Options refineOptions;
    refineOptions._sparse = false;

    for (int i = 1; i <= (int)options.refinementLevel; ++i) {
        refineOptions._faceTopologyOnly =
            options.fullTopologyInLastLevel ? false : (i == options.refinementLevel);

        Vtr::Level& parentLevel = getLevel(i-1);
        Vtr::Level& childLevel  = *(new Vtr::Level);

        Vtr::Refinement* refinement = 0;
        if (splitType == Sdc::SPLIT_TO_QUADS) {
            refinement = new Vtr::QuadRefinement(parentLevel, childLevel, _subdivOptions);
        } else {
            refinement = new Vtr::TriRefinement(parentLevel, childLevel, _subdivOptions);
        }
        refinement->refine(refineOptions);

        _levels.push_back(&childLevel);
        _refinements.push_back(refinement);
    }
}


void
TopologyRefiner::RefineAdaptive(AdaptiveOptions options) {

    assert(_levels[0]->getNumVertices() > 0);  //  Make sure the base level has been initialized

    //
    //  Allocate the stack of levels and the refinements between them:
    //
    _isUniform = false;
    _maxLevel = options.isolationLevel;
    _useSingleCreasePatch = options.useSingleCreasePatch;

    //
    //  Initialize refinement options for Vtr:
    //
    Vtr::Refinement::Options refineOptions;

    refineOptions._sparse           = true;
    refineOptions._faceTopologyOnly = not options.fullTopologyInLastLevel;

    Sdc::Split splitType = (_subdivType == Sdc::SCHEME_LOOP) ? Sdc::SPLIT_TO_TRIS : Sdc::SPLIT_TO_QUADS;

    for (int i = 1; i <= (int)options.isolationLevel; ++i) {
        //  Keeping full topology on for debugging -- may need to go back a level and "prune"
        //  its topology if we don't use the full depth
        refineOptions._faceTopologyOnly = false;

        Vtr::Level& parentLevel     = getLevel(i-1);
        Vtr::Level& childLevel      = *(new Vtr::Level);

        Vtr::Refinement* refinement = 0;
        if (splitType == Sdc::SPLIT_TO_QUADS) {
            refinement = new Vtr::QuadRefinement(parentLevel, childLevel, _subdivOptions);
        } else {
            refinement = new Vtr::TriRefinement(parentLevel, childLevel, _subdivOptions);
        }

        //
        //  Initialize a Selector to mark a sparse set of components for refinement.  If
        //  nothing was selected, discard the new refinement and child level, trim the
        //  maximum level and stop refinining any further.  Otherwise, refine and append
        //  the new refinement and child.
        //
        //  Note that if we support the "full topology at last level" option properly,
        //  we should prune the previous level generated, as it is now the last...
        //
        Vtr::SparseSelector selector(*refinement);

        selectFeatureAdaptiveComponents(selector);
        if (selector.isSelectionEmpty()) {
            _maxLevel = i - 1;

            delete refinement;
            delete &childLevel;
            break;
        }

        refinement->refine(refineOptions);

        _levels.push_back(&childLevel);
        _refinements.push_back(refinement);

        //childLevel.print(refinement);
        //assert(childLevel.validateTopology());
    }
}

//
//   Method for selecting components for sparse refinement based on the feature-adaptive needs
//   of patch generation.
//
//   It assumes we have a freshly initialized Vtr::SparseSelector (i.e. nothing already selected)
//   and will select all relevant topological features for inclusion in the subsequent sparse
//   refinement.
//
//   This was originally written specific to the quad-centric Catmark scheme and was since
//   generalized to support Loop given the enhanced tagging of components based on the scheme.
//   Any further enhancements here, e.g. new approaches for dealing with infinitely sharp
//   creases, should be aware of the intended generality.  Ultimately it may not be worth
//   trying to keep this general and we will be better off specializing it for each scheme.
//   The fact that this method is intimately tied to patch generation also begs for it to
//   become part of a class that encompasses both the feature adaptive tagging and the
//   identification of the intended patch that result from it.
//
void
TopologyRefiner::selectFeatureAdaptiveComponents(Vtr::SparseSelector& selector) {

    Vtr::Level const& level = selector.getRefinement().parent();

    int  regularFaceSize             =  selector.getRefinement()._regFaceSize;
    bool considerSingleCreasePatch   = _useSingleCreasePatch && (regularFaceSize == 4);

    //
    //  Face-varying consideration when isolating features:
    //      - there must obviously be face-varying channels for any consideration
    //      - we can ignore all purely linear face-varying channels -- a common case that
    //        will allow us to avoid the repeated per-face inspection of FVar data
    //      - may allow a subset of face-varying channels to be considered in future:
    //
    //  Note that some of this consideration can be given at the highest level and then
    //  reflected potentially in the Selector, e.g. when all FVar channels are linear,
    //  any request to inspect them can be overridden for all levels and not repeatedly
    //  reassessed here for each level.
    //
    int  numFVarChannels      = level.getNumFVarChannels();
    bool considerFVarChannels = (numFVarChannels > 0);

    if (considerFVarChannels) {
        considerFVarChannels = false;

        for (int channel = 0; channel < numFVarChannels; ++channel) {
            if (not level._fvarChannels[channel]->isLinear()) {
                considerFVarChannels = true;
                break;
            }
        }
    }

    //
    //  Inspect each face and the properties tagged at all of its corners:
    //
    for (Vtr::Index face = 0; face < level.getNumFaces(); ++face) {

        if (level.isHole(face)) {
            continue;
        }

        Vtr::ConstIndexArray faceVerts = level.getFaceVertices(face);

        //
        //  Testing irregular faces is only necessary at level 0, and potentially warrants
        //  separating out as the caller can detect these:
        //
        if (faceVerts.size() != regularFaceSize) {
            //
            //  We need to also ensure that all adjacent faces to this are selected, so we
            //  select every face incident every vertex of the face.  This is the only place
            //  where other faces are selected as a side effect and somewhat undermines the
            //  whole intent of the per-face traversal.
            //
            Vtr::ConstIndexArray fVerts = level.getFaceVertices(face);
            for (int i = 0; i < fVerts.size(); ++i) {
                ConstIndexArray fVertFaces = level.getVertexFaces(fVerts[i]);
                for (int j = 0; j < fVertFaces.size(); ++j) {
                    selector.selectFace(fVertFaces[j]);
                }
            }
            continue;
        }

        //
        //  Combine the tags for all vertices of the face and quickly accept/reject based on
        //  the presence/absence of properties where we can (further inspection is likely to
        //  be necessary in some cases, particularly when we start trying to be clever about
        //  minimizing refinement for inf-sharp creases, etc.):
        //
        Vtr::Level::VTag compFaceVTag = level.getFaceCompositeVTag(faceVerts);
        if (compFaceVTag._incomplete) {
            continue;
        }

        bool selectFace = false;
        if (compFaceVTag._xordinary) {
            selectFace = true;
        } else if (compFaceVTag._nonManifold) {
            //  Warrants further inspection in future -- isolate for now
            //    - will want to defer inf-sharp treatment to below
            selectFace = true;
        } else if (compFaceVTag._rule == Sdc::Crease::RULE_SMOOTH) {
            //  Avoid isolation when ALL vertices are Smooth.  All vertices must be regular by
            //  now and all vertices Smooth implies they are all interior vertices.  (If any
            //  adjacent faces are not regular, this face will have been previously selected).
            selectFace = false;
        } else if (compFaceVTag._rule & Sdc::Crease::RULE_DART) {
            //  Any occurrence of a Dart vertex requires isolation
            selectFace = true;
        } else if (not (compFaceVTag._rule & Sdc::Crease::RULE_SMOOTH)) {
            //  None of the vertices is Smooth, so we have all vertices either Crease or Corner.
            //  Though some may be regular patches, this currently warrants isolation as we only
            //  support regular patches with one corner or one boundary, i.e. with one or more
            //  smooth interior vertices.
            selectFace = true;
        } else if (compFaceVTag._semiSharp || compFaceVTag._semiSharpEdges) {
            //  Any semi-sharp feature at or around the vertex warrants isolation -- unless we
            //  optimize for the single-crease patch, i.e. only edge sharpness of a constant value
            //  along the entire regular patch boundary (quickly exclude the Corner case first):
            if (considerSingleCreasePatch && not (compFaceVTag._rule & Sdc::Crease::RULE_CORNER)) {
                selectFace = not level.isSingleCreasePatch(face);
            } else {
                selectFace = true;
            }
        } else if (not compFaceVTag._boundary) {
            //  At this point we are left with a mix of smooth and inf-sharp features.  If not
            //  on a boundary, the interior inf-sharp features need isolation -- unless we are
            //  again optimizing for the single-crease patch, infinitely sharp in this case.
            //
            //  Note this case of detecting a single-crease patch, while similar to the above,
            //  is kept separate for the inf-sharp case:  a separate and much more efficient
            //  test can be made for the inf-sharp case, and there are other opportunities here
            //  to optimize for regular patches at infinitely sharp corners.
            if (considerSingleCreasePatch && not (compFaceVTag._rule & Sdc::Crease::RULE_CORNER)) {
                selectFace = not level.isSingleCreasePatch(face);
            } else {
                selectFace = true;
            }
        } else if (not (compFaceVTag._rule & Sdc::Crease::RULE_CORNER)) {
            //  We are now left with boundary faces -- if no Corner vertex, we have a mix of both
            //  regular Smooth and Crease vertices on a boundary face, which can only be a regular
            //  boundary patch, so don't isolate.
            selectFace = false;
        } else {
            //  The last case with at least one Corner vertex and one Smooth (interior) vertex --
            //  distinguish the regular corner case from others:
            if (not compFaceVTag._corner) {
                //  We may consider interior sharp corners as regular in future, but for now we
                //  only accept a topological corner for the regular corner case:
                selectFace = true;
            } else if (level.getDepth() > 0) {
                //  A true corner at a subdivided level -- adjacent verts must be Crease and the
                //  opposite Smooth so we must have a regular corner:
                selectFace = false;
            } else {
                //  Make sure the adjacent boundary vertices were not sharpened, or equivalently,
                //  that only one corner is sharp:
                unsigned int infSharpCount = level._vertTags[faceVerts[0]]._infSharp;
                for (int i = 1; i < faceVerts.size(); ++i) {
                    infSharpCount += level._vertTags[faceVerts[i]]._infSharp;
                }
                selectFace = (infSharpCount != 1);
            }
        }

        //
        //  If still not selected, inspect the face-varying channels (when present) for similar
        //  irregular features requiring isolation:
        //
        if (not selectFace and considerFVarChannels) {
            for (int channel = 0; not selectFace && (channel < numFVarChannels); ++channel) {
                Vtr::FVarLevel const & fvarLevel = *level._fvarChannels[channel];

                //
                //  Retrieve the counterpart to the face-vertices composite tag for the face-values
                //  for this channel.  We can make some quick accept/reject tests but eventually we
                //  will need to combine the face-vertex and face-varying topology to determine the
                //  regularity of faces along face-varying boundaries.
                //
                Vtr::ConstIndexArray faceValues = fvarLevel.getFaceValues(face);

                Vtr::FVarLevel::ValueTag compFVarFaceTag =
                        fvarLevel.getFaceCompositeValueTag(faceValues, faceVerts);

                //  No mismatch in topology -> no need to further isolate...
                if (not compFVarFaceTag._mismatch) continue;

                if (compFVarFaceTag._xordinary) {
                    //  An xordinary boundary value always requires isolation:
                    selectFace = true;
                } else {
                    //  Combine the FVar topology tags (ValueTags) at corners with the vertex topology
                    //  tags (VTags), then make similar inferences from the combined tags as was done
                    //  for the face.
                    Vtr::Level::VTag fvarVTags[4];
                    Vtr::Level::VTag compFVarVTag =
                            fvarLevel.getFaceCompositeValueAndVTag(faceValues, faceVerts, fvarVTags);

                    if (not (compFVarVTag._rule & Sdc::Crease::RULE_SMOOTH)) {
                        //  No Smooth corners so too many boundaries/corners -- need to isolate:
                        selectFace = true;
                    } else if (not (compFVarVTag._rule & Sdc::Crease::RULE_CORNER)) {
                        //  A mix of Smooth and Crease corners -- must be regular so don't isolate:
                        selectFace = false;
                    } else {
                        //  Since FVar boundaries can be "sharpened" based on the linear interpolation
                        //  rules, we again have to inspect more closely (as we did with the original
                        //  face) to ensure we have a regular corner and not a sharpened crease:
                        unsigned int boundaryCount = fvarVTags[0]._boundary,
                                     infSharpCount = fvarVTags[0]._infSharp;
                        for (int i = 1; i < faceVerts.size(); ++i) {
                            boundaryCount += fvarVTags[i]._boundary;
                            infSharpCount += fvarVTags[i]._infSharp;
                        }
                        selectFace = (boundaryCount != 3) || (infSharpCount != 1);

                        //  There is a possibility of a false positive at level 0 --  Smooth interior
                        //  vertex with adjacent Corner and two opposite boundary Crease vertices
                        //  (the topological corner tag catches this above).  Verify that the corner
                        //  vertex is opposite the smooth vertex (and consider doing this above)...
                        //
                        if (not selectFace && (level.getDepth() == 0)) {
                            if (fvarVTags[0]._infSharp && fvarVTags[2]._boundary) selectFace = true;
                            if (fvarVTags[1]._infSharp && fvarVTags[3]._boundary) selectFace = true;
                            if (fvarVTags[2]._infSharp && fvarVTags[0]._boundary) selectFace = true;
                            if (fvarVTags[3]._infSharp && fvarVTags[1]._boundary) selectFace = true;
                        }
                    }
                }
            }
        }

        //  Finally, select the face for further refinement:
        if (selectFace) {
            selector.selectFace(face);
        }
    }
}

} // end namespace Far

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv
