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

#include <cassert>
#include <cstdio>


namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Far {

//
//  Relatively trivial construction/destruction -- the base level (level[0]) needs
//  to be explicitly initialized after construction and refinement then applied
//
TopologyRefiner::TopologyRefiner(Sdc::Type schemeType, Sdc::Options schemeOptions) :
    _subdivType(schemeType),
    _subdivOptions(schemeOptions),
    _isUniform(true),
    _maxLevel(0),
    _useSingleCreasePatch(false) {

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

void
TopologyRefiner::Clear() {
    _levels.clear();
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

//
//  Ptex information accessors
//
template <Sdc::Type SCHEME_TYPE> void
computePtexIndices(Vtr::Level const & coarseLevel, std::vector<int> & ptexIndices) {
    int nfaces = coarseLevel.getNumFaces();
    ptexIndices.resize(nfaces+1);
    int ptexID=0;
    for (int i = 0; i < nfaces; ++i) {
        ptexIndices[i] = ptexID;
        Vtr::IndexArray fverts = coarseLevel.getFaceVertices(i);
        ptexID += fverts.size()==Sdc::TypeTraits<SCHEME_TYPE>::RegularFaceValence() ? 1 : fverts.size();
    }
    // last entry contains the number of ptex texture faces
    ptexIndices[nfaces]=ptexID;
}
void
TopologyRefiner::initializePtexIndices() const {
    std::vector<int> & indices = const_cast<std::vector<int> &>(_ptexIndices);
    switch (GetSchemeType()) {
        case Sdc::TYPE_BILINEAR:
            computePtexIndices<Sdc::TYPE_BILINEAR>(getLevel(0), indices); break;
        case Sdc::TYPE_CATMARK :
            computePtexIndices<Sdc::TYPE_CATMARK>(getLevel(0), indices); break;
        case Sdc::TYPE_LOOP    :
            computePtexIndices<Sdc::TYPE_LOOP>(getLevel(0), indices); break;
    }
}
int
TopologyRefiner::GetNumPtexFaces() const {
    if (_ptexIndices.empty()) {
        initializePtexIndices();
    }
    // see computePtexIndices()
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
        IndexArray adjFaces = level.getEdgeFaces(edge);
        if (adjFaces.size()!=2) {
            return -1;
        }
        return (adjFaces[0]==face) ? adjFaces[1] : adjFaces[0];
    }
}

void
TopologyRefiner::GetPtexAdjacency(int face, int quadrant,
        int adjFaces[4], int adjEdges[4]) const {

    assert(GetSchemeType()==Sdc::TYPE_CATMARK);

    if (_ptexIndices.empty()) {
        initializePtexIndices();
    }

    Vtr::Level const & level = getLevel(0);

    IndexArray fedges = level.getFaceEdges(face);

    if (fedges.size()==4) {

        // Regular ptex quad face
        for (int i=0; i<4; ++i) {
            int edge = fedges[i];
            IndexArray efaces = level.getEdgeFaces(edge);
            Index adjface = getAdjacentFace(level, edge, face);
            if (adjface==-1) {
                adjFaces[i] = -1;  // boundary or non-manifold
                adjEdges[i] = 0;
            } else {

                IndexArray aedges = level.getFaceEdges(adjface);
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
                IndexArray afedges = level.getFaceEdges(adjface0);
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
                IndexArray afedges = level.getFaceEdges(adjface3);
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
TopologyRefiner::RefineUniform(int maxLevel, bool fullTopology) {

    assert(_levels[0]->getNumVertices() > 0);  //  Make sure the base level has been initialized
    assert(_subdivType == Sdc::TYPE_CATMARK);

    //
    //  Allocate the stack of levels and the refinements between them:
    //
    _isUniform = true;
    _maxLevel = maxLevel;

    //
    //  Initialize refinement options for Vtr -- adjusting full-topology for the last level:
    //
    Vtr::Refinement::Options refineOptions;
    refineOptions._sparse = false;

    for (int i = 1; i <= maxLevel; ++i) {
        refineOptions._faceTopologyOnly = fullTopology ? false : (i == maxLevel);

        Vtr::Level& parentLevel     = getLevel(i-1);
        Vtr::Level& childLevel      = *(new Vtr::Level);
        Vtr::Refinement& refinement = *(new Vtr::Refinement);

        refinement.setScheme(_subdivType, _subdivOptions);
        refinement.initialize(parentLevel, childLevel);
        refinement.refine(refineOptions);

        _levels.push_back(&childLevel);
        _refinements.push_back(&refinement);
    }
}


void
TopologyRefiner::RefineAdaptive(int subdivLevel, bool fullTopology, bool useSingleCreasePatch) {

    assert(_levels[0]->getNumVertices() > 0);  //  Make sure the base level has been initialized
    assert(_subdivType == Sdc::TYPE_CATMARK);

    //
    //  Allocate the stack of levels and the refinements between them:
    //
    _isUniform = false;
    _maxLevel = subdivLevel;
    _useSingleCreasePatch = useSingleCreasePatch;

    //
    //  Initialize refinement options for Vtr:
    //
    Vtr::Refinement::Options refineOptions;

    refineOptions._sparse           = true;
    refineOptions._faceTopologyOnly = !fullTopology;

    for (int i = 1; i <= subdivLevel; ++i) {
        //  Keeping full topology on for debugging -- may need to go back a level and "prune"
        //  its topology if we don't use the full depth
        refineOptions._faceTopologyOnly = false;

        Vtr::Level& parentLevel     = getLevel(i-1);
        Vtr::Level& childLevel      = *(new Vtr::Level);
        Vtr::Refinement& refinement = *(new Vtr::Refinement);

        refinement.setScheme(_subdivType, _subdivOptions);
        refinement.initialize(parentLevel, childLevel);

        //
        //  Initialize a Selector to mark a sparse set of components for refinement.  If
        //  nothing was selected, discard the new refinement and child level, trim the
        //  maximum level and stop refinining any further.  Otherwise, refine and append
        //  the new refinement and child.
        //
        //  Note that if we support the "full topology at last level" option properly,
        //  we should prune the previous level generated, as it is now the last...
        //
        Vtr::SparseSelector selector(refinement);

        catmarkFeatureAdaptiveSelector(selector);
        if (selector.isSelectionEmpty()) {
            _maxLevel = i - 1;

            delete &refinement;
            delete &childLevel;
            break;
        }

        refinement.refine(refineOptions);

        _levels.push_back(&childLevel);
        _refinements.push_back(&refinement);

        //childLevel.print(&refinement);
        //assert(childLevel.validateTopology());
    }
}

//
//   Catmark-specific method for feature-adaptive selection for sparse refinement at each level.
//
//   It assumes we have a freshly initialized Vtr::SparseSelector (i.e. nothing already selected)
//   and will select all relevant topological features for inclusion in the subsequent sparse
//   refinement.
//
//   With appropriate topological tags on the components, i.e. which vertices are extra-ordinary,
//   non-manifold, etc., there's no reason why this can't be written in a way that is independent
//   of the subdivision scheme.  All of the creasing cases are independent, leaving only the
//   regularity associated with the scheme.
//
void
TopologyRefiner::catmarkFeatureAdaptiveSelector(Vtr::SparseSelector& selector) {

    Vtr::Level const& level = selector.getRefinement().parent();

    for (Vtr::Index face = 0; face < level.getNumFaces(); ++face) {
        Vtr::IndexArray const faceVerts = level.getFaceVertices(face);

        //
        //  Testing irregular faces is only necessary at level 0, and potentially warrants
        //  separating out as the caller can detect these (and generically as long as we
        //  can identify an irregular face for all schemes):
        //
        if (faceVerts.size() != 4) {
            //
            //  We need to also ensure that all adjacent faces to this are selected, so we
            //  select every face incident every vertex of the face.  This is the only place
            //  where other faces are selected as a side effect and somewhat undermines the
            //  whole intent of the per-face traversal.
            //
            Vtr::IndexArray const fVerts = level.getFaceVertices(face);
            for (int i = 0; i < fVerts.size(); ++i) {
                IndexArray const fVertFaces = level.getVertexFaces(fVerts[i]);
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
        Vtr::Level::VTag compFaceTag = level.getFaceCompositeVTag(faceVerts);
        if (compFaceTag._incomplete) {
            continue;
        }

        bool selectFace = false;
        if (compFaceTag._xordinary) {
            selectFace = true;
        } else if (compFaceTag._rule & Sdc::Crease::RULE_DART) {
            //  Get this case out of the way before testing hard features
            selectFace = true;
        } else if (compFaceTag._nonManifold) {
            //  Warrants further inspection -- isolate for now
            //    - will want to defer inf-sharp treatment to below
            selectFace = true;
        } else if (!(compFaceTag._rule & Sdc::Crease::RULE_SMOOTH)) {
            //  None of the vertices is Smooth, so we have all vertices either Crease or Corner,
            //  though some may be regular patches, this currently warrants isolation as we only
            //  support regular patches with one corner or one boundary.
            selectFace = true;
        } else if (compFaceTag._semiSharp) {
            // if this is regular and the adjacent edges have same sharpness
            // and no vertex corner sharpness,
            // we can stop refinning and use single-crease patch.
            if (_useSingleCreasePatch) {
                selectFace = ! level.isSingleCreasePatch(face);
            } else {
                selectFace = true;
            }
        } else {
            //  This leaves us with at least one Smooth vertex (and so two smooth adjacent edges
            //  of the quad) and the rest hard Creases or Corners.  This includes the regular
            //  corner and boundary cases that we don't want to isolate, but leaves a few others
            //  that do warrant isolation -- needing further inspection.
            //
            //  For now go with the boundary cases and don't isolate...
            selectFace = false;
        }
        if (selectFace) {
            selector.selectFace(face);
        }
    }
}

#ifdef _VTR_COMPUTE_MASK_WEIGHTS_ENABLED
void
TopologyRefiner::ComputeMaskWeights() {

    assert(_subdivType == Sdc::TYPE_CATMARK);

    for (int i = 0; i < _maxLevel; ++i) {
        _refinements[i]->computeMaskWeights();
    }
}
#endif

} // end namespace Far

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv
