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
#include "../sdc/types.h"
#include "../sdc/crease.h"
#include "../vtr/array.h"
#include "../vtr/refinement.h"
#include "../vtr/fvarLevel.h"

#include "../vtr/fvarRefinement.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <algorithm>


//
//  FVarRefinement:
//      Analogous to Refinement -- retains data to facilitate refinement and
//  population of refined face-varying data channels.
//
namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Vtr {

//
//  Simple (for now) constructor and destructor:
//
FVarRefinement::FVarRefinement(Refinement const& refinement,
                               FVarLevel&        parentFVarLevel,
                               FVarLevel&        childFVarLevel) :
    _refinement(refinement),
    _parentLevel(*refinement._parent),
    _parentFVar(parentFVarLevel),
    _childLevel(*refinement._child),
    _childFVar(childFVarLevel) {
}

FVarRefinement::~FVarRefinement() {
}


//
// Methods supporting the refinement of face-varying data that has previously
// been applied to the Refinment member. So these methods already have access
// to fully refined child components.
//

void
FVarRefinement::applyRefinement() {

    //
    //  Transfer basic properties from the parent to child level:
    //
    _childFVar._options = _parentFVar._options;

    _childFVar._isLinear              = _parentFVar._isLinear;
    _childFVar._hasSmoothBoundaries   = _parentFVar._hasSmoothBoundaries;
    _childFVar._hasDependentSharpness = _parentFVar._hasDependentSharpness;

    //
    //  It's difficult to know immediately how many child values arise from the
    //  refinement -- particularly when sparse, so we get a close upper bound,
    //  resize for that number and trim when finished:
    //
    estimateAndAllocateChildValues();
    populateChildValues();
    trimAndFinalizeChildValues();

    propagateEdgeTags();
    propagateValueTags();
    if (_childFVar._hasSmoothBoundaries) {
        propagateValueCreases();
        reclassifySemisharpValues();
    }

    //
    //  The refined face-values are technically redundant as they can be constructed
    //  from the face-vertex siblings -- do so here as a post-process
    //
    if (_childFVar.getNumValues() > _childLevel.getNumVertices()) {
        _childFVar.initializeFaceValuesFromVertexFaceSiblings();
    } else {
        _childFVar.initializeFaceValuesFromFaceVertices();
    }

    //printf("FVar refinement to level %d:\n", _childLevel.getDepth());
    //_childFVar.print();

    //printf("Validating refinement to level %d:\n", _childLevel.getDepth());
    //_childFVar.validate();
    //assert(_childFVar.validate());
}

//
//  Quickly estimate the memory required for face-varying vertex-values in the child
//  and allocate them.  For uniform refinement this estimate should exactly match the
//  desired result.  For sparse refinement the excess should generally be low as the
//  sparse boundary components generally occur where face-varying data is continuous.
//
void
FVarRefinement::estimateAndAllocateChildValues() {

    int maxVertexValueCount = _refinement.getNumChildVerticesFromFaces();

    Index cVert    = _refinement.getFirstChildVertexFromEdges();
    Index cVertEnd = cVert + _refinement.getNumChildVerticesFromEdges();
    for ( ; cVert < cVertEnd; ++cVert) {
        Index pEdge = _refinement.getChildVertexParentIndex(cVert);

        maxVertexValueCount += _parentFVar.edgeTopologyMatches(pEdge)
                             ? 1 : _parentLevel.getEdgeFaces(pEdge).size();
    }

    cVert    = _refinement.getFirstChildVertexFromVertices();
    cVertEnd = cVert + _refinement.getNumChildVerticesFromVertices();
    for ( ; cVert < cVertEnd; ++cVert) {
        assert(!_refinement._childVertexTag[cVert]._incomplete);
        Index pVert = _refinement.getChildVertexParentIndex(cVert);

        maxVertexValueCount += _parentFVar.getNumVertexValues(pVert);
    }

    //
    //  Now allocate/initialize for the maximum -- use resize() and trim the size later
    //  to avoid the constant growing with reserve() and incremental sizing.  We know
    //  the estimate should be close and memory wasted should be small, so initialize
    //  all to zero as well to avoid writing in all but affected areas:
    //
    //  Resize vectors that mirror the component counts:
    _childFVar.resizeComponents();

    //  Resize the vertex-value tags in the child level:
    _childFVar._vertValueTags.resize(maxVertexValueCount);

    //  Resize the vertex-value "parent source" mapping in the refinement:
    _childValueParentSource.resize(maxVertexValueCount, 0);
}

void
FVarRefinement::trimAndFinalizeChildValues() {

    _childFVar._vertValueTags.resize(_childFVar._valueCount);
    if (_childFVar._hasSmoothBoundaries) {
        _childFVar._vertValueCreaseEnds.resize(_childFVar._valueCount);
    }

    _childValueParentSource.resize(_childFVar._valueCount);

    //  Allocate and initialize the vector of indices (redundant after level 0):
    _childFVar._vertValueIndices.resize(_childFVar._valueCount);
    for (int i = 0; i < _childFVar._valueCount; ++i) {
        _childFVar._vertValueIndices[i] = i;
    }
}

inline int
FVarRefinement::populateChildValuesForEdgeVertex(Index cVert, Index pEdge) {

    //  If we have a boundary edge with a mismatched end vertex, we only have one
    //  value and such cases were already initialized on construction, so return:
    //
    ConstIndexArray  pEdgeFaces = _parentLevel.getEdgeFaces(pEdge);

    if (pEdgeFaces.size() == 1) return 1;
    assert(pEdgeFaces.size() == 2);

    //  Determine the number of sibling values for the child vertex:
    //
    ConstIndexArray  cVertFaces = _childLevel.getVertexFaces(cVert);

    int cValueCount = 1;
    if (cVertFaces.size() > 2) {
        cValueCount = 2;
    } else if (cVertFaces.size() == 2) {
        cValueCount = 1 + (cVertFaces[0] != cVertFaces[1]);
    }

    //
    //  We may have no sibling values here when the edge was discts, but only in the
    //  case where the refinement was sparse (and so the child vertex "incomplete").
    //  In such a case the source of the single value needs to be identified:
    //
    Index cValueIndex = _childFVar.getVertexValueOffset(cVert);

    if (cValueCount == 1) {
        _childValueParentSource[cValueIndex] = (cVertFaces[0] != pEdgeFaces[0]);
    } else {
        assert(cValueCount == 2);

        //  Update the parent-source of any siblings (typically one):
        for (int j = 1; j < cValueCount; ++j) {
            _childValueParentSource[cValueIndex + j] = (LocalIndex) j;
        }

        //  Update the vertex-face siblings:
        FVarLevel::SiblingArray cVertFaceSiblings = _childFVar.getVertexFaceSiblings(cVert);
        for (int j = 0; j < cVertFaceSiblings.size(); ++j) {
            if (_refinement.getChildFaceParentFace(cVertFaces[j]) == pEdgeFaces[1]) {
                cVertFaceSiblings[j] = 1;
            }
        }
    }
    return cValueCount;
}

inline int
FVarRefinement::populateChildValuesForVertexVertex(Index cVert, Index pVert) {

    //
    //  We should not be getting incomplete vertex-vertices from feature-adaptive
    //  refinement (as neighboring vertices will be face-vertices or edge-vertices).
    //  This will get messy when we do (i.e. sparse refinement of Bilinear or more
    //  flexible and specific sparse refinement of Catmark) but for now assume 1-to-1.
    //
    assert(!_refinement._childVertexTag[cVert]._incomplete);

    //  Number of child values is same as number of parent values since complete:
    int cValueCount = _parentFVar.getNumVertexValues(pVert);

    if (cValueCount > 1) {
        Index cValueIndex = _childFVar.getVertexValueOffset(cVert);

        // Update the parent source for all child values:
        for (int j = 1; j < cValueCount; ++j) {
            _childValueParentSource[cValueIndex + j] = (LocalIndex) j;
        }

        // Update the vertex-face siblings:
        FVarLevel::ConstSiblingArray pVertFaceSiblings = _parentFVar.getVertexFaceSiblings(pVert);
        FVarLevel::SiblingArray      cVertFaceSiblings = _childFVar.getVertexFaceSiblings(cVert);
        for (int j = 0; j < cVertFaceSiblings.size(); ++j) {
            cVertFaceSiblings[j] = pVertFaceSiblings[j];
        }
    }
    return cValueCount;
}

void
FVarRefinement::populateChildValues() {

    //
    //  Be sure to match the same vertex ordering as Refinement, i.e. face-vertices
    //  first vs vertex-vertices first, etc.  A few optimizations within the use of
    //  face-varying data take advantage of this assumption.
    //
    //  Right now there are only two orderings under consideration, and its unclear
    //  whether only one will be supported or both.  Until that's determined, assert
    //  the conditions we expect for these two.
    //
    _childFVar._valueCount = 0;

    if (_refinement.getFirstChildVertexFromVertices() > 0) {
        assert((_refinement.getFirstChildVertexFromFaces() <=
                _refinement.getFirstChildVertexFromEdges()) &&
               (_refinement.getFirstChildVertexFromEdges() <
                _refinement.getFirstChildVertexFromVertices()));

        populateChildValuesFromFaceVertices();
        populateChildValuesFromEdgeVertices();
        populateChildValuesFromVertexVertices();
    } else {
        assert((_refinement.getFirstChildVertexFromVertices() <
                _refinement.getFirstChildVertexFromFaces()) &&
               (_refinement.getFirstChildVertexFromFaces() <=
                _refinement.getFirstChildVertexFromEdges()));

        populateChildValuesFromVertexVertices();
        populateChildValuesFromFaceVertices();
        populateChildValuesFromEdgeVertices();
    }
}

void
FVarRefinement::populateChildValuesFromFaceVertices() {

    Index cVert    = _refinement.getFirstChildVertexFromFaces();
    Index cVertEnd = cVert + _refinement.getNumChildVerticesFromFaces();
    for ( ; cVert < cVertEnd; ++cVert) {
        _childFVar._vertSiblingOffsets[cVert] = _childFVar._valueCount;
        _childFVar._vertSiblingCounts[cVert]  = 1;
        _childFVar._valueCount ++;
    }
}
void
FVarRefinement::populateChildValuesFromEdgeVertices() {

    Index cVert    = _refinement.getFirstChildVertexFromEdges();
    Index cVertEnd = cVert + _refinement.getNumChildVerticesFromEdges();
    for ( ; cVert < cVertEnd; ++cVert) {
        Index pEdge = _refinement.getChildVertexParentIndex(cVert);

        _childFVar._vertSiblingOffsets[cVert] = _childFVar._valueCount;
        if (_parentFVar.edgeTopologyMatches(pEdge)) {
            _childFVar._vertSiblingCounts[cVert] = 1;
            _childFVar._valueCount ++;
        } else {
            int cValueCount = populateChildValuesForEdgeVertex(cVert, pEdge);
            _childFVar._vertSiblingCounts[cVert] = (LocalIndex)cValueCount;
            _childFVar._valueCount += cValueCount;
        }
    }
}
void
FVarRefinement::populateChildValuesFromVertexVertices() {

    Index cVert    = _refinement.getFirstChildVertexFromVertices();
    Index cVertEnd = cVert + _refinement.getNumChildVerticesFromVertices();
    for ( ; cVert < cVertEnd; ++cVert) {
        Index pVert = _refinement.getChildVertexParentIndex(cVert);

        _childFVar._vertSiblingOffsets[cVert] = _childFVar._valueCount;
        if (_parentFVar.valueTopologyMatches(_parentFVar.getVertexValueOffset(pVert))) {
            _childFVar._vertSiblingCounts[cVert] = 1;
            _childFVar._valueCount ++;
        } else {
            int cValueCount = populateChildValuesForVertexVertex(cVert, pVert);
            _childFVar._vertSiblingCounts[cVert] = (LocalIndex)cValueCount;
            _childFVar._valueCount += cValueCount;
        }
    }
}

void
FVarRefinement::propagateEdgeTags() {

    //
    //  Edge tags correspond to child edges and originate from faces or edges:
    //      Face-edges:
    //          - tag can be initialized as cts (*)
    //              * what was this comment:  "discts based on parent face-edges at ends"
    //      Edge-edges:
    //          - tag propagated from parent edge
    //          - need to modify if parent edge was discts at one end
    //              - child edge for the matching end inherits tag
    //              - child edge at the other end is doubly discts
    //
    FVarLevel::ETag eTagMatch;
    eTagMatch.clear();
    eTagMatch._mismatch = false;

    for (int eIndex = 0; eIndex < _refinement._childEdgeFromFaceCount; ++eIndex) {
        _childFVar._edgeTags[eIndex] = eTagMatch;
    }
    for (int eIndex = _refinement._childEdgeFromFaceCount; eIndex < _childLevel.getNumEdges(); ++eIndex) {
        Index pEdge = _refinement.getChildEdgeParentIndex(eIndex);

        _childFVar._edgeTags[eIndex] = _parentFVar._edgeTags[pEdge];
    }
}

void
FVarRefinement::propagateValueTags() {

    //
    //  Value tags correspond to vertex-values and originate from all three sources:
    //      Face-values:
    //          - trivially initialized as matching
    //      Edge-values:
    //          - conditionally initialized based on parent edge continuity
    //          - should be trivial though (unlike edge-tags for the child edges)
    //      Vertex-values:
    //          - if complete, trivially propagated/inherited
    //          - if incomplete, need to map to child subset
    //

    //
    //  Values from face-vertices -- all match and are sequential:
    //
    FVarLevel::ValueTag valTagMatch;
    valTagMatch.clear();

    Index cVert      = _refinement.getFirstChildVertexFromFaces();
    Index cVertEnd   = cVert + _refinement.getNumChildVerticesFromFaces();
    Index cVertValue = _childFVar.getVertexValueOffset(cVert);
    for ( ; cVert < cVertEnd; ++cVert, ++cVertValue) {
        _childFVar._vertValueTags[cVertValue] = valTagMatch;
    }

    //
    //  Values from edge-vertices -- for edges that are split, tag as mismatched and tag
    //  as corner or crease depending on the presence of creases in the parent:
    //
    FVarLevel::ValueTag valTagMismatch = valTagMatch;
    valTagMismatch._mismatch = true;

    FVarLevel::ValueTag valTagCrease = valTagMismatch;
    valTagCrease._crease = true;

    FVarLevel::ValueTag& valTagSplitEdge = _parentFVar._hasSmoothBoundaries ? valTagCrease : valTagMismatch;

    cVert    = _refinement.getFirstChildVertexFromEdges();
    cVertEnd = cVert + _refinement.getNumChildVerticesFromEdges();
    for ( ; cVert < cVertEnd; ++cVert) {
        Index pEdge = _refinement.getChildVertexParentIndex(cVert);

        FVarLevel::ValueTagArray cValueTags = _childFVar.getVertexValueTags(cVert);

        if (_parentFVar.edgeTopologyMatches(pEdge)) {
            std::fill(cValueTags.begin(), cValueTags.end(), valTagMatch);
        } else {
            std::fill(cValueTags.begin(), cValueTags.end(), valTagSplitEdge);
        }
    }

    //
    //  Values from vertex-vertices -- inherit tags from parent values when complete
    //  otherwise (not yet supported) need to identify the parent value for each child:
    //
    cVert    = _refinement.getFirstChildVertexFromVertices();
    cVertEnd = cVert + _refinement.getNumChildVerticesFromVertices();
    for ( ; cVert < cVertEnd; ++cVert) {
        Index pVert = _refinement.getChildVertexParentIndex(cVert);
        assert(!_refinement._childVertexTag[cVert]._incomplete);

        FVarLevel::ConstValueTagArray pValueTags = _parentFVar.getVertexValueTags(pVert);
        FVarLevel::ValueTagArray cValueTags = _childFVar.getVertexValueTags(cVert);

        memcpy(cValueTags.begin(), pValueTags.begin(),
            pValueTags.size()*sizeof(FVarLevel::ValueTag));
    }
}

void
FVarRefinement::propagateValueCreases() {

    assert(_childFVar._hasSmoothBoundaries);

    //  Skip child vertices from faces:

    //
    //  For each child vertex from an edge that has FVar values and is complete, initialize
    //  the crease-ends for those values tagged as smooth boundaries
    //
    //  Note that this does depend on the nature of the topological split, i.e. how many
    //  child faces are incident the new child vertex for each face that becomes a crease,
    //  so identify constants to be used in each iteration first:
    //
    LocalIndex crease0StartFace = 0;
    LocalIndex crease0EndFace   = 1;
    LocalIndex crease1StartFace = 2;
    LocalIndex crease1EndFace   = 3;

    if (_refinement._splitType == Sdc::SPLIT_TO_TRIS) {
        crease0EndFace   = 2;
        crease1StartFace = 3;
        crease1EndFace   = 5;
    }

    Index cVert    = _refinement.getFirstChildVertexFromEdges();
    Index cVertEnd = cVert + _refinement.getNumChildVerticesFromEdges();
    for ( ; cVert < cVertEnd; ++cVert) {
        FVarLevel::ValueTagArray cValueTags = _childFVar.getVertexValueTags(cVert);

        if (!cValueTags[0].isMismatch()) continue;
        if (_refinement._childVertexTag[cVert]._incomplete) continue;

        FVarLevel::CreaseEndPairArray cValueCreaseEnds = _childFVar.getVertexValueCreaseEnds(cVert);

        if (!cValueTags[0].isInfSharp()) {
            cValueCreaseEnds[0]._startFace = crease0StartFace;
            cValueCreaseEnds[0]._endFace   = crease0EndFace;
        }
        if ((cValueTags.size() > 1) && !cValueTags[1].isInfSharp()) {
            cValueCreaseEnds[1]._startFace = crease1StartFace;
            cValueCreaseEnds[1]._endFace   = crease1EndFace;
        }
    }

    //
    //  For each child vertex from a vertex that has FVar values and is complete, initialize
    //  the crease-ends for those values tagged as smooth or semi-sharp (to become smooth
    //  eventually):
    //
    cVert    = _refinement.getFirstChildVertexFromVertices();
    cVertEnd = cVert + _refinement.getNumChildVerticesFromVertices();
    for ( ; cVert < cVertEnd; ++cVert) {
        FVarLevel::ValueTagArray cValueTags = _childFVar.getVertexValueTags(cVert);

        if (!cValueTags[0].isMismatch()) continue;
        if (_refinement._childVertexTag[cVert]._incomplete) continue;

        Index pVert = _refinement.getChildVertexParentIndex(cVert);

        FVarLevel::ConstCreaseEndPairArray pCreaseEnds = _parentFVar.getVertexValueCreaseEnds(pVert);
        FVarLevel::CreaseEndPairArray cCreaseEnds = _childFVar.getVertexValueCreaseEnds(cVert);

        for (int j = 0; j < cValueTags.size(); ++j) {
            if (!cValueTags[j].isInfSharp()) {
                cCreaseEnds[j] = pCreaseEnds[j];
            }
        }
    }
}

void
FVarRefinement::reclassifySemisharpValues() {

    //
    //  Reclassify the tags of semi-sharp vertex values to smooth creases according to
    //  changes in sharpness:
    //
    //  Vertex values introduced on edge-verts can never be semi-sharp as they will be
    //  introduced on discts edges, which are implicitly infinitely sharp, so we can
    //  skip them entirely.
    //
    //  So we just need to deal with those values descended from parent vertices that
    //  were semi-sharp.  The child values will have inherited the semi-sharp tag from
    //  their parent values -- we will be able to clear it in many simple cases but
    //  ultimately will need to inspect each value:
    //
    bool hasDependentSharpness = _parentFVar._hasDependentSharpness;

    FVarLevel::ValueTag valTagCrease;
    valTagCrease.clear();
    valTagCrease._mismatch = true;
    valTagCrease._crease   = true;

    FVarLevel::ValueTag valTagSemiSharp;
    valTagSemiSharp.clear();
    valTagSemiSharp._mismatch = true;
    valTagSemiSharp._semiSharp = true;

    Index cVert    = _refinement.getFirstChildVertexFromVertices();
    Index cVertEnd = cVert + _refinement.getNumChildVerticesFromVertices();

    for ( ; cVert < cVertEnd; ++cVert) {
        FVarLevel::ValueTagArray cValueTags = _childFVar.getVertexValueTags(cVert);

        if (!cValueTags[0].isMismatch()) continue;
        if (_refinement._childVertexTag[cVert]._incomplete) continue;

        //  If the parent vertex wasn't semi-sharp, the child vertex and values can't be:
        Index pVert = _refinement.getChildVertexParentIndex(cVert);
        if (!_parentLevel._vertTags[pVert]._semiSharp) continue;

        //  If the child vertex is still sharp, all values remain unaffected:
        if (!Sdc::Crease::IsSmooth(_childLevel._vertSharpness[cVert])) continue;

        //  If the child is no longer semi-sharp, we can just clear those values marked
        //  (i.e. make them creases, others may remain corners) and continue:
        //
        if (!_childLevel._vertTags[cVert]._semiSharp) {
            for (int j = 0; j < cValueTags.size(); ++j) {
                if (cValueTags[j]._semiSharp) {
                    cValueTags[j] = valTagCrease;
                }
            }
            continue;
        }

        //  There are some semi-sharp edges left -- for those values tagged as semi-sharp,
        //  see if they are still semi-sharp and clear those that are not:
        //
        FVarLevel::CreaseEndPairArray const cValueCreaseEnds = _childFVar.getVertexValueCreaseEnds(cVert);

        ConstIndexArray  cVertEdges = _childLevel.getVertexEdges(cVert);

        for (int j = 0; j < cValueTags.size(); ++j) {
            if (cValueTags[j]._semiSharp && !cValueTags[j]._depSharp) {
                LocalIndex vStartFace = cValueCreaseEnds[j]._startFace;
                LocalIndex vEndFace   = cValueCreaseEnds[j]._endFace;

                bool isStillSemiSharp = false;
                if (vEndFace > vStartFace) {
                    for (int k = vStartFace + 1; !isStillSemiSharp && (k <= vEndFace); ++k) {
                        isStillSemiSharp = _childLevel._edgeTags[cVertEdges[k]]._semiSharp;
                    }
                } else if (vStartFace > vEndFace) {
                    for (int k = vStartFace + 1; !isStillSemiSharp && (k < cVertEdges.size()); ++k) {
                        isStillSemiSharp = _childLevel._edgeTags[cVertEdges[k]]._semiSharp;
                    }
                    for (int k = 0; !isStillSemiSharp && (k <= vEndFace); ++k) {
                        isStillSemiSharp = _childLevel._edgeTags[cVertEdges[k]]._semiSharp;
                    }
                }
                if (!isStillSemiSharp) {
                    cValueTags[j] = valTagCrease;
                }
            }
        }

        //
        //  Now account for "dependent sharpness" (only matters when we have two values) --
        //  if one value was dependent/sharpened based on the other, clear the dependency
        //  tag if it is no longer sharp:
        //
        if ((cValueTags.size() == 2) && hasDependentSharpness) {
            if (cValueTags[0]._depSharp && !cValueTags[1]._semiSharp) {
                cValueTags[0]._depSharp = false;
            } else if (cValueTags[1]._depSharp && !cValueTags[0]._semiSharp) {
                cValueTags[1]._depSharp = false;
            }
        }
    }
}

float
FVarRefinement::getFractionalWeight(Index pVert, LocalIndex pSibling,
                                    Index cVert, LocalIndex /* cSibling */) const {

    //  Should only be called when the parent was semi-sharp but this child vertex
    //  value (not necessarily the child vertex as a whole) is no longer semi-sharp:
    assert(_parentLevel._vertTags[pVert]._semiSharp);
    assert(!_childLevel._vertTags[cVert]._incomplete);

    //
    //  Need to identify sharpness values for edges within the spans for both the
    //  parent and child...
    //
    //  Consider gathering the complete parent and child sharpness vectors outside
    //  this method and re-using them for each sibling, i.e. passing them to this
    //  method somehow.  We may also need them there for mask-related purposes...
    //
    ConstIndexArray  pVertEdges = _parentLevel.getVertexEdges(pVert);
    ConstIndexArray  cVertEdges = _childLevel.getVertexEdges(cVert);

    float * pEdgeSharpness = (float*) alloca(2 * pVertEdges.size() * sizeof(float));
    float * cEdgeSharpness = pEdgeSharpness + pVertEdges.size();

    FVarLevel::CreaseEndPair pValueCreaseEnds = _parentFVar.getVertexValueCreaseEnds(pVert)[pSibling];

    LocalIndex pStartFace = pValueCreaseEnds._startFace;
    LocalIndex pEndFace   = pValueCreaseEnds._endFace;

    int interiorEdgeCount = 0;
    if (pEndFace > pStartFace) {
        for (int i = pStartFace + 1; i <= pEndFace; ++i, ++interiorEdgeCount) {
            pEdgeSharpness[interiorEdgeCount] = _parentLevel._edgeSharpness[pVertEdges[i]];
            cEdgeSharpness[interiorEdgeCount] = _childLevel._edgeSharpness[cVertEdges[i]];
        }
    } else if (pStartFace > pEndFace) {
        for (int i = pStartFace + 1; i < pVertEdges.size(); ++i, ++interiorEdgeCount) {
            pEdgeSharpness[interiorEdgeCount] = _parentLevel._edgeSharpness[pVertEdges[i]];
            cEdgeSharpness[interiorEdgeCount] = _childLevel._edgeSharpness[cVertEdges[i]];
        }
        for (int i = 0; i <= pEndFace; ++i, ++interiorEdgeCount) {
            pEdgeSharpness[interiorEdgeCount] = _parentLevel._edgeSharpness[pVertEdges[i]];
            cEdgeSharpness[interiorEdgeCount] = _childLevel._edgeSharpness[cVertEdges[i]];
        }
    }
    return Sdc::Crease(_refinement._options).ComputeFractionalWeightAtVertex(
            _parentLevel._vertSharpness[pVert], _childLevel._vertSharpness[cVert],
            interiorEdgeCount, pEdgeSharpness, cEdgeSharpness);
}

} // end namespace Vtr

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv
