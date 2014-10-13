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
#include "../sdc/type.h"
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

typedef FVarLevel::Sibling      Sibling;
typedef FVarLevel::SiblingArray SiblingArray;

//
//  Simple (for now) constructor and destructor:
//
FVarRefinement::FVarRefinement(Refinement const& refinement,
                                     FVarLevel&        parentFVarLevel,
                                     FVarLevel&        childFVarLevel) :
    _refinement(refinement),
    _parent(&parentFVarLevel),
    _child(&childFVarLevel),
    _childSiblingFromEdgeCount(0),
    _childSiblingFromVertCount(0) {
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
    _child->_options             = _parent->_options;
    _child->_isLinear            = _parent->_isLinear;
    _child->_hasSmoothBoundaries = _parent->_hasSmoothBoundaries;

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
    if (_child->_hasSmoothBoundaries) {
        propagateValueCreases();
        reclassifySemisharpValues();
    }

    //
    //  The refined face-values are technically redundant as they can be constructed
    //  from the face-vertex siblings -- do so here as a post-process
    //
    if (_childSiblingFromEdgeCount || _childSiblingFromVertCount) {
        _child->initializeFaceValuesFromVertexFaceSiblings(_refinement._childVertFromFaceCount);
    } else {
        _child->initializeFaceValuesFromFaceVertices();
    }

    //printf("FVar refinement to level %d:\n", _child->getDepth());
    //_child->print();

    //printf("Validating refinement to level %d:\n", _child->getDepth());
    //_child->validate();
    //assert(_child->validate());
}

//
//  Quickly estimate the memory required for face-varying vertex-values in the child
//  and allocate them.  For uniform refinement this estimate should exactly match the
//  desired result.  For sparse refinement the excess should generally be low as the
//  sparse boundary components generally occur where face-varying data is continuous.
//
void
FVarRefinement::estimateAndAllocateChildValues() {

    int maxVertexValueCount = _refinement._childVertFromFaceCount;

    Index cVert = _refinement._childVertFromFaceCount;
    for (int i = 0; i < _refinement._childVertFromEdgeCount; ++i, ++cVert) {
        Index pEdge = _refinement.getChildVertexParentIndex(cVert);

        if ( _parent->_edgeTags[pEdge]._mismatch) {
            maxVertexValueCount += _refinement._parent->getEdgeFaces(pEdge).size();
        } else {
            maxVertexValueCount += 1;
        }
    }
    for (int i = 0; i < _refinement._childVertFromVertCount; ++i, ++cVert) {
        assert(!_refinement._childVertexTag[cVert]._incomplete);
        Index pVert = _refinement.getChildVertexParentIndex(cVert);

        if (_parent->_vertValueTags[pVert]._mismatch) {
            maxVertexValueCount += _parent->getNumVertexValues(pVert);
        } else {
            maxVertexValueCount += 1;
        }
    }

    //
    //  Now allocate/initialize for the maximum -- use resize() and trim the size later
    //  to avoid the constant growing with reserve() and incremental sizing.  We know
    //  the estimate should be close and memory wasted should be small, so initialize
    //  all to zero as well to avoid writing in all but affected areas:
    //
    //  Resize vectors that mirror the component counts:
    _child->resizeComponents();

    //  Resize the vertex-value tags in the child level:
    _child->_vertValueTags.resize(maxVertexValueCount);

    //  Resize the vertex-value "parent source" mapping in the refinement:
    _childValueParentSource.resize(maxVertexValueCount, 0);
}

void
FVarRefinement::trimAndFinalizeChildValues() {

    _child->_valueCount = _child->getNumVertices() + _childSiblingFromEdgeCount + _childSiblingFromVertCount;

    _child->_vertValueTags.resize(_child->_valueCount);
    if (_child->_hasSmoothBoundaries) {
        _child->_vertValueCreaseEnds.resize(2 * _child->_valueCount);
    }

    _childValueParentSource.resize(_child->_valueCount);

    //  ? - Update vertex sibling offsets based on now-accurate sibling counts:

    //  Allocate/initialize the indices (potentially redundant after level 0):
    _child->_vertValueIndices.resize(_child->_valueCount);
    for (int i = 0; i < _child->_valueCount; ++i) {
        _child->_vertValueIndices[i] = i;
    }
}

inline int
FVarRefinement::populateChildValuesForEdgeVertex(Index cVert, Index pEdge, int siblingOffset) {

    //  If we have a boundary edge with a mismatched end vertex, we only have one
    //  value and such cases were already initialized on construction, so return:
    //
    IndexArray const pEdgeFaces = _refinement._parent->getEdgeFaces(pEdge);

    if (pEdgeFaces.size() == 1) return 0;
    assert(pEdgeFaces.size() == 2);

    //  Determine the number of sibling values for the child vertex:
    //
    IndexArray const cVertFaces = _refinement._child->getVertexFaces(cVert);

    int cSiblingCount = 0;
    if (cVertFaces.size() > 2) {
        cSiblingCount = 1;
    } else if (cVertFaces.size() == 2) {
        cSiblingCount = (cVertFaces[0] != cVertFaces[1]);
    }

    //
    //  We may have no sibling values here when the edge was discts, but only in the
    //  case where the refinement was sparse (and so the child vertex "incomplete").
    //  In such a case the source of the single value needs to be identified:
    //
    if (cSiblingCount == 0) {
        _childValueParentSource[cVert] = (cVertFaces[0] != pEdgeFaces[0]);
    } else {
        assert (cSiblingCount == 1);

        //  Update the count/offset:
        _child->_vertSiblingCounts[cVert]  = (LocalIndex) cSiblingCount;
        _child->_vertSiblingOffsets[cVert] = siblingOffset;

        //  Update the parent-source of any siblings (typically one):
        for (int j = 0; j < cSiblingCount; ++j) {
            _childValueParentSource[siblingOffset + j] = (LocalIndex) (j + 1);
        }

        //  Update the vertex-face siblings:
        SiblingArray cVertFaceSiblings = _child->getVertexFaceSiblings(cVert);
        for (int j = 0; j < cVertFaceSiblings.size(); ++j) {
            if (_refinement.getChildFaceParentFace(cVertFaces[j]) == pEdgeFaces[1]) {
                cVertFaceSiblings[j] = 1;
            }
        }
    }
    return cSiblingCount;
}

inline int
FVarRefinement::populateChildValuesForVertexVertex(Index cVert, Index pVert, int siblingOffset) {

    //
    //  We should not be getting incomplete vertex-vertices from feature-adaptive
    //  refinement (as neighboring vertices will be face-vertices or edge-vertices).
    //  This will get messy when we do (i.e. sparse refinement of Bilinear or more
    //  flexible and specific sparse refinement of Catmark) but for now assume 1-to-1.
    //
    assert(!_refinement._childVertexTag[cVert]._incomplete);

    int cSiblingCount = _parent->_vertSiblingCounts[pVert];
    if (cSiblingCount) {
        _child->_vertSiblingCounts[cVert]  = (LocalIndex) cSiblingCount;
        _child->_vertSiblingOffsets[cVert] = siblingOffset;

        // Update the parent source:
        for (int j = 0; j < cSiblingCount; ++j) {
            _childValueParentSource[siblingOffset + j] = (LocalIndex)(j + 1);
        }

        // Update the vertex-face siblings:
        SiblingArray const pVertFaceSiblings = _parent->getVertexFaceSiblings(pVert);
        SiblingArray       cVertFaceSiblings = _child->getVertexFaceSiblings(cVert);
        for (int j = 0; j < cVertFaceSiblings.size(); ++j) {
            cVertFaceSiblings[j] = pVertFaceSiblings[j];
        }
    }
    return cSiblingCount;
}

void
FVarRefinement::populateChildValues() {

    //  For values from face-vertices, they are guaranteed to be continuous, so there is
    //  nothing we need do here -- other than skipping them and starting with the first
    //  child vertex after all face vertices.
    //
    //  For values from edge-vertices and vertex-vertices, identify all associated values
    //  for those that do not have matching topology -- those that do will have been
    //  initialized appropriately on construction.
    //
    //  The "sibling offset" is an offset into the member vectors where sibling values are
    //  "located".  Every vertex has at least one value and so that value is located at the
    //  index of the vertex -- all additional values are located after the one-per-vertex.
    //
    int siblingValueOffset = _child->getNumVertices();

    Index cVert = _refinement._childVertFromFaceCount;

    for (int i = 0; i < _refinement._childVertFromEdgeCount; ++i, ++cVert) {
        Index pEdge = _refinement.getChildVertexParentIndex(cVert);
        if (_parent->_edgeTags[pEdge]._mismatch) {
            int cSiblingCount = populateChildValuesForEdgeVertex(cVert, pEdge, siblingValueOffset);

            siblingValueOffset += cSiblingCount;
            _childSiblingFromEdgeCount += cSiblingCount;
        }
    }
    for (int i = 0; i < _refinement._childVertFromVertCount; ++i, ++cVert) {
        Index pVert = _refinement.getChildVertexParentIndex(cVert);
        if (_parent->_vertValueTags[pVert]._mismatch) {
            int cSiblingCount = populateChildValuesForVertexVertex(cVert, pVert, siblingValueOffset);

            siblingValueOffset += cSiblingCount;
            _childSiblingFromVertCount += cSiblingCount;
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
        _child->_edgeTags[eIndex] = eTagMatch;
    }
    for (int eIndex = _refinement._childEdgeFromFaceCount; eIndex < _child->getNumEdges(); ++eIndex) {
        Index pEdge = _refinement.getChildEdgeParentIndex(eIndex);

        _child->_edgeTags[eIndex] = _parent->_edgeTags[pEdge];
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
    //  Values from face-vertices -- all match:
    //
    FVarLevel::ValueTag valTagMatch;
    valTagMatch.clear();

    Index cVert = 0;
    for (cVert = 0; cVert < _refinement._childVertFromFaceCount; ++cVert) {
        _child->_vertValueTags[cVert] = valTagMatch;
    }

    //
    //  Values from edge-vertices -- for edges that are split, tag as mismatched and tag
    //  as corner or crease depending on the presence of creases in the parent:
    //
    FVarLevel::ValueTag valTagMismatch = valTagMatch;
    valTagMismatch._mismatch = true;

    FVarLevel::ValueTag valTagCrease = valTagMismatch;
    valTagCrease._crease = true;

    FVarLevel::ValueTag& valTagSplitEdge = _parent->_hasSmoothBoundaries ? valTagCrease : valTagMismatch;

    for (int i = 0; i < _refinement._childVertFromEdgeCount; ++i, ++cVert) {
        Index pEdge = _refinement.getChildVertexParentIndex(cVert);
        bool pEdgeIsSplit = _parent->_edgeTags[pEdge]._mismatch;

        FVarLevel::ValueTag const& cValueTag = pEdgeIsSplit ? valTagSplitEdge: valTagMatch;

        _child->_vertValueTags[cVert] = cValueTag;

        int vSiblingCount = _child->_vertSiblingCounts[cVert];
        if (vSiblingCount) {
            int cOffset = _child->_vertSiblingOffsets[cVert];
            for (int j = 0; j < vSiblingCount; ++j) {
                _child->_vertValueTags[cOffset + j] = cValueTag;
            }
        }
    }

    //
    //  Values from vertex-vertices -- inherit tags from parent values when complete
    //  otherwise (not yet supported) need to identify the parent value for each child:
    //
    for (int i = 0; i < _refinement._childVertFromVertCount; ++i, ++cVert) {
        assert(!_refinement._childVertexTag[cVert]._incomplete);

        Index pVert = _refinement.getChildVertexParentIndex(cVert);

        _child->_vertValueTags[cVert] = _parent->_vertValueTags[pVert];

        int vSiblingCount = _child->_vertSiblingCounts[cVert];
        if (vSiblingCount) {
            int cOffset = _child->_vertSiblingOffsets[cVert];
            int pOffset = _parent->_vertSiblingOffsets[pVert];
            for (int j = 0; j < vSiblingCount; ++j) {
                _child->_vertValueTags[cOffset + j] = _parent->_vertValueTags[pOffset + j];
            }
        }
    }
}

void
FVarRefinement::propagateValueCreases() {

    assert(_child->_hasSmoothBoundaries);

    //  Skip child vertices from faces:
    Index cVert = _refinement._childVertFromFaceCount;

    //
    //  For each child vertex from an edge that has FVar values and is complete, initialize
    //  the crease-ends for those values tagged as smooth boundaries:
    //
    for (int i = 0; i < _refinement._childVertFromEdgeCount; ++i, ++cVert) {
        if (!_child->_vertValueTags[cVert]._mismatch) continue;
        if (_refinement._childVertexTag[cVert]._incomplete) continue;

        if (!_child->isValueInfSharp(cVert)) {
            LocalIndex * sibling0Ends = &_child->_vertValueCreaseEnds[2 * cVert];
            sibling0Ends[0] = 0;
            sibling0Ends[1] = 1;
        }
        int vSiblingCount = _child->_vertSiblingCounts[cVert];
        if (vSiblingCount) {
            int cOffset = _child->_vertSiblingOffsets[cVert];
            if (!_child->isValueInfSharp(cOffset)) {
                LocalIndex * sibling1Ends = &_child->_vertValueCreaseEnds[2 * cOffset];
                sibling1Ends[0] = 2;
                sibling1Ends[1] = 3;
            }
        }
    }

    //
    //  For each child vertex from a vertex that has FVar values and is complete, initialize
    //  the crease-ends for those values tagged as smooth or semi-sharp (to become smooth
    //  eventually):
    //
    for (int i = 0; i < _refinement._childVertFromVertCount; ++i, ++cVert) {
        if (!_child->_vertValueTags[cVert]._mismatch) continue;
        if (_refinement._childVertexTag[cVert]._incomplete) continue;

        Index pVert = _refinement.getChildVertexParentIndex(cVert);

        if (!_child->isValueInfSharp(cVert)) {
            LocalIndex * pSiblingEnds = &_parent->_vertValueCreaseEnds[2 * pVert];
            LocalIndex * cSiblingEnds = &_child->_vertValueCreaseEnds[2 * cVert];

            cSiblingEnds[0] = pSiblingEnds[0];
            cSiblingEnds[1] = pSiblingEnds[1];
        }
        int vSiblingCount = _child->_vertSiblingCounts[cVert];
        if (vSiblingCount) {
            int cSiblingOffset = _child->_vertSiblingOffsets[cVert];
            int pSiblingOffset = _parent->_vertSiblingOffsets[pVert];

            LocalIndex * pSiblingEnds = &_parent->_vertValueCreaseEnds[2 * pSiblingOffset];
            LocalIndex * cSiblingEnds = &_child->_vertValueCreaseEnds[2 * cSiblingOffset];

            for (int j = 0; j < vSiblingCount; ++j, cSiblingEnds += 2, pSiblingEnds += 2) {
                if (!_child->isValueInfSharp(cSiblingOffset + j)) {
                    cSiblingEnds[0] = pSiblingEnds[0];
                    cSiblingEnds[1] = pSiblingEnds[1];
                }
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
    FVarLevel::ValueTag valTagCrease;
    valTagCrease.clear();
    valTagCrease._mismatch = true;
    valTagCrease._crease   = true;

    FVarLevel::ValueTag valTagSemiSharp;
    valTagSemiSharp.clear();
    valTagSemiSharp._mismatch = true;
    valTagSemiSharp._semiSharp = true;

    Index cVert = _refinement._childVertFromFaceCount + _refinement._childVertFromEdgeCount;

    for (int i = 0; i < _refinement._childVertFromVertCount; ++i, ++cVert) {
        if (!_child->_vertValueTags[cVert]._mismatch) continue;
        if (_refinement._childVertexTag[cVert]._incomplete) continue;

        //  Remember the child vertex may no longer be semisharp when the parent was:
        Index pVert = _refinement.getChildVertexParentIndex(cVert);
        if (!_refinement._parent->_vertTags[pVert]._semiSharp) continue;

        //  If the child vertex sharpness is non-zero, all values remain unaffected:
        if (!Sdc::Crease::IsSmooth(_refinement._child->_vertSharpness[cVert])) continue;

        //
        //  We are left with some or no semi-sharp edges in the child vertex.  If none
        //  left the child-vertex will no longer be semi-sharp and we can just clear
        //  those values marked as semi-sharp.  Otherwise, its simplest to assume all
        //  semi-sharp edges have decayed (so clearing them again) and using the
        //  remaining semi-sharp edges to reset those values that still are.
        //
        //  So convert all semi-sharp tags to creases and reset those that remain if
        //  the child vertex is still tagged as semi-sharp:
        //
        int vSiblingOffset = _child->_vertSiblingOffsets[cVert];
        int vSiblingCount  = _child->_vertSiblingCounts[cVert];
        int vValueCount    = 1 + vSiblingCount;
        for (int j = 0; j < vValueCount; ++j) {
            int vValueIndex = (j == 0) ? cVert : (vSiblingOffset + j - 1);

            if (_child->isValueSemiSharp(vValueIndex)) {
                _child->_vertValueTags[vValueIndex] = valTagCrease;
            }
        }
        if (!_refinement._child->_vertTags[cVert]._semiSharp) continue;

        //
        //  Identify the remaining semi-sharp edges and the corresponding values that
        //  should be retained as semi-sharp:
        //
        SiblingArray const vFaceSiblings = _child->getVertexFaceSiblings(cVert);

        IndexArray const cVertEdges = _refinement._child->getVertexEdges(cVert);
        for (int j = 0; j < cVertEdges.size(); ++j) {
            if (_refinement._child->_edgeTags[cVertEdges[j]]._semiSharp) {
                int jPrev = j ? (j - 1) : (vFaceSiblings.size() - 1);
                if (vFaceSiblings[jPrev] == vFaceSiblings[j]) {
                    int vSibling = vFaceSiblings[j];
                    int vValueIndex = (vSibling == 0) ? cVert : (vSiblingOffset + vSibling - 1);

                    _child->_vertValueTags[vValueIndex] = valTagSemiSharp;
                }
            }
        }
    }
}

float
FVarRefinement::getFractionalWeight(Index pVert, Sibling pSibling,
                                    Index cVert, Sibling /* cSibling */) const
{
    FVarLevel const& parentFVar = *_parent;
    Level     const& parent     = *_refinement._parent;
    Level     const& child      = *_refinement._child;

    //  Should only be called when the parent was semi-sharp but this child vertex
    //  value (not necessarily the child vertex as a whole) is no longer semi-sharp:
    assert(parent._vertTags[pVert]._semiSharp);
    assert(!child._vertTags[cVert]._incomplete);

    //
    //  Need to identify sharpness values for edges within the spans for both the
    //  parent and child...
    //
    //  Consider gathering the complete parent and child sharpness vectors outside
    //  this method and re-using them for each sibling, i.e. passing them to this
    //  method somehow.  We may also need them there for mask-related purposes...
    //
    IndexArray const pVertEdges = parent.getVertexEdges(pVert);
    IndexArray const cVertEdges = child.getVertexEdges(cVert);

    float * pEdgeSharpness = (float*) alloca(2 * pVertEdges.size() * sizeof(float));
    float * cEdgeSharpness = pEdgeSharpness + pVertEdges.size();

    int                pValueIndex  = parentFVar.getVertexValueIndex(pVert, pSibling);
    LocalIndex const * pSiblingEnds = &parentFVar._vertValueCreaseEnds[2 * pValueIndex];

    int interiorEdgeCount = 0;
    if (pSiblingEnds[1] > pSiblingEnds[0]) {
        for (int i = pSiblingEnds[0] + 1; i <= pSiblingEnds[1]; ++i, ++interiorEdgeCount) {
            pEdgeSharpness[interiorEdgeCount] = parent._edgeSharpness[pVertEdges[i]];
            cEdgeSharpness[interiorEdgeCount] = child._edgeSharpness[cVertEdges[i]];
        }
    } else if (pSiblingEnds[0] > pSiblingEnds[1]) {
        for (int i = pSiblingEnds[0] + 1; i < pVertEdges.size(); ++i, ++interiorEdgeCount) {
            pEdgeSharpness[interiorEdgeCount] = parent._edgeSharpness[pVertEdges[i]];
            cEdgeSharpness[interiorEdgeCount] = child._edgeSharpness[cVertEdges[i]];
        }
        for (int i = 0; i <= pSiblingEnds[1]; ++i, ++interiorEdgeCount) {
            pEdgeSharpness[interiorEdgeCount] = parent._edgeSharpness[pVertEdges[i]];
            cEdgeSharpness[interiorEdgeCount] = child._edgeSharpness[cVertEdges[i]];
        }
    }
    return Sdc::Crease(_refinement._schemeOptions).ComputeFractionalWeightAtVertex(
            parent._vertSharpness[pVert], child._vertSharpness[cVert],
            interiorEdgeCount, pEdgeSharpness, cEdgeSharpness);
}

} // end namespace Vtr

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv
