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
#include "../sdc/crease.h"
#include "../vtr/types.h"
#include "../vtr/level.h"
#include "../vtr/quadRefinement.h"

#include <cassert>
#include <cstdio>
#include <utility>


namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Vtr {

//
//  Simple constructor, destructor and basic initializers:
//
QuadRefinement::QuadRefinement(Level const & parent, Level & child, Sdc::Options const & options) :
    Refinement(parent, child, options) {

    _splitType   = Sdc::SPLIT_TO_QUADS;
    _regFaceSize = 4;
}

QuadRefinement::~QuadRefinement() {
}


//
//  Methods for construct the parent-to-child mapping
//
void
QuadRefinement::allocateParentChildIndices() {

    //
    //  Initialize the vectors of indices mapping parent components to those child components
    //  that will originate from each.
    //
    int faceChildFaceCount = (int) _parent->_faceVertIndices.size();
    int faceChildEdgeCount = (int) _parent->_faceEdgeIndices.size();
    int edgeChildEdgeCount = (int) _parent->_edgeVertIndices.size();

    int faceChildVertCount = _parent->getNumFaces();
    int edgeChildVertCount = _parent->getNumEdges();
    int vertChildVertCount = _parent->getNumVertices();

    //
    //  First reference the parent Level's face-vertex counts/offsets -- they can be used
    //  here for both the face-child-faces and face-child-edges as they both have one per
    //  face-vertex.
    //
    //  Given we will be ignoring initial values with uniform refinement and assigning all
    //  directly, initializing here is a waste...
    //
    Index initValue = 0;

    _faceChildFaceCountsAndOffsets = _parent->shareFaceVertCountsAndOffsets();
    _faceChildEdgeCountsAndOffsets = _parent->shareFaceVertCountsAndOffsets();

    _faceChildFaceIndices.resize(faceChildFaceCount, initValue);
    _faceChildEdgeIndices.resize(faceChildEdgeCount, initValue);
    _edgeChildEdgeIndices.resize(edgeChildEdgeCount, initValue);

    _faceChildVertIndex.resize(faceChildVertCount, initValue);
    _edgeChildVertIndex.resize(edgeChildVertCount, initValue);
    _vertChildVertIndex.resize(vertChildVertCount, initValue);
}


//
//  Methods to populate the face-vertex relation of the child Level:
//      - child faces only originate from parent faces
//
void
QuadRefinement::populateFaceVertexRelation() {

    //  Both face-vertex and face-edge share the face-vertex counts/offsets within a
    //  Level, so be sure not to re-initialize it if already done:
    //
    if (_child->_faceVertCountsAndOffsets.size() == 0) {
        populateFaceVertexCountsAndOffsets();
    }
    _child->_faceVertIndices.resize(_child->getNumFaces() * 4);

    populateFaceVerticesFromParentFaces();
}

void
QuadRefinement::populateFaceVertexCountsAndOffsets() {

    _child->_faceVertCountsAndOffsets.resize(_child->getNumFaces() * 2);

    for (int i = 0; i < _child->getNumFaces(); ++i) {
        _child->_faceVertCountsAndOffsets[i*2 + 0] = 4;
        _child->_faceVertCountsAndOffsets[i*2 + 1] = i << 2;
    }
}

void
QuadRefinement::populateFaceVerticesFromParentFaces() {

    //
    //  Algorithm:
    //    - iterate through parent face-child-face vector (could use back-vector)
    //    - use parent components incident the parent face:
    //        - use the interior face-vert, corner vert-vert and two edge-verts
    //
    for (Index pFace = 0; pFace < _parent->getNumFaces(); ++pFace) {
        ConstIndexArray pFaceVerts = _parent->getFaceVertices(pFace),
                        pFaceEdges = _parent->getFaceEdges(pFace),
                        pFaceChildren = getFaceChildFaces(pFace);

        int pFaceVertCount = pFaceVerts.size();
        for (int j = 0; j < pFaceVertCount; ++j) {
            Index cFace = pFaceChildren[j];
            if (IndexIsValid(cFace)) {
                int jPrev = j ? (j - 1) : (pFaceVertCount - 1);

                Index cVertOfFace  = _faceChildVertIndex[pFace];
                Index cVertOfEPrev = _edgeChildVertIndex[pFaceEdges[jPrev]];
                Index cVertOfVert  = _vertChildVertIndex[pFaceVerts[j]];
                Index cVertOfENext = _edgeChildVertIndex[pFaceEdges[j]];

                IndexArray cFaceVerts = _child->getFaceVertices(cFace);

                //  Note orientation wrt parent face -- quad vs non-quad...
                if (pFaceVertCount == 4) {
                    int jOpp  = jPrev ? (jPrev - 1) : 3;
                    int jNext = jOpp  ? (jOpp  - 1) : 3;

                    cFaceVerts[j]     = cVertOfVert;
                    cFaceVerts[jNext] = cVertOfENext;
                    cFaceVerts[jOpp]  = cVertOfFace;
                    cFaceVerts[jPrev] = cVertOfEPrev;
                } else {
                    cFaceVerts[0] = cVertOfVert;
                    cFaceVerts[1] = cVertOfENext;
                    cFaceVerts[2] = cVertOfFace;
                    cFaceVerts[3] = cVertOfEPrev;
                }
            }
        }
    }
}


//
//  Methods to populate the face-vertex relation of the child Level:
//      - child faces only originate from parent faces
//
void
QuadRefinement::populateFaceEdgeRelation() {

    //  Both face-vertex and face-edge share the face-vertex counts/offsets, so be sure
    //  not to re-initialize it if already done:
    //
    if (_child->_faceVertCountsAndOffsets.size() == 0) {
        populateFaceVertexCountsAndOffsets();
    }
    _child->_faceEdgeIndices.resize(_child->getNumFaces() * 4);

    populateFaceEdgesFromParentFaces();
}

void
QuadRefinement::populateFaceEdgesFromParentFaces() {

    //
    //  Algorithm:
    //    - iterate through parent face-child-face vector (could use back-vector)
    //    - use parent components incident the parent face:
    //        - use the two interior face-edges and the two boundary edge-edges
    //
    for (Index pFace = 0; pFace < _parent->getNumFaces(); ++pFace) {
        ConstIndexArray pFaceVerts = _parent->getFaceVertices(pFace),
                        pFaceEdges = _parent->getFaceEdges(pFace),
                        pFaceChildFaces = getFaceChildFaces(pFace),
                        pFaceChildEdges = getFaceChildEdges(pFace);

        int pFaceVertCount = pFaceVerts.size();

        for (int j = 0; j < pFaceVertCount; ++j) {
            Index cFace = pFaceChildFaces[j];
            if (IndexIsValid(cFace)) {
                IndexArray cFaceEdges = _child->getFaceEdges(cFace);

                int jPrev = j ? (j - 1) : (pFaceVertCount - 1);

                //
                //  We have two edges that are children of parent edges, and two child
                //  edges perpendicular to these from the interior of the parent face:
                //
                //  Identifying the former should be simpler -- after identifying the two
                //  parent edges, we have to identify which child-edge corresponds to this
                //  vertex.  This may be ambiguous with a degenerate edge (DEGEN) if tested
                //  this way, and may warrant higher level inspection of the parent face...
                //
                //  EDGE_IN_FACE -- having the edge-in-face local index would help to
                //  remove the ambiguity and simplify this.
                //
                Index pCornerVert = pFaceVerts[j];

                Index           pPrevEdge      = pFaceEdges[jPrev];
                ConstIndexArray pPrevEdgeVerts = _parent->getEdgeVertices(pPrevEdge);

                Index           pNextEdge      = pFaceEdges[j];
                ConstIndexArray pNextEdgeVerts = _parent->getEdgeVertices(pNextEdge);

                int cornerInPrevEdge = (pPrevEdgeVerts[0] != pCornerVert);
                int cornerInNextEdge = (pNextEdgeVerts[0] != pCornerVert);

                Index cEdgeOfEdgePrev = getEdgeChildEdges(pPrevEdge)[cornerInPrevEdge];
                Index cEdgeOfEdgeNext = getEdgeChildEdges(pNextEdge)[cornerInNextEdge];

                Index cEdgePerpEdgePrev = pFaceChildEdges[jPrev];
                Index cEdgePerpEdgeNext = pFaceChildEdges[j];

                //  Note orientation wrt parent face -- quad vs non-quad...
                if (pFaceVertCount == 4) {
                    int jOpp  = jPrev ? (jPrev - 1) : 3;
                    int jNext = jOpp  ? (jOpp  - 1) : 3;

                    cFaceEdges[j]     = cEdgeOfEdgeNext;
                    cFaceEdges[jNext] = cEdgePerpEdgeNext;
                    cFaceEdges[jOpp]  = cEdgePerpEdgePrev;
                    cFaceEdges[jPrev] = cEdgeOfEdgePrev;
                } else {
                    cFaceEdges[0] = cEdgeOfEdgeNext;
                    cFaceEdges[1] = cEdgePerpEdgeNext;
                    cFaceEdges[2] = cEdgePerpEdgePrev;
                    cFaceEdges[3] = cEdgeOfEdgePrev;
                }
            }
        }
    }
}

//
//  Methods to populate the edge-vertex relation of the child Level:
//      - child edges originate from parent faces and edges
//
void
QuadRefinement::populateEdgeVertexRelation() {

    _child->_edgeVertIndices.resize(_child->getNumEdges() * 2);

    populateEdgeVerticesFromParentFaces();
    populateEdgeVerticesFromParentEdges();
}

void
QuadRefinement::populateEdgeVerticesFromParentFaces() {

    //
    //  For each parent face's edge-children:
    //    - identify parent face's vert-child (note it is shared by all)
    //    - identify parent edge perpendicular to face's child edge:
    //        - identify parent edge's vert-child
    //
    for (Index pFace = 0; pFace < _parent->getNumFaces(); ++pFace) {
        ConstIndexArray pFaceEdges      = _parent->getFaceEdges(pFace),
                        pFaceChildEdges = getFaceChildEdges(pFace);

        for (int j = 0; j < pFaceEdges.size(); ++j) {
            Index cEdge = pFaceChildEdges[j];
            if (IndexIsValid(cEdge)) {
                IndexArray cEdgeVerts = _child->getEdgeVertices(cEdge);

                cEdgeVerts[0] = _faceChildVertIndex[pFace];
                cEdgeVerts[1] = _edgeChildVertIndex[pFaceEdges[j]];
            }
        }
    }
}

void
QuadRefinement::populateEdgeVerticesFromParentEdges() {

    //
    //  For each parent edge's edge-children:
    //    - identify parent edge's vert-child (potentially shared by both)
    //    - identify parent vert at end of child edge:
    //        - identify parent vert's vert-child
    //
    for (Index pEdge = 0; pEdge < _parent->getNumEdges(); ++pEdge) {
        ConstIndexArray pEdgeVerts = _parent->getEdgeVertices(pEdge),
                        pEdgeChildren = getEdgeChildEdges(pEdge);

        //  May want to unroll this trivial loop of 2...
        for (int j = 0; j < 2; ++j) {
            Index cEdge = pEdgeChildren[j];
            if (IndexIsValid(cEdge)) {
                IndexArray cEdgeVerts = _child->getEdgeVertices(cEdge);

                cEdgeVerts[0] = _edgeChildVertIndex[pEdge];
                cEdgeVerts[1] = _vertChildVertIndex[pEdgeVerts[j]];
            }
        }
    }
}

//
//  Methods to populate the edge-face relation of the child Level:
//      - child edges originate from parent faces and edges
//      - sparse refinement poses challenges with allocation here
//          - we need to update the counts/offsets as we populate
//
void
QuadRefinement::populateEdgeFaceRelation() {

    const Level& parent = *_parent;
          Level& child  = *_child;

    //
    //  Notes on allocating/initializing the edge-face counts/offsets vector:
    //
    //  Be aware of scheme-specific decisions here, e.g.:
    //      - inspection of sparse child faces for edges from faces
    //      - no guaranteed "neighborhood" around Bilinear verts from verts
    //
    //  If uniform subdivision, face count of a child edge will be:
    //      - 2 for new interior edges from parent faces
    //          == 2 * number of parent face verts for both quad- and tri-split
    //      - same as parent edge for edges from parent edges
    //  If sparse subdivision, face count of a child edge will be:
    //      - 1 or 2 for new interior edge depending on child faces in parent face
    //          - requires inspection if not all child faces present
    //      ? same as parent edge for edges from parent edges
    //          - given end vertex must have its full set of child faces
    //          - not for Bilinear -- only if neighborhood is non-zero
    //      - could at least make a quick traversal of components and use the above
    //        two points to get much closer estimate than what is used for uniform
    //
    int childEdgeFaceIndexSizeEstimate = (int)parent._faceVertIndices.size() * 2 +
                                         (int)parent._edgeFaceIndices.size() * 2;

    child._edgeFaceCountsAndOffsets.resize(child.getNumEdges() * 2);
    child._edgeFaceIndices.resize(childEdgeFaceIndexSizeEstimate);

    populateEdgeFacesFromParentFaces();
    populateEdgeFacesFromParentEdges();

    //  Revise the over-allocated estimate based on what is used (as indicated in the
    //  count/offset for the last vertex) and trim the index vector accordingly:
    childEdgeFaceIndexSizeEstimate = child.getNumEdgeFaces(child.getNumEdges()-1) +
                                     child.getOffsetOfEdgeFaces(child.getNumEdges()-1);
    child._edgeFaceIndices.resize(childEdgeFaceIndexSizeEstimate);

    child._maxEdgeFaces = parent._maxEdgeFaces;
}

void
QuadRefinement::populateEdgeFacesFromParentFaces() {

    //
    //  Note -- the edge-face counts/offsets vector is not known
    //  ahead of time and is populated incrementally, so we cannot
    //  thread this yet...
    //
    for (Index pFace = 0; pFace < _parent->getNumFaces(); ++pFace) {
        ConstIndexArray pFaceChildFaces = getFaceChildFaces(pFace),
                        pFaceChildEdges = getFaceChildEdges(pFace);

        int pFaceValence = _parent->getFaceVertices(pFace).size();

        for (int j = 0; j < pFaceValence; ++j) {
            Index cEdge = pFaceChildEdges[j];
            if (IndexIsValid(cEdge)) {
                //
                //  Reserve enough edge-faces, populate and trim as needed:
                //
                _child->resizeEdgeFaces(cEdge, 2);

                IndexArray cEdgeFaces = _child->getEdgeFaces(cEdge);

                //  One or two child faces may be assigned:
                int jNext = ((j + 1) < pFaceValence) ? (j + 1) : 0;

                int cEdgeFaceCount = 0;
                if (IndexIsValid(pFaceChildFaces[j])) {
                    cEdgeFaces[cEdgeFaceCount++] = pFaceChildFaces[j];
                }
                if (IndexIsValid(pFaceChildFaces[jNext])) {
                    cEdgeFaces[cEdgeFaceCount++] = pFaceChildFaces[jNext];
                }
                _child->trimEdgeFaces(cEdge, cEdgeFaceCount);
            }
        }
    }
}

void
QuadRefinement::populateEdgeFacesFromParentEdges() {

    //
    //  Note -- the edge-face counts/offsets vector is not known
    //  ahead of time and is populated incrementally, so we cannot
    //  thread this yet...
    //
    for (Index pEdge = 0; pEdge < _parent->getNumEdges(); ++pEdge) {
        ConstIndexArray pEdgeVerts = _parent->getEdgeVertices(pEdge),
                        pEdgeFaces = _parent->getEdgeFaces(pEdge),
                        pEdgeChildEdges = getEdgeChildEdges(pEdge);

        for (int j = 0; j < 2; ++j) {
            Index cEdge = pEdgeChildEdges[j];
            if (!IndexIsValid(cEdge)) continue;

            //
            //  Reserve enough edge-faces, populate and trim as needed:
            //
            _child->resizeEdgeFaces(cEdge, pEdgeFaces.size());

            IndexArray cEdgeFaces = _child->getEdgeFaces(cEdge);

            //
            //  Each parent face may contribute an incident child face:
            //
            //  EDGE_IN_FACE:
            //      This is awkward, and would be greatly simplified by storing the
            //  "edge in face" for each edge-face (as we do for "vert in face" of
            //  the vert-faces, etc.).  For each incident face we then immediately
            //  know the two child faces that are associated with the two child
            //  edges -- we just need to identify how to pair them based on the
            //  edge direction.
            //
            //      Note also here, that we could identify the pairs of child faces
            //  once for the parent before dealing with each child edge (we do the
            //  "find edge in face search" twice here as a result).  We will
            //  generally have 2 or 1 incident face to the parent edge so we
            //  can put the child-pairs on the stack.
            //
            //      Here's a more promising alternative -- instead of iterating
            //  through the child edges to "pull" data from the parent, iterate
            //  through the parent edges' faces and apply valid child faces to
            //  the appropriate child edge.  We should be able to use end-verts
            //  of the parent edge to get the corresponding child face for each,
            //  but we can't avoid a vert-in-face search and a subsequent parity
            //  test of the end-vert.
            //
            int cEdgeFaceCount = 0;

            for (int i = 0; i < pEdgeFaces.size(); ++i) {
                Index pFace = pEdgeFaces[i];

                ConstIndexArray pFaceEdges = _parent->getFaceEdges(pFace),
                                pFaceVerts = _parent->getFaceVertices(pFace),
                                pFaceChildren = getFaceChildFaces(pFace);

                int pFaceValence = pFaceVerts.size();

                //  EDGE_IN_FACE -- want to remove this search...
                int edgeInFace = 0;
                for ( ; pFaceEdges[edgeInFace] != pEdge; ++edgeInFace) ;

                //  Inspect either this child of the face or the next:
                int childInFace = edgeInFace + (pFaceVerts[edgeInFace] != pEdgeVerts[j]);
                if (childInFace == pFaceValence) childInFace = 0;

                if (IndexIsValid(pFaceChildren[childInFace])) {
                    cEdgeFaces[cEdgeFaceCount++] = pFaceChildren[childInFace];
                }
            }
            _child->trimEdgeFaces(cEdge, cEdgeFaceCount);
        }
    }
}


//
//  Methods to populate the vertex-face relation of the child Level:
//      - child vertices originate from parent faces, edges and vertices
//      - sparse refinement poses challenges with allocation here:
//          - we need to update the counts/offsets as we populate
//          - note this imposes ordering constraints and inhibits concurrency
//
void
QuadRefinement::populateVertexFaceRelation() {

    const Level& parent = *_parent;
          Level& child  = *_child;

    //
    //  Notes on allocating/initializing the vertex-face counts/offsets vector:
    //
    //  Be aware of scheme-specific decisions here, e.g.:
    //      - no verts from parent faces for Loop (unless N-gons supported)
    //      - more interior edges and faces for verts from parent edges for Loop
    //      - no guaranteed "neighborhood" around Bilinear verts from verts
    //
    //  If uniform subdivision, vert-face count will be (catmark or loop):
    //      - 4 or 0 for verts from parent faces (for catmark)
    //      - 2x or 3x number in parent edge for verts from parent edges
    //      - same as parent vert for verts from parent verts
    //  If sparse subdivision, vert-face count will be:
    //      - the number of child faces in parent face
    //      - 1 or 2x number in parent edge for verts from parent edges
    //          - where the 1 or 2 is number of child edges of parent edge
    //      - same as parent vert for verts from parent verts (catmark)
    //
    int childVertFaceIndexSizeEstimate = (int)parent._faceVertIndices.size()
                                       + (int)parent._edgeFaceIndices.size() * 2
                                       + (int)parent._vertFaceIndices.size();

    child._vertFaceCountsAndOffsets.resize(child.getNumVertices() * 2);
    child._vertFaceIndices.resize(         childVertFaceIndexSizeEstimate);
    child._vertFaceLocalIndices.resize(    childVertFaceIndexSizeEstimate);

    if (getFirstChildVertexFromVertices() == 0) {
        populateVertexFacesFromParentVertices();
        populateVertexFacesFromParentFaces();
        populateVertexFacesFromParentEdges();
    } else {
        populateVertexFacesFromParentFaces();
        populateVertexFacesFromParentEdges();
        populateVertexFacesFromParentVertices();
    }

    //  Revise the over-allocated estimate based on what is used (as indicated in the
    //  count/offset for the last vertex) and trim the index vectors accordingly:
    childVertFaceIndexSizeEstimate = child.getNumVertexFaces(child.getNumVertices()-1) +
                                     child.getOffsetOfVertexFaces(child.getNumVertices()-1);
    child._vertFaceIndices.resize(     childVertFaceIndexSizeEstimate);
    child._vertFaceLocalIndices.resize(childVertFaceIndexSizeEstimate);
}

void
QuadRefinement::populateVertexFacesFromParentFaces() {

    const Level& parent = *this->_parent;
          Level& child  = *this->_child;

    for (int fIndex = 0; fIndex < parent.getNumFaces(); ++fIndex) {
        int cVertIndex = this->_faceChildVertIndex[fIndex];
        if (!IndexIsValid(cVertIndex)) continue;

        //
        //  Inspect the parent face first:
        //
        int pFaceVertCount  = parent.getFaceVertices(fIndex).size();

        ConstIndexArray pFaceChildren = this->getFaceChildFaces(fIndex);

        //
        //  Reserve enough vert-faces, populate and trim to the actual size:
        //
        child.resizeVertexFaces(cVertIndex, pFaceVertCount);

        IndexArray      cVertFaces  = child.getVertexFaces(cVertIndex);
        LocalIndexArray cVertInFace = child.getVertexFaceLocalIndices(cVertIndex);

        int cVertFaceCount = 0;
        for (int j = 0; j < pFaceVertCount; ++j) {
            if (IndexIsValid(pFaceChildren[j])) {
                //  Note orientation wrt parent face -- quad vs non-quad...
                LocalIndex vertInFace =
                    (LocalIndex)((pFaceVertCount == 4) ? ((j+2) & 3) : 2);

                cVertFaces[cVertFaceCount]  = pFaceChildren[j];
                cVertInFace[cVertFaceCount] = vertInFace;
                cVertFaceCount++;
            }
        }
        child.trimVertexFaces(cVertIndex, cVertFaceCount);
    }
}

void
QuadRefinement::populateVertexFacesFromParentEdges() {

    const Level& parent = *this->_parent;
          Level& child  = *this->_child;

    for (int pEdgeIndex = 0; pEdgeIndex < parent.getNumEdges(); ++pEdgeIndex) {
        int cVertIndex = this->_edgeChildVertIndex[pEdgeIndex];
        if (!IndexIsValid(cVertIndex)) continue;

        //
        //  Inspect the parent edge first:
        //
        ConstIndexArray pEdgeFaces = parent.getEdgeFaces(pEdgeIndex);

        //
        //  Reserve enough vert-faces, populate and trim to the actual size:
        //
        child.resizeVertexFaces(cVertIndex, 2 * pEdgeFaces.size());

        IndexArray      cVertFaces  = child.getVertexFaces(cVertIndex);
        LocalIndexArray cVertInFace = child.getVertexFaceLocalIndices(cVertIndex);

        int cVertFaceCount = 0;
        for (int i = 0; i < pEdgeFaces.size(); ++i) {
            //
            //  EDGE_IN_FACE:
            //      Identify the parent edge within this parent face -- this is where
            //  augmenting the edge-face relation with the "child index" is useful:
            //
            Index pFaceIndex  = pEdgeFaces[i];

            ConstIndexArray pFaceEdges = parent.getFaceEdges(pFaceIndex),
                            pFaceChildren = this->getFaceChildFaces(pFaceIndex);

            //
            //  Identify the corresponding two child faces for this parent face and
            //  assign those of the two that are valid:
            //
            int pFaceEdgeCount = pFaceEdges.size();

            int faceChild0 = 0;
            for ( ; pFaceEdges[faceChild0] != pEdgeIndex; ++faceChild0) ;

            int faceChild1 = faceChild0 + 1;
            if (faceChild1 == pFaceEdgeCount) faceChild1 = 0;

            //  For counter-clockwise ordering of faces, consider the second face first:
            //
            //  Note orientation wrt incident parent faces -- quad vs non-quad...
            if (IndexIsValid(pFaceChildren[faceChild1])) {
                cVertFaces[cVertFaceCount] = pFaceChildren[faceChild1];
                cVertInFace[cVertFaceCount] = (LocalIndex)((pFaceEdgeCount == 4) ? faceChild0 : 3);
                cVertFaceCount++;
            }
            if (IndexIsValid(pFaceChildren[faceChild0])) {
                cVertFaces[cVertFaceCount] = pFaceChildren[faceChild0];
                cVertInFace[cVertFaceCount] = (LocalIndex)((pFaceEdgeCount == 4) ? faceChild1 : 1);
                cVertFaceCount++;
            }
        }
        child.trimVertexFaces(cVertIndex, cVertFaceCount);
    }
}

void
QuadRefinement::populateVertexFacesFromParentVertices() {

    const Level& parent = *this->_parent;
          Level& child  = *this->_child;

    for (int vIndex = 0; vIndex < parent.getNumVertices(); ++vIndex) {
        int cVertIndex = this->_vertChildVertIndex[vIndex];
        if (!IndexIsValid(cVertIndex)) continue;

        //
        //  Inspect the parent vert's faces:
        //
        ConstIndexArray      pVertFaces  = parent.getVertexFaces(vIndex);
        ConstLocalIndexArray pVertInFace = parent.getVertexFaceLocalIndices(vIndex);

        //
        //  Reserve enough vert-faces, populate and trim to the actual size:
        //
        child.resizeVertexFaces(cVertIndex, pVertFaces.size());

        IndexArray      cVertFaces  = child.getVertexFaces(cVertIndex);
        LocalIndexArray cVertInFace = child.getVertexFaceLocalIndices(cVertIndex);

        int cVertFaceCount = 0;
        for (int i = 0; i < pVertFaces.size(); ++i) {
            Index      pFace      = pVertFaces[i];
            LocalIndex pFaceChild = pVertInFace[i];

            Index cFace = this->getFaceChildFaces(pFace)[pFaceChild];
            if (IndexIsValid(cFace)) {
                //  Note orientation wrt incident parent faces -- quad vs non-quad...
                int pFaceCount = parent.getFaceVertices(pFace).size();

                cVertFaces[cVertFaceCount] = cFace;
                cVertInFace[cVertFaceCount] = (LocalIndex)((pFaceCount == 4) ? pFaceChild : 0);
                cVertFaceCount++;
            }
        }
        child.trimVertexFaces(cVertIndex, cVertFaceCount);
    }
}

//
//  Methods to populate the vertex-edge relation of the child Level:
//      - child vertices originate from parent faces, edges and vertices
//      - sparse refinement poses challenges with allocation here:
//          - we need to update the counts/offsets as we populate
//          - note this imposes ordering constraints and inhibits concurrency
//
void
QuadRefinement::populateVertexEdgeRelation() {

    const Level& parent = *_parent;
          Level& child  = *_child;

    //
    //  Notes on allocating/initializing the vertex-edge counts/offsets vector:
    //
    //  Be aware of scheme-specific decisions here, e.g.:
    //      - no verts from parent faces for Loop
    //      - more interior edges and faces for verts from parent edges for Loop
    //      - no guaranteed "neighborhood" around Bilinear verts from verts
    //
    //  If uniform subdivision, vert-edge count will be:
    //      - 4 or 0 for verts from parent faces (for catmark)
    //      - 2 + N or 2 + 2*N faces incident parent edge for verts from parent edges
    //      - same as parent vert for verts from parent verts
    //  If sparse subdivision, vert-edge count will be:
    //      - non-trivial function of child faces in parent face
    //          - 1 child face will always result in 2 child edges
    //          * 2 child faces can mean 3 or 4 child edges
    //          - 3 child faces will always result in 4 child edges
    //      - 1 or 2 + N faces incident parent edge for verts from parent edges
    //          - where the 1 or 2 is number of child edges of parent edge
    //          - any end vertex will require all N child faces (catmark)
    //      - same as parent vert for verts from parent verts (catmark)
    //
    int childVertEdgeIndexSizeEstimate = (int)parent._faceVertIndices.size()
                                       + (int)parent._edgeFaceIndices.size() + parent.getNumEdges() * 2
                                       + (int)parent._vertEdgeIndices.size();

    child._vertEdgeCountsAndOffsets.resize(child.getNumVertices() * 2);
    child._vertEdgeIndices.resize(         childVertEdgeIndexSizeEstimate);
    child._vertEdgeLocalIndices.resize(    childVertEdgeIndexSizeEstimate);

    if (getFirstChildVertexFromVertices() == 0) {
        populateVertexEdgesFromParentVertices();
        populateVertexEdgesFromParentFaces();
        populateVertexEdgesFromParentEdges();
    } else {
        populateVertexEdgesFromParentFaces();
        populateVertexEdgesFromParentEdges();
        populateVertexEdgesFromParentVertices();
    }

    //  Revise the over-allocated estimate based on what is used (as indicated in the
    //  count/offset for the last vertex) and trim the index vectors accordingly:
    childVertEdgeIndexSizeEstimate = child.getNumVertexEdges(child.getNumVertices()-1) +
                                     child.getOffsetOfVertexEdges(child.getNumVertices()-1);
    child._vertEdgeIndices.resize(     childVertEdgeIndexSizeEstimate);
    child._vertEdgeLocalIndices.resize(childVertEdgeIndexSizeEstimate);
}

void
QuadRefinement::populateVertexEdgesFromParentFaces() {

    const Level& parent = *this->_parent;
          Level& child  = *this->_child;

    for (int fIndex = 0; fIndex < parent.getNumFaces(); ++fIndex) {
        int cVertIndex = this->_faceChildVertIndex[fIndex];
        if (!IndexIsValid(cVertIndex)) continue;

        //
        //  Inspect the parent face first:
        //
        ConstIndexArray pFaceVerts = parent.getFaceVertices(fIndex),
                        pFaceChildEdges = this->getFaceChildEdges(fIndex);

        //
        //  Reserve enough vert-edges, populate and trim to the actual size:
        //
        child.resizeVertexEdges(cVertIndex, pFaceVerts.size());

        IndexArray      cVertEdges  = child.getVertexEdges(cVertIndex);
        LocalIndexArray cVertInEdge = child.getVertexEdgeLocalIndices(cVertIndex);

        //
        //  Need to ensure correct ordering here when complete -- we want the "leading"
        //  edge of each child face first.  The child vert is in the center of a new
        //  face to boundaries only occur when incomplete...
        //
        int cVertEdgeCount = 0;
        for (int j = 0; j < pFaceVerts.size(); ++j) {
            int jLeadingEdge = j ? (j - 1) : (pFaceVerts.size() - 1);
            if (IndexIsValid(pFaceChildEdges[jLeadingEdge])) {
                cVertEdges[cVertEdgeCount] = pFaceChildEdges[jLeadingEdge];
                cVertInEdge[cVertEdgeCount] = 0;
                cVertEdgeCount++;
            }
        }
        child.trimVertexEdges(cVertIndex, cVertEdgeCount);
    }
}
void
QuadRefinement::populateVertexEdgesFromParentEdges() {

    const Level& parent = *this->_parent;
          Level& child  = *this->_child;

    for (int eIndex = 0; eIndex < parent.getNumEdges(); ++eIndex) {
        int cVertIndex = this->_edgeChildVertIndex[eIndex];
        if (!IndexIsValid(cVertIndex)) continue;

        //
        //  First inspect the parent edge -- its parent faces then its child edges:
        //
        ConstIndexArray pEdgeFaces      = parent.getEdgeFaces(eIndex),
                        pEdgeChildEdges = this->getEdgeChildEdges(eIndex);

        //
        //  Reserve enough vert-edges, populate and trim to the actual size:
        //
        child.resizeVertexEdges(cVertIndex, pEdgeFaces.size() + 2);

        IndexArray      cVertEdges  = child.getVertexEdges(cVertIndex);
        LocalIndexArray cVertInEdge = child.getVertexEdgeLocalIndices(cVertIndex);

        //
        //  We need to order the incident edges around the vertex appropriately:
        //      - one child edge of the parent edge ("leading" in face 0)
        //      - child edge of face 0
        //      - the other child edge of the parent edge ("trailing" in face 0)
        //      - child edges of all remaining faces
        //  This is a bit awkward with the current implmentation -- given the way
        //  the child edge of a face is indentified.  Until we clean it up, deal
        //  with the two child edges of the parent edge first followed by all faces
        //  then swap the second child of the parent with the child of the first
        //  face.
        //
        //  Also be careful to place the child edges of the parent edge correctly.
        //  As edges are not directed their orientation may vary.
        //
        int cVertEdgeCount = 0;

        if (IndexIsValid(pEdgeChildEdges[0])) {
            cVertEdges[cVertEdgeCount] = pEdgeChildEdges[0];
            cVertInEdge[cVertEdgeCount] = 0;
            cVertEdgeCount++;
        }
        if (IndexIsValid(pEdgeChildEdges[1])) {
            cVertEdges[cVertEdgeCount] = pEdgeChildEdges[1];
            cVertInEdge[cVertEdgeCount] = 0;
            cVertEdgeCount++;
        }

        bool swapChildEdgesOfParent    = false;
        bool swapChildEdgeAndFace0Edge = false;
        for (int i = 0; i < pEdgeFaces.size(); ++i) {
            Index pFace = pEdgeFaces[i];

            ConstIndexArray pFaceEdges      = parent.getFaceEdges(pFace),
                            pFaceChildEdges = this->getFaceChildEdges(pFace);

            //
            //  EDGE_IN_FACE:
            //      Identify the parent edge within this parent face -- this is where
            //  augmenting the edge-face relation with the "local index" is useful:
            //
            int edgeInFace = 0;
            for ( ; pFaceEdges[edgeInFace] != eIndex; ++edgeInFace) ;

            if ((i == 0) && (cVertEdgeCount == 2)) {
                swapChildEdgeAndFace0Edge = IndexIsValid(pFaceChildEdges[edgeInFace]);
                if (swapChildEdgeAndFace0Edge) {
                    swapChildEdgesOfParent = (parent.getFaceVertices(pFace)[edgeInFace] ==
                                              parent.getEdgeVertices(eIndex)[0]);
                }
            }

            if (IndexIsValid(pFaceChildEdges[edgeInFace])) {
                cVertEdges[cVertEdgeCount] = pFaceChildEdges[edgeInFace];
                cVertInEdge[cVertEdgeCount] = 1;
                cVertEdgeCount++;
            }
        }

        //  Now swap the child edges of the parent as needed:
        if (swapChildEdgeAndFace0Edge) {
            if (swapChildEdgesOfParent) {
                std::swap(cVertEdges[0],  cVertEdges[1]);
                //  both local indices 0 -- no need to swap
            }
            std::swap(cVertEdges[1],  cVertEdges[2]);
            std::swap(cVertInEdge[1], cVertInEdge[2]);
        }

        child.trimVertexEdges(cVertIndex, cVertEdgeCount);
    }
}
void
QuadRefinement::populateVertexEdgesFromParentVertices() {

    const Level& parent = *this->_parent;
          Level& child  = *this->_child;

    for (int vIndex = 0; vIndex < parent.getNumVertices(); ++vIndex) {
        int cVertIndex = this->_vertChildVertIndex[vIndex];
        if (!IndexIsValid(cVertIndex)) continue;

        //
        //  Inspect the parent vert's edges first:
        //
        ConstIndexArray      pVertEdges  = parent.getVertexEdges(vIndex);
        ConstLocalIndexArray pVertInEdge = parent.getVertexEdgeLocalIndices(vIndex);

        //
        //  Reserve enough vert-edges, populate and trim to the actual size:
        //
        child.resizeVertexEdges(cVertIndex, pVertEdges.size());

        IndexArray      cVertEdges  = child.getVertexEdges(cVertIndex);
        LocalIndexArray cVertInEdge = child.getVertexEdgeLocalIndices(cVertIndex);

        int cVertEdgeCount = 0;
        for (int i = 0; i < pVertEdges.size(); ++i) {
            Index      pEdgeIndex  = pVertEdges[i];
            LocalIndex pEdgeVert = pVertInEdge[i];

            Index pEdgeChildIndex = this->getEdgeChildEdges(pEdgeIndex)[pEdgeVert];
            if (IndexIsValid(pEdgeChildIndex)) {
                cVertEdges[cVertEdgeCount] = pEdgeChildIndex;
                cVertInEdge[cVertEdgeCount] = 1;
                cVertEdgeCount++;
            }
        }
        child.trimVertexEdges(cVertIndex, cVertEdgeCount);
    }
}

//
//  Methods to populate child-component indices for sparse selection:
//
//  Need to find a better place for these anon helper methods now that they are required
//  both in the base class and the two subclasses for quad- and tri-splitting...
//
namespace {
    Index const IndexSparseMaskNeighboring = (1 << 0);
    Index const IndexSparseMaskSelected    = (1 << 1);

    inline void markSparseIndexNeighbor(Index& index) { index = IndexSparseMaskNeighboring; }
    inline void markSparseIndexSelected(Index& index) { index = IndexSparseMaskSelected; }
}

void
QuadRefinement::markSparseFaceChildren() {

    assert(_parentFaceTag.size() > 0);

    //
    //  For each parent face:
    //      All boundary edges will be adequately marked as a result of the pass over the
    //  edges above and boundary vertices marked by selection.  So all that remains is to
    //  identify the child faces and interior child edges for a face requiring neighboring
    //  child faces.
    //      For each corner vertex selected, we need to mark the corresponding child face,
    //  the two interior child edges and shared child vertex in the middle.
    //
    assert(_splitType == Sdc::SPLIT_TO_QUADS);

    for (Index pFace = 0; pFace < parent().getNumFaces(); ++pFace) {
        //
        //  Mark all descending child components of a selected face.  Otherwise inspect
        //  its incident vertices to see if anything neighboring has been selected --
        //  requiring partial refinement of this face.
        //
        //  Remember that a selected face cannot be transitional, and that only a
        //  transitional face will be partially refined.
        //
        IndexArray fChildFaces = getFaceChildFaces(pFace);
        IndexArray fChildEdges = getFaceChildEdges(pFace);

        ConstIndexArray fVerts = parent().getFaceVertices(pFace);

        SparseTag& pFaceTag = _parentFaceTag[pFace];

        if (pFaceTag._selected) {
            for (int i = 0; i < fVerts.size(); ++i) {
                markSparseIndexSelected(fChildFaces[i]);
                markSparseIndexSelected(fChildEdges[i]);
            }
            markSparseIndexSelected(_faceChildVertIndex[pFace]);

            pFaceTag._transitional = 0;
        } else {
            int marked = false;

            for (int i = 0; i < fVerts.size(); ++i) {
                //  NOTE - the mod 4 here will not work for N-gons (and want to avoid % anyway)
                int iPrev = (i+3) % 4;

                if (_parentVertexTag[fVerts[i]]._selected) {
                    markSparseIndexNeighbor(fChildFaces[i]);

                    markSparseIndexNeighbor(fChildEdges[i]);
                    markSparseIndexNeighbor(fChildEdges[iPrev]);

                    marked = true;
                }
            }
            if (marked) {
                markSparseIndexNeighbor(_faceChildVertIndex[pFace]);

                //
                //  Assign selection and transitional tags to faces when required:
                //
                //  Only non-selected faces may be "transitional", and we need to inspect
                //  all tags on its boundary edges to be sure.  Since we're inspecting each
                //  now (and may need to later) retain the transitional state of each in a
                //  4-bit mask that reflects the full transitional topology for later.
                //
                ConstIndexArray fEdges = parent().getFaceEdges(pFace);
                if (fEdges.size() == 4) {
                    pFaceTag._transitional = (unsigned char)
                           ((_parentEdgeTag[fEdges[0]]._transitional << 0) |
                            (_parentEdgeTag[fEdges[1]]._transitional << 1) |
                            (_parentEdgeTag[fEdges[2]]._transitional << 2) |
                            (_parentEdgeTag[fEdges[3]]._transitional << 3));
                } else if (fEdges.size() == 3) {
                    pFaceTag._transitional = (unsigned char)
                           ((_parentEdgeTag[fEdges[0]]._transitional << 0) |
                            (_parentEdgeTag[fEdges[1]]._transitional << 1) |
                            (_parentEdgeTag[fEdges[2]]._transitional << 2));
                } else {
                    pFaceTag._transitional = 0;
                    for (int i = 0; i < fEdges.size(); ++i) {
                        pFaceTag._transitional |= _parentEdgeTag[fEdges[i]]._transitional;
                    }
                }
            }
        }
    }
}

} // end namespace Vtr

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv
