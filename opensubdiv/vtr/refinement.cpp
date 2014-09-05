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
#include "../sdc/catmarkScheme.h"
#include "../sdc/bilinearScheme.h"
#include "../vtr/types.h"
#include "../vtr/level.h"
#include "../vtr/refinement.h"
#include "../vtr/fvarLevel.h"
#include "../vtr/fvarRefinement.h"
#include "../vtr/maskInterfaces.h"

#include <cassert>
#include <cstdio>


//
//  NOTES on short-cuts and assumptions...
//      This was developed to prototype some ideas, and some issues were simplified
//  (or just plain ignored) in order to validate the more general cases first.  Use
//  the following list of keywords to search for notes in comments where these issues
//  are discussed:
//
//  ORDERING:
//      - shortcuts that may not order incident components as ultimately desired
//          - typically the vert-face and/or vert-edge relations
//          - vert-face much more critical than vert-edge (which is debatable)
//      - these won't affect refinement but will affect patch construction
//      - failure to address will lead to topology validation failures once ordering
//        has been added to validation
//
namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Vtr {

//
//  Simple constructor and destructor -- consider inline if they remain simple...
//
Refinement::Refinement() :
    _parent(0),
    _child(0),
    _schemeType(Sdc::TYPE_CATMARK),
    _schemeOptions(),
    _quadSplit(true),
    _childFaceFromFaceCount(0),
    _childEdgeFromFaceCount(0),
    _childEdgeFromEdgeCount(0),
    _childVertFromFaceCount(0),
    _childVertFromEdgeCount(0),
    _childVertFromVertCount(0) {
}

Refinement::~Refinement() {

    for (int i = 0; i < (int)_fvarChannels.size(); ++i) {
        delete _fvarChannels[i];
    }
}


void
Refinement::initialize(Level& parent, Level& child) {

    //  Make sure we are getting a fresh child...
    assert((child.getDepth() == 0) && (child.getNumVertices() == 0));

    //
    //  Do we want anything more here for "initialization", e.g. the subd scheme,
    //  options, etc. -- or will those be specified/assigned on refinement?
    //
    _parent = &parent;
    _child  = &child;

    child._depth = 1 + parent.getDepth();
}

void
Refinement::setScheme(Sdc::Type const& schemeType, Sdc::Options const& schemeOptions) {

    _schemeType = schemeType;
    _schemeOptions = schemeOptions;
}


//
//  Methods for preparing for refinement:
//
void
Refinement::allocateParentToChildMapping() {

    //
    //  Initialize the vectors of indices mapping parent components to those child components
    //  that will originate from each.
    //
    //  There is some history here regarding whether to initialize all entries or not, and if
    //  so, what default value to use (marking for sparse refinement):
    //
    int faceChildFaceCount;
    int faceChildEdgeCount;
    int edgeChildEdgeCount;

    int faceChildVertCount;
    int edgeChildVertCount;
    int vertChildVertCount;

    if (_quadSplit) {
        faceChildFaceCount = (int) _parent->_faceVertIndices.size();
        faceChildEdgeCount = (int) _parent->_faceEdgeIndices.size();
        edgeChildEdgeCount = (int) _parent->_edgeVertIndices.size();

        faceChildVertCount = _parent->getNumFaces();
        edgeChildVertCount = _parent->getNumEdges();
        vertChildVertCount = _parent->getNumVertices();
    } else {
        assert("Non-quad splitting not yet supported\n" == 0);

        faceChildFaceCount = _parent->getNumFaces() * 4;
        faceChildEdgeCount = (int) _parent->_faceEdgeIndices.size();
        edgeChildEdgeCount = (int) _parent->_edgeVertIndices.size();

        faceChildVertCount = 0;
        edgeChildVertCount = _parent->getNumEdges();
        vertChildVertCount = _parent->getNumVertices();
    }

    //
    //  Given we will be ignoring initial values with uniform refinement and assigning all
    //  directly, initializing here is a waste...
    //
    Index initValue = 0;

    _faceChildFaceIndices.resize(faceChildFaceCount, initValue);
    _faceChildEdgeIndices.resize(faceChildEdgeCount, initValue);
    _edgeChildEdgeIndices.resize(edgeChildEdgeCount, initValue);

    _faceChildVertIndex.resize(faceChildVertCount, initValue);
    _edgeChildVertIndex.resize(edgeChildVertCount, initValue);
    _vertChildVertIndex.resize(vertChildVertCount, initValue);
}

void
Refinement::printParentToChildMapping() const {

    printf("Parent-to-child component mapping:\n");
    for (Index pFace = 0; pFace < _parent->getNumFaces(); ++pFace) {
        printf("  Face %d:\n", pFace);
        printf("    Child vert:  %d\n", _faceChildVertIndex[pFace]);

        printf("    Child faces: ");
        IndexArray const childFaces = getFaceChildFaces(pFace);
        for (int i = 0; i < childFaces.size(); ++i) {
            printf(" %d", childFaces[i]);
        }
        printf("\n");

        printf("    Child edges: ");
        IndexArray const childEdges = getFaceChildEdges(pFace);
        for (int i = 0; i < childEdges.size(); ++i) {
            printf(" %d", childEdges[i]);
        }
        printf("\n");
    }
    for (Index pEdge = 0; pEdge < _parent->getNumEdges(); ++pEdge) {
        printf("  Edge %d:\n", pEdge);
        printf("    Child vert:  %d\n", _edgeChildVertIndex[pEdge]);

        IndexArray const childEdges = getEdgeChildEdges(pEdge);
        printf("    Child edges: %d %d\n", childEdges[0], childEdges[1]);
    }
    for (Index pVert = 0; pVert < _parent->getNumVertices(); ++pVert) {
        printf("  Vert %d:\n", pVert);
        printf("    Child vert:  %d\n", _vertChildVertIndex[pVert]);
    }
}


namespace {
    inline bool isSparseIndexMarked(Index index)   { return index != 0; }

    inline int
    sequenceSparseIndexVector(IndexVector& indexVector, int baseValue = 0) {
        int validCount = 0;
        for (int i = 0; i < (int) indexVector.size(); ++i) {
            indexVector[i] = isSparseIndexMarked(indexVector[i])
                           ? (baseValue + validCount++) : INDEX_INVALID;
        }
        return validCount;
    }

    inline int
    sequenceFullIndexVector(IndexVector& indexVector, int baseValue = 0) {
        int indexCount = (int) indexVector.size();
        for (int i = 0; i < indexCount; ++i) {
            indexVector[i] = baseValue++;
        }
        return indexCount;
    }
}

void
Refinement::initializeSparseSelectionTags() {

    _parentFaceTag.resize(_parent->getNumFaces());
    _parentEdgeTag.resize(_parent->getNumEdges());
    _parentVertexTag.resize(_parent->getNumVertices());
}

void
Refinement::populateParentToChildMapping() {

    allocateParentToChildMapping();

    if (_uniform) {
        //
        //  We should be able to replace this separate initialization and sequencing in one
        //  iteration -- we just need to separate the allocation and initialization that are
        //  currently combined in the following initialization method:
        //
        //  child faces:
        _childFaceFromFaceCount = sequenceFullIndexVector(_faceChildFaceIndices);

        //  child edges:
        _childEdgeFromFaceCount = sequenceFullIndexVector(_faceChildEdgeIndices);
        _childEdgeFromEdgeCount = sequenceFullIndexVector(_edgeChildEdgeIndices, _childEdgeFromFaceCount);

        //  child vertices:
        _childVertFromFaceCount = sequenceFullIndexVector(_faceChildVertIndex);
        _childVertFromEdgeCount = sequenceFullIndexVector(_edgeChildVertIndex, _childVertFromFaceCount);
        _childVertFromVertCount = sequenceFullIndexVector(_vertChildVertIndex, _childVertFromFaceCount +
                                                                               _childVertFromEdgeCount);
    } else {
        //
        //  Since the parent-to-child mapping is not sparse, we initialize it and then mark
        //  child components that are required from the parent components selected, or the
        //  neighborhoods around them to support their limit.
        //
        //  Make sure the selection was non-empty -- currently unsupported...
        if (_parentVertexTag.size() == 0) {
            assert("Unsupported empty sparse refinement detected in Refinement" == 0);
        }
        markSparseChildComponents();

        //
        //  Assign the sequences of child components by inspecting what was marked:
        //
        //  Note that for vertices (and edges) the sequence is assembled from three source vectors
        //  for vertices originating from faces, edges and vertices:
        //
        //  child faces:
        _childFaceFromFaceCount = sequenceSparseIndexVector(_faceChildFaceIndices);

        //  child edges:
        _childEdgeFromFaceCount = sequenceSparseIndexVector(_faceChildEdgeIndices);
        _childEdgeFromEdgeCount = sequenceSparseIndexVector(_edgeChildEdgeIndices, _childEdgeFromFaceCount);

        //  child vertices:
        _childVertFromFaceCount = sequenceSparseIndexVector(_faceChildVertIndex);
        _childVertFromEdgeCount = sequenceSparseIndexVector(_edgeChildVertIndex, _childVertFromFaceCount);
        _childVertFromVertCount = sequenceSparseIndexVector(_vertChildVertIndex, _childVertFromFaceCount +
                                                                                 _childVertFromEdgeCount);
    }
}

void
Refinement::createChildComponents() {

    //
    //  Assign the child's component counts/inventory based on the child components identified:
    //
    _child->_faceCount = _childFaceFromFaceCount;
    _child->_edgeCount = _childEdgeFromFaceCount + _childEdgeFromEdgeCount;
    _child->_vertCount = _childVertFromFaceCount + _childVertFromEdgeCount + _childVertFromVertCount;
}


//
//  Before we can refine the topological relations, we want to have the vectors
//  sized appropriately so that we can (potentially) distribute the computation
//  over chunks of these vectors.
//
//  This is non-trivial in the case of sparse subdivision, but there are still
//  some opportunities for concurrency within these...
//
//  We will know the maximal size of each relation based on the origin of the
//  child component -- so we could over-allocate.  If the refinement is very
//  sparse the greatest savings will come from the small number of components
//  and the overhead in the relations of each should be miminal (there should
//  be a mix between maximal and minimal size given the mix of both interior
//  and exterior components when sparse).
//
//  Is it worth populating the counts concurrently, then making a single
//  pass through them to assemble the offsets?
//
void
Refinement::initializeFaceVertexCountsAndOffsets() {

    Level& child = *_child;

    //
    //  Be aware of scheme-specific decisions here, e.g. the current use
    //  of 4 for quads for Catmark -- must adjust for Loop, Bilinear and
    //  account for possibility of both quads and tris...
    //
    child._faceVertCountsAndOffsets.resize(child.getNumFaces() * 2);
    for (int i = 0; i < child.getNumFaces(); ++i) {
        child._faceVertCountsAndOffsets[i*2 + 0] = 4;
        child._faceVertCountsAndOffsets[i*2 + 1] = i << 2;
    }
}
void
Refinement::initializeEdgeFaceCountsAndOffsets() {

    //
    //  Be aware of scheme-specific decisions here, e.g.:
    //      - inspection of sparse child faces for edges from faces
    //      - no guaranteed "neighborhood" around Bilinear verts from verts
    //
    //  If uniform subdivision, face count of a child edge will be:
    //      - 2 for edges from parent faces
    //      - same as parent edge for edges from parent edges
    //  If sparse subdivision, face count of a child edge will be:
    //      - 1 or 2 depending on child faces in parent face
    //          - requires inspection if not all child faces present
    //      ? same as parent edge for edges from parent edges
    //          - given end vertex must have its full set of child faces
    //          - not for Bilinear -- only if neighborhood is non-zero
    //
}
void
Refinement::initializeVertexFaceCountsAndOffsets() {

    //
    //  Be aware of scheme-specific decisions here, e.g.:
    //      - no verts from parent faces for Loop
    //      - more interior edges and faces for verts from parent edges for Loop
    //      - no guaranteed "neighborhood" around Bilinear verts from verts
    //
    //  If uniform subdivision, vert-face count will be:
    //      - 4 for verts from parent faces (for catmark)
    //      - 2x number in parent edge for verts from parent edges
    //      - same as parent vert for verts from parent verts
    //  If sparse subdivision, vert-face count will be:
    //      - the number of child faces in parent face
    //      - 1 or 2x number in parent edge for verts from parent edges
    //          - where the 1 or 2 is number of child edges of parent edge
    //      - same as parent vert for verts from parent verts (catmark)
    //
}
void
Refinement::initializeVertexEdgeCountsAndOffsets() {

    //
    //  Be aware of scheme-specific decisions here, e.g.:
    //      - no verts from parent faces for Loop
    //      - more interior edges and faces for verts from parent edges for Loop
    //      - no guaranteed "neighborhood" around Bilinear verts from verts
    //
    //  If uniform subdivision, vert-edge count will be:
    //      - 4 for verts from parent faces (for catmark)
    //      - 2 + N faces incident parent edge for verts from parent edges
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
}

//
//  Face-vert and face-edge topology propagation -- faces only originate from faces:
//
void
Refinement::populateFaceVerticesFromParentFaces() {

    //
    //  Algorithm:
    //    - iterate through parent face-child-face vector (could use back-vector)
    //    - use parent components incident the parent face:
    //        - use the interior face-vert, corner vert-vert and two edge-verts
    //
    for (Index pFace = 0; pFace < _parent->getNumFaces(); ++pFace) {
        IndexArray const pFaceVerts = _parent->getFaceVertices(pFace);
        IndexArray const pFaceEdges = _parent->getFaceEdges(pFace);

        IndexArray const pFaceChildren = getFaceChildFaces(pFace);

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

void
Refinement::populateFaceEdgesFromParentFaces() {

    //
    //  Algorithm:
    //    - iterate through parent face-child-face vector (could use back-vector)
    //    - use parent components incident the parent face:
    //        - use the two interior face-edges and the two boundary edge-edges
    //
    for (Index pFace = 0; pFace < _parent->getNumFaces(); ++pFace) {
        IndexArray const pFaceVerts = _parent->getFaceVertices(pFace);
        IndexArray const pFaceEdges = _parent->getFaceEdges(pFace);

        IndexArray const pFaceChildFaces = getFaceChildFaces(pFace);
        IndexArray const pFaceChildEdges = getFaceChildEdges(pFace);

        int pFaceVertCount = pFaceVerts.size();

        for (int j = 0; j < pFaceVertCount; ++j) {
            Index cFace = pFaceChildFaces[j];
            if (IndexIsValid(cFace)) {
                IndexArray cFaceEdges = _child->getFaceEdges(cFace);

                int jPrev = j ? (j - 1) : (pFaceVertCount - 1);

                //
                //  We have two edges thate are children of parent edges, and two child
                //  edges perpendicular to these from the interior of the parent face:
                //
                //  Identifying the former should be simpler -- after identifying the two
                //  parent edges, we have to identify which child-edge corresponds to this
                //  vertex.  This may be ambiguous with a degenerate edge (DEGEN) if tested
                //  this way, and may warrant higher level inspection of the parent face...
                //
                Index pCornerVert = pFaceVerts[j];

                Index            pPrevEdge      = pFaceEdges[jPrev];
                IndexArray const pPrevEdgeVerts = _parent->getEdgeVertices(pPrevEdge);

                Index            pNextEdge      = pFaceEdges[j];
                IndexArray const pNextEdgeVerts = _parent->getEdgeVertices(pNextEdge);

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
//  Edge-vert topology propagation -- two functions for face or edge origin:
//
void
Refinement::populateEdgeVerticesFromParentFaces() {

    //
    //  For each parent face's edge-children:
    //    - identify parent face's vert-child (note it is shared by all)
    //    - identify parent edge perpendicular to face's child edge:
    //        - identify parent edge's vert-child
    //
    for (Index pFace = 0; pFace < _parent->getNumFaces(); ++pFace) {
        IndexArray const pFaceEdges = _parent->getFaceEdges(pFace);

        IndexArray const pFaceChildren = getFaceChildEdges(pFace);

        for (int j = 0; j < pFaceEdges.size(); ++j) {
            Index cEdge = pFaceChildren[j];
            if (IndexIsValid(cEdge)) {
                IndexArray cEdgeVerts = _child->getEdgeVertices(cEdge);

                cEdgeVerts[0] = _faceChildVertIndex[pFace];
                cEdgeVerts[1] = _edgeChildVertIndex[pFaceEdges[j]];
            }
        }
    }
}

void
Refinement::populateEdgeVerticesFromParentEdges() {

    //
    //  For each parent edge's edge-children:
    //    - identify parent edge's vert-child (potentially shared by both)
    //    - identify parent vert at end of child edge:
    //        - identify parent vert's vert-child
    //
    for (Index pEdge = 0; pEdge < _parent->getNumEdges(); ++pEdge) {
        IndexArray const pEdgeVerts = _parent->getEdgeVertices(pEdge);

        IndexArray const pEdgeChildren = getEdgeChildEdges(pEdge);

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
//  Edge-face topology propagation -- two functions for face or edge origin:
//
void
Refinement::populateEdgeFacesFromParentFaces() {

    //
    //  Note -- the edge-face counts/offsets vector is not known
    //  ahead of time and is populated incrementally, so we cannot
    //  thread this yet...
    //
    for (Index pFace = 0; pFace < _parent->getNumFaces(); ++pFace) {
        IndexArray const pFaceChildFaces = getFaceChildFaces(pFace);
        IndexArray const pFaceChildEdges = getFaceChildEdges(pFace);

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
Refinement::populateEdgeFacesFromParentEdges() {

    //
    //  Note -- the edge-face counts/offsets vector is not known
    //  ahead of time and is populated incrementally, so we cannot
    //  thread this yet...
    //
    for (Index pEdge = 0; pEdge < _parent->getNumEdges(); ++pEdge) {
        IndexArray const pEdgeVerts = _parent->getEdgeVertices(pEdge);
        IndexArray const pEdgeFaces = _parent->getEdgeFaces(pEdge);

        IndexArray const pEdgeChildEdges = getEdgeChildEdges(pEdge);

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

                IndexArray const pFaceEdges = _parent->getFaceEdges(pFace);
                IndexArray const pFaceVerts = _parent->getFaceVertices(pFace);

                IndexArray const pFaceChildren = getFaceChildFaces(pFace);

                int pFaceValence = pFaceVerts.size();

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
//  Vert-face topology propagation -- three functions for face, edge or vert origin:
//
//  Remember for these that the corresponding counts/offsets for each component are
//  not yet known and so are also populated by these functions.  This does impose an
//  ordering requirement here and inhibits concurrency.
//
void
Refinement::populateVertexFacesFromParentFaces() {

    const Level& parent = *this->_parent;
          Level& child  = *this->_child;

    for (int fIndex = 0; fIndex < parent.getNumFaces(); ++fIndex) {
        int cVertIndex = this->_faceChildVertIndex[fIndex];
        if (!IndexIsValid(cVertIndex)) continue;

        //
        //  Inspect the parent face first:
        //
        int pFaceVertCount  = parent.getFaceVertices(fIndex).size();

        IndexArray const pFaceChildren = this->getFaceChildFaces(fIndex);

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
Refinement::populateVertexFacesFromParentEdges() {

    const Level& parent = *this->_parent;
          Level& child  = *this->_child;

    for (int pEdgeIndex = 0; pEdgeIndex < parent.getNumEdges(); ++pEdgeIndex) {
        int cVertIndex = this->_edgeChildVertIndex[pEdgeIndex];
        if (!IndexIsValid(cVertIndex)) continue;

        //
        //  Inspect the parent edge first:
        //
        IndexArray const pEdgeFaces = parent.getEdgeFaces(pEdgeIndex);

        //
        //  Reserve enough vert-faces, populate and trim to the actual size:
        //
        child.resizeVertexFaces(cVertIndex, 2 * pEdgeFaces.size());

        IndexArray      cVertFaces  = child.getVertexFaces(cVertIndex);
        LocalIndexArray cVertInFace = child.getVertexFaceLocalIndices(cVertIndex);

        int cVertFaceCount = 0;
        for (int i = 0; i < pEdgeFaces.size(); ++i) {
            //
            //  identify the parent edge within this parent face -- this is where
            //  augmenting the edge-face relation with the "child index" is useful:
            //
            Index pFaceIndex  = pEdgeFaces[i];

            IndexArray const pFaceEdges = parent.getFaceEdges(pFaceIndex);

            IndexArray const pFaceChildren = this->getFaceChildFaces(pFaceIndex);

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
Refinement::populateVertexFacesFromParentVertices() {

    const Level& parent = *this->_parent;
          Level& child  = *this->_child;

    for (int vIndex = 0; vIndex < parent.getNumVertices(); ++vIndex) {
        int cVertIndex = this->_vertChildVertIndex[vIndex];
        if (!IndexIsValid(cVertIndex)) continue;

        //
        //  Inspect the parent vert's faces:
        //
        IndexArray const      pVertFaces  = parent.getVertexFaces(vIndex);
        LocalIndexArray const pVertInFace = parent.getVertexFaceLocalIndices(vIndex);

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
//  Vert-edge topology propagation -- three functions for face, edge or vert origin:
//
void
Refinement::populateVertexEdgesFromParentFaces() {

    const Level& parent = *this->_parent;
          Level& child  = *this->_child;

    for (int fIndex = 0; fIndex < parent.getNumFaces(); ++fIndex) {
        int cVertIndex = this->_faceChildVertIndex[fIndex];
        if (!IndexIsValid(cVertIndex)) continue;

        //
        //  Inspect the parent face first:
        //
        IndexArray const pFaceVerts = parent.getFaceVertices(fIndex);

        IndexArray const pFaceChildren = this->getFaceChildEdges(fIndex);

        //
        //  Reserve enough vert-edges, populate and trim to the actual size:
        //
        child.resizeVertexEdges(cVertIndex, pFaceVerts.size());

        IndexArray      cVertEdges  = child.getVertexEdges(cVertIndex);
        LocalIndexArray cVertInEdge = child.getVertexEdgeLocalIndices(cVertIndex);

        int cVertEdgeCount = 0;
        for (int j = 0; j < pFaceVerts.size(); ++j) {
            if (IndexIsValid(pFaceChildren[j])) {
                cVertEdges[cVertEdgeCount] = pFaceChildren[j];
                cVertInEdge[cVertEdgeCount] = 0;
                cVertEdgeCount++;
            }
        }
        child.trimVertexEdges(cVertIndex, cVertEdgeCount);
    }
}
void
Refinement::populateVertexEdgesFromParentEdges() {

    const Level& parent = *this->_parent;
          Level& child  = *this->_child;

    for (int eIndex = 0; eIndex < parent.getNumEdges(); ++eIndex) {
        int cVertIndex = this->_edgeChildVertIndex[eIndex];
        if (!IndexIsValid(cVertIndex)) continue;

        //
        //  First inspect the parent edge -- its parent faces then its child edges:
        //
        IndexArray const pEdgeFaces = parent.getEdgeFaces(eIndex);

        IndexArray const pEdgeChild = this->getEdgeChildEdges(eIndex);

        //
        //  Reserve enough vert-edges, populate and trim to the actual size:
        //
        child.resizeVertexEdges(cVertIndex, pEdgeFaces.size() + 2);

        IndexArray      cVertEdges  = child.getVertexEdges(cVertIndex);
        LocalIndexArray cVertInEdge = child.getVertexEdgeLocalIndices(cVertIndex);

        //
        //  For each face incident the parent edge -- assign the edge within
        //  that parent face (perpendicular to the parent edge) if valid:
        //
        int cVertEdgeCount = 0;

        for (int i = 0; i < pEdgeFaces.size(); ++i) {
            Index pFace = pEdgeFaces[i];

            IndexArray const pFaceEdges = parent.getFaceEdges(pFace);

            IndexArray const pFaceChildEdges = this->getFaceChildEdges(pFace);

            //
            //  Identify the parent edge within this parent face -- this is where
            //  augmenting the edge-face relation with the "child index" is useful:
            //
            int edgeInFace = 0;
            for ( ; pFaceEdges[edgeInFace] != eIndex; ++edgeInFace) ;

            if (IndexIsValid(pFaceChildEdges[edgeInFace])) {
                cVertEdges[cVertEdgeCount] = pFaceChildEdges[edgeInFace];
                cVertInEdge[cVertEdgeCount] = 1;
                cVertEdgeCount++;
            }
        }
        int edgeFromFaceCount = cVertEdgeCount;

        //
        //  For the two possible child edges of the parent edge -- assign if valid:
        //
        for (int i = 0; i < 2; ++i) {
            if (IndexIsValid(pEdgeChild[i])) {
                cVertEdges[cVertEdgeCount] = pEdgeChild[i];
                cVertInEdge[cVertEdgeCount] = 0;
                cVertEdgeCount++;
            }
        }
        int edgeFromEdgeCount = cVertEdgeCount - edgeFromFaceCount;

        //
        //  Note we have ignored the ORDERING of the edges here -- generating those
        //  perpendicular to the parent edge first and then its children.  A simple
        //  permutation of the results can produce a more desirable ordering, but we
        //  need a bit more information gathered above, e.g. (f0,f1,ex) is ambiguous
        //  as to whether it should be (f0,e0,f1) or (f1,e1,f0)...
        //
        //  Do we need to bother here?  Is ORDERING of vert-edges ever necessary?
        //
        if (cVertEdgeCount == 4) {
        } else if (edgeFromFaceCount == 2) {
        } else if (edgeFromEdgeCount == 2) {
        }

        child.trimVertexEdges(cVertIndex, cVertEdgeCount);
    }
}
void
Refinement::populateVertexEdgesFromParentVertices() {

    const Level& parent = *this->_parent;
          Level& child  = *this->_child;

    for (int vIndex = 0; vIndex < parent.getNumVertices(); ++vIndex) {
        int cVertIndex = this->_vertChildVertIndex[vIndex];
        if (!IndexIsValid(cVertIndex)) continue;

        //
        //  Inspect the parent vert's edges first:
        //
        IndexArray const      pVertEdges  = parent.getVertexEdges(vIndex);
        LocalIndexArray const pVertInEdge = parent.getVertexEdgeLocalIndices(vIndex);

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
//  The main refinement method...
//
//  Note the kinds of child components that are generated:
//    - faces generate child faces, edges and verts
//    - edges generate child edges and verts
//    - verts generate child verts only
//
//  The general approach here will be as follows:
//    - mark incident child components to be generated:
//        - assume all for now but need this for holes, adaptive, etc.
//        - unclear what strategy of child component marking is best:
//            - beginning with a set of parent faces, hole markers, etc.
//            - need to mark children of all parent faces, but also all/some
//              of the adjacent parent faces
//            - marking of child faces needs to also identify/mark child edges
//            - marking of child faces needs to also identify/mark child verts
//    - take inventory of what is marked to resize child vectors
//        ? new members to retain these counts for other purposes:
//            - mFaceChildFaceCount
//            - mFaceChildEdgeCount
//            - mFaceChildVertCount
//            - mEdgeChildEdgeCount
//            - mEdgeChildVertCount
//            - mVertChildVertCount
//        - child components the sum of the above:
//            - new face count = face-faces
//            - new edge count = face-edges + edge-edges
//            - new vert count = face-verts + edge-verts + vert-verts
//    - generate/identify child components between parent and child level:
//        - child vertices from all parent faces
//        - child vertices from all parent edges
//        - child vertices from all parent vertices
//        - child edges from all parent edges
//        - child edges from all parent faces
//        - child faces from all parent faces
//    - populate topology relations:
//        - iterate through each child of each parent component
//            - populate "uninherited" relations directly
//                - includes face-vert, edge-vert, face-edge
//                - these will always be completely defined
//            - populate "inherited" relations by mapping parent->child
//                - includes vert-face, vert-edge, edge-face
//                - assume one-to-one mapping with parent relation for now
//                    - will be subset of parent for holes, adaptive, etc.
//        - need full complement of parent-child relations
//    - subdivide sharpness values for edges or verts, if present
//        ? hier-edits for sharpness to be applied before/after?
//    ? re-classify/propagate vertex "masks"
//    ? compute mask weights for all child verts
//        - should mirror the parents topological relations
//    * interpolate -- now deferred externally
//
void
Refinement::refine(Options refineOptions) {

    assert(_parent && _child);

    _uniform = !refineOptions._sparse;

    //
    //  First determine the inventory of child components by populating the parent-to-child
    //  vectors and identifying the future child components as we take inventory:
    //
    populateParentToChildMapping();

    createChildComponents();

    //
    //  The child-to-parent mapping is used subsequently to iterate through components.
    //  Consider merging the tag propation with it as the required iteration through the
    //  parent components and conditional assigment could serve both purposes:
    //
    //  Note that some tags will need to be revised after subdividing sharpness values
    //
    populateChildToParentMapping();

    propagateComponentTags();

    //
    //  We can often suppress full topology generation in the last level -- the parent topology
    //  and subdivided sharpness values are enough to be able to interpolate vertex data.  How
    //  to best identify and specify what aspects should be refined is still unclear.
    //
    Relations relationsToPopulate;
    if (refineOptions._faceTopologyOnly) {
        relationsToPopulate.setAll(false);
        relationsToPopulate._faceVertices = true;
    } else {
        relationsToPopulate.setAll(true);
    }

    subdivideTopology(relationsToPopulate);

    //
    //  Subdividing sharpness values and classifying child vertices -- note the issue relating
    //  the subdivision of edge-sharpness and classification of the vertex of a parent edge:  a
    //  case can be made for classifying while computing sharpness values.
    //
    //  WORK IN PROGRESS:
    //      Considering modifying both of these to make use of the new component tags and the
    //  child-to-parent mapping.  In particular, if the edge or vertex is not semi-sharp, the
    //  value can just be copied from the parent.  Otherwise a newly computed semi-sharp edge
    //  of vertex sharpness will also need to be inspected and the "semisharp" tag adjusted
    //  accordingly.  And, in the case of the vertex, the "rule" needs to be recomputed for
    //  any "semisharp" vertex as any of its "semisharp" edges may have changed.
    //
    subdivideEdgeSharpness();
    subdivideVertexSharpness();

    //  Until reclassification of semisharp features is handled when subdivided, we will need
    //  to call a post-process to adjust the semisharp and rule tags:
    //
    reclassifySemisharpVertices();

    //
    //  Refine the topology for any face-varying channels:
    //
    bool refineOptions_faceVaryingChannels = true;

    if (refineOptions_faceVaryingChannels) {
        subdivideFVarChannels();
    }
}


//
//  The main method to subdivide the topology -- requires sizing and initialization of
//  child topology vectors and then appropriate assignment of them.
//
void
Refinement::subdivideTopology(Relations const& applyTo) {

    const Level& parent = *_parent;
          Level& child  = *_child;

    //
    //  Sizing and population of the topology vectors:
    //
    //  We have 6 relations to populate, but the way a child component's topology is
    //  determined depends on the nature of its origin from its parent components:
    //
    //      - face-verts:  originate from parent faces
    //      - face-edges:  originate from parent faces
    //      - edge-verts:  originate from parent faces or edges
    //      - edge-faces:  originate from parent faces or edges
    //      - vert-faces:  originate from parent faces or edges or verts
    //      - vert-edges:  originate from parent faces or edges or verts
    //
    //  Each relation has 1-3 cases of origin and so 1-3 methods of population.
    //
    //  It may be worth pairing up the two relations for each child component type,
    //  i.e. populating them together as they share the same type of origin, but keeping
    //  them separate may prove more efficient in general (writing to one destination at
    //  at a time) as well as provide opportunities for threading.
    //
    //  Note on sizing:
    //      All faces now quads or tris, so the face-* relation sizes should be known.
    //  For edge and vert relations, where the number of incident components is more
    //  variable, we allocate/offset assuming 1-to-1 correspondence between child and
    //  parent incident components (which will be true for uniform subdivision), then
    //  "compress" later on-the-fly as we determine the subset of parent components
    //  that contribute children to the neighborhood (not really "compressing" here
    //  but more initializing as we go and adjusting the counts and offsets to reflect
    //  the actual valences -- so we never pack/move the vector elements).
    //      Some of the above inhibits concurrency in the mapping of the child indices
    //  from the parent.  Knowing all of the counts/offsets before applying the mapping
    //  would be helpful.  Current count/offset vectors are:
    //
    //      - face-verts, also used by/shared with face-edges
    //      - edge-faces
    //      - vert-faces
    //      - vert-edges, not shared with vert-faces to allow non-manifold cases
    //
    //  So we have 6 relations to transform and 4 count/offset vectors required -- 3
    //  of which are currently assembled incrementally.  The two that are not build:
    //
    //      - face-edges, we share the face-vert counts/offsets
    //      - edge-verts, no need -- constant size of 2
    //
    //  There are many times were using a constant offset may be preferable, e.g. when
    //  all faces are quads or tris, there are no fin-edges, etc.  Not retrieving the
    //  offsets from memory would be beneficial, but branching at a fine-grain level
    //  to do so may defeat the purpose...
    //

    //
    //  Face relations -- both face-verts and face-edges share the same counts/offsets:
    //
    if (applyTo._faceVertices || applyTo._faceEdges) {
        initializeFaceVertexCountsAndOffsets();

        //  Face-verts -- allocate and populate:
        if (applyTo._faceVertices) {
            child._faceVertIndices.resize(child.getNumFaces() * 4);
            populateFaceVerticesFromParentFaces();
        }
        //  Face-edges -- allocate and populate:
        if (applyTo._faceEdges) {
            child._faceEdgeIndices.resize(child.getNumFaces() * 4);
            populateFaceEdgesFromParentFaces();
        }
    }

    //
    //  Edge relations -- edge-verts and edge-faces can be populated independently:
    //
    //  Edge-verts -- allocate and populate:
    if (applyTo._edgeVertices) {
        child._edgeVertIndices.resize(child.getNumEdges() * 2);

        populateEdgeVerticesFromParentFaces();
        populateEdgeVerticesFromParentEdges();
    }

    //  Edge-faces:
    //      NOTE we do not know the exact counts/offsets here because of the
    //  potentially sparse subdivision.  If we must compute this incrementally,
    //  i.e. as we populate the edge-face indices, we will not be able to thread
    //  that operation.  See populateEdgeFaceCountsAndOffsets() for possibilities
    //  of pre-computing these.
    //
    if (applyTo._edgeFaces) {
        //  Empty stub to eventually replace the incremental assembly that follows:
        initializeEdgeFaceCountsAndOffsets();

        int childEdgeFaceIndexSizeEstimate = (int)parent._faceVertIndices.size() * 2 +
                                             (int)parent._edgeFaceIndices.size() * 2;

        child._edgeFaceCountsAndOffsets.resize(child.getNumEdges() * 2);
        child._edgeFaceIndices.resize(         childEdgeFaceIndexSizeEstimate);

        populateEdgeFacesFromParentFaces();
        populateEdgeFacesFromParentEdges();

        //  Revise the over-allocated estimate based on what is used (as indicated in the
        //  count/offset for the last vertex) and trim the index vector accordingly:
        childEdgeFaceIndexSizeEstimate = child.getNumEdgeFaces(child.getNumEdges()-1) +
                                         child.getOffsetOfEdgeFaces(child.getNumEdges()-1);
        child._edgeFaceIndices.resize(childEdgeFaceIndexSizeEstimate);

        child._maxEdgeFaces = parent._maxEdgeFaces;
    }

    //
    //  Vert relations -- vert-faces and vert-edges can be populated independently:
    //
    //  Vert-faces:
    //      We know the number of counts/offsets required, but we do not yet know how
    //  many total incident faces we will have -- given we are generating a potential
    //  subset of the number of children possible, allocate for the maximum and trim
    //  once its determined how many were used.
    //      The upper-bound for allocation is determined by the incidence vectors for
    //  the parent components, e.g. assuming fully populated, the total number of faces
    //  incident all vertices that are generated from parent faces is equal to the size
    //  of the parent's face-vert index vector.  Similar deductions can be made for
    //  for those originating from parent edges and verts.
    //      If this over-allocation proves excessive, we can get a more reasonable
    //  estimate, or even the exact size, by iterating though topology/child vectors.
    //
    //      The bigger issue here though is that not having the counts/offsets known
    //  and updating it incrementally inhibits threading.  So consider figuring out the
    //  counts/offsets before remapping the topology:
    //      - if uniform subdivision, vert-face count will be:
    //          - 4 for verts from parent faces (for catmark)
    //          - 2x number in parent edge for verts from parent edges
    //          - same as parent vert for verts from parent verts
    //      - if sparse subdivision, vert-face count will be:
    //          - the number of child faces in parent face
    //          - 1 or 2x number in parent edge for verts from parent edges
    //              - where the 1 or 2 is number of child edges of parent edge
    //          - same as parent vert for verts from parent verts (catmark)
    //
    //  Note a couple of these metrics are Catmark-dependent, i.e. assuming 4 child
    //  faces per face, but more subtely the assumption that a child vert from a
    //  parent vert will have the same valence -- this will only be true if the
    //  subd Scheme requires a "neighborhood of 1", which will have been taken into
    //  account when marking/generating children.
    //
    if (applyTo._vertexFaces) {
        //  Empty stub to eventually replace the incremental assembly that follows:
        initializeVertexFaceCountsAndOffsets();

        int childVertFaceIndexSizeEstimate = (int)parent._faceVertIndices.size()
                                            + (int)parent._edgeFaceIndices.size() * 2
                                            + (int)parent._vertFaceIndices.size();

        child._vertFaceCountsAndOffsets.resize(child.getNumVertices() * 2);
        child._vertFaceIndices.resize(         childVertFaceIndexSizeEstimate);
        child._vertFaceLocalIndices.resize(    childVertFaceIndexSizeEstimate);

        populateVertexFacesFromParentFaces();
        populateVertexFacesFromParentEdges();
        populateVertexFacesFromParentVertices();

        //  Revise the over-allocated estimate based on what is used (as indicated in the
        //  count/offset for the last vertex) and trim the index vectors accordingly:
        childVertFaceIndexSizeEstimate = child.getNumVertexFaces(child.getNumVertices()-1) +
                                         child.getOffsetOfVertexFaces(child.getNumVertices()-1);
        child._vertFaceIndices.resize(     childVertFaceIndexSizeEstimate);
        child._vertFaceLocalIndices.resize(childVertFaceIndexSizeEstimate);
    }

    //  Vert-edges:
    //      As for vert-faces, we know the number of counts/offsets required, but we do
    //  not yet know how many total incident edges we will have.  So over-estimate and
    //  trim after population, similar to above for face-verts.
    //
    //  See also notes above regarding attempts to determine counts/offsets before we
    //  start to remap the components:
    //      - if uniform subdivision, vert-edge count will be:
    //          - 4 for verts from parent faces (for catmark)
    //          - 2 + N faces incident parent edge for verts from parent edges
    //          - same as parent vert for verts from parent verts
    //      - if sparse subdivision, vert-edge count will be:
    //          - non-trivial function of child faces in parent face
    //              - 1 child face will always result in 2 child edges
    //              * 2 child faces can mean 3 or 4 child edges
    //              - 3 child faces will always result in 4 child edges
    //          - 1 or 2 + N faces incident parent edge for verts from parent edges
    //              - where the 1 or 2 is number of child edges of parent edge
    //              - any end vertex will require all N child faces (catmark)
    //          - same as parent vert for verts from parent verts (catmark)
    //
    if (applyTo._vertexEdges) {
        //  Empty stub to eventually replace the incremental assembly that follows:
        initializeVertexEdgeCountsAndOffsets();

        int childVertEdgeIndexSizeEstimate = (int)parent._faceVertIndices.size()
                                           + (int)parent._edgeFaceIndices.size() + parent.getNumEdges() * 2
                                           + (int)parent._vertEdgeIndices.size();

        child._vertEdgeCountsAndOffsets.resize(child.getNumVertices() * 2);
        child._vertEdgeIndices.resize(         childVertEdgeIndexSizeEstimate);
        child._vertEdgeLocalIndices.resize(    childVertEdgeIndexSizeEstimate);

        populateVertexEdgesFromParentFaces();
        populateVertexEdgesFromParentEdges();
        populateVertexEdgesFromParentVertices();

        //  Revise the over-allocated estimate based on what is used (as indicated in the
        //  count/offset for the last vertex) and trim the index vectors accordingly:
        childVertEdgeIndexSizeEstimate = child.getNumVertexEdges(child.getNumVertices()-1) +
                                         child.getOffsetOfVertexEdges(child.getNumVertices()-1);
        child._vertEdgeIndices.resize(     childVertEdgeIndexSizeEstimate);
        child._vertEdgeLocalIndices.resize(childVertEdgeIndexSizeEstimate);
    }
    child._maxValence = parent._maxValence;
}


//
//  Sharpness subdivision methods:
//      Subdividing vertex sharpness is relatively trivial -- most complexity is in
//  the edge sharpness given its varying computation methods.
//
void
Refinement::subdivideEdgeSharpness() {

    Sdc::Crease creasing(_schemeOptions);

    _child->_edgeSharpness.clear();
    _child->_edgeSharpness.resize(_child->getNumEdges(), Sdc::Crease::SHARPNESS_SMOOTH);

    //
    //  Edge sharpness is passed to child-edges using the parent edge and the
    //  parent vertex for which the child corresponds.  Child-edges are created
    //  from both parent faces and parent edges, but those child-edges created
    //  from a parent face should be within the face's interior and so smooth
    //  (and so previously initialized).
    //
    //  The presence/validity of each parent edges child vert indicates one or
    //  more child edges.
    //
    //  NOTE -- It is also useful at this time to classify the child vert of
    //  this edge based on the creasing information here, particularly when a
    //  non-trivial creasing method like Chaikin is used.  This is not being
    //  done now but is worth considering...
    //
    //  Only deal with the subrange of edges originating from edges:
    int cEdgeBegin = _childEdgeFromFaceCount;
    int cEdgeEnd   =  _child->getNumEdges();

    float * pVertEdgeSharpness = 0;
    if (!creasing.IsUniform()) {
        pVertEdgeSharpness = (float *)alloca(_parent->getMaxValence() * sizeof(float));
    }

    for (Index cEdge = cEdgeBegin; cEdge < cEdgeEnd; ++cEdge) {
        Sharpness&   cSharpness = _child->_edgeSharpness[cEdge];
        Level::ETag& cEdgeTag   = _child->_edgeTags[cEdge];

        if (cEdgeTag._infSharp) {
            cSharpness = Sdc::Crease::SHARPNESS_INFINITE;
        } else if (cEdgeTag._semiSharp) {
            Index       pEdge      = _childEdgeParentIndex[cEdge];
            Sharpness   pSharpness = _parent->_edgeSharpness[pEdge];

            if (creasing.IsUniform()) {
                cSharpness = creasing.SubdivideUniformSharpness(pSharpness);
            } else {
                IndexArray const pEdgeVerts = _parent->getEdgeVertices(pEdge);
                Index            pVert      = pEdgeVerts[_childEdgeTag[cEdge]._indexInParent];
                IndexArray const pVertEdges = _parent->getVertexEdges(pVert);

                for (int i = 0; i < pVertEdges.size(); ++i) {
                    pVertEdgeSharpness[i] = _parent->_edgeSharpness[pVertEdges[i]];
                }
                cSharpness = creasing.SubdivideEdgeSharpnessAtVertex(pSharpness, pVertEdges.size(),
                                                                         pVertEdgeSharpness);
            }
            cEdgeTag._semiSharp = Sdc::Crease::IsSharp(cSharpness);
        }
    }
}

void
Refinement::subdivideVertexSharpness() {

    Sdc::Crease creasing(_schemeOptions);

    _child->_vertSharpness.clear();
    _child->_vertSharpness.resize(_child->getNumVertices(), Sdc::Crease::SHARPNESS_SMOOTH);

    //
    //  All child-verts originating from faces or edges are initialized as smooth
    //  above.  Only those originating from vertices require "subdivided" values:
    //
    //  Only deal with the subrange of vertices originating from vertices:
    int cVertBegin = _childVertFromFaceCount + _childVertFromEdgeCount;
    int cVertEnd   = _child->getNumVertices();

    for (Index cVert = cVertBegin; cVert < cVertEnd; ++cVert) {
        Sharpness&   cSharpness = _child->_vertSharpness[cVert];
        Level::VTag& cVertTag   = _child->_vertTags[cVert];

        if (cVertTag._infSharp) {
            cSharpness = Sdc::Crease::SHARPNESS_INFINITE;
        } else if (cVertTag._semiSharp) {
            Index       pVert      = _childVertexParentIndex[cVert];
            Sharpness   pSharpness = _parent->_vertSharpness[pVert];

            cSharpness = creasing.SubdivideVertexSharpness(pSharpness);

            if (!Sdc::Crease::IsSharp(cSharpness)) {
                //  Need to visit edge neighborhood to determine if still semisharp...
                //      cVertTag._infSharp = ...?
                //  See the "reclassify" method below...
            }
        }
    }
}

void
Refinement::reclassifySemisharpVertices() {

    typedef Level::VTag::VTagSize VTagSize;

    Sdc::Crease creasing(_schemeOptions);

    //
    //  Inspect all vertices derived from edges -- for those whose parent edges were semisharp,
    //  reset the semisharp tag and the associated Rule according to the sharpness pair for the
    //  subdivided edges (note this may be better handled when the edge sharpness is computed):
    //
    int vertFromEdgeMin = _childVertFromFaceCount;
    int vertFromEdgeMax = vertFromEdgeMin + _childVertFromEdgeCount;

    for (Index cVert = vertFromEdgeMin; cVert < vertFromEdgeMax; ++cVert) {
        Level::VTag& cVertTag = _child->_vertTags[cVert];
        if (!cVertTag._semiSharp) continue;

        Index pEdge = _childVertexParentIndex[cVert];

        IndexArray const cEdges = getEdgeChildEdges(pEdge);

        if (_childVertexTag[cVert]._incomplete) {
            //  One child edge likely missing -- assume Crease if remaining edge semi-sharp:
            cVertTag._semiSharp = (IndexIsValid(cEdges[0]) && _child->_edgeTags[cEdges[0]]._semiSharp) ||
                                  (IndexIsValid(cEdges[1]) && _child->_edgeTags[cEdges[1]]._semiSharp);
            cVertTag._rule      = (VTagSize)(cVertTag._semiSharp ? Sdc::Crease::RULE_CREASE : Sdc::Crease::RULE_SMOOTH);
        } else {
            int sharpEdgeCount = _child->_edgeTags[cEdges[0]]._semiSharp + _child->_edgeTags[cEdges[1]]._semiSharp;

            cVertTag._semiSharp = (sharpEdgeCount > 0);
            cVertTag._rule      = (VTagSize)(creasing.DetermineVertexVertexRule(0.0, sharpEdgeCount));
        }
    }

    //
    //  Inspect all vertices derived from vertices -- for those whose parent vertices were
    //  semisharp, reset the semisharp tag and the associated Rule based on the neighborhood
    //  of child edges around the child vertex.
    //
    //  We should never find such a vertex "incomplete" in a sparse refinement as a parent
    //  vertex is either selected or not, but never neighboring.  So the only complication
    //  here is whether the local topology of child edges exists -- it may have been pruned
    //  from the last level to reduce memory.  If so, we use the parent to identify the
    //  child edges.
    //
    //  In both cases, we count the number of sharp and semisharp child edges incident the
    //  child vertex and adjust the "semisharp" and "rule" tags accordingly.
    //
    int vertFromVertMin = vertFromEdgeMax;
    int vertFromVertMax = vertFromVertMin + _childVertFromVertCount;

    for (Index cVert = vertFromVertMin; cVert < vertFromVertMax; ++cVert) {
        Level::VTag& cVertTag = _child->_vertTags[cVert];
        if (!cVertTag._semiSharp) continue;

        //  If the vertex is still sharp, it remains the semisharp Corner its parent was...
        if (_child->_vertSharpness[cVert] > 0.0) continue;

        //
        //  See if we can use the vert-edges of the child vertex:
        //
        int sharpEdgeCount = 0;
        int semiSharpEdgeCount = 0;

        bool cVertEdgesPresent = (_child->getNumVertexEdgesTotal() > 0);
        if (cVertEdgesPresent) {
            IndexArray const cEdges = _child->getVertexEdges(cVert);

            for (int i = 0; i < cEdges.size(); ++i) {
                Level::ETag cEdgeTag = _child->_edgeTags[cEdges[i]];

                sharpEdgeCount     += cEdgeTag._semiSharp || cEdgeTag._infSharp;
                semiSharpEdgeCount += cEdgeTag._semiSharp;
            }
        } else {
            Index pVert  = _childVertexParentIndex[cVert];

            IndexArray const      pEdges      = _parent->getVertexEdges(pVert);
            LocalIndexArray const pVertInEdge = _parent->getVertexEdgeLocalIndices(pVert);

            for (int i = 0; i < pEdges.size(); ++i) {
                IndexArray const cEdgePair = getEdgeChildEdges(pEdges[i]);

                Index       cEdge    = cEdgePair[pVertInEdge[i]];
                Level::ETag cEdgeTag = _child->_edgeTags[cEdge];

                sharpEdgeCount     += cEdgeTag._semiSharp || cEdgeTag._infSharp;
                semiSharpEdgeCount += cEdgeTag._semiSharp;
            }
        }

        cVertTag._semiSharp = (semiSharpEdgeCount > 0);
        cVertTag._rule      = (VTagSize)(creasing.DetermineVertexVertexRule(0.0, sharpEdgeCount));
    }
}

void
Refinement::subdivideFVarChannels() {

//printf("Refinement::subdivideFVarChannels() -- level %d...\n", _child->_depth);
    assert(_child->_fvarChannels.size() == 0);
    assert(this->_fvarChannels.size() == 0);

    int channelCount = _parent->getNumFVarChannels();

    for (int channel = 0; channel < channelCount; ++channel) {
        FVarLevel* parentFVar = _parent->_fvarChannels[channel];

        FVarLevel*      childFVar  = new FVarLevel(*_child);
        FVarRefinement* refineFVar = new FVarRefinement(*this, *parentFVar, *childFVar);

        refineFVar->applyRefinement();

        _child->_fvarChannels.push_back(childFVar);
        this->_fvarChannels.push_back(refineFVar);
    }
}


//
//  Methods to propagate/initialize component tags according to their parent component:
//
void
Refinement::propagateFaceTagsFromParentFaces() {

    //
    //  Tags for faces originating from faces are inherited from the parent face:
    //
    for (Index cFace = 0; cFace < _child->getNumFaces(); ++cFace) {
        _child->_faceTags[cFace] = _parent->_faceTags[_childFaceParentIndex[cFace]];
    }
    /*
    for (int pFace = 0; pFace < _parent->getNumFaces(); ++pFace) {
        Level::FTag const& fTag = _parent->_faceTags[pFace];

        IndexArray const pFaceChildFaces = getFaceChildFaces(pFace);
        for (int i = 0; i < pFaceChildFaces.size(); ++i) {
            Index cFace = pFaceChildFaces[i];

            if (IndexIsValid(cFace)) _child->_faceTags[cFace] = fTag;
        }
    }
    */
}
void
Refinement::propagateEdgeTagsFromParentFaces() {

    //
    //  Tags for edges originating from faces are all constant:
    //
    Level::ETag eTag;
    eTag._nonManifold = 0;
    eTag._boundary    = 0;
    eTag._infSharp    = 0;
    eTag._semiSharp   = 0;

    for (Index cEdge = 0; cEdge < _childEdgeFromFaceCount; ++cEdge) {
        _child->_edgeTags[cEdge] = eTag;
    }
}
void
Refinement::propagateVertexTagsFromParentFaces() {

    //
    //  Similarly, tags for vertices originating from faces are all constant -- with the
    //  unfortunate exception of refining level 0, where the faces may be N-sided and so
    //  introduce new vertices that need to be tagged as extra-ordinary:
    //
    Level::VTag vTag;
    vTag._nonManifold = 0;
    vTag._xordinary   = 0;
    vTag._boundary    = 0;
    vTag._infSharp    = 0;
    vTag._semiSharp   = 0;
    vTag._rule        = Sdc::Crease::RULE_SMOOTH;

    if (_parent->_depth > 0) {
        for (Index cVert = 0; cVert < _childVertFromFaceCount; ++cVert) {
            _child->_vertTags[cVert] = vTag;
        }
    } else {
        for (Index cVert = 0; cVert < _childVertFromFaceCount; ++cVert) {
            _child->_vertTags[cVert] = vTag;
            if (_parent->getNumFaceVertices(_childVertexParentIndex[cVert]) != 4) {
                _child->_vertTags[cVert]._xordinary = true;
            }
        }
    }
}
void
Refinement::propagateEdgeTagsFromParentEdges() {

    //
    //  Tags for edges originating from edges are inherited from the parent edge:
    //
    for (Index cEdge = _childEdgeFromFaceCount; cEdge < _child->getNumEdges(); ++cEdge) {
        _child->_edgeTags[cEdge] = _parent->_edgeTags[_childEdgeParentIndex[cEdge]];
    }
    /*
    for (Index pEdge = 0; pEdge < _parent->getNumEdges(); ++pEdge) {
        Level::ETag const& pEdgeTag = _parent->_edgeTags[pEdge];
        IndexArray const   cEdges   = getEdgeChildEdges(pEdge);

        if (IndexIsValid(cEdges[0])) _child->_edgeTags[cEdges[0]] = pEdgeTag;
        if (IndexIsValid(cEdges[1])) _child->_edgeTags[cEdges[1]] = pEdgeTag;
    }
    */
}
void
Refinement::propagateVertexTagsFromParentEdges() {

    //
    //  Tags for vertices originating from edges are initialized according to the tags
    //  of the parent edge:
    //
    for (Index pEdge = 0; pEdge < _parent->getNumEdges(); ++pEdge) {
        Index cVert = _edgeChildVertIndex[pEdge];
        if (!IndexIsValid(cVert)) continue;

        Level::ETag const& pEdgeTag = _parent->_edgeTags[pEdge];
        Level::VTag&       cVertTag = _child->_vertTags[cVert];

        cVertTag._nonManifold = pEdgeTag._nonManifold;
        cVertTag._xordinary   = false;
        cVertTag._boundary    = pEdgeTag._boundary;
        cVertTag._infSharp    = false;

        cVertTag._semiSharp = pEdgeTag._semiSharp;
        cVertTag._rule = (Level::VTag::VTagSize)((pEdgeTag._semiSharp || pEdgeTag._infSharp)
                       ? Sdc::Crease::RULE_CREASE : Sdc::Crease::RULE_SMOOTH);
    }
}
void
Refinement::propagateVertexTagsFromParentVertices() {

    //
    //  Tags for vertices originating from vertices are inherited from the parent vertex:
    //
    for (Index cVert = _childVertFromFaceCount + _childVertFromEdgeCount;
                  cVert < _child->getNumVertices(); ++cVert) {
        _child->_vertTags[cVert] = _parent->_vertTags[_childVertexParentIndex[cVert]];
    }
    /*
    for (Index pVert = 0; pVert < _parent->getNumVertices(); ++pVert) {
        Index cVert = _vertChildVertIndex[pVert];

        if (IndexIsValid(cVert)) _child->_vertTags[cVert] = _parent->_vertTags[pVert];
    }
    */
}

void
Refinement::propagateComponentTags() {

    _child->_faceTags.resize(_child->getNumFaces());
    _child->_edgeTags.resize(_child->getNumEdges());
    _child->_vertTags.resize(_child->getNumVertices());

    propagateFaceTagsFromParentFaces();

    propagateEdgeTagsFromParentFaces();
    propagateEdgeTagsFromParentEdges();

    propagateVertexTagsFromParentFaces();
    propagateVertexTagsFromParentEdges();
    propagateVertexTagsFromParentVertices();
}

void
Refinement::populateChildToParentTags() {

    ChildTag childTags[2][4];
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 4; ++j) {
            ChildTag& tag = childTags[i][j];
            tag._incomplete    = (unsigned char)i;
            tag._parentType    = 0;
            tag._indexInParent = (unsigned char)j;
        }
    }

    if (_uniform) {
        Index cFace = 0;
        for (Index pFace = 0; pFace < _parent->getNumFaces(); ++pFace) {
            IndexArray pVerts = _parent->getFaceVertices(pFace);
            if (pVerts.size() == 4) {
                _childFaceTag[cFace + 0] = childTags[0][0];
                _childFaceTag[cFace + 1] = childTags[0][1];
                _childFaceTag[cFace + 2] = childTags[0][2];
                _childFaceTag[cFace + 3] = childTags[0][3];

                _childFaceParentIndex[cFace + 0] = pFace;
                _childFaceParentIndex[cFace + 1] = pFace;
                _childFaceParentIndex[cFace + 2] = pFace;
                _childFaceParentIndex[cFace + 3] = pFace;

                cFace += 4;
            } else {
                for (int i = 0; i < pVerts.size(); ++i, ++cFace) {
                    _childFaceTag[cFace] = childTags[0][i];
                    _childFaceParentIndex[cFace] = pFace;
                }
            }
        }

        Index cEdge = 0;
        for (Index pFace = 0; pFace < _parent->getNumFaces(); ++pFace) {
            IndexArray pVerts = _parent->getFaceVertices(pFace);
            if (pVerts.size() == 4) {
                _childEdgeTag[cEdge + 0] = childTags[0][0];
                _childEdgeTag[cEdge + 1] = childTags[0][1];
                _childEdgeTag[cEdge + 2] = childTags[0][2];
                _childEdgeTag[cEdge + 3] = childTags[0][3];

                _childEdgeParentIndex[cEdge + 0] = pFace;
                _childEdgeParentIndex[cEdge + 1] = pFace;
                _childEdgeParentIndex[cEdge + 2] = pFace;
                _childEdgeParentIndex[cEdge + 3] = pFace;

                cEdge += 4;
            } else {
                for (int i = 0; i < pVerts.size(); ++i, ++cEdge) {
                    _childEdgeTag[cEdge] = childTags[0][i];
                    _childEdgeParentIndex[cEdge] = pFace;
                }
            }
        }
        for (Index pEdge = 0; pEdge < _parent->getNumEdges(); ++pEdge, cEdge += 2) {
            _childEdgeTag[cEdge + 0] = childTags[0][0];
            _childEdgeTag[cEdge + 1] = childTags[0][1];

            _childEdgeParentIndex[cEdge + 0] = pEdge;
            _childEdgeParentIndex[cEdge + 1] = pEdge;
        }

        Index cVert = 0;
        for (Index pFace = 0; pFace < _parent->getNumFaces(); ++pFace, ++cVert) {
            _childVertexTag[cVert] = childTags[0][0];
            _childVertexParentIndex[cVert] = pFace;
        }
        for (Index pEdge = 0; pEdge < _parent->getNumEdges(); ++pEdge, ++cVert) {
            _childVertexTag[cVert] = childTags[0][0];
            _childVertexParentIndex[cVert] = pEdge;
        }
        for (Index pVert = 0; pVert < _parent->getNumVertices(); ++pVert, ++cVert) {
            _childVertexTag[cVert] = childTags[0][0];
            _childVertexParentIndex[cVert] = pVert;
        }
    } else {
        //  Child faces of faces:
        for (Index pFace = 0; pFace < _parent->getNumFaces(); ++pFace) {
            bool incomplete = !_parentFaceTag[pFace]._selected;

            IndexArray cFaces = getFaceChildFaces(pFace);
            if (!incomplete && (cFaces.size() == 4)) {
                _childFaceTag[cFaces[0]] = childTags[0][0];
                _childFaceTag[cFaces[1]] = childTags[0][1];
                _childFaceTag[cFaces[2]] = childTags[0][2];
                _childFaceTag[cFaces[3]] = childTags[0][3];

                _childFaceParentIndex[cFaces[0]] = pFace;
                _childFaceParentIndex[cFaces[1]] = pFace;
                _childFaceParentIndex[cFaces[2]] = pFace;
                _childFaceParentIndex[cFaces[3]] = pFace;
            } else {
                for (int i = 0; i < cFaces.size(); ++i) {
                    if (IndexIsValid(cFaces[i])) {
                        _childFaceTag[cFaces[i]] = childTags[incomplete][i];
                        _childFaceParentIndex[cFaces[i]] = pFace;
                    }
                }
            }
        }

        //  Child edges of faces and edges:
        for (Index pFace = 0; pFace < _parent->getNumFaces(); ++pFace) {
            bool incomplete = !_parentFaceTag[pFace]._selected;

            IndexArray cEdges = getFaceChildEdges(pFace);
            if (!incomplete && (cEdges.size() == 4)) {
                _childEdgeTag[cEdges[0]] = childTags[0][0];
                _childEdgeTag[cEdges[1]] = childTags[0][1];
                _childEdgeTag[cEdges[2]] = childTags[0][2];
                _childEdgeTag[cEdges[3]] = childTags[0][3];

                _childEdgeParentIndex[cEdges[0]] = pFace;
                _childEdgeParentIndex[cEdges[1]] = pFace;
                _childEdgeParentIndex[cEdges[2]] = pFace;
                _childEdgeParentIndex[cEdges[3]] = pFace;
            } else {
                for (int i = 0; i < cEdges.size(); ++i) {
                    if (IndexIsValid(cEdges[i])) {
                        _childEdgeTag[cEdges[i]] = childTags[incomplete][i];
                        _childEdgeParentIndex[cEdges[i]] = pFace;
                    }
                }
            }
        }
        for (Index pEdge = 0; pEdge < _parent->getNumEdges(); ++pEdge) {
            bool incomplete = !_parentEdgeTag[pEdge]._selected;

            IndexArray cEdges = getEdgeChildEdges(pEdge);
            if (!incomplete) {
                _childEdgeTag[cEdges[0]] = childTags[0][0];
                _childEdgeTag[cEdges[1]] = childTags[0][1];

                _childEdgeParentIndex[cEdges[0]] = pEdge;
                _childEdgeParentIndex[cEdges[1]] = pEdge;
            } else {
                for (int i = 0; i < 2; ++i) {
                    if (IndexIsValid(cEdges[i])) {
                        _childEdgeTag[cEdges[i]] = childTags[incomplete][i];
                        _childEdgeParentIndex[cEdges[i]] = pEdge;
                    }
                }
            }
        }

        //  Child vertices of faces, edges and vertices:
        for (Index pFace = 0; pFace < _parent->getNumFaces(); ++pFace) {
            Index cVert = _faceChildVertIndex[pFace];
            if (IndexIsValid(cVert)) {
                bool incomplete = !_parentFaceTag[pFace]._selected;

                _childVertexTag[cVert] = childTags[incomplete][0];
                _childVertexParentIndex[cVert] = pFace;
            }
        }
        for (Index pEdge = 0; pEdge < _parent->getNumEdges(); ++pEdge) {
            Index cVert = _edgeChildVertIndex[pEdge];
            if (IndexIsValid(cVert)) {
                bool incomplete = !_parentEdgeTag[pEdge]._selected;

                _childVertexTag[cVert] = childTags[incomplete][0];
                _childVertexParentIndex[cVert] = pEdge;
            }
        }
        for (Index pVert = 0; pVert < _parent->getNumVertices(); ++pVert) {
            Index cVert = _vertChildVertIndex[pVert];
            if (IndexIsValid(cVert)) {
                _childVertexTag[cVert] = childTags[0][0];
                _childVertexParentIndex[cVert] = pVert;
            }
        }
    }
}

void
Refinement::populateChildToParentIndices() {
}

void
Refinement::populateChildToParentMapping() {
    assert(_quadSplit);

    _childVertexParentIndex.resize(_child->getNumVertices());
    _childEdgeParentIndex.resize(_child->getNumEdges());
    _childFaceParentIndex.resize(_child->getNumFaces());

    _childVertexTag.resize(_child->getNumVertices());
    _childEdgeTag.resize(_child->getNumEdges());
    _childFaceTag.resize(_child->getNumFaces());

    populateChildToParentIndices();

    populateChildToParentTags();
}

#ifdef _VTR_COMPUTE_MASK_WEIGHTS_ENABLED
void
Refinement::computeMaskWeights() {

    const Level& parent = *this->_parent;
          Level& child  = *this->_child;

    assert(child.getNumVertices() != 0);

    _faceVertWeights.resize(parent.faceVertCount());
    _edgeVertWeights.resize(parent.edgeVertCount());
    _edgeFaceWeights.resize(parent.edgeFaceCount());
    _vertVertWeights.resize(parent.getNumVertices());
    _vertEdgeWeights.resize(parent.vertEdgeCount());
    _vertFaceWeights.resize(parent.vertEdgeCount());

    //
    //  Hard-coding this for Catmark temporarily for testing...
    //
    assert(_schemeType == Sdc::TYPE_CATMARK);

    Sdc::Scheme<Sdc::TYPE_CATMARK> scheme(_schemeOptions);

    if (_childVertFromFaceCount) {
        for (int pFace = 0; pFace < parent.getNumFaces(); ++pFace) {
            Index cVert = _faceChildVertIndex[pFace];
            if (!IndexIsValid(cVert)) continue;

            int    fVertCount   = parent.faceVertCount(pFace);
            int    fVertOffset  = parent.faceVertOffset(pFace);
            float* fVertWeights = &_faceVertWeights[fVertOffset];

            MaskInterface fMask(fVertWeights, 0, 0);
            FaceInterface fHood(fVertCount);

            scheme.ComputeFaceVertexMask(fHood, fMask);
        }
    }
    if (_childVertFromEdgeCount) {
        EdgeInterface eHood(parent);

        for (int pEdge = 0; pEdge < parent.getNumEdges(); ++pEdge) {
            Index cVert = _edgeChildVertIndex[pEdge];
            if (!IndexIsValid(cVert)) continue;

            //
            //  Update the locations for the mask weights:
            //
            int    eFaceCount   = parent.edgeFaceCount(pEdge);
            int    eFaceOffset  = parent.edgeFaceOffset(pEdge);
            float* eFaceWeights = &_edgeFaceWeights[eFaceOffset];
            float* eVertWeights = &_edgeVertWeights[2 * pEdge];

            MaskInterface eMask(eVertWeights, 0, eFaceWeights);

            //
            //  Identify the parent and child and compute weights -- note that the face
            //  weights may not be populated, so set them to zero if not:
            //
            eHood.SetIndex(pEdge);

            Sdc::Rule pRule = (parent._edgeSharpness[pEdge] > 0.0) ? Sdc::Crease::RULE_CREASE : Sdc::Crease::RULE_SMOOTH;
            Sdc::Rule cRule = child.getVertexRule(cVert);

            scheme.ComputeEdgeVertexMask(eHood, eMask, pRule, cRule);

            if (eMask.GetFaceWeightCount() == 0) {
                std::fill(eFaceWeights, eFaceWeights + eFaceCount, 0.0);
            }
        }
    }
    if (_childVertFromVertCount) {
        VertexInterface vHood(parent, child);

        for (int pVert = 0; pVert < parent.getNumVertices(); ++pVert) {
            Index cVert = _vertChildVertIndex[pVert];
            if (!IndexIsValid(cVert)) continue;

            //
            //  Update the locations for the mask weights:
            //
            float* vVertWeights = &_vertVertWeights[pVert];

            int    vEdgeCount   = parent.vertEdgeCount(pVert);
            int    vEdgeOffset  = parent.vertEdgeOffset(pVert);
            float* vEdgeWeights = &_vertEdgeWeights[vEdgeOffset];

            int    vFaceCount   = parent.vertFaceCount(pVert);
            int    vFaceOffset  = parent.vertFaceOffset(pVert);
            float* vFaceWeights = &_vertFaceWeights[vFaceOffset];

            MaskInterface vMask(vVertWeights, vEdgeWeights, vFaceWeights);

            //
            //  Initialize the neighborhood and gather the pre-determined Rules:
            //
            vHood.SetIndex(pVert, cVert);

            Sdc::Rule pRule = parent.vertRule(pVert);
            Sdc::Rule cRule = child.vertRule(cVert);

            scheme.ComputeVertexVertexMask(vHood, vMask, pRule, cRule);

            if (vMask.GetEdgeWeightCount() == 0) {
                std::fill(vEdgeWeights, vEdgeWeights + vEdgeCount, 0.0);
            }
            if (vMask.GetFaceWeightCount() == 0) {
                std::fill(vFaceWeights, vFaceWeights + vFaceCount, 0.0);
            }
        }
    }
}
#endif


//
//  Marking of sparse child components -- including those selected and those neighboring...
//
//      For schemes requiring neighboring support, this is the equivalent of the "guarantee
//  neighbors" in Hbr -- it ensures that all components required to define the limit of
//  those "selected" are also generated in the refinement.
//
//  The difference with Hbr is that we do this in a single pass for all components once
//  "selection" of components of interest has been completed.
//
//  Considering two approaches:
//      1) By Vertex neighborhoods:
//          - for each base vertex
//              - for each incident face
//                  - test and mark components for its child face
//  or
//      2) By Edge and Face contents:
//          - for each base edge
//              - test and mark local components
//          - for each base face
//              - test and mark local components
//
//  Given a typical quad mesh with N verts, N faces and 2*N edges, determine which is more
//  efficient...
//
//  Going with (2) initially for simplicity -- certain aspects of (1) are awkward, i.e. the
//  identification of child-edges to be marked (trivial in (2).  We are also guaranteed with
//  (2) that we only visit each component once, i.e. each edge and each face.
//
//  Revising the above assessment... (2) has gotten WAY more complicated once the ability to
//  select child faces is provided.  Given that feature is important to Manuel for support
//  of the FarStencilTables we have to assume it will be needed.  So we'll try (1) out as it
//  will be simpler to get it correct -- we can work on improving performance later.
//
//  Complexity added by child component selection:
//      - the child vertex of the component can now be selected as part of a child face or
//  edge, and so the parent face or edge is not fully selected.  So we've had to add another
//  bit to the marking masks to indicate when a parent component is "fully selected".
//      - selecting a child face creates the situation where child edges of parent edges do
//  not have any selected vertex at their ends -- both can be neighboring.  This complicated
//  the marking of neighboring child edges, which was otherwise trivial -- if any end vertex
//  of a child edge (of a parent edge) was selected, the child edge was at least neighboring.
//
//  Final note on the marking technique:
//      There are currently two values to the marking of child components, which are no
//  longer that useful.  It is now sufficient, and not likely to be necessary, to distinguish
//  between what was selected or added to support it.  Ultimately that will be determined by
//  inspecting the selected flag on the parent component once the child-to-parent map is in
//  place.
//
namespace {
    Index const IndexSparseMaskNeighboring = (1 << 0);
    Index const IndexSparseMaskSelected    = (1 << 1);

    inline void markSparseIndexNeighbor(Index& index) { index = IndexSparseMaskNeighboring; }
    inline void markSparseIndexSelected(Index& index) { index = IndexSparseMaskSelected; }
}

void
Refinement::markSparseChildComponents() {

    assert(_parentEdgeTag.size() > 0);
    assert(_parentFaceTag.size() > 0);
    assert(_parentVertexTag.size() > 0);

    //
    //  For each parent vertex:
    //      - mark the descending child vertex for each selected vertex
    //
    for (Index pVert = 0; pVert < parent().getNumVertices(); ++pVert) {
        if (_parentVertexTag[pVert]._selected) {
            markSparseIndexSelected(_vertChildVertIndex[pVert]);
        }
    }

    //
    //  For each parent edge:
    //      - mark the descending child edges and vertex for each selected edge
    //      - test each end vertex of unselected edges to see if selected:
    //          - mark both the child edge and the middle child vertex if so
    //      - set transitional bit for all edges based on selection of incident faces
    //
    //  Note that no edges have been marked "fully selected" -- only their vertices have
    //  been marked and marking of their child edges deferred to visiting each edge only
    //  once here.
    //
    for (Index pEdge = 0; pEdge < parent().getNumEdges(); ++pEdge) {
        IndexArray       eChildEdges = getEdgeChildEdges(pEdge);
        IndexArray const eVerts      = parent().getEdgeVertices(pEdge);

        SparseTag& pEdgeTag = _parentEdgeTag[pEdge];

        if (pEdgeTag._selected) {
            markSparseIndexSelected(eChildEdges[0]);
            markSparseIndexSelected(eChildEdges[1]);
            markSparseIndexSelected(_edgeChildVertIndex[pEdge]);
        } else {
            if (_parentVertexTag[eVerts[0]]._selected) {
                markSparseIndexNeighbor(eChildEdges[0]);
                markSparseIndexNeighbor(_edgeChildVertIndex[pEdge]);
            }
            if (_parentVertexTag[eVerts[1]]._selected) {
                markSparseIndexNeighbor(eChildEdges[1]);
                markSparseIndexNeighbor(_edgeChildVertIndex[pEdge]);
            }
        }

        //
        //  TAG the parent edges as "transitional" here if only one was selected (or in
        //  the more general non-manifold case, they are not all selected the same way).
        //  We use the transitional tags on the edges to TAG the parent face below.
        //
        //  Note -- this is best done now rather than as a post-process as we have more
        //  explicit information about the selected components.  Unless we also tag the
        //  parent faces as selected, we can't easily tell from the child-faces of the
        //  edge's incident faces which were generated by selection or neighboring...
        //
        IndexArray const eFaces = parent().getEdgeFaces(pEdge);
        if (eFaces.size() == 2) {
            pEdgeTag._transitional = (_parentFaceTag[eFaces[0]]._selected !=
                                      _parentFaceTag[eFaces[1]]._selected);
        } else if (eFaces.size() < 2) {
            pEdgeTag._transitional = false;
        } else {
            bool isFace0Selected = _parentFaceTag[eFaces[0]]._selected;

            pEdgeTag._transitional = false;
            for (int i = 1; i < eFaces.size(); ++i) {
                if (_parentFaceTag[eFaces[i]]._selected != isFace0Selected) {
                    pEdgeTag._transitional = true;
                    break;
                }
            }
        }
    }

    //
    //  For each parent face:
    //      All boundary edges will be adequately marked as a result of the pass over the
    //  edges above and boundary vertices marked by selection.  So all that remains is to
    //  identify the child faces and interior child edges for a face requiring neighboring
    //  child faces.
    //      For each corner vertex selected, we need to mark the corresponding child face,
    //  the two interior child edges and shared child vertex in the middle.
    //
    assert(_quadSplit);

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

        IndexArray const fVerts = parent().getFaceVertices(pFace);

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
                IndexArray const fEdges = parent().getFaceEdges(pFace);
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
