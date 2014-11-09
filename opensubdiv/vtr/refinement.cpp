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
#include <utility>


namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Vtr {

//
//  Simple constructor, destructor and basic initializers:
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
    _childVertFromVertCount(0),
    _firstChildFaceFromFace(0),
    _firstChildEdgeFromFace(0),
    _firstChildEdgeFromEdge(0),
    _firstChildVertFromFace(0),
    _firstChildVertFromEdge(0),
    _firstChildVertFromVert(0) {
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

    assert((schemeType == Sdc::TYPE_CATMARK) || (schemeType == Sdc::TYPE_BILINEAR));
    _quadSplit = true;
}

void
Refinement::initializeChildComponentCounts() {

    //
    //  Assign the child's component counts/inventory based on the child components identified:
    //
    _child->_faceCount = _childFaceFromFaceCount;
    _child->_edgeCount = _childEdgeFromFaceCount + _childEdgeFromEdgeCount;
    _child->_vertCount = _childVertFromFaceCount + _childVertFromEdgeCount + _childVertFromVertCount;
}

void
Refinement::initializeSparseSelectionTags() {

    _parentFaceTag.resize(_parent->getNumFaces());
    _parentEdgeTag.resize(_parent->getNumEdges());
    _parentVertexTag.resize(_parent->getNumVertices());
}


//
//  The main refinement method -- provides a high-level overview of refinement:
//
//  The refinement process is as follows:
//      - determine a mapping from parent components to their potential child components
//          - for sparse refinement this mapping will be partial
//      - determine the reverse mapping from chosen child components back to their parents
//          - previously this was optional -- not strictly necessary and comes at added cost
//          - does simplify iteration of child components when refinement is sparse
//      - propagate/initialize component Tags from parents to their children
//          - knowing these Tags for a child component simplifies dealing with it later
//      - subdivide the topology, i.e. populate all topology relations for the child Level
//          - any subset of the 6 relations in a Level can be created
//          - using the minimum required in the last Level is very advantageous
//      - subdivide the sharpness values in the child Level
//      - subdivide face-varying channels in the child Level
//
void
Refinement::refine(Options refineOptions) {

    //  This will become redundant when/if assigned on construction:
    assert(_parent && _child);

    _uniform = !refineOptions._sparse;

    //
    //  Initialize the parent-to-child and reverse child-to-parent mappings and propagate
    //  component tags to the new child components:
    //
    populateParentToChildMapping();

    initializeChildComponentCounts();

    populateChildToParentMapping();

    propagateComponentTags();

    //
    //  Subdivide the topology -- populating only those of the 6 relations specified:
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
    //  Subdivide the sharpness values and face-varying channels:
    //    - note there is some dependency of the vertex tag/Rule for semi-sharp vertices
    //
    subdivideSharpnessValues();

    //  We may have an option here to suppress face-varying channels...
    bool refineOptions_faceVaryingChannels = true;
    if (refineOptions_faceVaryingChannels) {
        subdivideFVarChannels();
    }

    //  Various debugging support:
    //
    //printf("Vertex refinement to level %d completed...\n", _child->getDepth());
    //_child->print();
    //printf("  validating refinement to level %d...\n", _child->getDepth());
    //_child->validateTopology();
    //assert(_child->validateTopology());
}


//
//  Methods for construct the parent-to-child mapping
//
void
Refinement::populateParentToChildMapping() {

    allocateParentChildIndices();

    //
    //  If sparse refinement, mark indices of any components in addition to those selected
    //  so that we have the full neighborhood for selected components:
    //
    if (!_uniform) {
        //  Make sure the selection was non-empty -- currently unsupported...
        if (_parentVertexTag.size() == 0) {
            assert("Unsupported empty sparse refinement detected in Refinement" == 0);
        }
        markSparseChildComponentIndices();
    }

    populateParentChildIndices();
}

void
Refinement::allocateParentChildIndices() {

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

        //  Beware these child-counts when Loop subdivision supports N-sided faces in the cage
        //      - there will 2*(N-2) additional face-child-faces for each N-sided face
        //      - there will 2*(N-2)+1 additional face-child-edges for each N-sided face
        //      - there will 1 face-child-vertex for each N-sided face
        //  Can consider these reasonable estimates and grow as needed later -- but be clear
        //  about it if so.

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
Refinement::populateParentChildIndices() {

    //
    //  Two vertex orderings are under consideration -- the current/original orders
    //  vertices originating from faces first (historically these were relied upon
    //  to compute the rest of the vertices) while ordering vertices from vertices
    //  first is being considered (advantageous as it preserves the index of a parent
    //  vertex at all subsequent levels).
    //
    //  Other than defining the same ordering for refinement face-varying channels
    //  (which can be inferred from settings here) the rest of the code should be
    //  invariant to vertex ordering.
    //
    bool faceVertsFirst = true;

    //
    //  These two blocks now differ only in the utility function that assigns the
    //  sequential values to the index vectors -- so parameterization/simplification
    //  is now possible...
    //
    if (_uniform) {
        //  child faces:
        _firstChildFaceFromFace = 0;
        _childFaceFromFaceCount = sequenceFullIndexVector(_faceChildFaceIndices, _firstChildFaceFromFace);

        //  child edges:
        _firstChildEdgeFromFace = 0;
        _childEdgeFromFaceCount = sequenceFullIndexVector(_faceChildEdgeIndices, _firstChildEdgeFromFace);

        _firstChildEdgeFromEdge = _childEdgeFromFaceCount;
        _childEdgeFromEdgeCount = sequenceFullIndexVector(_edgeChildEdgeIndices, _firstChildEdgeFromEdge);

        //  child vertices:
        if (faceVertsFirst) {
            _firstChildVertFromFace = 0;
            _childVertFromFaceCount = sequenceFullIndexVector(_faceChildVertIndex, _firstChildVertFromFace);

            _firstChildVertFromEdge = _firstChildVertFromFace + _childVertFromFaceCount;
            _childVertFromEdgeCount = sequenceFullIndexVector(_edgeChildVertIndex, _firstChildVertFromEdge);

            _firstChildVertFromVert = _firstChildVertFromEdge + _childVertFromEdgeCount;
            _childVertFromVertCount = sequenceFullIndexVector(_vertChildVertIndex, _firstChildVertFromVert);
        } else {
            _firstChildVertFromVert = 0;
            _childVertFromVertCount = sequenceFullIndexVector(_vertChildVertIndex, _firstChildVertFromVert);

            _firstChildVertFromFace = _firstChildVertFromVert + _childVertFromVertCount;
            _childVertFromFaceCount = sequenceFullIndexVector(_faceChildVertIndex, _firstChildVertFromFace);

            _firstChildVertFromEdge = _firstChildVertFromFace + _childVertFromFaceCount;
            _childVertFromEdgeCount = sequenceFullIndexVector(_edgeChildVertIndex, _firstChildVertFromEdge);
        }
    } else {
        //  child faces:
        _firstChildFaceFromFace = 0;
        _childFaceFromFaceCount = sequenceSparseIndexVector(_faceChildFaceIndices, _firstChildFaceFromFace);

        //  child edges:
        _firstChildEdgeFromFace = 0;
        _childEdgeFromFaceCount = sequenceSparseIndexVector(_faceChildEdgeIndices, _firstChildEdgeFromFace);

        _firstChildEdgeFromEdge = _childEdgeFromFaceCount;
        _childEdgeFromEdgeCount = sequenceSparseIndexVector(_edgeChildEdgeIndices, _firstChildEdgeFromEdge);

        //  child vertices:
        if (faceVertsFirst) {
            _firstChildVertFromFace = 0;
            _childVertFromFaceCount = sequenceSparseIndexVector(_faceChildVertIndex, _firstChildVertFromFace);

            _firstChildVertFromEdge = _firstChildVertFromFace + _childVertFromFaceCount;
            _childVertFromEdgeCount = sequenceSparseIndexVector(_edgeChildVertIndex, _firstChildVertFromEdge);

            _firstChildVertFromVert = _firstChildVertFromEdge + _childVertFromEdgeCount;
            _childVertFromVertCount = sequenceSparseIndexVector(_vertChildVertIndex, _firstChildVertFromVert);
        } else {
            _firstChildVertFromVert = 0;
            _childVertFromVertCount = sequenceSparseIndexVector(_vertChildVertIndex, _firstChildVertFromVert);

            _firstChildVertFromFace = _firstChildVertFromVert + _childVertFromVertCount;
            _childVertFromFaceCount = sequenceSparseIndexVector(_faceChildVertIndex, _firstChildVertFromFace);

            _firstChildVertFromEdge = _firstChildVertFromFace + _childVertFromFaceCount;
            _childVertFromEdgeCount = sequenceSparseIndexVector(_edgeChildVertIndex, _firstChildVertFromEdge);
        }
    }
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


//
//  Methods to construct the child-to-parent mapping:
//
void
Refinement::populateChildToParentMapping() {

    ChildTag initialChildTags[2][4];
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 4; ++j) {
            ChildTag & tag = initialChildTags[i][j];

            tag._incomplete    = (unsigned char)i;
            tag._parentType    = 0;
            tag._indexInParent = (unsigned char)j;
        }
    }

    populateFaceParentVectors(initialChildTags);
    populateEdgeParentVectors(initialChildTags);
    populateVertexParentVectors(initialChildTags);
}

void
Refinement::populateFaceParentVectors(ChildTag const initialChildTags[2][4]) {

    _childFaceTag.resize(_child->getNumFaces());
    _childFaceParentIndex.resize(_child->getNumFaces());

    populateFaceParentFromParentFaces(initialChildTags);
}
void
Refinement::populateFaceParentFromParentFaces(ChildTag const initialChildTags[2][4]) {

    if (_uniform) {
        Index cFace = getFirstChildFaceFromFaces();
        for (Index pFace = 0; pFace < _parent->getNumFaces(); ++pFace) {
            IndexArray pVerts = _parent->getFaceVertices(pFace);

            //  Make this dependent on parent face's child count not its face-verts
            assert(_quadSplit);
            if (pVerts.size() == 4) {
                _childFaceTag[cFace + 0] = initialChildTags[0][0];
                _childFaceTag[cFace + 1] = initialChildTags[0][1];
                _childFaceTag[cFace + 2] = initialChildTags[0][2];
                _childFaceTag[cFace + 3] = initialChildTags[0][3];

                _childFaceParentIndex[cFace + 0] = pFace;
                _childFaceParentIndex[cFace + 1] = pFace;
                _childFaceParentIndex[cFace + 2] = pFace;
                _childFaceParentIndex[cFace + 3] = pFace;

                cFace += 4;
            } else {
                for (int i = 0; i < pVerts.size(); ++i, ++cFace) {
                    _childFaceTag[cFace] = initialChildTags[0][i];
                    _childFaceParentIndex[cFace] = pFace;
                }
            }
        }
    } else {
        //  Child faces of faces:
        for (Index pFace = 0; pFace < _parent->getNumFaces(); ++pFace) {
            bool incomplete = !_parentFaceTag[pFace]._selected;

            IndexArray cFaces = getFaceChildFaces(pFace);
            assert(_quadSplit);
            if (!incomplete && (cFaces.size() == 4)) {
                _childFaceTag[cFaces[0]] = initialChildTags[0][0];
                _childFaceTag[cFaces[1]] = initialChildTags[0][1];
                _childFaceTag[cFaces[2]] = initialChildTags[0][2];
                _childFaceTag[cFaces[3]] = initialChildTags[0][3];

                _childFaceParentIndex[cFaces[0]] = pFace;
                _childFaceParentIndex[cFaces[1]] = pFace;
                _childFaceParentIndex[cFaces[2]] = pFace;
                _childFaceParentIndex[cFaces[3]] = pFace;
            } else {
                for (int i = 0; i < cFaces.size(); ++i) {
                    if (IndexIsValid(cFaces[i])) {
                        _childFaceTag[cFaces[i]] = initialChildTags[incomplete][i];
                        _childFaceParentIndex[cFaces[i]] = pFace;
                    }
                }
            }
        }
    }
}

void
Refinement::populateEdgeParentVectors(ChildTag const initialChildTags[2][4]) {

    _childEdgeTag.resize(_child->getNumEdges());
    _childEdgeParentIndex.resize(_child->getNumEdges());

    populateEdgeParentFromParentFaces(initialChildTags);
    populateEdgeParentFromParentEdges(initialChildTags);
}
void
Refinement::populateEdgeParentFromParentFaces(ChildTag const initialChildTags[2][4]) {

    if (_uniform) {
        Index cEdge = getFirstChildEdgeFromFaces();
        for (Index pFace = 0; pFace < _parent->getNumFaces(); ++pFace) {
            IndexArray pVerts = _parent->getFaceVertices(pFace);

            //  Make this dependent on parent face's child count not its face-verts
            assert(_quadSplit);
            if (pVerts.size() == 4) {
                _childEdgeTag[cEdge + 0] = initialChildTags[0][0];
                _childEdgeTag[cEdge + 1] = initialChildTags[0][1];
                _childEdgeTag[cEdge + 2] = initialChildTags[0][2];
                _childEdgeTag[cEdge + 3] = initialChildTags[0][3];

                _childEdgeParentIndex[cEdge + 0] = pFace;
                _childEdgeParentIndex[cEdge + 1] = pFace;
                _childEdgeParentIndex[cEdge + 2] = pFace;
                _childEdgeParentIndex[cEdge + 3] = pFace;

                cEdge += 4;
            } else {
                for (int i = 0; i < pVerts.size(); ++i, ++cEdge) {
                    _childEdgeTag[cEdge] = initialChildTags[0][i];
                    _childEdgeParentIndex[cEdge] = pFace;
                }
            }
        }
    } else {
        for (Index pFace = 0; pFace < _parent->getNumFaces(); ++pFace) {
            bool incomplete = !_parentFaceTag[pFace]._selected;

            IndexArray cEdges = getFaceChildEdges(pFace);
            assert(_quadSplit);
            if (!incomplete && (cEdges.size() == 4)) {
                _childEdgeTag[cEdges[0]] = initialChildTags[0][0];
                _childEdgeTag[cEdges[1]] = initialChildTags[0][1];
                _childEdgeTag[cEdges[2]] = initialChildTags[0][2];
                _childEdgeTag[cEdges[3]] = initialChildTags[0][3];

                _childEdgeParentIndex[cEdges[0]] = pFace;
                _childEdgeParentIndex[cEdges[1]] = pFace;
                _childEdgeParentIndex[cEdges[2]] = pFace;
                _childEdgeParentIndex[cEdges[3]] = pFace;
            } else {
                for (int i = 0; i < cEdges.size(); ++i) {
                    if (IndexIsValid(cEdges[i])) {
                        _childEdgeTag[cEdges[i]] = initialChildTags[incomplete][i];
                        _childEdgeParentIndex[cEdges[i]] = pFace;
                    }
                }
            }
        }
    }
}
void
Refinement::populateEdgeParentFromParentEdges(ChildTag const initialChildTags[2][4]) {

    if (_uniform) {
        Index cEdge = getFirstChildEdgeFromEdges();
        for (Index pEdge = 0; pEdge < _parent->getNumEdges(); ++pEdge, cEdge += 2) {
            _childEdgeTag[cEdge + 0] = initialChildTags[0][0];
            _childEdgeTag[cEdge + 1] = initialChildTags[0][1];

            _childEdgeParentIndex[cEdge + 0] = pEdge;
            _childEdgeParentIndex[cEdge + 1] = pEdge;
        }
    } else {
        for (Index pEdge = 0; pEdge < _parent->getNumEdges(); ++pEdge) {
            bool incomplete = !_parentEdgeTag[pEdge]._selected;

            IndexArray cEdges = getEdgeChildEdges(pEdge);
            if (!incomplete) {
                _childEdgeTag[cEdges[0]] = initialChildTags[0][0];
                _childEdgeTag[cEdges[1]] = initialChildTags[0][1];

                _childEdgeParentIndex[cEdges[0]] = pEdge;
                _childEdgeParentIndex[cEdges[1]] = pEdge;
            } else {
                for (int i = 0; i < 2; ++i) {
                    if (IndexIsValid(cEdges[i])) {
                        _childEdgeTag[cEdges[i]] = initialChildTags[incomplete][i];
                        _childEdgeParentIndex[cEdges[i]] = pEdge;
                    }
                }
            }
        }
    }
}

void
Refinement::populateVertexParentVectors(ChildTag const initialChildTags[2][4]) {

    if (_uniform) {
        _childVertexTag.resize(_child->getNumVertices(), initialChildTags[0][0]);
    } else {
        _childVertexTag.resize(_child->getNumVertices(), initialChildTags[1][0]);
    }
    _childVertexParentIndex.resize(_child->getNumVertices());

    populateVertexParentFromParentFaces(initialChildTags);
    populateVertexParentFromParentEdges(initialChildTags);
    populateVertexParentFromParentVertices(initialChildTags);
}
void
Refinement::populateVertexParentFromParentFaces(ChildTag const initialChildTags[2][4]) {

    if (_uniform) {
        Index cVert = getFirstChildVertexFromFaces();
        for (Index pFace = 0; pFace < _parent->getNumFaces(); ++pFace, ++cVert) {
            //  Child tag was initialized as the complete and only child when allocated

            _childVertexParentIndex[cVert] = pFace;
        }
    } else {
        ChildTag const & completeChildTag = initialChildTags[0][0];

        for (Index pFace = 0; pFace < _parent->getNumFaces(); ++pFace) {
            Index cVert = _faceChildVertIndex[pFace];
            if (IndexIsValid(cVert)) {
                //  Child tag was initialized as incomplete -- reset if complete:
                if (_parentFaceTag[pFace]._selected) {
                    _childVertexTag[cVert] = completeChildTag;
                }
                _childVertexParentIndex[cVert] = pFace;
            }
        }
    }
}
void
Refinement::populateVertexParentFromParentEdges(ChildTag const initialChildTags[2][4]) {

    if (_uniform) {
        Index cVert = getFirstChildVertexFromEdges();
        for (Index pEdge = 0; pEdge < _parent->getNumEdges(); ++pEdge, ++cVert) {
            //  Child tag was initialized as the complete and only child when allocated

            _childVertexParentIndex[cVert] = pEdge;
        }
    } else {
        ChildTag const & completeChildTag = initialChildTags[0][0];

        for (Index pEdge = 0; pEdge < _parent->getNumEdges(); ++pEdge) {
            Index cVert = _edgeChildVertIndex[pEdge];
            if (IndexIsValid(cVert)) {
                //  Child tag was initialized as incomplete -- reset if complete:
                if (_parentEdgeTag[pEdge]._selected) {
                    _childVertexTag[cVert] = completeChildTag;
                }
                _childVertexParentIndex[cVert] = pEdge;
            }
        }
    }
}
void
Refinement::populateVertexParentFromParentVertices(ChildTag const initialChildTags[2][4]) {

    if (_uniform) {
        Index cVert = getFirstChildVertexFromVertices();
        for (Index pVert = 0; pVert < _parent->getNumVertices(); ++pVert, ++cVert) {
            //  Child tag was initialized as the complete and only child when allocated

            _childVertexParentIndex[cVert] = pVert;
        }
    } else {
        ChildTag const & completeChildTag = initialChildTags[0][0];

        for (Index pVert = 0; pVert < _parent->getNumVertices(); ++pVert) {
            Index cVert = _vertChildVertIndex[pVert];
            if (IndexIsValid(cVert)) {
                //  Child tag was initialized as incomplete but these should be complete:
                _childVertexTag[cVert] = completeChildTag;
                _childVertexParentIndex[cVert] = pVert;
            }
        }
    }
}


//
//  Methods to propagate/initialize child component tags from their parent component:
//
void
Refinement::propagateComponentTags() {

    populateFaceTagVectors();
    populateEdgeTagVectors();
    populateVertexTagVectors();
}

void
Refinement::populateFaceTagVectors() {

    _child->_faceTags.resize(_child->getNumFaces());

    populateFaceTagsFromParentFaces();
}
void
Refinement::populateFaceTagsFromParentFaces() {

    //
    //  Tags for faces originating from faces are inherited from the parent face:
    //
    Index cFace    = getFirstChildFaceFromFaces();
    Index cFaceEnd = cFace + getNumChildFacesFromFaces();
    for ( ; cFace < cFaceEnd; ++cFace) {
        _child->_faceTags[cFace] = _parent->_faceTags[_childFaceParentIndex[cFace]];
    }
}

void
Refinement::populateEdgeTagVectors() {

    _child->_edgeTags.resize(_child->getNumEdges());

    populateEdgeTagsFromParentFaces();
    populateEdgeTagsFromParentEdges();
}
void
Refinement::populateEdgeTagsFromParentFaces() {

    //
    //  Tags for edges originating from faces are all constant:
    //
    Level::ETag eTag;
    eTag._nonManifold = 0;
    eTag._boundary    = 0;
    eTag._infSharp    = 0;
    eTag._semiSharp   = 0;

    Index cEdge    = getFirstChildEdgeFromFaces();
    Index cEdgeEnd = cEdge + getNumChildEdgesFromFaces();
    for ( ; cEdge < cEdgeEnd; ++cEdge) {
        _child->_edgeTags[cEdge] = eTag;
    }
}
void
Refinement::populateEdgeTagsFromParentEdges() {

    //
    //  Tags for edges originating from edges are inherited from the parent edge:
    //
    Index cEdge    = getFirstChildEdgeFromEdges();
    Index cEdgeEnd = cEdge + getNumChildEdgesFromEdges();
    for ( ; cEdge < cEdgeEnd; ++cEdge) {
        _child->_edgeTags[cEdge] = _parent->_edgeTags[_childEdgeParentIndex[cEdge]];
    }
}

void
Refinement::populateVertexTagVectors() {

    _child->_vertTags.resize(_child->getNumVertices());

    populateVertexTagsFromParentFaces();
    populateVertexTagsFromParentEdges();
    populateVertexTagsFromParentVertices();

    if (!_uniform) {
        for (Index cVert = 0; cVert < _child->getNumVertices(); ++cVert) {
            if (_childVertexTag[cVert]._incomplete) {
                _child->_vertTags[cVert]._incomplete = true;
            }
        }
    }
}
void
Refinement::populateVertexTagsFromParentFaces() {

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
    vTag._incomplete  = 0;

    Index cVert    = getFirstChildVertexFromFaces();
    Index cVertEnd = cVert + getNumChildVerticesFromFaces();

    if (_parent->_depth > 0) {
        for ( ; cVert < cVertEnd; ++cVert) {
            _child->_vertTags[cVert] = vTag;
        }
    } else {
        int regFaceVertCount = _quadSplit ? 4 : 3;

        for ( ; cVert < cVertEnd; ++cVert) {
            _child->_vertTags[cVert] = vTag;

            if (_parent->getNumFaceVertices(_childVertexParentIndex[cVert]) != regFaceVertCount) {
                _child->_vertTags[cVert]._xordinary = true;
            }
        }
    }
}
void
Refinement::populateVertexTagsFromParentEdges() {

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
        cVertTag._incomplete = 0;
    }
}
void
Refinement::populateVertexTagsFromParentVertices() {

    //
    //  Tags for vertices originating from vertices are inherited from the parent vertex:
    //
    Index cVert    = getFirstChildVertexFromVertices();
    Index cVertEnd = cVert + getNumChildVerticesFromVertices();
    for ( ; cVert < cVertEnd; ++cVert) {
        _child->_vertTags[cVert] = _parent->_vertTags[_childVertexParentIndex[cVert]];
    }
}



//
//  Methods to subdivide the topology:
//
//  The main method to subdivide topology is fairly simple -- given a set of relations
//  to populate it simply tests and populates each relation separately.  The method for
//  each relation is responsible for appropriate allocation and initialization of all
//  data involved.
//
void
Refinement::subdivideTopology(Relations const& applyTo) {

    if (applyTo._faceVertices) {
        populateFaceVertexRelation();
    }
    if (applyTo._faceEdges) {
        populateFaceEdgeRelation();
    }
    if (applyTo._edgeVertices) {
        populateEdgeVertexRelation();
    }
    if (applyTo._edgeFaces) {
        populateEdgeFaceRelation();
    }
    if (applyTo._vertexFaces) {
        populateVertexFaceRelation();
    }
    if (applyTo._vertexEdges) {
        populateVertexEdgeRelation();
    }

    //
    //  Additional members of the child Level not specific to any relation...
    //      - note in the case of max-valence, the child's max-valence may be less
    //  than the parent if that maximal parent vertex was not included in the sparse
    //  refinement (possible when sparse refinement is more general).
    //
    _child->_maxValence = _parent->_maxValence;
}


//
//  Methods to populate the face-vertex relation of the child Level:
//      - child faces only originate from parent faces
//
void
Refinement::populateFaceVertexRelation() {

    //  Both face-vertex and face-edge share the face-vertex counts/offsets, so be sure
    //  not to re-initialize it if already done:
    //
    if (_child->_faceVertCountsAndOffsets.size() == 0) {
        populateFaceVertexCountsAndOffsets();
    }
    _child->_faceVertIndices.resize(_child->getNumFaces() * (_quadSplit ? 4 : 3));

    populateFaceVerticesFromParentFaces();
}

void
Refinement::populateFaceVertexCountsAndOffsets() {

    Level& child = *_child;

    //
    //  Be aware of scheme-specific decisions here, e.g. the current use
    //  of 4 for quads for Catmark -- must adjust for Loop, Bilinear and
    //  account for possibility of both quads and tris...
    //
    child._faceVertCountsAndOffsets.resize(child.getNumFaces() * 2);
    if (_quadSplit) {
        for (int i = 0; i < child.getNumFaces(); ++i) {
            child._faceVertCountsAndOffsets[i*2 + 0] = 4;
            child._faceVertCountsAndOffsets[i*2 + 1] = i << 2;
        }
    } else {
        for (int i = 0; i < child.getNumFaces(); ++i) {
            child._faceVertCountsAndOffsets[i*2 + 0] = 3;
            child._faceVertCountsAndOffsets[i*2 + 1] = i * 3;
        }
    }
}

void
Refinement::populateFaceVerticesFromParentFaces() {

    //
    //  Algorithm:
    //    - iterate through parent face-child-face vector (could use back-vector)
    //    - use parent components incident the parent face:
    //        - use the interior face-vert, corner vert-vert and two edge-verts
    //
    assert(_quadSplit);
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


//
//  Methods to populate the face-vertex relation of the child Level:
//      - child faces only originate from parent faces
//
void
Refinement::populateFaceEdgeRelation() {

    //  Both face-vertex and face-edge share the face-vertex counts/offsets, so be sure
    //  not to re-initialize it if already done:
    //
    if (_child->_faceVertCountsAndOffsets.size() == 0) {
        populateFaceVertexCountsAndOffsets();
    }
    _child->_faceEdgeIndices.resize(_child->getNumFaces() * (_quadSplit ? 4 : 3));

    populateFaceEdgesFromParentFaces();
}

void
Refinement::populateFaceEdgesFromParentFaces() {

    //
    //  Algorithm:
    //    - iterate through parent face-child-face vector (could use back-vector)
    //    - use parent components incident the parent face:
    //        - use the two interior face-edges and the two boundary edge-edges
    //
    assert(_quadSplit);
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
//  Methods to populate the edge-vertex relation of the child Level:
//      - child edges originate from parent faces and edges
//
void
Refinement::populateEdgeVertexRelation() {

    _child->_edgeVertIndices.resize(_child->getNumEdges() * 2);

    populateEdgeVerticesFromParentFaces();
    populateEdgeVerticesFromParentEdges();
}

void
Refinement::populateEdgeVerticesFromParentFaces() {

    //
    //  For each parent face's edge-children:
    //    - identify parent face's vert-child (note it is shared by all)
    //    - identify parent edge perpendicular to face's child edge:
    //        - identify parent edge's vert-child
    //
    assert(_quadSplit);
    for (Index pFace = 0; pFace < _parent->getNumFaces(); ++pFace) {
        IndexArray const pFaceEdges      = _parent->getFaceEdges(pFace);
        IndexArray const pFaceChildEdges = getFaceChildEdges(pFace);

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
//  Methods to populate the edge-face relation of the child Level:
//      - child edges originate from parent faces and edges
//      - sparse refinement poses challenges with allocation here
//          - we need to update the counts/offsets as we populate
//
void
Refinement::populateEdgeFaceRelation() {

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
Refinement::populateEdgeFacesFromParentFaces() {

    //
    //  Note -- the edge-face counts/offsets vector is not known
    //  ahead of time and is populated incrementally, so we cannot
    //  thread this yet...
    //
    assert(_quadSplit);
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
    assert(_quadSplit);
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

                IndexArray const pFaceEdges = _parent->getFaceEdges(pFace);
                IndexArray const pFaceVerts = _parent->getFaceVertices(pFace);

                IndexArray const pFaceChildren = getFaceChildFaces(pFace);

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
Refinement::populateVertexFaceRelation() {

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
    int childVertFaceIndexSizeEstimate;
    if (_quadSplit) {
        childVertFaceIndexSizeEstimate = (int)parent._faceVertIndices.size()
                                       + (int)parent._edgeFaceIndices.size() * 2
                                       + (int)parent._vertFaceIndices.size();
    } else {
        childVertFaceIndexSizeEstimate = (int)parent._edgeFaceIndices.size() * 3
                                       + (int)parent._vertFaceIndices.size();
    }

    child._vertFaceCountsAndOffsets.resize(child.getNumVertices() * 2);
    child._vertFaceIndices.resize(         childVertFaceIndexSizeEstimate);
    child._vertFaceLocalIndices.resize(    childVertFaceIndexSizeEstimate);

    if (getFirstChildVertexFromFaces() == 0) {
        populateVertexFacesFromParentFaces();
        populateVertexFacesFromParentEdges();
        populateVertexFacesFromParentVertices();
    } else {
        populateVertexFacesFromParentVertices();
        populateVertexFacesFromParentFaces();
        populateVertexFacesFromParentEdges();
    }

    //  Revise the over-allocated estimate based on what is used (as indicated in the
    //  count/offset for the last vertex) and trim the index vectors accordingly:
    childVertFaceIndexSizeEstimate = child.getNumVertexFaces(child.getNumVertices()-1) +
                                     child.getOffsetOfVertexFaces(child.getNumVertices()-1);
    child._vertFaceIndices.resize(     childVertFaceIndexSizeEstimate);
    child._vertFaceLocalIndices.resize(childVertFaceIndexSizeEstimate);
}

void
Refinement::populateVertexFacesFromParentFaces() {

    const Level& parent = *this->_parent;
          Level& child  = *this->_child;

    assert(_quadSplit);
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

    assert(_quadSplit);
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
            //  EDGE_IN_FACE:
            //      Identify the parent edge within this parent face -- this is where
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
                assert(_quadSplit);
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
Refinement::populateVertexEdgeRelation() {

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
    int childVertEdgeIndexSizeEstimate;
    if (_quadSplit) {
        childVertEdgeIndexSizeEstimate = (int)parent._faceVertIndices.size()
                                       + (int)parent._edgeFaceIndices.size() + parent.getNumEdges() * 2
                                       + (int)parent._vertEdgeIndices.size();
    } else {
        childVertEdgeIndexSizeEstimate = (int)parent._edgeFaceIndices.size() * 2 + parent.getNumEdges() * 2
                                       + (int)parent._vertEdgeIndices.size();
    }

    child._vertEdgeCountsAndOffsets.resize(child.getNumVertices() * 2);
    child._vertEdgeIndices.resize(         childVertEdgeIndexSizeEstimate);
    child._vertEdgeLocalIndices.resize(    childVertEdgeIndexSizeEstimate);

    if (getFirstChildVertexFromFaces() == 0) {
        populateVertexEdgesFromParentFaces();
        populateVertexEdgesFromParentEdges();
        populateVertexEdgesFromParentVertices();
    } else {
        populateVertexEdgesFromParentVertices();
        populateVertexEdgesFromParentFaces();
        populateVertexEdgesFromParentEdges();
    }

    //  Revise the over-allocated estimate based on what is used (as indicated in the
    //  count/offset for the last vertex) and trim the index vectors accordingly:
    childVertEdgeIndexSizeEstimate = child.getNumVertexEdges(child.getNumVertices()-1) +
                                     child.getOffsetOfVertexEdges(child.getNumVertices()-1);
    child._vertEdgeIndices.resize(     childVertEdgeIndexSizeEstimate);
    child._vertEdgeLocalIndices.resize(childVertEdgeIndexSizeEstimate);
}

void
Refinement::populateVertexEdgesFromParentFaces() {

    const Level& parent = *this->_parent;
          Level& child  = *this->_child;

    assert(_quadSplit);
    for (int fIndex = 0; fIndex < parent.getNumFaces(); ++fIndex) {
        int cVertIndex = this->_faceChildVertIndex[fIndex];
        if (!IndexIsValid(cVertIndex)) continue;

        //
        //  Inspect the parent face first:
        //
        IndexArray const pFaceVerts = parent.getFaceVertices(fIndex);

        IndexArray const pFaceChildEdges = this->getFaceChildEdges(fIndex);

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
Refinement::populateVertexEdgesFromParentEdges() {

    const Level& parent = *this->_parent;
          Level& child  = *this->_child;

    assert(_quadSplit);
    for (int eIndex = 0; eIndex < parent.getNumEdges(); ++eIndex) {
        int cVertIndex = this->_edgeChildVertIndex[eIndex];
        if (!IndexIsValid(cVertIndex)) continue;

        //
        //  First inspect the parent edge -- its parent faces then its child edges:
        //
        IndexArray const pEdgeFaces      = parent.getEdgeFaces(eIndex);
        IndexArray const pEdgeChildEdges = this->getEdgeChildEdges(eIndex);

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

            IndexArray const pFaceEdges      = parent.getFaceEdges(pFace);
            IndexArray const pFaceChildEdges = this->getFaceChildEdges(pFace);

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
//  Methods to subdivide sharpness values:
//
void
Refinement::subdivideSharpnessValues() {

    //
    //  Subdividing edge and vertex sharpness values are independent, but in order
    //  to maintain proper classification/tagging of components as semi-sharp, both
    //  must be computed and the neighborhood inspected to properly update the
    //  status.
    //
    //  It is possible to clear the semi-sharp status when propagating the tags and
    //  to reset it (potentially multiple times) when updating the sharpness values.
    //  The vertex subdivision Rule is also affected by this, which complicates the
    //  process.  So for now we apply a post-process to explicitly handle all
    //  semi-sharp vertices.
    //
    subdivideEdgeSharpness();
    subdivideVertexSharpness();

    reclassifySemisharpVertices();
}

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
    float * pVertEdgeSharpness = 0;
    if (!creasing.IsUniform()) {
        pVertEdgeSharpness = (float *)alloca(_parent->getMaxValence() * sizeof(float));
    }

    Index cEdge    = getFirstChildEdgeFromEdges();
    Index cEdgeEnd = cEdge + getNumChildEdgesFromEdges();
    for ( ; cEdge < cEdgeEnd; ++cEdge) {
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
    Index cVertBegin = getFirstChildVertexFromVertices();
    Index cVertEnd   = cVertBegin + getNumChildVerticesFromVertices();

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
    Index vertFromEdgeBegin = getFirstChildVertexFromEdges();
    Index vertFromEdgeEnd   = vertFromEdgeBegin + getNumChildVerticesFromEdges();

    for (Index cVert = vertFromEdgeBegin; cVert < vertFromEdgeEnd; ++cVert) {
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
    Index vertFromVertBegin = getFirstChildVertexFromVertices();
    Index vertFromVertEnd   = vertFromVertBegin + getNumChildVerticesFromVertices();

    for (Index cVert = vertFromVertBegin; cVert < vertFromVertEnd; ++cVert) {
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

//
//  Methods to subdivide face-varying channels:
//
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
Refinement::markSparseChildComponentIndices() {

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

} // end namespace Vtr

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv
