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
#include "../vtr/level.h"

#include "../vtr/fvarLevel.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <algorithm>


//
//  FVarLevel:
//      Simple container of face-varying topology, associated with a particular
//  level.  It is typically constructed and initialized similarly to levels -- the
//  base level in a Factory and subsequent levels by refinement.
//
namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Vtr {

//
//  Simple (for now) constructor and destructor:
//
FVarLevel::FVarLevel(Level const& level) :
    _level(level),
    _isLinear(false),
    _hasSmoothBoundaries(false),
    _hasDependentSharpness(false),
    _valueCount(0) {
}

FVarLevel::~FVarLevel() {
}

//
//  Initialization and sizing methods to allocate space:
//
void
FVarLevel::setOptions(Sdc::Options const& options) {
    _options = options;
}

void
FVarLevel::resizeComponents() {

    //  Per-face members:
    _faceVertValues.resize(_level.getNumFaceVerticesTotal());

    //  Per-edge members:
    ETag edgeTagMatch;
    edgeTagMatch.clear();
    _edgeTags.resize(_level.getNumEdges(), edgeTagMatch);

    //  Per-vertex members:
    _vertSiblingCounts.resize(_level.getNumVertices());
    _vertSiblingOffsets.resize(_level.getNumVertices());

    _vertFaceSiblings.resize(_level.getNumVertexFacesTotal(), 0);
}

void
FVarLevel::resizeVertexValues(int vertexValueCount) {

    _vertValueIndices.resize(vertexValueCount);

    ValueTag valueTagMatch;
    valueTagMatch.clear();
    _vertValueTags.resize(vertexValueCount, valueTagMatch);

    if (_hasSmoothBoundaries) {
        _vertValueCreaseEnds.resize(vertexValueCount);
    }
}

void
FVarLevel::resizeValues(int valueCount) {
    _valueCount = valueCount;
}


//
//  Initialize the component tags once all face-values have been assigned...
//
//  Constructing the mapping between vertices and their face-varying values involves:
//
//      - iteration through all vertices to mark edge discontinuities and classify
//      - allocation of vectors mapping vertices to their multiple (sibling) values
//      - iteration through all vertices and their distinct values to tag topologically
//
//  Once values have been identified for each vertex and tagged, refinement propagates
//  the tags to child values using more simplified logic (child values inherit the
//  topology of their parent) and no futher analysis is required.
//
void
FVarLevel::completeTopologyFromFaceValues() {

    //
    //  Assign some members and local variables based on the interpolation options (the
    //  members support queries that are expected later):
    //
    //  Given the growing number of options and behaviors to support, this is likely going
    //  to get another pass.  It may be worth identifying the behavior for each "feature",
    //  i.e. determine smooth or sharp for corners, creases and darts, but the fact that
    //  the rule for one value may be dependent on that of another complicates this.
    //
    using Sdc::Options;

    Options::VtxBoundaryInterpolation geomOptions = _options.GetVtxBoundaryInterpolation();
    Options::FVarLinearInterpolation   fvarOptions = _options.GetFVarLinearInterpolation();

    _isLinear = (fvarOptions == Options::FVAR_LINEAR_ALL);

    _hasSmoothBoundaries = (fvarOptions != Options::FVAR_LINEAR_ALL) &&
                           (fvarOptions != Options::FVAR_LINEAR_BOUNDARIES);

    _hasDependentSharpness = (fvarOptions == Options::FVAR_LINEAR_CORNERS_PLUS1) ||
                             (fvarOptions == Options::FVAR_LINEAR_CORNERS_PLUS2);

    bool geomCornersAreSmooth = (geomOptions != Options::VTX_BOUNDARY_EDGE_AND_CORNER);
    bool fvarCornersAreSharp  = (fvarOptions != Options::FVAR_LINEAR_NONE);

    bool makeSmoothCornersSharp = geomCornersAreSmooth && fvarCornersAreSharp;

    bool sharpenBothIfOneCorner  = (fvarOptions == Options::FVAR_LINEAR_CORNERS_PLUS2);

    bool sharpenDarts = sharpenBothIfOneCorner || !_hasSmoothBoundaries;

    //
    //  Its awkward and potentially inefficient to try and accomplish everything in one
    //  pass over the vertices...
    //
    //  Make a first pass through the vertices to identify discts edges and to determine
    //  the number of values-per-vertex for subsequent allocation.  The presence of a
    //  discts edge warrants marking vertices at BOTH ends as having mismatched topology
    //  wrt the vertices (part of why full topological analysis is deferred).
    //
    //  So this first pass will allocate/initialize the overall structure of the topology.
    //  Given N vertices and M (as yet unknown) sibling values, the first pass achieves
    //  the following:
    //
    //      - assigns a local vector indicating which of the N vertices "match"
    //          - requires a single value but must also have no discts incident edges
    //      - determines the number of values associated with each of the N vertices
    //      - assigns an offset to the first value for each of the N vertices
    //      - initializes the vert-face "siblings" for all N vertices
    //  and
    //      - tags any incident edges as discts
    //
    //  The second pass initializes remaining members based on the total number of siblings
    //  M after allocating appropriate vectors dependent on M.
    //
    std::vector<LocalIndex> vertexMismatch(_level.getNumVertices(), 0);

    _vertFaceSiblings.resize(_level.getNumVertexFacesTotal(), 0);

    int const maxValence = _level.getMaxValence();

    Index *   indexBuffer   = (Index *)alloca(maxValence*sizeof(Index));
    int *     valueBuffer   = (int *)alloca(maxValence*sizeof(int));
    Sibling * siblingBuffer = (Sibling *)alloca(maxValence*sizeof(Sibling));

    int * uniqueValues = valueBuffer;

    int totalValueCount = 0;
    for (int vIndex = 0; vIndex < _level.getNumVertices(); ++vIndex) {
        ConstIndexArray       vEdges  = _level.getVertexEdges(vIndex);
        ConstLocalIndexArray  vInEdge = _level.getVertexEdgeLocalIndices(vIndex);

        ConstIndexArray       vFaces  = _level.getVertexFaces(vIndex);
        ConstLocalIndexArray  vInFace = _level.getVertexFaceLocalIndices(vIndex);

        //  Store the value for each vert-face locally -- we will identify the index
        //  of its sibling as we inspect them:
        Index * vValues = indexBuffer;
        Sibling * vSiblings = siblingBuffer;
        for (int i = 0; i < vFaces.size(); ++i) {
            vValues[i] = _faceVertValues[_level.getOffsetOfFaceVertices(vFaces[i]) + vInFace[i]];
        }

        int uniqueValueCount = 0;
        uniqueValues[uniqueValueCount++] = vValues[0];
        vSiblings[0] = 0;

        bool vIsBoundary = _level._vertTags[vIndex]._boundary;
        for (int i = vIsBoundary; i < vFaces.size(); ++i) {
            int iPrev = i ? (i - 1) : (vFaces.size() - 1);

            if (vValues[iPrev] == vValues[i]) {
                //  Beware the logic here when setting vSiblings[]...
                if (i > 0) {
                    vSiblings[i] = vSiblings[iPrev];
                }
            } else {
                //  Tag the corresponding edge as discts:
                Index eIndex = vEdges[i];
                ETag& eTag   = _edgeTags[eIndex];

                if (vInEdge[i] == 0) {
                    eTag._disctsV0 = true;
                } else {
                    eTag._disctsV1 = true;
                }
                eTag._mismatch = true;

                //  Tag both end vertices as not matching topology:
                ConstIndexArray eVerts = _level.getEdgeVertices(eIndex);
                vertexMismatch[eVerts[0]] = true;
                vertexMismatch[eVerts[1]] = true;

                //  Add the "new" value if not already present:
                if (i > 0) {
                    //
                    //  Use the block in the second pass which has been simplified -- we need to
                    //  assign vSiblings[] here whether it matches or not...
                    //
                    vSiblings[i] = (Sibling) uniqueValueCount;

                    if (uniqueValueCount == 1) {
                        uniqueValues[uniqueValueCount++] = vValues[i];
                    } else if ((uniqueValueCount == 2) && (uniqueValues[0] != vValues[i])) {
                        uniqueValues[uniqueValueCount++] = vValues[i];
                    } else {
                        int* valueLast  = uniqueValues + uniqueValueCount;
                        int* valueFound = std::find(uniqueValues, valueLast, vValues[i]);
                        if (valueFound == valueLast) {
                            uniqueValues[uniqueValueCount++] = vValues[i];
                        } else {
                            vSiblings[i] = (Sibling) (valueFound - uniqueValues);
                        }
                    }
                }
            }
        }

        //
        //  While we've tagged the vertex as having mismatched FVar topology in the presence of
        //  any discts edges, we also need to account for different treatment of vertices along
        //  geometric boundaries if the FVar interpolation rules affect them:
        //
        if (vIsBoundary && !vertexMismatch[vIndex]) {
            if (vFaces.size() == 1) {
                if (makeSmoothCornersSharp) {
                    vertexMismatch[vIndex] = true;
                }
            } else if (!_hasSmoothBoundaries) {
                vertexMismatch[vIndex] = true;
            }
        }

        //
        //  Update the value count and offset for this vertex and cumulative totals:
        //
        _vertSiblingCounts[vIndex]  = (LocalIndex) uniqueValueCount;
        _vertSiblingOffsets[vIndex] = totalValueCount;

        totalValueCount += uniqueValueCount;

        //  Update the vert-face siblings from the local array above:
        if (uniqueValueCount > 1) {
            SiblingArray vFaceSiblings = getVertexFaceSiblings(vIndex);
            for (int i = 0; i < vFaces.size(); ++i) {
                vFaceSiblings[i] = vSiblings[i];
            }
        }
    }

    //
    //  Now that we know the total number of additional sibling values (M values in addition
    //  to the N vertex values) allocate space to accomodate all N + M vertex values.  The
    //  vertex value tags will be initialized to match, and we proceed to sparsely mark the
    //  vertices that mismatch, so initialize a few local ValueTag constants for that purpose
    //  (assigning entire Tag structs is much more efficient than setting individual bits)
    //
    resizeVertexValues(totalValueCount);

    ValueTag valueTagMismatch;
    valueTagMismatch.clear();
    valueTagMismatch._mismatch = true;

    ValueTag valueTagCrease = valueTagMismatch;
    valueTagCrease._crease = true;

    ValueTag valueTagSemiSharp = valueTagMismatch;
    valueTagSemiSharp._semiSharp = true;

    ValueTag valueTagDepSharp = valueTagSemiSharp;
    valueTagDepSharp._depSharp = true;

    //
    //  Now the second pass through the vertices to identify the values associated with the
    //  vertex and to inspect and tag local face-varying topology for those that don't match:
    //
    for (int vIndex = 0; vIndex < _level.getNumVertices(); ++vIndex) {
        ConstIndexArray       vFaces  = _level.getVertexFaces(vIndex);
        ConstLocalIndexArray  vInFace = _level.getVertexFaceLocalIndices(vIndex);

        //
        //  First step is to assign the values associated with the faces by retrieving them
        //  from the faces.  If the face-varying topology around this vertex matches the vertex
        //  topology, there is little more to do as other members were bulk-initialized to
        //  match, so we can continue immediately:
        //
        IndexArray vValues = getVertexValues(vIndex);

        vValues[0] = _faceVertValues[_level.getOffsetOfFaceVertices(vFaces[0]) + vInFace[0]];
        if (!vertexMismatch[vIndex]) {
            continue;
        }
        if (vValues.size() > 1) {
            ConstSiblingArray vFaceSiblings = getVertexFaceSiblings(vIndex);

            for (int i = 1, nextSibling = 1; i < vFaces.size(); ++i) {
                if (vFaceSiblings[i] == nextSibling) {
                    vValues[nextSibling++] = _faceVertValues[_level.getOffsetOfFaceVertices(vFaces[i]) + vInFace[i]];
                }
            }
        }

        //
        //  If all values for this vertex are to be designated as sharp, the value tags
        //  have already been initialized for this by default, so we can continue.  On
        //  further inspection there may be other cases where all are determined to be
        //  sharp, but use what information we can now to avoid that inspection:
        //
        //  Regarding sharpness of the vertex itself, its vertex tags reflect the inf-
        //  or semi-sharp nature of the vertex and edges around it, so be careful not
        //  to assume too much from say, the presence of an incident inf-sharp edge.
        //  We can make clear decisions based on the sharpness of the vertex itself.
        //
        ValueTagArray vValueTags = getVertexValueTags(vIndex);

        bool  vIsBoundary = _level._vertTags[vIndex]._boundary;
        float vSharpness  = _level._vertSharpness[vIndex];

        bool allCornersAreSharp = !_hasSmoothBoundaries ||
                                  Sdc::Crease::IsInfinite(vSharpness) ||
                                  (_hasDependentSharpness && (vValues.size() > 2)) ||
                                  (sharpenDarts && (vValues.size() == 1) && !vIsBoundary) ||
                                   _level._vertTags[vIndex]._nonManifold;
        if (allCornersAreSharp) {
            std::fill(vValueTags.begin(), vValueTags.end(), valueTagMismatch);
            continue;
        }

        //
        //  Values may be a mix of sharp corners and smooth boundaries -- start by
        //  gathering information about the "span" of faces for each value:
        //
        assert(sizeof(ValueSpan) <= sizeof(int));
        ValueSpan * vValueSpans = (ValueSpan *) indexBuffer;
        memset(vValueSpans, 0, vValues.size() * sizeof(ValueSpan));

        gatherValueSpans(vIndex, vValueSpans);

        //
        //  Spans are identified as sharp (disjoint) or smooth based on their own local
        //  topology, but depending on the specified options, the sharpness of one span
        //  may be dependent on the sharpness of another.  Mark both as infinitely sharp
        //  where possible to avoid re-assessing this dependency as sharpness is reduced
        //  during refinement.
        //
        allCornersAreSharp = false;

        bool hasDependentValuesToSharpen = false;
        if (_hasDependentSharpness && (vValues.size() == 2)) {
            //  Detect interior inf-sharp (or discts) edge:
            allCornersAreSharp = vValueSpans[0]._disjoint || vValueSpans[1]._disjoint;

            //  Detect a sharp corner, making both sharp:
            if (sharpenBothIfOneCorner) {
                allCornersAreSharp |= (vValueSpans[0]._size == 1) || (vValueSpans[1]._size == 1);
            }

            //  If only one semi-sharp, need to mark the other as dependent on it:
            hasDependentValuesToSharpen = vValueSpans[0]._semiSharp != vValueSpans[1]._semiSharp;
        }
        if (allCornersAreSharp) {
            std::fill(vValueTags.begin(), vValueTags.end(), valueTagMismatch);
            continue;
        }

        //
        //  Inspect each vertex value to determine if it is a smooth boundary (crease) and tag
        //  it accordingly.  If not semi-sharp, be sure to consider those values sharpened by
        //  the topology of other values.
        //
        CreaseEndPairArray vValueCreaseEnds = getVertexValueCreaseEnds(vIndex);

        for (int i = 0; i < vValues.size(); ++i) {
            ValueSpan const & vSpan = vValueSpans[i];

            if (vSpan._disjoint || ((vSpan._size == 1) && fvarCornersAreSharp)) {
                vValueTags[i] = valueTagMismatch;
            } else {
                if ((vSpan._semiSharp > 0) || Sdc::Crease::IsSharp(vSharpness)) {
                    vValueTags[i] = valueTagSemiSharp;
                } else if (hasDependentValuesToSharpen) {
                    vValueTags[i] = valueTagDepSharp;
                } else {
                    vValueTags[i] = valueTagCrease;
                }

                vValueCreaseEnds[i]._startFace = vSpan._start;
                if ((i == 0) && (vSpan._start != 0)) {
                    vValueCreaseEnds[i]._endFace = (LocalIndex) (vSpan._start + vSpan._size - 1 - vFaces.size());
                } else {
                    vValueCreaseEnds[i]._endFace = (LocalIndex) (vSpan._start + vSpan._size - 1);
                }
            }
        }
    }
    //printf("completed fvar topology...\n");
    //print();
    //printf("validating...\n");
    //assert(validate());
}

//
//  Values tagged as creases have their two "end values" identified relative to the incident
//  faces of the vertex for compact storage and quick retrieval.  This methods identifies the
//  values for the two ends of such a crease value:
//
void
FVarLevel::getVertexCreaseEndValues(Index vIndex, Sibling vSibling, Index endValues[2]) const {

    ConstCreaseEndPairArray vValueCreaseEnds = getVertexValueCreaseEnds(vIndex);

    ConstIndexArray      vFaces  = _level.getVertexFaces(vIndex);
    ConstLocalIndexArray vInFace = _level.getVertexFaceLocalIndices(vIndex);

    LocalIndex vertFace0 = vValueCreaseEnds[vSibling]._startFace;
    LocalIndex vertFace1 = vValueCreaseEnds[vSibling]._endFace;

    ConstIndexArray face0Values = getFaceValues(vFaces[vertFace0]);
    ConstIndexArray face1Values = getFaceValues(vFaces[vertFace1]);

    int endInFace0 = vInFace[vertFace0];
    int endInFace1 = vInFace[vertFace1];

    endInFace0 = (endInFace0 == (face0Values.size() - 1)) ? 0 : (endInFace0 + 1);
    endInFace1 = (endInFace1 ? endInFace1 : face1Values.size()) - 1;

    endValues[0] = face0Values[endInFace0];
    endValues[1] = face1Values[endInFace1];
}

//
//  Debugging aids...
//
bool
FVarLevel::validate() const {

    //
    //  Verify that member sizes match sizes for the associated level:
    //
    if ((int)_vertSiblingCounts.size() != _level.getNumVertices()) {
        printf("Error:  vertex count mismatch\n");
        return false;
    }
    if ((int)_edgeTags.size() != _level.getNumEdges()) {
        printf("Error:  edge count mismatch\n");
        return false;
    }
    if ((int)_faceVertValues.size() != _level.getNumFaceVerticesTotal()) {
        printf("Error:  face-value/face-vert count mismatch\n");
        return false;
    }
    if (_level.getDepth() > 0) {
        if (_valueCount != (int)_vertValueIndices.size()) {
            printf("Error:  value/vertex-value count mismatch\n");
            return false;
        }
    }

    //
    //  Verify that face-verts and (locally computed) face-vert siblings yield the
    //  expected face-vert values:
    //
    std::vector<Sibling> fvSiblingVector;
    buildFaceVertexSiblingsFromVertexFaceSiblings(fvSiblingVector);

    for (int fIndex = 0; fIndex < _level.getNumFaces(); ++fIndex) {
        ConstIndexArray     fVerts = _level.getFaceVertices(fIndex);
        ConstIndexArray    fValues = getFaceValues(fIndex);
        Sibling const  * fSiblings = &fvSiblingVector[_level.getOffsetOfFaceVertices(fIndex)];

        for (int fvIndex = 0; fvIndex < fVerts.size(); ++fvIndex) {
            Index vIndex = fVerts[fvIndex];

            Index   fvValue   = fValues[fvIndex];
            Sibling fvSibling = fSiblings[fvIndex];
            if (fvSibling >= getNumVertexValues(vIndex)) {
                printf("Error:  invalid sibling %d for face-vert %d.%d = %d\n", fvSibling, fIndex, fvIndex, vIndex);
                return false;
            }

            Index testValue = getVertexValue(vIndex, fvSibling);
            if (testValue != fvValue) {
                printf("Error:  unexpected value %d for sibling %d of face-vert %d.%d = %d (expecting %d)\n",
                        testValue, fvSibling, fIndex, fvIndex, vIndex, fvValue);
                return false;
            }
        }
    }

    //
    //  Verify that the vert-face siblings yield the expected value:
    //
    for (int vIndex = 0; vIndex < _level.getNumVertices(); ++vIndex) {
        ConstIndexArray      vFaces    = _level.getVertexFaces(vIndex);
        ConstLocalIndexArray vInFace   = _level.getVertexFaceLocalIndices(vIndex);
        ConstSiblingArray    vSiblings = getVertexFaceSiblings(vIndex);

        for (int j = 0; j < vFaces.size(); ++j) {
            Sibling vSibling = vSiblings[j];
            if (vSibling >= getNumVertexValues(vIndex)) {
                printf("Error:  invalid sibling %d at vert-face %d.%d\n", vSibling, vIndex, j);
                return false;
            }

            Index fIndex  = vFaces[j];
            int   fvIndex = vInFace[j];
            Index fvValue = getFaceValues(fIndex)[fvIndex];

            Index vValue = getVertexValue(vIndex, vSibling);
            if (vValue != fvValue) {
                printf("Error:  value mismatch between face-vert %d.%d and vert-face %d.%d (%d != %d)\n",
                        fIndex, fvIndex, vIndex, j, fvValue, vValue);
                return false;
            }
        }
    }
    return true;
}

void
FVarLevel::print() const {

    std::vector<Sibling> fvSiblingVector;
    buildFaceVertexSiblingsFromVertexFaceSiblings(fvSiblingVector);

    printf("Face-varying data channel:\n");
    printf("  Inventory:\n");
    printf("    vertex count       = %d\n", _level.getNumVertices());
    printf("    source value count = %d\n", _valueCount);
    printf("    vertex value count = %d\n", (int)_vertValueIndices.size());

    printf("  Face values:\n");
    for (int i = 0; i < _level.getNumFaces(); ++i) {
        ConstIndexArray  fVerts    = _level.getFaceVertices(i);
        ConstIndexArray  fValues   = getFaceValues(i);
        Sibling const  * fSiblings = &fvSiblingVector[_level.getOffsetOfFaceVertices(i)];

        printf("    face%4d:  ", i);

        printf("verts =");
        for (int j = 0; j < fVerts.size(); ++j) {
            printf("%4d", fVerts[j]);
        }
        printf(",  values =");
        for (int j = 0; j < fValues.size(); ++j) {
            printf("%4d", fValues[j]);
        }
        printf(",  siblings =");
        for (int j = 0; j < fVerts.size(); ++j) {
            printf("%4d", (int)fSiblings[j]);
        }
        printf("\n");
    }

    printf("  Vertex values:\n");
    for (int i = 0; i < _level.getNumVertices(); ++i) {
        int vCount  = getNumVertexValues(i);
        int vOffset = getVertexValueOffset(i);

        printf("    vert%4d:  vcount = %1d, voffset =%4d, ", i, vCount, vOffset);

        ConstIndexArray vValues = getVertexValues(i);

        printf("values =");
        for (int j = 0; j < vValues.size(); ++j) {
            printf("%4d", vValues[j]);
        }
        if (vCount > 1) {
            ConstValueTagArray vValueTags = getVertexValueTags(i);

            printf(", crease =");
            for (int j = 0; j < vValueTags.size(); ++j) {
                printf("%4d", vValueTags[j]._crease);
            }
            printf(", semi-sharp =");
            for (int j = 0; j < vValueTags.size(); ++j) {
                printf("%2d", vValueTags[j]._semiSharp);
            }
        }
        printf("\n");
    }

    printf("  Edge discontinuities:\n");
    for (int i = 0; i < _level.getNumEdges(); ++i) {
        ETag const eTag = getEdgeTag(i);
        if (eTag._mismatch) {
            ConstIndexArray eVerts = _level.getEdgeVertices(i);
            printf("    edge%4d:  verts = [%4d%4d], discts = [%d,%d]\n", i, eVerts[0], eVerts[1],
                    eTag._disctsV0, eTag._disctsV1);
        }
    }
}



void
FVarLevel::initializeFaceValuesFromFaceVertices() {

    Index const* srcFaceVerts  = &_level._faceVertIndices[0];
    Index *      dstFaceValues = &_faceVertValues[0];

    std::memcpy(dstFaceValues, srcFaceVerts, getNumFaceValuesTotal() * sizeof(Index));
}


void
FVarLevel::initializeFaceValuesFromVertexFaceSiblings() {
    //
    //  Iterate through all face-values first and initialize them with the first value
    //  associated with each face-vertex.  Then make a second sparse pass through the
    //  vertex-faces to offset those with multiple values.  This turns out to be much
    //  more efficient than a single iteration through the vertex-faces since the first
    //  pass is much more memory coherent.
    //
    int fvCount = (int) _level._faceVertIndices.size();
    for (int i = 0; i < fvCount; ++i) {
        _faceVertValues[i] = getVertexValueOffset(_level._faceVertIndices[i]);
    }

    //
    //  Now use the vert-face-siblings to populate the face-vert-values:
    //
    for (int vIndex = 0; vIndex < _level.getNumVertices(); ++vIndex) {
        if (getNumVertexValues(vIndex) > 1) {
            ConstIndexArray      vFaces    = _level.getVertexFaces(vIndex);
            ConstLocalIndexArray vInFace   = _level.getVertexFaceLocalIndices(vIndex);
            ConstSiblingArray    vSiblings = getVertexFaceSiblings(vIndex);

            for (int j = 0; j < vFaces.size(); ++j) {
                if (vSiblings[j]) {
                    int fvOffset = _level.getOffsetOfFaceVertices(vFaces[j]);

                    _faceVertValues[fvOffset + vInFace[j]] += vSiblings[j];
                }
            }
        }
    }
}

void
FVarLevel::buildFaceVertexSiblingsFromVertexFaceSiblings(std::vector<Sibling>& fvSiblings) const {

    fvSiblings.resize(_level.getNumFaceVerticesTotal());
    std::memset(&fvSiblings[0], 0, _level.getNumFaceVerticesTotal() * sizeof(Sibling));

    for (int vIndex = 0; vIndex < _level.getNumVertices(); ++vIndex) {
        //  We can skip cases of one sibling as we initialized to 0...
        if (getNumVertexValues(vIndex) > 1) {
            ConstIndexArray      vFaces    = _level.getVertexFaces(vIndex);
            ConstLocalIndexArray vInFace   = _level.getVertexFaceLocalIndices(vIndex);
            ConstSiblingArray    vSiblings = getVertexFaceSiblings(vIndex);

            for (int j = 0; j < vFaces.size(); ++j) {
                if (vSiblings[j] > 0) {
                    fvSiblings[_level.getOffsetOfFaceVertices(vFaces[j]) + vInFace[j]] = vSiblings[j];
                }
            }
        }
    }
}


//
//  Higher-level topological queries, i.e. values in a neighborhood:
//    - given an edge, return values corresponding to its vertices within a given face
//    - given a vertex, return values corresponding to verts at the ends of its edges
//
void
FVarLevel::getEdgeFaceValues(Index eIndex, int fIncToEdge, Index valuesPerVert[2]) const {

    ConstIndexArray eVerts = _level.getEdgeVertices(eIndex);
    if ((getNumVertexValues(eVerts[0]) > 1) || (getNumVertexValues(eVerts[1]) > 1)) {
        Index eFace = _level.getEdgeFaces(eIndex)[fIncToEdge];

        //  This is another of those irritating times where I want to have the edge-in-face
        //  local indices stored with each edge-face...
        //
        //  This is about as fast as we're going to get with a search and still leaves
        //  this function showing a considerable part of edge-vertex interpolation (down
        //  from 40% to 25%)...
        //
        ConstIndexArray fEdges  = _level.getFaceEdges(eFace);
        ConstIndexArray fValues = getFaceValues(eFace);

        int edgeInFace = 0;
        if (fEdges.size() == 4) {
            if (fEdges[0] == eIndex) {
                valuesPerVert[0] = fValues[0];
                valuesPerVert[1] = fValues[1];
                edgeInFace = 0;
            } else if (fEdges[1] == eIndex) {
                valuesPerVert[0] = fValues[1];
                valuesPerVert[1] = fValues[2];
                edgeInFace = 1;
            } else if (fEdges[2] == eIndex) {
                valuesPerVert[0] = fValues[2];
                valuesPerVert[1] = fValues[3];
                edgeInFace = 2;
            } else {
                valuesPerVert[0] = fValues[3];
                valuesPerVert[1] = fValues[0];
                edgeInFace = 3;
            }
        } else if (fEdges.size() == 3) {
            if (fEdges[0] == eIndex) {
                valuesPerVert[0] = fValues[0];
                valuesPerVert[1] = fValues[1];
                edgeInFace = 0;
            } else if (fEdges[1] == eIndex) {
                valuesPerVert[0] = fValues[1];
                valuesPerVert[1] = fValues[2];
                edgeInFace = 1;
            } else {
                valuesPerVert[0] = fValues[2];
                valuesPerVert[1] = fValues[0];
                edgeInFace = 2;
            }
        } else {
            for (int i = 0; i < fEdges.size(); ++i) {
                if (fEdges[i] == eIndex) {
                    valuesPerVert[0] = fValues[i];
                    valuesPerVert[1] = fValues[(i+1) % fEdges.size()];
                    edgeInFace = i;
                    break;
                }
            }
        }

        //  Given the way these two end-values are used (both weights the same) we really
        //  don't need to ensure the value pair matches the vertex pair...
        if (eVerts[0] != _level.getFaceVertices(eFace)[edgeInFace]) {
            std::swap(valuesPerVert[0], valuesPerVert[1]);
        }
    } else {
        valuesPerVert[0] = getVertexValue(eVerts[0]);
        valuesPerVert[1] = getVertexValue(eVerts[1]);
    }
}

void
FVarLevel::getVertexEdgeValues(Index vIndex, Index valuesPerEdge[]) const {

    ConstIndexArray      vEdges  = _level.getVertexEdges(vIndex);
    ConstLocalIndexArray vInEdge = _level.getVertexEdgeLocalIndices(vIndex);

    ConstIndexArray      vFaces  = _level.getVertexFaces(vIndex);
    ConstLocalIndexArray vInFace = _level.getVertexFaceLocalIndices(vIndex);

    bool vIsBoundary = (vEdges.size() > vFaces.size());
    bool isBaseLevel = (_level.getDepth() == 0);

    for (int i = 0; i < vEdges.size(); ++i) {
        Index           eIndex = vEdges[i];
        ConstIndexArray eVerts = _level.getEdgeVertices(eIndex);

        //  Remember this method is for presumed continuous edges around the vertex:
        assert(edgeTopologyMatches(eIndex));

        Index vOther = eVerts[!vInEdge[i]];
        if (getNumVertexValues(vOther) == 1) {
            valuesPerEdge[i] = isBaseLevel ? getVertexValue(vOther) : getVertexValueOffset(vOther);
        } else if (vIsBoundary && (i == (vEdges.size() - 1))) {
            ConstIndexArray fValues = getFaceValues(vFaces[i-1]);

            int prevInFace = vInFace[i-1] ? (vInFace[i-1] - 1) : (fValues.size() - 1);
            valuesPerEdge[i] = fValues[prevInFace];
        } else {
            ConstIndexArray fValues = getFaceValues(vFaces[i]);

            int nextInFace = (vInFace[i] == (fValues.size() - 1)) ? 0 : (vInFace[i] + 1);
            valuesPerEdge[i] = fValues[nextInFace];
        }
    }
}

//
//  Gather information about the "span" of faces for each value:
//
//  This method is only invoked when the spans for values may be smooth boundaries and
//  other criteria that make all sharp (e.g. a non-manifold vertex) have been considered.
//
//  The "size" (number of faces in which each value occurs), is most immediately useful
//  in determining whether a value is a corner or smooth boundary, while other properties
//  such as the first face and whether or not the span is interrupted by a discts edge
//  (and so made "disjoint") or semi-sharp or infinite edges, are useful to fully qualify
//  smooth boundaries by the caller.
//
void
FVarLevel::gatherValueSpans(Index vIndex, ValueSpan * vValueSpans) const {

    ConstIndexArray vEdges = _level.getVertexEdges(vIndex);
    ConstIndexArray vFaces = _level.getVertexFaces(vIndex);

    ConstSiblingArray vFaceSiblings = getVertexFaceSiblings(vIndex);

    bool vHasSingleValue = (getNumVertexValues(vIndex) == 1);
    bool vIsBoundary = vEdges.size() > vFaces.size();

    if (vHasSingleValue) {
        //  Mark an interior dart disjoint if more than one discts edge:
        for (int i = 0; i < vEdges.size(); ++i) {
            if (_edgeTags[vEdges[i]]._mismatch) {
                if (vValueSpans[0]._size) {
                    vValueSpans[0]._disjoint = true;
                    break;
                } else {
                    vValueSpans[0]._size  = (LocalIndex) vFaces.size();
                    vValueSpans[0]._start = (LocalIndex) i;
                }
            } else if (_level._edgeTags[vEdges[i]]._infSharp) {
                vValueSpans[0]._disjoint = true;
                break;
            } else if (_level._edgeTags[vEdges[i]]._semiSharp) {
                ++ vValueSpans[0]._semiSharp;
            }
        }
    } else {
        //  Walk around the vertex and accumulate span info for each value -- be
        //  careful about the span for the first value "wrapping" around:
        vValueSpans[0]._size  = 1;
        vValueSpans[0]._start = 0;
        if (!vIsBoundary && (vFaceSiblings[vFaces.size() - 1] == 0)) {
            if (_edgeTags[vEdges[0]]._mismatch) {
                vValueSpans[0]._disjoint = true;
            } else if (_level._edgeTags[vEdges[0]]._infSharp) {
                vValueSpans[0]._disjoint = true;
            } else if (_level._edgeTags[vEdges[0]]._semiSharp) {
                ++ vValueSpans[0]._semiSharp;
            }
        }
        for (int i = 1; i < vFaces.size(); ++i) {
            if (vFaceSiblings[i] == vFaceSiblings[i-1]) {
                if (_edgeTags[vEdges[i]]._mismatch) {
                    ++ vValueSpans[vFaceSiblings[i]]._disjoint;
                } else if (_level._edgeTags[vEdges[i]]._infSharp) {
                    vValueSpans[vFaceSiblings[i]]._disjoint = true;
                } else if (_level._edgeTags[vEdges[i]]._semiSharp) {
                    ++ vValueSpans[vFaceSiblings[i]]._semiSharp;
                }
            } else {
                //  If we have already set the span for this value, mark disjoint
                if (vValueSpans[vFaceSiblings[i]]._size > 0) {
                    ++ vValueSpans[vFaceSiblings[i]]._disjoint;
                }
                vValueSpans[vFaceSiblings[i]]._start = (LocalIndex) i;
            }
            ++ vValueSpans[vFaceSiblings[i]]._size;
        }
        //  If the span for value 0 has wrapped around, decrement the disjoint added
        //  at the interior edge where it started the closing part of the span:
        if ((vFaceSiblings[vFaces.size() - 1] == 0) && !vIsBoundary) {
            -- vValueSpans[0]._disjoint;
        }
    }
}

} // end namespace Vtr

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv
