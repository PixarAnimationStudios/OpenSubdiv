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
    _level(level), _isLinear(false), _hasSmoothBoundaries(false), _valueCount(0) {
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
FVarLevel::resizeValues(int valueCount) {
    _valueCount = valueCount;
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
    _vertSiblingCounts.resize(_level.getNumVertices(), 0);
    _vertSiblingOffsets.resize(_level.getNumVertices());

    _vertFaceSiblings.resize(_level.getNumVertexFacesTotal(), 0);
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
    //  members reflect queries that are made elsewhere):
    //
    //  Given the growing number of options and behaviors to support, this is likely going
    //  to get another pass.  It may be worth identifying the behavior for each "feature",
    //  i.e. determine smooth or sharp for corners, creases and darts...
    //
    using Sdc::Options;

    Options::VVarBoundaryInterpolation geomOptions = _options.GetVVarBoundaryInterpolation();
    Options::FVarLinearInterpolation   fvarOptions = _options.GetFVarLinearInterpolation();

    _isLinear = (fvarOptions == Options::FVAR_LINEAR_ALL);

    _hasSmoothBoundaries = (fvarOptions != Options::FVAR_LINEAR_ALL) &&
                           (fvarOptions != Options::FVAR_LINEAR_BOUNDARIES);

    bool geomCornersAreSmooth = (geomOptions != Options::VVAR_BOUNDARY_EDGE_AND_CORNER);
    bool fvarCornersAreSharp  = (fvarOptions != Options::FVAR_LINEAR_NONE);

    bool makeCornersSharp = geomCornersAreSmooth && fvarCornersAreSharp;

    //
    //  Two "options" conditionally sharpen all values for a vertex depending on the collective topology
    //  around the vertex rather than of the topology of the value itself within its disjoint region.
    //
    //  Historically Hbr will sharpen all values if there are more than 3 values associated with a vertex
    //  (making at least 3 discts or boundary edges) while the option to sharpen corners is enabled.  This
    //  is still being done for now but is not desirable in some cases and may become optional (consider
    //  two adjacent UV sets with smooth UV boundaries, then splitting one of them in two -- the other
    //  UV set will have its boundary sharpened at the vertex where the other was split).
    //
    //  The "propogate corners" associated with the smooth boundary and sharp corners option will sharpen
    //  all values if any one is a corner.  This preserves the sharpness of a concave corner in regular
    //  areas where one face is made a corner.  Since the above option has historically sharpened all
    //  values in cases where there are more than 2 values at a vertex, its unclear what the intent of
    //  "propagate corners" is if more than 2 are present.
    //
    bool cornersPlus1 = (fvarOptions == Options::FVAR_LINEAR_CORNERS_PLUS1);
    bool cornersPlus2 = (fvarOptions == Options::FVAR_LINEAR_CORNERS_PLUS2);

    bool considerEntireVertex = cornersPlus1 || cornersPlus2;

    bool sharpenAllIfMoreThan2 = considerEntireVertex;
    bool sharpenAllIfAnyCorner = cornersPlus2;
    bool sharpenDarts          = cornersPlus2 || !_hasSmoothBoundaries;

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
    //      - assigns the number of siblings for each of the N vertices
    //          - determining the total number of siblings M in the process
    //      - assigns the sibling offsets for each of the N vertices
    //      - assigns the vert-value tags (partially) for the first N vertices (matches or not)
    //      - initializes the vert-face siblings for all N vertices
    //  and
    //      - tags any incident edges as discts
    //
    //  The second pass initializes remaining members based on the total number of siblings
    //  M after allocating appropriate vectors dependent on M.
    //
    //  Still looking or opportunities to economize effort between the two passes...
    //
    ValueTag valueTagMatch;
    valueTagMatch.clear();

    ValueTag valueTagMismatch = valueTagMatch;
    valueTagMismatch._mismatch = true;

    _vertValueTags.resize(_level.getNumVertices(), valueTagMatch);
    _vertFaceSiblings.resize(_level.getNumVertexFacesTotal(), 0);

    int const maxValence = _level.getMaxValence();

    Index *   indexBuffer   = (Index *)alloca(maxValence*sizeof(Index));
    int *     valueBuffer   = (int *)alloca(maxValence*sizeof(int));
    Sibling * siblingBuffer = (Sibling *)alloca(maxValence*sizeof(Sibling));

    int * uniqueValues = valueBuffer;

    int totalValueCount = _level.getNumVertices();
    for (int vIndex = 0; vIndex < _level.getNumVertices(); ++vIndex) {
        IndexArray const      vEdges  = _level.getVertexEdges(vIndex);
        LocalIndexArray const vInEdge = _level.getVertexEdgeLocalIndices(vIndex);

        IndexArray const      vFaces  = _level.getVertexFaces(vIndex);
        LocalIndexArray const vInFace = _level.getVertexFaceLocalIndices(vIndex);

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
                IndexArray const eVerts = _level.getEdgeVertices(eIndex);
                _vertValueTags[eVerts[0]] = valueTagMismatch;
                _vertValueTags[eVerts[1]] = valueTagMismatch;

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
        if (vIsBoundary && !_vertValueTags[vIndex]._mismatch) {
            if (vFaces.size() == 1) {
                if (makeCornersSharp) {
                    _vertValueTags[vIndex]._mismatch = true;
                }
            } else if (!_hasSmoothBoundaries) {
                _vertValueTags[vIndex]._mismatch = true;
            }
        }

        //
        //  Make note of any extra values that will be added after one-per-vertex:
        //
        int siblingCount = uniqueValueCount - 1;

        _vertSiblingCounts[vIndex]  = (LocalIndex) siblingCount;
        _vertSiblingOffsets[vIndex] = totalValueCount;

        totalValueCount += siblingCount;

        //  Update the vert-face siblings from the local array above:
        if (siblingCount) {
            SiblingArray vFaceSiblings = getVertexFaceSiblings(vIndex);
            for (int i = 0; i < vFaces.size(); ++i) {
                vFaceSiblings[i] = vSiblings[i];
            }
        }
    }

    //
    //  Now that we know the total number of additional sibling values (M values in addition
    //  to the N vertex values) allocate space to accomodate all N + M values.  Note that we
    //  already have tags for the first N partially initialized (matching or not) so just
    //  append tags for the additional M sibling values (all M initialized as mismatched).
    //
    //  Tags for the additional M values are intentionally initialized as mismatched and as
    //  sharp corners so that we only need update tags requiring smooth boundaries.
    //
    _vertValueIndices.resize(totalValueCount);
    _vertValueTags.resize(totalValueCount, valueTagMismatch);

    if (_hasSmoothBoundaries) {
        _vertValueCreaseEnds.resize(totalValueCount * 2);
    }

    //
    //  Now the second pass through the vertices to identify the values associated with the
    //  vertex and to inspect local face-varying topology in more detail when necessary:
    //
    ValueTag valueTagCrease = valueTagMismatch;
    valueTagCrease._crease = true;

    ValueTag valueTagSemiSharp = valueTagMismatch;
    valueTagSemiSharp._semiSharp = true;

    for (int vIndex = 0; vIndex < _level.getNumVertices(); ++vIndex) {
        IndexArray const      vFaces  = _level.getVertexFaces(vIndex);
        LocalIndexArray const vInFace = _level.getVertexFaceLocalIndices(vIndex);

        //
        //  First step is to assign the values associated with the faces by retrieving them
        //  from the faces.  If the face-varying topology around this vertex matches the
        //  vertex topology, there is little more to do and we can continue immediately:
        //
        _vertValueIndices[vIndex] = _faceVertValues[_level.getOffsetOfFaceVertices(vFaces[0]) + vInFace[0]];
        if (!_vertValueTags[vIndex]._mismatch) {
            continue;
        }

        int vSiblingOffset = _vertSiblingOffsets[vIndex];
        int vSiblingCount  = _vertSiblingCounts[vIndex];
        int vValueCount    = 1 + vSiblingCount;

        SiblingArray const vFaceSiblings = getVertexFaceSiblings(vIndex);

        if (vValueCount > 1) {
            Index * vertValueSiblingIndices = &_vertValueIndices[vSiblingOffset];
            int vSiblingIndex = 1;
            for (int i = 1; i < vFaces.size(); ++i) {
                if (vFaceSiblings[i] == vSiblingIndex) {
                    *vertValueSiblingIndices++ = _faceVertValues[_level.getOffsetOfFaceVertices(vFaces[i]) + vInFace[i]];
                    vSiblingIndex++;
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
        bool  vIsBoundary = _level._vertTags[vIndex]._boundary;
        float vSharpness  = _level._vertSharpness[vIndex];

        bool allCornersAreSharp = !_hasSmoothBoundaries ||
                                  Sdc::Crease::IsInfinite(vSharpness) ||
                                  (sharpenAllIfMoreThan2 && (vValueCount > 2)) ||
                                  (sharpenDarts && (vValueCount == 1) && !vIsBoundary) ||
                                   _level._vertTags[vIndex]._nonManifold;
        if (allCornersAreSharp) {
            continue;
        }

        //
        //  Values may be a mix of sharp corners and smooth boundaries...
        //
        //  Gather information about the "span" of faces for each value.  Use the results
        //  in one last chance for all values to be made sharp via interpolation options...
        //
        assert(sizeof(ValueSpan) <= sizeof(int));
        ValueSpan * vValueSpans = (ValueSpan *) indexBuffer;
        memset(vValueSpans, 0, vValueCount * sizeof(ValueSpan));

        gatherValueSpans(vIndex, vValueSpans);

        bool testForSmoothBoundaries = true;
        if (sharpenAllIfAnyCorner) {
            for (int i = 0; i < vValueCount; ++i) {
                if (vValueSpans[i]._size == 1) {
                    testForSmoothBoundaries = false;
                    break;
                }
            }
        }

        //
        //  Test each vertex value to determine if it is a smooth boundary (crease) -- if not
        //  disjoint and not a sharp corner, tag as a smooth boundary and identify its ends.
        //  Note if semi-sharp, and do not tag it as a crease now -- the refinement will tag
        //  it once all sharpness has decayed to zero:
        //
        if (testForSmoothBoundaries) {
            for (int i = 0; i < vValueCount; ++i) {
                ValueSpan& vSpan = vValueSpans[i];

                if (!vSpan._disjoint && ((vSpan._size > 1) || !fvarCornersAreSharp)) {
                    Index valueIndex = (i == 0) ? vIndex : (vSiblingOffset + i - 1);

                    if ((vSpan._semiSharp > 0) || (vSharpness > 0)) {
                        _vertValueTags[valueIndex] = valueTagSemiSharp;
                    } else {
                        _vertValueTags[valueIndex] = valueTagCrease;
                    }

                    LocalIndex * endFaces = &_vertValueCreaseEnds[2 * valueIndex];

                    endFaces[0] = vSpan._start;
                    if ((i == 0) && (vSpan._start != 0)) {
                        endFaces[1] = (LocalIndex) (vSpan._start + vSpan._size - 1 - vFaces.size());
                    } else {
                        endFaces[1] = vSpan._start + vSpan._size - 1;
                    }
                }
            }
        }
    }
//    printf("completed fvar topology...\n");
//    print();
//    printf("validating...\n");
//    assert(validate());
}

//
//  Values tagged as creases have their two "end values" identified relative to the incident
//  faces of the vertex for compact storage and quick retrieval.  This methods identifies the
//  values for the two ends of such a crease value:
//
void
FVarLevel::getVertexCreaseEndValues(Index vIndex, Sibling vSibling, Index endValues[2]) const
{
    int creaseEndOffset = 2 * getVertexValueIndex(vIndex, vSibling);

    LocalIndex vertFace0 = _vertValueCreaseEnds[creaseEndOffset];
    LocalIndex vertFace1 = _vertValueCreaseEnds[creaseEndOffset + 1];

    IndexArray const      vFaces  = _level.getVertexFaces(vIndex);
    LocalIndexArray const vInFace = _level.getVertexFaceLocalIndices(vIndex);

    LocalIndex endInFace0 = vInFace[vertFace0];
    LocalIndex endInFace1 = vInFace[vertFace1];
    if (_level.getDepth() > 0) {
        endInFace0 = (endInFace0 + 1) % 4;
        endInFace1 = (endInFace1 + 3) % 4;
    } else {
        //  Avoid the costly % N for potential N-sided faces at level 0...
        endInFace0++;
        if (endInFace0 == _level.getNumFaceVertices(vFaces[vertFace0])) {
            endInFace0 = 0;
        }
        endInFace1 = (endInFace1 ? endInFace1 : _level.getNumFaceVertices(vFaces[vertFace1])) - 1;
    }

    endValues[0] = _faceVertValues[_level.getOffsetOfFaceVertices(vFaces[vertFace0]) + endInFace0];
    endValues[1] = _faceVertValues[_level.getOffsetOfFaceVertices(vFaces[vertFace1]) + endInFace1];
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
        IndexArray const fVerts    = _level.getFaceVertices(fIndex);
        IndexArray const fValues   = getFaceValues(fIndex);
        Sibling const*      fSiblings = &fvSiblingVector[_level.getOffsetOfFaceVertices(fIndex)];

        for (int fvIndex = 0; fvIndex < fVerts.size(); ++fvIndex) {
            Index vIndex = fVerts[fvIndex];

            Index fvValue   = fValues[fvIndex];
            Sibling  fvSibling = fSiblings[fvIndex];
            //  Remember the "sibling count" is 0 when a single value, i.e. no siblings
            if (fvSibling > _vertSiblingCounts[vIndex]) {
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
        IndexArray const      vFaces    = _level.getVertexFaces(vIndex);
        LocalIndexArray const vInFace   = _level.getVertexFaceLocalIndices(vIndex);
        SiblingArray const       vSiblings = getVertexFaceSiblings(vIndex);

        for (int j = 0; j < vFaces.size(); ++j) {
            Sibling vSibling = vSiblings[j];
            //  Remember the "sibling count" is 0 when a single value, i.e. no siblings
            if (vSibling > _vertSiblingCounts[vIndex]) {
                printf("Error:  invalid sibling %d at vert-face %d.%d\n", vSibling, vIndex, j);
                return false;
            }

            Index fIndex  = vFaces[j];
            int      fvIndex = vInFace[j];
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
    printf("    vertex count = %d\n", _level.getNumVertices());
    printf("    value count  = %d\n", _valueCount);
    printf("    vert-values  = %d\n", (int)_vertValueIndices.size());

    printf("  Face values:\n");
    for (int i = 0; i < _level.getNumFaces(); ++i) {
        IndexArray const fVerts = _level.getFaceVertices(i);
        IndexArray const fValues = getFaceValues(i);
        Sibling const*      fSiblings = &fvSiblingVector[_level.getOffsetOfFaceVertices(i)];

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
        int sCount  = _vertSiblingCounts[i];
        int sOffset = _vertSiblingOffsets[i];

        printf("    vert%4d:  scount = %1d, soffset =%4d, ", i, sCount, sOffset);

        printf("values =%4d", _vertValueIndices[i]);
        for (int j = 0; j < sCount; ++j) {
            printf("%4d", _vertValueIndices[sOffset + j]);
        }
        if (sCount) {
            printf(", crease =%4d", _vertValueTags[i]._crease);
            for (int j = 0; j < sCount; ++j) {
                printf("%4d", _vertValueTags[sOffset + j]._crease);
            }
            printf(", semi-sharp =%2d", (int)_vertValueTags[i]._semiSharp);
            for (int j = 0; j < sCount; ++j) {
                printf("%2d", _vertValueTags[sOffset + j]._semiSharp);
            }
        }
        printf("\n");
    }

    printf("  Edge discontinuities:\n");
    for (int i = 0; i < _level.getNumEdges(); ++i) {
        ETag const& eTag = _edgeTags[i];
        if (eTag._mismatch) {
            IndexArray eVerts = _level.getEdgeVertices(i);
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
FVarLevel::initializeFaceValuesFromVertexFaceSiblings(int vFirstSibling)
{
    //
    //  Now use the vert-face-siblings to populate the face-siblings and face-values:
    //
    //  The face-siblings will have been initialized to 0, so start by copying face-verts
    //  of all child faces to the face-values.  Then iterate through the vert-faces and
    //  apply any non-zero siblings to both -- assigning sibling offset and the value index
    //  to the appropriate face-vert.
    //
    //  Note that while we are updating sparsely, we are potentially writing to scattered
    //  locations in memory given the face-vert locations for each vertex.  It may be
    //  worth doing the face-siblings first and then updating the face-values in order
    //  (each being an indirect "sum" of face-vertex and face-sibling).
    //
    initializeFaceValuesFromFaceVertices();

    //
    //  We can deal with all vertices similarly, regardless of whether a sibling originated
    //  from a parent edge or vertex (as this is typically constructed as part of refinement)
    //
    for (int vIndex = vFirstSibling; vIndex < getNumVertices(); ++vIndex) {
        int vSiblingCount = _vertSiblingCounts[vIndex];
        if (vSiblingCount) {
            SiblingArray const       vSiblings = getVertexFaceSiblings(vIndex);
            IndexArray const      vFaces    = _level.getVertexFaces(vIndex);
            LocalIndexArray const vInFace   = _level.getVertexFaceLocalIndices(vIndex);

            for (int j = 0; j < vFaces.size(); ++j) {
                if (vSiblings[j]) {
                    getFaceValues(vFaces[j])[vInFace[j]] = getVertexValueIndex(vIndex, vSiblings[j]);
                }
            }
        }
    }
}

void
FVarLevel::buildFaceVertexSiblingsFromVertexFaceSiblings(std::vector<Sibling>& fvSiblings) const {

    fvSiblings.resize(_level.getNumFaceVerticesTotal());
    std::memset(&fvSiblings[0], 0, _level.getNumFaceVerticesTotal() * sizeof(Sibling));

    for (int vIndex = 0; vIndex < getNumVertices(); ++vIndex) {
        int vSiblingCount = _vertSiblingCounts[vIndex];
        if (vSiblingCount) {
            SiblingArray const       vSiblings = getVertexFaceSiblings(vIndex);
            IndexArray const      vFaces    = _level.getVertexFaces(vIndex);
            LocalIndexArray const vInFace   = _level.getVertexFaceLocalIndices(vIndex);

            for (int j = 0; j < vFaces.size(); ++j) {
                if (vSiblings[j]) {
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

    IndexArray const eVerts = _level.getEdgeVertices(eIndex);
    if (_vertSiblingCounts[eVerts[0]] || _vertSiblingCounts[eVerts[1]]) {
        Index eFace = _level.getEdgeFaces(eIndex)[fIncToEdge];

        //  This is another of those irritating times where I want to have the edge-in-face
        //  local indices stored with each edge-face...
        /*
        IndexArray const fEdges  = _level.getFaceEdges(eFace);
        IndexArray const fVerts  = _level.getFaceVertices(eFace);
        IndexArray const fValues = getFaceValues(eFace);

        for (int i = 0; i < fEdges.size(); ++i) {
            if (fEdges[i] == eIndex) {
                int i0 = i;
                int i1 = (i + 1) % fEdges.size();

                if (fVerts[i0] != eVerts[0]) {
                    std::swap(i0, i1);
                }
                valuesPerVert[0] = fValues[i0];
                valuesPerVert[1] = fValues[i1];
                return;
            }
        }
        */

        //  This is about as fast as we're going to get with a search and still leaves
        //  this function showing a considerable part of edge-vertex interpolation (down
        //  from 40% to 25%)...
        //
        IndexArray const fEdges  = _level.getFaceEdges(eFace);
        IndexArray const fValues = getFaceValues(eFace);

        if (fEdges.size() == 4) {
            if (fEdges[0] == eIndex) {
                valuesPerVert[0] = fValues[0];
                valuesPerVert[1] = fValues[1];
            } else if (fEdges[1] == eIndex) {
                valuesPerVert[0] = fValues[1];
                valuesPerVert[1] = fValues[2];
            } else if (fEdges[2] == eIndex) {
                valuesPerVert[0] = fValues[2];
                valuesPerVert[1] = fValues[3];
            } else {
                valuesPerVert[0] = fValues[3];
                valuesPerVert[1] = fValues[0];
            }
        } else if (fEdges.size() == 3) {
            if (fEdges[0] == eIndex) {
                valuesPerVert[0] = fValues[0];
                valuesPerVert[1] = fValues[1];
            } else if (fEdges[1] == eIndex) {
                valuesPerVert[0] = fValues[1];
                valuesPerVert[1] = fValues[2];
            } else {
                valuesPerVert[0] = fValues[2];
                valuesPerVert[1] = fValues[0];
            }
        } else {
            assert(fEdges.size() <= 4);
        }
    } else {
        valuesPerVert[0] = _vertValueIndices[eVerts[0]];
        valuesPerVert[1] = _vertValueIndices[eVerts[1]];
    }
}

void
FVarLevel::getVertexEdgeValues(Index vIndex, Index valuesPerEdge[]) const {

    IndexArray const      vEdges  = _level.getVertexEdges(vIndex);
    LocalIndexArray const vInEdge = _level.getVertexEdgeLocalIndices(vIndex);

    IndexArray const      vFaces  = _level.getVertexFaces(vIndex);
    LocalIndexArray const vInFace = _level.getVertexFaceLocalIndices(vIndex);

    bool vIsBoundary = (vEdges.size() > vFaces.size());

//printf("                    Gathering edge-values for vertex %d:\n", vIndex);
    for (int i = 0; i < vEdges.size(); ++i) {
        Index            eIndex = vEdges[i];
        IndexArray const eVerts = _level.getEdgeVertices(eIndex);
//printf("                      edge %d - boundary = %d\n", eIndex, _level._edgeTags[eIndex]._boundary);

        //  Remember this method is for presumed continuous edges around the vertex:
        if (_edgeTags[eIndex]._mismatch) {
printf("WARNING - unexpected mismatched edge %d gathering edge-values for vertex %d:\n", eIndex, vIndex);
printf("    vertex tag mismatch = %d\n", _vertValueTags[vIndex]._mismatch);
printf("    edge[] tag mismatch = %d\n", _edgeTags[eIndex]._mismatch);
        }
        assert(_edgeTags[eIndex]._mismatch == false);

        Index vOther = eVerts[!vInEdge[i]];
        if (_vertSiblingCounts[vOther] == 0) {
//printf("                        singular\n");
            valuesPerEdge[i] = _vertValueIndices[vOther];
        } else {
            //
            //  We need to identify an incident face for this edge from which to get
            //  a value at the end of the edge.  If we had the edge-in-face local-index
            //  we could quickly identify the edge in any adjacent face -- but for now
            //  we would have to do a search in any incident face (e.g. the first).
            //
            //  We can use the ordering of edges and faces to identify such a face,
            //  and use the vert-in-face local-index to then identify the value...
            //  NO WAIT -- we can't right now because refined vert-edges are not oriented
            //  correctly!
            //
            bool useOrientedEdges = false;
            if (useOrientedEdges) {
                if (vIsBoundary && (i == (vEdges.size() - 1))) {
                    valuesPerEdge[i] = getFaceValues(vFaces[i-1])[(vInFace[i-1] + 3) % 4];
                } else {
                    valuesPerEdge[i] = getFaceValues(vFaces[i])[(vInFace[i] + 1) % 4];
                }
            } else {
                Index            eFace  = _level.getEdgeFaces(eIndex)[0];
                IndexArray const fVerts = _level.getFaceVertices(eFace);
                for (int j = 0; j < fVerts.size(); ++j) {
                    if (fVerts[j] == vOther) {
                        valuesPerEdge[i] = getFaceValues(eFace)[j];
                        break;
                    }
                }
            }
        }
//printf("                            edge-value[%d] = %4d\n", i, valuesPerEdge[i]);
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

    IndexArray const vEdges = _level.getVertexEdges(vIndex);
    IndexArray const vFaces = _level.getVertexFaces(vIndex);

    SiblingArray const vFaceSiblings = getVertexFaceSiblings(vIndex);

    bool vHasSingleValue = (_vertSiblingCounts[vIndex] == 0);
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
