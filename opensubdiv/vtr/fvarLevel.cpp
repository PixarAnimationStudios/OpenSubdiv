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
    _level(level), _isLinear(false), _valueCount(0) {
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
    _edgeTags.resize(_level.getNumEdges());
    std::memset(&_edgeTags[0], 0, _level.getNumEdges() * sizeof(ETag));

    //  Per-vertex members:
    _vertSiblingCounts.resize(_level.getNumVertices(), 0);
    _vertSiblingOffsets.resize(_level.getNumVertices());

    _vertFaceSiblings.resize(_level.getNumVertexFacesTotal(), 0);
}


//
//  Initialize the component tags once all face-values have been assigned:
//      - edge traversal:
//          - finding and testing the values at matching ends of the edge
//          - marking the bitfields according to the result
//          - is it worth marking vertices as "mismatch"?
//              - need more vertex analysis later
//              - will only be marking discts verts
//              - no iteration of edge-verts, unlike vert-edges
//          - (consider) marking incident faces as "mismatch"
//      - vertex traversal:
//          - mismatches previously identified in edge traversal:
//              - should be able to skip vertices that match
//          - need to identify number/index of values for each
//          - should be able to identify topology per value during iteration:
//              - one "in a row" will be a corner, "crease" otherwise
//
//  WAIT -- consider the following vertex-oriented alternative...
//      - vertex traversal:
//          - iterate through all successive face-vert values:
//              - each successive mismatch identifies an edge discty at one end
//                  - mark the edge mismatch
//                  - mark the discts end (which is trivially known from local index)
//              - "spans" of values identifies their topology
//                  - beware disjoint spans -- effectively corners
//              - set of unique values gathered in process
//          * NOTE - does not capture the dart/boundary case
//              - single value consistent around vertex -- no discts edge detected
//              - remember this is not a Dart but a true boundary, i.e. a Crease
//              - could mark vertex when marking edge, but order-dependent:
//                  - if just need to mark mismatch, could be sufficient:
//                      - one value but mismatch implies Dart
//              - single value may be at the center of multiple discts edges:
//                  - value becomes Corner then instead of Crease
//
//  Implications:
//      - both the edge-traversal and the vertex traversal want to compare indices
//      - pros/cons for vertex traversal:
//          - pro:  1/2 as many verts as edges
//          - pro:  directly identifies adjacent face-vert values (via local index)
//          - CON:  knowledge of discts edges absent (cts at vertex, discts end of edge)
//      - pros/cons for edge traversal:
//          - pro:  can mark incident verts and faces mismatched in process
//          - CON:  2x as many edges as verts
//          - CON:  requires search to find edge in each adjacent face
//      - vertex traversal seems preferable, but CON is significant:
//          - could need to test "other end of each edge":
//              - unfortunate in that pair-wise test succeeding now slowed down
//                in ALL cases -- even when everything matches
//          - could mark "other end vertex" when marking any edge discts:
//              - would result in "revisiting" a vertex
//              - what are implications of identifying opposite end-splits later?
//                  - currently matched becomes mismatched, crease
//                  - single mismatched becomes mismatched, corner
//                  - multiple mismatched is awkward to deal with:
//                      - spans that may be crease fragmented -- how?
//                          - depending on span size may produce:
//                              - two Creases, two Corners, one of each...
//  Best of both worlds:
//      - double vertex traversal:
//          - first intializes edge tags and marks opposite vertex mismatched
//              - may also identify value-per-vertex counts
//          - second (sparse) will analyze local topology
//
void
FVarLevel::completeTopologyFromFaceValues() {

    //
    //  REMEMBER!!!
    //      .... that "mismatches" may also occur where the topology matches, but options
    //  specify different treatment, e.g. boundary and corner interpolation rules.  So we
    //  should inspect both the VVar and FVar boundary rules and determine when to mark
    //  such features as mismatched.
    //
    //  Note that this affects typical exterior boundaries in addition to corners -- a
    //  geometrically smooth disk may end up with piecewise linear UV boundaries.
    //
    //  Not sure what the precedence is here, particuarly "propagate corner" vs the choice
    //  of "fvar boundary interpolation" -- will need to check with Hbr.
    //
    _isLinear = (_options.GetFVarBoundaryInterpolation() == Sdc::Options::FVAR_BOUNDARY_BILINEAR);

    bool geomCornersAreSharp = (_options.GetVVarBoundaryInterpolation() == Sdc::Options::VVAR_BOUNDARY_EDGE_AND_CORNER);
    bool fvarCornersAreSharp = (_options.GetFVarBoundaryInterpolation() != Sdc::Options::FVAR_BOUNDARY_EDGE_ONLY);
    bool fvarPropagateCorner = false;

    fvarCornersAreSharp = fvarPropagateCorner ? geomCornersAreSharp : fvarCornersAreSharp;

    bool mismatchCorners = (fvarCornersAreSharp != geomCornersAreSharp);

    //
    //  Iterate through the vertices first to identify discts edges -- this is better done
    //  than iterating through the edges because:
    //
    //      - there are typically twice as many edges as vertices
    //
    //      - the ordering of incident faces of a vertex make retrieval/comparison of
    //        adjacent pairs of face-varying values much more efficient
    //
    //  Since an edge may be discts at only one end, but the vertex at the cts end still
    //  needs to be aware of such a discontinuity, we mark the opposite vertex for any
    //  edge that is found to be discts.
    //
    //  In the process, we will initialize the "sibling" values associated with both the
    //  face-verts and the vert-faces.  These are both 0-initialized and only updated when
    //  discontinuities are encountered.
    //
    ValueTag valueTagMatch(false);
    ValueTag valueTagMismatch(true);

    _vertValueTags.resize(_level.getNumVertices(), valueTagMatch);
    _vertFaceSiblings.resize(_level.getNumVertexFacesTotal(), 0);

    int const maxValence = _level.getMaxValence();

    Index * indexBuffer   = (Index *)alloca(maxValence*sizeof(Index));
    int *      valueBuffer   = (int *)alloca(maxValence*sizeof(int));
    Sibling  * siblingBuffer = (Sibling *)alloca(maxValence*sizeof(Sibling));

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

        bool vIsBoundary = (vEdges.size() != vFaces.size());
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
                ETag&    eTag   = _edgeTags[eIndex];

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
    //  Allocate space for the values -- we already have tags for those corresponding to each
    //  vertex (and those tags have been updated above), so append tags indicating the mismatch
    //  for all of the sibling values detected:
    //
    _vertValueIndices.resize(totalValueCount);
    _vertValueTags.resize(totalValueCount, valueTagMismatch);

    //
    //  Now a second pass through the vertices to identify the local face-varying topology
    //  in more detail -- tagging each FVar value associated with the vertex:
    //
    //  REMEMBER -- we want a ValueTag for each instance of the value at each vertex, i.e.
    //  per vertex-value.  At level 0, given user input, the number of vertex-values may
    //  exceed the number of values as they may have been used for multiple vertices (e.g.
    //  the case of each quad getting UV coordinates mapping the unit square).  The number
    //  of vertex-values is what we need for refinement to "unweld" them, and the tag for
    //  each will indicate refinement rules for its children.
    //
    for (int vIndex = 0; vIndex < _level.getNumVertices(); ++vIndex) {
        ValueTag& vTag = _vertValueTags[vIndex];

        IndexArray const      vFaces  = _level.getVertexFaces(vIndex);
        LocalIndexArray const vInFace = _level.getVertexFaceLocalIndices(vIndex);

        //
        //  At this point, mismatches have only been determined by the presence of edges
        //  that are discontinuous.  We may still have boundary/corner vertices that match
        //  topologically, but need to be treated differently (i.e. they do not match) due
        //  to boundary interpolation rules.  So update the "matched" status of boundary
        //  vertices before continuing to the general treatment for all cases:
        //
        bool vIsBoundary = _level._vertTags[vIndex]._boundary;
        if (vIsBoundary && !vTag._mismatch) {
            bool vIsCorner = (vFaces.size() == 1);
            if (vIsCorner) {
                vTag._mismatch = mismatchCorners;
            }
        }

        //
        //  If the face-varying topology around this vertex matches the vertex topology,
        //  there is little more to do -- except for level 0 where there may not be a
        //  1-to-1 correspondence between given values and vertex-values:
        //
        if (!vTag._mismatch) {
            if (_level.getDepth() == 0) {
                _vertValueIndices[vIndex] = _faceVertValues[_level.getOffsetOfFaceVertices(vFaces[0]) + vInFace[0]];
            } else {
                _vertValueIndices[vIndex] = vIndex;
            }
            continue;
        }

        //
        //  We have a mismatch in topology...  We may still have only one value for the
        //  vertex, but more than likely there will be other "sibling" values that will
        //  be appended to the set of 1-per-vertex.
        //
        //  When the topology does not match, we need to gather the unique values around
        //  this vertex and tag each according to its more localized topology.
        //
        //  Special cases to consider:
        //      - one mismatched value:  should be a crease (corner if linear or boundary?)
        //      - one value per face-vert:  all will be corners (hard or smooth?)
        //
        //  The general case will involve identifying "spans" via discts edges.  Any value
        //  that has more than one instance split over multiple spans must be a hard corner.
        //
        //  For now, make each unique value a hard corner to get something working:
        //
        IndexArray const      vEdges  = _level.getVertexEdges(vIndex);
        LocalIndexArray const vInEdge = _level.getVertexEdgeLocalIndices(vIndex);

        int vSiblingCount  = _vertSiblingCounts[vIndex];
        int vSiblingOffset = _vertSiblingOffsets[vIndex];

        //
        //  Gather the unique set of values and relevant topological information for each
        //  as we go...
        //
        int vvCount = 1 + vSiblingCount;
        int * vvIndices = indexBuffer;
        //int vvSrcFaces[vvCount];

        Index lastValue = _faceVertValues[_level.getOffsetOfFaceVertices(vFaces[0]) + vInFace[0]];

        vvCount = 1;
        vvIndices[0]  = lastValue;
        //vvSrcFaces[0] = 0;

        for (int i = 1; i < vFaces.size(); ++i) {
            Index nextValue = _faceVertValues[_level.getOffsetOfFaceVertices(vFaces[i]) + vInFace[i]];

            if (lastValue != nextValue) {
                bool nextValueIsNew = (vvCount == 1) || ((vvCount == 2) && (vvIndices[0] != nextValue)) ||
                                      (std::find(vvIndices, vvIndices + vvCount, nextValue) == vvIndices + vvCount);
                if (nextValueIsNew) {
                    vvIndices[vvCount]  = nextValue;
                    //vvSrcFaces[vvCount] = i;
                    vvCount ++;
                }
            }
            lastValue = nextValue;
        }
        assert(vvCount == (1 + vSiblingCount));

        //
        //  Assign tags to the vertex' primary value and sibling values:
        //
        ValueTag valueTag;
        valueTag._mismatch = true;
        valueTag._corner = true;
        valueTag._crease = false;

        _vertValueIndices[vIndex] = vvIndices[0];
        _vertValueTags[vIndex]    = valueTag;

        for (int i = 0; i < vSiblingCount; ++i) {
            _vertValueIndices[vSiblingOffset + i] = vvIndices[1 + i];
            _vertValueTags[vSiblingOffset + i]    = valueTag;
        }
    }
    //printf("completed topology...\n");
    //print();
    //printf("validating...\n");
    //assert(validate());
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

} // end namespace Vtr

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv
