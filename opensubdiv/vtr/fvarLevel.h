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
#ifndef VTR_FVAR_LEVEL_H
#define VTR_FVAR_LEVEL_H

#include "../version.h"

#include "../sdc/type.h"
#include "../sdc/crease.h"
#include "../sdc/options.h"
#include "../vtr/types.h"
#include "../vtr/level.h"

#include <vector>
#include <cassert>
#include <cstring>


namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Vtr {

//
//  FVarLevel:
//      A "face-varying channel" includes the topology for a set of face-varying
//  data, relative to the topology of the Level with which it is associated.
//
//  Analogous to a set of vertices and face-vertices that define the topology for
//  the geometry, a channel requires a set of "values" and "face-values".  The
//  "values" are indices of entries in a set of face-varying data, just as vertices
//  are indices into a set of vertex data.  The face-values identify a value for
//  each vertex of the face, and so define topology for the values that may be
//  unique to each channel.
//
//  In addition to the value size and the vector of face-values (which matches the
//  size of the geometry's face-vertices), tags are associated with each component
//  to identify deviations of the face-varying topology from the vertex topology.
//  And since there may be a one-to-many mapping between vertices and face-varying
//  values, that mapping is also allocated.
//
//  It turns out that the mapping used is able to completely encode the set of
//  face-values and is more amenable to refinement.  Currently the face-values
//  take up almost half the memory of this representation, so if memory does
//  become a concern, we do not need to store them.  The only reason we do so now
//  is that the face-value interface for specifying base topology and inspecting
//  subsequent levels is very familar to that of face-vertices for clients.  So
//  having them available for such access is convenient.
//
//  Regarding scope and access...
//      Unclear at this early state, but leaning towards nesting this class within
//  Level, given the intimate dependency between the two.
//      Everything is being declared public for now to facilitate access until its
//  clearer how this functionality will be provided.
//

class FVarLevel {

public:
    typedef LocalIndex      Sibling;
    typedef LocalIndexArray SiblingArray;

public:
    //
    //  Component tags -- trying to minimize the types needed here:
    //
    //  Tag per Edge:
    //      - facilitates topological analysis around each vertex
    //      - required during refinement to spawn one or more edge-values
    //
    struct ETag {
        ETag() { }

        void clear() { std::memset(this, 0, sizeof(ETag)); }

        typedef unsigned char ETagSize;

        ETagSize _mismatch : 1;  // local FVar topology does not match
        ETagSize _boundary : 1;  // not continuous at both ends
        ETagSize _disctsV0 : 1;  // discontinuous at vertex 0
        ETagSize _disctsV1 : 1;  // discontinuous at vertex 1
    };

    //
    //  Tag per Value:
    //      - informs both refinement and interpolation
    //          - every value spawns a child value in refinement
    //      - given ordering of values (1-per-vertex first) serves as a vertex tag
    //
    struct ValueTag {
        ValueTag() { }

        void clear() { std::memset(this, 0, sizeof(ValueTag)); }

        typedef unsigned char ValueTagSize;

        ValueTagSize _mismatch  : 1;  // local FVar topology does not match
        ValueTagSize _crease    : 1;  // value is a crease, otherwise a corner
        ValueTagSize _semiSharp : 1;  // value is a corner decaying to crease
    };

public:
    FVarLevel(Level const& level);
    ~FVarLevel();

    //  Const methods:
    //
    //  Inventory of the face-varying level itself:
    Level const& getLevel() const { return _level; }

    int getDepth() const       { return _level.getDepth(); }
    int getNumFaces() const    { return _level.getNumFaces(); }
    int getNumEdges() const    { return _level.getNumEdges(); }
    int getNumVertices() const { return _level.getNumVertices(); }

    int getNumValues() const          { return _valueCount; }
    int getNumFaceValuesTotal() const { return (int) _faceVertValues.size(); }

    //  Queries per face:
    IndexArray const getFaceValues(Index fIndex) const;

    //  Queries per vertex (and its potential sibling values):
    bool vertexTopologyMatches(Index vIndex) const { return !_vertValueTags[vIndex]._mismatch; }

    int   getNumVertexValues(Index vIndex) const;
    Index getVertexValueIndex(Index vIndex, Sibling sibling = 0) const;
    Index getVertexValue(Index vIndex, Sibling sibling = 0) const;

    SiblingArray const getVertexFaceSiblings(Index faceIndex) const;

    //  Queries specific to values:
    bool isValueCrease(Index valueIndex) const    { return _vertValueTags[valueIndex]._crease; }
    bool isValueCorner(Index valueIndex) const    { return !_vertValueTags[valueIndex]._crease; }
    bool isValueSemiSharp(Index valueIndex) const { return _vertValueTags[valueIndex]._semiSharp; }
    bool isValueInfSharp(Index valueIndex) const  { return !_vertValueTags[valueIndex]._semiSharp &&
                                                           !_vertValueTags[valueIndex]._crease; }
                                                           

    //  Higher-level topological queries, i.e. values in a neighborhood:
    void getEdgeFaceValues(Index eIndex, int fIncToEdge, Index valuesPerVert[2]) const;
    void getVertexEdgeValues(Index vIndex, Index valuesPerEdge[]) const;
    void getVertexCreaseEndValues(Index vIndex, Sibling sibling, Index endValues[2]) const;

    //  Currently the sibling value storage and indexing is being reconsidered...
    //    int                 getNumVertexSiblings(Index vIndex) const;
    //    IndexArray const getVertexSiblingValues(Index vIndex) const;

    //  Non-const methods -- modifiers to be protected:
    //
    //  Array modifiers for the per-face and vertex-face data:
    IndexArray getFaceValues(Index fIndex);
    SiblingArray  getVertexFaceSiblings(Index vIndex);

    void setOptions(Sdc::Options const& options);
    void resizeValues(int numValues);
    void resizeComponents();

    void completeTopologyFromFaceValues();
    void initializeFaceValuesFromFaceVertices();
    void initializeFaceValuesFromVertexFaceSiblings(int firstVertex = 0);
    void buildFaceVertexSiblingsFromVertexFaceSiblings(std::vector<Sibling>& fvSiblings) const;

    //  Information about the "span" for a value:
    struct ValueSpan {
        LocalIndex _size;
        LocalIndex _start;
        LocalIndex _disjoint;
        LocalIndex _semiSharp;
    };
    void gatherValueSpans(Index vIndex, ValueSpan * vValueSpans) const;

    bool validate() const;
    void print() const;

public:
    Level const & _level;

    //  Options vary between channels:
    Sdc::Options _options;

    bool _isLinear;
    bool _hasSmoothBoundaries;
    int  _valueCount;

    //
    //  Vectors recording face-varying topology -- values-per-face, which edges
    //  are discts wrt the FVar data, the one-to-many mapping between vertices and 
    //  their sibling values, etc.  We use 8-bit "local indices" where possible.
    //
    //  Per-face (matches face-verts of corresponding level):
    std::vector<Index> _faceVertValues;

    //  Per-edge:
    std::vector<ETag> _edgeTags;

    //  Per-vertex:
    std::vector<Sibling>  _vertSiblingCounts;
    std::vector<int>      _vertSiblingOffsets;
    std::vector<Sibling>  _vertFaceSiblings;

    //  Per-value:
    std::vector<Index>      _vertValueIndices;
    std::vector<ValueTag>   _vertValueTags;
    std::vector<LocalIndex> _vertValueCreaseEnds;
};

//
//  Access/modify the values associated with each face:
//
inline IndexArray const
FVarLevel::getFaceValues(Index fIndex) const {

    int vCount  = _level._faceVertCountsAndOffsets[fIndex*2];
    int vOffset = _level._faceVertCountsAndOffsets[fIndex*2+1];
    return IndexArray(&_faceVertValues[vOffset], vCount);
}
inline IndexArray
FVarLevel::getFaceValues(Index fIndex) {

    int vCount  = _level._faceVertCountsAndOffsets[fIndex*2];
    int vOffset = _level._faceVertCountsAndOffsets[fIndex*2+1];
    return IndexArray(&_faceVertValues[vOffset], vCount);
}

inline FVarLevel::SiblingArray const
FVarLevel::getVertexFaceSiblings(Index vIndex) const {

    int vCount  = _level._vertFaceCountsAndOffsets[vIndex*2];
    int vOffset = _level._vertFaceCountsAndOffsets[vIndex*2+1];
    return SiblingArray(&_vertFaceSiblings[vOffset], vCount);
}
inline FVarLevel::SiblingArray
FVarLevel::getVertexFaceSiblings(Index vIndex) {

    int vCount  = _level._vertFaceCountsAndOffsets[vIndex*2];
    int vOffset = _level._vertFaceCountsAndOffsets[vIndex*2+1];
    return SiblingArray(&_vertFaceSiblings[vOffset], vCount);
}

//
//  Access the values associated with each vertex:
//
/*
inline int
FVarLevel::getNumVertexSiblings(Index vertexIndex) const
{
    return _vertSiblingCounts[vertexIndex];
}
inline IndexArray const
FVarLevel::getVertexSiblingValues(Index vIndex) const
{
    int vCount  = _vertSiblingCounts[vIndex];
    int vOffset = _vertSiblingOffsets[vIndex];
    return IndexArray(&_vertValueIndices[vOffset], vCount);
}
*/
inline int
FVarLevel::getNumVertexValues(Index vertexIndex) const {
    return 1 + _vertSiblingCounts[vertexIndex];
}
inline Index
FVarLevel::getVertexValueIndex(Index vIndex, Sibling vSibling) const {
    return vSibling ? (_vertSiblingOffsets[vIndex] + vSibling - 1) : vIndex;
}
inline Index
FVarLevel::getVertexValue(Index vIndex, Sibling vSibling) const {
    return _vertValueIndices[getVertexValueIndex(vIndex, vSibling)];
}

} // end namespace Vtr

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;
} // end namespace OpenSubdiv

#endif /* VTR_FVAR_LEVEL_H */
