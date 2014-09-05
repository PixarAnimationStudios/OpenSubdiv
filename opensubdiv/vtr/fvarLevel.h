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
    typedef Sdc::Options::FVarBoundaryInterpolation   BoundaryInterpolation;

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
        ETag(bool mismatch) : _mismatch(mismatch) { }

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
        ValueTag(bool mismatch) : _mismatch(mismatch) { }

        typedef unsigned char ValueTagSize;

        ValueTagSize _mismatch : 1;  // local FVar topology does not match
        ValueTagSize _corner   : 1;  // value is a corner (unlike its vertex)
        ValueTagSize _crease   : 1;  // value is a crease (unlike its vertex)

        //  Note that the corner/crease distinction will take into account the boundary
        //  interpolation options and other topological factors.  For example, a value
        //  on a Crease that is set to have Linear interpolation rules can be set to be a
        //  Corner for interpolation purposes, and conversely a Corner value with smooth
        //  corner rules will be set to be a Crease.

        //  Since the Crease is the only non-trivial case, want to store more information
        //  here for Crease so that we can quickly identify the values involved when it
        //  comes time to interpolate (though considering vertex interpolation, the set
        //  of edges is searched to identify the two corresponding to the crease when the
        //  mask is computed, so perhaps this effort is unwarranted).
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

    int      getNumVertexValues(Index vIndex) const;
    Index getVertexValueIndex(Index vIndex, Sibling sibling = 0) const;
    Index getVertexValue(Index vIndex, Sibling sibling = 0) const;

//    int                 getNumVertexSiblings(Index vIndex) const;
//    IndexArray const getVertexSiblingValues(Index vIndex) const;

    SiblingArray const getVertexFaceSiblings(Index faceIndex) const;

    //  Higher-level topological queries, i.e. values in a neighborhood:
    void getEdgeFaceValues(Index eIndex, int fIncToEdge, Index valuesPerVert[2]) const;
    void getVertexEdgeValues(Index vIndex, Index valuesPerEdge[]) const;

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

    bool validate() const;
    void print() const;

public:
    Level const & _level;

    //
    //  It's a little bit unclear at present how FVarBoundaryInterpolation works.
    //  I would have though the default would be to inherit the same interpolation
    //  rules from the geometry, but I don't see an enumeration for that -- so if
    //  that is desirable, an explicit internal initialization/assignment will be
    //  warranted.
    //
    //  Remember that the VVarBoundaryInterpolation has three enums that can now
    //  be reduced to two, so some revision to FVarBoundaryInterpolation may also
    //  be considered.
    //
    //  Options are stored locally here and they may vary between channels.  By
    //  default the options member is initialized from whatever contains it --
    //  which may apply a common set to all channels or vary them individually.
    //
    Sdc::Options _options;
    bool       _isLinear;

    int _valueCount;

    //
    //  Members that distinguish the face-varying "topology" from the Level to
    //  which the data set is associated.
    //
    //  Like the geometric topology, the face-varying topology is specified by
    //  a set of per-face-vertex indices -- analogous to vertices -- which are
    //  referred to as "values".  Additional vectors associated with vertices
    //  are constructed to identify the set of values incident each vertex.
    //  There is typically a single value associated with each vertex but also
    //  a set of "sibling" values when the face-varying values around a vertex
    //  are not all the same.  The neighborhood of each vertex is expressed in
    //  terms of these local sibling indices, i.e. local indices typically 0,
    //  1 or generally very low, and not exceeding the limit we use for vertex
    //  valence (and N-sided faces).
    //
    //  As the unique values are identified and associated with each vertex,
    //  the local neighborhood is also inspected and each value "tagged" to
    //  indicate its topology.  Foremost is a bit indicating whether the value
    //  "matches" the topology of its vertex (which can only occur when there
    //  is only one value for that vertex), but additionally bits are added to
    //  describe the topological neighborhood to indicate how it and all of its
    //  refined descendants should be treated.
    //
    //  Tags are associated with the "vertex values", i.e. each instance of a
    //  value for each vertex.  When a vertex has more than one value associated
    //  with it, the subdivision rules applicable to each are independent and
    //  fixed throughout refinement.
    //
    //  Level 0 as a special case:
    //      Given that users can specify the input values arbitrarily, it is
    //  necessary to add a little extra to accomodate this.
    //      For subsequent levels of refinement, the values associated with
    //  each vertex are exclusive to that vertex, e.g. if vertex V has values
    //  A and B incident to it, no other vertex will share A and B.  Any child
    //  values of A and B are then local to the child of V.
    //      This is not the case in level 0.  Considering a quad mesh there
    //  may be only 4 values for the corners of a unit square, and all vertices
    //  share combinations of those values.  In the
    //
    //  Notes on memory usage:
    //      The largest contributor to memory here is the set of face-values,
    //  which matches the size of the geometric face-vertices, i.e. typically
    //  4 ints per face.  It turns out this can be reconstructed from the rest,
    //  so whether it should always be updated/retained vs computed when needed
    //  is up for debate (right now it is constructed from the other members in
    //  a method as the last step in refinement).
    //      The most critical vector stores the sibling index for the values
    //  incident each vertex, i.e. it is the same size as the set of vert-faces
    //  (typically 4 ints per vertex) but a fraction of the size since we use
    //  8-bit indices for the siblings.
    //      The rest are typically an integer or two per vertex or value and
    //  8-bit tags per edge and value.
    //      The bulk of the memory usage is in the face-values, and these are
    //  not needed for refinement.
    //
    //  Memory cost (bytes) for members (N verts, N faces, 2*N edges, M > N value):
    //     16*N  per-face-vert values
    //      2*N  per-edge tags
    //        N  per-vertex sibling counts
    //      4*N  per-vertex sibling offsets
    //      4*N  per-vert-face siblings
    //      4*M  per-value indices (redundant after level 0)
    //        M  per-value tags
    //    - total:  27*N + 5*M
    //    - consider size of M:
    //        - worst case M = 4*N at level 0, but M -> N as level increases
    //        - typical size may be M ~= 1.5*N, so say M = 8/5 * N
    //            - current examples indicate far less:  1.1*N to 1.2*N
    //            - so M = 6/5 * N may be more realistic
    //        - total = 35*N, i.e. 8-9 ints-per-face (or per-vertex)
    //            * roughly an extra int for each face-vertex index
    //    - bare minimum (*):
    //        - 21*N:
    //            - 16*N face values specified as input
    //            -  4*N for some kind of vertex-to-value index/offset/mapping
    //            -    N for some kind of vertex/match indication tag
    //        * assuming face-values retained in user-specified form
    //            - compute on-demand by vert-face-siblings reduces by 12*N
    //        - note typical UV data size of M = N*8/5 values:
    //            - float[2] for each value -> data = 8*M = 13*N
    //    - potentially redundant:
    //        - 6*N (4*M) value indices for level > 0
    //    - possible extras:
    //        - 2*M (3*N) for the 2 ends of each value that is a crease
    //            - allocated for all M vertex-values
    //            - only populated for those tagged as creases
    //            ? how quickly can we look this up instead?
    //
    //  Per-face:
    std::vector<Index> _faceVertValues;     // matches face-verts of level (16*N)

    //  Per-edge:
    std::vector<ETag>     _edgeTags;           // 1 per edge (2*N)

    //  Per-vertex:
    std::vector<Sibling>  _vertSiblingCounts;  // 1 per vertex (1*N)
    std::vector<int>      _vertSiblingOffsets; // 1 per vertex (4*N)
    std::vector<Sibling>  _vertFaceSiblings;   // matches face-verts of level (4*N)

    //  Per-value:
    std::vector<Index> _vertValueIndices;   // variable per vertex (4*M>N)
    std::vector<ValueTag> _vertValueTags;      // variable per vertex (1*M>N)
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
