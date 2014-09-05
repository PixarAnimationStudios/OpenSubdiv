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
#ifndef VTR_REFINEMENT_H
#define VTR_REFINEMENT_H

#include "../version.h"

#include "../sdc/type.h"
#include "../sdc/options.h"
#include "../vtr/types.h"
#include "../vtr/level.h"

#include <vector>

//
//  Declaration for the main refinement class (Refinement) and its pre-requisites:
//
namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Far {
    class TopologyRefiner;
    class PatchTablesFactory;
}

namespace Vtr {

class SparseSelector;
class FVarRefinement;

//
//  Refinement:
//      A refinement is a mapping between two levels -- relating the components in the original
//  (parent) level to the one refined (child).  The refinement may be complete (uniform) or sparse
//  (adaptive or otherwise selective), so not all components in the parent level will spawn
//  components in the child level.
//
//  At a high level, all that is necessary in terms of interface is to construct, initialize
//  (linking the two levels), optionally select components for sparse refinement (via use of the
//  SparseSelector) and call the refine() method.  This is the usage expected in FarTopologyRefiner.
//
//  Since we really want this class to be restricted from public access eventually, all methods
//  begin with lower case (as is the convention for protected methods) and the list of friends
//  will be maintained more strictly.
//
class Refinement {

public:
    Refinement();
    ~Refinement();

    void setScheme(Sdc::Type const& schemeType, Sdc::Options const& schemeOptions);
    void initialize(Level& parent, Level& child);

    Level const& parent() const { return *_parent; }

    Level const& child() const  { return *_child; }
    Level&       child()        { return *_child; }

    //
    //  Options associated with the actual refinement operation, which are going to get
    //  quite involved to ensure that the refinement of data that is not of interest can
    //  be suppressed.  For now we have:
    //
    //      "sparse": the alternative to uniform refinement, which requires that
    //          components be previously selected/marked to be included.
    //
    //      "face topology only": this is one that may get broken down into a finer
    //          set of options.  It suppresses "full topology" in the child level
    //          and only generates what is necessary to define the list of faces.
    //          This is only one of the six possible topological relations that
    //          can be generated -- we may eventually want a flag for each.
    //
    //      "compute masks": this is intended to be temporary, along with the data
    //          members associated with it -- it will trigger the computation and
    //          storage of mask weights for all child vertices.  This is naively
    //          stored at this point and exists only for reference.
    //
    //  Its still up for debate as to how finely these should be controlled, e.g.
    //  for sparse refinement, we likely want full topology at the finest level to
    //  allow for subsequent patch construction...
    //
    struct Options {
        Options() : _sparse(0),
                    _faceTopologyOnly(0)
                    { }

        unsigned int _sparse           : 1;
        unsigned int _faceTopologyOnly : 1;

        //  Currently under consideration:
        //unsigned int _childToParentMap : 1;
        //unsigned int _computeMasks     : 1;
    };

    void refine(Options options = Options());

public:
    //
    //  Access to members -- some testing classes (involving vertex interpolation)
    //  currently make use of these:
    //
    int getNumChildVerticesFromFaces() const    { return _childVertFromFaceCount; }
    int getNumChildVerticesFromEdges() const    { return _childVertFromEdgeCount; }
    int getNumChildVerticesFromVertices() const { return _childVertFromVertCount; }

    Index getFaceChildVertex(Index f) const   { return _faceChildVertIndex[f]; }
    Index getEdgeChildVertex(Index e) const   { return _edgeChildVertIndex[e]; }
    Index getVertexChildVertex(Index v) const { return _vertChildVertIndex[v]; }

    IndexArray const getFaceChildFaces(Index parentFace) const;
    IndexArray const getFaceChildEdges(Index parentFace) const;
    IndexArray const getEdgeChildEdges(Index parentEdge) const;

    //  Child-to-parent relationships (not yet complete -- unclear how we will define the
    //  "type" of the parent component, e.g. vertex, edge or face):
    Index getChildFaceParentFace(Index f) const     { return _childFaceParentIndex[f]; }
    int      getChildFaceInParentFace(Index f) const   { return _childFaceTag[f]._indexInParent; }

    Index getChildEdgeParentIndex(Index e) const    { return _childEdgeParentIndex[e]; }

    Index getChildVertexParentIndex(Index v) const  { return _childVertexParentIndex[v]; }

//
//  Non-public methods:
//
protected:

    friend class FVarRefinement;
    friend class SparseSelector;

    friend class Far::TopologyRefiner;
    friend class Far::PatchTablesFactory;


    IndexArray getFaceChildFaces(Index parentFace);
    IndexArray getFaceChildEdges(Index parentFace);
    IndexArray getEdgeChildEdges(Index parentEdge);

protected:
    //
    //  Work in progress...
    //
    //  Tags have now been added per-component in Level, but there is additional need to tag
    //  components within Refinement -- we can't tag the parent level components for any
    //  refinement (in order to keep it const) and tags associated with children that are
    //  specific to the child-to-parent mapping may not be warranted in the child level.
    //
    //  Parent tags are only required for sparse refinement, and so a single SparseTag is used
    //  for all three component types.  The main property to tag is whether a component was
    //  selected.  Tagging if a component is "transitional" is also useful.  This may only be
    //  necessary for edges but is currently packed into a mask per-edge for faces -- that may
    //  be deferred, in which case "transitional" can be a single bit.
    //
    //  Child tags are to become part of the new child-to-parent mapping, which is to consist
    //  of the parent component index for each child component, plus a set of tags for the child
    //  indicating more about its relationship to its parent, e.g. is it completely defined,
    //  what the parent component type is, what is the index of the chile within its parent,
    //  etc.
    //
    struct SparseTag {
        SparseTag() : _selected(0), _transitional(0) { }

        unsigned char _selected     : 1;  // component specifically selected for refinement
        unsigned char _transitional : 4;  // adjacent to a refined component (4-bits for face)
    };

    struct ChildTag {
        ChildTag() { }

        unsigned char _incomplete    : 1;  // incomplete neighborhood to represent limit of parent
        unsigned char _parentType    : 2;  // type of parent component:  vertex, edge or face
        unsigned char _indexInParent : 2;  // index of child wrt parent:  0-3, or iterative if N > 4
    };

//
//  Remaining methods really should remain private...
//
private:
    //
    //  Methods involved in constructing the parent-to-child mapping -- when the
    //  refinement is sparse, additional methods are needed to identify the selection:
    //
    void initializeSparseSelectionTags();
    void markSparseChildComponents();

    void allocateParentToChildMapping();
    void populateParentToChildMapping();
    void printParentToChildMapping() const;

    //
    //  Methods involved in constructing the child-to-parent mapping:
    //
    void createChildComponents();
    void populateChildToParentTags();
    void populateChildToParentIndices();
    void populateChildToParentMapping();

    //
    //  Methods involved in subdividing/propagating component tags and sharpness:
    //
    void propagateComponentTags();

    void propagateFaceTagsFromParentFaces();
    void propagateEdgeTagsFromParentFaces();
    void propagateEdgeTagsFromParentEdges();
    void propagateVertexTagsFromParentFaces();
    void propagateVertexTagsFromParentEdges();
    void propagateVertexTagsFromParentVertices();

    void subdivideVertexSharpness();
    void subdivideEdgeSharpness();
    void reclassifySemisharpVertices();

    //
    //  Methods involved in subdividing face-varying topology:
    //
    void subdivideFVarChannels();

    //
    //  Methods (and types) involved in subdividing the topology:
    //
    //  Simple struct defining the types of topological relations to subdivide:
    struct Relations {
        unsigned int   _faceVertices : 1;
        unsigned int   _faceEdges    : 1;
        unsigned int   _edgeVertices : 1;
        unsigned int   _edgeFaces    : 1;
        unsigned int   _vertexFaces  : 1;
        unsigned int   _vertexEdges  : 1;

        void setAll(bool value) {
            _faceVertices = value;
            _faceEdges    = value;
            _edgeVertices = value;
            _edgeFaces    = value;
            _vertexFaces  = value;
            _vertexEdges  = value;
        }
    };

    void subdivideTopology(Relations const& relationsToSubdivide);

    //  Methods for sizing the child topology vectors (note we only need four of the
    //  expected six here as face-edge shares face-vert and edge-vert is trivial):
    void initializeFaceVertexCountsAndOffsets();
    void initializeEdgeFaceCountsAndOffsets();
    void initializeVertexFaceCountsAndOffsets();
    void initializeVertexEdgeCountsAndOffsets();

    //  Methods for populating sections of child topology relations based on their origin
    //  in the parent -- 12 in all
    //
    //  These iterate through all components -- currently iterating through the parents
    //  and updating only those children marked valid.  We may want to change the iteration
    //  strategy here, particularly for sparse refinement.  Plans are to factor these out
    //  to work between a specific parent and child pair and control the iteration through
    //  other means.
    //
    void populateFaceVerticesFromParentFaces();
    void populateFaceEdgesFromParentFaces();
    void populateEdgeVerticesFromParentFaces();
    void populateEdgeVerticesFromParentEdges();
    void populateEdgeFacesFromParentFaces();
    void populateEdgeFacesFromParentEdges();
    void populateVertexFacesFromParentFaces();
    void populateVertexFacesFromParentEdges();
    void populateVertexFacesFromParentVertices();
    void populateVertexEdgesFromParentFaces();
    void populateVertexEdgesFromParentEdges();
    void populateVertexEdgesFromParentVertices();

private:
    friend class Level;  //  Access for some debugging information

    Level* _parent;
    Level* _child;

    Sdc::Type    _schemeType;
    Sdc::Options _schemeOptions;

    bool _quadSplit;  // generalize this to Sdc::Split later
    bool _uniform;

    //
    //  Inventory of the types of child components:
    //      There are six types of child components:  child faces can only originate from
    //  faces, child edges can originate from faces and edges, while child vertices can
    //  originate from all three component types.
    //
    //  While the Refinement populates the Level containing the children, knowing the
    //  counts for the originating type, and generating them in blocks according to these
    //  types, allows us to process them more specifically.
    //
    int _childFaceFromFaceCount;  // arguably redundant (all faces originate from faces)
    int _childEdgeFromFaceCount;
    int _childEdgeFromEdgeCount;
    int _childVertFromFaceCount;
    int _childVertFromEdgeCount;
    int _childVertFromVertCount;

    //
    //  Members involved in the parent-to-child mapping:
    //      These are vectors sized according to the number of parent components (and
    //  their topology) that contain references/indices to the child components that
    //  result from them by refinement.  When refinement is sparse, parent components
    //  that have not spawned all child components will have their missing children
    //  marked as invalid.
    //
    //  Vectors for each of the six child component types are sized according to a
    //  topological relation of the originating parent type, i.e. for child faces and
    //  edges originating from parent faces, there will be a 1-to-1 correspondence
    //  between a parents face-verts and its child faces (not true for Loop) and
    //  similarly its child edges.  Given this correspondence with the topology vectors,
    //  we use the same counts/offsets of the topology vectors when we can.
    //
    IndexVector _faceChildFaceIndices;  // *cannot* always use face-vert counts/offsets
    IndexVector _faceChildEdgeIndices;  // can use face-vert counts/offsets
    IndexVector _faceChildVertIndex;

    IndexVector _edgeChildEdgeIndices;  // trivial/corresponding pair for each
    IndexVector _edgeChildVertIndex;

    IndexVector _vertChildVertIndex;

    //
    //  Members involved in the child-to-parent mapping:
    //      In many cases just having the parent-to-child mapping is adequate to refine
    //  everything we need, but some analysis of the resulting refinement wants to "walk
    //  up the hierarchy" for a component to gather information about its ancestry (e.g.
    //  PTex coordinates).  It is also useful when the refinement is sparse -- the more
    //  sparse, the more it makes sense to iterate through the child components and seek
    //  their parents rather than the reverse.
    //
    //  So the reverse child-to-parent mapping definitely useful, but it is relatively
    //  costly (both space and time) to initialize and propagate though the hierarchy, so
    //  I would prefer to only enable it when needed, i.e. make it optional.
    //
    //  The mapping itself is simply an integer index for each child component, along with
    //  a corresponding tag for each with information about its parent:  is the child
    //  fully defined wrt its parent (when refinement is sparse), what is the index of the
    //  child within its parent, and what is the type of the parent.  (The latter can be
    //  inferred from the index of the child and so may be redudant.)
    //
    //
    IndexVector _childFaceParentIndex;
    IndexVector _childEdgeParentIndex;
    IndexVector _childVertexParentIndex;

    std::vector<ChildTag> _childFaceTag;
    std::vector<ChildTag> _childEdgeTag;
    std::vector<ChildTag> _childVertexTag;

    //
    //  Additional tags per parent component are also used when refinement is sparse.
    //
    std::vector<SparseTag> _parentFaceTag;
    std::vector<SparseTag> _parentEdgeTag;
    std::vector<SparseTag> _parentVertexTag;

    //  References to components in the base level (top-most ancestor) may be useful
    //  to copy non-interpolatible properties to all descendant components.

    //  Refinement data for face-varying channels:
    std::vector<FVarRefinement*> _fvarChannels;

public:
    //  TEMPORARY -- FOR ILLUSTRATIVE PURPOSES ONLY...
    //
    //  Mask for the child vertices stored relative to parent topology, i.e. weights
    //  for a child face-vertex are stored relative to the parent face -- a weight for
    //  each of the parent face's vertices.
    //
    //  Currently the full complement of weights is used and expected to be applied, e.g.
    //  for a crease in the interior, there may be no face-vert weights in the stencil
    //  and so no need to apply them, but we require all and so set these to zero for now.
    //  We will need some kind of stencil type associated with each child vertex if we
    //  are to avoid doing so in order to detect the difference.
    //
    //  Note this is potentially extremely wasteful in terms of memory when the set of
    //  refined components in the child is small relative to the parent.  All possible
    //  stencil weights (i.e. for uniform refinement) will be allocated here if the
    //  corresonding counts/offset of the parent are to be used.
    //
//#define _VTR_COMPUTE_MASK_WEIGHTS_ENABLED
#ifdef _VTR_COMPUTE_MASK_WEIGHTS_ENABLED
    void computeMaskWeights();

    std::vector<float> _faceVertWeights;  // matches parent face vert counts and offsets
    std::vector<float> _edgeVertWeights;  // trivially 2 per parent edge
    std::vector<float> _edgeFaceWeights;  // matches parent edge face counts and offsets
    std::vector<float> _vertVertWeights;  // trivially 1 per parent vert
    std::vector<float> _vertEdgeWeights;  // matches parent vert edge counts and offsets
    std::vector<float> _vertFaceWeights;  // matches parent vert face counts and offsets
#endif
};

inline IndexArray const
Refinement::getFaceChildFaces(Index parentFace) const {

    //
    //  Note this will need to vary based on the topological split applied...
    //
    return IndexArray(&_faceChildFaceIndices[_parent->getOffsetOfFaceVertices(parentFace)],
                                                _parent->getNumFaceVertices(parentFace));
}
inline IndexArray
Refinement::getFaceChildFaces(Index parentFace) {

    return IndexArray(&_faceChildFaceIndices[_parent->getOffsetOfFaceVertices(parentFace)],
                                                _parent->getNumFaceVertices(parentFace));
}

inline IndexArray const
Refinement::getFaceChildEdges(Index parentFace) const {

    //
    //  Note this *may* need to vary based on the topological split applied...
    //
    return IndexArray(&_faceChildEdgeIndices[_parent->getOffsetOfFaceVertices(parentFace)],
                                                _parent->getNumFaceVertices(parentFace));
}
inline IndexArray
Refinement::getFaceChildEdges(Index parentFace) {

    return IndexArray(&_faceChildEdgeIndices[_parent->getOffsetOfFaceVertices(parentFace)],
                                                _parent->getNumFaceVertices(parentFace));
}

inline IndexArray const
Refinement::getEdgeChildEdges(Index parentEdge) const {

    return IndexArray(&_edgeChildEdgeIndices[parentEdge*2], 2);
}

inline IndexArray
Refinement::getEdgeChildEdges(Index parentEdge) {

    return IndexArray(&_edgeChildEdgeIndices[parentEdge*2], 2);
}

} // end namespace Vtr

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;
} // end namespace OpenSubdiv

#endif /* VTR_REFINEMENT_H */
