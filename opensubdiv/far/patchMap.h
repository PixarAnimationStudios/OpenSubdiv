//
//     Copyright 2013 Pixar
//
//     Licensed under the Apache License, Version 2.0 (the "License");
//     you may not use this file except in compliance with the License
//     and the following modification to it: Section 6 Trademarks.
//     deleted and replaced with:
//
//     6. Trademarks. This License does not grant permission to use the
//     trade names, trademarks, service marks, or product names of the
//     Licensor and its affiliates, except as required for reproducing
//     the content of the NOTICE file.
//
//     You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
//     Unless required by applicable law or agreed to in writing,
//     software distributed under the License is distributed on an
//     "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
//     either express or implied.  See the License for the specific
//     language governing permissions and limitations under the
//     License.
//

#ifndef FAR_PATCH_MAP_H
#define FAR_PATCH_MAP_H

#include "../version.h"

#include "../far/patchTables.h"

#include <cassert>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

/// \brief An quadtree-based map connecting coarse faces to their sub-patches
///
/// FarPatchTables::PatchArrays contain lists of patches that represent the limit
/// surface of a mesh, sorted by their topological type. These arrays break the
/// connection between coarse faces and their sub-patches. 
///
/// The PatchMap provides a quad-tree based lookup structure that, given a singular
/// parametric location, can efficiently return a handle to the sub-patch that
/// contains this location.
///
class FarPatchMap {
public:

    /// \brief Handle that can be used as unique patch identifier within FarPatchTables
    struct Handle {
        unsigned int patchArrayIdx,  // OsdPatchArray containing the patch
                     patchIdx,       // Absolute index of the patch
                     vertexOffset;   // Offset to the first CV of the patch
    };

    /// \brief Constructor
    ///
    /// @param patchTables  A valid set of FarPatchTables
    ///
    FarPatchMap( FarPatchTables const & patchTables );

    /// \brief Returns a handle to the sub-patch of the face at the given (u,v).
    /// Note : the faceid corresponds to quadrangulated face indices (ie. quads
    /// count as 1 index, non-quads add as many indices as they have vertices)
    ///
    /// @param faceid  The index of the face
    ///
    /// @param u       Local u parameter
    ///
    /// @param v       Local v parameter
    ///
    /// @return        A patch handle or NULL if the face does not exist or the
    ///                limit surface is tagged as a hole at the given location
    ///
    Handle const * FindPatch( int faceid, float u, float v ) const;
    
private:
    void initialize( FarPatchTables const & patchTables );

    // Quadtree node with 4 children
    struct QuadNode {
        struct Child {
            unsigned int isSet:1,    // true if the child has been set
                         isLeaf:1,   // true if the child is a QuadNode
                         idx:30;     // child index (either QuadNode or Handle)
        };

        // sets all the children to point to the patch of index patchIdx
        void SetChild(int patchIdx);

        // sets the child in "quadrant" to point to the node or patch of the given index
        void SetChild(unsigned char quadrant, int child, bool isLeaf=true);
        
        Child children[4];
    };

    typedef std::vector<QuadNode> QuadTree;

    // adds a child to a parent node and pushes it back on the tree
    static QuadNode * addChild( QuadTree & quadtree, QuadNode * parent, int quadrant );
    
    // given a median, transforms the (u,v) to the quadrant they point to, and
    // return the quadrant index.
    //
    // Quadrants indexing:
    //
    //   (0,0) o-----o-----o
    //         |     |     |
    //         |  0  |  3  |
    //         |     |     |
    //         o-----o-----o
    //         |     |     |
    //         |  1  |  2  |
    //         |     |     |
    //         o-----o-----o (1,1)
    //
    template <class T> static int resolveQuadrant(T & median, T & u, T & v);

    std::vector<Handle>   _handles;  // all the patches in the FarPatchTable
    std::vector<QuadNode> _quadtree; // quadtree nodes
};

// Constructor
inline
FarPatchMap::FarPatchMap( FarPatchTables const & patchTables ) {
    initialize( patchTables );
}

// sets all the children to point to the patch of index patchIdx
inline void
FarPatchMap::QuadNode::SetChild(int patchIdx) {
    for (int i=0; i<4; ++i) {
        children[i].isSet=true;
        children[i].isLeaf=true;
        children[i].idx=patchIdx;
    }
}

// sets the child in "quadrant" to point to the node or patch of the given index
inline void 
FarPatchMap::QuadNode::SetChild(unsigned char quadrant, int idx, bool isLeaf) {
    assert(quadrant<4);
    children[quadrant].isSet  = true;
    children[quadrant].isLeaf = isLeaf;
    children[quadrant].idx    = idx;
}

// adds a child to a parent node and pushes it back on the tree
inline FarPatchMap::QuadNode * 
FarPatchMap::addChild( QuadTree & quadtree, QuadNode * parent, int quadrant ) {
    quadtree.push_back(QuadNode());
    int idx = (int)quadtree.size()-1;
    parent->SetChild(quadrant, idx, false);
    return &(quadtree[idx]);
}

// given a median, transforms the (u,v) to the quadrant they point to, and
// return the quadrant index.
template <class T> int 
FarPatchMap::resolveQuadrant(T & median, T & u, T & v) {
    int quadrant = -1;

    if (u<median) {
        if (v<median) {
            quadrant = 0;
        } else {
            quadrant = 1;
            v-=median;
        }
    } else {
        if (v<median) {
            quadrant = 3;
        } else {
            quadrant = 2;
            v-=median;
        }
        u-=median;
    }
    return quadrant;
}

/// Returns a handle to the sub-patch of the face at the given (u,v).
inline FarPatchMap::Handle const * 
FarPatchMap::FindPatch( int faceid, float u, float v ) const {
    
    if (faceid>(int)_quadtree.size())
        return NULL;

    assert( (u>=0.0f) and (u<=1.0f) and (v>=0.0f) and (v<=1.0f) );

    QuadNode const * node = &_quadtree[faceid];

    float half = 0.5f;

    // 0xFF : we should never have depths greater than k_InfinitelySharp
    for (int depth=0; depth<0xFF; ++depth) {

        float delta = half * 0.5f;
        
        int quadrant = resolveQuadrant( half, u, v );
        assert(quadrant>=0);
        
        // is the quadrant a hole ?
        if (not node->children[quadrant].isSet)
            return 0;
        
        if (node->children[quadrant].isLeaf) {
            return &_handles[node->children[quadrant].idx];
        } else {
            node = &_quadtree[node->children[quadrant].idx];
        }
        
        half = delta;
    }

    assert(0);
    return 0;
}

// Constructor
inline void
FarPatchMap::initialize( FarPatchTables const & patchTables ) {

    int nfaces = 0, npatches = (int)patchTables.GetNumPatches();
        
    if (not npatches)
        return;
        
    FarPatchTables::PatchArrayVector const & patchArrays =
        patchTables.GetPatchArrayVector();

    FarPatchTables::PatchParamTable const & paramTable =
        patchTables.GetPatchParamTable();

    // populate subpatch handles vector
    _handles.resize(npatches);
    for (int arrayIdx=0, current=0; arrayIdx<(int)patchArrays.size(); ++arrayIdx) {
    
        FarPatchTables::PatchArray const & parray = patchArrays[arrayIdx];

        int ringsize = parray.GetDescriptor().GetNumControlVertices();
        
        for (unsigned int j=0; j < parray.GetNumPatches(); ++j) {
            
            FarPatchParam const & param = paramTable[parray.GetPatchIndex()+j];
            
            Handle & h = _handles[current];

            h.patchArrayIdx = arrayIdx;
            h.patchIdx      = (unsigned int)current;
            h.vertexOffset  = j * ringsize;

            nfaces = std::max(nfaces, (int)param.faceIndex);
            
            ++current;
        }
    }
    ++nfaces;

    // temporary vector to hold the quadtree while under construction
    std::vector<QuadNode> quadtree;

    // reserve memory for the octree nodes (size is a worse-case approximation)
    quadtree.reserve( nfaces + npatches );
    
    // each coarse face has a root node associated to it that we need to initialize
    quadtree.resize(nfaces);
    
    // populate the quadtree from the FarPatchArrays sub-patches
    for (int i=0, handleIdx=0; i<(int)patchArrays.size(); ++i) {
    
        FarPatchTables::PatchArray const & parray = patchArrays[i];

        for (unsigned int j=0; j < parray.GetNumPatches(); ++j, ++handleIdx) {
        
            FarPatchParam const & param = paramTable[parray.GetPatchIndex()+j];

            FarPatchParam::BitField bits = param.bitField;

            unsigned char depth = bits.GetDepth();
            
            QuadNode * node = &quadtree[ param.faceIndex ];
            
            if (depth==(bits.NonQuadRoot() ? 1 : 0)) {
                // special case : regular BSpline face w/ no sub-patches
                node->SetChild( handleIdx );
                continue;
            } 
                  
            short u = bits.GetU(),
                  v = bits.GetV(),
                  pdepth = bits.NonQuadRoot() ? depth-2 : depth-1,
                  half = 1 << pdepth;
            
            for (unsigned char k=0; k<depth; ++k) {

                short delta = half >> 1;
                
                int quadrant = resolveQuadrant(half, u, v);
                assert(quadrant>=0);

                half = delta;

                if (k==pdepth) {
                   // we have reached the depth of the sub-patch : add a leaf
                   assert( not node->children[quadrant].isSet );
                   node->SetChild(quadrant, handleIdx, true);
                   break;
                } else {
                    // travel down the child node of the corresponding quadrant
                    if (not node->children[quadrant].isSet) {
                        // create a new branch in the quadrant
                        node = addChild(quadtree, node, quadrant);
                    } else {
                        // travel down an existing branch
                        node = &(quadtree[ node->children[quadrant].idx ]);
                    }
                }
            }
        }
    }

    // copy the resulting quadtree to eliminate un-unused vector capacity
    _quadtree = quadtree;
}

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* FAR_PATCH_PARAM */
