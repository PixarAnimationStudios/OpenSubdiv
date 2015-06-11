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

#include "../far/patchMap.h"

#include <algorithm>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Far {

// Constructor
PatchMap::PatchMap( PatchTable const & patchTable ) {
    initialize( patchTable );
}

// sets all the children to point to the patch of index patchIdx
void
PatchMap::QuadNode::SetChild(int patchIdx) {
    for (int i=0; i<4; ++i) {
        children[i].isSet=true;
        children[i].isLeaf=true;
        children[i].idx=patchIdx;
    }
}

// sets the child in "quadrant" to point to the node or patch of the given index
void
PatchMap::QuadNode::SetChild(unsigned char quadrant, int idx, bool isLeaf) {
    assert(quadrant<4);
    children[quadrant].isSet  = true;
    children[quadrant].isLeaf = isLeaf;
    children[quadrant].idx    = idx;
}

// adds a child to a parent node and pushes it back on the tree
PatchMap::QuadNode *
PatchMap::addChild( QuadTree & quadtree, QuadNode * parent, int quadrant ) {
    quadtree.push_back(QuadNode());
    int idx = (int)quadtree.size()-1;
    parent->SetChild(quadrant, idx, false);
    return &(quadtree[idx]);
}

void
PatchMap::initialize( PatchTable const & patchTable ) {

    int nfaces = 0,
        narrays = (int)patchTable.GetNumPatchArrays(),
        npatches = (int)patchTable.GetNumPatchesTotal();

    if (not narrays or not npatches)
        return;

    // populate subpatch handles vector
    _handles.resize(npatches);

    for (int parray=0, current=0; parray<narrays; ++parray) {

        ConstPatchParamArray params = patchTable.GetPatchParams(parray);

        int ringsize = patchTable.GetPatchArrayDescriptor(parray).GetNumControlVertices();

        for (Index j=0; j < patchTable.GetNumPatches(parray); ++j) {

            Handle & h = _handles[current];

            h.arrayIndex = parray;
            h.patchIndex = current;
            h.vertIndex  = j * ringsize;

            nfaces = std::max(nfaces, (int)params[j].GetFaceId());

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
    for (Index parray=0, handleIndex=0; parray<narrays; ++parray) {

        ConstPatchParamArray params = patchTable.GetPatchParams(parray);

        for (int i=0; i < patchTable.GetNumPatches(parray); ++i, ++handleIndex) {

            PatchParam const & param = params[i];

            unsigned short depth = param.GetDepth();

            QuadNode * node = &quadtree[ params[i].GetFaceId() ];

            if (depth==(param.NonQuadRoot() ? 1 : 0)) {
                // special case : regular BSpline face w/ no sub-patches
                node->SetChild( handleIndex );
                continue;
            }

            int u = param.GetU(),
                v = param.GetV(),
                pdepth = param.NonQuadRoot() ? depth-2 : depth-1,
                half = 1 << pdepth;

            for (unsigned char j=0; j<depth; ++j) {

                int delta = half >> 1;

                int quadrant = resolveQuadrant(half, u, v);
                assert(quadrant>=0);

                half = delta;

                if (j==pdepth) {
                   // we have reached the depth of the sub-patch : add a leaf
                   assert( not node->children[quadrant].isSet );
                   node->SetChild(quadrant, handleIndex, true);
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


} // end namespace Far

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv
