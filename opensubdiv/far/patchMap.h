//
//   Copyright 2013 Pixar
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

#ifndef OPENSUBDIV3_FAR_PATCH_MAP_H
#define OPENSUBDIV3_FAR_PATCH_MAP_H

#include "../version.h"

#include "../far/patchTable.h"

#include <cassert>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Far {

/// \brief An quadtree-based map connecting coarse faces to their sub-patches
///
/// PatchTable::PatchArrays contain lists of patches that represent the limit
/// surface of a mesh, sorted by their topological type. These arrays break the
/// connection between coarse faces and their sub-patches.
///
/// The PatchMap provides a quad-tree based lookup structure that, given a singular
/// parametric location, can efficiently return a handle to the sub-patch that
/// contains this location.
///
class PatchMap {
public:

    typedef PatchTable::PatchHandle Handle;

    /// \brief Constructor
    ///
    /// @param patchTable  A valid set of PatchTable
    ///
    PatchMap( PatchTable const & patchTable );

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
    Handle const * FindPatch( int faceid, double u, double v ) const;

private:

    inline void initialize( PatchTable const & patchTable );

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

    // identify and transform a (u,v) pair by its containing quadrant
    template <class T>
    static int transformToQuadQuadrant(T const & median, T & u, T & v);
    template <class T>
    static int transformToTriQuadrant(T const & median, T & u, T & v, bool & rotated);

private:

    bool _patchesAreTriangular;

    std::vector<Handle>   _handles;  // all the patches in the PatchTable
    std::vector<QuadNode> _quadtree; // quadtree nodes
};

//
//  Given a median value for both U and V, these methods transform a (u,v) pair
//  into the quadrant that contains them and returns the quadrant index.
//
//  Quadrant indexing for tri and quad patches -- consistent with PatchParam's
//  usage of UV bits:
//
//      (0,1) o                       (0,1) o-----o-----o (1,1)
//            |\                            |     |     |
//            |  \                          |  2  |  3  |
//            | 2  \                        |     |     |
//            o-----o                       o-----o-----o
//            |\  3 |\                      |     |     |
//            |  \  |  \                    |  0  |  1  |
//            | 0  \| 1  \                  |     |     |
//      (0,0) o-----o-----o (1,0)     (0,0) o-----o-----o (1,0)
//
//  The triangular case also takes and returns/affects the rotation of the
//  quadrant being searched and identified (quadrant 3 imparts a rotation).
//
template <class T>
inline int
PatchMap::transformToQuadQuadrant(T const & median, T & u, T & v) {

    int uHalf = (u >= median);
    if (uHalf) u -= median;

    int vHalf = (v >= median);
    if (vHalf) v -= median;

    return (vHalf << 1) | uHalf;
}
    
template <class T>
int inline
PatchMap::transformToTriQuadrant(T const & median, T & u, T & v, bool & rotated) {
    
    if (!rotated) {
        if (u >= median) {
            u -= median;
            return 1;
        }
        if (v >= median) {
            v -= median;
            return 2;
        }
        if ((u + v) >= median) {
            rotated = !rotated;
            return 3;
        }
        return 0;
    } else {
        if (u < median) return 1;
        if (v < median) return 2;
        u -= median;
        v -= median;
        if ((u + v) >= median) {
            rotated = !rotated;
            return 0;
        }
        return 3;
    }
}

/// Returns a handle to the sub-patch of the face at the given (u,v).
inline PatchMap::Handle const *
PatchMap::FindPatch( int faceid, double u, double v ) const {

    if (faceid>=(int)_quadtree.size())
        return 0;

    assert( (u>=0.0) && (u<=1.0) && (v>=0.0) && (v<=1.0) );

    QuadNode const * node = &_quadtree[faceid];

    bool triRotated = false;

    double half = 0.5;

    // Patch depth is limited to 4 bits by PatchParam
    int maxDepth = (1 << 4);
    for (int depth = 0; depth < maxDepth; ++depth, half *= 0.5) {

        int quadrant = _patchesAreTriangular
                     ? transformToTriQuadrant(half, u, v, triRotated)
                     : transformToQuadQuadrant(half, u, v);

        // is the quadrant a hole ?
        if (! node->children[quadrant].isSet)
            return 0;

        if (node->children[quadrant].isLeaf) {
            return &_handles[node->children[quadrant].idx];
        } else {
            node = &_quadtree[node->children[quadrant].idx];
        }
    }
    assert(0);
    return 0;
}

} // end namespace Far

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* OPENSUBDIV3_FAR_PATCH_PARAM */
