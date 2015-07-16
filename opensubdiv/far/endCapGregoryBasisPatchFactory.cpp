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

#include "../far/gregoryBasis.h"
#include "../far/endCapGregoryBasisPatchFactory.h"
#include "../far/error.h"
#include "../far/stencilTableFactory.h"
#include "../far/topologyRefiner.h"

#include <cassert>
#include <cmath>
#include <cstring>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Far {

//
// EndCapGregoryBasisPatchFactory for Vertex StencilTable
//
EndCapGregoryBasisPatchFactory::EndCapGregoryBasisPatchFactory(
    TopologyRefiner const & refiner, bool shareBoundaryVertices) :
    _refiner(&refiner), _shareBoundaryVertices(shareBoundaryVertices),
    _numGregoryBasisVertices(0), _numGregoryBasisPatches(0) {

    // Sanity check: the mesh must be adaptively refined
    assert(not refiner.IsUniform());
}

//
// Stateless EndCapGregoryBasisPatchFactory
//
GregoryBasis const *
EndCapGregoryBasisPatchFactory::Create(TopologyRefiner const & refiner,
    Index faceIndex, int fvarChannel) {

    // Gregory patches are end-caps: they only exist on max-level
    Vtr::internal::Level const & level = refiner.getLevel(refiner.GetMaxLevel());

    GregoryBasis::ProtoBasis basis(level, faceIndex, 0, fvarChannel);
    GregoryBasis * result = new GregoryBasis;
    basis.Copy(result);

    // note: this function doesn't create varying stencils.
    return result;
}

bool
EndCapGregoryBasisPatchFactory::addPatchBasis(Index faceIndex,
                                              bool verticesMask[4][5],
                                              int levelVertOffset) {

    // Gregory patches only exist on the hight
    Vtr::internal::Level const & level = _refiner->getLevel(_refiner->GetMaxLevel());

    // Gather the CVs that influence the Gregory patch and their relative
    // weights in a basis
    GregoryBasis::ProtoBasis basis(level, faceIndex, levelVertOffset, -1);

    for (int i = 0; i < 4; ++i) {
        if (verticesMask[i][0]) {
            _vertexStencils.push_back(basis.P[i]);
            _varyingStencils.push_back(basis.V[i]);
        }
        if (verticesMask[i][1]) {
            _vertexStencils.push_back(basis.Ep[i]);
            _varyingStencils.push_back(basis.V[i]);
        }
        if (verticesMask[i][2]) {
            _vertexStencils.push_back(basis.Em[i]);
            _varyingStencils.push_back(basis.V[i]);
        }
        if (verticesMask[i][3]) {
            _vertexStencils.push_back(basis.Fp[i]);
            _varyingStencils.push_back(basis.V[i]);
        }
        if (verticesMask[i][4]) {
            _vertexStencils.push_back(basis.Fm[i]);
            _varyingStencils.push_back(basis.V[i]);
        }
    }
    return true;
}

//
// Populates the topology table used by Gregory-basis patches
//
// Note  : 'faceIndex' values are expected to be sorted in ascending order !!!
// Note 2: this code attempts to identify basis vertices shared along
//         gregory patch edges
ConstIndexArray
EndCapGregoryBasisPatchFactory::GetPatchPoints(
    Vtr::internal::Level const * level, Index faceIndex,
    PatchTableFactory::PatchFaceTag const * levelPatchTags,
    int levelVertOffset) {
    // allocate indices (awkward)
    // assert(Vtr::INDEX_INVALID==0xFFFFFFFF);
    for (int i = 0; i < 20; ++i) {
        _patchPoints.push_back(Vtr::INDEX_INVALID);
    }
    Index * dest = &_patchPoints[_numGregoryBasisPatches * 20];

    int gregoryVertexOffset = _refiner->GetNumVerticesTotal();

    if (_shareBoundaryVertices) {
        ConstIndexArray fedges = level->getFaceEdges(faceIndex);
        assert(fedges.size()==4);

        for (int i=0; i<4; ++i) {
            Index edge = fedges[i], adjface = 0;

            { // Gather adjacent faces
                ConstIndexArray adjfaces = level->getEdgeFaces(edge);
                for (int j=0; j<adjfaces.size(); ++j) {
                    if (adjfaces[j]==faceIndex) {
                        // XXXX manuelk if 'edge' is non-manifold, arbitrarily pick the
                        // next face in the list of adjacent faces
                        adjface = (adjfaces[(j+1)%adjfaces.size()]);
                        break;
                    }
                }
            }
            // We are looking for adjacent faces that:
            // - exist (no boundary)
            // - have already been processed (known CV indices)
            // - are also Gregory basis patches
            if (adjface!=Vtr::INDEX_INVALID and (adjface < faceIndex) and
                (not levelPatchTags[adjface]._isRegular)) {

                ConstIndexArray aedges = level->getFaceEdges(adjface);
                int aedge = aedges.FindIndexIn4Tuple(edge);
                assert(aedge!=Vtr::INDEX_INVALID);

                // Find index of basis in the list of basis already generated
                struct compare {
                    static int op(void const * a, void const * b) {
                        return *(Index const*)a - *(Index const*)b;
                    }
                };

                Index * ptr = (Index *)std::bsearch(&adjface,
                                                    &_faceIndices[0],
                                                    _faceIndices.size(),
                                                    sizeof(Index), compare::op);

                int srcBasisIdx = (int)(ptr - &_faceIndices[0]);

                if (!ptr) {
                    // if the adjface is hole, it won't be found
                    break;
                }
                assert(ptr
                       and srcBasisIdx>=0
                       and srcBasisIdx<(int)_faceIndices.size());

                // Copy the indices of CVs from the face on the other side of the
                // shared edge
                static int const gregoryEdgeVerts[4][4] = { { 0,  1,  7,  5},
                                                            { 5,  6, 12, 10},
                                                            {10, 11, 17, 15},
                                                            {15, 16,  2,  0} };
                Index * src = &_patchPoints[srcBasisIdx*20];
                for (int j=0; j<4; ++j) {
                    // invert direction
                    // note that src  indices have already been offsetted.
                    dest[gregoryEdgeVerts[i][3-j]] = src[gregoryEdgeVerts[aedge][j]];
                }
            }
        }
    }

    bool newVerticesMask[4][5];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 5; ++j) {
            if (dest[i*5+j]==Vtr::INDEX_INVALID) {
                // assign new vertex
                dest[i*5+j] =
                    _numGregoryBasisVertices + gregoryVertexOffset;
                ++_numGregoryBasisVertices;
                newVerticesMask[i][j] = true;
            } else {
                // share vertex
                newVerticesMask[i][j] = false;
            }
        }
    }
    _faceIndices.push_back(faceIndex);

    // add basis
    addPatchBasis(faceIndex, newVerticesMask, levelVertOffset);

    ++_numGregoryBasisPatches;

    // return cvs;
    return ConstIndexArray(dest, 20);
}

} // end namespace Far

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv
