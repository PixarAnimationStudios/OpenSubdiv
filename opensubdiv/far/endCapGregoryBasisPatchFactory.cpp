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
    TopologyRefiner const & refiner,
    StencilTable * vertexStencils,
    StencilTable * varyingStencils,
    bool shareBoundaryVertices) :
    _vertexStencils(vertexStencils), _varyingStencils(varyingStencils),
    _refiner(&refiner), _shareBoundaryVertices(shareBoundaryVertices),
    _numGregoryBasisVertices(0), _numGregoryBasisPatches(0) {

    // Sanity check: the mesh must be adaptively refined
    assert(! refiner.IsUniform());

    // Reserve the patch point stencils. Ideally topology refiner
    // would have an API to return how many endcap patches will be required.
    // Instead we conservatively estimate by the number of patches at the
    // finest level.
    int numMaxLevelFaces = refiner.GetLevel(refiner.GetMaxLevel()).GetNumFaces();

    int numPatchPointsExpected = numMaxLevelFaces * 20;
    // limits to 100M (=800M bytes) entries for the reserved size.
    int numStencilsExpected = std::min(numPatchPointsExpected * 16,
                                       100*1024*1024);
    _vertexStencils->reserve(numPatchPointsExpected, numStencilsExpected);
    if (_varyingStencils) {
        // varying stencils use only 1 index with weight=1.0
        _varyingStencils->reserve(numPatchPointsExpected, numPatchPointsExpected);
    }
}

bool
EndCapGregoryBasisPatchFactory::addPatchBasis(Vtr::internal::Level const & level, Index faceIndex,
                                              Vtr::internal::Level::VSpan const cornerSpans[],
                                              bool verticesMask[4][5],
                                              int levelVertOffset,
                                              int fvarChannel) {

    // Gather the CVs that influence the Gregory patch and their relative
    // weights in a basis
    GregoryBasis::ProtoBasis basis(level, faceIndex, cornerSpans, levelVertOffset, fvarChannel);

    for (int i = 0; i < 4; ++i) {
        if (verticesMask[i][0]) {
            GregoryBasis::AppendToStencilTable(basis.P[i], _vertexStencils);
            if (_varyingStencils) {
                GregoryBasis::AppendToStencilTable(basis.varyingIndex[i], _varyingStencils);
            }
        }
        if (verticesMask[i][1]) {
            GregoryBasis::AppendToStencilTable(basis.Ep[i], _vertexStencils);
            if (_varyingStencils) {
                GregoryBasis::AppendToStencilTable(basis.varyingIndex[i], _varyingStencils);
            }
        }
        if (verticesMask[i][2]) {
            GregoryBasis::AppendToStencilTable(basis.Em[i], _vertexStencils);
            if (_varyingStencils) {
                GregoryBasis::AppendToStencilTable(basis.varyingIndex[i], _varyingStencils);
            }
        }
        if (verticesMask[i][3]) {
            GregoryBasis::AppendToStencilTable(basis.Fp[i], _vertexStencils);
            if (_varyingStencils) {
                GregoryBasis::AppendToStencilTable(basis.varyingIndex[i], _varyingStencils);
            }
        }
        if (verticesMask[i][4]) {
            GregoryBasis::AppendToStencilTable(basis.Fm[i], _vertexStencils);
            if (_varyingStencils) {
                GregoryBasis::AppendToStencilTable(basis.varyingIndex[i], _varyingStencils);
            }
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
    Vtr::internal::Level::VSpan const cornerSpans[],
    int levelVertOffset, int fvarChannel) {

    // allocate indices (awkward)
    // assert(Vtr::INDEX_INVALID==0xFFFFFFFF);
    for (int i = 0; i < 20; ++i) {
        _patchPoints.push_back(Vtr::INDEX_INVALID);
    }
    Index * dest = &_patchPoints[_numGregoryBasisPatches * 20];

    int gregoryVertexOffset = (fvarChannel < 0)
                            ? _refiner->GetNumVerticesTotal()
                            : _refiner->GetNumFVarValuesTotal(fvarChannel);

    if (_shareBoundaryVertices) {
        int levelIndex = level->getDepth();

        //  Simple struct with encoding of <level,face> index as an unsigned int and a
        //  comparison method for use with std::bsearch
        struct LevelAndFaceIndex {
            static inline unsigned int create(unsigned int levelIndex, Index faceIndex) {
                return (levelIndex << 28) | (unsigned int) faceIndex;
            }
            static int compare(void const * a, void const * b) {
                return *(unsigned int const*)a - *(unsigned int const*)b;
            }
        };

        ConstIndexArray fedges = level->getFaceEdges(faceIndex);
        assert(fedges.size()==4);

        Vtr::internal::Level::ETag etags[4];
        level->getFaceETags(faceIndex, etags, fvarChannel);

        for (int i=0; i<4; ++i) {
            // Ignore boundary edges (or those with a face-varying discontinuity)
            if (etags[i]._boundary) continue;

            Index edge = fedges[i];
            Index adjFaceIndex = 0;

            { // Gather adjacent faces
                ConstIndexArray adjfaces = level->getEdgeFaces(edge);
                for (int j=0; j<adjfaces.size(); ++j) {
                    if (adjfaces[j]==faceIndex) {
                        // XXXX manuelk if 'edge' is non-manifold, arbitrarily pick the
                        // next face in the list of adjacent faces
                        adjFaceIndex = (adjfaces[(j+1)%adjfaces.size()]);
                        break;
                    }
                }
            }
            // We are looking for adjacent faces that:
            // - exist (no boundary)
            // - have already been processed (known CV indices)
            // - are also Gregory basis patches
            if ((adjFaceIndex != Vtr::INDEX_INVALID) && (adjFaceIndex < faceIndex)) {

                if (_levelAndFaceIndices.empty()) {
                    break;
                }

                ConstIndexArray aedges = level->getFaceEdges(adjFaceIndex);
                int aedge = aedges.FindIndexIn4Tuple(edge);
                assert(aedge!=Vtr::INDEX_INVALID);

                // Find index of basis in the list of bases already generated
                unsigned int adjLevelAndFaceIndex = LevelAndFaceIndex::create(levelIndex, adjFaceIndex);
                unsigned int * ptr = (unsigned int *)std::bsearch(&adjLevelAndFaceIndex,
                                                                  &_levelAndFaceIndices[0],
                                                                 _levelAndFaceIndices.size(),
                                                                 sizeof(unsigned int),
                                                                 LevelAndFaceIndex::compare);
                if (ptr == 0) {
                    break;
                }

                int adjPatchIndex = (int)(ptr - &_levelAndFaceIndices[0]);
                assert(adjPatchIndex>=0 && adjPatchIndex<(int)_levelAndFaceIndices.size());

                // Copy the indices of CVs from the face on the other side of the shared edge
                static int const gregoryEdgeVerts[4][4] = { { 0,  1,  7,  5},
                                                            { 5,  6, 12, 10},
                                                            {10, 11, 17, 15},
                                                            {15, 16,  2,  0} };
                Index * src = &_patchPoints[adjPatchIndex*20];
                for (int j=0; j<4; ++j) {
                    // invert direction
                    // note that src indices have already been offset.
                    dest[gregoryEdgeVerts[i][3-j]] = src[gregoryEdgeVerts[aedge][j]];
                }
            }
        }
        _levelAndFaceIndices.push_back(LevelAndFaceIndex::create(levelIndex, faceIndex));
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

    // add basis
    addPatchBasis(*level, faceIndex, cornerSpans, newVerticesMask, levelVertOffset, fvarChannel);

    ++_numGregoryBasisPatches;

    // return cvs;
    return ConstIndexArray(dest, 20);
}

} // end namespace Far

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv
