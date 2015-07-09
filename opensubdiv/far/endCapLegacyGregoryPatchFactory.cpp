//
//   Copyright 2015 Pixar
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

#include "../far/error.h"
#include "../far/endCapLegacyGregoryPatchFactory.h"
#include "../far/patchTable.h"
#include "../far/topologyRefiner.h"
#include "../vtr/level.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Far {

EndCapLegacyGregoryPatchFactory::EndCapLegacyGregoryPatchFactory(
    TopologyRefiner const &refiner) :
    _refiner(refiner) {
}

ConstIndexArray
EndCapLegacyGregoryPatchFactory::GetPatchPoints(
    Vtr::internal::Level const * level, Index faceIndex,
    PatchTableFactory::PatchFaceTag const * levelPatchTags,
    int levelVertOffset) {

    PatchTableFactory::PatchFaceTag patchTag = levelPatchTags[faceIndex];

    // Gregory Regular Patch (4 CVs + quad-offsets / valence tables)
    Vtr::ConstIndexArray faceVerts = level->getFaceVertices(faceIndex);

    if (patchTag._boundaryCount) {
        for (int j = 0; j < 4; ++j) {
            // apply level offset
            _gregoryBoundaryTopology.push_back(faceVerts[j] + levelVertOffset);
        }
        _gregoryBoundaryFaceIndices.push_back(faceIndex);
        return ConstIndexArray(&_gregoryBoundaryTopology[_gregoryBoundaryTopology.size()-4], 4);
    } else {
        for (int j = 0; j < 4; ++j) {
            // apply level offset
            _gregoryTopology.push_back(faceVerts[j] + levelVertOffset);
        }
        _gregoryFaceIndices.push_back(faceIndex);
        return ConstIndexArray(&_gregoryTopology[_gregoryTopology.size()-4], 4);
    }
}

//
//  Populate the quad-offsets table used by Gregory patches
//
static void getQuadOffsets(
    Vtr::internal::Level const& level, Index faceIndex, unsigned int offsets[]) {

    Vtr::ConstIndexArray fVerts = level.getFaceVertices(faceIndex);

    for (int i = 0; i < 4; ++i) {

        Vtr::Index vIndex = fVerts[i];
        Vtr::ConstIndexArray vFaces = level.getVertexFaces(vIndex),
                      vEdges = level.getVertexEdges(vIndex);

        int thisFaceInVFaces = -1;
        for (int j = 0; j < vFaces.size(); ++j) {
            if (faceIndex == vFaces[j]) {
                thisFaceInVFaces = j;
                break;
            }
        }
        assert(thisFaceInVFaces != -1);

        Index vOffsets[2];
        vOffsets[0] = thisFaceInVFaces;
        vOffsets[1] = (thisFaceInVFaces + 1)%vEdges.size();
        // we have to use the number of incident edges to modulo the local index
        // because there could be 2 consecutive edges in the face belonging to
        // the Gregory patch.
        offsets[i] = vOffsets[0] | (vOffsets[1] << 8);
    }
}

void
EndCapLegacyGregoryPatchFactory::Finalize(
    int maxValence, 
    PatchTable::QuadOffsetsTable *quadOffsetsTable,
    PatchTable::VertexValenceTable *vertexValenceTable)
{
    // populate quad offsets

    size_t numGregoryPatches = _gregoryFaceIndices.size();
    size_t numGregoryBoundaryPatches = _gregoryBoundaryFaceIndices.size();
    size_t numTotalGregoryPatches = 
        numGregoryPatches + numGregoryBoundaryPatches;

    Vtr::internal::Level const &maxLevel = _refiner.getLevel(_refiner.GetMaxLevel());

    quadOffsetsTable->resize(numTotalGregoryPatches*4);

    if (numTotalGregoryPatches > 0) {
        PatchTable::QuadOffsetsTable::value_type *p = 
            &((*quadOffsetsTable)[0]);
        for (size_t i = 0; i < numGregoryPatches; ++i) {
            getQuadOffsets(maxLevel, _gregoryFaceIndices[i], p);
            p += 4;
        }
        for (size_t i = 0; i < numGregoryBoundaryPatches; ++i) {
            getQuadOffsets(maxLevel, _gregoryBoundaryFaceIndices[i], p);
            p += 4;
        }
    }

    // populate vertex valences
    //
    //  Now deal with the "vertex valence" table for Gregory patches -- this 
    //  table contains the one-ring of vertices around each vertex.  Currently
    //  it is extremely wasteful for the following reasons:
    //      - it allocates 2*maxvalence+1 for ALL vertices
    //      - it initializes the one-ring for ALL vertices
    //  We use the full size expected (not sure what else relies on that) but 
    //  we avoiding initializing
    //  the vast majority of vertices that are not associated with gregory 
    //  patches -- by having previously marked those that are associated above
    //  and skipping all others.
    //
    const int SizePerVertex = 2*maxValence + 1;

    PatchTable::VertexValenceTable & vTable = (*vertexValenceTable);
    vTable.resize(_refiner.GetNumVerticesTotal() * SizePerVertex);

    int vOffset = 0;
    int levelLast = _refiner.GetMaxLevel();
    for (int i = 0; i <= levelLast; ++i) {

        Vtr::internal::Level const * level = &_refiner.getLevel(i);

        if (i == levelLast) {

            int vTableOffset = vOffset * SizePerVertex;

            for (int vIndex = 0; vIndex < level->getNumVertices(); ++vIndex) {
                int* vTableEntry = &vTable[vTableOffset];

                //
                //  If not marked as a vertex of a gregory patch, just set to 0 to ignore.  Otherwise
                //  gather the one-ring around the vertex and set its resulting size (note the negative
                //  size used to distinguish between boundary/interior):
                //
                //if (!gregoryVertexFlags[vIndex + vOffset]) {
                vTableEntry[0] = 0;
                //} else {

                int * ringDest = vTableEntry + 1,
                    ringSize = level->gatherQuadRegularRingAroundVertex(vIndex, ringDest);

                for (int j = 0; j < ringSize; ++j) {
                    ringDest[j] += vOffset;
                }
                if (ringSize & 1) {
                    // boundary vertex : duplicate boundary vertex index
                    // and store negative valence.
                    ringSize++;
                    vTableEntry[ringSize]=vTableEntry[ringSize-1];
                    vTableEntry[0] = -ringSize/2;
                } else {
                    vTableEntry[0] = ringSize/2;
                }
                //}
                vTableOffset += SizePerVertex;
            }
        }
        vOffset += level->getNumVertices();
    }
}


} // end namespace Far

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv

