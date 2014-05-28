//
//   Copyright 2014 Pixar
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

#ifndef OSDUTIL_PATCH_PARTITIONER_H
#define OSDUTIL_PATCH_PARTITIONER_H

#include "../version.h"
#include "../far/patchTables.h"

#include <algorithm>
#include <vector>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

// OsdUtilPatchPartitioner is a utility class which performs patch array
// partitioning for the given ID assignments.
//
// input patchtable:
// +-----------+-------------------+-----------------+-------------------+
// |Type       |Regular            |Boundary         | ...               |
// +-----------+-------------------+-----------------+-------------------+
// input patcharray
//             |a:b:c:d:e:f:.......|p:q:r:s:t:.......| ...
//              \ single drawcall /
//
//
// Flattening idsOnPtexFaces into all patches gives
// ID assignments on each patch
// +-----------+-------------------+-----------------+-------------------+
// |PartitionID|0:2:1:1:2:0:.......|4:1:1:1:2:.......| ...               |
// +-----------+-------------------+-----------------+-------------------+
//
//
// Then OsdUtilPatchPartitioner reorders indices table, patch param table
// and quadoffset table by partition ID, and also generates an array of
// patchArrayVector that corresponds to each partition ID.
//
//
// output patchtable
// +-----------+-------------------+-----------------+-------------------+
// |Type       |Regular            |Boundary         | ...               |
// +-----------+-------------------+-----------------+-------------------+
// |Reordered  |0:0:1:1:2:2:.......|1:1:1:2:4:.......| ...               |
// +-----------+-------------------+-----------------+-------------------+
//
// output patcharray subsets
//    for ID=0 |a:f|
//    for ID=1     |c:d|           |q:r:s|
//    for ID=2         |b:e|             |t|
//    ...             /    |
//            single drawcall
//

class OsdUtilPatchPartitioner {
public:
    /// Constructor.
    OsdUtilPatchPartitioner(FarPatchTables const *srcPatchTables,
                            std::vector<int> const &idsOnPtexFaces);

    /// Returns the number of partitions.
    int GetNumPartitions() const {
        return (int)_partitionedPatchArrays.size();
    }

    /// Returns the reordered patch tables.
    FarPatchTables const &GetPatchTables() const {
        return _patchTables;
    }

    /// Returns the subset of patcharrays which contains patches associated with the partition ID.
    FarPatchTables::PatchArrayVector const & GetPatchArrays(int partitionID) const {
        return _partitionedPatchArrays[partitionID];
    }

private:
    FarPatchTables _patchTables;
    std::vector<FarPatchTables::PatchArrayVector> _partitionedPatchArrays;
};


inline
OsdUtilPatchPartitioner::OsdUtilPatchPartitioner(FarPatchTables const *srcPatchTables,
                                                 std::vector<int> const &idsOnPtexFaces) :
        _patchTables(FarPatchTables::PatchArrayVector(),
                     FarPatchTables::PTable(), NULL, NULL, NULL, NULL, 0, 0)
{
    int numPartitions = 0;
    for (int i = 0; i < (int)idsOnPtexFaces.size(); ++i) {
        numPartitions = std::max(numPartitions, idsOnPtexFaces[i]);
    }
    ++numPartitions;
    _partitionedPatchArrays.resize(numPartitions);

    // src tables
    FarPatchTables::PatchParamTable const &srcPatchParamTable =
        srcPatchTables->GetPatchParamTable();
    FarPatchTables::PTable const &srcPTable =
        srcPatchTables->GetPatchTable();
    FarPatchTables::QuadOffsetTable const &srcQuadOffsetTable =
        srcPatchTables->GetQuadOffsetTable();
    FarPatchTables::FVarData const &srcFVarData =
        srcPatchTables->GetFVarData();

    // dst tables
    FarPatchTables::PatchParamTable newPatchParamTable;
    FarPatchTables::PTable newPTable;
    FarPatchTables::QuadOffsetTable newQuadOffsetTable;
    std::vector<float> newFVarDataTable;
    bool hasFVarData = !srcFVarData.GetAllData().empty() &&
        srcFVarData.GetData(1) == NULL; // multi-level face-varying data not supported
    int fvarWidth = hasFVarData ? srcFVarData.GetFVarWidth() : 0;

    // iterate over all patch
    for (FarPatchTables::PatchArrayVector::const_iterator paIt =
        srcPatchTables->GetPatchArrayVector().begin();
            paIt != srcPatchTables->GetPatchArrayVector().end(); ++paIt) {

        FarPatchTables::Descriptor desc = paIt->GetDescriptor();

        int vertexOffset = (int)newPTable.size();
        int patchOffset = (int)newPatchParamTable.size();
        int quadOffsetOffset = (int)newQuadOffsetTable.size();
        int fvarOffset = (int)newFVarDataTable.size();

        // shuffle this range in partition order
        std::vector<std::pair<int, int> > sortProxy(paIt->GetNumPatches());
        std::vector<int> numPatches(numPartitions);
        for (int i = 0; i < (int)paIt->GetNumPatches(); ++i) {
            int patchIndex = paIt->GetPatchIndex() + i;
            int ptexIndex = srcPatchParamTable[patchIndex].faceIndex;
            int partitionID = idsOnPtexFaces[ptexIndex];
            sortProxy[i] = std::make_pair(partitionID, patchIndex);
            ++numPatches[partitionID];
        }
        // sort by partitionID
        std::sort(sortProxy.begin(), sortProxy.end());

        // for each single patch
        for (int i = 0; i < (int)sortProxy.size(); ++i) {
            // reorder corresponding index table entries (16, 12, 9, 4, 3 integers)
            int patchIndex = sortProxy[i].second - paIt->GetPatchIndex();

            FarPatchTables::PTable::const_iterator begin =
                srcPTable.begin() + paIt->GetVertIndex() +
                (int)(patchIndex * desc.GetNumControlVertices());

            FarPatchTables::PTable::const_iterator end =
                begin + (desc.GetNumControlVertices());

            std::copy(begin, end, std::back_inserter(newPTable));

            // reorder corresponding patchparam table entry
            newPatchParamTable.push_back(
                srcPatchParamTable[patchIndex + paIt->GetPatchIndex()]);

            // reorder corresponding quadoffset table entries (4 int)
            if (desc.GetType() == FarPatchTables::GREGORY or
                desc.GetType() == FarPatchTables::GREGORY_BOUNDARY) {
                for (int j = 0; j < 4; ++j) {
                    newQuadOffsetTable.push_back(
                        srcQuadOffsetTable[patchIndex*4+j + quadOffsetOffset]);
                }
            }

            // reorder corresponding face-varying table entry
            if (hasFVarData) {
                int fvarVerts = desc.GetType() == FarPatchTables::TRIANGLES ? 3 : 4;
                for (int j = 0; j < fvarVerts * fvarWidth; ++j) {
                    newFVarDataTable.push_back(
                        srcFVarData.GetAllData()[patchIndex*fvarVerts*fvarWidth+j + fvarOffset]);
                }
            }
        }

        // split patch array
        for (int i = 0; i < numPartitions; ++i) {
            _partitionedPatchArrays[i].push_back(
                FarPatchTables::PatchArray(desc,
                                           vertexOffset,
                                           patchOffset,
                                           numPatches[i],
                                           quadOffsetOffset));

            vertexOffset += numPatches[i] * (desc.GetNumControlVertices());
            patchOffset += numPatches[i];
            quadOffsetOffset += numPatches[i] * 4;
        }
    }

    // create reordered patch tables.
    _patchTables = FarPatchTables(srcPatchTables->GetPatchArrayVector(),
                                  newPTable,
                                  &srcPatchTables->GetVertexValenceTable(),
                                  &newQuadOffsetTable,
                                  &newPatchParamTable,
                                  hasFVarData ? &newFVarDataTable : NULL,
                                  fvarWidth,
                                  srcPatchTables->GetMaxValence());
}

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OSDUTIL_PATCH_PARTITIONER_H
