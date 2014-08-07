//
//   Copyright 2014 DigitalFish, Inc.
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

#ifndef OSDUTIL_VERTEX_SPLIT_H
#define OSDUTIL_VERTEX_SPLIT_H

#include "../version.h"
#include "../far/mesh.h"
#include "../far/meshFactory.h"

#include <algorithm>
#include <map>
#include <vector>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

/// \brief Duplicates vertices at the finest subdivision level to produce a
/// \brief minimal vertex-varying data table for the face-varying data
///
/// Modifies the mesh to add duplicate vertices at the finest subdivision level
/// to produce a minimal vertex-varying data table for the face-varying data.
/// The new vertices are inserted after their original vertex, and the patch
/// control vertices are reindexed accordingly.
///
template <class T>
class OsdUtilVertexSplit {
public:
    // A table of vertex-varying data at the finest subdivision level
    typedef std::vector<float> VVarDataTable;

    /// \brief Constructor
    OsdUtilVertexSplit(FarMesh<T> * mesh);

    /// \brief Returns the table of vertex-varying data for the finest
    /// \brief subdivision level
    VVarDataTable const &GetVVarDataTable() const
    {
        return _vvarDataTable;
    }

private:
    VVarDataTable _vvarDataTable;       // the table of vertex-varying data
};

template <class T>
OsdUtilVertexSplit<T>::OsdUtilVertexSplit(FarMesh<T> * mesh)
{
    typedef std::multimap<int, int> VertexToFVarMultimap;

    const FarKernelBatchVector& kernelBatchVector = mesh->GetKernelBatches();
    const FarPatchTables* patchTables = mesh->GetPatchTables();
    const FarSubdivisionTables* subdivisionTables =
        mesh->GetSubdivisionTables();
    const FarPatchTables::PTable& patchTable = patchTables->GetPatchTable();
    const FarPatchTables::FVarData& fvarData = patchTables->GetFVarData();
    const std::vector<float>& fvarDataTable = fvarData.GetAllData();
    int fvarWidth = fvarData.GetFVarWidth();
    if (fvarWidth == 0)
        return;

    // Determine which vertices to split.
    typename FarMeshFactory<T>::SplitTable splitTable(patchTable.size());
    VertexToFVarMultimap vertexToFVarMultimap;
    for (int i = 0; i < (int)patchTable.size(); ++i) {
        int vertex = patchTable[i];
        std::pair<VertexToFVarMultimap::const_iterator,
            VertexToFVarMultimap::const_iterator> vertexRange =
            vertexToFVarMultimap.equal_range(vertex);

        int j;
        for (j = 0; vertexRange.first != vertexRange.second;
            ++vertexRange.first, ++j)
        {
            int fvar = vertexRange.first->second;
            const float* fvarData = &fvarDataTable[fvar * fvarWidth];
            if (std::equal(fvarData, fvarData + fvarWidth,
                &fvarDataTable[i * fvarWidth]))
            {
                splitTable[i] = j;
                goto split_vertex;
            }
        }

        splitTable[i] = j;
        vertexToFVarMultimap.insert(std::make_pair(vertex, i));

    split_vertex:;
    }

    // Duplicate vertices in the kernel batches from the last subdivision level.
    for (int i = (int)kernelBatchVector.size() - 1; i >= 0; --i) {
        const FarKernelBatch& kernelBatch = kernelBatchVector[i];
        if (kernelBatch.GetLevel() != subdivisionTables->GetMaxLevel() - 1)
            break;

        int vertexOffset = kernelBatch.GetVertexOffset();
        int firstVertex = vertexOffset + kernelBatch.GetStart();
        int lastVertex = vertexOffset + kernelBatch.GetEnd();

        // Select the vertices to duplicate from this kernel batch.
        typename FarMeshFactory<T>::VertexList duplicateList;
        for (int j = firstVertex; j < lastVertex; ++j) {
            std::pair<std::multimap<int, int>::const_iterator,
                std::multimap<int, int>::const_iterator> vertexRange =
                vertexToFVarMultimap.equal_range(j);
            for (++vertexRange.first; vertexRange.first != vertexRange.second;
                ++vertexRange.first)
            {
                duplicateList.push_back(j);
            }
        }

        // Duplicate vertices in this kernel batch.
        FarMeshFactory<T>::DuplicateVertices(mesh, duplicateList);

        // Interleave the duplicate vertices after their original vertex.
        int duplicateVertex = lastVertex;
        int nextVertex = firstVertex;
        typename FarMeshFactory<T>::VertexPermutation vertexPermutation;
        for (int j = firstVertex; j < lastVertex; ++j) {
            vertexPermutation[j] = nextVertex++;

            std::pair<std::multimap<int, int>::const_iterator,
                std::multimap<int, int>::const_iterator> vertexRange =
                vertexToFVarMultimap.equal_range(j);
            for (++vertexRange.first; vertexRange.first != vertexRange.second;
                ++vertexRange.first)
            {
                vertexPermutation[duplicateVertex++] = nextVertex++;
            }
        }

        FarMeshFactory<T>::PermuteVertices(mesh, vertexPermutation);
    }

    // Split the vertices in the mesh.
    FarMeshFactory<T>::SplitVertices(mesh, splitTable);

    // Map each vertex to its associated face-varying data.
    typedef std::map<int, int> VertexToFVarMap;
    VertexToFVarMap vertexToFVarMap;
    for (int i = 0; i < (int)patchTable.size(); ++i) {
        vertexToFVarMap.insert(std::make_pair(patchTable[i], i));
    }

    // Create the vertex-varying data table.
    int lastLevel = subdivisionTables->GetMaxLevel() - 1;
    int firstVertex = subdivisionTables->GetFirstVertexOffset(lastLevel);
    int numVertices = subdivisionTables->GetNumVertices(lastLevel);

    _vvarDataTable.resize(numVertices * fvarWidth);
    for (int i = 0; i < (int)patchTable.size(); ++i) {
        int vertex = patchTable[i];
        int fvar = vertexToFVarMap.find(vertex)->second;
        for (int j = 0; j < fvarWidth; ++j) {
            _vvarDataTable[(vertex - firstVertex) * fvarWidth + j] =
                fvarDataTable[fvar * fvarWidth + j];
        }
    }
}

} // namespace OPENSUBDIV_VERSION

using namespace OPENSUBDIV_VERSION;

} // namespace OpenSubdiv

#endif
