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

#ifndef FAR_BILINEAR_SUBDIVISION_TABLES_FACTORY_H
#define FAR_BILINEAR_SUBDIVISION_TABLES_FACTORY_H

#include "../version.h"

#include "../far/bilinearSubdivisionTables.h"
#include "../far/meshFactory.h"
#include "../far/subdivisionTablesFactory.h"

#include <cassert>
#include <vector>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

template <class T, class U> class FarMeshFactory;

/// \brief A specialized factory for FarBilinearSubdivisionTables
///
/// Separating the factory allows us to isolate Far data structures from Hbr dependencies.
///
template <class T, class U> class FarBilinearSubdivisionTablesFactory {
protected:
    template <class X, class Y> friend class FarMeshFactory;

    /// \brief Creates a FarBilinearSubdivisiontables instance.
    ///
    /// @param meshFactory  a valid FarMeshFactory instance
    ///
    /// @param farMesh
    ///
    /// @param batches      a vector of Kernel refinement batches : the factory 
    ///                     will reserve and append refinement tasks
    ///
    static FarBilinearSubdivisionTables<U> * Create( FarMeshFactory<T,U> * meshFactory, FarMesh<U> * farMesh, FarKernelBatchVector *batches );
};

// This factory walks the Hbr vertices and accumulates the weights and adjacency
// (valance) information specific to the bilinear subdivision scheme. The results
// are stored in a FarBilinearSubdivisionTable<U>
template <class T, class U> FarBilinearSubdivisionTables<U> * 
FarBilinearSubdivisionTablesFactory<T,U>::Create( FarMeshFactory<T,U> * meshFactory, FarMesh<U> * farMesh, FarKernelBatchVector * batches ) {

    assert( meshFactory and farMesh );
     
    int maxlevel = meshFactory->GetMaxLevel();
    
    std::vector<int> & remap = meshFactory->getRemappingTable();
    
    FarSubdivisionTablesFactory<T,U> tablesFactory( meshFactory->GetHbrMesh(),  maxlevel, remap );

    FarBilinearSubdivisionTables<U> * result = new FarBilinearSubdivisionTables<U>(farMesh, maxlevel);

    // Allocate memory for the indexing tables
    result->_F_ITa.resize(tablesFactory.GetNumFaceVerticesTotal(maxlevel)*2);
    result->_F_IT.resize(tablesFactory.GetFaceVertsValenceSum());

    result->_E_IT.resize(tablesFactory.GetNumEdgeVerticesTotal(maxlevel)*2);

    result->_V_ITa.resize((tablesFactory.GetNumVertexVerticesTotal(maxlevel)
                           - tablesFactory.GetNumVertexVerticesTotal(0))); // subtract corase cage vertices

    // Prepare batch table
    batches->reserve(maxlevel*5);

    int vertexOffset = 0;
    int F_IT_offset = 0;
    int faceTableOffset = 0;
    int edgeTableOffset = 0;
    int vertTableOffset = 0;

    int * F_ITa = result->_F_ITa.empty() ? 0 : &result->_F_ITa[0];
    unsigned int * F_IT = result->_F_IT.empty() ? 0 : &result->_F_IT[0];
    int * E_IT = result->_E_IT.empty() ? 0 : &result->_E_IT[0];
    int * V_ITa = result->_V_ITa.empty() ? 0 : &result->_V_ITa[0];

    for (int level=1; level<=maxlevel; ++level) {

        // pointer to the first vertex corresponding to this level
        vertexOffset = tablesFactory._vertVertIdx[level-1] +
            (int)tablesFactory._vertVertsList[level-1].size();
        result->_vertsOffsets[level] = vertexOffset;

        // Face vertices
        // "For each vertex, gather all the vertices from the parent face."
        int nFaceVertices = (int)tablesFactory._faceVertsList[level].size();
        if (nFaceVertices > 0)
            batches->push_back(FarKernelBatch( FarKernelBatch::BILINEAR_FACE_VERTEX,
                                               level,
                                               /*tableIndex=*/0,
                                               /*start=*/0,
                                               /*end=*/nFaceVertices,
                                               faceTableOffset,
                                               vertexOffset) );
        vertexOffset += nFaceVertices;
        faceTableOffset += nFaceVertices;

        for (int i=0; i < nFaceVertices; ++i) {

            HbrVertex<T> * v = tablesFactory._faceVertsList[level][i];
            assert(v);

            HbrFace<T> * f=v->GetParentFace();
            assert(f);

            int valence = f->GetNumVertices();

            F_ITa[2*i+0] = F_IT_offset;
            F_ITa[2*i+1] = valence;

            for (int j=0; j<valence; ++j)
                F_IT[F_IT_offset++] = remap[f->GetVertex(j)->GetID()];
        }
        F_ITa += nFaceVertices*2;

        // Edge vertices

        // "Average the end-points of the parent edge"
        int nEdgeVertices = (int)tablesFactory._edgeVertsList[level].size();
        if (nEdgeVertices > 0)
            batches->push_back(FarKernelBatch( FarKernelBatch::BILINEAR_EDGE_VERTEX,
                                               level,
                                               /*tableIndex=*/0,
                                               /*start=*/0,
                                               /*end=*/nEdgeVertices,
                                               edgeTableOffset,
                                               vertexOffset) );
        vertexOffset += nEdgeVertices;
        edgeTableOffset += nEdgeVertices;
        for (int i=0; i < nEdgeVertices; ++i) {

            HbrVertex<T> * v = tablesFactory._edgeVertsList[level][i];
            assert(v);
            HbrHalfedge<T> * e = v->GetParentEdge();
            assert(e);

            // get the indices 2 vertices from the parent edge
            E_IT[2*i+0] = remap[e->GetOrgVertex()->GetID()];
            E_IT[2*i+1] = remap[e->GetDestVertex()->GetID()];

        }
        E_IT += nEdgeVertices*2;

        // Vertex vertices

        // "Pass down the parent vertex"
        int nVertVertices = (int)tablesFactory._vertVertsList[level].size();
        batches->push_back(FarKernelBatch( FarKernelBatch::BILINEAR_VERT_VERTEX,
                                           level,
                                           /*tableIndex=*/0,
                                           /*start=*/0,
                                           /*end=*/nVertVertices,
                                           vertTableOffset,
                                           vertexOffset));
        vertexOffset += nVertVertices;
        vertTableOffset += nVertVertices;
        
        for (int i=0; i < nVertVertices; ++i) {

            HbrVertex<T> * v = tablesFactory._vertVertsList[level][i],
                         * pv = v->GetParentVertex();
            assert(v and pv);

            V_ITa[i] = remap[pv->GetID()];

        }
        V_ITa += nVertVertices;
    }
    result->_vertsOffsets[maxlevel+1] = vertexOffset;
    return result;
}

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* FAR_BILINEAR_SUBDIVISION_TABLES_FACTORY_H */
