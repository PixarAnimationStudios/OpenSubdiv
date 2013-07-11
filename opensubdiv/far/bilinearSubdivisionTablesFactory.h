//
//     Copyright (C) Pixar. All rights reserved.
//
//     This license governs use of the accompanying software. If you
//     use the software, you accept this license. If you do not accept
//     the license, do not use the software.
//
//     1. Definitions
//     The terms "reproduce," "reproduction," "derivative works," and
//     "distribution" have the same meaning here as under U.S.
//     copyright law.  A "contribution" is the original software, or
//     any additions or changes to the software.
//     A "contributor" is any person or entity that distributes its
//     contribution under this license.
//     "Licensed patents" are a contributor's patent claims that read
//     directly on its contribution.
//
//     2. Grant of Rights
//     (A) Copyright Grant- Subject to the terms of this license,
//     including the license conditions and limitations in section 3,
//     each contributor grants you a non-exclusive, worldwide,
//     royalty-free copyright license to reproduce its contribution,
//     prepare derivative works of its contribution, and distribute
//     its contribution or any derivative works that you create.
//     (B) Patent Grant- Subject to the terms of this license,
//     including the license conditions and limitations in section 3,
//     each contributor grants you a non-exclusive, worldwide,
//     royalty-free license under its licensed patents to make, have
//     made, use, sell, offer for sale, import, and/or otherwise
//     dispose of its contribution in the software or derivative works
//     of the contribution in the software.
//
//     3. Conditions and Limitations
//     (A) No Trademark License- This license does not grant you
//     rights to use any contributor's name, logo, or trademarks.
//     (B) If you bring a patent claim against any contributor over
//     patents that you claim are infringed by the software, your
//     patent license from such contributor to the software ends
//     automatically.
//     (C) If you distribute any portion of the software, you must
//     retain all copyright, patent, trademark, and attribution
//     notices that are present in the software.
//     (D) If you distribute any portion of the software in source
//     code form, you may do so only under this license by including a
//     complete copy of this license with your distribution. If you
//     distribute any portion of the software in compiled or object
//     code form, you may only do so under a license that complies
//     with this license.
//     (E) The software is licensed "as-is." You bear the risk of
//     using it. The contributors give no express warranties,
//     guarantees or conditions. You may have additional consumer
//     rights under your local laws which this license cannot change.
//     To the extent permitted under your local laws, the contributors
//     exclude the implied warranties of merchantability, fitness for
//     a particular purpose and non-infringement.
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
