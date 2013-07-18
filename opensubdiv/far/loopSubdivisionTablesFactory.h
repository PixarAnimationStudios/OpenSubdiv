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

#ifndef FAR_LOOP_SUBDIVISION_TABLES_FACTORY_H
#define FAR_LOOP_SUBDIVISION_TABLES_FACTORY_H

#include "../version.h"

#include "../far/loopSubdivisionTables.h"
#include "../far/meshFactory.h"
#include "../far/kernelBatchFactory.h"
#include "../far/subdivisionTablesFactory.h"

#include <cassert>
#include <vector>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

template <class T, class U> class FarMeshFactory;

/// \brief A specialized factory for FarLoopSubdivisionTables
///
/// Separating the factory allows us to isolate Far data structures from Hbr dependencies.
///
template <class T, class U> class FarLoopSubdivisionTablesFactory {
protected:
    template <class X, class Y> friend class FarMeshFactory;

    /// \brief Creates a FarLoopSubdivisiontables instance.
    ///
    /// @param meshFactory  a valid FarMeshFactory instance
    ///
    /// @param farMesh
    ///
    /// @param batches      a vector of Kernel refinement batches : the factory 
    ///                     will reserve and append refinement tasks
    ///
    static FarLoopSubdivisionTables<U> * Create( FarMeshFactory<T,U> * meshFactory, FarMesh<U> * farMesh, FarKernelBatchVector * batches );
};

// This factory walks the Hbr vertices and accumulates the weights and adjacency
// (valance) information specific to the loop subdivision scheme. The results
// are stored in a FarLoopSubdivisionTable<U>.
template <class T, class U> FarLoopSubdivisionTables<U> * 
FarLoopSubdivisionTablesFactory<T,U>::Create( FarMeshFactory<T,U> * meshFactory, FarMesh<U> * farMesh, FarKernelBatchVector * batches ) {

    assert( meshFactory and farMesh );

    int maxlevel = meshFactory->GetMaxLevel();
    
    std::vector<int> & remap = meshFactory->getRemappingTable();
    
    FarSubdivisionTablesFactory<T,U> tablesFactory( meshFactory->GetHbrMesh(),  maxlevel, remap );

    FarLoopSubdivisionTables<U> * result = new FarLoopSubdivisionTables<U>(farMesh, maxlevel);

    // Allocate memory for the indexing tables
    result->_E_IT.resize(tablesFactory.GetNumEdgeVerticesTotal(maxlevel)*4);
    result->_E_W.resize(tablesFactory.GetNumEdgeVerticesTotal(maxlevel)*2);

    result->_V_ITa.resize((tablesFactory.GetNumVertexVerticesTotal(maxlevel)
                           - tablesFactory.GetNumVertexVerticesTotal(0))*5); // subtract coarse cage vertices
    result->_V_IT.resize(tablesFactory.GetVertVertsValenceSum());
    result->_V_W.resize(tablesFactory.GetNumVertexVerticesTotal(maxlevel)
                        - tablesFactory.GetNumVertexVerticesTotal(0));

    // Prepare batch table
    batches->reserve(maxlevel*5);

    int vertexOffset = 0;
    int V_IT_offset = 0;
    int edgeTableOffset = 0;
    int vertTableOffset = 0;

    int * E_IT = result->_E_IT.empty() ? 0 : &result->_E_IT[0];
    float * E_W = result->_E_W.empty() ? 0 : &result->_E_W[0];
    int * V_ITa = result->_V_ITa.empty() ? 0 : &result->_V_ITa[0];
    unsigned int * V_IT = result->_V_IT.empty() ? 0 : &result->_V_IT[0];
    float * V_W = result->_V_W.empty() ? 0 : &result->_V_W[0];

    for (int level=1; level<=maxlevel; ++level) {

        // pointer to the first vertex corresponding to this level
        vertexOffset = tablesFactory._vertVertIdx[level-1] + 
            (int)tablesFactory._vertVertsList[level-1].size();
        result->_vertsOffsets[level] = vertexOffset;

        // Edge vertices
        int nEdgeVertices = (int)tablesFactory._edgeVertsList[level].size();
        if (nEdgeVertices > 0) 
            batches->push_back(FarKernelBatch( FarKernelBatch::LOOP_EDGE_VERTEX,
                                               level,
                                               0,
                                               0,
                                               nEdgeVertices,
                                               edgeTableOffset,
                                               vertexOffset) );
        vertexOffset += nEdgeVertices;
        edgeTableOffset += nEdgeVertices;

        for (int i=0; i < nEdgeVertices; ++i) {

            HbrVertex<T> * v = tablesFactory._edgeVertsList[level][i];
            assert(v);
            HbrHalfedge<T> * e = v->GetParentEdge();
            assert(e);

            float esharp = e->GetSharpness(),
                  endPtWeight = 0.5f,
                  oppPtWeight = 0.5f;

            E_IT[4*i+0]= remap[e->GetOrgVertex()->GetID()];
            E_IT[4*i+1]= remap[e->GetDestVertex()->GetID()];

            if (!e->IsBoundary() && esharp <= 1.0f) {
                endPtWeight = 0.375f + esharp * (0.5f - 0.375f);
                oppPtWeight = 0.125f * (1 - esharp);

                HbrHalfedge<T>* ee = e->GetNext();
                E_IT[4*i+2]= remap[ee->GetDestVertex()->GetID()];
                ee = e->GetOpposite()->GetNext();
                E_IT[4*i+3]= remap[ee->GetDestVertex()->GetID()];
            } else {
                E_IT[4*i+2]= -1;
                E_IT[4*i+3]= -1;
            }
            E_W[2*i+0] = endPtWeight;
            E_W[2*i+1] = oppPtWeight;
        }
        E_IT += 4 * nEdgeVertices;
        E_W += 2 * nEdgeVertices;

        // Vertex vertices

        FarVertexKernelBatchFactory batchFactory((int)tablesFactory._vertVertsList[level].size(), 0);

        int nVertVertices = (int)tablesFactory._vertVertsList[level].size();
        for (int i=0; i < nVertVertices; ++i) {

            HbrVertex<T> * v = tablesFactory._vertVertsList[level][i],
                         * pv = v->GetParentVertex();
            assert(v and pv);

            // Look at HbrCatmarkSubdivision<T>::Subdivide for more details about
            // the multi-pass interpolation
            unsigned char masks[2];
            int npasses;
            float weights[2];
            masks[0] = pv->GetMask(false);
            masks[1] = pv->GetMask(true);

            // If the masks are identical, only a single pass is necessary. If the
            // vertex is transitioning to another rule, two passes are necessary,
            // except when transitioning from k_Dart to k_Smooth : the same
            // compute kernel is applied twice. Combining this special case allows
            // to batch the compute kernels into fewer calls.
            if (masks[0] != masks[1] and (
                not (masks[0]==HbrVertex<T>::k_Smooth and
                     masks[1]==HbrVertex<T>::k_Dart))) {
                weights[1] = pv->GetFractionalMask();
                weights[0] = 1.0f - weights[1];
                npasses = 2;
            } else {
                weights[0] = 1.0f;
                weights[1] = 0.0f;
                npasses = 1;
            }

            int rank = FarSubdivisionTablesFactory<T,U>::GetMaskRanking(masks[0], masks[1]);

            V_ITa[5*i+0] = V_IT_offset;
            V_ITa[5*i+1] = 0;
            V_ITa[5*i+2] = remap[ pv->GetID() ];
            V_ITa[5*i+3] = -1;
            V_ITa[5*i+4] = -1;

            for (int p=0; p<npasses; ++p)
                switch (masks[p]) {
                    case HbrVertex<T>::k_Smooth :
                    case HbrVertex<T>::k_Dart : {
                        HbrHalfedge<T> *e = pv->GetIncidentEdge(),
                                       *start = e;
                        while (e) {
                            V_ITa[5*i+1]++;

                            V_IT[V_IT_offset++] = remap[ e->GetDestVertex()->GetID() ];

                            e = e->GetPrev()->GetOpposite();

                            if (e==start) break;
                        }
                        break;
                    }
                    case HbrVertex<T>::k_Crease : {

                        class GatherCreaseEdgesOperator : public HbrHalfedgeOperator<T> {
                        public:
                            HbrVertex<T> * vertex; int eidx[2]; int count; bool next;

                            GatherCreaseEdgesOperator(HbrVertex<T> * v, bool n) : vertex(v), count(0), next(n) { eidx[0]=-1; eidx[1]=-1; }

                            virtual void operator() (HbrHalfedge<T> &e) {
                                if (e.IsSharp(next) and count < 2) {
                                    HbrVertex<T> * a = e.GetDestVertex();
                                    if (a==vertex)
                                        a = e.GetOrgVertex();
                                    eidx[count++]=a->GetID();
                                }
                            }
                        };

                        GatherCreaseEdgesOperator op( pv, p==1 );
                        pv->ApplyOperatorSurroundingEdges( op );

                        assert(V_ITa[5*i+3]==-1 and V_ITa[5*i+4]==-1);
                        assert(op.eidx[0]!=-1 and op.eidx[1]!=-1);
                        V_ITa[5*i+3] = remap[op.eidx[0]];
                        V_ITa[5*i+4] = remap[op.eidx[1]];
                        break;
                    }
                    case HbrVertex<T>::k_Corner :
                        // in the case of a k_Crease / k_Corner pass combination, we
                        // need to set the valence to -1 to tell the "B" Kernel to
                        // switch to k_Corner rule (as edge indices won't be -1)
                        if (V_ITa[5*i+1]==0)
                            V_ITa[5*i+1] = -1;

                    default : break;
                }

            if (rank>7)
                // the k_Corner and k_Crease single-pass cases apply a weight of 1.0
                // but this value is inverted in the kernel
                V_W[i] = 0.0;
            else
                V_W[i] = weights[0];

            batchFactory.AddVertex( i, rank );
        }
        V_ITa += nVertVertices*5;
        V_W += nVertVertices;

        batchFactory.AppendLoopBatches(level, vertTableOffset, vertexOffset, batches);
        vertexOffset += nVertVertices;
        vertTableOffset += nVertVertices;
    }
    result->_vertsOffsets[maxlevel+1] = vertexOffset;
    return result;
}

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* FAR_LOOP_SUBDIVISION_TABLES_FACTORY_H */
