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

#ifndef FAR_LOOP_SUBDIVISION_TABLES_FACTORY_H
#define FAR_LOOP_SUBDIVISION_TABLES_FACTORY_H

#include "../version.h"

#include "../far/loopSubdivisionTables.h"
#include "../far/meshFactory.h"
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

    /// Creates a FarLoopSubdivisiontables instance.
    static FarLoopSubdivisionTables<U> * Create( FarMeshFactory<T,U> * meshFactory, FarMesh<U> * farMesh );
};

// This factory walks the Hbr vertices and accumulates the weights and adjacency
// (valance) information specific to the loop subdivision scheme. The results
// are stored in a FarLoopSubdivisionTable<U>.
template <class T, class U> FarLoopSubdivisionTables<U> * 
FarLoopSubdivisionTablesFactory<T,U>::Create( FarMeshFactory<T,U> * meshFactory, FarMesh<U> * farMesh ) {

    assert( meshFactory and farMesh );

    int maxlevel = meshFactory->GetMaxLevel();
    
    std::vector<int> & remap = meshFactory->getRemappingTable();
    
    FarSubdivisionTablesFactory<T,U> tablesFactory( meshFactory->GetHbrMesh(),  maxlevel, remap );

    FarLoopSubdivisionTables<U> * result = new FarLoopSubdivisionTables<U>(farMesh, maxlevel);

    // Allocate memory for the indexing tables
    result->_E_IT.Resize(tablesFactory.GetNumEdgeVerticesTotal(maxlevel)*4);
    result->_E_W.Resize(tablesFactory.GetNumEdgeVerticesTotal(maxlevel)*2);

    result->_V_ITa.Resize(tablesFactory.GetNumVertexVerticesTotal(maxlevel)*5);
    result->_V_IT.Resize(tablesFactory.GetVertVertsValenceSum());
    result->_V_W.Resize(tablesFactory.GetNumVertexVerticesTotal(maxlevel));

    for (int level=1; level<=maxlevel; ++level) {

        // pointer to the first vertex corresponding to this level
        result->_vertsOffsets[level] = tablesFactory._vertVertIdx[level-1] + 
                                       (int)tablesFactory._vertVertsList[level-1].size();

        typename FarSubdivisionTables<U>::VertexKernelBatch * batch = & (result->_batches[level-1]);

        // Edge vertices
        int * E_IT = result->_E_IT[level-1];
        float * E_W = result->_E_W[level-1];
        batch->kernelE = (int)tablesFactory._edgeVertsList[level].size();
        for (int i=0; i < batch->kernelE; ++i) {

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
        result->_E_IT.SetMarker(level, &E_IT[4*batch->kernelE]);
        result->_E_W.SetMarker(level, &E_W[2*batch->kernelE]);

        // Vertex vertices

        batch->InitVertexKernels( (int)tablesFactory._vertVertsList[level].size(), 0 );

        int offset = 0;
        int * V_ITa = result->_V_ITa[level-1];
        unsigned int * V_IT = result->_V_IT[level-1];
        float * V_W = result->_V_W[level-1];
        int nverts = (int)tablesFactory._vertVertsList[level].size();
        for (int i=0; i < nverts; ++i) {

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

            V_ITa[5*i+0] = offset;
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

                            V_IT[offset++] = remap[ e->GetDestVertex()->GetID() ];

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

            batch->AddVertex( i, rank );
        }
        result->_V_ITa.SetMarker(level, &V_ITa[5*nverts]);
        result->_V_IT.SetMarker(level, &V_IT[offset]);
        result->_V_W.SetMarker(level, &V_W[nverts]);

        if (nverts>0) {
            batch->kernelB.second++;
            batch->kernelA1.second++;
            batch->kernelA2.second++;
        }
    }
    return result;
}

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* FAR_LOOP_SUBDIVISION_TABLES_FACTORY_H */
