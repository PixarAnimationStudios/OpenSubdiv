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

#ifndef FAR_CATMARK_SUBDIVISION_TABLES_FACTORY_H
#define FAR_CATMARK_SUBDIVISION_TABLES_FACTORY_H

#include <cassert>
#include <vector>

#include "../version.h"

#include "../far/subdivisionTables.h"
#include "../far/meshFactory.h"
#include "../far/kernelBatchFactory.h"
#include "../far/subdivisionTablesFactory.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

template <class T, class U> class FarMeshFactory;

/// \brief A specialized factory for catmark FarSubdivisionTables
///
/// Separating the factory allows us to isolate Far data structures from Hbr dependencies.
///
template <class T, class U> class FarCatmarkSubdivisionTablesFactory {
protected:
    template <class X, class Y> friend class FarMeshFactory;

    /// \brief Creates a FarSubdivisiontables instance with Catmark scheme.
    ///
    /// @param meshFactory  a valid FarMeshFactory instance
    ///
    /// @param batches      a vector of Kernel refinement batches : the factory
    ///                     will reserve and append refinement tasks
    ///
    static FarSubdivisionTables * Create( FarMeshFactory<T,U> * meshFactory, FarKernelBatchVector *batches  );

    // Compares vertices based on their topological configuration
    // (see subdivisionTables::GetMaskRanking for more details)
    static bool CompareVertices( HbrVertex<T> const *x, HbrVertex<T> const *y );
};

// This factory walks the Hbr vertices and accumulates the weights and adjacency
// (valance) information specific to the catmark subdivision scheme. The results
// are stored in a FarSubdivisionTable.
template <class T, class U> FarSubdivisionTables *
FarCatmarkSubdivisionTablesFactory<T,U>::Create( FarMeshFactory<T,U> * meshFactory, FarKernelBatchVector *batches ) {

    assert( meshFactory );

    int maxlevel = meshFactory->GetMaxLevel();

    std::vector<int> & remap = meshFactory->getRemappingTable();

    FarSubdivisionTablesFactory<T,U> tablesFactory( meshFactory->GetHbrMesh(), maxlevel, remap, CompareVertices );

    FarSubdivisionTables * result = new FarSubdivisionTables(maxlevel, FarSubdivisionTables::CATMARK);

    // Calculate the size of the face-vertex indexing tables
    int minCoarseFaceValence = tablesFactory.GetMinCoarseFaceValence();
    int maxCoarseFaceValence = tablesFactory.GetMaxCoarseFaceValence();
    bool coarseMeshAllQuadFaces = minCoarseFaceValence == 4 and maxCoarseFaceValence == 4;
    bool coarseMeshAllTriQuadFaces = minCoarseFaceValence >= 3 and maxCoarseFaceValence <= 4;
    bool hasQuadFaceVertexKernel = meshFactory->IsKernelTypeSupported(FarKernelBatch::CATMARK_QUAD_FACE_VERTEX);
    bool hasTriQuadFaceVertexKernel = meshFactory->IsKernelTypeSupported(FarKernelBatch::CATMARK_TRI_QUAD_FACE_VERTEX);

    int F_ITa_size = 0;
    if (not hasQuadFaceVertexKernel and not hasTriQuadFaceVertexKernel)
        F_ITa_size = tablesFactory.GetNumFaceVerticesTotal(maxlevel) * 2;
    else if (not coarseMeshAllTriQuadFaces or not hasTriQuadFaceVertexKernel)
        F_ITa_size = tablesFactory.GetNumFaceVerticesTotal(1) * 2;

    int F_IT_size = tablesFactory.GetFaceVertsValenceSum();
    if (coarseMeshAllTriQuadFaces and hasTriQuadFaceVertexKernel)
        F_IT_size += tablesFactory.GetNumCoarseTriangleFaces(); // add padding for tri faces

    // Triangular interpolation mode :
    // see "smoothtriangle" tag introduced in prman 3.9 and HbrCatmarkSubdivision<T>
    typename HbrCatmarkSubdivision<T>::TriangleSubdivision triangleMethod =
        dynamic_cast<HbrCatmarkSubdivision<T> *>(meshFactory->GetHbrMesh()->GetSubdivision())->GetTriangleSubdivisionMethod();
    bool hasFractionalEdgeSharpness = tablesFactory.HasFractionalEdgeSharpness();
    bool useRestrictedEdgeVertexKernel = meshFactory->IsKernelTypeSupported(FarKernelBatch::CATMARK_RESTRICTED_EDGE_VERTEX);
    useRestrictedEdgeVertexKernel &= not hasFractionalEdgeSharpness and triangleMethod != HbrCatmarkSubdivision<T>::k_New;

    bool hasFractionalVertexSharpness = tablesFactory.HasFractionalVertexSharpness();
    bool hasStandardVertexVertexKernels = meshFactory->IsKernelTypeSupported(FarKernelBatch::CATMARK_VERT_VERTEX_A1) and
                                          meshFactory->IsKernelTypeSupported(FarKernelBatch::CATMARK_VERT_VERTEX_A2) and
                                          meshFactory->IsKernelTypeSupported(FarKernelBatch::CATMARK_VERT_VERTEX_B);
    bool useRestrictedVertexVertexKernels = meshFactory->IsKernelTypeSupported(FarKernelBatch::CATMARK_RESTRICTED_VERT_VERTEX_A) and
                                            meshFactory->IsKernelTypeSupported(FarKernelBatch::CATMARK_RESTRICTED_VERT_VERTEX_B1) and
                                            meshFactory->IsKernelTypeSupported(FarKernelBatch::CATMARK_RESTRICTED_VERT_VERTEX_B2);
    useRestrictedVertexVertexKernels &= not hasFractionalVertexSharpness and not hasFractionalEdgeSharpness;

    // Allocate memory for the indexing tables
    result->_F_ITa.resize(F_ITa_size);
    result->_F_IT.resize(F_IT_size);

    result->_E_IT.resize(tablesFactory.GetNumEdgeVerticesTotal(maxlevel)*4);
    if (not useRestrictedEdgeVertexKernel)
        result->_E_W.resize(tablesFactory.GetNumEdgeVerticesTotal(maxlevel)*2);

    result->_V_ITa.resize((tablesFactory.GetNumVertexVerticesTotal(maxlevel)
                           - tablesFactory.GetNumVertexVerticesTotal(0))*5); // subtract coarse cage vertices
    result->_V_IT.resize(tablesFactory.GetVertVertsValenceSum()*2);
    if (not useRestrictedVertexVertexKernels)
        result->_V_W.resize(tablesFactory.GetNumVertexVerticesTotal(maxlevel)
                            - tablesFactory.GetNumVertexVerticesTotal(0));

    // Prepare batch table
    batches->reserve(maxlevel*5);

    int vertexOffset = 0;
    int F_IT_offset = 0;
    int V_IT_offset = 0;
    int faceTableOffset = 0;
    int edgeTableOffset = 0;
    int vertTableOffset = 0;

    unsigned int * F_IT = result->_F_IT.empty() ? 0 : &result->_F_IT[0];
    int * F_ITa = result->_F_ITa.empty() ? 0 : &result->_F_ITa[0];
    int * E_IT = result->_E_IT.empty() ? 0 : &result->_E_IT[0];
    float * E_W = result->_E_W.empty() ? 0 : &result->_E_W[0];
    int * V_ITa = result->_V_ITa.empty() ? 0 : &result->_V_ITa[0];
    unsigned int * V_IT = result->_V_IT.empty() ? 0 : &result->_V_IT[0];
    float * V_W = result->_V_W.empty() ? 0 : &result->_V_W[0];

    for (int level=1; level<=maxlevel; ++level) {

        // pointer to the first vertex corresponding to this level
        vertexOffset = tablesFactory._faceVertIdx[level];
        result->_vertsOffsets[level] = vertexOffset;

        // Face vertices
        // "For each vertex, gather all the vertices from the parent face."
        int nFaceVertices = (int)tablesFactory._faceVertsList[level].size();

        // choose the kernel type that best fits the face topology
        int kernelType = FarKernelBatch::CATMARK_FACE_VERTEX;
        if (level == 1) {
            if (coarseMeshAllQuadFaces and hasQuadFaceVertexKernel)
                kernelType = FarKernelBatch::CATMARK_QUAD_FACE_VERTEX;
            else if (coarseMeshAllTriQuadFaces and hasTriQuadFaceVertexKernel)
                kernelType = FarKernelBatch::CATMARK_TRI_QUAD_FACE_VERTEX;
        } else {
            if (hasQuadFaceVertexKernel)
                kernelType = FarKernelBatch::CATMARK_QUAD_FACE_VERTEX;
            if (hasTriQuadFaceVertexKernel)
                kernelType = FarKernelBatch::CATMARK_TRI_QUAD_FACE_VERTEX;
        }

        // add a batch for face vertices
        if (nFaceVertices > 0) {  // in torus case, nfacevertices could be zero
            assert(meshFactory->IsKernelTypeSupported(kernelType));
            if (kernelType == FarKernelBatch::CATMARK_FACE_VERTEX) {
                batches->push_back(FarKernelBatch( kernelType,
                                                   level,
                                                   0,
                                                   0,
                                                   nFaceVertices,
                                                   faceTableOffset,
                                                   vertexOffset) );
            } else {
                // quad and tri-quad kernels store the offset of the first vertex in the table offset
                batches->push_back(FarKernelBatch( kernelType,
                                                   level,
                                                   0,
                                                   0,
                                                   nFaceVertices,
                                                   F_IT_offset,
                                                   vertexOffset) );
            }
        }

        vertexOffset += nFaceVertices;
        if (kernelType == FarKernelBatch::CATMARK_FACE_VERTEX)
            faceTableOffset += nFaceVertices;

        for (int i=0; i < nFaceVertices; ++i) {

            HbrVertex<T> * v = tablesFactory._faceVertsList[level][i];
            assert(v);

            HbrFace<T> * f=v->GetParentFace();
            assert(f);

            int valence = f->GetNumVertices();

            if (kernelType == FarKernelBatch::CATMARK_FACE_VERTEX) {
                *F_ITa++ = F_IT_offset;
                *F_ITa++ = valence;
            }

            for (int j=0; j<valence; ++j)
                F_IT[F_IT_offset++] = remap[f->GetVertex(j)->GetID()];

            if (kernelType == FarKernelBatch::CATMARK_TRI_QUAD_FACE_VERTEX and valence == 3)
                F_IT[F_IT_offset++] = remap[f->GetVertex(2)->GetID()]; // repeat last index
        }

        // Edge vertices

        // "For each vertex, gather the 2 vertices from the parent edege and the
        // 2 child vertices from the faces to the left and right of that edge.
        // Adjust if edge has a crease or is on a boundary."
        int nEdgeVertices = (int)tablesFactory._edgeVertsList[level].size();

        // add a batch for edge vertices
        kernelType = (useRestrictedEdgeVertexKernel ?
            FarKernelBatch::CATMARK_RESTRICTED_EDGE_VERTEX :
            FarKernelBatch::CATMARK_EDGE_VERTEX);

        if (nEdgeVertices > 0) {
            assert(meshFactory->IsKernelTypeSupported(kernelType));
            batches->push_back(FarKernelBatch( kernelType,
                                               level,
                                               0,
                                               0,
                                               nEdgeVertices,
                                               edgeTableOffset,
                                               vertexOffset) );
        }

        vertexOffset += nEdgeVertices;
        edgeTableOffset += nEdgeVertices;

        for (int i=0; i < nEdgeVertices; ++i) {

            HbrVertex<T> * v = tablesFactory._edgeVertsList[level][i];
            assert(v);
            HbrHalfedge<T> * e = v->GetParentEdge();
            assert(e);

            float esharp = e->GetSharpness();

            // get the indices 2 vertices from the parent edge
            E_IT[4*i+0] = remap[e->GetOrgVertex()->GetID()];
            E_IT[4*i+1] = remap[e->GetDestVertex()->GetID()];

            float faceWeight=0.5f, vertWeight=0.5f;

            if (kernelType == FarKernelBatch::CATMARK_RESTRICTED_EDGE_VERTEX) {
                // in the case of a sharp edge, repeat the endpoint vertices
                if (not e->IsBoundary() and esharp == HbrHalfedge<T>::k_Smooth) {
                    HbrFace<T>* rf = e->GetRightFace();
                    HbrFace<T>* lf = e->GetLeftFace();

                    E_IT[4*i+2] = remap[lf->Subdivide()->GetID()];
                    E_IT[4*i+3] = remap[rf->Subdivide()->GetID()];
                } else {
                    E_IT[4*i+2] = E_IT[4*i+0];
                    E_IT[4*i+3] = E_IT[4*i+1];
                }
            } else if (not e->IsBoundary() and esharp <= HbrHalfedge<T>::k_Sharp) {
                // in the case of a fractional sharpness, set the adjacent faces vertices

                float leftWeight, rightWeight;
                HbrFace<T>* rf = e->GetRightFace();
                HbrFace<T>* lf = e->GetLeftFace();

                leftWeight = ( triangleMethod == HbrCatmarkSubdivision<T>::k_New && lf->GetNumVertices() == 3) ? HBR_SMOOTH_TRI_EDGE_WEIGHT : 0.25f;
                rightWeight = ( triangleMethod == HbrCatmarkSubdivision<T>::k_New && rf->GetNumVertices() == 3) ? HBR_SMOOTH_TRI_EDGE_WEIGHT : 0.25f;

                faceWeight = 0.5f * (leftWeight + rightWeight);
                vertWeight = 0.5f * (1.0f - 2.0f * faceWeight);

                faceWeight *= (1.0f - esharp);

                vertWeight = 0.5f * esharp + (1.0f - esharp) * vertWeight;

                E_IT[4*i+2] = remap[lf->Subdivide()->GetID()];
                E_IT[4*i+3] = remap[rf->Subdivide()->GetID()];
            } else {
                E_IT[4*i+2] = -1;
                E_IT[4*i+3] = -1;
            }
            if (kernelType == FarKernelBatch::CATMARK_EDGE_VERTEX) {
                E_W[2*i+0] = vertWeight;
                E_W[2*i+1] = faceWeight;
            }
        }
        E_IT += 4 * nEdgeVertices;
        if (kernelType == FarKernelBatch::CATMARK_EDGE_VERTEX)
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

                            V_IT[V_IT_offset++] = remap[ e->GetLeftFace()->Subdivide()->GetID() ];

                            e = e->GetPrev()->GetOpposite();

                            if (e==start) break;
                        }
                        break;
                    }
                    case HbrVertex<T>::k_Crease : {

                        class GatherCreaseEdgesOperator : public HbrHalfedgeOperator<T> {
                        public:
                            HbrVertex<T> * vertex; int eidx[2]; int count; bool next;

                            GatherCreaseEdgesOperator(HbrVertex<T> * vtx, bool n) : vertex(vtx), count(0), next(n) { eidx[0]=-1; eidx[1]=-1; }

                            ~GatherCreaseEdgesOperator() { }

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
                        if (not useRestrictedVertexVertexKernels) {
                            // in the case of a k_Crease / k_Corner pass combination, we
                            // need to set the valence to -1 to tell the "B" Kernel to
                            // switch to k_Corner rule (as edge indices won't be -1)
                            if (V_ITa[5*i+1]==0)
                                V_ITa[5*i+1] = -1;
                        } else {
                            // in the case of a k_Corner, repeat the vertex
                            V_ITa[5*i+3] = V_ITa[5*i+2];
                            V_ITa[5*i+4] = V_ITa[5*i+2];
                        }

                    default : break;
                }

            if (not useRestrictedVertexVertexKernels) {
                if (rank>7)
                    // the k_Corner and k_Crease single-pass cases apply a weight of 1.0
                    // but this value is inverted in the kernel
                    V_W[i] = 0.0;
                else
                    V_W[i] = weights[0];
            }

            if (not useRestrictedVertexVertexKernels)
                batchFactory.AddVertex( i, rank );
            else
                batchFactory.AddCatmarkRestrictedVertex( i, rank, V_ITa[5*i+1] );
        }
        V_ITa += nVertVertices*5;
        if (not useRestrictedVertexVertexKernels)
            V_W += nVertVertices;

        // add batches for vert vertices
        if (nVertVertices > 0) {
            if (not useRestrictedVertexVertexKernels) {
                assert(hasStandardVertexVertexKernels);
                batchFactory.AppendCatmarkBatches(level, vertTableOffset, vertexOffset, batches);
            } else {
                batchFactory.AppendCatmarkRestrictedBatches(level, vertTableOffset, vertexOffset, batches);
            }
        }

        vertexOffset += nVertVertices;
        vertTableOffset += nVertVertices;
    }
    result->_vertsOffsets[maxlevel+1] = vertexOffset;
    return result;
}

template <class T, class U> bool
FarCatmarkSubdivisionTablesFactory<T,U>::CompareVertices( HbrVertex<T> const * x, HbrVertex<T> const * y ) {

    // Masks of the parent vertex decide for the current vertex.
    HbrVertex<T> * px=x->GetParentVertex(),
                 * py=y->GetParentVertex();

    int rankx = FarSubdivisionTablesFactory<T,U>::GetMaskRanking(px->GetMask(false), px->GetMask(true) );
    int ranky = FarSubdivisionTablesFactory<T,U>::GetMaskRanking(py->GetMask(false), py->GetMask(true) );

    assert( rankx!=0xFF and ranky!=0xFF );

    // Arrange regular vertices before irregular vertices within the same kernel
    if ((rankx <= 2 and ranky <= 2) or (rankx >= 3 and rankx <= 7 and ranky >= 3 and ranky <= 7) or (rankx >= 8 and ranky >= 8))
        return x->GetValence() == 4 and y->GetValence() != 4;
    else
        return rankx < ranky;
}

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* FAR_CATMARK_SUBDIVISION_TABLES_FACTORY_H */
