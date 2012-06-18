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
#ifndef FAR_CATMARK_SUBDIVISION_TABLES_H
#define FAR_CATMARK_SUBDIVISION_TABLES_H

#include <assert.h>
#include <vector>
#include <utility>

#include "../hbr/mesh.h"
#include "../hbr/catmark.h"

#include "../version.h"

#include "../far/subdivisionTables.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

// Catmull-Clark tables store the indexing tables required in order to compute
// the refined positions of a mesh without the help of a hierarchical data
// structure. The advantage of this representation is its ability to be executed
// in a massively parallel environment without data dependencies.
//

template <class T, class U=T> class FarCatmarkSubdivisionTables : public FarSubdivisionTables<T,U> {

public:

    // Memory required to store the indexing tables
    virtual int GetMemoryUsed() const;
   
    // Compute the positions of refined vertices using the specified kernels
    virtual void Refine( int level, void * data=0 ) const;   
    
    // Table accessors
    typename FarSubdivisionTables<T,U>::template Table<unsigned int> const & Get_F_IT( ) const { return _F_IT; } 

    typename FarSubdivisionTables<T,U>::template Table<int> const & Get_F_ITa( ) const { return _F_ITa; } 

    // Returns the number of indexing tables needed to represent this particular
    // subdivision scheme.
    virtual int GetNumTables() const { return 7; }

private:

    friend class FarMeshFactory<T,U>;
    friend class FarDispatcher<T,U>;
    
    // Constructor : build level table at depth 'level'
    FarCatmarkSubdivisionTables( FarMeshFactory<T,U> const & factory, FarMesh<T,U> * mesh, int level );

    // Compute-kernel applied to vertices resulting from the refinement of a face.
    void computeFacePoints(int offset, int level, int start, int end, void * clientdata) const;

    // Compute-kernel applied to vertices resulting from the refinement of an edge.
    void computeEdgePoints(int offset, int level, int start, int end, void * clientdata) const;

    // Compute-kernel applied to vertices resulting from the refinement of a vertex
    // Kernel "A" Handles the k_Smooth and k_Dart rules
    void computeVertexPointsA(int offset, bool pass, int level, int start, int end, void * clientdata) const;
    
    // Compute-kernel applied to vertices resulting from the refinement of a vertex
    // Kernel "B" Handles the k_Crease and k_Corner rules
    void computeVertexPointsB(int offset, int level, int start, int end, void * clientdata) const;
    
private:
   
    typename FarSubdivisionTables<T,U>::template Table<int>           _F_ITa;
    typename FarSubdivisionTables<T,U>::template Table<unsigned int>  _F_IT;
};

template <class T, class U> int
FarCatmarkSubdivisionTables<T,U>::GetMemoryUsed() const {
    return FarSubdivisionTables<T,U>::GetMemoryUsed()+
        _F_ITa.GetMemoryUsed()+
        _F_IT.GetMemoryUsed();
}

// Constructor - generates indexing tables matching the Catmull-Clark subdivision scheme.
//
// tables codices detail :
//
// _F_ITa[0] : offset into _F_IT array of vertices making up the face
// _F_ITa[1] : valence of the face
//
// _E_ITa[0] : index of the org / dest vertices of the parent edge
// _E_ITa[1] : 
// _E_ITa[2] : index of vertices refined from the faces left / right
// _E_ITa[3] : of the parent edge
//
// _V_ITa[0] : offset to the corresponding adjacent vertices into _V0_IT
// _V_ITa[1] : number of adjacent indices
// _V_ITa[2] : index of the parent vertex
// _V_ITa[3] : index of adjacent edge 0 (k_Crease rule)
// _V_ITa[4] : index of adjacent edge 1 (k_Crease rule)
//
template <class T, class U>
FarCatmarkSubdivisionTables<T,U>::FarCatmarkSubdivisionTables( FarMeshFactory<T,U> const & factory, FarMesh<T,U> * mesh, int maxlevel ) : 
    FarSubdivisionTables<T,U>(mesh, maxlevel),
    _F_ITa(maxlevel+1),
    _F_IT(maxlevel+1)
{
    std::vector<int> const & remap = factory._remapTable;

    // Allocate memory for the indexing tables
    _F_ITa.Resize(factory.GetNumFaceVerticesTotal(maxlevel)*2);
    _F_IT.Resize(factory.GetNumFacesTotal(maxlevel) - factory.GetNumFacesTotal(0));

    this->_E_IT.Resize(factory.GetNumEdgeVerticesTotal(maxlevel)*4);
    this->_E_W.Resize(factory.GetNumEdgeVerticesTotal(maxlevel)*2);

    this->_V_ITa.Resize(factory.GetNumVertexVerticesTotal(maxlevel)*5);
    this->_V_IT.Resize(factory.GetNumAdjacentVertVerticesTotal(maxlevel)*2);
    this->_V_W.Resize(factory.GetNumVertexVerticesTotal(maxlevel));

    for (int level=1; level<=maxlevel; ++level) {

        // pointer to the first vertex corresponding to this level
        this->_vertsOffsets[level] = factory._vertVertIdx[level-1] + 
                                     factory._vertVertsList[level-1].size();

        typename FarSubdivisionTables<T,U>::VertexKernelBatch * batch = & (this->_batches[level-1]);

        // Face vertices 
        // "For each vertex, gather all the vertices from the parent face."
        int offset = 0;
        int * F_ITa = this->_F_ITa[level-1];
        unsigned int * F_IT = this->_F_IT[level-1];
        batch->kernelF = (int)factory._faceVertsList[level].size();
        for (int i=0; i < batch->kernelF; ++i) {

            HbrVertex<T> * v = factory._faceVertsList[level][i];
            assert(v);

            HbrFace<T> * f=v->GetParentFace();
            assert(f);

            int valence = f->GetNumVertices();

            F_ITa[2*i+0] = offset;
            F_ITa[2*i+1] = valence;

            for (int j=0; j<valence; ++j)
                F_IT[offset++] = remap[f->GetVertex(j)->GetID()];
        }
        _F_ITa.SetMarker(level, &F_ITa[2*batch->kernelF]);
        _F_IT.SetMarker(level, &F_IT[offset]);

        // Edge vertices 

        // Triangular interpolation mode :
        // see "smoothtriangle" tag introduced in prman 3.9 and HbrCatmarkSubdivision<T>
        typename HbrCatmarkSubdivision<T>::TriangleSubdivision triangleMethod = 
            dynamic_cast<HbrCatmarkSubdivision<T> *>(factory._hbrMesh->GetSubdivision())->GetTriangleSubdivisionMethod();

        // "For each vertex, gather the 2 vertices from the parent edege and the
        // 2 child vertices from the faces to the left and right of that edge.
        // Adjust if edge has a crease or is on a boundary."
        unsigned int * E_IT = this->_E_IT[level-1];
        float * E_W = this->_E_W[level-1];
        batch->kernelE = (int)factory._edgeVertsList[level].size();
        for (int i=0; i < batch->kernelE; ++i) {

            HbrVertex<T> * v = factory._edgeVertsList[level][i];
            assert(v);
            HbrHalfedge<T> * e = v->GetParentEdge();
            assert(e);

            float esharp = e->GetSharpness();

            // get the indices 2 vertices from the parent edge
            E_IT[4*i+0] = remap[e->GetOrgVertex()->GetID()];
            E_IT[4*i+1] = remap[e->GetDestVertex()->GetID()];

            float faceWeight=0.5f, vertWeight=0.5f;

            // in the case of a fractional sharpness, set the adjacent faces vertices
            if (!e->IsBoundary() && esharp <= 1.0f) {

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
            E_W[2*i+0] = vertWeight;
            E_W[2*i+1] = faceWeight;
        }
        this->_E_IT.SetMarker(level, &E_IT[4*batch->kernelE]);
        this->_E_W.SetMarker(level, &E_W[2*batch->kernelE]);
        
        // Vertex vertices 

        batch->InitVertexKernels( factory._vertVertsList[level].size(), 0 );

        offset = 0;
        int * V_ITa = this->_V_ITa[level-1];
        unsigned int * V_IT = this->_V_IT[level-1];
        float * V_W = this->_V_W[level-1];
        int nverts = (int)factory._vertVertsList[level].size();
        for (int i=0; i < nverts; ++i) {

            HbrVertex<T> * v = factory._vertVertsList[level][i],
                         * pv = v->GetParentVertex();
            assert(v and pv);

            // Look at HbrCatmarkSubdivision<T>::Subdivide for more details about
            // the multi-pass interpolation 
            int masks[2], npasses;
            float weights[2];
            masks[0] = pv->GetMask(false);
            masks[1] = pv->GetMask(true);

            // If the masks are identical, only a single pass is necessary. If the
            // vertex is transitionning to another rule, two passes are necessary,
            // except when transitionning from k_Dart to k_Smooth : the same
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

            int rank = this->getMaskRanking(masks[0], masks[1]);

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
                             
                            V_IT[offset++] = remap[ e->GetLeftFace()->Subdivide()->GetID() ];

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
        this->_V_ITa.SetMarker(level, &V_ITa[5*nverts]);
        this->_V_IT.SetMarker(level, &V_IT[offset]);
        this->_V_W.SetMarker(level, &V_W[nverts]);

        batch->kernelB.second++;
        batch->kernelA1.second++;
        batch->kernelA2.second++;
    }
}

template <class T, class U> void
FarCatmarkSubdivisionTables<T,U>::Refine( int level, void * clientdata ) const {
    
    assert(this->_mesh and level>0);
    
    typename FarSubdivisionTables<T,U>::VertexKernelBatch const * batch = & (this->_batches[level-1]);

    FarDispatcher<T,U> const * dispatch = this->_mesh->GetDispatcher();
    assert(dispatch);

    int offset = this->GetFirstVertexOffset(level);
    if (batch->kernelF>0)
        dispatch->ApplyCatmarkFaceVerticesKernel(this->_mesh, offset, level, 0, batch->kernelF, clientdata);

    offset += this->GetNumFaceVertices(level);
    if (batch->kernelE>0)
        dispatch->ApplyCatmarkEdgeVerticesKernel(this->_mesh, offset, level, 0, batch->kernelE, clientdata);

    offset += this->GetNumEdgeVertices(level);
    if (batch->kernelB.first < batch->kernelB.second)
        dispatch->ApplyCatmarkVertexVerticesKernelB(this->_mesh, offset, level, batch->kernelB.first, batch->kernelB.second, clientdata);
    if (batch->kernelA1.first < batch->kernelA1.second)
        dispatch->ApplyCatmarkVertexVerticesKernelA(this->_mesh, offset, false, level, batch->kernelA1.first, batch->kernelA1.second, clientdata);
    if (batch->kernelA2.first < batch->kernelA2.second)
        dispatch->ApplyCatmarkVertexVerticesKernelA(this->_mesh, offset, true, level, batch->kernelA2.first, batch->kernelA2.second, clientdata);
}

//
// Face-vertices compute Kernel - completely re-entrant
//

template <class T, class U> void 
FarCatmarkSubdivisionTables<T,U>::computeFacePoints( int offset, int level, int start, int end, void * clientdata ) const {

    assert(this->_mesh);
    
    U * vsrc = &this->_mesh->GetVertices().at(0),
      * vdst = vsrc + offset + start;

    const int * F_ITa = _F_ITa[level-1];
    const unsigned int * F_IT = _F_IT[level-1];

    for (int i=start; i<end; ++i, ++vdst ) {
        
        vdst->Clear(clientdata);
        
        int h = F_ITa[2*i  ], 
            n = F_ITa[2*i+1];
        float weight = 1.0f/n;

        for (int j=0; j<n; ++j) {
             vdst->AddWithWeight( vsrc[ F_IT[h+j] ], weight, clientdata );
             vdst->AddVaryingWithWeight( vsrc[ F_IT[h+j] ], weight, clientdata );
        }
    }
}

//
// Edge-vertices compute Kernel - completely re-entrant
//

template <class T, class U> void 
FarCatmarkSubdivisionTables<T,U>::computeEdgePoints( int offset,  int level, int start, int end, void * clientdata ) const {

    assert(this->_mesh);
    
    U * vsrc = &this->_mesh->GetVertices().at(0),
      * vdst = vsrc + offset + start;

    const unsigned int * E_IT = this->_E_IT[level-1];
    const float * E_W = this->_E_W[level-1];
      
    for (int i=start; i<end; ++i, ++vdst ) {
    
        vdst->Clear(clientdata);

        int eidx0 = E_IT[4*i+0],
            eidx1 = E_IT[4*i+1],
            eidx2 = E_IT[4*i+2],
            eidx3 = E_IT[4*i+3];
            
        float vertWeight = E_W[i*2+0];

        // Fully sharp edge : vertWeight = 0.5f
        vdst->AddWithWeight( vsrc[eidx0], vertWeight, clientdata );
        vdst->AddWithWeight( vsrc[eidx1], vertWeight, clientdata );
        
        if (eidx2!=-1) {
            // Apply fractional sharpness
            float faceWeight = E_W[i*2+1];
            
            vdst->AddWithWeight( vsrc[eidx2], faceWeight, clientdata );
            vdst->AddWithWeight( vsrc[eidx3], faceWeight, clientdata );
        }

        vdst->AddVaryingWithWeight( vsrc[eidx0], 0.5f, clientdata );
        vdst->AddVaryingWithWeight( vsrc[eidx1], 0.5f, clientdata );
    }
}

//
// Vertex-vertices compute Kernels "A" and "B" - completely re-entrant
//

// multi-pass kernel handling k_Crease and k_Corner rules
template <class T, class U> void 
FarCatmarkSubdivisionTables<T,U>::computeVertexPointsA( int offset, bool pass, int level, int start, int end, void * clientdata ) const {

    assert(this->_mesh);
    
    U * vsrc = &this->_mesh->GetVertices().at(0),
      * vdst = vsrc + offset + start;

    const int * V_ITa = this->_V_ITa[level-1];
    const float * V_W = this->_V_W[level-1];

    for (int i=start; i<end; ++i, ++vdst ) {

        if (not pass)
            vdst->Clear(clientdata);
        
        int     n=V_ITa[5*i+1],   // number of vertices in the _VO_IT array (valence)
                p=V_ITa[5*i+2],   // index of the parent vertex 
            eidx0=V_ITa[5*i+3],   // index of the first crease rule edge
            eidx1=V_ITa[5*i+4];   // index of the second crease rule edge

        float weight = pass ? V_W[i] : 1.0f - V_W[i];

        // In the case of fractional weight, the weight must be inverted since 
        // the value is shared with the k_Smooth kernel (statistically the 
        // k_Smooth kernel runs much more often than this one)
        if (weight>0.0f and weight<1.0f and n>0)
            weight=1.0f-weight;

        // In the case of a k_Corner / k_Crease combination, the edge indices
        // won't be null,  so we use a -1 valence to detect that particular case
        if (eidx0==-1 or (pass==false and (n==-1)) ) {
            // k_Corner case
            vdst->AddWithWeight( vsrc[p], weight, clientdata );
        } else {
            // k_Crease case
            vdst->AddWithWeight( vsrc[p], weight * 0.75f, clientdata );
            vdst->AddWithWeight( vsrc[eidx0], weight * 0.125f, clientdata );
            vdst->AddWithWeight( vsrc[eidx1], weight * 0.125f, clientdata );
        }
        vdst->AddVaryingWithWeight( vsrc[p], 1.0f, clientdata );
    }
}

// multi-pass kernel handling k_Dart and k_Smooth rules
template <class T, class U> void 
FarCatmarkSubdivisionTables<T,U>::computeVertexPointsB( int offset, int level, int start, int end, void * clientdata ) const {

    assert(this->_mesh);
    
    U * vsrc = &this->_mesh->GetVertices().at(0),
      * vdst = vsrc + offset + start;

    const int * V_ITa = this->_V_ITa[level-1];
    const unsigned int * V_IT = this->_V_IT[level-1];
    const float * V_W = this->_V_W[level-1];

    for (int i=start; i<end; ++i, ++vdst ) {
         
        vdst->Clear(clientdata);
        
        int h = V_ITa[5*i  ],     // offset of the vertices in the _V0_IT array
            n = V_ITa[5*i+1],     // number of vertices in the _VO_IT array (valence)
            p = V_ITa[5*i+2];     // index of the parent vertex
            
        float weight = V_W[i],
                  wp = 1.0f/(n*n),
                  wv = (n-2.0f)*n*wp;

        vdst->AddWithWeight( vsrc[p], weight * wv, clientdata );
        
        for (int j=0; j<n; ++j) {
            vdst->AddWithWeight( vsrc[V_IT[h+j*2  ]], weight * wp, clientdata );
            vdst->AddWithWeight( vsrc[V_IT[h+j*2+1]], weight * wp, clientdata );
        }
        vdst->AddVaryingWithWeight( vsrc[p], 1.0f, clientdata );
    }    
}

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* FAR_CATMARK_SUBDIVISION_TABLES_H */
