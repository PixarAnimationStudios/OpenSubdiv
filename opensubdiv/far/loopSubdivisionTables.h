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
#ifndef FAR_LOOP_SUBDIVISION_TABLES_H
#define FAR_LOOP_SUBDIVISION_TABLES_H

#include <cassert>
#include <cmath>
#include <vector>

#include "../version.h"

#include "../far/subdivisionTables.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

/// \brief Loop subdivision scheme tables.
///
/// Loop tables store the indexing tables required in order to compute
/// the refined positions of a mesh without the help of a hierarchical data
/// structure. The advantage of this representation is its ability to be executed
/// in a massively parallel environment without data dependencies.
///

template <class U> class FarLoopSubdivisionTables : public FarSubdivisionTables<U> {

public:

    /// Compute the positions of refined vertices using the specified kernels
    virtual void Apply( int level, void * data=0 ) const;


private:
    template <class X, class Y> friend struct FarLoopSubdivisionTablesFactory;
    friend class FarDispatcher<U>;

    FarLoopSubdivisionTables( FarMesh<U> * mesh, int maxlevel );

    // Compute-kernel applied to vertices resulting from the refinement of an edge.
    void computeEdgePoints(int offset, int level, int start, int end, void * clientdata) const;

    // Compute-kernel applied to vertices resulting from the refinement of a vertex
    // Kernel "A" Handles the k_Smooth and k_Dart rules
    void computeVertexPointsA(int offset, bool pass, int level, int start, int end, void * clientdata) const;

    // Compute-kernel applied to vertices resulting from the refinement of a vertex
    // Kernel "B" Handles the k_Crease and k_Corner rules
    void computeVertexPointsB(int offset,int level, int start, int end, void * clientdata) const;
};

template <class U>
FarLoopSubdivisionTables<U>::FarLoopSubdivisionTables( FarMesh<U> * mesh, int maxlevel ) :
    FarSubdivisionTables<U>(mesh, maxlevel)
{ }


template <class U> void
FarLoopSubdivisionTables<U>::Apply( int level, void * clientdata ) const
{
    assert(this->_mesh and level>0);

    typename FarSubdivisionTables<U>::VertexKernelBatch const * batch = & (this->_batches[level-1]);

    FarDispatcher<U> const * dispatch = this->_mesh->GetDispatcher();
    assert(dispatch);

    int offset = this->GetFirstVertexOffset(level);
    if (batch->kernelE>0)
        dispatch->ApplyLoopEdgeVerticesKernel(this->_mesh, offset, level, 0, batch->kernelE, clientdata);

    offset += this->GetNumEdgeVertices(level);
    if (batch->kernelB.first < batch->kernelB.second)
        dispatch->ApplyLoopVertexVerticesKernelB(this->_mesh, offset, level, batch->kernelB.first, batch->kernelB.second, clientdata);
    if (batch->kernelA1.first < batch->kernelA1.second)
        dispatch->ApplyLoopVertexVerticesKernelA(this->_mesh, offset, false, level, batch->kernelA1.first, batch->kernelA1.second, clientdata);
    if (batch->kernelA2.first < batch->kernelA2.second)
        dispatch->ApplyLoopVertexVerticesKernelA(this->_mesh, offset, true, level, batch->kernelA2.first, batch->kernelA2.second, clientdata);
}

//
// Edge-vertices compute Kernel - completely re-entrant
//

template <class U> void
FarLoopSubdivisionTables<U>::computeEdgePoints( int offset, int level, int start, int end, void * clientdata ) const {

    assert(this->_mesh);

    U * vsrc = &this->_mesh->GetVertices().at(0),
      * vdst = vsrc + offset + start;

    const int * E_IT = this->_E_IT[level-1];
    const float * E_W = this->_E_W[level-1];

    for (int i=start; i<end; ++i, ++vdst ) {

        vdst->Clear(clientdata);

        int eidx0 = E_IT[4*i+0],
            eidx1 = E_IT[4*i+1],
            eidx2 = E_IT[4*i+2],
            eidx3 = E_IT[4*i+3];

        float endPtWeight = E_W[i*2+0];

        // Fully sharp edge : endPtWeight = 0.5f
        vdst->AddWithWeight( vsrc[eidx0], endPtWeight, clientdata );
        vdst->AddWithWeight( vsrc[eidx1], endPtWeight, clientdata );

        if (eidx2!=-1) {
            // Apply fractional sharpness
            float oppPtWeight = E_W[i*2+1];

            vdst->AddWithWeight( vsrc[eidx2], oppPtWeight, clientdata );
            vdst->AddWithWeight( vsrc[eidx3], oppPtWeight, clientdata );
        }

        vdst->AddVaryingWithWeight( vsrc[eidx0], 0.5f, clientdata );
        vdst->AddVaryingWithWeight( vsrc[eidx1], 0.5f, clientdata );
    }
}

//
// Vertex-vertices compute Kernels "A" and "B" - completely re-entrant
//

// multi-pass kernel handling k_Crease and k_Corner rules
template <class U> void
FarLoopSubdivisionTables<U>::computeVertexPointsA( int offset, bool pass, int level, int start, int end, void * clientdata ) const {

    assert(this->_mesh);

    U * vsrc = &this->_mesh->GetVertices().at(0),
      * vdst = vsrc + offset + start;

    const int * V_ITa = this->_V_ITa[level-1];
    const float * V_W = this->_V_W[level-1];

    for (int i=start; i<end; ++i, ++vdst ) {

        if (not pass)
            vdst->Clear(clientdata);

        int     n=V_ITa[5*i+1], // number of vertices in the _VO_IT array (valence)
                p=V_ITa[5*i+2], // index of the parent vertex
            eidx0=V_ITa[5*i+3], // index of the first crease rule edge
            eidx1=V_ITa[5*i+4]; // index of the second crease rule edge

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
template <class U> void
FarLoopSubdivisionTables<U>::computeVertexPointsB( int offset, int level, int start, int end, void * clientdata ) const {

    assert(this->_mesh);

    U * vsrc = &this->_mesh->GetVertices().at(0),
      * vdst = vsrc + offset + start;

    const int * V_ITa = this->_V_ITa[level-1];
    const unsigned int * V_IT = this->_V_IT[level-1];
    const float * V_W = this->_V_W[level-1];

    for (int i=start; i<end; ++i, ++vdst ) {

        vdst->Clear(clientdata);

        int h = V_ITa[5*i  ], // offset of the vertices in the _V0_IT array
            n = V_ITa[5*i+1], // number of vertices in the _VO_IT array (valence)
            p = V_ITa[5*i+2]; // index of the parent vertex

        float weight = V_W[i],
                  wp = 1.0f/n,
                beta = 0.25f * cosf((float)M_PI * 2.0f * wp) + 0.375f;
        beta = beta*beta;
        beta = (0.625f-beta)*wp;

        vdst->AddWithWeight( vsrc[p], weight * (1.0f-(beta*n)), clientdata);

        for (int j=0; j<n; ++j)
            vdst->AddWithWeight( vsrc[V_IT[h+j]], weight * beta );

        vdst->AddVaryingWithWeight( vsrc[p], 1.0f, clientdata );
    }
}

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* FAR_LOOP_SUBDIVISION_TABLES_H */
