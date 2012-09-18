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

#include <cassert>
#include <vector>

#include "../version.h"

#include "../far/bilinearSubdivisionTables.h"
#include "../far/meshFactory.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

template <class T, class U> class FarMeshFactory;

// A specialized factory for FarBilinearSubdivisionTables
// Separating the factory allows us to isolate Far data structures from Hbr dependencies.
//
template <class T, class U> struct FarBilinearSubdivisionTablesFactory {
    static FarBilinearSubdivisionTables<U> * Create( FarMeshFactory<T,U> const * factory, FarMesh<U> * mesh, int maxlevel );
};

template <class T, class U> FarBilinearSubdivisionTables<U> * 
FarBilinearSubdivisionTablesFactory<T,U>::Create( FarMeshFactory<T,U> const * factory, FarMesh<U> * mesh, int maxlevel ) {

    assert( factory and mesh );

    FarBilinearSubdivisionTables<U> * result = new FarBilinearSubdivisionTables<U>(mesh, maxlevel);

    std::vector<int> const & remap = factory->_remapTable;

    // Allocate memory for the indexing tables
    result->_F_ITa.Resize(factory->GetNumFaceVerticesTotal(maxlevel)*2);
    result->_F_IT.Resize(factory->GetNumFacesTotal(maxlevel) - factory->GetNumFacesTotal(0));

    result->_E_IT.Resize(factory->GetNumEdgeVerticesTotal(maxlevel)*2);

    result->_V_ITa.Resize(factory->GetNumVertexVerticesTotal(maxlevel));

    for (int level=1; level<=maxlevel; ++level) {

        // pointer to the first vertex corresponding to this level
        result->_vertsOffsets[level] = factory->_vertVertIdx[level-1] +
                                     (int)factory->_vertVertsList[level-1].size();

        typename FarSubdivisionTables<U>::VertexKernelBatch * batch = & (result->_batches[level-1]);

        // Face vertices
        // "For each vertex, gather all the vertices from the parent face."
        int offset = 0;
        int * F_ITa = result->_F_ITa[level-1];
        unsigned int * F_IT = result->_F_IT[level-1];
        batch->kernelF = (int)factory->_faceVertsList[level].size();
        for (int i=0; i < batch->kernelF; ++i) {

            HbrVertex<T> * v = factory->_faceVertsList[level][i];
            assert(v);

            HbrFace<T> * f=v->GetParentFace();
            assert(f);

            int valence = f->GetNumVertices();

            F_ITa[2*i+0] = offset;
            F_ITa[2*i+1] = valence;

            for (int j=0; j<valence; ++j)
                F_IT[offset++] = remap[f->GetVertex(j)->GetID()];
        }
        result->_F_ITa.SetMarker(level, &F_ITa[2*batch->kernelF]);
        result->_F_IT.SetMarker(level, &F_IT[offset]);

        // Edge vertices

        // "Average the end-points of the parent edge"
        int * E_IT = result->_E_IT[level-1];
        batch->kernelE = (int)factory->_edgeVertsList[level].size();
        for (int i=0; i < batch->kernelE; ++i) {

            HbrVertex<T> * v = factory->_edgeVertsList[level][i];
            assert(v);
            HbrHalfedge<T> * e = v->GetParentEdge();
            assert(e);

            // get the indices 2 vertices from the parent edge
            E_IT[2*i+0] = remap[e->GetOrgVertex()->GetID()];
            E_IT[2*i+1] = remap[e->GetDestVertex()->GetID()];

        }
        result->_E_IT.SetMarker(level, &E_IT[2*batch->kernelE]);

        // Vertex vertices

        // "Pass down the parent vertex"
        offset = 0;
        int * V_ITa = result->_V_ITa[level-1];
        batch->kernelB.first = 0;
        batch->kernelB.second = (int)factory->_vertVertsList[level].size();
        for (int i=0; i < batch->kernelB.second; ++i) {

            HbrVertex<T> * v = factory->_vertVertsList[level][i],
                         * pv = v->GetParentVertex();
            assert(v and pv);

            V_ITa[i] = remap[pv->GetID()];

        }
        result->_V_ITa.SetMarker(level, &V_ITa[batch->kernelB.second]);
    }
    return result;
}

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* FAR_BILINEAR_SUBDIVISION_TABLES_FACTORY_H */
