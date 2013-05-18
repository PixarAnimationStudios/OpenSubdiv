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

#ifndef FAR_BILINEAR_SUBDIVISION_TABLES_H
#define FAR_BILINEAR_SUBDIVISION_TABLES_H

#include "../version.h"

#include "../far/subdivisionTables.h"

#include <cassert>
#include <vector>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

/// \brief Bilinear subdivision scheme tables.
///
/// Bilinear tables store the indexing tables required in order to compute
/// the refined positions of a mesh without the help of a hierarchical data
/// structure. The advantage of this representation is its ability to be executed
/// in a massively parallel environment without data dependencies.
///
template <class U> class FarBilinearSubdivisionTables : public FarSubdivisionTables<U> {

public:

    /// Returns the number of indexing tables needed to represent this particular
    /// subdivision scheme.
    virtual int GetNumTables() const { return 7; }

private:
    template <class X, class Y> friend class FarBilinearSubdivisionTablesFactory;
    template <class X, class Y> friend class FarMultiMeshFactory;
    template <class CONTROLLER> friend class FarComputeController;

    FarBilinearSubdivisionTables( FarMesh<U> * mesh, int maxlevel );

    // Compute-kernel applied to vertices resulting from the refinement of a face.
    void computeFacePoints(int vertexOffset, int tableOffset, int start, int end, void * clientdata) const;

    // Compute-kernel applied to vertices resulting from the refinement of an edge.
    void computeEdgePoints(int vertexOffset, int tableOffset, int start, int end, void * clientdata) const;

    // Compute-kernel applied to vertices resulting from the refinement of a vertex
    void computeVertexPoints(int vertexOffset, int tableOffset, int start, int end, void * clientdata) const;

};

template <class U>
FarBilinearSubdivisionTables<U>::FarBilinearSubdivisionTables( FarMesh<U> * mesh, int maxlevel ) :
    FarSubdivisionTables<U>(mesh, maxlevel)
{ }

//
// Face-vertices compute Kernel - completely re-entrant
//

template <class U> void
FarBilinearSubdivisionTables<U>::computeFacePoints( int offset, int tableOffset, int start, int end, void * clientdata ) const {

    assert(this->_mesh);

    U * vsrc = &this->_mesh->GetVertices().at(0),
      * vdst = vsrc + offset + start;

    for (int i=start+tableOffset; i<end+tableOffset; ++i, ++vdst ) {

        vdst->Clear(clientdata);

        int h = this->_F_ITa[2*i  ],
            n = this->_F_ITa[2*i+1];
        float weight = 1.0f/n;

        for (int j=0; j<n; ++j) {
             vdst->AddWithWeight( vsrc[ this->_F_IT[h+j] ], weight, clientdata );
             vdst->AddVaryingWithWeight( vsrc[ this->_F_IT[h+j] ], weight, clientdata );
        }
    }
}

//
// Edge-vertices compute Kernel - completely re-entrant
//

template <class U> void
FarBilinearSubdivisionTables<U>::computeEdgePoints( int offset,  int tableOffset, int start, int end, void * clientdata ) const {

    assert(this->_mesh);

    U * vsrc = &this->_mesh->GetVertices().at(0),
      * vdst = vsrc + offset + start;

    for (int i=start+tableOffset; i<end+tableOffset; ++i, ++vdst ) {

        vdst->Clear(clientdata);

        int eidx0 = this->_E_IT[2*i+0],
            eidx1 = this->_E_IT[2*i+1];

        vdst->AddWithWeight( vsrc[eidx0], 0.5f, clientdata );
        vdst->AddWithWeight( vsrc[eidx1], 0.5f, clientdata );

        vdst->AddVaryingWithWeight( vsrc[eidx0], 0.5f, clientdata );
        vdst->AddVaryingWithWeight( vsrc[eidx1], 0.5f, clientdata );
    }
}

//
// Vertex-vertices compute Kernel - completely re-entrant
//

template <class U> void
FarBilinearSubdivisionTables<U>::computeVertexPoints( int offset, int tableOffset, int start, int end, void * clientdata ) const {

    assert(this->_mesh);

    U * vsrc = &this->_mesh->GetVertices().at(0),
      * vdst = vsrc + offset + start;

    for (int i=start+tableOffset; i<end+tableOffset; ++i, ++vdst ) {

        vdst->Clear(clientdata);

        int p = this->_V_ITa[i];   // index of the parent vertex

        vdst->AddWithWeight( vsrc[p], 1.0f, clientdata );
        vdst->AddVaryingWithWeight( vsrc[p], 1.0f, clientdata );
    }
}

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* FAR_BILINEAR_SUBDIVISION_TABLES_H */
