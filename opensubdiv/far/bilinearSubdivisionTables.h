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

#include "assert.h"

#include <vector>
#include <utility>

#include "../hbr/mesh.h"
#include "../hbr/bilinear.h"

#include "../version.h"

#include "../far/subdivisionTables.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

// Bilinear tables store the indexing tables required in order to compute
// the refined positions of a mesh without the help of a hierarchical data
// structure. The advantage of this representation is its ability to be executed
// in a massively parallel environment without data dependencies.
//
template <class T, class U=T> class FarBilinearSubdivisionTables : public FarSubdivisionTables<T,U> {

public:

    // Memory required to store the indexing tables
    virtual int GetMemoryUsed() const;
   
    // Compute the positions of refined vertices using the specified kernels
    virtual void Refine( int level, void * data=0 ) const;   
    
    // Table accessors
    typename FarSubdivisionTables<T,U>::template Table<unsigned int> const & Get_F_IT( ) const { return _F_IT; } 

    typename FarSubdivisionTables<T,U>::template Table<int> const & Get_F_ITa( ) const { return _F_ITa; } 

private:

    friend class FarMeshFactory<T,U>;
    friend class FarDispatcher<T,U>;
    
    // Constructor : build level table at depth 'level'
    FarBilinearSubdivisionTables( FarMeshFactory<T,U> const & factory, FarMesh<T,U> * mesh, int level );

    // Compute-kernel applied to vertices resulting from the refinement of a face.
    void computeFacePoints(int offset, int level, int start, int end, void * clientdata) const;

    // Compute-kernel applied to vertices resulting from the refinement of an edge.
    void computeEdgePoints(int offset, int level, int start, int end, void * clientdata) const;

    // Compute-kernel applied to vertices resulting from the refinement of a vertex
    void computeVertexPoints(int offset, int level, int start, int end, void * clientdata) const;
    
    
private:
   
    typename FarSubdivisionTables<T,U>::template Table<int>	   _F_ITa;
    typename FarSubdivisionTables<T,U>::template Table<unsigned int>  _F_IT;
};

template <class T, class U> int
FarBilinearSubdivisionTables<T,U>::GetMemoryUsed() const {
    return FarSubdivisionTables<T,U>::GetMemoryUsed()+
        _F_ITa.GetMemoryUsed()+
        _F_IT.GetMemoryUsed();
}

// Constructor - generates indexing tables matching the bilinear subdivision scheme.
//
// tables codices detail :
//
// codices detail :
//
// _F_ITa[0] : offset into _F_IT array of vertices making up the face
// _F_ITa[1] : valence of the face
//
// _E_ITa[0] : index of the org / dest vertices of the parent edge
// _E_ITa[1] : 
//
// _V_ITa[0] : index of the parent vertex
//
template <class T, class U>
FarBilinearSubdivisionTables<T,U>::FarBilinearSubdivisionTables( FarMeshFactory<T,U> const & factory, FarMesh<T,U> * mesh, int maxlevel ) : 
    FarSubdivisionTables<T,U>(mesh, maxlevel),
    _F_ITa(maxlevel+1),
    _F_IT(maxlevel+1)
{
    std::vector<int> const & remap = factory._remapTable;

    // Allocate memory for the indexing tables
    _F_ITa.Resize(factory.GetNumFaceVerticesTotal(maxlevel)*2);
    _F_IT.Resize(factory.GetNumFacesTotal(maxlevel) - factory.GetNumFacesTotal(0));

    this->_E_IT.Resize(factory.GetNumEdgeVerticesTotal(maxlevel)*2);

    this->_V_ITa.Resize(factory.GetNumVertexVerticesTotal(maxlevel));

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

        // "Average the end-points of the parent edge"
	unsigned int * E_IT = this->_E_IT[level-1];
	batch->kernelE = (int)factory._edgeVertsList[level].size();
	for (int i=0; i < batch->kernelE; ++i) {

	    HbrVertex<T> * v = factory._edgeVertsList[level][i];
            assert(v);
	    HbrHalfedge<T> * e = v->GetParentEdge();
	    assert(e);

	    // get the indices 2 vertices from the parent edge
            E_IT[2*i+0] = remap[e->GetOrgVertex()->GetID()];
            E_IT[2*i+1] = remap[e->GetDestVertex()->GetID()];

	}
        this->_E_IT.SetMarker(level, &E_IT[2*batch->kernelE]);
	
	// Vertex vertices 

        // "Pass down the parent vertex"
        offset = 0;
	int * V_ITa = this->_V_ITa[level-1];
	batch->kernelB.first = 0;
	batch->kernelB.second = (int)factory._vertVertsList[level].size();
	for (int i=0; i < batch->kernelB.second; ++i) {

            HbrVertex<T> * v = factory._vertVertsList[level][i],
	        	 * pv = v->GetParentVertex();
	    assert(v and pv);

            V_ITa[i] = remap[pv->GetID()];

	}
        this->_V_ITa.SetMarker(level, &V_ITa[batch->kernelB.second]);
    }
}

template <class T, class U> void
FarBilinearSubdivisionTables<T,U>::Refine( int level, void * clientdata ) const {
    
    assert(this->_mesh and level>0);
    
    typename FarSubdivisionTables<T,U>::VertexKernelBatch const * batch = & (this->_batches[level-1]);

    FarDispatcher<T,U> const * dispatch = this->_mesh->GetDispatcher();
    assert(dispatch);

    int offset = this->GetFirstVertexOffset(level);
    if (batch->kernelF>0)
        dispatch->ApplyBilinearFaceVerticesKernel(this->_mesh, offset, level, 0, batch->kernelF, clientdata);

    offset += this->GetNumFaceVertices(level);
    if (batch->kernelE>0)
        dispatch->ApplyBilinearEdgeVerticesKernel(this->_mesh, offset, level, 0, batch->kernelE, clientdata);

    offset += this->GetNumEdgeVertices(level);
    if (batch->kernelB.first < batch->kernelB.second)
        dispatch->ApplyBilinearVertexVerticesKernel(this->_mesh, offset, level, batch->kernelB.first, batch->kernelB.second, clientdata);
}

//
// Face-vertices compute Kernel - completely re-entrant
//

template <class T, class U> void 
FarBilinearSubdivisionTables<T,U>::computeFacePoints( int offset, int level, int start, int end, void * clientdata ) const {

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
FarBilinearSubdivisionTables<T,U>::computeEdgePoints( int offset,  int level, int start, int end, void * clientdata ) const {

    assert(this->_mesh);
    
    U * vsrc = &this->_mesh->GetVertices().at(0),
      * vdst = vsrc + offset + start;

    const unsigned int * E_IT = this->_E_IT[level-1];
      
    for (int i=start; i<end; ++i, ++vdst ) {
    
        vdst->Clear(clientdata);

        int eidx0 = E_IT[2*i+0],
            eidx1 = E_IT[2*i+1];
	    
	vdst->AddWithWeight( vsrc[eidx0], 0.5f, clientdata );
	vdst->AddWithWeight( vsrc[eidx1], 0.5f, clientdata );
	
        vdst->AddVaryingWithWeight( vsrc[eidx0], 0.5f, clientdata );
        vdst->AddVaryingWithWeight( vsrc[eidx1], 0.5f, clientdata );
    }
}

//
// Vertex-vertices compute Kernel - completely re-entrant
//

template <class T, class U> void 
FarBilinearSubdivisionTables<T,U>::computeVertexPoints( int offset, int level, int start, int end, void * clientdata ) const {

    assert(this->_mesh);
    
    U * vsrc = &this->_mesh->GetVertices().at(0),
      * vdst = vsrc + offset + start;

    const int * V_ITa = this->_V_ITa[level-1];

    for (int i=start; i<end; ++i, ++vdst ) {

        vdst->Clear(clientdata);
	
        int p=V_ITa[i];   // index of the parent vertex 

	vdst->AddWithWeight( vsrc[p], 1.0f, clientdata );
	vdst->AddVaryingWithWeight( vsrc[p], 1.0f, clientdata );
    }
}

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* FAR_BILINEAR_SUBDIVISION_TABLES_H */
