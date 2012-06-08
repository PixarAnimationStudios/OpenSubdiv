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
#ifndef FAR_DISPATCHER_H
#define FAR_DISPATCHER_H

#include "../version.h"

#include "../far/mesh.h"
#include "../far/subdivisionTables.h"
#include "../far/bilinearSubdivisionTables.h"
#include "../far/catmarkSubdivisionTables.h"
#include "../far/loopSubdivisionTables.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

// Compute dispatcher : allows client code to this API to customize parts or the
// entire computation process. This pattern aims at hiding the logic specific to
// the subdivision algorithms and expose a simplified access to minimalistic  
// compute kernels. By default, meshes revert to a default dispatcher that implements
// single-threaded CPU kernels.
//
// - derive a dispatcher class from this one
// - override the virtual functions
// - pass the derived dispatched to the factory (one instance can be shared by many meshes)
// - call the FarMesh::Subdivide() to trigger computations
//
// Note : the caller is responsible for deleting a custom dispatcher
template <class T, class U=T> class FarDispatcher {


protected:
    friend class FarBilinearSubdivisionTables<T,U>;
    friend class FarCatmarkSubdivisionTables<T,U>;
    friend class FarLoopSubdivisionTables<T,U>;
    friend class FarMesh<T,U>;
    
    virtual void Refine(FarMesh<T,U> * mesh, int maxlevel, void * clientdata=0) const;
    

    virtual void ApplyBilinearFaceVerticesKernel(FarMesh<T,U> * mesh, int offset, int level, int start, int end, void * clientdata) const;

    virtual void ApplyBilinearEdgeVerticesKernel(FarMesh<T,U> * mesh, int offset, int level, int start, int end, void * clientdata) const;

    virtual void ApplyBilinearVertexVerticesKernel(FarMesh<T,U> * mesh, int offset, int level, int start, int end, void * clientdata) const;


    virtual void ApplyCatmarkFaceVerticesKernel(FarMesh<T,U> * mesh, int offset, int level, int start, int end, void * clientdata) const;

    virtual void ApplyCatmarkEdgeVerticesKernel(FarMesh<T,U> * mesh, int offset, int level, int start, int end, void * clientdata) const;

    virtual void ApplyCatmarkVertexVerticesKernelB(FarMesh<T,U> * mesh, int offset, int level, int start, int end, void * clientdata) const;

    virtual void ApplyCatmarkVertexVerticesKernelA(FarMesh<T,U> * mesh, int offset, bool pass, int level, int start, int end, void * clientdata) const;


    virtual void ApplyLoopEdgeVerticesKernel(FarMesh<T,U> * mesh, int offset, int level, int start, int end, void * clientdata) const;

    virtual void ApplyLoopVertexVerticesKernelB(FarMesh<T,U> * mesh, int offset, int level, int start, int end, void * clientdata) const;

    virtual void ApplyLoopVertexVerticesKernelA(FarMesh<T,U> * mesh, int offset, bool pass, int level, int start, int end, void * clientdata) const;

private:
    friend class FarMeshFactory<T,U>;
    
    static FarDispatcher _DefaultDispatcher;
};


template<class T, class U> FarDispatcher<T,U> FarDispatcher<T,U>::_DefaultDispatcher;

template <class T,class U> void
FarDispatcher<T,U>::Refine( FarMesh<T,U> * mesh, int maxlevel, void * data) const {

    assert(mesh);

    FarSubdivisionTables<T,U> const * tables = mesh->GetSubdivision();

    if ( (maxlevel < 0) )
        maxlevel=tables->GetMaxLevel();
    else
        maxlevel = std::min(maxlevel, tables->GetMaxLevel());
    
    for (int i=1; i<maxlevel; ++i)
        tables->Refine(i, data);
}

template <class T, class U> void 
FarDispatcher<T,U>::ApplyBilinearFaceVerticesKernel(FarMesh<T,U> * mesh, int offset, int level, int start, int end, void * clientdata) const {
    FarBilinearSubdivisionTables<T,U> const * subdivision = 
        dynamic_cast<FarBilinearSubdivisionTables<T,U> const *>(mesh->GetSubdivision());
    assert(subdivision);
    subdivision->computeFacePoints(offset, level, start, end, clientdata);
}

template <class T, class U> void 
FarDispatcher<T,U>::ApplyBilinearEdgeVerticesKernel(FarMesh<T,U> * mesh, int offset, int level, int start, int end, void * clientdata) const {
    FarBilinearSubdivisionTables<T,U> const * subdivision = 
        dynamic_cast<FarBilinearSubdivisionTables<T,U> const *>(mesh->GetSubdivision());
    assert(subdivision);
    subdivision->computeEdgePoints(offset, level, start, end, clientdata);
}

template <class T, class U> void 
FarDispatcher<T,U>::ApplyBilinearVertexVerticesKernel(FarMesh<T,U> * mesh, int offset, int level, int start, int end, void * clientdata) const {
    FarBilinearSubdivisionTables<T,U> const * subdivision = 
        dynamic_cast<FarBilinearSubdivisionTables<T,U> const *>(mesh->GetSubdivision());
    assert(subdivision);
    subdivision->computeVertexPoints(offset, level, start, end, clientdata);
}

template <class T, class U> void 
FarDispatcher<T,U>::ApplyCatmarkFaceVerticesKernel(FarMesh<T,U> * mesh, int offset, int level, int start, int end, void * clientdata) const {
    FarCatmarkSubdivisionTables<T,U> const * subdivision = 
        dynamic_cast<FarCatmarkSubdivisionTables<T,U> const *>(mesh->GetSubdivision());
    assert(subdivision);
    subdivision->computeFacePoints(offset, level, start, end, clientdata);
}

template <class T, class U> void 
FarDispatcher<T,U>::ApplyCatmarkEdgeVerticesKernel(FarMesh<T,U> * mesh, int offset, int level, int start, int end, void * clientdata) const {
    FarCatmarkSubdivisionTables<T,U> const * subdivision = 
        dynamic_cast<FarCatmarkSubdivisionTables<T,U> const *>(mesh->GetSubdivision());
    assert(subdivision);
    subdivision->computeEdgePoints(offset, level, start, end, clientdata);
}

template <class T, class U> void 
FarDispatcher<T,U>::ApplyCatmarkVertexVerticesKernelB(FarMesh<T,U> * mesh, int offset, int level, int start, int end, void * clientdata) const {
    FarCatmarkSubdivisionTables<T,U> const * subdivision = 
        dynamic_cast<FarCatmarkSubdivisionTables<T,U> const *>(mesh->GetSubdivision());
    assert(subdivision);
    subdivision->computeVertexPointsB(offset, level, start, end, clientdata);
}

template <class T, class U> void 
FarDispatcher<T,U>::ApplyCatmarkVertexVerticesKernelA(FarMesh<T,U> * mesh, int offset, bool pass, int level, int start, int end, void * clientdata) const {
    FarCatmarkSubdivisionTables<T,U> const * subdivision = 
        dynamic_cast<FarCatmarkSubdivisionTables<T,U> const *>(mesh->GetSubdivision());
    assert(subdivision);
    subdivision->computeVertexPointsA(offset, pass, level, start, end, clientdata);
}

template <class T, class U> void 
FarDispatcher<T,U>::ApplyLoopEdgeVerticesKernel(FarMesh<T,U> * mesh, int offset, int level, int start, int end, void * clientdata) const {
    FarLoopSubdivisionTables<T,U> const * subdivision = 
        dynamic_cast<FarLoopSubdivisionTables<T,U> const *>(mesh->GetSubdivision());
    assert(subdivision);
    subdivision->computeEdgePoints(offset, level, start, end, clientdata);
}

template <class T, class U> void 
FarDispatcher<T,U>::ApplyLoopVertexVerticesKernelB(FarMesh<T,U> * mesh, int offset, int level, int start, int end, void * clientdata) const {
    FarLoopSubdivisionTables<T,U> const * subdivision = 
        dynamic_cast<FarLoopSubdivisionTables<T,U> const *>(mesh->GetSubdivision());
    assert(subdivision);
    subdivision->computeVertexPointsB(offset, level, start, end, clientdata);
}

template <class T, class U> void 
FarDispatcher<T,U>::ApplyLoopVertexVerticesKernelA(FarMesh<T,U> * mesh, int offset, bool pass, int level, int start, int end, void * clientdata) const {
    FarLoopSubdivisionTables<T,U> const * subdivision = 
        dynamic_cast<FarLoopSubdivisionTables<T,U> const *>(mesh->GetSubdivision());
    assert(subdivision);
    subdivision->computeVertexPointsA(offset, pass, level, start, end, clientdata);
}

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* FAR_DISPATCHER_H */
