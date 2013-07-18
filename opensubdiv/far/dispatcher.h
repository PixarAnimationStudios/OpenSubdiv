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
#ifndef FAR_DISPATCHER_H
#define FAR_DISPATCHER_H

#include "../version.h"

#include "../far/mesh.h"
#include "../far/bilinearSubdivisionTables.h"
#include "../far/catmarkSubdivisionTables.h"
#include "../far/loopSubdivisionTables.h"
#include "../far/vertexEditTables.h"
#include "../far/kernelBatch.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

/// \brief Subdivision process encapsulation layer.
///
/// The Compute dispatcher allows client code to customize parts or the entire
/// computation process. This pattern aims at hiding the logic specific to
/// the subdivision algorithms and expose a simplified access to minimalistic
/// compute kernels. By default, meshes revert to a default dispatcher that
/// implements single-threaded CPU kernels.
///
/// - derive a dispatcher class from this one
/// - override the virtual functions
/// - pass the derived dispatcher to the factory (one instance can be shared by many meshes)
/// - call the FarMesh::Subdivide() to trigger computations
///
/// Note : the caller is responsible for deleting a custom dispatcher
///
class FarDispatcher {
public:
    /// \brief Launches the processing of a vector of kernel batches
    ///
    /// @param controller  refinement controller implementation
    ///
    /// @param batches     batches of kernels that need to be processed
    ///
    /// @param maxlevel    process vertex batches up to this level
    ///
    /// @param clientdata  custom client data passed to the controller
    ///
    template <class CONTROLLER>
    static void Refine(CONTROLLER const *controller, FarKernelBatchVector const & batches, int maxlevel, void * clientdata=0);
};

template <class CONTROLLER> void
FarDispatcher::Refine(CONTROLLER const *controller, FarKernelBatchVector const & batches, int maxlevel, void * clientdata) {

    for (int i = 0; i < (int)batches.size(); ++i) {
        const FarKernelBatch &batch = batches[i];

        if (maxlevel >= 0 && batch.GetLevel() >= maxlevel) continue;

        switch(batch.GetKernelType()) {
            case FarKernelBatch::CATMARK_FACE_VERTEX:
                controller->ApplyCatmarkFaceVerticesKernel(batch, clientdata);
                break;
            case FarKernelBatch::CATMARK_EDGE_VERTEX:
                controller->ApplyCatmarkEdgeVerticesKernel(batch, clientdata);
                break;
            case FarKernelBatch::CATMARK_VERT_VERTEX_B:
                controller->ApplyCatmarkVertexVerticesKernelB(batch, clientdata);
                break;
            case FarKernelBatch::CATMARK_VERT_VERTEX_A1:
                controller->ApplyCatmarkVertexVerticesKernelA1(batch, clientdata);
                break;
            case FarKernelBatch::CATMARK_VERT_VERTEX_A2:
                controller->ApplyCatmarkVertexVerticesKernelA2(batch, clientdata);
                break;

            case FarKernelBatch::LOOP_EDGE_VERTEX:
                controller->ApplyLoopEdgeVerticesKernel(batch, clientdata);
                break;
            case FarKernelBatch::LOOP_VERT_VERTEX_B:
                controller->ApplyLoopVertexVerticesKernelB(batch, clientdata);
                break;
            case FarKernelBatch::LOOP_VERT_VERTEX_A1:
                controller->ApplyLoopVertexVerticesKernelA1(batch, clientdata);
                break;
            case FarKernelBatch::LOOP_VERT_VERTEX_A2:
                controller->ApplyLoopVertexVerticesKernelA2(batch, clientdata);
                break;

            case FarKernelBatch::BILINEAR_FACE_VERTEX:
                controller->ApplyBilinearFaceVerticesKernel(batch, clientdata);
                break;
            case FarKernelBatch::BILINEAR_EDGE_VERTEX:
                controller->ApplyBilinearEdgeVerticesKernel(batch, clientdata);
                break;
            case FarKernelBatch::BILINEAR_VERT_VERTEX:
                controller->ApplyBilinearVertexVerticesKernel(batch, clientdata);
                break;

            case FarKernelBatch::HIERARCHICAL_EDIT:
                controller->ApplyVertexEdits(batch, clientdata);
                break;
        }
    }
}

// -----------------------------------------------------------------------------

/// \brief Far default controller implementation
///
/// This is Far's default implementation of a kernal batch controller.
///
template <class U>
class FarComputeController {

public:
    void Refine(FarMesh<U> * mesh, int maxlevel=-1) const;

    void ApplyBilinearFaceVerticesKernel(FarKernelBatch const &batch, void * clientdata) const;

    void ApplyBilinearEdgeVerticesKernel(FarKernelBatch const &batch, void * clientdata) const;

    void ApplyBilinearVertexVerticesKernel(FarKernelBatch const &batch, void * clientdata) const;


    void ApplyCatmarkFaceVerticesKernel(FarKernelBatch const &batch, void * clientdata) const;

    void ApplyCatmarkEdgeVerticesKernel(FarKernelBatch const &batch, void * clientdata) const;

    void ApplyCatmarkVertexVerticesKernelB(FarKernelBatch const &batch, void * clientdata) const;

    void ApplyCatmarkVertexVerticesKernelA1(FarKernelBatch const &batch, void * clientdata) const;

    void ApplyCatmarkVertexVerticesKernelA2(FarKernelBatch const &batch, void * clientdata) const;

    void ApplyLoopEdgeVerticesKernel(FarKernelBatch const &batch, void * clientdata) const;

    void ApplyLoopVertexVerticesKernelB(FarKernelBatch const &batch, void * clientdata) const;

    void ApplyLoopVertexVerticesKernelA1(FarKernelBatch const &batch, void * clientdata) const;

    void ApplyLoopVertexVerticesKernelA2(FarKernelBatch const &batch, void * clientdata) const;

    void ApplyVertexEdits(FarKernelBatch const &batch, void * clientdata) const;

    static FarComputeController _DefaultController;
    
private:

};

template<class U> FarComputeController<U> FarComputeController<U>::_DefaultController;

template <class U> void
FarComputeController<U>::Refine(FarMesh<U> *mesh, int maxlevel) const {

    FarDispatcher::Refine(this, mesh->GetKernelBatches(), maxlevel, mesh);
}

template <class U> void
FarComputeController<U>::ApplyBilinearFaceVerticesKernel(FarKernelBatch const &batch, void * clientdata) const {

    FarMesh<U> * mesh = static_cast<FarMesh<U> *>(clientdata);

    FarBilinearSubdivisionTables<U> const * subdivision =
        dynamic_cast<FarBilinearSubdivisionTables<U> const *>(mesh->GetSubdivisionTables());
    assert(subdivision);

    subdivision->computeFacePoints( batch.GetVertexOffset(),
                                    batch.GetTableOffset(),
                                    batch.GetStart(),
                                    batch.GetEnd(),
                                    clientdata );
}

template <class U> void
FarComputeController<U>::ApplyBilinearEdgeVerticesKernel(FarKernelBatch const &batch, void * clientdata) const {

    FarMesh<U> * mesh = static_cast<FarMesh<U> *>(clientdata);

    FarBilinearSubdivisionTables<U> const * subdivision =
        dynamic_cast<FarBilinearSubdivisionTables<U> const *>(mesh->GetSubdivisionTables());
    assert(subdivision);

    subdivision->computeEdgePoints( batch.GetVertexOffset(),
                                    batch.GetTableOffset(),
                                    batch.GetStart(),
                                    batch.GetEnd(),
                                    clientdata );
}

template <class U> void
FarComputeController<U>::ApplyBilinearVertexVerticesKernel(FarKernelBatch const &batch, void * clientdata) const {

    FarMesh<U> * mesh = static_cast<FarMesh<U> *>(clientdata);

    FarBilinearSubdivisionTables<U> const * subdivision =
        dynamic_cast<FarBilinearSubdivisionTables<U> const *>(mesh->GetSubdivisionTables());

    assert(subdivision);

    subdivision->computeVertexPoints( batch.GetVertexOffset(),
                                      batch.GetTableOffset(),
                                      batch.GetStart(),
                                      batch.GetEnd(),
                                      clientdata );
}

template <class U> void
FarComputeController<U>::ApplyCatmarkFaceVerticesKernel(FarKernelBatch const &batch, void * clientdata) const {

    FarMesh<U> * mesh = static_cast<FarMesh<U> *>(clientdata);

    FarCatmarkSubdivisionTables<U> const * subdivision =
        dynamic_cast<FarCatmarkSubdivisionTables<U> const *>(mesh->GetSubdivisionTables());

    assert(subdivision);

    subdivision->computeFacePoints( batch.GetVertexOffset(),
                                    batch.GetTableOffset(),
                                    batch.GetStart(),
                                    batch.GetEnd(),
                                    clientdata );
}

template <class U> void
FarComputeController<U>::ApplyCatmarkEdgeVerticesKernel(FarKernelBatch const &batch, void * clientdata) const {

    FarMesh<U> * mesh = static_cast<FarMesh<U> *>(clientdata);

    FarCatmarkSubdivisionTables<U> const * subdivision =
        dynamic_cast<FarCatmarkSubdivisionTables<U> const *>(mesh->GetSubdivisionTables());

    assert(subdivision);

    subdivision->computeEdgePoints( batch.GetVertexOffset(),
                                    batch.GetTableOffset(),
                                    batch.GetStart(),
                                    batch.GetEnd(),
                                    clientdata );
}

template <class U> void
FarComputeController<U>::ApplyCatmarkVertexVerticesKernelB(FarKernelBatch const &batch, void * clientdata) const {

    FarMesh<U> * mesh = static_cast<FarMesh<U> *>(clientdata);

    FarCatmarkSubdivisionTables<U> const * subdivision =
        dynamic_cast<FarCatmarkSubdivisionTables<U> const *>(mesh->GetSubdivisionTables());

    assert(subdivision);

    subdivision->computeVertexPointsB( batch.GetVertexOffset(),
                                       batch.GetTableOffset(),
                                       batch.GetStart(),
                                       batch.GetEnd(),
                                       clientdata );
}

template <class U> void
FarComputeController<U>::ApplyCatmarkVertexVerticesKernelA1(FarKernelBatch const &batch, void * clientdata) const {

    FarMesh<U> * mesh = static_cast<FarMesh<U> *>(clientdata);

    FarCatmarkSubdivisionTables<U> const * subdivision =
        dynamic_cast<FarCatmarkSubdivisionTables<U> const *>(mesh->GetSubdivisionTables());

    assert(subdivision);

    subdivision->computeVertexPointsA( batch.GetVertexOffset(),
                                       false,
                                       batch.GetTableOffset(),
                                       batch.GetStart(),
                                       batch.GetEnd(),
                                       clientdata );
}

template <class U> void
FarComputeController<U>::ApplyCatmarkVertexVerticesKernelA2(FarKernelBatch const &batch, void * clientdata) const {

    FarMesh<U> * mesh = static_cast<FarMesh<U> *>(clientdata);

    FarCatmarkSubdivisionTables<U> const * subdivision =
        dynamic_cast<FarCatmarkSubdivisionTables<U> const *>(mesh->GetSubdivisionTables());

    assert(subdivision);

    subdivision->computeVertexPointsA( batch.GetVertexOffset(),
                                       true,
                                       batch.GetTableOffset(),
                                       batch.GetStart(),
                                       batch.GetEnd(),
                                       clientdata );
}

template <class U> void
FarComputeController<U>::ApplyLoopEdgeVerticesKernel(FarKernelBatch const &batch, void * clientdata) const {

    FarMesh<U> * mesh = static_cast<FarMesh<U> *>(clientdata);

    FarLoopSubdivisionTables<U> const * subdivision =
        dynamic_cast<FarLoopSubdivisionTables<U> const *>(mesh->GetSubdivisionTables());

    assert(subdivision);

    subdivision->computeEdgePoints( batch.GetVertexOffset(),
                                    batch.GetTableOffset(),
                                    batch.GetStart(),
                                    batch.GetEnd(),
                                    clientdata );
}

template <class U> void
FarComputeController<U>::ApplyLoopVertexVerticesKernelB(FarKernelBatch const &batch, void * clientdata) const {

    FarMesh<U> * mesh = static_cast<FarMesh<U> *>(clientdata);

    FarLoopSubdivisionTables<U> const * subdivision =
        dynamic_cast<FarLoopSubdivisionTables<U> const *>(mesh->GetSubdivisionTables());

    assert(subdivision);

    subdivision->computeVertexPointsB( batch.GetVertexOffset(),
                                       batch.GetTableOffset(),
                                       batch.GetStart(),
                                       batch.GetEnd(),
                                       clientdata );
}

template <class U> void
FarComputeController<U>::ApplyLoopVertexVerticesKernelA1(FarKernelBatch const &batch, void * clientdata) const {

    FarMesh<U> * mesh = static_cast<FarMesh<U> *>(clientdata);

    FarLoopSubdivisionTables<U> const * subdivision =
        dynamic_cast<FarLoopSubdivisionTables<U> const *>(mesh->GetSubdivisionTables());

    assert(subdivision);

    subdivision->computeVertexPointsA( batch.GetVertexOffset(),
                                       false,
                                       batch.GetTableOffset(),
                                       batch.GetStart(),
                                       batch.GetEnd(),
                                       clientdata );
}

template <class U> void
FarComputeController<U>::ApplyLoopVertexVerticesKernelA2(FarKernelBatch const &batch, void * clientdata) const {

    FarMesh<U> * mesh = static_cast<FarMesh<U> *>(clientdata);

    FarLoopSubdivisionTables<U> const * subdivision =
        dynamic_cast<FarLoopSubdivisionTables<U> const *>(mesh->GetSubdivisionTables());

    assert(subdivision);

    subdivision->computeVertexPointsA( batch.GetVertexOffset(),
                                       true,
                                       batch.GetTableOffset(),
                                       batch.GetStart(),
                                       batch.GetEnd(),
                                       clientdata);
}

template <class U> void
FarComputeController<U>::ApplyVertexEdits(FarKernelBatch const &batch, void * clientdata) const {

    FarMesh<U> * mesh = static_cast<FarMesh<U> *>(clientdata);

    FarVertexEditTables<U> const * vertEdit = mesh->GetVertexEdit();

    if (vertEdit)
        vertEdit->computeVertexEdits( batch.GetTableIndex(),
                                      batch.GetVertexOffset(),
                                      batch.GetTableOffset(),
                                      batch.GetStart(),
                                      batch.GetEnd(),
                                      clientdata );
}

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* FAR_DISPATCHER_H */
