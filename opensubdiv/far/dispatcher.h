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

#ifndef FAR_DISPATCHER_H
#define FAR_DISPATCHER_H

#include "../version.h"

#include "../far/subdivisionTables.h"
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
    /// \brief Launches the processing of a kernel batch
    /// returns true if the batch is handled, otherwise returns false (i.e. user defined kernel)
    ///
    /// @param controller  refinement controller implementation
    ///
    /// @param context     refinement context implementation
    ///
    /// @param batch       a batch of kernel that need to be processed
    ///
    template <class CONTROLLER, class CONTEXT>
    static bool ApplyKernel(CONTROLLER *controller, CONTEXT *context, FarKernelBatch const &batch);

    /// \brief Launches the processing of a vector of kernel batches
    /// this is a convenient API for controllers which don't have any user defined kernels.
    ///
    /// @param controller  refinement controller implementation (vertex array)
    ///
    /// @param context     refinement context implementation (subdivision tables)
    ///                    passed to the controller.
    ///
    /// @param batches     batches of kernels that need to be processed
    ///
    /// @param maxlevel    process vertex batches up to this level
    ///
    template <class CONTROLLER, class CONTEXT>
    static void Refine(CONTROLLER *controller, CONTEXT *context, FarKernelBatchVector const & batches, int maxlevel);

};

template <class CONTROLLER, class CONTEXT> bool
FarDispatcher::ApplyKernel(CONTROLLER *controller, CONTEXT *context, FarKernelBatch const &batch) {

    switch(batch.GetKernelType()) {
        case FarKernelBatch::CATMARK_FACE_VERTEX:
            controller->ApplyCatmarkFaceVerticesKernel(batch, context);
            break;
        case FarKernelBatch::CATMARK_QUAD_FACE_VERTEX:
            controller->ApplyCatmarkQuadFaceVerticesKernel(batch, context);
            break;
        case FarKernelBatch::CATMARK_TRI_QUAD_FACE_VERTEX:
            controller->ApplyCatmarkTriQuadFaceVerticesKernel(batch, context);
            break;
        case FarKernelBatch::CATMARK_EDGE_VERTEX:
            controller->ApplyCatmarkEdgeVerticesKernel(batch, context);
            break;
        case FarKernelBatch::CATMARK_RESTRICTED_EDGE_VERTEX:
            controller->ApplyCatmarkRestrictedEdgeVerticesKernel(batch, context);
            break;
        case FarKernelBatch::CATMARK_VERT_VERTEX_B:
            controller->ApplyCatmarkVertexVerticesKernelB(batch, context);
            break;
        case FarKernelBatch::CATMARK_VERT_VERTEX_A1:
            controller->ApplyCatmarkVertexVerticesKernelA1(batch, context);
            break;
        case FarKernelBatch::CATMARK_VERT_VERTEX_A2:
            controller->ApplyCatmarkVertexVerticesKernelA2(batch, context);
            break;
        case FarKernelBatch::CATMARK_RESTRICTED_VERT_VERTEX_B1:
            controller->ApplyCatmarkRestrictedVertexVerticesKernelB1(batch, context);
            break;
        case FarKernelBatch::CATMARK_RESTRICTED_VERT_VERTEX_B2:
            controller->ApplyCatmarkRestrictedVertexVerticesKernelB2(batch, context);
            break;
        case FarKernelBatch::CATMARK_RESTRICTED_VERT_VERTEX_A:
            controller->ApplyCatmarkRestrictedVertexVerticesKernelA(batch, context);
            break;

        case FarKernelBatch::LOOP_EDGE_VERTEX:
            controller->ApplyLoopEdgeVerticesKernel(batch, context);
            break;
        case FarKernelBatch::LOOP_VERT_VERTEX_B:
            controller->ApplyLoopVertexVerticesKernelB(batch, context);
            break;
        case FarKernelBatch::LOOP_VERT_VERTEX_A1:
            controller->ApplyLoopVertexVerticesKernelA1(batch, context);
            break;
        case FarKernelBatch::LOOP_VERT_VERTEX_A2:
            controller->ApplyLoopVertexVerticesKernelA2(batch, context);
            break;

        case FarKernelBatch::BILINEAR_FACE_VERTEX:
            controller->ApplyBilinearFaceVerticesKernel(batch, context);
            break;
        case FarKernelBatch::BILINEAR_EDGE_VERTEX:
            controller->ApplyBilinearEdgeVerticesKernel(batch, context);
            break;
        case FarKernelBatch::BILINEAR_VERT_VERTEX:
            controller->ApplyBilinearVertexVerticesKernel(batch, context);
            break;

        case FarKernelBatch::HIERARCHICAL_EDIT:
            controller->ApplyVertexEdits(batch, context);
            break;

        default: // user defined kernel type
            return false;
    }

    return true;
}

template <class CONTROLLER, class CONTEXT> void
FarDispatcher::Refine(CONTROLLER *controller, CONTEXT *context, FarKernelBatchVector const & batches, int maxlevel) {

    for (int i = 0; i < (int)batches.size(); ++i) {
        const FarKernelBatch &batch = batches[i];

        if (maxlevel >= 0 && batch.GetLevel() >= maxlevel) continue;

        ApplyKernel(controller, context, batch);
    }
}

// -----------------------------------------------------------------------------

/// \brief Far default controller implementation
///
/// This is Far's default implementation of a kernal batch controller.
///
class FarComputeController {

public:
    template <class CONTEXT>
    void Refine(CONTEXT * context, int maxlevel=-1) const;

    template <class CONTEXT>
    void ApplyBilinearFaceVerticesKernel(FarKernelBatch const &batch, CONTEXT *context) const;

    template <class CONTEXT>
    void ApplyBilinearEdgeVerticesKernel(FarKernelBatch const &batch, CONTEXT *context) const;

    template <class CONTEXT>
    void ApplyBilinearVertexVerticesKernel(FarKernelBatch const &batch, CONTEXT *context) const;

    template <class CONTEXT>
    void ApplyCatmarkFaceVerticesKernel(FarKernelBatch const &batch, CONTEXT *context) const;

    template <class CONTEXT>
    void ApplyCatmarkQuadFaceVerticesKernel(FarKernelBatch const &batch, CONTEXT *context) const;

    template <class CONTEXT>
    void ApplyCatmarkTriQuadFaceVerticesKernel(FarKernelBatch const &batch, CONTEXT *context) const;

    template <class CONTEXT>
    void ApplyCatmarkEdgeVerticesKernel(FarKernelBatch const &batch, CONTEXT *context) const;

    template <class CONTEXT>
    void ApplyCatmarkRestrictedEdgeVerticesKernel(FarKernelBatch const &batch, CONTEXT *context) const;

    template <class CONTEXT>
    void ApplyCatmarkVertexVerticesKernelB(FarKernelBatch const &batch, CONTEXT *context) const;

    template <class CONTEXT>
    void ApplyCatmarkVertexVerticesKernelA1(FarKernelBatch const &batch, CONTEXT *context) const;

    template <class CONTEXT>
    void ApplyCatmarkVertexVerticesKernelA2(FarKernelBatch const &batch, CONTEXT *context) const;

    template <class CONTEXT>
    void ApplyCatmarkRestrictedVertexVerticesKernelB1(FarKernelBatch const &batch, CONTEXT *context) const;

    template <class CONTEXT>
    void ApplyCatmarkRestrictedVertexVerticesKernelB2(FarKernelBatch const &batch, CONTEXT *context) const;

    template <class CONTEXT>
    void ApplyCatmarkRestrictedVertexVerticesKernelA(FarKernelBatch const &batch, CONTEXT *context) const;

    template <class CONTEXT>
    void ApplyLoopEdgeVerticesKernel(FarKernelBatch const &batch, CONTEXT *context) const;

    template <class CONTEXT>
    void ApplyLoopVertexVerticesKernelB(FarKernelBatch const &batch, CONTEXT *context) const;

    template <class CONTEXT>
    void ApplyLoopVertexVerticesKernelA1(FarKernelBatch const &batch, CONTEXT *context) const;

    template <class CONTEXT>
    void ApplyLoopVertexVerticesKernelA2(FarKernelBatch const &batch, CONTEXT *context) const;

    template <class CONTEXT>
    void ApplyVertexEdits(FarKernelBatch const &batch, CONTEXT *context) const;

private:

};

template <class CONTEXT> void
FarComputeController::Refine(CONTEXT *context, int maxlevel) const {

    FarDispatcher::Refine(this, context, context->GetKernelBatches(), maxlevel);

}

template <class CONTEXT> void
FarComputeController::ApplyBilinearFaceVerticesKernel(FarKernelBatch const &batch, CONTEXT *context) const {

    typename CONTEXT::VertexType *vsrc = &context->GetVertices().at(0);

    FarSubdivisionTables const * subdivision = context->GetSubdivisionTables();
    assert(subdivision);

    subdivision->computeBilinearFacePoints( batch.GetVertexOffset(),
                                            batch.GetTableOffset(),
                                            batch.GetStart(),
                                            batch.GetEnd(),
                                            vsrc );
}

template <class CONTEXT> void
FarComputeController::ApplyBilinearEdgeVerticesKernel(FarKernelBatch const &batch, CONTEXT *context) const {

    typename CONTEXT::VertexType *vsrc = &context->GetVertices().at(0);

    FarSubdivisionTables const * subdivision = context->GetSubdivisionTables();
    assert(subdivision);

    subdivision->computeBilinearEdgePoints( batch.GetVertexOffset(),
                                            batch.GetTableOffset(),
                                            batch.GetStart(),
                                            batch.GetEnd(),
                                            vsrc );
}

template <class CONTEXT> void
FarComputeController::ApplyBilinearVertexVerticesKernel(FarKernelBatch const &batch, CONTEXT *context) const {

    typename CONTEXT::VertexType *vsrc = &context->GetVertices().at(0);

    FarSubdivisionTables const * subdivision = context->GetSubdivisionTables();
    assert(subdivision);

    subdivision->computeBilinearVertexPoints( batch.GetVertexOffset(),
                                              batch.GetTableOffset(),
                                              batch.GetStart(),
                                              batch.GetEnd(),
                                              vsrc );
}

template <class CONTEXT> void
FarComputeController::ApplyCatmarkFaceVerticesKernel(FarKernelBatch const &batch, CONTEXT *context) const {

    typename CONTEXT::VertexType *vsrc = &context->GetVertices().at(0);

    FarSubdivisionTables const * subdivision = context->GetSubdivisionTables();
    assert(subdivision);

    subdivision->computeCatmarkFacePoints( batch.GetVertexOffset(),
                                           batch.GetTableOffset(),
                                           batch.GetStart(),
                                           batch.GetEnd(),
                                           vsrc );
}

template <class CONTEXT> void
FarComputeController::ApplyCatmarkQuadFaceVerticesKernel(FarKernelBatch const &batch, CONTEXT *context) const {

    typename CONTEXT::VertexType *vsrc = &context->GetVertices().at(0);

    FarSubdivisionTables const * subdivision = context->GetSubdivisionTables();
    assert(subdivision);

    subdivision->computeCatmarkQuadFacePoints( batch.GetVertexOffset(),
                                               batch.GetTableOffset(),
                                               batch.GetStart(),
                                               batch.GetEnd(),
                                               vsrc );
}

template <class CONTEXT> void
FarComputeController::ApplyCatmarkTriQuadFaceVerticesKernel(FarKernelBatch const &batch, CONTEXT *context) const {

    typename CONTEXT::VertexType *vsrc = &context->GetVertices().at(0);

    FarSubdivisionTables const * subdivision = context->GetSubdivisionTables();
    assert(subdivision);

    subdivision->computeCatmarkTriQuadFacePoints( batch.GetVertexOffset(),
                                                  batch.GetTableOffset(),
                                                  batch.GetStart(),
                                                  batch.GetEnd(),
                                                  vsrc );
}

template <class CONTEXT> void
FarComputeController::ApplyCatmarkEdgeVerticesKernel(FarKernelBatch const &batch, CONTEXT *context) const {

    typename CONTEXT::VertexType *vsrc = &context->GetVertices().at(0);

    FarSubdivisionTables const * subdivision = context->GetSubdivisionTables();
    assert(subdivision);

    subdivision->computeCatmarkEdgePoints( batch.GetVertexOffset(),
                                           batch.GetTableOffset(),
                                           batch.GetStart(),
                                           batch.GetEnd(),
                                           vsrc );
}

template <class CONTEXT> void
FarComputeController::ApplyCatmarkRestrictedEdgeVerticesKernel(FarKernelBatch const &batch, CONTEXT *context) const {

    typename CONTEXT::VertexType *vsrc = &context->GetVertices().at(0);

    FarSubdivisionTables const * subdivision = context->GetSubdivisionTables();
    assert(subdivision);

    subdivision->computeCatmarkRestrictedEdgePoints( batch.GetVertexOffset(),
                                                     batch.GetTableOffset(),
                                                     batch.GetStart(),
                                                     batch.GetEnd(),
                                                     vsrc );
}

template <class CONTEXT> void
FarComputeController::ApplyCatmarkVertexVerticesKernelB(FarKernelBatch const &batch, CONTEXT *context) const {

    typename CONTEXT::VertexType *vsrc = &context->GetVertices().at(0);

    FarSubdivisionTables const * subdivision = context->GetSubdivisionTables();
    assert(subdivision);

    subdivision->computeCatmarkVertexPointsB( batch.GetVertexOffset(),
                                              batch.GetTableOffset(),
                                              batch.GetStart(),
                                              batch.GetEnd(),
                                              vsrc );
}

template <class CONTEXT> void
FarComputeController::ApplyCatmarkVertexVerticesKernelA1(FarKernelBatch const &batch, CONTEXT *context) const {

    typename CONTEXT::VertexType *vsrc = &context->GetVertices().at(0);

    FarSubdivisionTables const * subdivision = context->GetSubdivisionTables();
    assert(subdivision);

    subdivision->computeCatmarkVertexPointsA( batch.GetVertexOffset(),
                                              false,
                                              batch.GetTableOffset(),
                                              batch.GetStart(),
                                              batch.GetEnd(),
                                              vsrc );
}

template <class CONTEXT> void
FarComputeController::ApplyCatmarkVertexVerticesKernelA2(FarKernelBatch const &batch, CONTEXT *context) const {

    typename CONTEXT::VertexType *vsrc = &context->GetVertices().at(0);

    FarSubdivisionTables const * subdivision = context->GetSubdivisionTables();
    assert(subdivision);

    subdivision->computeCatmarkVertexPointsA( batch.GetVertexOffset(),
                                              true,
                                              batch.GetTableOffset(),
                                              batch.GetStart(),
                                              batch.GetEnd(),
                                              vsrc );
}

template <class CONTEXT> void
FarComputeController::ApplyCatmarkRestrictedVertexVerticesKernelB1(FarKernelBatch const &batch, CONTEXT *context) const {

    typename CONTEXT::VertexType *vsrc = &context->GetVertices().at(0);

    FarSubdivisionTables const * subdivision = context->GetSubdivisionTables();
    assert(subdivision);

    subdivision->computeCatmarkRestrictedVertexPointsB1( batch.GetVertexOffset(),
                                                         batch.GetTableOffset(),
                                                         batch.GetStart(),
                                                         batch.GetEnd(),
                                                         vsrc );
}

template <class CONTEXT> void
FarComputeController::ApplyCatmarkRestrictedVertexVerticesKernelB2(FarKernelBatch const &batch, CONTEXT *context) const {

    typename CONTEXT::VertexType *vsrc = &context->GetVertices().at(0);

    FarSubdivisionTables const * subdivision = context->GetSubdivisionTables();
    assert(subdivision);

    subdivision->computeCatmarkRestrictedVertexPointsB2( batch.GetVertexOffset(),
                                                         batch.GetTableOffset(),
                                                         batch.GetStart(),
                                                         batch.GetEnd(),
                                                         vsrc );
}

template <class CONTEXT> void
FarComputeController::ApplyCatmarkRestrictedVertexVerticesKernelA(FarKernelBatch const &batch, CONTEXT *context) const {

    typename CONTEXT::VertexType *vsrc = &context->GetVertices().at(0);

    FarSubdivisionTables const * subdivision = context->GetSubdivisionTables();
    assert(subdivision);

    subdivision->computeCatmarkRestrictedVertexPointsA( batch.GetVertexOffset(),
                                                         batch.GetTableOffset(),
                                                         batch.GetStart(),
                                                         batch.GetEnd(),
                                                         vsrc );
}

template <class CONTEXT> void
FarComputeController::ApplyLoopEdgeVerticesKernel(FarKernelBatch const &batch, CONTEXT *context) const {

    typename CONTEXT::VertexType *vsrc = &context->GetVertices().at(0);

    FarSubdivisionTables const * subdivision = context->GetSubdivisionTables();
    assert(subdivision);

    subdivision->computeLoopEdgePoints( batch.GetVertexOffset(),
                                        batch.GetTableOffset(),
                                        batch.GetStart(),
                                        batch.GetEnd(),
                                        vsrc );
}

template <class CONTEXT> void
FarComputeController::ApplyLoopVertexVerticesKernelB(FarKernelBatch const &batch, CONTEXT *context) const {

    typename CONTEXT::VertexType *vsrc = &context->GetVertices().at(0);

    FarSubdivisionTables const * subdivision = context->GetSubdivisionTables();
    assert(subdivision);

    subdivision->computeLoopVertexPointsB( batch.GetVertexOffset(),
                                           batch.GetTableOffset(),
                                           batch.GetStart(),
                                           batch.GetEnd(),
                                           vsrc );
}

template <class CONTEXT> void
FarComputeController::ApplyLoopVertexVerticesKernelA1(FarKernelBatch const &batch, CONTEXT *context) const {

    typename CONTEXT::VertexType *vsrc = &context->GetVertices().at(0);

    FarSubdivisionTables const * subdivision = context->GetSubdivisionTables();
    assert(subdivision);

    subdivision->computeLoopVertexPointsA( batch.GetVertexOffset(),
                                           false,
                                           batch.GetTableOffset(),
                                           batch.GetStart(),
                                           batch.GetEnd(),
                                           vsrc );
}

template <class CONTEXT> void
FarComputeController::ApplyLoopVertexVerticesKernelA2(FarKernelBatch const &batch, CONTEXT *context) const {

    typename CONTEXT::VertexType *vsrc = &context->GetVertices().at(0);

    FarSubdivisionTables const * subdivision = context->GetSubdivisionTables();
    assert(subdivision);

    subdivision->computeLoopVertexPointsA( batch.GetVertexOffset(),
                                           true,
                                           batch.GetTableOffset(),
                                           batch.GetStart(),
                                           batch.GetEnd(),
                                           vsrc );
}

template <class CONTEXT> void
FarComputeController::ApplyVertexEdits(FarKernelBatch const &batch, CONTEXT *context) const {

    typename CONTEXT::VertexType *vsrc = &context->GetVertices().at(0);

    FarVertexEditTables const * vertEdit = context->GetVertexEditTables();

    if (vertEdit)
        vertEdit->computeVertexEdits( batch.GetTableIndex(),
                                      batch.GetVertexOffset(),
                                      batch.GetTableOffset(),
                                      batch.GetStart(),
                                      batch.GetEnd(),
                                      vsrc );
}

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* FAR_DISPATCHER_H */
