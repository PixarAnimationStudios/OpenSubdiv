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

#ifndef FAR_KERNELBATCH_DISPATCHER_H
#define FAR_KERNELBATCH_DISPATCHER_H

#include "../version.h"

#include "../far/kernelBatch.h"
#include "../far/stencilTables.h"

#include <cassert>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Far {

/// \brief Subdivision refinement encapsulation layer.
///
/// The kernel dispatcher allows client code to customize parts or the entire
/// computation process. This pattern aims at hiding the logic specific to
/// the subdivision algorithms and expose a simplified access to minimalistic
/// compute kernels. By default, meshes revert to a default dispatcher that
/// implements single-threaded CPU kernels.
///
/// - derive a dispatcher class from this one
/// - override the virtual functions
/// - pass the derived dispatcher to the factory (one instance can be shared by many meshes)
///
/// Note : the caller is responsible for deleting a custom dispatcher
///
class KernelBatchDispatcher {
public:

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
    template <class CONTROLLER, class CONTEXT> static void Apply(
        CONTROLLER *controller, CONTEXT *context, KernelBatchVector const & batches, int maxlevel);

protected:

    /// \brief Launches the processing of a kernel batch
    /// returns true if the batch is handled, otherwise returns false (i.e. user defined kernel)
    ///
    /// @param controller  refinement controller implementation
    ///
    /// @param context     refinement context implementation
    ///
    /// @param batch       a batch of kernel that need to be processed
    ///
    template <class CONTROLLER, class CONTEXT> static bool ApplyKernel(
        CONTROLLER *controller, CONTEXT *context, KernelBatch const &batch);

};

///
/// \brief Far default controller implementation
///
/// This is Far's default implementation of a kernal batch controller.
///
class DefaultController {
public:

    template <class CONTEXT> void ApplyStencilTableKernel(
        KernelBatch const &batch, CONTEXT *context) const;

};


// Launches the processing of a kernel batch
template <class CONTROLLER, class CONTEXT> bool
KernelBatchDispatcher::ApplyKernel(CONTROLLER *controller, CONTEXT *context,
    KernelBatch const &batch) {

    if (batch.end==0) {
        return true;
    }

    switch(batch.kernelType) {

        case KernelBatch::KERNEL_UNKNOWN:
            assert(0);

        case KernelBatch::KERNEL_STENCIL_TABLE:
            controller->ApplyStencilTableKernel(batch, context);
            break;

        default: // user defined kernel type
            return false;
    }

    return true;
}

// Launches the processing of a vector of kernel batches
template <class CONTROLLER, class CONTEXT> void
KernelBatchDispatcher::Apply(CONTROLLER *controller, CONTEXT *context,
    KernelBatchVector const & batches, int maxlevel) {

    for (int i = 0; i < (int)batches.size(); ++i) {

        const KernelBatch &batch = batches[i];

        if (maxlevel>=0 and batch.level>=maxlevel) {
            continue;
        }

        ApplyKernel(controller, context, batch);
    }
}

template <class CONTEXT> void
DefaultController::ApplyStencilTableKernel(
    KernelBatch const &batch, CONTEXT *context) const {

    StencilTables const * stencilTables = context->GetStencilTables();
    assert(stencilTables);

    typename CONTEXT::VertexType *vsrc = &context->GetVertices().at(0),
                                 *vdst = vsrc + batch.start + stencilTables->GetNumControlVertices();

    stencilTables->UpdateValues(vsrc, vdst, batch.start, batch.end);
}

} // end namespace Far

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* FAR_KERNELBATCH_DISPATCHER_H */
