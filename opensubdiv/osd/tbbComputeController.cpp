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

#include "../osd/cpuComputeContext.h"
#include "../osd/tbbComputeController.h"
#include "../osd/tbbKernel.h"
#include "../osd/table.h"

#ifdef OPENSUBDIV_HAS_TBB 
    #include <tbb/task_scheduler_init.h>
#endif

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {


OsdTbbComputeController::OsdTbbComputeController(int numThreads) {
    _numThreads = numThreads;
    if(_numThreads == -1)
        tbb::task_scheduler_init init();
    else
        tbb::task_scheduler_init init(numThreads);
}


void
OsdTbbComputeController::ApplyBilinearFaceVerticesKernel(
    FarKernelBatch const &batch, void *clientdata) const {

    OsdCpuComputeContext * context =
        static_cast<OsdCpuComputeContext*>(clientdata);
    assert(context);

    OsdTbbComputeFace(
        context->GetVertexDescriptor(),
        context->GetCurrentVertexBuffer(),
        context->GetCurrentVaryingBuffer(),
        (const int*)context->GetTable(Table::F_IT)->GetBuffer(),
        (const int*)context->GetTable(Table::F_ITa)->GetBuffer(),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdTbbComputeController::ApplyBilinearEdgeVerticesKernel(
    FarKernelBatch const &batch, void *clientdata) const {

    OsdCpuComputeContext * context =
        static_cast<OsdCpuComputeContext*>(clientdata);
    assert(context);

    OsdTbbComputeBilinearEdge(
        context->GetVertexDescriptor(),
        context->GetCurrentVertexBuffer(),
        context->GetCurrentVaryingBuffer(),
        (const int*)context->GetTable(Table::E_IT)->GetBuffer(),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdTbbComputeController::ApplyBilinearVertexVerticesKernel(
    FarKernelBatch const &batch, void *clientdata) const {

    OsdCpuComputeContext * context =
        static_cast<OsdCpuComputeContext*>(clientdata);
    assert(context);

    OsdTbbComputeBilinearVertex(
        context->GetVertexDescriptor(),
        context->GetCurrentVertexBuffer(),
        context->GetCurrentVaryingBuffer(),
        (const int*)context->GetTable(Table::V_ITa)->GetBuffer(),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdTbbComputeController::ApplyCatmarkFaceVerticesKernel(
    FarKernelBatch const &batch, void *clientdata) const {

    OsdCpuComputeContext * context =
        static_cast<OsdCpuComputeContext*>(clientdata);
    assert(context);

    OsdTbbComputeFace(
        context->GetVertexDescriptor(),
        context->GetCurrentVertexBuffer(),
        context->GetCurrentVaryingBuffer(),
        (const int*)context->GetTable(Table::F_IT)->GetBuffer(),
        (const int*)context->GetTable(Table::F_ITa)->GetBuffer(),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdTbbComputeController::ApplyCatmarkEdgeVerticesKernel(
    FarKernelBatch const &batch, void *clientdata) const {

    OsdCpuComputeContext * context =
        static_cast<OsdCpuComputeContext*>(clientdata);
    assert(context);

    OsdTbbComputeEdge(
        context->GetVertexDescriptor(),
        context->GetCurrentVertexBuffer(),
        context->GetCurrentVaryingBuffer(),
        (const int*)context->GetTable(Table::E_IT)->GetBuffer(),
        (const float*)context->GetTable(Table::E_W)->GetBuffer(),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdTbbComputeController::ApplyCatmarkVertexVerticesKernelB(
    FarKernelBatch const &batch, void *clientdata) const {

    OsdCpuComputeContext * context =
        static_cast<OsdCpuComputeContext*>(clientdata);
    assert(context);

    OsdTbbComputeVertexB(
        context->GetVertexDescriptor(),
        context->GetCurrentVertexBuffer(),
        context->GetCurrentVaryingBuffer(),
        (const int*)context->GetTable(Table::V_ITa)->GetBuffer(),
        (const int*)context->GetTable(Table::V_IT)->GetBuffer(),
        (const float*)context->GetTable(Table::V_W)->GetBuffer(),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdTbbComputeController::ApplyCatmarkVertexVerticesKernelA1(
    FarKernelBatch const &batch, void *clientdata) const {

    OsdCpuComputeContext * context =
        static_cast<OsdCpuComputeContext*>(clientdata);
    assert(context);

    OsdTbbComputeVertexA(
        context->GetVertexDescriptor(),
        context->GetCurrentVertexBuffer(),
        context->GetCurrentVaryingBuffer(),
        (const int*)context->GetTable(Table::V_ITa)->GetBuffer(),
        (const float*)context->GetTable(Table::V_W)->GetBuffer(),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd(), false);
}

void
OsdTbbComputeController::ApplyCatmarkVertexVerticesKernelA2(
    FarKernelBatch const &batch, void *clientdata) const {

    OsdCpuComputeContext * context =
        static_cast<OsdCpuComputeContext*>(clientdata);
    assert(context);

    OsdTbbComputeVertexA(
        context->GetVertexDescriptor(),
        context->GetCurrentVertexBuffer(),
        context->GetCurrentVaryingBuffer(),
        (const int*)context->GetTable(Table::V_ITa)->GetBuffer(),
        (const float*)context->GetTable(Table::V_W)->GetBuffer(),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd(), true);
}

void
OsdTbbComputeController::ApplyLoopEdgeVerticesKernel(
    FarKernelBatch const &batch, void *clientdata) const {

    OsdCpuComputeContext * context =
        static_cast<OsdCpuComputeContext*>(clientdata);
    assert(context);

    OsdTbbComputeEdge(
        context->GetVertexDescriptor(),
        context->GetCurrentVertexBuffer(),
        context->GetCurrentVaryingBuffer(),
        (const int*)context->GetTable(Table::E_IT)->GetBuffer(),
        (const float*)context->GetTable(Table::E_W)->GetBuffer(),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdTbbComputeController::ApplyLoopVertexVerticesKernelB(
    FarKernelBatch const &batch, void *clientdata) const {

    OsdCpuComputeContext * context =
        static_cast<OsdCpuComputeContext*>(clientdata);
    assert(context);

    OsdTbbComputeLoopVertexB(
        context->GetVertexDescriptor(),
        context->GetCurrentVertexBuffer(),
        context->GetCurrentVaryingBuffer(),
        (const int*)context->GetTable(Table::V_ITa)->GetBuffer(),
        (const int*)context->GetTable(Table::V_IT)->GetBuffer(),
        (const float*)context->GetTable(Table::V_W)->GetBuffer(),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdTbbComputeController::ApplyLoopVertexVerticesKernelA1(
    FarKernelBatch const &batch, void *clientdata) const {

    OsdCpuComputeContext * context =
        static_cast<OsdCpuComputeContext*>(clientdata);
    assert(context);

    OsdTbbComputeVertexA(
        context->GetVertexDescriptor(),
        context->GetCurrentVertexBuffer(),
        context->GetCurrentVaryingBuffer(),
        (const int*)context->GetTable(Table::V_ITa)->GetBuffer(),
        (const float*)context->GetTable(Table::V_W)->GetBuffer(),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd(), false);
}

void
OsdTbbComputeController::ApplyLoopVertexVerticesKernelA2(
    FarKernelBatch const &batch, void *clientdata) const {

    OsdCpuComputeContext * context =
        static_cast<OsdCpuComputeContext*>(clientdata);
    assert(context);

    OsdTbbComputeVertexA(
        context->GetVertexDescriptor(),
        context->GetCurrentVertexBuffer(),
        context->GetCurrentVaryingBuffer(),
        (const int*)context->GetTable(Table::V_ITa)->GetBuffer(),
        (const float*)context->GetTable(Table::V_W)->GetBuffer(),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd(), true);
}

void
OsdTbbComputeController::ApplyVertexEdits(
    FarKernelBatch const &batch, void * clientdata) const {

    OsdCpuComputeContext * context =
        static_cast<OsdCpuComputeContext*>(clientdata);
    assert(context);

    const OsdCpuHEditTable *edit = context->GetEditTable(batch.GetTableIndex());
    assert(edit);

    const OsdCpuTable * primvarIndices = edit->GetPrimvarIndices();
    const OsdCpuTable * editValues = edit->GetEditValues();

    if (edit->GetOperation() == FarVertexEdit::Add) {
        OsdTbbEditVertexAdd(context->GetVertexDescriptor(),
                            context->GetCurrentVertexBuffer(),
                            edit->GetPrimvarOffset(),
                            edit->GetPrimvarWidth(),
                            batch.GetVertexOffset(), 
                            batch.GetTableOffset(), 
                            batch.GetStart(), 
                            batch.GetEnd(),
                            static_cast<unsigned int*>(primvarIndices->GetBuffer()),
                            static_cast<float*>(editValues->GetBuffer()));
    } else if (edit->GetOperation() == FarVertexEdit::Set) {
        OsdTbbEditVertexSet(context->GetVertexDescriptor(),
                            context->GetCurrentVertexBuffer(),
                            edit->GetPrimvarOffset(),
                            edit->GetPrimvarWidth(),
                            batch.GetVertexOffset(), 
                            batch.GetTableOffset(), 
                            batch.GetStart(), 
                            batch.GetEnd(),
                            static_cast<unsigned int*>(primvarIndices->GetBuffer()),
                            static_cast<float*>(editValues->GetBuffer()));
    }
}

void
OsdTbbComputeController::Synchronize() {
    // XXX: 
}

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv

