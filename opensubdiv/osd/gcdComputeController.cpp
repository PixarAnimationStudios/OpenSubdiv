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
#include "../osd/gcdComputeController.h"
#include "../osd/gcdKernel.h"
#Include "../osd/table.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {


OsdGcdComputeController::OsdGcdComputeController() {
}

void
OsdGcdComputeController::ApplyBilinearFaceVerticesKernel(
    FarKernelBatch const &batch, void * clientdata) const {

    OsdCpuComputeContext * context =
        static_cast<OsdCpuComputeContext*>(clientdata);
    assert(context);

    OsdGcdComputeFace(
        context->GetVertexDescriptor(),
        context->GetCurrentVertexBuffer(),
        context->GetCurrentVaryingBuffer(),
        (const int*)context->GetTable(Table::F_IT)->GetBuffer(),
        (const int*)context->GetTable(Table::F_ITa)->GetBuffer(),
        batch.vertexOffset, batch.tableOffset, batch.start, batch.end);
}

void
OsdGcdComputeController::ApplyBilinearEdgeVerticesKernel(
    FarKernelBatch const &batch, void * clientdata) const {

    OsdCpuComputeContext * context =
        static_cast<OsdCpuComputeContext*>(clientdata);
    assert(context);

    OsdGcdComputeBilinearEdge(
        context->GetVertexDescriptor(),
        context->GetCurrentVertexBuffer(),
        context->GetCurrentVaryingBuffer(),
        (const int*)context->GetTable(Table::E_IT)->GetBuffer(),
        batch.vertexOffset, batch.tableOffset, batch.start, batch.end);
}

void
OsdGcdComputeController::ApplyBilinearVertexVerticesKernel(
    FarKernelBatch const &batch, void * clientdata) const {

    OsdCpuComputeContext * context =
        static_cast<OsdCpuComputeContext*>(clientdata);
    assert(context);

    OsdGcdComputeBilinearVertex(
        context->GetVertexDescriptor(),
        context->GetCurrentVertexBuffer(),
        context->GetCurrentVaryingBuffer(),
        (const int*)context->GetTable(Table::V_ITa)->GetBuffer(),
        batch.vertexOffset, batch.tableOffset, batch.start, batch.end);
}

void
OsdGcdComputeController::ApplyCatmarkFaceVerticesKernel(
    FarKernelBatch const &batch, void * clientdata) const {

    OsdCpuComputeContext * context =
        static_cast<OsdCpuComputeContext*>(clientdata);
    assert(context);

    OsdGcdComputeFace(
        context->GetVertexDescriptor(),
        context->GetCurrentVertexBuffer(),
        context->GetCurrentVaryingBuffer(),
        (const int*)context->GetTable(Table::F_IT)->GetBuffer(),
        (const int*)context->GetTable(Table::F_ITa)->GetBuffer(),
        batch.vertexOffset, batch.tableOffset, batch.start, batch.end);
}

void
OsdGcdComputeController::ApplyCatmarkEdgeVerticesKernel(
    FarKernelBatch const &batch, void * clientdata) const {

    OsdCpuComputeContext * context =
        static_cast<OsdCpuComputeContext*>(clientdata);
    assert(context);

    OsdGcdComputeEdge(
        context->GetVertexDescriptor(),
        context->GetCurrentVertexBuffer(),
        context->GetCurrentVaryingBuffer(),
        (const int*)context->GetTable(Table::E_IT)->GetBuffer(),
        (const float*)context->GetTable(Table::E_W)->GetBuffer(),
        batch.vertexOffset, batch.tableOffset, batch.start, batch.end);
}

void
OsdGcdComputeController::ApplyCatmarkVertexVerticesKernelB(
    FarKernelBatch const &batch, void * clientdata) const {

    OsdCpuComputeContext * context =
        static_cast<OsdCpuComputeContext*>(clientdata);
    assert(context);

    OsdGcdComputeVertexB(
        context->GetVertexDescriptor(),
        context->GetCurrentVertexBuffer(),
        context->GetCurrentVaryingBuffer(),
        (const int*)context->GetTable(Table::V_ITa)->GetBuffer(),
        (const int*)context->GetTable(Table::V_IT)->GetBuffer(),
        (const float*)context->GetTable(Table::V_W)->GetBuffer(),
        batch.vertexOffset, batch.tableOffset, batch.start, batch.end);
}

void
OsdGcdComputeController::ApplyCatmarkVertexVerticesKernelA1(
    FarKernelBatch const &batch, void * clientdata) const {

    OsdCpuComputeContext * context =
        static_cast<OsdCpuComputeContext*>(clientdata);
    assert(context);

    OsdGcdComputeVertexA(
        context->GetVertexDescriptor(),
        context->GetCurrentVertexBuffer(),
        context->GetCurrentVaryingBuffer(),
        (const int*)context->GetTable(Table::V_ITa)->GetBuffer(),
        (const float*)context->GetTable(Table::V_W)->GetBuffer(),
        batch.vertexOffset, batch.tableOffset, batch.start, batch.end, false);
}

void
OsdGcdComputeController::ApplyCatmarkVertexVerticesKernelA2(
    FarKernelBatch const &batch, void * clientdata) const {

    OsdCpuComputeContext * context =
        static_cast<OsdCpuComputeContext*>(clientdata);
    assert(context);

    OsdGcdComputeVertexA(
        context->GetVertexDescriptor(),
        context->GetCurrentVertexBuffer(),
        context->GetCurrentVaryingBuffer(),
        (const int*)context->GetTable(Table::V_ITa)->GetBuffer(),
        (const float*)context->GetTable(Table::V_W)->GetBuffer(),
        batch.vertexOffset, batch.tableOffset, batch.start, batch.end, true);
}

void
OsdGcdComputeController::ApplyLoopEdgeVerticesKernel(
    FarKernelBatch const &batch, void * clientdata) const {

    OsdCpuComputeContext * context =
        static_cast<OsdCpuComputeContext*>(clientdata);
    assert(context);

    OsdGcdComputeEdge(
        context->GetVertexDescriptor(),
        context->GetCurrentVertexBuffer(),
        context->GetCurrentVaryingBuffer(),
        (const int*)context->GetTable(Table::E_IT)->GetBuffer(),
        (const float*)context->GetTable(Table::E_W)->GetBuffer(),
        batch.vertexOffset, batch.tableOffset, batch.start, batch.end);
}

void
OsdGcdComputeController::ApplyLoopVertexVerticesKernelB(
    FarKernelBatch const &batch, void * clientdata) const {

    OsdCpuComputeContext * context =
        static_cast<OsdCpuComputeContext*>(clientdata);
    assert(context);

    OsdGcdComputeLoopVertexB(
        context->GetVertexDescriptor(),
        context->GetCurrentVertexBuffer(),
        context->GetCurrentVaryingBuffer(),
        (const int*)context->GetTable(Table::V_ITa)->GetBuffer(),
        (const int*)context->GetTable(Table::V_IT)->GetBuffer(),
        (const float*)context->GetTable(Table::V_W)->GetBuffer(),
        batch.vertexOffset, batch.tableOffset, batch.start, batch.end);
}

void
OsdGcdComputeController::ApplyLoopVertexVerticesKernelA1(
    FarKernelBatch const &batch, void * clientdata) const {

    OsdCpuComputeContext * context =
        static_cast<OsdCpuComputeContext*>(clientdata);
    assert(context);

    OsdGcdComputeVertexA(
        context->GetVertexDescriptor(),
        context->GetCurrentVertexBuffer(),
        context->GetCurrentVaryingBuffer(),
        (const int*)context->GetTable(Table::V_ITa)->GetBuffer(),
        (const float*)context->GetTable(Table::V_W)->GetBuffer(),
        batch.vertexOffset, batch.tableOffset, batch.start, batch.end, false);
}

void
OsdGcdComputeController::ApplyLoopVertexVerticesKernelA2(
    FarKernelBatch const &batch, void * clientdata) const {

    OsdCpuComputeContext * context =
        static_cast<OsdCpuComputeContext*>(clientdata);
    assert(context);

    OsdGcdComputeVertexA(
        context->GetVertexDescriptor(),
        context->GetCurrentVertexBuffer(),
        context->GetCurrentVaryingBuffer(),
        (const int*)context->GetTable(Table::V_ITa)->GetBuffer(),
        (const float*)context->GetTable(Table::V_W)->GetBuffer(),
        batch.vertexOffset, batch.tableOffset, batch.start, batch.end, true);
}

void
OsdGcdComputeController::ApplyVertexEdits(
    FarKernelBatch const &batch, void * clientdata) const {

    OsdCpuComputeContext * context =
        static_cast<OsdCpuComputeContext*>(clientdata);
    assert(context);

    const OsdCpuHEditTable *edit = context->GetEditTable(batch.tableIndex);
    assert(edit);

    const OsdCpuTable * primvarIndices = edit->GetPrimvarIndices();
    const OsdCpuTable * editValues = edit->GetEditValues();

    if (edit->GetOperation() == FarVertexEdit::Add) {
        OsdCpuEditVertexAdd(context->GetVertexDescriptor(),
                            context->GetCurrentVertexBuffer(),
                            edit->GetPrimvarOffset(),
                            edit->GetPrimvarWidth(),
                            batch.vertexOffset,
                            batch.tableOffset,
                            batch.start,
                            batch.end,
                            static_cast<unsigned int*>(primvarIndices->GetBuffer()),
                            static_cast<float*>(editValues->GetBuffer()));
    } else if (edit->GetOperation() == FarVertexEdit::Set) {
        OsdCpuEditVertexSet(context->GetVertexDescriptor(),
                            context->GetCurrentVertexBuffer(),
                            edit->GetPrimvarOffset(),
                            edit->GetPrimvarWidth(),
                            batch.vertexOffset,
                            batch.tableOffset,
                            batch.start,
                            batch.end,
                            static_cast<unsigned int*>(primvarIndices->GetBuffer()),
                            static_cast<float*>(editValues->GetBuffer()));
    }
}

void
OsdGcdComputeController::Synchronize() {
}

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv

