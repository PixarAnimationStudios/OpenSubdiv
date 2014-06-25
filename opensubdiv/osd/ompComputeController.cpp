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

#include "../osd/cpuComputeContext.h"
#include "../osd/ompComputeController.h"
#include "../osd/ompKernel.h"

#ifdef OPENSUBDIV_HAS_OPENMP
    #include <omp.h>
#endif

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {


OsdOmpComputeController::OsdOmpComputeController(int numThreads) {

    _numThreads = (numThreads == -1) ? omp_get_max_threads() : numThreads;
}


void
OsdOmpComputeController::ApplyBilinearFaceVerticesKernel(
    FarKernelBatch const &batch, OsdCpuComputeContext const *context) const {

    assert(context);

    OsdOmpComputeFace(
        _currentBindState.vertexBuffer, _currentBindState.varyingBuffer,
        _currentBindState.vertexDesc, _currentBindState.varyingDesc,
        (const int*)context->GetTable(FarSubdivisionTables::F_IT)->GetBuffer(),
        (const int*)context->GetTable(FarSubdivisionTables::F_ITa)->GetBuffer(),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdOmpComputeController::ApplyBilinearEdgeVerticesKernel(
    FarKernelBatch const &batch, OsdCpuComputeContext const *context) const {

    assert(context);

    OsdOmpComputeBilinearEdge(
        _currentBindState.vertexBuffer, _currentBindState.varyingBuffer,
        _currentBindState.vertexDesc, _currentBindState.varyingDesc,
        (const int*)context->GetTable(FarSubdivisionTables::E_IT)->GetBuffer(),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdOmpComputeController::ApplyBilinearVertexVerticesKernel(
    FarKernelBatch const &batch, OsdCpuComputeContext const *context) const {

    assert(context);

    OsdOmpComputeBilinearVertex(
        _currentBindState.vertexBuffer, _currentBindState.varyingBuffer,
        _currentBindState.vertexDesc, _currentBindState.varyingDesc,
        (const int*)context->GetTable(FarSubdivisionTables::V_ITa)->GetBuffer(),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdOmpComputeController::ApplyCatmarkFaceVerticesKernel(
    FarKernelBatch const &batch, OsdCpuComputeContext const *context) const {

    assert(context);

    OsdOmpComputeFace(
        _currentBindState.vertexBuffer, _currentBindState.varyingBuffer,
        _currentBindState.vertexDesc, _currentBindState.varyingDesc,
        (const int*)context->GetTable(FarSubdivisionTables::F_IT)->GetBuffer(),
        (const int*)context->GetTable(FarSubdivisionTables::F_ITa)->GetBuffer(),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdOmpComputeController::ApplyCatmarkQuadFaceVerticesKernel(
    FarKernelBatch const &batch, OsdCpuComputeContext const *context) const {

    assert(context);

    OsdOmpComputeQuadFace(
        _currentBindState.vertexBuffer, _currentBindState.varyingBuffer,
        _currentBindState.vertexDesc, _currentBindState.varyingDesc,
        (const int*)context->GetTable(FarSubdivisionTables::F_IT)->GetBuffer(),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdOmpComputeController::ApplyCatmarkTriQuadFaceVerticesKernel(
    FarKernelBatch const &batch, OsdCpuComputeContext const *context) const {

    assert(context);

    OsdOmpComputeTriQuadFace(
        _currentBindState.vertexBuffer, _currentBindState.varyingBuffer,
        _currentBindState.vertexDesc, _currentBindState.varyingDesc,
        (const int*)context->GetTable(FarSubdivisionTables::F_IT)->GetBuffer(),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdOmpComputeController::ApplyCatmarkEdgeVerticesKernel(
    FarKernelBatch const &batch, OsdCpuComputeContext const *context) const {

    assert(context);

    OsdOmpComputeEdge(
        _currentBindState.vertexBuffer, _currentBindState.varyingBuffer,
        _currentBindState.vertexDesc, _currentBindState.varyingDesc,
        (const int*)context->GetTable(FarSubdivisionTables::E_IT)->GetBuffer(),
        (const float*)context->GetTable(FarSubdivisionTables::E_W)->GetBuffer(),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdOmpComputeController::ApplyCatmarkRestrictedEdgeVerticesKernel(
    FarKernelBatch const &batch, OsdCpuComputeContext const *context) const {

    assert(context);

    OsdOmpComputeRestrictedEdge(
        _currentBindState.vertexBuffer, _currentBindState.varyingBuffer,
        _currentBindState.vertexDesc, _currentBindState.varyingDesc,
        (const int*)context->GetTable(FarSubdivisionTables::E_IT)->GetBuffer(),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdOmpComputeController::ApplyCatmarkVertexVerticesKernelB(
    FarKernelBatch const &batch, OsdCpuComputeContext const *context) const {

    assert(context);

    OsdOmpComputeVertexB(
        _currentBindState.vertexBuffer, _currentBindState.varyingBuffer,
        _currentBindState.vertexDesc, _currentBindState.varyingDesc,
        (const int*)context->GetTable(FarSubdivisionTables::V_ITa)->GetBuffer(),
        (const int*)context->GetTable(FarSubdivisionTables::V_IT)->GetBuffer(),
        (const float*)context->GetTable(FarSubdivisionTables::V_W)->GetBuffer(),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdOmpComputeController::ApplyCatmarkVertexVerticesKernelA1(
    FarKernelBatch const &batch, OsdCpuComputeContext const *context) const {

    assert(context);

    OsdOmpComputeVertexA(
        _currentBindState.vertexBuffer, _currentBindState.varyingBuffer,
        _currentBindState.vertexDesc, _currentBindState.varyingDesc,
        (const int*)context->GetTable(FarSubdivisionTables::V_ITa)->GetBuffer(),
        (const float*)context->GetTable(FarSubdivisionTables::V_W)->GetBuffer(),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd(), false);
}

void
OsdOmpComputeController::ApplyCatmarkVertexVerticesKernelA2(
    FarKernelBatch const &batch, OsdCpuComputeContext const *context) const {

    assert(context);

    OsdOmpComputeVertexA(
        _currentBindState.vertexBuffer, _currentBindState.varyingBuffer,
        _currentBindState.vertexDesc, _currentBindState.varyingDesc,
        (const int*)context->GetTable(FarSubdivisionTables::V_ITa)->GetBuffer(),
        (const float*)context->GetTable(FarSubdivisionTables::V_W)->GetBuffer(),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd(), true);
}

void
OsdOmpComputeController::ApplyCatmarkRestrictedVertexVerticesKernelB1(
    FarKernelBatch const &batch, OsdCpuComputeContext const *context) const {

    assert(context);

    OsdOmpComputeRestrictedVertexB1(
        _currentBindState.vertexBuffer, _currentBindState.varyingBuffer,
        _currentBindState.vertexDesc, _currentBindState.varyingDesc,
        (const int*)context->GetTable(FarSubdivisionTables::V_ITa)->GetBuffer(),
        (const int*)context->GetTable(FarSubdivisionTables::V_IT)->GetBuffer(),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdOmpComputeController::ApplyCatmarkRestrictedVertexVerticesKernelB2(
    FarKernelBatch const &batch, OsdCpuComputeContext const *context) const {

    assert(context);

    OsdOmpComputeRestrictedVertexB2(
        _currentBindState.vertexBuffer, _currentBindState.varyingBuffer,
        _currentBindState.vertexDesc, _currentBindState.varyingDesc,
        (const int*)context->GetTable(FarSubdivisionTables::V_ITa)->GetBuffer(),
        (const int*)context->GetTable(FarSubdivisionTables::V_IT)->GetBuffer(),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdOmpComputeController::ApplyCatmarkRestrictedVertexVerticesKernelA(
    FarKernelBatch const &batch, OsdCpuComputeContext const *context) const {

    assert(context);

    OsdOmpComputeRestrictedVertexA(
        _currentBindState.vertexBuffer, _currentBindState.varyingBuffer,
        _currentBindState.vertexDesc, _currentBindState.varyingDesc,
        (const int*)context->GetTable(FarSubdivisionTables::V_ITa)->GetBuffer(),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdOmpComputeController::ApplyLoopEdgeVerticesKernel(
    FarKernelBatch const &batch, OsdCpuComputeContext const *context) const {

    assert(context);

    OsdOmpComputeEdge(
        _currentBindState.vertexBuffer, _currentBindState.varyingBuffer,
        _currentBindState.vertexDesc, _currentBindState.varyingDesc,
        (const int*)context->GetTable(FarSubdivisionTables::E_IT)->GetBuffer(),
        (const float*)context->GetTable(FarSubdivisionTables::E_W)->GetBuffer(),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdOmpComputeController::ApplyLoopVertexVerticesKernelB(
    FarKernelBatch const &batch, OsdCpuComputeContext const *context) const {

    assert(context);

    OsdOmpComputeLoopVertexB(
        _currentBindState.vertexBuffer, _currentBindState.varyingBuffer,
        _currentBindState.vertexDesc, _currentBindState.varyingDesc,
        (const int*)context->GetTable(FarSubdivisionTables::V_ITa)->GetBuffer(),
        (const int*)context->GetTable(FarSubdivisionTables::V_IT)->GetBuffer(),
        (const float*)context->GetTable(FarSubdivisionTables::V_W)->GetBuffer(),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdOmpComputeController::ApplyLoopVertexVerticesKernelA1(
    FarKernelBatch const &batch, OsdCpuComputeContext const *context) const {

    assert(context);

    OsdOmpComputeVertexA(
        _currentBindState.vertexBuffer, _currentBindState.varyingBuffer,
        _currentBindState.vertexDesc, _currentBindState.varyingDesc,
        (const int*)context->GetTable(FarSubdivisionTables::V_ITa)->GetBuffer(),
        (const float*)context->GetTable(FarSubdivisionTables::V_W)->GetBuffer(),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd(), false);
}

void
OsdOmpComputeController::ApplyLoopVertexVerticesKernelA2(
    FarKernelBatch const &batch, OsdCpuComputeContext const *context) const {

    assert(context);

    OsdOmpComputeVertexA(
        _currentBindState.vertexBuffer, _currentBindState.varyingBuffer,
        _currentBindState.vertexDesc, _currentBindState.varyingDesc,
        (const int*)context->GetTable(FarSubdivisionTables::V_ITa)->GetBuffer(),
        (const float*)context->GetTable(FarSubdivisionTables::V_W)->GetBuffer(),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd(), true);
}

void
OsdOmpComputeController::ApplyVertexEdits(
    FarKernelBatch const &batch, OsdCpuComputeContext const *context) const {

    assert(context);

    const OsdCpuHEditTable *edit = context->GetEditTable(batch.GetTableIndex());
    assert(edit);

    const OsdCpuTable * primvarIndices = edit->GetPrimvarIndices();
    const OsdCpuTable * editValues = edit->GetEditValues();

    if (edit->GetOperation() == FarVertexEdit::Add) {
        OsdOmpEditVertexAdd(_currentBindState.vertexBuffer,
                            _currentBindState.vertexDesc,
                            edit->GetPrimvarOffset(),
                            edit->GetPrimvarWidth(),
                            batch.GetVertexOffset(), 
                            batch.GetTableOffset(), 
                            batch.GetStart(), 
                            batch.GetEnd(),
                            static_cast<unsigned int*>(primvarIndices->GetBuffer()),
                            static_cast<float*>(editValues->GetBuffer()));
    } else if (edit->GetOperation() == FarVertexEdit::Set) {
        OsdOmpEditVertexSet(_currentBindState.vertexBuffer,
                            _currentBindState.vertexDesc,
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
OsdOmpComputeController::Synchronize() {
    // XXX: 
}

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv

