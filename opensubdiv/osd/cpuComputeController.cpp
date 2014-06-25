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
#include "../osd/cpuComputeController.h"
#include "../osd/cpuKernel.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {


OsdCpuComputeController::OsdCpuComputeController() {
}

OsdCpuComputeController::~OsdCpuComputeController() {
}

void
OsdCpuComputeController::ApplyBilinearFaceVerticesKernel(
    FarKernelBatch const &batch, OsdCpuComputeContext const *context) const {

    assert(context);

    OsdCpuComputeFace(
        _currentBindState.vertexBuffer, _currentBindState.varyingBuffer,
        _currentBindState.vertexDesc, _currentBindState.varyingDesc,
        (const int*)context->GetTable(FarSubdivisionTables::F_IT)->GetBuffer(),
        (const int*)context->GetTable(FarSubdivisionTables::F_ITa)->GetBuffer(),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdCpuComputeController::ApplyBilinearEdgeVerticesKernel(
    FarKernelBatch const &batch, OsdCpuComputeContext const *context) const {

    assert(context);

    OsdCpuComputeBilinearEdge(
        _currentBindState.vertexBuffer, _currentBindState.varyingBuffer,
        _currentBindState.vertexDesc, _currentBindState.varyingDesc,
        (const int*)context->GetTable(FarSubdivisionTables::E_IT)->GetBuffer(),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdCpuComputeController::ApplyBilinearVertexVerticesKernel(
    FarKernelBatch const &batch, OsdCpuComputeContext const *context) const {

    assert(context);

    OsdCpuComputeBilinearVertex(
        _currentBindState.vertexBuffer, _currentBindState.varyingBuffer,
        _currentBindState.vertexDesc, _currentBindState.varyingDesc,
        (const int*)context->GetTable(FarSubdivisionTables::V_ITa)->GetBuffer(),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdCpuComputeController::ApplyCatmarkFaceVerticesKernel(
    FarKernelBatch const &batch, OsdCpuComputeContext const *context) const {

    assert(context);

    OsdCpuComputeFace(
        _currentBindState.vertexBuffer, _currentBindState.varyingBuffer,
        _currentBindState.vertexDesc, _currentBindState.varyingDesc,
        (const int*)context->GetTable(FarSubdivisionTables::F_IT)->GetBuffer(),
        (const int*)context->GetTable(FarSubdivisionTables::F_ITa)->GetBuffer(),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdCpuComputeController::ApplyCatmarkQuadFaceVerticesKernel(
    FarKernelBatch const &batch, OsdCpuComputeContext const *context) const {

    assert(context);

    OsdCpuComputeQuadFace(
        _currentBindState.vertexBuffer, _currentBindState.varyingBuffer,
        _currentBindState.vertexDesc, _currentBindState.varyingDesc,
        (const int*)context->GetTable(FarSubdivisionTables::F_IT)->GetBuffer(),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdCpuComputeController::ApplyCatmarkTriQuadFaceVerticesKernel(
    FarKernelBatch const &batch, OsdCpuComputeContext const *context) const {

    assert(context);

    OsdCpuComputeTriQuadFace(
        _currentBindState.vertexBuffer, _currentBindState.varyingBuffer,
        _currentBindState.vertexDesc, _currentBindState.varyingDesc,
        (const int*)context->GetTable(FarSubdivisionTables::F_IT)->GetBuffer(),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdCpuComputeController::ApplyCatmarkEdgeVerticesKernel(
    FarKernelBatch const &batch, OsdCpuComputeContext const *context) const {

    assert(context);

    OsdCpuComputeEdge(
        _currentBindState.vertexBuffer, _currentBindState.varyingBuffer,
        _currentBindState.vertexDesc, _currentBindState.varyingDesc,
        (const int*)context->GetTable(FarSubdivisionTables::E_IT)->GetBuffer(),
        (const float*)context->GetTable(FarSubdivisionTables::E_W)->GetBuffer(),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdCpuComputeController::ApplyCatmarkRestrictedEdgeVerticesKernel(
    FarKernelBatch const &batch, OsdCpuComputeContext const *context) const {

    assert(context);

    OsdCpuComputeRestrictedEdge(
        _currentBindState.vertexBuffer, _currentBindState.varyingBuffer,
        _currentBindState.vertexDesc, _currentBindState.varyingDesc,
        (const int*)context->GetTable(FarSubdivisionTables::E_IT)->GetBuffer(),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdCpuComputeController::ApplyCatmarkVertexVerticesKernelB(
    FarKernelBatch const &batch, OsdCpuComputeContext const *context) const {

    assert(context);

    OsdCpuComputeVertexB(
        _currentBindState.vertexBuffer, _currentBindState.varyingBuffer,
        _currentBindState.vertexDesc, _currentBindState.varyingDesc,
        (const int*)context->GetTable(FarSubdivisionTables::V_ITa)->GetBuffer(),
        (const int*)context->GetTable(FarSubdivisionTables::V_IT)->GetBuffer(),
        (const float*)context->GetTable(FarSubdivisionTables::V_W)->GetBuffer(),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdCpuComputeController::ApplyCatmarkVertexVerticesKernelA1(
    FarKernelBatch const &batch, OsdCpuComputeContext const *context) const {

    assert(context);

    OsdCpuComputeVertexA(
        _currentBindState.vertexBuffer, _currentBindState.varyingBuffer,
        _currentBindState.vertexDesc, _currentBindState.varyingDesc,
        (const int*)context->GetTable(FarSubdivisionTables::V_ITa)->GetBuffer(),
        (const float*)context->GetTable(FarSubdivisionTables::V_W)->GetBuffer(),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd(), false);
}

void
OsdCpuComputeController::ApplyCatmarkVertexVerticesKernelA2(
    FarKernelBatch const &batch, OsdCpuComputeContext const *context) const {

    assert(context);

    OsdCpuComputeVertexA(
        _currentBindState.vertexBuffer, _currentBindState.varyingBuffer,
        _currentBindState.vertexDesc, _currentBindState.varyingDesc,
        (const int*)context->GetTable(FarSubdivisionTables::V_ITa)->GetBuffer(),
        (const float*)context->GetTable(FarSubdivisionTables::V_W)->GetBuffer(),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd(), true);
}

void
OsdCpuComputeController::ApplyCatmarkRestrictedVertexVerticesKernelB1(
    FarKernelBatch const &batch, OsdCpuComputeContext const *context) const {

    assert(context);

    OsdCpuComputeRestrictedVertexB1(
        _currentBindState.vertexBuffer, _currentBindState.varyingBuffer,
        _currentBindState.vertexDesc, _currentBindState.varyingDesc,
        (const int*)context->GetTable(FarSubdivisionTables::V_ITa)->GetBuffer(),
        (const int*)context->GetTable(FarSubdivisionTables::V_IT)->GetBuffer(),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdCpuComputeController::ApplyCatmarkRestrictedVertexVerticesKernelB2(
    FarKernelBatch const &batch, OsdCpuComputeContext const *context) const {

    assert(context);

    OsdCpuComputeRestrictedVertexB2(
        _currentBindState.vertexBuffer, _currentBindState.varyingBuffer,
        _currentBindState.vertexDesc, _currentBindState.varyingDesc,
        (const int*)context->GetTable(FarSubdivisionTables::V_ITa)->GetBuffer(),
        (const int*)context->GetTable(FarSubdivisionTables::V_IT)->GetBuffer(),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdCpuComputeController::ApplyCatmarkRestrictedVertexVerticesKernelA(
    FarKernelBatch const &batch, OsdCpuComputeContext const *context) const {

    assert(context);

    OsdCpuComputeRestrictedVertexA(
        _currentBindState.vertexBuffer, _currentBindState.varyingBuffer,
        _currentBindState.vertexDesc, _currentBindState.varyingDesc,
        (const int*)context->GetTable(FarSubdivisionTables::V_ITa)->GetBuffer(),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdCpuComputeController::ApplyLoopEdgeVerticesKernel(
    FarKernelBatch const &batch, OsdCpuComputeContext const *context) const {

    assert(context);

    OsdCpuComputeEdge(
        _currentBindState.vertexBuffer, _currentBindState.varyingBuffer,
        _currentBindState.vertexDesc, _currentBindState.varyingDesc,
        (const int*)context->GetTable(FarSubdivisionTables::E_IT)->GetBuffer(),
        (const float*)context->GetTable(FarSubdivisionTables::E_W)->GetBuffer(),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdCpuComputeController::ApplyLoopVertexVerticesKernelB(
    FarKernelBatch const &batch, OsdCpuComputeContext const *context) const {

    assert(context);

    OsdCpuComputeLoopVertexB(
        _currentBindState.vertexBuffer, _currentBindState.varyingBuffer,
        _currentBindState.vertexDesc, _currentBindState.varyingDesc,
        (const int*)context->GetTable(FarSubdivisionTables::V_ITa)->GetBuffer(),
        (const int*)context->GetTable(FarSubdivisionTables::V_IT)->GetBuffer(),
        (const float*)context->GetTable(FarSubdivisionTables::V_W)->GetBuffer(),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdCpuComputeController::ApplyLoopVertexVerticesKernelA1(
    FarKernelBatch const &batch, OsdCpuComputeContext const *context) const {

    assert(context);

    OsdCpuComputeVertexA(
        _currentBindState.vertexBuffer, _currentBindState.varyingBuffer,
        _currentBindState.vertexDesc, _currentBindState.varyingDesc,
        (const int*)context->GetTable(FarSubdivisionTables::V_ITa)->GetBuffer(),
        (const float*)context->GetTable(FarSubdivisionTables::V_W)->GetBuffer(),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd(), false);
}

void
OsdCpuComputeController::ApplyLoopVertexVerticesKernelA2(
    FarKernelBatch const &batch, OsdCpuComputeContext const *context) const {

    assert(context);

    OsdCpuComputeVertexA(
        _currentBindState.vertexBuffer, _currentBindState.varyingBuffer,
        _currentBindState.vertexDesc, _currentBindState.varyingDesc,
        (const int*)context->GetTable(FarSubdivisionTables::V_ITa)->GetBuffer(),
        (const float*)context->GetTable(FarSubdivisionTables::V_W)->GetBuffer(),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd(), true);
}

void
OsdCpuComputeController::ApplyVertexEdits(
    FarKernelBatch const &batch, OsdCpuComputeContext const *context) const {

    assert(context);

    const OsdCpuHEditTable *edit = context->GetEditTable(batch.GetTableIndex());
    assert(edit);

    const OsdCpuTable * primvarIndices = edit->GetPrimvarIndices();
    const OsdCpuTable * editValues = edit->GetEditValues();

    if (edit->GetOperation() == FarVertexEdit::Add) {
        OsdCpuEditVertexAdd(_currentBindState.vertexBuffer,
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
        OsdCpuEditVertexSet(_currentBindState.vertexBuffer,
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
OsdCpuComputeController::Synchronize() {
}

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
