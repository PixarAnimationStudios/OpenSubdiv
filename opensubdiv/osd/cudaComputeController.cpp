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

#include "../osd/cudaComputeContext.h"
#include "../osd/cudaComputeController.h"

#include <cuda_runtime.h>
#include <string.h>

extern "C" {

void OsdCudaComputeFace(float *vertex, float *varying,
                        int vertexLength, int vertexStride,
                        int varyingLength, int varyingStride,
                        int *F_IT, int *F_ITa, int offset, int tableOffset, int start, int end);

void OsdCudaComputeQuadFace(float *vertex, float *varying,
                            int vertexLength, int vertexStride,
                            int varyingLength, int varyingStride,
                            int *F_IT, int offset, int tableOffset, int start, int end);

void OsdCudaComputeTriQuadFace(float *vertex, float *varying,
                               int vertexLength, int vertexStride,
                               int varyingLength, int varyingStride,
                               int *F_IT, int offset, int tableOffset, int start, int end);

void OsdCudaComputeEdge(float *vertex, float *varying,
                        int vertexLength, int vertexStride,
                        int varyingLength, int varyingStride,
                        int *E_IT, float *E_W, int offset, int tableOffset, int start, int end);

void OsdCudaComputeRestrictedEdge(float *vertex, float *varying,
                                  int vertexLength, int vertexStride,
                                  int varyingLength, int varyingStride,
                                  int *E_IT, int offset, int tableOffset, int start, int end);

void OsdCudaComputeVertexA(float *vertex, float *varying,
                           int vertexLength, int vertexStride,
                           int varyingLength, int varyingStride,
                           int *V_ITa, float *V_W, int offset, int tableOffset,
                           int start, int end, int pass);

void OsdCudaComputeVertexB(float *vertex, float *varying,
                           int vertexLength, int vertexStride,
                           int varyingLength, int varyingStride,
                           int *V_ITa, int *V_IT, float *V_W, int offset, int tableOffset,
                           int start, int end);

void OsdCudaComputeRestrictedVertexA(float *vertex, float *varying,
                                     int vertexLength, int vertexStride,
                                     int varyingLength, int varyingStride,
                                     int *V_ITa, int offset, int tableOffset,
                                     int start, int end);

void OsdCudaComputeRestrictedVertexB1(float *vertex, float *varying,
                                      int vertexLength, int vertexStride,
                                      int varyingLength, int varyingStride,
                                      int *V_ITa, int *V_IT, int offset, int tableOffset,
                                      int start, int end);

void OsdCudaComputeRestrictedVertexB2(float *vertex, float *varying,
                                      int vertexLength, int vertexStride,
                                      int varyingLength, int varyingStride,
                                      int *V_ITa, int *V_IT, int offset, int tableOffset,
                                      int start, int end);

void OsdCudaComputeLoopVertexB(float *vertex, float *varying,
                               int vertexLength, int vertexStride,
                               int varyingLength, int varyingStride,
                               int *V_ITa, int *V_IT, float *V_W, int offset, int tableOffset,
                               int start, int end);

void OsdCudaComputeBilinearEdge(float *vertex, float *varying,
                                int vertexLength, int vertexStride,
                                int varyingLength, int varyingStride,
                                int *E_IT, int offset, int tableOffset, int start, int end);

void OsdCudaComputeBilinearVertex(float *vertex, float *varying,
                                  int vertexLength, int vertexStride,
                                  int varyingLength, int varyingStride,
                                  int *V_ITa, int offset, int tableOffset, int start, int end);

void OsdCudaEditVertexAdd(float *vertex,
                          int vertexLength, int vertexStride,
                          int primVarOffset, int primVarWidth,
                          int offset, int tableOffset,
                          int start, int end, int *editIndices, float *editValues);

}

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

OsdCudaComputeController::OsdCudaComputeController() {
}

OsdCudaComputeController::~OsdCudaComputeController() {
}

void
OsdCudaComputeController::ApplyBilinearFaceVerticesKernel(
    FarKernelBatch const &batch, OsdCudaComputeContext const *context) const {

    assert(context);

    const OsdCudaTable * F_IT = context->GetTable(FarSubdivisionTables::F_IT);
    const OsdCudaTable * F_ITa = context->GetTable(FarSubdivisionTables::F_ITa);
    assert(F_IT);
    assert(F_ITa);

    float *vertex = _currentBindState.GetOffsettedVertexBuffer();
    float *varying = _currentBindState.GetOffsettedVaryingBuffer();

    OsdCudaComputeFace(
        vertex, varying,
        _currentBindState.vertexDesc.length, _currentBindState.vertexDesc.stride,
        _currentBindState.varyingDesc.length, _currentBindState.varyingDesc.stride,
        static_cast<int*>(F_IT->GetCudaMemory()),
        static_cast<int*>(F_ITa->GetCudaMemory()),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdCudaComputeController::ApplyBilinearEdgeVerticesKernel(
    FarKernelBatch const &batch, OsdCudaComputeContext const *context) const {

    assert(context);

    const OsdCudaTable * E_IT = context->GetTable(FarSubdivisionTables::E_IT);
    assert(E_IT);

    float *vertex = _currentBindState.GetOffsettedVertexBuffer();
    float *varying = _currentBindState.GetOffsettedVaryingBuffer();

    OsdCudaComputeBilinearEdge(
        vertex, varying,
        _currentBindState.vertexDesc.length, _currentBindState.vertexDesc.stride,
        _currentBindState.varyingDesc.length, _currentBindState.varyingDesc.stride,
        static_cast<int*>(E_IT->GetCudaMemory()),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdCudaComputeController::ApplyBilinearVertexVerticesKernel(
    FarKernelBatch const &batch, OsdCudaComputeContext const *context) const {

    assert(context);

    const OsdCudaTable * V_ITa = context->GetTable(FarSubdivisionTables::V_ITa);
    assert(V_ITa);

    float *vertex = _currentBindState.GetOffsettedVertexBuffer();
    float *varying = _currentBindState.GetOffsettedVaryingBuffer();

    OsdCudaComputeBilinearVertex(
        vertex, varying,
        _currentBindState.vertexDesc.length, _currentBindState.vertexDesc.stride,
        _currentBindState.varyingDesc.length, _currentBindState.varyingDesc.stride,
        static_cast<int*>(V_ITa->GetCudaMemory()),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdCudaComputeController::ApplyCatmarkFaceVerticesKernel(
    FarKernelBatch const &batch, OsdCudaComputeContext const *context) const {

    assert(context);

    const OsdCudaTable * F_IT = context->GetTable(FarSubdivisionTables::F_IT);
    const OsdCudaTable * F_ITa = context->GetTable(FarSubdivisionTables::F_ITa);
    assert(F_IT);
    assert(F_ITa);

    float *vertex = _currentBindState.GetOffsettedVertexBuffer();
    float *varying = _currentBindState.GetOffsettedVaryingBuffer();

    OsdCudaComputeFace(
        vertex, varying,
        _currentBindState.vertexDesc.length, _currentBindState.vertexDesc.stride,
        _currentBindState.varyingDesc.length, _currentBindState.varyingDesc.stride,
        static_cast<int*>(F_IT->GetCudaMemory()),
        static_cast<int*>(F_ITa->GetCudaMemory()),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdCudaComputeController::ApplyCatmarkQuadFaceVerticesKernel(
    FarKernelBatch const &batch, OsdCudaComputeContext const *context) const {

    assert(context);

    const OsdCudaTable * F_IT = context->GetTable(FarSubdivisionTables::F_IT);
    assert(F_IT);

    float *vertex = _currentBindState.GetOffsettedVertexBuffer();
    float *varying = _currentBindState.GetOffsettedVaryingBuffer();

    OsdCudaComputeQuadFace(
        vertex, varying,
        _currentBindState.vertexDesc.length, _currentBindState.vertexDesc.stride,
        _currentBindState.varyingDesc.length, _currentBindState.varyingDesc.stride,
        static_cast<int*>(F_IT->GetCudaMemory()),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdCudaComputeController::ApplyCatmarkTriQuadFaceVerticesKernel(
    FarKernelBatch const &batch, OsdCudaComputeContext const *context) const {

    assert(context);

    const OsdCudaTable * F_IT = context->GetTable(FarSubdivisionTables::F_IT);
    assert(F_IT);

    float *vertex = _currentBindState.GetOffsettedVertexBuffer();
    float *varying = _currentBindState.GetOffsettedVaryingBuffer();

    OsdCudaComputeTriQuadFace(
        vertex, varying,
        _currentBindState.vertexDesc.length, _currentBindState.vertexDesc.stride,
        _currentBindState.varyingDesc.length, _currentBindState.varyingDesc.stride,
        static_cast<int*>(F_IT->GetCudaMemory()),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdCudaComputeController::ApplyCatmarkEdgeVerticesKernel(
    FarKernelBatch const &batch, OsdCudaComputeContext const *context) const {

    assert(context);

    const OsdCudaTable * E_IT = context->GetTable(FarSubdivisionTables::E_IT);
    const OsdCudaTable * E_W = context->GetTable(FarSubdivisionTables::E_W);
    assert(E_IT);
    assert(E_W);

    float *vertex = _currentBindState.GetOffsettedVertexBuffer();
    float *varying = _currentBindState.GetOffsettedVaryingBuffer();

    OsdCudaComputeEdge(
        vertex, varying,
        _currentBindState.vertexDesc.length, _currentBindState.vertexDesc.stride,
        _currentBindState.varyingDesc.length, _currentBindState.varyingDesc.stride,
        static_cast<int*>(E_IT->GetCudaMemory()),
        static_cast<float*>(E_W->GetCudaMemory()),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdCudaComputeController::ApplyCatmarkRestrictedEdgeVerticesKernel(
    FarKernelBatch const &batch, OsdCudaComputeContext const *context) const {

    assert(context);

    const OsdCudaTable * E_IT = context->GetTable(FarSubdivisionTables::E_IT);
    assert(E_IT);

    float *vertex = _currentBindState.GetOffsettedVertexBuffer();
    float *varying = _currentBindState.GetOffsettedVaryingBuffer();

    OsdCudaComputeRestrictedEdge(
        vertex, varying,
        _currentBindState.vertexDesc.length, _currentBindState.vertexDesc.stride,
        _currentBindState.varyingDesc.length, _currentBindState.varyingDesc.stride,
        static_cast<int*>(E_IT->GetCudaMemory()),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdCudaComputeController::ApplyCatmarkVertexVerticesKernelB(
    FarKernelBatch const &batch, OsdCudaComputeContext const *context) const {

    assert(context);

    const OsdCudaTable * V_ITa = context->GetTable(FarSubdivisionTables::V_ITa);
    const OsdCudaTable * V_IT = context->GetTable(FarSubdivisionTables::V_IT);
    const OsdCudaTable * V_W = context->GetTable(FarSubdivisionTables::V_W);
    assert(V_ITa);
    assert(V_IT);
    assert(V_W);

    float *vertex = _currentBindState.GetOffsettedVertexBuffer();
    float *varying = _currentBindState.GetOffsettedVaryingBuffer();

    OsdCudaComputeVertexB(
        vertex, varying,
        _currentBindState.vertexDesc.length, _currentBindState.vertexDesc.stride,
        _currentBindState.varyingDesc.length, _currentBindState.varyingDesc.stride,
        static_cast<int*>(V_ITa->GetCudaMemory()),
        static_cast<int*>(V_IT->GetCudaMemory()),
        static_cast<float*>(V_W->GetCudaMemory()),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdCudaComputeController::ApplyCatmarkVertexVerticesKernelA1(
    FarKernelBatch const &batch, OsdCudaComputeContext const *context) const {

    assert(context);

    const OsdCudaTable * V_ITa = context->GetTable(FarSubdivisionTables::V_ITa);
    const OsdCudaTable * V_W = context->GetTable(FarSubdivisionTables::V_W);
    assert(V_ITa);
    assert(V_W);

    float *vertex = _currentBindState.GetOffsettedVertexBuffer();
    float *varying = _currentBindState.GetOffsettedVaryingBuffer();

    OsdCudaComputeVertexA(
        vertex, varying,
        _currentBindState.vertexDesc.length, _currentBindState.vertexDesc.stride,
        _currentBindState.varyingDesc.length, _currentBindState.varyingDesc.stride,
        static_cast<int*>(V_ITa->GetCudaMemory()),
        static_cast<float*>(V_W->GetCudaMemory()),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd(), false);
}

void
OsdCudaComputeController::ApplyCatmarkVertexVerticesKernelA2(
    FarKernelBatch const &batch, OsdCudaComputeContext const *context) const {

    assert(context);

    const OsdCudaTable * V_ITa = context->GetTable(FarSubdivisionTables::V_ITa);
    const OsdCudaTable * V_W = context->GetTable(FarSubdivisionTables::V_W);
    assert(V_ITa);
    assert(V_W);

    float *vertex = _currentBindState.GetOffsettedVertexBuffer();
    float *varying = _currentBindState.GetOffsettedVaryingBuffer();

    OsdCudaComputeVertexA(
        vertex, varying,
        _currentBindState.vertexDesc.length, _currentBindState.vertexDesc.stride,
        _currentBindState.varyingDesc.length, _currentBindState.varyingDesc.stride,
        static_cast<int*>(V_ITa->GetCudaMemory()),
        static_cast<float*>(V_W->GetCudaMemory()),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd(), true);
}

void
OsdCudaComputeController::ApplyCatmarkRestrictedVertexVerticesKernelB1(
    FarKernelBatch const &batch, OsdCudaComputeContext const *context) const {

    assert(context);

    const OsdCudaTable * V_ITa = context->GetTable(FarSubdivisionTables::V_ITa);
    const OsdCudaTable * V_IT = context->GetTable(FarSubdivisionTables::V_IT);
    assert(V_ITa);
    assert(V_IT);

    float *vertex = _currentBindState.GetOffsettedVertexBuffer();
    float *varying = _currentBindState.GetOffsettedVaryingBuffer();

    OsdCudaComputeRestrictedVertexB1(
        vertex, varying,
        _currentBindState.vertexDesc.length, _currentBindState.vertexDesc.stride,
        _currentBindState.varyingDesc.length, _currentBindState.varyingDesc.stride,
        static_cast<int*>(V_ITa->GetCudaMemory()),
        static_cast<int*>(V_IT->GetCudaMemory()),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdCudaComputeController::ApplyCatmarkRestrictedVertexVerticesKernelB2(
    FarKernelBatch const &batch, OsdCudaComputeContext const *context) const {

    assert(context);

    const OsdCudaTable * V_ITa = context->GetTable(FarSubdivisionTables::V_ITa);
    const OsdCudaTable * V_IT = context->GetTable(FarSubdivisionTables::V_IT);
    assert(V_ITa);
    assert(V_IT);

    float *vertex = _currentBindState.GetOffsettedVertexBuffer();
    float *varying = _currentBindState.GetOffsettedVaryingBuffer();

    OsdCudaComputeRestrictedVertexB2(
        vertex, varying,
        _currentBindState.vertexDesc.length, _currentBindState.vertexDesc.stride,
        _currentBindState.varyingDesc.length, _currentBindState.varyingDesc.stride,
        static_cast<int*>(V_ITa->GetCudaMemory()),
        static_cast<int*>(V_IT->GetCudaMemory()),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdCudaComputeController::ApplyCatmarkRestrictedVertexVerticesKernelA(
    FarKernelBatch const &batch, OsdCudaComputeContext const *context) const {

    assert(context);

    const OsdCudaTable * V_ITa = context->GetTable(FarSubdivisionTables::V_ITa);
    assert(V_ITa);

    float *vertex = _currentBindState.GetOffsettedVertexBuffer();
    float *varying = _currentBindState.GetOffsettedVaryingBuffer();

    OsdCudaComputeRestrictedVertexA(
        vertex, varying,
        _currentBindState.vertexDesc.length, _currentBindState.vertexDesc.stride,
        _currentBindState.varyingDesc.length, _currentBindState.varyingDesc.stride,
        static_cast<int*>(V_ITa->GetCudaMemory()),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdCudaComputeController::ApplyLoopEdgeVerticesKernel(
    FarKernelBatch const &batch, OsdCudaComputeContext const *context) const {

    assert(context);

    const OsdCudaTable * E_IT = context->GetTable(FarSubdivisionTables::E_IT);
    const OsdCudaTable * E_W = context->GetTable(FarSubdivisionTables::E_W);
    assert(E_IT);
    assert(E_W);

    float *vertex = _currentBindState.GetOffsettedVertexBuffer();
    float *varying = _currentBindState.GetOffsettedVaryingBuffer();

    OsdCudaComputeEdge(
        vertex, varying,
        _currentBindState.vertexDesc.length, _currentBindState.vertexDesc.stride,
        _currentBindState.varyingDesc.length, _currentBindState.varyingDesc.stride,
        static_cast<int*>(E_IT->GetCudaMemory()),
        static_cast<float*>(E_W->GetCudaMemory()),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdCudaComputeController::ApplyLoopVertexVerticesKernelB(
    FarKernelBatch const &batch, OsdCudaComputeContext const *context) const {

    assert(context);

    const OsdCudaTable * V_ITa = context->GetTable(FarSubdivisionTables::V_ITa);
    const OsdCudaTable * V_IT = context->GetTable(FarSubdivisionTables::V_IT);
    const OsdCudaTable * V_W = context->GetTable(FarSubdivisionTables::V_W);
    assert(V_ITa);
    assert(V_IT);
    assert(V_W);

    float *vertex = _currentBindState.GetOffsettedVertexBuffer();
    float *varying = _currentBindState.GetOffsettedVaryingBuffer();

    OsdCudaComputeLoopVertexB(
        vertex, varying,
        _currentBindState.vertexDesc.length, _currentBindState.vertexDesc.stride,
        _currentBindState.varyingDesc.length, _currentBindState.varyingDesc.stride,
        static_cast<int*>(V_ITa->GetCudaMemory()),
        static_cast<int*>(V_IT->GetCudaMemory()),
        static_cast<float*>(V_W->GetCudaMemory()),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdCudaComputeController::ApplyLoopVertexVerticesKernelA1(
    FarKernelBatch const &batch, OsdCudaComputeContext const *context) const {

    assert(context);

    const OsdCudaTable * V_ITa = context->GetTable(FarSubdivisionTables::V_ITa);
    const OsdCudaTable * V_W = context->GetTable(FarSubdivisionTables::V_W);
    assert(V_ITa);
    assert(V_W);

    float *vertex = _currentBindState.GetOffsettedVertexBuffer();
    float *varying = _currentBindState.GetOffsettedVaryingBuffer();

    OsdCudaComputeVertexA(
        vertex, varying,
        _currentBindState.vertexDesc.length, _currentBindState.vertexDesc.stride,
        _currentBindState.varyingDesc.length, _currentBindState.varyingDesc.stride,
        static_cast<int*>(V_ITa->GetCudaMemory()),
        static_cast<float*>(V_W->GetCudaMemory()),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd(), false);
}

void
OsdCudaComputeController::ApplyLoopVertexVerticesKernelA2(
    FarKernelBatch const &batch, OsdCudaComputeContext const *context) const {

    assert(context);

    const OsdCudaTable * V_ITa = context->GetTable(FarSubdivisionTables::V_ITa);
    const OsdCudaTable * V_W = context->GetTable(FarSubdivisionTables::V_W);
    assert(V_ITa);
    assert(V_W);

    float *vertex = _currentBindState.GetOffsettedVertexBuffer();
    float *varying = _currentBindState.GetOffsettedVaryingBuffer();

    OsdCudaComputeVertexA(
        vertex, varying,
        _currentBindState.vertexDesc.length, _currentBindState.vertexDesc.stride,
        _currentBindState.varyingDesc.length, _currentBindState.varyingDesc.stride,
        static_cast<int*>(V_ITa->GetCudaMemory()),
        static_cast<float*>(V_W->GetCudaMemory()),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd(), true);
}

void
OsdCudaComputeController::ApplyVertexEdits(
    FarKernelBatch const &batch, OsdCudaComputeContext const *context) const {

    assert(context);

    const OsdCudaHEditTable *edit = context->GetEditTable(batch.GetTableIndex());
    assert(edit);

    const OsdCudaTable * primvarIndices = edit->GetPrimvarIndices();
    const OsdCudaTable * editValues = edit->GetEditValues();

    float *vertex = _currentBindState.GetOffsettedVertexBuffer();

    if (edit->GetOperation() == FarVertexEdit::Add) {
        OsdCudaEditVertexAdd(
            vertex,
            _currentBindState.vertexDesc.length, _currentBindState.vertexDesc.stride,
            edit->GetPrimvarOffset(),
            edit->GetPrimvarWidth(),
            batch.GetVertexOffset(),
            batch.GetTableOffset(),
            batch.GetStart(),
            batch.GetEnd(),
            static_cast<int*>(primvarIndices->GetCudaMemory()),
            static_cast<float*>(editValues->GetCudaMemory()));
    } else if (edit->GetOperation() == FarVertexEdit::Set) {
        // XXXX TODO
    }
}

void
OsdCudaComputeController::Synchronize() {

    cudaThreadSynchronize();
}

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv
