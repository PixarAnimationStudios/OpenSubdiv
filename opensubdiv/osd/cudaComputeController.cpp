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

#include "../osd/cudaComputeContext.h"
#include "../osd/cudaComputeController.h"

#include <cuda_runtime.h>
#include <string.h>

extern "C" {

void OsdCudaComputeFace(float *vertex, float *varying,
                        int numUserVertexElements, int numVaryingElements,
                        int *F_IT, int *F_ITa, int offset, int tableOffset, int start, int end);

void OsdCudaComputeEdge(float *vertex, float *varying,
                        int numUserVertexElements, int numVaryingElements,
                        int *E_IT, float *E_W, int offset, int tableOffset, int start, int end);

void OsdCudaComputeVertexA(float *vertex, float *varying,
                           int numUserVertexElements, int numVaryingElements,
                           int *V_ITa, float *V_W, int offset, int tableOffset,
                           int start, int end, int pass);

void OsdCudaComputeVertexB(float *vertex, float *varying,
                           int numUserVertexElements, int numVaryingElements,
                           int *V_ITa, int *V_IT, float *V_W, int offset, int tableOffset,
                           int start, int end);

void OsdCudaComputeLoopVertexB(float *vertex, float *varying,
                               int numUserVertexElements,
                               int numVaryingElements,
                               int *V_ITa, int *V_IT, float *V_W, int offset, int tableOffset,
                               int start, int end);

void OsdCudaComputeBilinearEdge(float *vertex, float *varying,
                                int numUserVertexElements,
                                int numVaryingElements,
                                int *E_IT, int offset, int tableOffset, int start, int end);

void OsdCudaComputeBilinearVertex(float *vertex, float *varying,
                                  int numUserVertexElements,
                                  int numVaryingElements,
                                  int *V_ITa, int offset, int tableOffset, int start, int end);

void OsdCudaEditVertexAdd(float *vertex, int numUserVertexElements,
                          int primVarOffset, int primVarWidth,
                          int vertexOffset, int tableOffset,
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
    FarKernelBatch const &batch, void * clientdata) const {

    OsdCudaComputeContext * context =
        static_cast<OsdCudaComputeContext*>(clientdata);
    assert(context);

    const OsdCudaTable * F_IT = context->GetTable(FarSubdivisionTables<OsdVertex>::F_IT);
    const OsdCudaTable * F_ITa = context->GetTable(FarSubdivisionTables<OsdVertex>::F_ITa);
    assert(F_IT);
    assert(F_ITa);

    OsdCudaComputeFace(
        context->GetCurrentVertexBuffer(),
        context->GetCurrentVaryingBuffer(),
        context->GetVertexDescriptor().numVertexElements-3,
        context->GetVertexDescriptor().numVaryingElements,
        static_cast<int*>(F_IT->GetCudaMemory()),
        static_cast<int*>(F_ITa->GetCudaMemory()),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdCudaComputeController::ApplyBilinearEdgeVerticesKernel(
    FarKernelBatch const &batch, void * clientdata) const {

    OsdCudaComputeContext * context =
        static_cast<OsdCudaComputeContext*>(clientdata);
    assert(context);

    const OsdCudaTable * E_IT = context->GetTable(FarSubdivisionTables<OsdVertex>::E_IT);
    assert(E_IT);

    OsdCudaComputeBilinearEdge(
        context->GetCurrentVertexBuffer(),
        context->GetCurrentVaryingBuffer(),
        context->GetVertexDescriptor().numVertexElements-3,
        context->GetVertexDescriptor().numVaryingElements,
        static_cast<int*>(E_IT->GetCudaMemory()),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdCudaComputeController::ApplyBilinearVertexVerticesKernel(
    FarKernelBatch const &batch, void * clientdata) const {

    OsdCudaComputeContext * context =
        static_cast<OsdCudaComputeContext*>(clientdata);
    assert(context);

    const OsdCudaTable * V_ITa = context->GetTable(FarSubdivisionTables<OsdVertex>::V_ITa);
    assert(V_ITa);

    OsdCudaComputeBilinearVertex(
        context->GetCurrentVertexBuffer(),
        context->GetCurrentVaryingBuffer(),
        context->GetVertexDescriptor().numVertexElements-3,
        context->GetVertexDescriptor().numVaryingElements,
        static_cast<int*>(V_ITa->GetCudaMemory()),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdCudaComputeController::ApplyCatmarkFaceVerticesKernel(
    FarKernelBatch const &batch, void * clientdata) const {

    OsdCudaComputeContext * context =
        static_cast<OsdCudaComputeContext*>(clientdata);
    assert(context);

    const OsdCudaTable * F_IT = context->GetTable(FarSubdivisionTables<OsdVertex>::F_IT);
    const OsdCudaTable * F_ITa = context->GetTable(FarSubdivisionTables<OsdVertex>::F_ITa);
    assert(F_IT);
    assert(F_ITa);

    OsdCudaComputeFace(
        context->GetCurrentVertexBuffer(),
        context->GetCurrentVaryingBuffer(),
        context->GetVertexDescriptor().numVertexElements-3,
        context->GetVertexDescriptor().numVaryingElements,
        static_cast<int*>(F_IT->GetCudaMemory()),
        static_cast<int*>(F_ITa->GetCudaMemory()),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdCudaComputeController::ApplyCatmarkEdgeVerticesKernel(
    FarKernelBatch const &batch, void * clientdata) const {

    OsdCudaComputeContext * context =
        static_cast<OsdCudaComputeContext*>(clientdata);
    assert(context);

    const OsdCudaTable * E_IT = context->GetTable(FarSubdivisionTables<OsdVertex>::E_IT);
    const OsdCudaTable * E_W = context->GetTable(FarSubdivisionTables<OsdVertex>::E_W);
    assert(E_IT);
    assert(E_W);

    OsdCudaComputeEdge(
        context->GetCurrentVertexBuffer(),
        context->GetCurrentVaryingBuffer(),
        context->GetVertexDescriptor().numVertexElements-3,
        context->GetVertexDescriptor().numVaryingElements,
        static_cast<int*>(E_IT->GetCudaMemory()),
        static_cast<float*>(E_W->GetCudaMemory()),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdCudaComputeController::ApplyCatmarkVertexVerticesKernelB(
    FarKernelBatch const &batch, void * clientdata) const {

    OsdCudaComputeContext * context =
        static_cast<OsdCudaComputeContext*>(clientdata);
    assert(context);

    const OsdCudaTable * V_ITa = context->GetTable(FarSubdivisionTables<OsdVertex>::V_ITa);
    const OsdCudaTable * V_IT = context->GetTable(FarSubdivisionTables<OsdVertex>::V_IT);
    const OsdCudaTable * V_W = context->GetTable(FarSubdivisionTables<OsdVertex>::V_W);
    assert(V_ITa);
    assert(V_IT);
    assert(V_W);

    OsdCudaComputeVertexB(
        context->GetCurrentVertexBuffer(),
        context->GetCurrentVaryingBuffer(),
        context->GetVertexDescriptor().numVertexElements-3,
        context->GetVertexDescriptor().numVaryingElements,
        static_cast<int*>(V_ITa->GetCudaMemory()),
        static_cast<int*>(V_IT->GetCudaMemory()),
        static_cast<float*>(V_W->GetCudaMemory()),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdCudaComputeController::ApplyCatmarkVertexVerticesKernelA1(
    FarKernelBatch const &batch, void * clientdata) const {

    OsdCudaComputeContext * context =
        static_cast<OsdCudaComputeContext*>(clientdata);
    assert(context);

    const OsdCudaTable * V_ITa = context->GetTable(FarSubdivisionTables<OsdVertex>::V_ITa);
    const OsdCudaTable * V_W = context->GetTable(FarSubdivisionTables<OsdVertex>::V_W);
    assert(V_ITa);
    assert(V_W);

    OsdCudaComputeVertexA(
        context->GetCurrentVertexBuffer(),
        context->GetCurrentVaryingBuffer(),
        context->GetVertexDescriptor().numVertexElements-3,
        context->GetVertexDescriptor().numVaryingElements,
        static_cast<int*>(V_ITa->GetCudaMemory()),
        static_cast<float*>(V_W->GetCudaMemory()),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd(), false);
}

void
OsdCudaComputeController::ApplyCatmarkVertexVerticesKernelA2(
    FarKernelBatch const &batch, void * clientdata) const {

    OsdCudaComputeContext * context =
        static_cast<OsdCudaComputeContext*>(clientdata);
    assert(context);

    const OsdCudaTable * V_ITa = context->GetTable(FarSubdivisionTables<OsdVertex>::V_ITa);
    const OsdCudaTable * V_W = context->GetTable(FarSubdivisionTables<OsdVertex>::V_W);
    assert(V_ITa);
    assert(V_W);

    OsdCudaComputeVertexA(
        context->GetCurrentVertexBuffer(),
        context->GetCurrentVaryingBuffer(),
        context->GetVertexDescriptor().numVertexElements-3,
        context->GetVertexDescriptor().numVaryingElements,
        static_cast<int*>(V_ITa->GetCudaMemory()),
        static_cast<float*>(V_W->GetCudaMemory()),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd(), true);
}

void
OsdCudaComputeController::ApplyLoopEdgeVerticesKernel(
    FarKernelBatch const &batch, void * clientdata) const {

    OsdCudaComputeContext * context =
        static_cast<OsdCudaComputeContext*>(clientdata);
    assert(context);

    const OsdCudaTable * E_IT = context->GetTable(FarSubdivisionTables<OsdVertex>::E_IT);
    const OsdCudaTable * E_W = context->GetTable(FarSubdivisionTables<OsdVertex>::E_W);
    assert(E_IT);
    assert(E_W);

    OsdCudaComputeEdge(
        context->GetCurrentVertexBuffer(),
        context->GetCurrentVaryingBuffer(),
        context->GetVertexDescriptor().numVertexElements-3,
        context->GetVertexDescriptor().numVaryingElements,
        static_cast<int*>(E_IT->GetCudaMemory()),
        static_cast<float*>(E_W->GetCudaMemory()),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdCudaComputeController::ApplyLoopVertexVerticesKernelB(
    FarKernelBatch const &batch, void * clientdata) const {

    OsdCudaComputeContext * context =
        static_cast<OsdCudaComputeContext*>(clientdata);
    assert(context);

    const OsdCudaTable * V_ITa = context->GetTable(FarSubdivisionTables<OsdVertex>::V_ITa);
    const OsdCudaTable * V_IT = context->GetTable(FarSubdivisionTables<OsdVertex>::V_IT);
    const OsdCudaTable * V_W = context->GetTable(FarSubdivisionTables<OsdVertex>::V_W);
    assert(V_ITa);
    assert(V_IT);
    assert(V_W);

    OsdCudaComputeLoopVertexB(
        context->GetCurrentVertexBuffer(),
        context->GetCurrentVaryingBuffer(),
        context->GetVertexDescriptor().numVertexElements-3,
        context->GetVertexDescriptor().numVaryingElements,
        static_cast<int*>(V_ITa->GetCudaMemory()),
        static_cast<int*>(V_IT->GetCudaMemory()),
        static_cast<float*>(V_W->GetCudaMemory()),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdCudaComputeController::ApplyLoopVertexVerticesKernelA1(
    FarKernelBatch const &batch, void * clientdata) const {

    OsdCudaComputeContext * context =
        static_cast<OsdCudaComputeContext*>(clientdata);
    assert(context);

    const OsdCudaTable * V_ITa = context->GetTable(FarSubdivisionTables<OsdVertex>::V_ITa);
    const OsdCudaTable * V_W = context->GetTable(FarSubdivisionTables<OsdVertex>::V_W);
    assert(V_ITa);
    assert(V_W);

    OsdCudaComputeVertexA(
        context->GetCurrentVertexBuffer(),
        context->GetCurrentVaryingBuffer(),
        context->GetVertexDescriptor().numVertexElements-3,
        context->GetVertexDescriptor().numVaryingElements,
        static_cast<int*>(V_ITa->GetCudaMemory()),
        static_cast<float*>(V_W->GetCudaMemory()),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd(), false);
}

void
OsdCudaComputeController::ApplyLoopVertexVerticesKernelA2(
    FarKernelBatch const &batch, void * clientdata) const {

    OsdCudaComputeContext * context =
        static_cast<OsdCudaComputeContext*>(clientdata);
    assert(context);

    const OsdCudaTable * V_ITa = context->GetTable(FarSubdivisionTables<OsdVertex>::V_ITa);
    const OsdCudaTable * V_W = context->GetTable(FarSubdivisionTables<OsdVertex>::V_W);
    assert(V_ITa);
    assert(V_W);

    OsdCudaComputeVertexA(
        context->GetCurrentVertexBuffer(),
        context->GetCurrentVaryingBuffer(),
        context->GetVertexDescriptor().numVertexElements-3,
        context->GetVertexDescriptor().numVaryingElements,
        static_cast<int*>(V_ITa->GetCudaMemory()),
        static_cast<float*>(V_W->GetCudaMemory()),
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd(), true);
}

void
OsdCudaComputeController::ApplyVertexEdits(
    FarKernelBatch const &batch, void * clientdata) const {

    OsdCudaComputeContext * context =
        static_cast<OsdCudaComputeContext*>(clientdata);
    assert(context);

    const OsdCudaHEditTable *edit = context->GetEditTable(batch.GetTableIndex());
    assert(edit);

    const OsdCudaTable * primvarIndices = edit->GetPrimvarIndices();
    const OsdCudaTable * editValues = edit->GetEditValues();

    if (edit->GetOperation() == FarVertexEdit::Add) {
        OsdCudaEditVertexAdd(
            context->GetCurrentVertexBuffer(),
            context->GetVertexDescriptor().numVertexElements-3,
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
