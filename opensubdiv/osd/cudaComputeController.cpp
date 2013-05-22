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
