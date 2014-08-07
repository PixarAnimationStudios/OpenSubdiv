// Copyright 2014 Google Inc. All rights reserved.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "../osd/cpuKernel.h"
#include "../osd/neonComputeContext.h"
#include "../osd/neonComputeController.h"
#include "../osd/neonKernel.h"

#include <cassert>

#define USE_ASM_KERNELS 1

#if USE_ASM_KERNELS
extern "C" {

void OsdNeonComputeQuadFace_ASM(float *vertex, const int *F_IT,
    int vertexOffset, int tableOffset, int batchSize);

void OsdNeonComputeTriQuadFace_ASM(float *vertex, const int *F_IT,
    int vertexOffset, int tableOffset, int batchSize);

void OsdNeonComputeRestrictedEdge_ASM(float *vertex, const int *E_IT,
    int vertexOffset, int tableOffset, int batchSize);

void OsdNeonComputeRestrictedVertexB1_ASM(float *vertex, const int *V_ITa,
    const int *V_IT, int vertexOffset, int tableOffset, int start, int end);

void OsdNeonComputeRestrictedVertexB2_ASM(float *vertex, const int *V_ITa,
    const int *V_IT, int vertexOffset, int tableOffset, int start, int end);

void OsdNeonComputeRestrictedVertexA_ASM(float *vertex, const int *V_ITa,
    int vertexOffset, int tableOffset, int start, int end);

}
#endif // #if USE_ASM_KERNELS

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

OsdNeonComputeController::OsdNeonComputeController()
{
}

OsdNeonComputeController::~OsdNeonComputeController()
{
}

void OsdNeonComputeController::ApplyCatmarkQuadFaceVerticesKernel(
    FarKernelBatch const &batch, OsdNeonComputeContext const *context) const
{
    assert(context);

    if (neonKernelsSupported()) {
        assert(batch.GetStart() == 0);

        const int* F_IT = static_cast<const int*>(
            context->GetTable(FarSubdivisionTables::F_IT)->GetBuffer());
#if USE_ASM_KERNELS
        OsdNeonComputeQuadFace_ASM(getVertexBuffer(), F_IT,
            batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetEnd());
#else
        OsdNeonComputeQuadFace(getVertexBuffer(), F_IT,
            batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetEnd());
#endif // #if USE_ASM_KERNELS
    } else {
        OsdCpuComputeController::ApplyCatmarkQuadFaceVerticesKernel(batch,
            context);
    }
}

void OsdNeonComputeController::ApplyCatmarkTriQuadFaceVerticesKernel(
    FarKernelBatch const &batch, OsdNeonComputeContext const *context) const
{
    assert(context);

    if (neonKernelsSupported()) {
        assert(batch.GetStart() == 0);

        const int* F_IT = static_cast<const int*>(
            context->GetTable(FarSubdivisionTables::F_IT)->GetBuffer());
#if USE_ASM_KERNELS
        OsdNeonComputeTriQuadFace_ASM(getVertexBuffer(), F_IT,
            batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetEnd());
#else
        OsdNeonComputeTriQuadFace(getVertexBuffer(), F_IT,
            batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetEnd());
#endif // #if USE_ASM_KERNELS
    } else {
        OsdCpuComputeController::ApplyCatmarkTriQuadFaceVerticesKernel(batch,
            context);
    }
}

void OsdNeonComputeController::ApplyCatmarkRestrictedEdgeVerticesKernel(
    FarKernelBatch const &batch, OsdNeonComputeContext const *context) const
{
    assert(context);

    if (neonKernelsSupported()) {
        assert(batch.GetStart() == 0);

        const int* E_IT = static_cast<const int*>(
            context->GetTable(FarSubdivisionTables::E_IT)->GetBuffer());
#if USE_ASM_KERNELS
        OsdNeonComputeRestrictedEdge_ASM(getVertexBuffer(), E_IT,
            batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetEnd());
#else
        OsdNeonComputeRestrictedEdge(getVertexBuffer(), E_IT,
            batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetEnd());
#endif // #if USE_ASM_KERNELS
    } else {
        OsdCpuComputeController::ApplyCatmarkRestrictedEdgeVerticesKernel(batch,
            context);
    }
}

void OsdNeonComputeController::ApplyCatmarkRestrictedVertexVerticesKernelB1(
    FarKernelBatch const &batch, OsdNeonComputeContext const *context) const
{
    assert(context);

    if (neonKernelsSupported()) {
        const int* V_ITa = static_cast<const int*>(
            context->GetTable(FarSubdivisionTables::V_ITa)->GetBuffer());
        const int* V_IT = static_cast<const int*>(
            context->GetTable(FarSubdivisionTables::V_IT)->GetBuffer());
#if USE_ASM_KERNELS
        OsdNeonComputeRestrictedVertexB1_ASM(getVertexBuffer(), V_ITa, V_IT,
            batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(),
            batch.GetEnd());
#else
        OsdNeonComputeRestrictedVertexB1(getVertexBuffer(), V_ITa, V_IT,
            batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(),
            batch.GetEnd());
#endif // #if USE_ASM_KERNELS
    } else {
        OsdCpuComputeController::ApplyCatmarkRestrictedVertexVerticesKernelB1(
            batch, context);
    }
}

void OsdNeonComputeController::ApplyCatmarkRestrictedVertexVerticesKernelB2(
    FarKernelBatch const &batch, OsdNeonComputeContext const *context) const
{
    assert(context);

    if (neonKernelsSupported() && context->GetMaxVertexValence() <= 10) {
        const int* V_ITa = static_cast<const int*>(
            context->GetTable(FarSubdivisionTables::V_ITa)->GetBuffer());
        const int* V_IT = static_cast<const int*>(
            context->GetTable(FarSubdivisionTables::V_IT)->GetBuffer());
#if USE_ASM_KERNELS
        OsdNeonComputeRestrictedVertexB2_ASM(getVertexBuffer(), V_ITa, V_IT,
            batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(),
            batch.GetEnd());
#else
        OsdNeonComputeRestrictedVertexB2(getVertexBuffer(), V_ITa, V_IT,
            batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(),
            batch.GetEnd());
#endif // #if USE_ASM_KERNELS
    } else {
        OsdCpuComputeController::ApplyCatmarkRestrictedVertexVerticesKernelB2(
            batch, context);
    }
}

void OsdNeonComputeController::ApplyCatmarkRestrictedVertexVerticesKernelA(
    FarKernelBatch const &batch, OsdNeonComputeContext const *context) const
{
    assert(context);

    if (neonKernelsSupported()) {
        const int* V_ITa = static_cast<const int*>(
            context->GetTable(FarSubdivisionTables::V_ITa)->GetBuffer());
#if USE_ASM_KERNELS
        OsdNeonComputeRestrictedVertexA_ASM(getVertexBuffer(), V_ITa,
            batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(),
            batch.GetEnd());
#else
        OsdNeonComputeRestrictedVertexA(getVertexBuffer(), V_ITa,
            batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(),
            batch.GetEnd());
#endif // #if USE_ASM_KERNELS
    } else {
        OsdCpuComputeController::ApplyCatmarkRestrictedVertexVerticesKernelA(
            batch, context);
    }
}

void OsdNeonComputeController::Synchronize()
{
}

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
