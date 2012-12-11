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

#include "../osd/cpuDispatcher.h"
#include "../osd/cpuKernel.h"
#include "../osd/cpuComputeContext.h"

#include <stdlib.h>
#include <string.h>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

OsdCpuKernelDispatcher::OsdCpuKernelDispatcher() {
}

OsdCpuKernelDispatcher::~OsdCpuKernelDispatcher() {

}

void
OsdCpuKernelDispatcher::Refine(FarMesh<OsdVertex> * mesh,
                               OsdCpuComputeContext *context) const {

    FarDispatcher<OsdVertex>::Refine(mesh, /*maxlevel =*/ -1, context);
}

OsdCpuKernelDispatcher *
OsdCpuKernelDispatcher::GetInstance() {

    static OsdCpuKernelDispatcher instance;
    return &instance;
}

void
OsdCpuKernelDispatcher::ApplyBilinearFaceVerticesKernel(
    FarMesh<OsdVertex> * mesh, int offset, int level,
    int start, int end, void * clientdata) const {

    OsdCpuComputeContext * context =
        static_cast<OsdCpuComputeContext*>(clientdata);
    assert(context);

    OsdCpuComputeFace(
        context->GetVertexDescriptor(),
        context->GetCurrentVertexBuffer(),
        context->GetCurrentVaryingBuffer(),
        (const int*)context->GetTablePtr(Table::F_IT, level-1),
        (const int*)context->GetTablePtr(Table::F_ITa, level-1),
        offset, start, end);
}

void
OsdCpuKernelDispatcher::ApplyBilinearEdgeVerticesKernel(
    FarMesh<OsdVertex> * mesh, int offset, int level,
    int start, int end, void * clientdata) const {

    OsdCpuComputeContext * context =
        static_cast<OsdCpuComputeContext*>(clientdata);
    assert(context);

    OsdCpuComputeBilinearEdge(
        context->GetVertexDescriptor(),
        context->GetCurrentVertexBuffer(),
        context->GetCurrentVaryingBuffer(),
        (const int*)context->GetTablePtr(Table::E_IT, level-1),
        offset, start, end);
}

void
OsdCpuKernelDispatcher::ApplyBilinearVertexVerticesKernel(
    FarMesh<OsdVertex> * mesh, int offset, int level,
    int start, int end, void * clientdata) const {

    OsdCpuComputeContext * context =
        static_cast<OsdCpuComputeContext*>(clientdata);
    assert(context);

    OsdCpuComputeBilinearVertex(
        context->GetVertexDescriptor(),
        context->GetCurrentVertexBuffer(),
        context->GetCurrentVaryingBuffer(),
        (const int*)context->GetTablePtr(Table::V_ITa, level-1),
        offset, start, end);
}

void
OsdCpuKernelDispatcher::ApplyCatmarkFaceVerticesKernel(
    FarMesh<OsdVertex> * mesh, int offset, int level,
    int start, int end, void * clientdata) const {

    OsdCpuComputeContext * context =
        static_cast<OsdCpuComputeContext*>(clientdata);
    assert(context);

    OsdCpuComputeFace(
        context->GetVertexDescriptor(),
        context->GetCurrentVertexBuffer(),
        context->GetCurrentVaryingBuffer(),
        (const int*)context->GetTablePtr(Table::F_IT, level-1),
        (const int*)context->GetTablePtr(Table::F_ITa, level-1),
        offset, start, end);
}

void
OsdCpuKernelDispatcher::ApplyCatmarkEdgeVerticesKernel(
    FarMesh<OsdVertex> * mesh, int offset, int level,
    int start, int end, void * clientdata) const {

    OsdCpuComputeContext * context =
        static_cast<OsdCpuComputeContext*>(clientdata);
    assert(context);

    OsdCpuComputeEdge(
        context->GetVertexDescriptor(),
        context->GetCurrentVertexBuffer(),
        context->GetCurrentVaryingBuffer(),
        (const int*)context->GetTablePtr(Table::E_IT, level-1),
        (const float*)context->GetTablePtr(Table::E_W, level-1),
        offset, start, end);
}

void
OsdCpuKernelDispatcher::ApplyCatmarkVertexVerticesKernelB(
    FarMesh<OsdVertex> * mesh, int offset, int level,
    int start, int end, void * clientdata) const {

    OsdCpuComputeContext * context =
        static_cast<OsdCpuComputeContext*>(clientdata);
    assert(context);

    OsdCpuComputeVertexB(
        context->GetVertexDescriptor(),
        context->GetCurrentVertexBuffer(),
        context->GetCurrentVaryingBuffer(),
        (const int*)context->GetTablePtr(Table::V_ITa, level-1),
        (const int*)context->GetTablePtr(Table::V_IT, level-1),
        (const float*)context->GetTablePtr(Table::V_W, level-1),
        offset, start, end);
}

void
OsdCpuKernelDispatcher::ApplyCatmarkVertexVerticesKernelA(
    FarMesh<OsdVertex> * mesh, int offset, bool pass, int level,
    int start, int end, void * clientdata) const {

    OsdCpuComputeContext * context =
        static_cast<OsdCpuComputeContext*>(clientdata);
    assert(context);

    OsdCpuComputeVertexA(
        context->GetVertexDescriptor(),
        context->GetCurrentVertexBuffer(),
        context->GetCurrentVaryingBuffer(),
        (const int*)context->GetTablePtr(Table::V_ITa, level-1),
        (const float*)context->GetTablePtr(Table::V_W, level-1),
        offset, start, end, pass);
}

void
OsdCpuKernelDispatcher::ApplyLoopEdgeVerticesKernel(
    FarMesh<OsdVertex> * mesh, int offset, int level,
    int start, int end, void * clientdata) const {

    OsdCpuComputeContext * context =
        static_cast<OsdCpuComputeContext*>(clientdata);
    assert(context);

    OsdCpuComputeEdge(
        context->GetVertexDescriptor(),
        context->GetCurrentVertexBuffer(),
        context->GetCurrentVaryingBuffer(),
        (const int*)context->GetTablePtr(Table::E_IT, level-1),
        (const float*)context->GetTablePtr(Table::E_W, level-1),
        offset, start, end);
}

void
OsdCpuKernelDispatcher::ApplyLoopVertexVerticesKernelB(
    FarMesh<OsdVertex> * mesh, int offset, int level,
    int start, int end, void * clientdata) const {

    OsdCpuComputeContext * context =
        static_cast<OsdCpuComputeContext*>(clientdata);
    assert(context);

    OsdCpuComputeLoopVertexB(
        context->GetVertexDescriptor(),
        context->GetCurrentVertexBuffer(),
        context->GetCurrentVaryingBuffer(),
        (const int*)context->GetTablePtr(Table::V_ITa, level-1),
        (const int*)context->GetTablePtr(Table::V_IT, level-1),
        (const float*)context->GetTablePtr(Table::V_W, level-1),
        offset, start, end);
}

void
OsdCpuKernelDispatcher::ApplyLoopVertexVerticesKernelA(
    FarMesh<OsdVertex> * mesh, int offset, bool pass, int level,
    int start, int end, void * clientdata) const {

    OsdCpuComputeContext * context =
        static_cast<OsdCpuComputeContext*>(clientdata);
    assert(context);

    OsdCpuComputeVertexA(
        context->GetVertexDescriptor(),
        context->GetCurrentVertexBuffer(),
        context->GetCurrentVaryingBuffer(),
        (const int*)context->GetTablePtr(Table::V_ITa, level-1),
        (const float*)context->GetTablePtr(Table::V_W, level-1),
        offset, start, end, pass);
}

void
OsdCpuKernelDispatcher::ApplyVertexEdits(
    FarMesh<OsdVertex> *mesh, int offset, int level, void *clientdata) const {

    OsdCpuComputeContext * context =
        static_cast<OsdCpuComputeContext*>(clientdata);
    assert(context);

    int numEdits = context->GetNumEditTables();

    for (int i = 0; i < numEdits; ++i) {

        const FarVertexEditTables<OsdVertex>::VertexEditBatch * edit =
            context->GetEditTable(i);
        assert(edit);

        const FarTable<unsigned int> &primvarIndices = edit->GetVertexIndices();
        const FarTable<float> &editValues = edit->GetValues();

        // XXX: how about edits for varying...?

        if (edit->GetOperation() == FarVertexEdit::Add) {
            OsdCpuEditVertexAdd(context->GetVertexDescriptor(),
                                context->GetCurrentVertexBuffer(),
                                edit->GetPrimvarIndex(),
                                edit->GetPrimvarWidth(),
                                primvarIndices.GetNumElements(level-1),
                                (const int*)primvarIndices[level-1],
                                (const float*)editValues[level-1]);
        } else if (edit->GetOperation() == FarVertexEdit::Set) {
            OsdCpuEditVertexSet(context->GetVertexDescriptor(),
                                context->GetCurrentVertexBuffer(),
                                edit->GetPrimvarIndex(),
                                edit->GetPrimvarWidth(),
                                primvarIndices.GetNumElements(level-1),
                                (const int*)primvarIndices[level-1],
                                (const float*)editValues[level-1]);
        }
    }
}

}  // end namespace OPENSUBDIV_VERSION

}  // end namespace OpenSubdiv
