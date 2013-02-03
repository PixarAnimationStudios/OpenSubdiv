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

#include "../osd/gcdDispatcher.h"
#include "../osd/gcdKernel.h"
#include "../osd/cpuComputeContext.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

OsdGcdKernelDispatcher::OsdGcdKernelDispatcher() {
    _gcd_queue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);
}

OsdGcdKernelDispatcher::~OsdGcdKernelDispatcher() {
}

void
OsdGcdKernelDispatcher::Refine(FarMesh<OsdVertex> * mesh,
                               OsdCpuComputeContext *context) const {

    FarDispatcher<OsdVertex>::Refine(mesh, /*maxlevel =*/ -1, context);
}

OsdGcdKernelDispatcher *
OsdGcdKernelDispatcher::GetInstance() {

    static OsdGcdKernelDispatcher instance;
    return &instance;
}

void
OsdGcdKernelDispatcher::ApplyBilinearFaceVerticesKernel(
    FarMesh<OsdVertex> * mesh, int offset, int level,
    int start, int end, void * clientdata) const {

    OsdCpuComputeContext * context =
        static_cast<OsdCpuComputeContext*>(clientdata);
    assert(context);

    OsdGcdComputeFace(
        context->GetVertexDescriptor(),
        context->GetCurrentVertexBuffer(),
        context->GetCurrentVaryingBuffer(),
        (const int*)context->GetTablePtr(Table::F_IT, level-1),
        (const int*)context->GetTablePtr(Table::F_ITa, level-1),
        offset, start, end,
        _gcd_queue);
}

void
OsdGcdKernelDispatcher::ApplyBilinearEdgeVerticesKernel(
    FarMesh<OsdVertex> * mesh, int offset, int level,
    int start, int end, void * clientdata) const {

    OsdCpuComputeContext * context =
        static_cast<OsdCpuComputeContext*>(clientdata);
    assert(context);

    OsdGcdComputeBilinearEdge(
        context->GetVertexDescriptor(),
        context->GetCurrentVertexBuffer(),
        context->GetCurrentVaryingBuffer(),
        (const int*)context->GetTablePtr(Table::E_IT, level-1),
        offset, start, end,
        _gcd_queue);
}

void
OsdGcdKernelDispatcher::ApplyBilinearVertexVerticesKernel(
    FarMesh<OsdVertex> * mesh, int offset, int level,
    int start, int end, void * clientdata) const {

    OsdCpuComputeContext * context =
        static_cast<OsdCpuComputeContext*>(clientdata);
    assert(context);

    OsdGcdComputeBilinearVertex(
        context->GetVertexDescriptor(),
        context->GetCurrentVertexBuffer(),
        context->GetCurrentVaryingBuffer(),
        (const int*)context->GetTablePtr(Table::V_ITa, level-1),
        offset, start, end,
        _gcd_queue);
}

void
OsdGcdKernelDispatcher::ApplyCatmarkFaceVerticesKernel(
    FarMesh<OsdVertex> * mesh, int offset, int level,
    int start, int end, void * clientdata) const {

    OsdCpuComputeContext * context =
        static_cast<OsdCpuComputeContext*>(clientdata);
    assert(context);

    OsdGcdComputeFace(
        context->GetVertexDescriptor(),
        context->GetCurrentVertexBuffer(),
        context->GetCurrentVaryingBuffer(),
        (const int*)context->GetTablePtr(Table::F_IT, level-1),
        (const int*)context->GetTablePtr(Table::F_ITa, level-1),
        offset, start, end,
        _gcd_queue);
}

void
OsdGcdKernelDispatcher::ApplyCatmarkEdgeVerticesKernel(
    FarMesh<OsdVertex> * mesh, int offset, int level,
    int start, int end, void * clientdata) const {

    OsdCpuComputeContext * context =
        static_cast<OsdCpuComputeContext*>(clientdata);
    assert(context);

    OsdGcdComputeEdge(
        context->GetVertexDescriptor(),
        context->GetCurrentVertexBuffer(),
        context->GetCurrentVaryingBuffer(),
        (const int*)context->GetTablePtr(Table::E_IT, level-1),
        (const float*)context->GetTablePtr(Table::E_W, level-1),
        offset, start, end,
        _gcd_queue);
}

void
OsdGcdKernelDispatcher::ApplyCatmarkVertexVerticesKernelB(
    FarMesh<OsdVertex> * mesh, int offset, int level,
    int start, int end, void * clientdata) const {

    OsdCpuComputeContext * context =
        static_cast<OsdCpuComputeContext*>(clientdata);
    assert(context);

    OsdGcdComputeVertexB(
        context->GetVertexDescriptor(),
        context->GetCurrentVertexBuffer(),
        context->GetCurrentVaryingBuffer(),
        (const int*)context->GetTablePtr(Table::V_ITa, level-1),
        (const int*)context->GetTablePtr(Table::V_IT, level-1),
        (const float*)context->GetTablePtr(Table::V_W, level-1),
        offset, start, end,
        _gcd_queue);
}

void
OsdGcdKernelDispatcher::ApplyCatmarkVertexVerticesKernelA(
    FarMesh<OsdVertex> * mesh, int offset, bool pass, int level,
    int start, int end, void * clientdata) const {

    OsdCpuComputeContext * context =
        static_cast<OsdCpuComputeContext*>(clientdata);
    assert(context);

    OsdGcdComputeVertexA(
        context->GetVertexDescriptor(),
        context->GetCurrentVertexBuffer(),
        context->GetCurrentVaryingBuffer(),
        (const int*)context->GetTablePtr(Table::V_ITa, level-1),
        (const float*)context->GetTablePtr(Table::V_W, level-1),
        offset, start, end, pass,
        _gcd_queue);
}

void
OsdGcdKernelDispatcher::ApplyLoopEdgeVerticesKernel(
    FarMesh<OsdVertex> * mesh, int offset, int level,
    int start, int end, void * clientdata) const {

    OsdCpuComputeContext * context =
        static_cast<OsdCpuComputeContext*>(clientdata);
    assert(context);

    OsdGcdComputeEdge(
        context->GetVertexDescriptor(),
        context->GetCurrentVertexBuffer(),
        context->GetCurrentVaryingBuffer(),
        (const int*)context->GetTablePtr(Table::E_IT, level-1),
        (const float*)context->GetTablePtr(Table::E_W, level-1),
        offset, start, end,
        _gcd_queue);
}

void
OsdGcdKernelDispatcher::ApplyLoopVertexVerticesKernelB(
    FarMesh<OsdVertex> * mesh, int offset, int level,
    int start, int end, void * clientdata) const {

    OsdCpuComputeContext * context =
        static_cast<OsdCpuComputeContext*>(clientdata);
    assert(context);

    OsdGcdComputeLoopVertexB(
        context->GetVertexDescriptor(),
        context->GetCurrentVertexBuffer(),
        context->GetCurrentVaryingBuffer(),
        (const int*)context->GetTablePtr(Table::V_ITa, level-1),
        (const int*)context->GetTablePtr(Table::V_IT, level-1),
        (const float*)context->GetTablePtr(Table::V_W, level-1),
        offset, start, end,
        _gcd_queue);
}

void
OsdGcdKernelDispatcher::ApplyLoopVertexVerticesKernelA(
    FarMesh<OsdVertex> * mesh, int offset, bool pass, int level,
    int start, int end, void * clientdata) const {

    OsdCpuComputeContext * context =
        static_cast<OsdCpuComputeContext*>(clientdata);
    assert(context);

    OsdGcdComputeVertexA(
        context->GetVertexDescriptor(),
        context->GetCurrentVertexBuffer(),
        context->GetCurrentVaryingBuffer(),
        (const int*)context->GetTablePtr(Table::V_ITa, level-1),
        (const float*)context->GetTablePtr(Table::V_W, level-1),
        offset, start, end, pass,
        _gcd_queue);
}

void
OsdGcdKernelDispatcher::ApplyVertexEdits(
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
            OsdGcdEditVertexAdd(context->GetVertexDescriptor(),
                                context->GetCurrentVertexBuffer(),
                                edit->GetPrimvarIndex(),
                                edit->GetPrimvarWidth(),
                                primvarIndices.GetNumElements(level-1),
                                (const int*)primvarIndices[level-1],
                                (const float*)editValues[level-1],
                                _gcd_queue);
        } else if (edit->GetOperation() == FarVertexEdit::Set) {
            OsdGcdEditVertexSet(context->GetVertexDescriptor(),
                                context->GetCurrentVertexBuffer(),
                                edit->GetPrimvarIndex(),
                                edit->GetPrimvarWidth(),
                                primvarIndices.GetNumElements(level-1),
                                (const int*)primvarIndices[level-1],
                                (const float*)editValues[level-1],
                                _gcd_queue);
        }
    }
}

}  // end namespace OPENSUBDIV_VERSION

}  // end namespace OpenSubdiv
