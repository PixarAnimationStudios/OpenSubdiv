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

#if defined(__APPLE__)
    #include <OpenGL/gl3.h>
#else
    #include <GL/glew.h>
#endif

#include "../osd/glslTransformFeedbackDispatcher.h"
#include "../osd/glslTransformFeedbackComputeContext.h"
#include "../osd/glslTransformFeedbackKernelBundle.h"

#include <cassert>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

OsdGLSLTransformFeedbackKernelDispatcher::OsdGLSLTransformFeedbackKernelDispatcher() {
}

OsdGLSLTransformFeedbackKernelDispatcher::~OsdGLSLTransformFeedbackKernelDispatcher() {
}

void
OsdGLSLTransformFeedbackKernelDispatcher::Refine(FarMesh<OsdVertex> * mesh,
                                OsdGLSLTransformFeedbackComputeContext *context) {

    FarDispatcher<OsdVertex>::Refine(mesh, /*maxlevel =*/ -1, context);
}

OsdGLSLTransformFeedbackKernelDispatcher *
OsdGLSLTransformFeedbackKernelDispatcher::GetInstance() {

    static OsdGLSLTransformFeedbackKernelDispatcher instance;
    return &instance;
}

void
OsdGLSLTransformFeedbackKernelDispatcher::ApplyBilinearFaceVerticesKernel(
    FarMesh<OsdVertex> * mesh, int offset, int level,
    int start, int end, void * clientdata) const {

    OsdGLSLTransformFeedbackComputeContext * context =
        static_cast<OsdGLSLTransformFeedbackComputeContext*>(clientdata);
    assert(context);

    OsdGLSLTransformFeedbackKernelBundle * kernelBundle = context->GetKernelBundle();

    kernelBundle->ApplyBilinearFaceVerticesKernel(
        context->GetCurrentVertexBuffer(),
        context->GetNumCurrentVertexElements(),
        context->GetCurrentVaryingBuffer(),
        context->GetNumCurrentVaryingElements(),
        context->GetTable(Table::F_IT)->GetMarker(level-1),
        context->GetTable(Table::F_ITa)->GetMarker(level-1),
        offset, start, end);
}

void
OsdGLSLTransformFeedbackKernelDispatcher::ApplyBilinearEdgeVerticesKernel(
    FarMesh<OsdVertex> * mesh, int offset, int level,
    int start, int end, void * clientdata) const {

    OsdGLSLTransformFeedbackComputeContext * context =
        static_cast<OsdGLSLTransformFeedbackComputeContext*>(clientdata);
    assert(context);

    OsdGLSLTransformFeedbackKernelBundle * kernelBundle = context->GetKernelBundle();

    kernelBundle->ApplyBilinearEdgeVerticesKernel(
        context->GetCurrentVertexBuffer(),
        context->GetNumCurrentVertexElements(),
        context->GetCurrentVaryingBuffer(),
        context->GetNumCurrentVaryingElements(),
        context->GetTable(Table::E_IT)->GetMarker(level-1),
        offset, start, end);
}

void
OsdGLSLTransformFeedbackKernelDispatcher::ApplyBilinearVertexVerticesKernel(
    FarMesh<OsdVertex> * mesh, int offset, int level,
    int start, int end, void * clientdata) const {

    OsdGLSLTransformFeedbackComputeContext * context =
        static_cast<OsdGLSLTransformFeedbackComputeContext*>(clientdata);
    assert(context);

    OsdGLSLTransformFeedbackKernelBundle * kernelBundle = context->GetKernelBundle();

    kernelBundle->ApplyBilinearVertexVerticesKernel(
        context->GetCurrentVertexBuffer(),
        context->GetNumCurrentVertexElements(),
        context->GetCurrentVaryingBuffer(),
        context->GetNumCurrentVaryingElements(),
        context->GetTable(Table::V_ITa)->GetMarker(level-1),
        offset, start, end);
}

void
OsdGLSLTransformFeedbackKernelDispatcher::ApplyCatmarkFaceVerticesKernel(
    FarMesh<OsdVertex> * mesh, int offset, int level,
    int start, int end, void * clientdata) const {

    OsdGLSLTransformFeedbackComputeContext * context =
        static_cast<OsdGLSLTransformFeedbackComputeContext*>(clientdata);
    assert(context);

    OsdGLSLTransformFeedbackKernelBundle * kernelBundle = context->GetKernelBundle();

    kernelBundle->ApplyCatmarkFaceVerticesKernel(
        context->GetCurrentVertexBuffer(),
        context->GetNumCurrentVertexElements(),
        context->GetCurrentVaryingBuffer(),
        context->GetNumCurrentVaryingElements(),
        context->GetTable(Table::F_IT)->GetMarker(level-1),
        context->GetTable(Table::F_ITa)->GetMarker(level-1),
        offset, start, end);
}



void
OsdGLSLTransformFeedbackKernelDispatcher::ApplyCatmarkEdgeVerticesKernel(
    FarMesh<OsdVertex> * mesh, int offset, int level,
    int start, int end, void * clientdata) const {

    OsdGLSLTransformFeedbackComputeContext * context =
        static_cast<OsdGLSLTransformFeedbackComputeContext*>(clientdata);
    assert(context);

    OsdGLSLTransformFeedbackKernelBundle * kernelBundle = context->GetKernelBundle();

    kernelBundle->ApplyCatmarkEdgeVerticesKernel(
        context->GetCurrentVertexBuffer(),
        context->GetNumCurrentVertexElements(),
        context->GetCurrentVaryingBuffer(),
        context->GetNumCurrentVaryingElements(),
        context->GetTable(Table::E_IT)->GetMarker(level-1),
        context->GetTable(Table::E_W)->GetMarker(level-1),
        offset, start, end);
}

void
OsdGLSLTransformFeedbackKernelDispatcher::ApplyCatmarkVertexVerticesKernelB(
    FarMesh<OsdVertex> * mesh, int offset, int level,
    int start, int end, void * clientdata) const {

    OsdGLSLTransformFeedbackComputeContext * context =
        static_cast<OsdGLSLTransformFeedbackComputeContext*>(clientdata);
    assert(context);

    OsdGLSLTransformFeedbackKernelBundle * kernelBundle = context->GetKernelBundle();

    kernelBundle->ApplyCatmarkVertexVerticesKernelB(
        context->GetCurrentVertexBuffer(),
        context->GetNumCurrentVertexElements(),
        context->GetCurrentVaryingBuffer(),
        context->GetNumCurrentVaryingElements(),
        context->GetTable(Table::V_IT)->GetMarker(level-1),
        context->GetTable(Table::V_ITa)->GetMarker(level-1),
        context->GetTable(Table::V_W)->GetMarker(level-1),
        offset, start, end);
}

void
OsdGLSLTransformFeedbackKernelDispatcher::ApplyCatmarkVertexVerticesKernelA(
    FarMesh<OsdVertex> * mesh, int offset, bool pass, int level,
    int start, int end, void * clientdata) const {

    OsdGLSLTransformFeedbackComputeContext * context =
        static_cast<OsdGLSLTransformFeedbackComputeContext*>(clientdata);
    assert(context);

    OsdGLSLTransformFeedbackKernelBundle * kernelBundle = context->GetKernelBundle();

    kernelBundle->ApplyCatmarkVertexVerticesKernelA(
        context->GetCurrentVertexBuffer(),
        context->GetNumCurrentVertexElements(),
        context->GetCurrentVaryingBuffer(),
        context->GetNumCurrentVaryingElements(),
        context->GetTable(Table::V_ITa)->GetMarker(level-1),
        context->GetTable(Table::V_W)->GetMarker(level-1),
        offset, pass, start, end);
}



void
OsdGLSLTransformFeedbackKernelDispatcher::ApplyLoopEdgeVerticesKernel(
    FarMesh<OsdVertex> * mesh, int offset, int level,
    int start, int end, void * clientdata) const {

    OsdGLSLTransformFeedbackComputeContext * context =
        static_cast<OsdGLSLTransformFeedbackComputeContext*>(clientdata);
    assert(context);

    OsdGLSLTransformFeedbackKernelBundle * kernelBundle = context->GetKernelBundle();

    kernelBundle->ApplyLoopEdgeVerticesKernel(
        context->GetCurrentVertexBuffer(),
        context->GetNumCurrentVertexElements(),
        context->GetCurrentVaryingBuffer(),
        context->GetNumCurrentVaryingElements(),
        context->GetTable(Table::E_IT)->GetMarker(level-1),
        context->GetTable(Table::E_W)->GetMarker(level-1),
        offset, start, end);
}

void
OsdGLSLTransformFeedbackKernelDispatcher::ApplyLoopVertexVerticesKernelB(
    FarMesh<OsdVertex> * mesh, int offset, int level,
    int start, int end, void * clientdata) const {

    OsdGLSLTransformFeedbackComputeContext * context =
        static_cast<OsdGLSLTransformFeedbackComputeContext*>(clientdata);
    assert(context);

    OsdGLSLTransformFeedbackKernelBundle * kernelBundle = context->GetKernelBundle();

    kernelBundle->ApplyLoopVertexVerticesKernelB(
        context->GetCurrentVertexBuffer(),
        context->GetNumCurrentVertexElements(),
        context->GetCurrentVaryingBuffer(),
        context->GetNumCurrentVaryingElements(),
        context->GetTable(Table::V_IT)->GetMarker(level-1),
        context->GetTable(Table::V_ITa)->GetMarker(level-1),
        context->GetTable(Table::V_W)->GetMarker(level-1),
        offset, start, end);
}

void
OsdGLSLTransformFeedbackKernelDispatcher::ApplyLoopVertexVerticesKernelA(
    FarMesh<OsdVertex> * mesh, int offset, bool pass,
    int level, int start, int end, void * clientdata) const {

    OsdGLSLTransformFeedbackComputeContext * context =
        static_cast<OsdGLSLTransformFeedbackComputeContext*>(clientdata);
    assert(context);

    OsdGLSLTransformFeedbackKernelBundle * kernelBundle = context->GetKernelBundle();

    kernelBundle->ApplyLoopVertexVerticesKernelA(
        context->GetCurrentVertexBuffer(),
        context->GetNumCurrentVertexElements(),
        context->GetCurrentVaryingBuffer(),
        context->GetNumCurrentVaryingElements(),
        context->GetTable(Table::V_ITa)->GetMarker(level-1),
        context->GetTable(Table::V_W)->GetMarker(level-1),
        offset, pass, start, end);
}


void
OsdGLSLTransformFeedbackKernelDispatcher::ApplyVertexEdits(
    FarMesh<OsdVertex> *mesh, int offset, int level,
    void * clientdata) const {

    OsdGLSLTransformFeedbackComputeContext * context =
        static_cast<OsdGLSLTransformFeedbackComputeContext*>(clientdata);
    assert(context);

    int numEditTables = context->GetNumEditTables();
    if (numEditTables == 0) return;

    OsdGLSLTransformFeedbackKernelBundle * kernelBundle = context->GetKernelBundle();

    for (int i=0; i < numEditTables; ++i) {

        const OsdGLSLTransformFeedbackHEditTable * edit = context->GetEditTable(i);
        assert(edit);

        const OsdGLSLTransformFeedbackTable * primvarIndices = edit->GetPrimvarIndices();
        const OsdGLSLTransformFeedbackTable * editValues = edit->GetEditValues();

        context->BindEditTextures(i);

        int indices_ofs = primvarIndices->GetMarker(level-1);
        int values_ofs = editValues->GetMarker(level-1);
        int numVertices = primvarIndices->GetNumElements(level-1);
        int primvarOffset = edit->GetPrimvarOffset();
        int primvarWidth = edit->GetPrimvarWidth();

        if (edit->GetOperation() == FarVertexEdit::Add) {
            kernelBundle->ApplyEditAdd(
                context->GetCurrentVertexBuffer(),
                context->GetNumCurrentVertexElements(),
                context->GetCurrentVaryingBuffer(),
                context->GetNumCurrentVaryingElements(),
                numVertices, indices_ofs, values_ofs,
                primvarOffset, primvarWidth);
        } else {
            // XXX: edit SET is not implemented yet.
        }

    }
    context->UnbindEditTextures();
}

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
