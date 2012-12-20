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
    #include "TargetConditionals.h"
    #if TARGET_OS_IPHONE or TARGET_IPHONE_SIMULATOR
        #include <OpenGLES/ES2/gl.h>
    #else
        #include <OpenGL/gl3.h>
    #endif
#elif defined(ANDROID)
    #include <GLES2/gl2.h>
#else
    #if defined(_WIN32)
        #include <windows.h>
    #endif
    #include <GL/glew.h>
#endif

#include "../osd/glslComputeContext.h"
#include "../osd/glslDispatcher.h"
#include "../osd/glslKernelBundle.h"

#include <cassert>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

OsdGLSLComputeKernelDispatcher::OsdGLSLComputeKernelDispatcher() {
}

OsdGLSLComputeKernelDispatcher::~OsdGLSLComputeKernelDispatcher() {
}

void
OsdGLSLComputeKernelDispatcher::Refine(FarMesh<OsdVertex> * mesh,
                                OsdGLSLComputeContext *context) {

    FarDispatcher<OsdVertex>::Refine(mesh, /*maxlevel =*/ -1, context);
}

OsdGLSLComputeKernelDispatcher *
OsdGLSLComputeKernelDispatcher::GetInstance() {

    static OsdGLSLComputeKernelDispatcher instance;
    return &instance;
}

void
OsdGLSLComputeKernelDispatcher::ApplyBilinearFaceVerticesKernel(
    FarMesh<OsdVertex> * mesh, int offset, int level,
    int start, int end, void * clientdata) const {

    OsdGLSLComputeContext * context =
        static_cast<OsdGLSLComputeContext*>(clientdata);
    assert(context);

    OsdGLSLComputeKernelBundle * kernelBundle = context->GetKernelBundle();

    kernelBundle->ApplyBilinearFaceVerticesKernel(
        context->GetTable(Table::F_IT)->GetMarker(level-1),
        context->GetTable(Table::F_ITa)->GetMarker(level-1),
        offset, start, end);
}

void
OsdGLSLComputeKernelDispatcher::ApplyBilinearEdgeVerticesKernel(
    FarMesh<OsdVertex> * mesh, int offset, int level,
    int start, int end, void * clientdata) const {

    OsdGLSLComputeContext * context =
        static_cast<OsdGLSLComputeContext*>(clientdata);
    assert(context);

    OsdGLSLComputeKernelBundle * kernelBundle = context->GetKernelBundle();

    kernelBundle->ApplyBilinearEdgeVerticesKernel(
        context->GetTable(Table::E_IT)->GetMarker(level-1),
        offset, start, end);
}

void
OsdGLSLComputeKernelDispatcher::ApplyBilinearVertexVerticesKernel(
    FarMesh<OsdVertex> * mesh, int offset, int level,
    int start, int end, void * clientdata) const {

    OsdGLSLComputeContext * context =
        static_cast<OsdGLSLComputeContext*>(clientdata);
    assert(context);

    OsdGLSLComputeKernelBundle * kernelBundle = context->GetKernelBundle();

    kernelBundle->ApplyBilinearVertexVerticesKernel(
        context->GetTable(Table::V_ITa)->GetMarker(level-1),
        offset, start, end);
}

void
OsdGLSLComputeKernelDispatcher::ApplyCatmarkFaceVerticesKernel(
    FarMesh<OsdVertex> * mesh, int offset, int level,
    int start, int end, void * clientdata) const {

    OsdGLSLComputeContext * context =
        static_cast<OsdGLSLComputeContext*>(clientdata);
    assert(context);

    OsdGLSLComputeKernelBundle * kernelBundle = context->GetKernelBundle();

    kernelBundle->ApplyCatmarkFaceVerticesKernel(
        context->GetTable(Table::F_IT)->GetMarker(level-1),
        context->GetTable(Table::F_ITa)->GetMarker(level-1),
        offset, start, end);
}



void
OsdGLSLComputeKernelDispatcher::ApplyCatmarkEdgeVerticesKernel(
    FarMesh<OsdVertex> * mesh, int offset, int level,
    int start, int end, void * clientdata) const {

    OsdGLSLComputeContext * context =
        static_cast<OsdGLSLComputeContext*>(clientdata);
    assert(context);

    OsdGLSLComputeKernelBundle * kernelBundle = context->GetKernelBundle();

    kernelBundle->ApplyCatmarkEdgeVerticesKernel(
        context->GetTable(Table::E_IT)->GetMarker(level-1),
        context->GetTable(Table::E_W)->GetMarker(level-1),
        offset, start, end);
}

void
OsdGLSLComputeKernelDispatcher::ApplyCatmarkVertexVerticesKernelB(
    FarMesh<OsdVertex> * mesh, int offset, int level,
    int start, int end, void * clientdata) const {

    OsdGLSLComputeContext * context =
        static_cast<OsdGLSLComputeContext*>(clientdata);
    assert(context);

    OsdGLSLComputeKernelBundle * kernelBundle = context->GetKernelBundle();

    kernelBundle->ApplyCatmarkVertexVerticesKernelB(
        context->GetTable(Table::V_IT)->GetMarker(level-1),
        context->GetTable(Table::V_ITa)->GetMarker(level-1),
        context->GetTable(Table::V_W)->GetMarker(level-1),
        offset, start, end);
}

void
OsdGLSLComputeKernelDispatcher::ApplyCatmarkVertexVerticesKernelA(
    FarMesh<OsdVertex> * mesh, int offset, bool pass, int level,
    int start, int end, void * clientdata) const {

    OsdGLSLComputeContext * context =
        static_cast<OsdGLSLComputeContext*>(clientdata);
    assert(context);

    OsdGLSLComputeKernelBundle * kernelBundle = context->GetKernelBundle();

    kernelBundle->ApplyCatmarkVertexVerticesKernelA(
        context->GetTable(Table::V_ITa)->GetMarker(level-1),
        context->GetTable(Table::V_W)->GetMarker(level-1),
        offset, pass, start, end);
}



void
OsdGLSLComputeKernelDispatcher::ApplyLoopEdgeVerticesKernel(
    FarMesh<OsdVertex> * mesh, int offset, int level,
    int start, int end, void * clientdata) const {

    OsdGLSLComputeContext * context =
        static_cast<OsdGLSLComputeContext*>(clientdata);
    assert(context);

    OsdGLSLComputeKernelBundle * kernelBundle = context->GetKernelBundle();

    kernelBundle->ApplyLoopEdgeVerticesKernel(
        context->GetTable(Table::E_IT)->GetMarker(level-1),
        context->GetTable(Table::E_W)->GetMarker(level-1),
        offset, start, end);
}

void
OsdGLSLComputeKernelDispatcher::ApplyLoopVertexVerticesKernelB(
    FarMesh<OsdVertex> * mesh, int offset, int level,
    int start, int end, void * clientdata) const {

    OsdGLSLComputeContext * context =
        static_cast<OsdGLSLComputeContext*>(clientdata);
    assert(context);

    OsdGLSLComputeKernelBundle * kernelBundle = context->GetKernelBundle();

    kernelBundle->ApplyLoopVertexVerticesKernelB(
        context->GetTable(Table::V_IT)->GetMarker(level-1),
        context->GetTable(Table::V_ITa)->GetMarker(level-1),
        context->GetTable(Table::V_W)->GetMarker(level-1),
        offset, start, end);
}

void
OsdGLSLComputeKernelDispatcher::ApplyLoopVertexVerticesKernelA(
    FarMesh<OsdVertex> * mesh, int offset, bool pass,
    int level, int start, int end, void * clientdata) const {

    OsdGLSLComputeContext * context =
        static_cast<OsdGLSLComputeContext*>(clientdata);
    assert(context);

    OsdGLSLComputeKernelBundle * kernelBundle = context->GetKernelBundle();

    kernelBundle->ApplyLoopVertexVerticesKernelA(
        context->GetTable(Table::V_ITa)->GetMarker(level-1),
        context->GetTable(Table::V_W)->GetMarker(level-1),
        offset, pass, start, end);
}

void
OsdGLSLComputeKernelDispatcher::ApplyVertexEdits(
    FarMesh<OsdVertex> *mesh, int offset, int level,
    void * clientdata) const {

    OsdGLSLComputeContext * context =
        static_cast<OsdGLSLComputeContext*>(clientdata);
    assert(context);

    int numEditTables = context->GetNumEditTables();
    if (numEditTables == 0) return;

    OsdGLSLComputeKernelBundle * kernelBundle = context->GetKernelBundle();

    for (int i=0; i < numEditTables; ++i) {

        const OsdGLSLComputeHEditTable * edit = context->GetEditTable(i);
        assert(edit);

        const OsdGLSLComputeTable * primvarIndices = edit->GetPrimvarIndices();
        const OsdGLSLComputeTable * editValues = edit->GetEditValues();

        context->BindEditShaderStorageBuffers(i);

        int indices_ofs = primvarIndices->GetMarker(level-1);
        int values_ofs = editValues->GetMarker(level-1);
        int numVertices = primvarIndices->GetNumElements(level-1);
        int primvarOffset = edit->GetPrimvarOffset();
        int primvarWidth = edit->GetPrimvarWidth();

        if (edit->GetOperation() == FarVertexEdit::Add) {
            kernelBundle->ApplyEditAdd(numVertices, indices_ofs, values_ofs,
                                       primvarOffset, primvarWidth);
        } else {
            // XXX: edit SET is not implemented yet.
        }

    }
    context->UnbindEditShaderStorageBuffers();
}



}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
