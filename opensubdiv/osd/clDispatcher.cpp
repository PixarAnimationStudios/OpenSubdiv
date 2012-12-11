#include <stdio.h>
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

#include "../osd/clDispatcher.h"
#include "../osd/clComputeContext.h"
#include "../osd/clKernelBundle.h"
#include "../osd/error.h"

#if defined(_WIN32)
    #include <windows.h>
#elif defined(__APPLE__)
    #include <OpenCL/opencl.h>
#else
    #include <CL/opencl.h>
#endif

#include <string.h>
#include <algorithm>

// XXX: Error handling
#ifdef NDEBUG
#define CL_CHECK_ERROR(x, ...)
#else
#define CL_CHECK_ERROR(x, ...) {                     \
        if (x != CL_SUCCESS) {                       \
            OsdError(OSD_CL_RUNTIME_ERROR, "%d", x); \
            OsdError(OSD_CL_RUNTIME_ERROR, __VA_ARGS__); } }
#endif

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

OsdCLKernelDispatcher::OsdCLKernelDispatcher() {
}

OsdCLKernelDispatcher::~OsdCLKernelDispatcher() {
}

void
OsdCLKernelDispatcher::Refine(FarMesh<OsdVertex> * mesh,
                              OsdCLComputeContext *context) {

    FarDispatcher<OsdVertex>::Refine(mesh, /*maxlevel =*/ -1, context);
}

OsdCLKernelDispatcher *
OsdCLKernelDispatcher::GetInstance() {

    static OsdCLKernelDispatcher instance;
    return &instance;
}

void
OsdCLKernelDispatcher::ApplyBilinearFaceVerticesKernel(
    FarMesh<OsdVertex> * mesh, int offset, int level,
    int start, int end, void * clientdata) const {

    ApplyCatmarkFaceVerticesKernel(mesh, offset, level, start, end, clientdata);
}

void
OsdCLKernelDispatcher::ApplyBilinearEdgeVerticesKernel(
    FarMesh<OsdVertex> * mesh, int offset, int level,
    int start, int end, void * clientdata) const {

    OsdCLComputeContext * context =
        static_cast<OsdCLComputeContext*>(clientdata);
    assert(context);

    cl_int ciErrNum;
    size_t globalWorkSize[1] = { end-start };
    cl_kernel kernel = context->GetKernelBundle()->GetBilinearEdgeKernel();

    cl_mem vertexBuffer = context->GetCurrentVertexBuffer();
    cl_mem varyingBuffer = context->GetCurrentVaryingBuffer();
    cl_mem E_IT = context->GetTable(Table::E_IT)->GetDevicePtr();
    int E_IT_ofs = context->GetTable(Table::E_IT)->GetMarker(level-1);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &vertexBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &varyingBuffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &E_IT);
    clSetKernelArg(kernel, 3, sizeof(int), &E_IT_ofs);
    clSetKernelArg(kernel, 4, sizeof(int), &offset);
    clSetKernelArg(kernel, 5, sizeof(int), &start);
    clSetKernelArg(kernel, 6, sizeof(int), &end);
    ciErrNum = clEnqueueNDRangeKernel(context->GetCommandQueue(),
                                      kernel, 1, NULL, globalWorkSize,
                                      NULL, 0, NULL, NULL);
    CL_CHECK_ERROR(ciErrNum, "bilinear edge kernel %d\n", ciErrNum);
}

void
OsdCLKernelDispatcher::ApplyBilinearVertexVerticesKernel(
    FarMesh<OsdVertex> * mesh, int offset, int level,
    int start, int end, void * clientdata) const {

    OsdCLComputeContext * context =
        static_cast<OsdCLComputeContext*>(clientdata);
    assert(context);

    cl_int ciErrNum;
    size_t globalWorkSize[1] = { end-start };
    cl_kernel kernel = context->GetKernelBundle()->GetBilinearVertexKernel();

    cl_mem vertexBuffer = context->GetCurrentVertexBuffer();
    cl_mem varyingBuffer = context->GetCurrentVaryingBuffer();
    cl_mem V_ITa = context->GetTable(Table::V_ITa)->GetDevicePtr();
    int V_ITa_ofs = context->GetTable(Table::V_ITa)->GetMarker(level-1);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &vertexBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &varyingBuffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &V_ITa);
    clSetKernelArg(kernel, 3, sizeof(int), &V_ITa_ofs);
    clSetKernelArg(kernel, 4, sizeof(int), &offset);
    clSetKernelArg(kernel, 5, sizeof(int), &start);
    clSetKernelArg(kernel, 6, sizeof(int), &end);
    ciErrNum = clEnqueueNDRangeKernel(context->GetCommandQueue(),
                                      kernel, 1, NULL, globalWorkSize,
                                      NULL, 0, NULL, NULL);
    CL_CHECK_ERROR(ciErrNum, "bilinear vertex kernel 1 %d\n", ciErrNum);
}

void
OsdCLKernelDispatcher::ApplyCatmarkFaceVerticesKernel(
    FarMesh<OsdVertex> * mesh, int offset, int level,
    int start, int end, void * clientdata) const {

    OsdCLComputeContext * context =
        static_cast<OsdCLComputeContext*>(clientdata);
    assert(context);

    cl_int ciErrNum;
    size_t globalWorkSize[1] = { end-start };
    cl_kernel kernel = context->GetKernelBundle()->GetCatmarkFaceKernel();

    cl_mem vertexBuffer = context->GetCurrentVertexBuffer();
    cl_mem varyingBuffer = context->GetCurrentVaryingBuffer();
    cl_mem F_IT = context->GetTable(Table::F_IT)->GetDevicePtr();
    cl_mem F_ITa = context->GetTable(Table::F_ITa)->GetDevicePtr();
    int F_IT_ofs = context->GetTable(Table::F_IT)->GetMarker(level-1);
    int F_ITa_ofs = context->GetTable(Table::F_ITa)->GetMarker(level-1);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &vertexBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &varyingBuffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &F_IT);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &F_ITa);
    clSetKernelArg(kernel, 4, sizeof(int), &F_IT_ofs);
    clSetKernelArg(kernel, 5, sizeof(int), &F_ITa_ofs);
    clSetKernelArg(kernel, 6, sizeof(int), &offset);
    clSetKernelArg(kernel, 7, sizeof(int), &start);
    clSetKernelArg(kernel, 8, sizeof(int), &end);

    ciErrNum = clEnqueueNDRangeKernel(context->GetCommandQueue(),
                                      kernel, 1, NULL, globalWorkSize,
                                      NULL, 0, NULL, NULL);
    CL_CHECK_ERROR(ciErrNum, "face kernel lv[%d] %d\n", level, ciErrNum);
}

void
OsdCLKernelDispatcher::ApplyCatmarkEdgeVerticesKernel(
    FarMesh<OsdVertex> * mesh, int offset, int level,
    int start, int end, void * clientdata) const {

    OsdCLComputeContext * context =
        static_cast<OsdCLComputeContext*>(clientdata);
    assert(context);

    cl_int ciErrNum;
    size_t globalWorkSize[1] = { end-start };
    cl_kernel kernel = context->GetKernelBundle()->GetCatmarkEdgeKernel();

    cl_mem vertexBuffer = context->GetCurrentVertexBuffer();
    cl_mem varyingBuffer = context->GetCurrentVaryingBuffer();
    cl_mem E_IT = context->GetTable(Table::E_IT)->GetDevicePtr();
    cl_mem E_W = context->GetTable(Table::E_W)->GetDevicePtr();
    int E_IT_ofs = context->GetTable(Table::E_IT)->GetMarker(level-1);
    int E_W_ofs = context->GetTable(Table::E_W)->GetMarker(level-1);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &vertexBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &varyingBuffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &E_IT);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &E_W);
    clSetKernelArg(kernel, 4, sizeof(int), &E_IT_ofs);
    clSetKernelArg(kernel, 5, sizeof(int), &E_W_ofs);
    clSetKernelArg(kernel, 6, sizeof(int), &offset);
    clSetKernelArg(kernel, 7, sizeof(int), &start);
    clSetKernelArg(kernel, 8, sizeof(int), &end);

    ciErrNum = clEnqueueNDRangeKernel(context->GetCommandQueue(),
                                      kernel, 1, NULL, globalWorkSize,
                                      NULL, 0, NULL, NULL);
    CL_CHECK_ERROR(ciErrNum, "edge kernel %d\n", ciErrNum);
}

void
OsdCLKernelDispatcher::ApplyCatmarkVertexVerticesKernelB(
    FarMesh<OsdVertex> * mesh, int offset, int level,
    int start, int end, void * clientdata) const {

    OsdCLComputeContext * context =
        static_cast<OsdCLComputeContext*>(clientdata);
    assert(context);

    cl_int ciErrNum;
    size_t globalWorkSize[1] = { end-start };
    cl_kernel kernel = context->GetKernelBundle()->GetCatmarkVertexKernelB();

    cl_mem vertexBuffer = context->GetCurrentVertexBuffer();
    cl_mem varyingBuffer = context->GetCurrentVaryingBuffer();
    cl_mem V_ITa = context->GetTable(Table::V_ITa)->GetDevicePtr();
    cl_mem V_IT = context->GetTable(Table::V_IT)->GetDevicePtr();
    cl_mem V_W = context->GetTable(Table::V_W)->GetDevicePtr();
    int V_ITa_ofs = context->GetTable(Table::V_ITa)->GetMarker(level-1);
    int V_IT_ofs = context->GetTable(Table::V_IT)->GetMarker(level-1);
    int V_W_ofs = context->GetTable(Table::V_W)->GetMarker(level-1);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &vertexBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &varyingBuffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &V_ITa);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &V_IT);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &V_W);
    clSetKernelArg(kernel, 5, sizeof(int), &V_ITa_ofs);
    clSetKernelArg(kernel, 6, sizeof(int), &V_IT_ofs);
    clSetKernelArg(kernel, 7, sizeof(int), &V_W_ofs);
    clSetKernelArg(kernel, 8, sizeof(int), &offset);
    clSetKernelArg(kernel, 9, sizeof(int), &start);
    clSetKernelArg(kernel, 10, sizeof(int), &end);

    ciErrNum = clEnqueueNDRangeKernel(context->GetCommandQueue(),
                                      kernel, 1, NULL, globalWorkSize,
                                      NULL, 0, NULL, NULL);
    CL_CHECK_ERROR(ciErrNum, "vertex kernel 1 %d\n", ciErrNum);
}

void
OsdCLKernelDispatcher::ApplyCatmarkVertexVerticesKernelA(
    FarMesh<OsdVertex> * mesh, int offset, bool pass, int level,
    int start, int end, void * clientdata) const {

    OsdCLComputeContext * context =
        static_cast<OsdCLComputeContext*>(clientdata);
    assert(context);

    cl_int ciErrNum;
    size_t globalWorkSize[1] = { end-start };
    int ipass = pass;
    cl_kernel kernel = context->GetKernelBundle()->GetCatmarkVertexKernelA();

    cl_mem vertexBuffer = context->GetCurrentVertexBuffer();
    cl_mem varyingBuffer = context->GetCurrentVaryingBuffer();
    cl_mem V_ITa = context->GetTable(Table::V_ITa)->GetDevicePtr();
    cl_mem V_W = context->GetTable(Table::V_W)->GetDevicePtr();
    int V_ITa_ofs = context->GetTable(Table::V_ITa)->GetMarker(level-1);
    int V_W_ofs = context->GetTable(Table::V_W)->GetMarker(level-1);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &vertexBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &varyingBuffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &V_ITa);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &V_W);
    clSetKernelArg(kernel, 4, sizeof(int), &V_ITa_ofs);
    clSetKernelArg(kernel, 5, sizeof(int), &V_W_ofs);
    clSetKernelArg(kernel, 6, sizeof(int), &offset);
    clSetKernelArg(kernel, 7, sizeof(int), &start);
    clSetKernelArg(kernel, 8, sizeof(int), &end);
    clSetKernelArg(kernel, 9, sizeof(int), &ipass);

    ciErrNum = clEnqueueNDRangeKernel(context->GetCommandQueue(),
                                      kernel, 1, NULL, globalWorkSize,
                                      NULL, 0, NULL, NULL);
    CL_CHECK_ERROR(ciErrNum, "vertex kernel 2 %d\n", ciErrNum);
}

void
OsdCLKernelDispatcher::ApplyLoopEdgeVerticesKernel(
    FarMesh<OsdVertex> * mesh, int offset, int level,
    int start, int end, void * clientdata) const {

    OsdCLComputeContext * context =
        static_cast<OsdCLComputeContext*>(clientdata);
    assert(context);

    cl_int ciErrNum;
    size_t globalWorkSize[1] = { end-start };
    cl_kernel kernel = context->GetKernelBundle()->GetLoopEdgeKernel();

    cl_mem vertexBuffer = context->GetCurrentVertexBuffer();
    cl_mem varyingBuffer = context->GetCurrentVaryingBuffer();
    cl_mem E_IT = context->GetTable(Table::E_IT)->GetDevicePtr();
    cl_mem E_W = context->GetTable(Table::E_W)->GetDevicePtr();
    int E_IT_ofs = context->GetTable(Table::E_IT)->GetMarker(level-1);
    int E_W_ofs = context->GetTable(Table::E_W)->GetMarker(level-1);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &vertexBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &varyingBuffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &E_IT);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &E_W);
    clSetKernelArg(kernel, 4, sizeof(int), &E_IT_ofs);
    clSetKernelArg(kernel, 5, sizeof(int), &E_W_ofs);
    clSetKernelArg(kernel, 6, sizeof(int), &offset);
    clSetKernelArg(kernel, 7, sizeof(int), &start);
    clSetKernelArg(kernel, 8, sizeof(int), &end);

    ciErrNum = clEnqueueNDRangeKernel(context->GetCommandQueue(),
                                      kernel, 1, NULL, globalWorkSize,
                                      NULL, 0, NULL, NULL);
    CL_CHECK_ERROR(ciErrNum, "edge kernel %d\n", ciErrNum);
}

void
OsdCLKernelDispatcher::ApplyLoopVertexVerticesKernelB(
    FarMesh<OsdVertex> * mesh, int offset, int level,
    int start, int end, void * clientdata) const {

    OsdCLComputeContext * context =
        static_cast<OsdCLComputeContext*>(clientdata);
    assert(context);

    cl_int ciErrNum;
    size_t globalWorkSize[1] = { end-start };
    cl_kernel kernel = context->GetKernelBundle()->GetLoopVertexKernelB();

    cl_mem vertexBuffer = context->GetCurrentVertexBuffer();
    cl_mem varyingBuffer = context->GetCurrentVaryingBuffer();
    cl_mem V_ITa = context->GetTable(Table::V_ITa)->GetDevicePtr();
    cl_mem V_IT = context->GetTable(Table::V_IT)->GetDevicePtr();
    cl_mem V_W = context->GetTable(Table::V_W)->GetDevicePtr();
    int V_ITa_ofs = context->GetTable(Table::V_ITa)->GetMarker(level-1);
    int V_IT_ofs = context->GetTable(Table::V_IT)->GetMarker(level-1);
    int V_W_ofs = context->GetTable(Table::V_W)->GetMarker(level-1);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &vertexBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &varyingBuffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &V_ITa);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &V_IT);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &V_W);
    clSetKernelArg(kernel, 5, sizeof(int), &V_ITa_ofs);
    clSetKernelArg(kernel, 6, sizeof(int), &V_IT_ofs);
    clSetKernelArg(kernel, 7, sizeof(int), &V_W_ofs);
    clSetKernelArg(kernel, 8, sizeof(int), &offset);
    clSetKernelArg(kernel, 9, sizeof(int), &start);
    clSetKernelArg(kernel, 10, sizeof(int), &end);

    ciErrNum = clEnqueueNDRangeKernel(context->GetCommandQueue(),
                                      kernel, 1, NULL, globalWorkSize,
                                      NULL, 0, NULL, NULL);
    CL_CHECK_ERROR(ciErrNum, "vertex kernel 1 %d\n", ciErrNum);
}

void
OsdCLKernelDispatcher::ApplyLoopVertexVerticesKernelA(
    FarMesh<OsdVertex> * mesh, int offset, bool pass,
    int level, int start, int end, void * clientdata) const {

    OsdCLComputeContext * context =
        static_cast<OsdCLComputeContext*>(clientdata);
    assert(context);

    cl_int ciErrNum;
    size_t globalWorkSize[1] = { end-start };
    int ipass = pass;
    cl_kernel kernel = context->GetKernelBundle()->GetLoopVertexKernelA();

    cl_mem vertexBuffer = context->GetCurrentVertexBuffer();
    cl_mem varyingBuffer = context->GetCurrentVaryingBuffer();
    cl_mem V_ITa = context->GetTable(Table::V_ITa)->GetDevicePtr();
    cl_mem V_W = context->GetTable(Table::V_W)->GetDevicePtr();
    int V_ITa_ofs = context->GetTable(Table::V_ITa)->GetMarker(level-1);
    int V_W_ofs = context->GetTable(Table::V_W)->GetMarker(level-1);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &vertexBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &varyingBuffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &V_ITa);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &V_W);
    clSetKernelArg(kernel, 4, sizeof(int), &V_ITa_ofs);
    clSetKernelArg(kernel, 5, sizeof(int), &V_W_ofs);
    clSetKernelArg(kernel, 6, sizeof(int), &offset);
    clSetKernelArg(kernel, 7, sizeof(int), &start);
    clSetKernelArg(kernel, 8, sizeof(int), &end);
    clSetKernelArg(kernel, 9, sizeof(int), &ipass);

    ciErrNum = clEnqueueNDRangeKernel(context->GetCommandQueue(),
                                      kernel, 1, NULL, globalWorkSize,
                                      NULL, 0, NULL, NULL);
    CL_CHECK_ERROR(ciErrNum, "vertex kernel 2 %d\n", ciErrNum);
}

void
OsdCLKernelDispatcher::ApplyVertexEdits(
    FarMesh<OsdVertex> *mesh, int offset, int level, void * clientdata) const {

    OsdCLComputeContext * context =
        static_cast<OsdCLComputeContext*>(clientdata);
    assert(context);

    cl_int ciErrNum;
    cl_mem vertexBuffer = context->GetCurrentVertexBuffer();

    int numEditTables = context->GetNumEditTables();
    for (int i=0; i < numEditTables; ++i) {

        const OsdCLHEditTable * edit = context->GetEditTable(i);
        assert(edit);

        const OsdCLTable * primvarIndices = edit->GetPrimvarIndices();
        const OsdCLTable * editValues = edit->GetEditValues();

        cl_mem indices = primvarIndices->GetDevicePtr();
        cl_mem values = editValues->GetDevicePtr();
        int indices_ofs = primvarIndices->GetMarker(level-1);
        int values_ofs = editValues->GetMarker(level-1);
        int numVertices = primvarIndices->GetNumElements(level-1);
        int primvarOffset = edit->GetPrimvarOffset();
        int primvarWidth = edit->GetPrimvarWidth();
        size_t globalWorkSize[1] = { numVertices };

        if (numVertices == 0) continue;

        if (edit->GetOperation() == FarVertexEdit::Add) {
            cl_kernel kernel = context->GetKernelBundle()->GetVertexEditAdd();

            clSetKernelArg(kernel, 0, sizeof(cl_mem), &vertexBuffer);
            clSetKernelArg(kernel, 1, sizeof(cl_mem), &indices);
            clSetKernelArg(kernel, 2, sizeof(cl_mem), &values);
            clSetKernelArg(kernel, 3, sizeof(int), &indices_ofs);
            clSetKernelArg(kernel, 4, sizeof(int), &values_ofs);
            clSetKernelArg(kernel, 5, sizeof(int), &primvarOffset);
            clSetKernelArg(kernel, 6, sizeof(int), &primvarWidth);

            ciErrNum = clEnqueueNDRangeKernel(context->GetCommandQueue(),
                                              kernel, 1, NULL, globalWorkSize,
                                              NULL, 0, NULL, NULL);

            CL_CHECK_ERROR(ciErrNum, "vertex edit %d %d\n", i, ciErrNum);

        } else if (edit->GetOperation() == FarVertexEdit::Set) {
             // XXXX TODO
        }
    }
}


}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
