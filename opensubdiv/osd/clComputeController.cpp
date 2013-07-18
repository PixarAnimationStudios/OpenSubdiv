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

#include "../osd/clComputeController.h"
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

OsdCLComputeController::OsdCLComputeController(cl_context clContext,
                                               cl_command_queue queue) :
    _clContext(clContext), _clQueue(queue) {
}

OsdCLComputeController::~OsdCLComputeController() {

    for (std::vector<OsdCLKernelBundle*>::iterator it = _kernelRegistry.begin();
        it != _kernelRegistry.end(); ++it) {
        delete *it;
    }
}

void
OsdCLComputeController::Synchronize() {

    clFinish(_clQueue);
}

OsdCLKernelBundle *
OsdCLComputeController::getKernelBundle(int numVertexElements,
                                        int numVaryingElements) {

    std::vector<OsdCLKernelBundle*>::iterator it =
        std::find_if(_kernelRegistry.begin(), _kernelRegistry.end(),
                     OsdCLKernelBundle::Match(numVertexElements,
                                              numVaryingElements));
    if (it != _kernelRegistry.end()) {
        return *it;
    } else {
        OsdCLKernelBundle *kernelBundle = new OsdCLKernelBundle();
        _kernelRegistry.push_back(kernelBundle);
        kernelBundle->Compile(_clContext,
                              numVertexElements,
                              numVaryingElements);
        return kernelBundle;
    }
}

void
OsdCLComputeController::ApplyBilinearFaceVerticesKernel(
    FarKernelBatch const &batch, void * clientdata) const {

    ApplyCatmarkFaceVerticesKernel(batch, clientdata);
}

void
OsdCLComputeController::ApplyBilinearEdgeVerticesKernel(
    FarKernelBatch const &batch, void * clientdata) const {

    OsdCLComputeContext * context =
        static_cast<OsdCLComputeContext*>(clientdata);
    assert(context);

    cl_int ciErrNum;
    size_t globalWorkSize[1] = { batch.GetEnd() - batch.GetStart() };
    cl_kernel kernel = context->GetKernelBundle()->GetBilinearEdgeKernel();

    cl_mem vertexBuffer = context->GetCurrentVertexBuffer();
    cl_mem varyingBuffer = context->GetCurrentVaryingBuffer();
    cl_mem E_IT = context->GetTable(FarSubdivisionTables<OsdVertex>::E_IT)->GetDevicePtr();

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &vertexBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &varyingBuffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &E_IT);
    clSetKernelArg(kernel, 3, sizeof(int), batch.GetVertexOffsetPtr());
    clSetKernelArg(kernel, 4, sizeof(int), batch.GetTableOffsetPtr());
    clSetKernelArg(kernel, 5, sizeof(int), batch.GetStartPtr());
    clSetKernelArg(kernel, 6, sizeof(int), batch.GetEndPtr());
    ciErrNum = clEnqueueNDRangeKernel(context->GetCommandQueue(),
                                      kernel, 1, NULL, globalWorkSize,
                                      NULL, 0, NULL, NULL);
    CL_CHECK_ERROR(ciErrNum, "bilinear edge kernel %d\n", ciErrNum);
}

void
OsdCLComputeController::ApplyBilinearVertexVerticesKernel(
    FarKernelBatch const &batch, void * clientdata) const {

    OsdCLComputeContext * context =
        static_cast<OsdCLComputeContext*>(clientdata);
    assert(context);

    cl_int ciErrNum;
    size_t globalWorkSize[1] = { batch.GetEnd() - batch.GetStart() };
    cl_kernel kernel = context->GetKernelBundle()->GetBilinearVertexKernel();

    cl_mem vertexBuffer = context->GetCurrentVertexBuffer();
    cl_mem varyingBuffer = context->GetCurrentVaryingBuffer();
    cl_mem V_ITa = context->GetTable(FarSubdivisionTables<OsdVertex>::V_ITa)->GetDevicePtr();

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &vertexBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &varyingBuffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &V_ITa);
    clSetKernelArg(kernel, 3, sizeof(int), batch.GetVertexOffsetPtr());
    clSetKernelArg(kernel, 4, sizeof(int), batch.GetTableOffsetPtr());
    clSetKernelArg(kernel, 5, sizeof(int), batch.GetStartPtr());
    clSetKernelArg(kernel, 6, sizeof(int), batch.GetEndPtr());
    ciErrNum = clEnqueueNDRangeKernel(context->GetCommandQueue(),
                                      kernel, 1, NULL, globalWorkSize,
                                      NULL, 0, NULL, NULL);
    CL_CHECK_ERROR(ciErrNum, "bilinear vertex kernel 1 %d\n", ciErrNum);
}

void
OsdCLComputeController::ApplyCatmarkFaceVerticesKernel(
    FarKernelBatch const &batch, void * clientdata) const {

    OsdCLComputeContext * context =
        static_cast<OsdCLComputeContext*>(clientdata);
    assert(context);

    cl_int ciErrNum;
    size_t globalWorkSize[1] = { batch.GetEnd() - batch.GetStart() };
    cl_kernel kernel = context->GetKernelBundle()->GetCatmarkFaceKernel();

    cl_mem vertexBuffer = context->GetCurrentVertexBuffer();
    cl_mem varyingBuffer = context->GetCurrentVaryingBuffer();
    cl_mem F_IT = context->GetTable(FarSubdivisionTables<OsdVertex>::F_IT)->GetDevicePtr();
    cl_mem F_ITa = context->GetTable(FarSubdivisionTables<OsdVertex>::F_ITa)->GetDevicePtr();

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &vertexBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &varyingBuffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &F_IT);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &F_ITa);
    clSetKernelArg(kernel, 4, sizeof(int), batch.GetVertexOffsetPtr());
    clSetKernelArg(kernel, 5, sizeof(int), batch.GetTableOffsetPtr());
    clSetKernelArg(kernel, 6, sizeof(int), batch.GetStartPtr());
    clSetKernelArg(kernel, 7, sizeof(int), batch.GetEndPtr());

    ciErrNum = clEnqueueNDRangeKernel(context->GetCommandQueue(),
                                      kernel, 1, NULL, globalWorkSize,
                                      NULL, 0, NULL, NULL);
    CL_CHECK_ERROR(ciErrNum, "face kernel %d\n", ciErrNum);
}

void
OsdCLComputeController::ApplyCatmarkEdgeVerticesKernel(
    FarKernelBatch const &batch, void * clientdata) const {

    OsdCLComputeContext * context =
        static_cast<OsdCLComputeContext*>(clientdata);
    assert(context);

    cl_int ciErrNum;
    size_t globalWorkSize[1] = { batch.GetEnd() - batch.GetStart() };
    cl_kernel kernel = context->GetKernelBundle()->GetCatmarkEdgeKernel();

    cl_mem vertexBuffer = context->GetCurrentVertexBuffer();
    cl_mem varyingBuffer = context->GetCurrentVaryingBuffer();
    cl_mem E_IT = context->GetTable(FarSubdivisionTables<OsdVertex>::E_IT)->GetDevicePtr();
    cl_mem E_W = context->GetTable(FarSubdivisionTables<OsdVertex>::E_W)->GetDevicePtr();

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &vertexBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &varyingBuffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &E_IT);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &E_W);
    clSetKernelArg(kernel, 4, sizeof(int), batch.GetVertexOffsetPtr());
    clSetKernelArg(kernel, 5, sizeof(int), batch.GetTableOffsetPtr());
    clSetKernelArg(kernel, 6, sizeof(int), batch.GetStartPtr());
    clSetKernelArg(kernel, 7, sizeof(int), batch.GetEndPtr());

    ciErrNum = clEnqueueNDRangeKernel(context->GetCommandQueue(),
                                      kernel, 1, NULL, globalWorkSize,
                                      NULL, 0, NULL, NULL);
    CL_CHECK_ERROR(ciErrNum, "edge kernel %d\n", ciErrNum);
}

void
OsdCLComputeController::ApplyCatmarkVertexVerticesKernelB(
    FarKernelBatch const &batch, void * clientdata) const {

    OsdCLComputeContext * context =
        static_cast<OsdCLComputeContext*>(clientdata);
    assert(context);

    cl_int ciErrNum;
    size_t globalWorkSize[1] = { batch.GetEnd() - batch.GetStart() };
    cl_kernel kernel = context->GetKernelBundle()->GetCatmarkVertexKernelB();

    cl_mem vertexBuffer = context->GetCurrentVertexBuffer();
    cl_mem varyingBuffer = context->GetCurrentVaryingBuffer();
    cl_mem V_ITa = context->GetTable(FarSubdivisionTables<OsdVertex>::V_ITa)->GetDevicePtr();
    cl_mem V_IT = context->GetTable(FarSubdivisionTables<OsdVertex>::V_IT)->GetDevicePtr();
    cl_mem V_W = context->GetTable(FarSubdivisionTables<OsdVertex>::V_W)->GetDevicePtr();

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &vertexBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &varyingBuffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &V_ITa);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &V_IT);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &V_W);
    clSetKernelArg(kernel, 5, sizeof(int), batch.GetVertexOffsetPtr());
    clSetKernelArg(kernel, 6, sizeof(int), batch.GetTableOffsetPtr());
    clSetKernelArg(kernel, 7, sizeof(int), batch.GetStartPtr());
    clSetKernelArg(kernel, 8, sizeof(int), batch.GetEndPtr());

    ciErrNum = clEnqueueNDRangeKernel(context->GetCommandQueue(),
                                      kernel, 1, NULL, globalWorkSize,
                                      NULL, 0, NULL, NULL);
    CL_CHECK_ERROR(ciErrNum, "vertex kernel 1 %d\n", ciErrNum);
}

void
OsdCLComputeController::ApplyCatmarkVertexVerticesKernelA1(
    FarKernelBatch const &batch, void * clientdata) const {

    OsdCLComputeContext * context =
        static_cast<OsdCLComputeContext*>(clientdata);
    assert(context);

    cl_int ciErrNum;
    size_t globalWorkSize[1] = { batch.GetEnd() - batch.GetStart() };
    int ipass = false;
    cl_kernel kernel = context->GetKernelBundle()->GetCatmarkVertexKernelA();

    cl_mem vertexBuffer = context->GetCurrentVertexBuffer();
    cl_mem varyingBuffer = context->GetCurrentVaryingBuffer();
    cl_mem V_ITa = context->GetTable(FarSubdivisionTables<OsdVertex>::V_ITa)->GetDevicePtr();
    cl_mem V_W = context->GetTable(FarSubdivisionTables<OsdVertex>::V_W)->GetDevicePtr();

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &vertexBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &varyingBuffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &V_ITa);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &V_W);
    clSetKernelArg(kernel, 4, sizeof(int), batch.GetVertexOffsetPtr());
    clSetKernelArg(kernel, 5, sizeof(int), batch.GetTableOffsetPtr());
    clSetKernelArg(kernel, 6, sizeof(int), batch.GetStartPtr());
    clSetKernelArg(kernel, 7, sizeof(int), batch.GetEndPtr());
    clSetKernelArg(kernel, 8, sizeof(int), &ipass);

    ciErrNum = clEnqueueNDRangeKernel(context->GetCommandQueue(),
                                      kernel, 1, NULL, globalWorkSize,
                                      NULL, 0, NULL, NULL);
    CL_CHECK_ERROR(ciErrNum, "vertex kernel 2 %d\n", ciErrNum);
}

void
OsdCLComputeController::ApplyCatmarkVertexVerticesKernelA2(
    FarKernelBatch const &batch, void * clientdata) const {

    OsdCLComputeContext * context =
        static_cast<OsdCLComputeContext*>(clientdata);
    assert(context);

    cl_int ciErrNum;
    size_t globalWorkSize[1] = { batch.GetEnd() - batch.GetStart() };
    int ipass = true;
    cl_kernel kernel = context->GetKernelBundle()->GetCatmarkVertexKernelA();

    cl_mem vertexBuffer = context->GetCurrentVertexBuffer();
    cl_mem varyingBuffer = context->GetCurrentVaryingBuffer();
    cl_mem V_ITa = context->GetTable(FarSubdivisionTables<OsdVertex>::V_ITa)->GetDevicePtr();
    cl_mem V_W = context->GetTable(FarSubdivisionTables<OsdVertex>::V_W)->GetDevicePtr();

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &vertexBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &varyingBuffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &V_ITa);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &V_W);
    clSetKernelArg(kernel, 4, sizeof(int), batch.GetVertexOffsetPtr());
    clSetKernelArg(kernel, 5, sizeof(int), batch.GetTableOffsetPtr());
    clSetKernelArg(kernel, 6, sizeof(int), batch.GetStartPtr());
    clSetKernelArg(kernel, 7, sizeof(int), batch.GetEndPtr());
    clSetKernelArg(kernel, 8, sizeof(int), &ipass);

    ciErrNum = clEnqueueNDRangeKernel(context->GetCommandQueue(),
                                      kernel, 1, NULL, globalWorkSize,
                                      NULL, 0, NULL, NULL);
    CL_CHECK_ERROR(ciErrNum, "vertex kernel 2 %d\n", ciErrNum);
}

void
OsdCLComputeController::ApplyLoopEdgeVerticesKernel(
    FarKernelBatch const &batch, void * clientdata) const {

    OsdCLComputeContext * context =
        static_cast<OsdCLComputeContext*>(clientdata);
    assert(context);

    cl_int ciErrNum;
    size_t globalWorkSize[1] = { batch.GetEnd() - batch.GetStart() };
    cl_kernel kernel = context->GetKernelBundle()->GetLoopEdgeKernel();

    cl_mem vertexBuffer = context->GetCurrentVertexBuffer();
    cl_mem varyingBuffer = context->GetCurrentVaryingBuffer();
    cl_mem E_IT = context->GetTable(FarSubdivisionTables<OsdVertex>::E_IT)->GetDevicePtr();
    cl_mem E_W = context->GetTable(FarSubdivisionTables<OsdVertex>::E_W)->GetDevicePtr();

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &vertexBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &varyingBuffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &E_IT);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &E_W);
    clSetKernelArg(kernel, 4, sizeof(int), batch.GetVertexOffsetPtr());
    clSetKernelArg(kernel, 5, sizeof(int), batch.GetTableOffsetPtr());
    clSetKernelArg(kernel, 6, sizeof(int), batch.GetStartPtr());
    clSetKernelArg(kernel, 7, sizeof(int), batch.GetEndPtr());

    ciErrNum = clEnqueueNDRangeKernel(context->GetCommandQueue(),
                                      kernel, 1, NULL, globalWorkSize,
                                      NULL, 0, NULL, NULL);
    CL_CHECK_ERROR(ciErrNum, "edge kernel %d\n", ciErrNum);
}

void
OsdCLComputeController::ApplyLoopVertexVerticesKernelB(
    FarKernelBatch const &batch, void * clientdata) const {

    OsdCLComputeContext * context =
        static_cast<OsdCLComputeContext*>(clientdata);
    assert(context);

    cl_int ciErrNum;
    size_t globalWorkSize[1] = { batch.GetEnd() - batch.GetStart() };
    cl_kernel kernel = context->GetKernelBundle()->GetLoopVertexKernelB();

    cl_mem vertexBuffer = context->GetCurrentVertexBuffer();
    cl_mem varyingBuffer = context->GetCurrentVaryingBuffer();
    cl_mem V_ITa = context->GetTable(FarSubdivisionTables<OsdVertex>::V_ITa)->GetDevicePtr();
    cl_mem V_IT = context->GetTable(FarSubdivisionTables<OsdVertex>::V_IT)->GetDevicePtr();
    cl_mem V_W = context->GetTable(FarSubdivisionTables<OsdVertex>::V_W)->GetDevicePtr();

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &vertexBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &varyingBuffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &V_ITa);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &V_IT);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &V_W);
    clSetKernelArg(kernel, 5, sizeof(int), batch.GetVertexOffsetPtr());
    clSetKernelArg(kernel, 6, sizeof(int), batch.GetTableOffsetPtr());
    clSetKernelArg(kernel, 7, sizeof(int), batch.GetStartPtr());
    clSetKernelArg(kernel, 8, sizeof(int), batch.GetEndPtr());

    ciErrNum = clEnqueueNDRangeKernel(context->GetCommandQueue(),
                                      kernel, 1, NULL, globalWorkSize,
                                      NULL, 0, NULL, NULL);
    CL_CHECK_ERROR(ciErrNum, "vertex kernel 1 %d\n", ciErrNum);
}

void
OsdCLComputeController::ApplyLoopVertexVerticesKernelA1(
    FarKernelBatch const &batch, void * clientdata) const {

    OsdCLComputeContext * context =
        static_cast<OsdCLComputeContext*>(clientdata);
    assert(context);

    cl_int ciErrNum;
    size_t globalWorkSize[1] = { batch.GetEnd() - batch.GetStart() };
    int ipass = false;
    cl_kernel kernel = context->GetKernelBundle()->GetLoopVertexKernelA();

    cl_mem vertexBuffer = context->GetCurrentVertexBuffer();
    cl_mem varyingBuffer = context->GetCurrentVaryingBuffer();
    cl_mem V_ITa = context->GetTable(FarSubdivisionTables<OsdVertex>::V_ITa)->GetDevicePtr();
    cl_mem V_W = context->GetTable(FarSubdivisionTables<OsdVertex>::V_W)->GetDevicePtr();

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &vertexBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &varyingBuffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &V_ITa);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &V_W);
    clSetKernelArg(kernel, 4, sizeof(int), batch.GetVertexOffsetPtr());
    clSetKernelArg(kernel, 5, sizeof(int), batch.GetTableOffsetPtr());
    clSetKernelArg(kernel, 6, sizeof(int), batch.GetStartPtr());
    clSetKernelArg(kernel, 7, sizeof(int), batch.GetEndPtr());
    clSetKernelArg(kernel, 8, sizeof(int), &ipass);

    ciErrNum = clEnqueueNDRangeKernel(context->GetCommandQueue(),
                                      kernel, 1, NULL, globalWorkSize,
                                      NULL, 0, NULL, NULL);
    CL_CHECK_ERROR(ciErrNum, "vertex kernel 2 %d\n", ciErrNum);
}

void
OsdCLComputeController::ApplyLoopVertexVerticesKernelA2(
    FarKernelBatch const &batch, void * clientdata) const {

    OsdCLComputeContext * context =
        static_cast<OsdCLComputeContext*>(clientdata);
    assert(context);

    cl_int ciErrNum;
    size_t globalWorkSize[1] = { batch.GetEnd() - batch.GetStart() };
    int ipass = true;
    cl_kernel kernel = context->GetKernelBundle()->GetLoopVertexKernelA();

    cl_mem vertexBuffer = context->GetCurrentVertexBuffer();
    cl_mem varyingBuffer = context->GetCurrentVaryingBuffer();
    cl_mem V_ITa = context->GetTable(FarSubdivisionTables<OsdVertex>::V_ITa)->GetDevicePtr();
    cl_mem V_W = context->GetTable(FarSubdivisionTables<OsdVertex>::V_W)->GetDevicePtr();

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &vertexBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &varyingBuffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &V_ITa);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &V_W);
    clSetKernelArg(kernel, 4, sizeof(int), batch.GetVertexOffsetPtr());
    clSetKernelArg(kernel, 5, sizeof(int), batch.GetTableOffsetPtr());
    clSetKernelArg(kernel, 6, sizeof(int), batch.GetStartPtr());
    clSetKernelArg(kernel, 7, sizeof(int), batch.GetEndPtr());
    clSetKernelArg(kernel, 8, sizeof(int), &ipass);

    ciErrNum = clEnqueueNDRangeKernel(context->GetCommandQueue(),
                                      kernel, 1, NULL, globalWorkSize,
                                      NULL, 0, NULL, NULL);
    CL_CHECK_ERROR(ciErrNum, "vertex kernel 2 %d\n", ciErrNum);
}

void
OsdCLComputeController::ApplyVertexEdits(
    FarKernelBatch const &batch, void * clientdata) const {

    OsdCLComputeContext * context =
        static_cast<OsdCLComputeContext*>(clientdata);
    assert(context);

    cl_int ciErrNum;
    size_t globalWorkSize[1] = { batch.GetEnd() - batch.GetStart() };
    cl_mem vertexBuffer = context->GetCurrentVertexBuffer();

    const OsdCLHEditTable * edit = context->GetEditTable(batch.GetTableIndex());
    assert(edit);

    const OsdCLTable * primvarIndices = edit->GetPrimvarIndices();
    const OsdCLTable * editValues = edit->GetEditValues();

    cl_mem indices = primvarIndices->GetDevicePtr();
    cl_mem values = editValues->GetDevicePtr();
    int primvarOffset = edit->GetPrimvarOffset();
    int primvarWidth = edit->GetPrimvarWidth();

    if (edit->GetOperation() == FarVertexEdit::Add) {
        cl_kernel kernel = context->GetKernelBundle()->GetVertexEditAdd();

        clSetKernelArg(kernel, 0, sizeof(cl_mem), &vertexBuffer);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &indices);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &values);
        clSetKernelArg(kernel, 3, sizeof(int), &primvarOffset);
        clSetKernelArg(kernel, 4, sizeof(int), &primvarWidth);
        clSetKernelArg(kernel, 5, sizeof(int), batch.GetVertexOffsetPtr());
        clSetKernelArg(kernel, 6, sizeof(int), batch.GetTableOffsetPtr());
        clSetKernelArg(kernel, 7, sizeof(int), batch.GetStartPtr());
        clSetKernelArg(kernel, 8, sizeof(int), batch.GetEndPtr());

        ciErrNum = clEnqueueNDRangeKernel(context->GetCommandQueue(),
                                          kernel, 1, NULL, globalWorkSize,
                                          NULL, 0, NULL, NULL);
        
        CL_CHECK_ERROR(ciErrNum, "vertex edit %d %d\n", batch.GetTableIndex(), ciErrNum);

    } else if (edit->GetOperation() == FarVertexEdit::Set) {
        // XXXX TODO
    }
}

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv

