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

#include "../osd/clComputeController.h"
#include "../osd/clComputeContext.h"
#include "../osd/clKernelBundle.h"
#include "../osd/error.h"
#include "../osd/opencl.h"

#if defined(_WIN32)
    #include <windows.h>
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
OsdCLComputeController::getKernelBundle(
    OsdVertexBufferDescriptor const &vertexDesc,
    OsdVertexBufferDescriptor const &varyingDesc) {

    std::vector<OsdCLKernelBundle*>::iterator it =
        std::find_if(_kernelRegistry.begin(), _kernelRegistry.end(),
                     OsdCLKernelBundle::Match(vertexDesc,
                                              varyingDesc));

    if (it != _kernelRegistry.end()) {
        return *it;
    } else {
        OsdCLKernelBundle *kernelBundle = new OsdCLKernelBundle();
        _kernelRegistry.push_back(kernelBundle);
        kernelBundle->Compile(_clContext,
                              vertexDesc,
                              varyingDesc);
        return kernelBundle;
    }
}

void
OsdCLComputeController::ApplyBilinearFaceVerticesKernel(
    FarKernelBatch const &batch, OsdCLComputeContext const *context) const {

    ApplyCatmarkFaceVerticesKernel(batch, context);
}

void
OsdCLComputeController::ApplyBilinearEdgeVerticesKernel(
    FarKernelBatch const &batch, OsdCLComputeContext const *context) const {

    assert(context);

    cl_int ciErrNum;
    size_t globalWorkSize[1] = { (size_t)(batch.GetEnd() - batch.GetStart()) };
    cl_kernel kernel = _currentBindState.kernelBundle->GetBilinearEdgeKernel();

    cl_mem E_IT = context->GetTable(FarSubdivisionTables::E_IT)->GetDevicePtr();

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &_currentBindState.vertexBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &_currentBindState.varyingBuffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &E_IT);
    clSetKernelArg(kernel, 3, sizeof(int), &_currentBindState.vertexDesc.offset);
    clSetKernelArg(kernel, 4, sizeof(int), &_currentBindState.varyingDesc.offset);
    clSetKernelArg(kernel, 5, sizeof(int), batch.GetVertexOffsetPtr());
    clSetKernelArg(kernel, 6, sizeof(int), batch.GetTableOffsetPtr());
    clSetKernelArg(kernel, 7, sizeof(int), batch.GetStartPtr());
    clSetKernelArg(kernel, 8, sizeof(int), batch.GetEndPtr());
    ciErrNum = clEnqueueNDRangeKernel(_clQueue,
                                      kernel, 1, NULL, globalWorkSize,
                                      NULL, 0, NULL, NULL);
    CL_CHECK_ERROR(ciErrNum, "bilinear edge kernel %d\n", ciErrNum);
}

void
OsdCLComputeController::ApplyBilinearVertexVerticesKernel(
    FarKernelBatch const &batch, OsdCLComputeContext const *context) const {

    assert(context);

    cl_int ciErrNum;
    size_t globalWorkSize[1] = { (size_t)(batch.GetEnd() - batch.GetStart()) };
    cl_kernel kernel = _currentBindState.kernelBundle->GetBilinearVertexKernel();

    cl_mem V_ITa = context->GetTable(FarSubdivisionTables::V_ITa)->GetDevicePtr();

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &_currentBindState.vertexBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &_currentBindState.varyingBuffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &V_ITa);
    clSetKernelArg(kernel, 3, sizeof(int), &_currentBindState.vertexDesc.offset);
    clSetKernelArg(kernel, 4, sizeof(int), &_currentBindState.varyingDesc.offset);
    clSetKernelArg(kernel, 5, sizeof(int), batch.GetVertexOffsetPtr());
    clSetKernelArg(kernel, 6, sizeof(int), batch.GetTableOffsetPtr());
    clSetKernelArg(kernel, 7, sizeof(int), batch.GetStartPtr());
    clSetKernelArg(kernel, 8, sizeof(int), batch.GetEndPtr());
    ciErrNum = clEnqueueNDRangeKernel(_clQueue,
                                      kernel, 1, NULL, globalWorkSize,
                                      NULL, 0, NULL, NULL);
    CL_CHECK_ERROR(ciErrNum, "bilinear vertex kernel 1 %d\n", ciErrNum);
}

void
OsdCLComputeController::ApplyCatmarkFaceVerticesKernel(
    FarKernelBatch const &batch, OsdCLComputeContext const *context) const {

    assert(context);

    cl_int ciErrNum;
    size_t globalWorkSize[1] = { (size_t)(batch.GetEnd() - batch.GetStart()) };
    cl_kernel kernel = _currentBindState.kernelBundle->GetCatmarkFaceKernel();

    cl_mem F_IT = context->GetTable(FarSubdivisionTables::F_IT)->GetDevicePtr();
    cl_mem F_ITa = context->GetTable(FarSubdivisionTables::F_ITa)->GetDevicePtr();

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &_currentBindState.vertexBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &_currentBindState.varyingBuffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &F_IT);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &F_ITa);
    clSetKernelArg(kernel, 4, sizeof(int), &_currentBindState.vertexDesc.offset);
    clSetKernelArg(kernel, 5, sizeof(int), &_currentBindState.varyingDesc.offset);
    clSetKernelArg(kernel, 6, sizeof(int), batch.GetVertexOffsetPtr());
    clSetKernelArg(kernel, 7, sizeof(int), batch.GetTableOffsetPtr());
    clSetKernelArg(kernel, 8, sizeof(int), batch.GetStartPtr());
    clSetKernelArg(kernel, 9, sizeof(int), batch.GetEndPtr());

    ciErrNum = clEnqueueNDRangeKernel(_clQueue,
                                      kernel, 1, NULL, globalWorkSize,
                                      NULL, 0, NULL, NULL);
    CL_CHECK_ERROR(ciErrNum, "face kernel %d\n", ciErrNum);
}

void
OsdCLComputeController::ApplyCatmarkQuadFaceVerticesKernel(
    FarKernelBatch const &batch, OsdCLComputeContext const *context) const {

    assert(context);

    cl_int ciErrNum;
    size_t globalWorkSize[1] = { (size_t)(batch.GetEnd() - batch.GetStart()) };
    cl_kernel kernel = _currentBindState.kernelBundle->GetCatmarkQuadFaceKernel();

    cl_mem F_IT = context->GetTable(FarSubdivisionTables::F_IT)->GetDevicePtr();

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &_currentBindState.vertexBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &_currentBindState.varyingBuffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &F_IT);
    clSetKernelArg(kernel, 3, sizeof(int), &_currentBindState.vertexDesc.offset);
    clSetKernelArg(kernel, 4, sizeof(int), &_currentBindState.varyingDesc.offset);
    clSetKernelArg(kernel, 5, sizeof(int), batch.GetVertexOffsetPtr());
    clSetKernelArg(kernel, 6, sizeof(int), batch.GetTableOffsetPtr());
    clSetKernelArg(kernel, 7, sizeof(int), batch.GetStartPtr());
    clSetKernelArg(kernel, 8, sizeof(int), batch.GetEndPtr());

    ciErrNum = clEnqueueNDRangeKernel(_clQueue,
                                      kernel, 1, NULL, globalWorkSize,
                                      NULL, 0, NULL, NULL);
    CL_CHECK_ERROR(ciErrNum, "quad face kernel %d\n", ciErrNum);
}

void
OsdCLComputeController::ApplyCatmarkTriQuadFaceVerticesKernel(
    FarKernelBatch const &batch, OsdCLComputeContext const *context) const {

    assert(context);

    cl_int ciErrNum;
    size_t globalWorkSize[1] = { (size_t)(batch.GetEnd() - batch.GetStart()) };
    cl_kernel kernel = _currentBindState.kernelBundle->GetCatmarkTriQuadFaceKernel();

    cl_mem F_IT = context->GetTable(FarSubdivisionTables::F_IT)->GetDevicePtr();

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &_currentBindState.vertexBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &_currentBindState.varyingBuffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &F_IT);
    clSetKernelArg(kernel, 3, sizeof(int), &_currentBindState.vertexDesc.offset);
    clSetKernelArg(kernel, 4, sizeof(int), &_currentBindState.varyingDesc.offset);
    clSetKernelArg(kernel, 5, sizeof(int), batch.GetVertexOffsetPtr());
    clSetKernelArg(kernel, 6, sizeof(int), batch.GetTableOffsetPtr());
    clSetKernelArg(kernel, 7, sizeof(int), batch.GetStartPtr());
    clSetKernelArg(kernel, 8, sizeof(int), batch.GetEndPtr());

    ciErrNum = clEnqueueNDRangeKernel(_clQueue,
                                      kernel, 1, NULL, globalWorkSize,
                                      NULL, 0, NULL, NULL);
    CL_CHECK_ERROR(ciErrNum, "tri quad face kernel %d\n", ciErrNum);
}

void
OsdCLComputeController::ApplyCatmarkEdgeVerticesKernel(
    FarKernelBatch const &batch, OsdCLComputeContext const *context) const {

    assert(context);

    cl_int ciErrNum;
    size_t globalWorkSize[1] = { (size_t)(batch.GetEnd() - batch.GetStart()) };
    cl_kernel kernel = _currentBindState.kernelBundle->GetCatmarkEdgeKernel();

    cl_mem E_IT = context->GetTable(FarSubdivisionTables::E_IT)->GetDevicePtr();
    cl_mem E_W = context->GetTable(FarSubdivisionTables::E_W)->GetDevicePtr();

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &_currentBindState.vertexBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &_currentBindState.varyingBuffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &E_IT);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &E_W);
    clSetKernelArg(kernel, 4, sizeof(int), &_currentBindState.vertexDesc.offset);
    clSetKernelArg(kernel, 5, sizeof(int), &_currentBindState.varyingDesc.offset);
    clSetKernelArg(kernel, 6, sizeof(int), batch.GetVertexOffsetPtr());
    clSetKernelArg(kernel, 7, sizeof(int), batch.GetTableOffsetPtr());
    clSetKernelArg(kernel, 8, sizeof(int), batch.GetStartPtr());
    clSetKernelArg(kernel, 9, sizeof(int), batch.GetEndPtr());

    ciErrNum = clEnqueueNDRangeKernel(_clQueue,
                                      kernel, 1, NULL, globalWorkSize,
                                      NULL, 0, NULL, NULL);
    CL_CHECK_ERROR(ciErrNum, "edge kernel %d\n", ciErrNum);
}

void
OsdCLComputeController::ApplyCatmarkRestrictedEdgeVerticesKernel(
    FarKernelBatch const &batch, OsdCLComputeContext const *context) const {

    assert(context);

    cl_int ciErrNum;
    size_t globalWorkSize[1] = { (size_t)(batch.GetEnd() - batch.GetStart()) };
    cl_kernel kernel = _currentBindState.kernelBundle->GetCatmarkRestrictedEdgeKernel();

    cl_mem E_IT = context->GetTable(FarSubdivisionTables::E_IT)->GetDevicePtr();

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &_currentBindState.vertexBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &_currentBindState.varyingBuffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &E_IT);
    clSetKernelArg(kernel, 3, sizeof(int), &_currentBindState.vertexDesc.offset);
    clSetKernelArg(kernel, 4, sizeof(int), &_currentBindState.varyingDesc.offset);
    clSetKernelArg(kernel, 5, sizeof(int), batch.GetVertexOffsetPtr());
    clSetKernelArg(kernel, 6, sizeof(int), batch.GetTableOffsetPtr());
    clSetKernelArg(kernel, 7, sizeof(int), batch.GetStartPtr());
    clSetKernelArg(kernel, 8, sizeof(int), batch.GetEndPtr());

    ciErrNum = clEnqueueNDRangeKernel(_clQueue,
                                      kernel, 1, NULL, globalWorkSize,
                                      NULL, 0, NULL, NULL);
    CL_CHECK_ERROR(ciErrNum, "restricted edge kernel %d\n", ciErrNum);
}

void
OsdCLComputeController::ApplyCatmarkVertexVerticesKernelB(
    FarKernelBatch const &batch, OsdCLComputeContext const *context) const {

    assert(context);

    cl_int ciErrNum;
    size_t globalWorkSize[1] = { (size_t)(batch.GetEnd() - batch.GetStart()) };
    cl_kernel kernel = _currentBindState.kernelBundle->GetCatmarkVertexKernelB();

    cl_mem V_ITa = context->GetTable(FarSubdivisionTables::V_ITa)->GetDevicePtr();
    cl_mem V_IT = context->GetTable(FarSubdivisionTables::V_IT)->GetDevicePtr();
    cl_mem V_W = context->GetTable(FarSubdivisionTables::V_W)->GetDevicePtr();

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &_currentBindState.vertexBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &_currentBindState.varyingBuffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &V_ITa);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &V_IT);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &V_W);
    clSetKernelArg(kernel, 5, sizeof(int), &_currentBindState.vertexDesc.offset);
    clSetKernelArg(kernel, 6, sizeof(int), &_currentBindState.varyingDesc.offset);
    clSetKernelArg(kernel, 7, sizeof(int), batch.GetVertexOffsetPtr());
    clSetKernelArg(kernel, 8, sizeof(int), batch.GetTableOffsetPtr());
    clSetKernelArg(kernel, 9, sizeof(int), batch.GetStartPtr());
    clSetKernelArg(kernel, 10, sizeof(int), batch.GetEndPtr());

    ciErrNum = clEnqueueNDRangeKernel(_clQueue,
                                      kernel, 1, NULL, globalWorkSize,
                                      NULL, 0, NULL, NULL);
    CL_CHECK_ERROR(ciErrNum, "vertex kernel B %d\n", ciErrNum);
}

void
OsdCLComputeController::ApplyCatmarkVertexVerticesKernelA1(
    FarKernelBatch const &batch, OsdCLComputeContext const *context) const {

    assert(context);

    cl_int ciErrNum;
    size_t globalWorkSize[1] = { (size_t)(batch.GetEnd() - batch.GetStart()) };
    int ipass = false;
    cl_kernel kernel = _currentBindState.kernelBundle->GetCatmarkVertexKernelA();

    cl_mem V_ITa = context->GetTable(FarSubdivisionTables::V_ITa)->GetDevicePtr();
    cl_mem V_W = context->GetTable(FarSubdivisionTables::V_W)->GetDevicePtr();

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &_currentBindState.vertexBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &_currentBindState.varyingBuffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &V_ITa);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &V_W);
    clSetKernelArg(kernel, 4, sizeof(int), &_currentBindState.vertexDesc.offset);
    clSetKernelArg(kernel, 5, sizeof(int), &_currentBindState.varyingDesc.offset);
    clSetKernelArg(kernel, 6, sizeof(int), batch.GetVertexOffsetPtr());
    clSetKernelArg(kernel, 7, sizeof(int), batch.GetTableOffsetPtr());
    clSetKernelArg(kernel, 8, sizeof(int), batch.GetStartPtr());
    clSetKernelArg(kernel, 9, sizeof(int), batch.GetEndPtr());
    clSetKernelArg(kernel, 10, sizeof(int), &ipass);

    ciErrNum = clEnqueueNDRangeKernel(_clQueue,
                                      kernel, 1, NULL, globalWorkSize,
                                      NULL, 0, NULL, NULL);
    CL_CHECK_ERROR(ciErrNum, "vertex kernel A1 %d\n", ciErrNum);
}

void
OsdCLComputeController::ApplyCatmarkVertexVerticesKernelA2(
    FarKernelBatch const &batch, OsdCLComputeContext const *context) const {

    assert(context);

    cl_int ciErrNum;
    size_t globalWorkSize[1] = { (size_t)(batch.GetEnd() - batch.GetStart()) };
    int ipass = true;
    cl_kernel kernel = _currentBindState.kernelBundle->GetCatmarkVertexKernelA();

    cl_mem V_ITa = context->GetTable(FarSubdivisionTables::V_ITa)->GetDevicePtr();
    cl_mem V_W = context->GetTable(FarSubdivisionTables::V_W)->GetDevicePtr();

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &_currentBindState.vertexBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &_currentBindState.varyingBuffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &V_ITa);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &V_W);
    clSetKernelArg(kernel, 4, sizeof(int), &_currentBindState.vertexDesc.offset);
    clSetKernelArg(kernel, 5, sizeof(int), &_currentBindState.varyingDesc.offset);
    clSetKernelArg(kernel, 6, sizeof(int), batch.GetVertexOffsetPtr());
    clSetKernelArg(kernel, 7, sizeof(int), batch.GetTableOffsetPtr());
    clSetKernelArg(kernel, 8, sizeof(int), batch.GetStartPtr());
    clSetKernelArg(kernel, 9, sizeof(int), batch.GetEndPtr());
    clSetKernelArg(kernel, 10, sizeof(int), &ipass);

    ciErrNum = clEnqueueNDRangeKernel(_clQueue,
                                      kernel, 1, NULL, globalWorkSize,
                                      NULL, 0, NULL, NULL);
    CL_CHECK_ERROR(ciErrNum, "vertex kernel A2 %d\n", ciErrNum);
}

void
OsdCLComputeController::ApplyCatmarkRestrictedVertexVerticesKernelB1(
    FarKernelBatch const &batch, OsdCLComputeContext const *context) const {

    assert(context);

    cl_int ciErrNum;
    size_t globalWorkSize[1] = { (size_t)(batch.GetEnd() - batch.GetStart()) };
    cl_kernel kernel = _currentBindState.kernelBundle->GetCatmarkRestrictedVertexKernelB1();

    cl_mem V_ITa = context->GetTable(FarSubdivisionTables::V_ITa)->GetDevicePtr();
    cl_mem V_IT = context->GetTable(FarSubdivisionTables::V_IT)->GetDevicePtr();

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &_currentBindState.vertexBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &_currentBindState.varyingBuffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &V_ITa);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &V_IT);
    clSetKernelArg(kernel, 4, sizeof(int), &_currentBindState.vertexDesc.offset);
    clSetKernelArg(kernel, 5, sizeof(int), &_currentBindState.varyingDesc.offset);
    clSetKernelArg(kernel, 6, sizeof(int), batch.GetVertexOffsetPtr());
    clSetKernelArg(kernel, 7, sizeof(int), batch.GetTableOffsetPtr());
    clSetKernelArg(kernel, 8, sizeof(int), batch.GetStartPtr());
    clSetKernelArg(kernel, 9, sizeof(int), batch.GetEndPtr());

    ciErrNum = clEnqueueNDRangeKernel(_clQueue,
                                      kernel, 1, NULL, globalWorkSize,
                                      NULL, 0, NULL, NULL);
    CL_CHECK_ERROR(ciErrNum, "restricted vertex kernel B1 %d\n", ciErrNum);
}

void
OsdCLComputeController::ApplyCatmarkRestrictedVertexVerticesKernelB2(
    FarKernelBatch const &batch, OsdCLComputeContext const *context) const {

    assert(context);

    cl_int ciErrNum;
    size_t globalWorkSize[1] = { (size_t)(batch.GetEnd() - batch.GetStart()) };
    cl_kernel kernel = _currentBindState.kernelBundle->GetCatmarkRestrictedVertexKernelB2();

    cl_mem V_ITa = context->GetTable(FarSubdivisionTables::V_ITa)->GetDevicePtr();
    cl_mem V_IT = context->GetTable(FarSubdivisionTables::V_IT)->GetDevicePtr();

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &_currentBindState.vertexBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &_currentBindState.varyingBuffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &V_ITa);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &V_IT);
    clSetKernelArg(kernel, 4, sizeof(int), &_currentBindState.vertexDesc.offset);
    clSetKernelArg(kernel, 5, sizeof(int), &_currentBindState.varyingDesc.offset);
    clSetKernelArg(kernel, 6, sizeof(int), batch.GetVertexOffsetPtr());
    clSetKernelArg(kernel, 7, sizeof(int), batch.GetTableOffsetPtr());
    clSetKernelArg(kernel, 8, sizeof(int), batch.GetStartPtr());
    clSetKernelArg(kernel, 9, sizeof(int), batch.GetEndPtr());

    ciErrNum = clEnqueueNDRangeKernel(_clQueue,
                                      kernel, 1, NULL, globalWorkSize,
                                      NULL, 0, NULL, NULL);
    CL_CHECK_ERROR(ciErrNum, "restricted vertex kernel B2 %d\n", ciErrNum);
}

void
OsdCLComputeController::ApplyCatmarkRestrictedVertexVerticesKernelA(
    FarKernelBatch const &batch, OsdCLComputeContext const *context) const {

    assert(context);

    cl_int ciErrNum;
    size_t globalWorkSize[1] = { (size_t)(batch.GetEnd() - batch.GetStart()) };
    cl_kernel kernel = _currentBindState.kernelBundle->GetCatmarkRestrictedVertexKernelA();

    cl_mem V_ITa = context->GetTable(FarSubdivisionTables::V_ITa)->GetDevicePtr();

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &_currentBindState.vertexBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &_currentBindState.varyingBuffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &V_ITa);
    clSetKernelArg(kernel, 3, sizeof(int), &_currentBindState.vertexDesc.offset);
    clSetKernelArg(kernel, 4, sizeof(int), &_currentBindState.varyingDesc.offset);
    clSetKernelArg(kernel, 5, sizeof(int), batch.GetVertexOffsetPtr());
    clSetKernelArg(kernel, 6, sizeof(int), batch.GetTableOffsetPtr());
    clSetKernelArg(kernel, 7, sizeof(int), batch.GetStartPtr());
    clSetKernelArg(kernel, 8, sizeof(int), batch.GetEndPtr());

    ciErrNum = clEnqueueNDRangeKernel(_clQueue,
                                      kernel, 1, NULL, globalWorkSize,
                                      NULL, 0, NULL, NULL);
    CL_CHECK_ERROR(ciErrNum, "restricted vertex kernel A %d\n", ciErrNum);
}

void
OsdCLComputeController::ApplyLoopEdgeVerticesKernel(
    FarKernelBatch const &batch, OsdCLComputeContext const *context) const {

    assert(context);

    cl_int ciErrNum;
    size_t globalWorkSize[1] = { (size_t)(batch.GetEnd() - batch.GetStart()) };
    cl_kernel kernel = _currentBindState.kernelBundle->GetLoopEdgeKernel();

    cl_mem E_IT = context->GetTable(FarSubdivisionTables::E_IT)->GetDevicePtr();
    cl_mem E_W = context->GetTable(FarSubdivisionTables::E_W)->GetDevicePtr();

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &_currentBindState.vertexBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &_currentBindState.varyingBuffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &E_IT);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &E_W);
    clSetKernelArg(kernel, 4, sizeof(int), &_currentBindState.vertexDesc.offset);
    clSetKernelArg(kernel, 5, sizeof(int), &_currentBindState.varyingDesc.offset);
    clSetKernelArg(kernel, 6, sizeof(int), batch.GetVertexOffsetPtr());
    clSetKernelArg(kernel, 7, sizeof(int), batch.GetTableOffsetPtr());
    clSetKernelArg(kernel, 8, sizeof(int), batch.GetStartPtr());
    clSetKernelArg(kernel, 9, sizeof(int), batch.GetEndPtr());

    ciErrNum = clEnqueueNDRangeKernel(_clQueue,
                                      kernel, 1, NULL, globalWorkSize,
                                      NULL, 0, NULL, NULL);
    CL_CHECK_ERROR(ciErrNum, "edge kernel %d\n", ciErrNum);
}

void
OsdCLComputeController::ApplyLoopVertexVerticesKernelB(
    FarKernelBatch const &batch, OsdCLComputeContext const *context) const {

    assert(context);

    cl_int ciErrNum;
    size_t globalWorkSize[1] = { (size_t)(batch.GetEnd() - batch.GetStart()) };
    cl_kernel kernel = _currentBindState.kernelBundle->GetLoopVertexKernelB();

    cl_mem V_ITa = context->GetTable(FarSubdivisionTables::V_ITa)->GetDevicePtr();
    cl_mem V_IT = context->GetTable(FarSubdivisionTables::V_IT)->GetDevicePtr();
    cl_mem V_W = context->GetTable(FarSubdivisionTables::V_W)->GetDevicePtr();

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &_currentBindState.vertexBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &_currentBindState.varyingBuffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &V_ITa);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &V_IT);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &V_W);
    clSetKernelArg(kernel, 5, sizeof(int), &_currentBindState.vertexDesc.offset);
    clSetKernelArg(kernel, 6, sizeof(int), &_currentBindState.varyingDesc.offset);
    clSetKernelArg(kernel, 7, sizeof(int), batch.GetVertexOffsetPtr());
    clSetKernelArg(kernel, 8, sizeof(int), batch.GetTableOffsetPtr());
    clSetKernelArg(kernel, 9, sizeof(int), batch.GetStartPtr());
    clSetKernelArg(kernel, 10, sizeof(int), batch.GetEndPtr());

    ciErrNum = clEnqueueNDRangeKernel(_clQueue,
                                      kernel, 1, NULL, globalWorkSize,
                                      NULL, 0, NULL, NULL);
    CL_CHECK_ERROR(ciErrNum, "vertex kernel 1 %d\n", ciErrNum);
}

void
OsdCLComputeController::ApplyLoopVertexVerticesKernelA1(
    FarKernelBatch const &batch, OsdCLComputeContext const *context) const {

    assert(context);

    cl_int ciErrNum;
    size_t globalWorkSize[1] = { (size_t)(batch.GetEnd() - batch.GetStart()) };
    int ipass = false;
    cl_kernel kernel = _currentBindState.kernelBundle->GetLoopVertexKernelA();

    cl_mem V_ITa = context->GetTable(FarSubdivisionTables::V_ITa)->GetDevicePtr();
    cl_mem V_W = context->GetTable(FarSubdivisionTables::V_W)->GetDevicePtr();

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &_currentBindState.vertexBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &_currentBindState.varyingBuffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &V_ITa);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &V_W);
    clSetKernelArg(kernel, 4, sizeof(int), &_currentBindState.vertexDesc.offset);
    clSetKernelArg(kernel, 5, sizeof(int), &_currentBindState.varyingDesc.offset);
    clSetKernelArg(kernel, 6, sizeof(int), batch.GetVertexOffsetPtr());
    clSetKernelArg(kernel, 7, sizeof(int), batch.GetTableOffsetPtr());
    clSetKernelArg(kernel, 8, sizeof(int), batch.GetStartPtr());
    clSetKernelArg(kernel, 9, sizeof(int), batch.GetEndPtr());
    clSetKernelArg(kernel, 10, sizeof(int), &ipass);

    ciErrNum = clEnqueueNDRangeKernel(_clQueue,
                                      kernel, 1, NULL, globalWorkSize,
                                      NULL, 0, NULL, NULL);
    CL_CHECK_ERROR(ciErrNum, "vertex kernel 2 %d\n", ciErrNum);
}

void
OsdCLComputeController::ApplyLoopVertexVerticesKernelA2(
    FarKernelBatch const &batch, OsdCLComputeContext const *context) const {

    assert(context);

    cl_int ciErrNum;
    size_t globalWorkSize[1] = { (size_t)(batch.GetEnd() - batch.GetStart()) };
    int ipass = true;
    cl_kernel kernel = _currentBindState.kernelBundle->GetLoopVertexKernelA();

    cl_mem V_ITa = context->GetTable(FarSubdivisionTables::V_ITa)->GetDevicePtr();
    cl_mem V_W = context->GetTable(FarSubdivisionTables::V_W)->GetDevicePtr();

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &_currentBindState.vertexBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &_currentBindState.varyingBuffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &V_ITa);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &V_W);
    clSetKernelArg(kernel, 4, sizeof(int), &_currentBindState.vertexDesc.offset);
    clSetKernelArg(kernel, 5, sizeof(int), &_currentBindState.varyingDesc.offset);
    clSetKernelArg(kernel, 6, sizeof(int), batch.GetVertexOffsetPtr());
    clSetKernelArg(kernel, 7, sizeof(int), batch.GetTableOffsetPtr());
    clSetKernelArg(kernel, 8, sizeof(int), batch.GetStartPtr());
    clSetKernelArg(kernel, 9, sizeof(int), batch.GetEndPtr());
    clSetKernelArg(kernel, 10, sizeof(int), &ipass);

    ciErrNum = clEnqueueNDRangeKernel(_clQueue,
                                      kernel, 1, NULL, globalWorkSize,
                                      NULL, 0, NULL, NULL);
    CL_CHECK_ERROR(ciErrNum, "vertex kernel 2 %d\n", ciErrNum);
}

void
OsdCLComputeController::ApplyVertexEdits(
    FarKernelBatch const &batch, OsdCLComputeContext const *context) const {

    assert(context);

    cl_int ciErrNum;
    size_t globalWorkSize[1] = { (size_t)(batch.GetEnd() - batch.GetStart()) };

    const OsdCLHEditTable * edit = context->GetEditTable(batch.GetTableIndex());
    assert(edit);

    const OsdCLTable * primvarIndices = edit->GetPrimvarIndices();
    const OsdCLTable * editValues = edit->GetEditValues();

    cl_mem indices = primvarIndices->GetDevicePtr();
    cl_mem values = editValues->GetDevicePtr();
    int primvarOffset = edit->GetPrimvarOffset();
    int primvarWidth = edit->GetPrimvarWidth();

    if (edit->GetOperation() == FarVertexEdit::Add) {
        cl_kernel kernel = _currentBindState.kernelBundle->GetVertexEditAdd();

        clSetKernelArg(kernel, 0, sizeof(cl_mem), &_currentBindState.vertexBuffer);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &indices);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &values);
        clSetKernelArg(kernel, 3, sizeof(int), &_currentBindState.vertexDesc.offset);
        clSetKernelArg(kernel, 4, sizeof(int), &primvarOffset);
        clSetKernelArg(kernel, 5, sizeof(int), &primvarWidth);
        clSetKernelArg(kernel, 6, sizeof(int), batch.GetVertexOffsetPtr());
        clSetKernelArg(kernel, 7, sizeof(int), batch.GetTableOffsetPtr());
        clSetKernelArg(kernel, 8, sizeof(int), batch.GetStartPtr());
        clSetKernelArg(kernel, 9, sizeof(int), batch.GetEndPtr());

        ciErrNum = clEnqueueNDRangeKernel(_clQueue,
                                          kernel, 1, NULL, globalWorkSize,
                                          NULL, 0, NULL, NULL);
        
        CL_CHECK_ERROR(ciErrNum, "vertex edit %d %d\n", batch.GetTableIndex(), ciErrNum);

    } else if (edit->GetOperation() == FarVertexEdit::Set) {
        // XXXX TODO
    }
}

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv

