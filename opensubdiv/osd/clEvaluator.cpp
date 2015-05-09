//
//   Copyright 2015 Pixar
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

#include "../osd/clEvaluator.h"

#include <sstream>
#include <string>
#include <vector>

#include "../osd/opencl.h"
#include "../far/error.h"
#include "../far/stencilTables.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

static const char *clSource =
#include "clKernel.gen.h"
;

// ----------------------------------------------------------------------------

template <class T> cl_mem
createCLBuffer(std::vector<T> const & src, cl_context clContext) {
    cl_int errNum = 0;
    cl_mem devicePtr = clCreateBuffer(clContext,
                                      CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,
                                      src.size()*sizeof(T),
                                      (void*)(&src.at(0)),
                                      &errNum);

    if (errNum != CL_SUCCESS) {
        Far::Error(Far::FAR_RUNTIME_ERROR, "clCreateBuffer: %d", errNum);
    }

    return devicePtr;
}

// ----------------------------------------------------------------------------

CLStencilTables::CLStencilTables(Far::StencilTables const *stencilTables,
                                 cl_context clContext) {
    _numStencils = stencilTables->GetNumStencils();

    if (_numStencils > 0) {
        _sizes   = createCLBuffer(stencilTables->GetSizes(), clContext);
        _offsets = createCLBuffer(stencilTables->GetOffsets(), clContext);
        _indices = createCLBuffer(stencilTables->GetControlIndices(),
                                  clContext);
        _weights = createCLBuffer(stencilTables->GetWeights(), clContext);
    } else {
        _sizes = _offsets = _indices = _weights = NULL;
    }
}

CLStencilTables::~CLStencilTables() {
    if (_sizes)   clReleaseMemObject(_sizes);
    if (_offsets) clReleaseMemObject(_offsets);
    if (_indices) clReleaseMemObject(_indices);
    if (_weights) clReleaseMemObject(_weights);
}

// ---------------------------------------------------------------------------

CLEvaluator::CLEvaluator(cl_context context, cl_command_queue queue)
    : _clContext(context), _clCommandQueue(queue),
      _program(NULL), _stencilsKernel(NULL) {
}

CLEvaluator::~CLEvaluator() {
    if (_stencilsKernel) clReleaseKernel(_stencilsKernel);
    if (_program) clReleaseProgram(_program);
}

bool
CLEvaluator::Compile(VertexBufferDescriptor const &srcDesc,
                   VertexBufferDescriptor const &dstDesc) {
    if (srcDesc.length > dstDesc.length) {
        Far::Error(Far::FAR_RUNTIME_ERROR,
                   "srcDesc length must be less than or equal to "
                   "dstDesc length.\n");
        return false;
    }

    cl_int errNum;

    std::ostringstream defines;
    defines << "#define LENGTH "     << srcDesc.length << "\n"
            << "#define SRC_STRIDE " << srcDesc.stride << "\n"
            << "#define DST_STRIDE " << dstDesc.stride << "\n";
    std::string defineStr = defines.str();

    const char *sources[] = { defineStr.c_str(), clSource };
    _program = clCreateProgramWithSource(_clContext, 2, sources, 0, &errNum);
    if (errNum != CL_SUCCESS) {
        Far::Error(Far::FAR_RUNTIME_ERROR,
                   "clCreateProgramWithSource (%d)", errNum);
    }

    errNum = clBuildProgram(_program, 0, NULL, NULL, NULL, NULL);
    if (errNum != CL_SUCCESS) {
        Far::Error(Far::FAR_RUNTIME_ERROR, "clBuildProgram (%d) \n", errNum);

        cl_int numDevices = 0;
        clGetContextInfo(
            _clContext, CL_CONTEXT_NUM_DEVICES,
            sizeof(cl_uint), &numDevices, NULL);

        cl_device_id *devices = new cl_device_id[numDevices];
        clGetContextInfo(_clContext, CL_CONTEXT_DEVICES,
                         sizeof(cl_device_id)*numDevices, devices, NULL);

        for (int i = 0; i < numDevices; ++i) {
            char cBuildLog[10240];
            clGetProgramBuildInfo(
                _program, devices[i],
                CL_PROGRAM_BUILD_LOG, sizeof(cBuildLog), cBuildLog, NULL);
            Far::Error(Far::FAR_RUNTIME_ERROR, cBuildLog);
        }
        delete[] devices;

        return false;
    }

    _stencilsKernel = clCreateKernel(_program, "computeStencils", &errNum);

    if (errNum != CL_SUCCESS) {
        Far::Error(Far::FAR_RUNTIME_ERROR, "buildKernel (%d)\n", errNum);
        return false;
    }
    return true;
}

bool
CLEvaluator::EvalStencils(cl_mem src,
                          VertexBufferDescriptor const &srcDesc,
                          cl_mem dst,
                          VertexBufferDescriptor const &dstDesc,
                          cl_mem sizes,
                          cl_mem offsets,
                          cl_mem indices,
                          cl_mem weights,
                          int start,
                          int end) const {
    if (end <= start) return true;

    size_t globalWorkSize = (size_t)(end - start);

    clSetKernelArg(_stencilsKernel, 0, sizeof(cl_mem), &src);
    clSetKernelArg(_stencilsKernel, 1, sizeof(int), &srcDesc.offset);
    clSetKernelArg(_stencilsKernel, 2, sizeof(cl_mem), &dst);
    clSetKernelArg(_stencilsKernel, 3, sizeof(int), &dstDesc.offset);
    clSetKernelArg(_stencilsKernel, 4, sizeof(cl_mem), &sizes);
    clSetKernelArg(_stencilsKernel, 5, sizeof(cl_mem), &offsets);
    clSetKernelArg(_stencilsKernel, 6, sizeof(cl_mem), &indices);
    clSetKernelArg(_stencilsKernel, 7, sizeof(cl_mem), &weights);
    clSetKernelArg(_stencilsKernel, 8, sizeof(int), &start);
    clSetKernelArg(_stencilsKernel, 9, sizeof(int), &end);

    cl_int errNum = clEnqueueNDRangeKernel(
        _clCommandQueue, _stencilsKernel, 1, NULL,
        &globalWorkSize, NULL, 0, NULL, NULL);

    if (errNum != CL_SUCCESS) {
        Far::Error(Far::FAR_RUNTIME_ERROR,
                   "ApplyStencilTableKernel (%d) ", errNum);
        return false;
    }

    clFinish(_clCommandQueue);
    return true;
}

/* static */
void
CLEvaluator::Synchronize(cl_command_queue clCommandQueue) {
    clFinish(clCommandQueue);
}

}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
