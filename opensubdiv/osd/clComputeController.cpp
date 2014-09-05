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
#include "../osd/error.h"

#if defined(_WIN32)
    #include <windows.h>
#endif

#include <algorithm>
#include <string.h>
#include <sstream>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

static const char *clSource =
#include "clKernel.gen.h"
;

// -----------------------------------------------------------------------------

static cl_kernel buildKernel(cl_program prog, const char * name) {

    cl_int errNum;
    cl_kernel k = clCreateKernel(prog, name, &errNum);

    if (errNum != CL_SUCCESS) {
        Error(OSD_CL_KERNEL_CREATE_ERROR, "buildKernel '%s' (%d)\n", name, errNum);
    }
    return k;
}

// -----------------------------------------------------------------------------

class CLComputeController::KernelBundle :
    NonCopyable<CLComputeController::KernelBundle> {

public:

    bool Compile(cl_context clContext, VertexBufferDescriptor const & desc) {

        cl_int errNum;

        _desc = VertexBufferDescriptor(0, desc.length, desc.stride);

        std::ostringstream defines;
        defines << "#define OFFSET " << _desc.offset << "\n"
                << "#define LENGTH " << _desc.length << "\n"
                << "#define STRIDE " << _desc.stride << "\n";
        std::string defineStr = defines.str();

        const char *sources[] = { defineStr.c_str(), clSource };
        _program = clCreateProgramWithSource(clContext, 2, sources, 0, &errNum);
        if (errNum!=CL_SUCCESS) {
            Error(OSD_CL_PROGRAM_BUILD_ERROR,
                "clCreateProgramWithSource (%d)", errNum);
        }

        errNum = clBuildProgram(_program, 0, NULL, NULL, NULL, NULL);
        if (errNum != CL_SUCCESS) {
            Error(OSD_CL_PROGRAM_BUILD_ERROR, "clBuildProgram (%d) \n", errNum);

            cl_int numDevices = 0;
            clGetContextInfo(clContext,
                CL_CONTEXT_NUM_DEVICES, sizeof(cl_uint), &numDevices, NULL);

            cl_device_id *devices = new cl_device_id[numDevices];
            clGetContextInfo(clContext, CL_CONTEXT_DEVICES,
                sizeof(cl_device_id)*numDevices, devices, NULL);

            for (int i = 0; i < numDevices; ++i) {
                char cBuildLog[10240];
                clGetProgramBuildInfo(_program, devices[i],
                    CL_PROGRAM_BUILD_LOG, sizeof(cBuildLog), cBuildLog, NULL);
                Error(OSD_CL_PROGRAM_BUILD_ERROR, cBuildLog);
            }
            delete[] devices;

            return false;
        }

        // compile all cl compute kernels
        _stencilsKernel = buildKernel(_program, "computeStencils");

        return true;
    }

    cl_kernel GetStencilsKernel() const {
        return _stencilsKernel;
    }

    struct Match {

        Match(VertexBufferDescriptor const & d) : desc(d) { }

        bool operator() (KernelBundle const * kernel) {
            return (desc.length==kernel->_desc.length and
                    desc.stride==kernel->_desc.stride);
        }

        VertexBufferDescriptor desc;
    };

private:

    cl_program _program;

    cl_kernel _stencilsKernel;

    VertexBufferDescriptor _desc;
};

// ----------------------------------------------------------------------------

void
CLComputeController::ApplyStencilTableKernel(
    Far::KernelBatch const &batch, ComputeContext const *context) {

    assert(context);

    cl_int errNum;

    size_t globalWorkSize[1] = { (size_t)(batch.end - batch.start) };

    int ncvs = context->GetNumControlVertices();

    if (context->HasVertexStencilTables()) {

        KernelBundle const * bundle = getKernel(_currentBindState.vertexDesc);

        cl_kernel kernel = bundle->GetStencilsKernel();

        cl_mem sizes = context->GetVertexStencilTablesSizes(),
               offsets = context->GetVertexStencilTablesOffsets(),
               indices = context->GetVertexStencilTablesIndices(),
               weights = context->GetVertexStencilTablesWeights();

        clSetKernelArg(kernel, 0, sizeof(cl_mem), &_currentBindState.vertexBuffer);

        clSetKernelArg(kernel, 1, sizeof(cl_mem), &sizes);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &offsets);
        clSetKernelArg(kernel, 3, sizeof(cl_mem), &indices);
        clSetKernelArg(kernel, 4, sizeof(cl_mem), &weights);

        clSetKernelArg(kernel, 5, sizeof(int), &batch.start);
        clSetKernelArg(kernel, 6, sizeof(int), &batch.end);

        clSetKernelArg(kernel, 7, sizeof(int), &_currentBindState.vertexDesc.offset);
        clSetKernelArg(kernel, 8, sizeof(int), &ncvs);

        errNum = clEnqueueNDRangeKernel(
            _clQueue, kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
        if (errNum!=CL_SUCCESS) {
            Error(OSD_CL_RUNTIME_ERROR,
                "ApplyStencilTableKernel (%d) ", errNum);
        }
    }

    if (context->HasVaryingStencilTables()) {

        KernelBundle const * bundle = getKernel(_currentBindState.varyingDesc);

        cl_kernel kernel = bundle->GetStencilsKernel();

        cl_mem sizes = context->GetVaryingStencilTablesSizes(),
               offsets = context->GetVaryingStencilTablesOffsets(),
               indices = context->GetVaryingStencilTablesIndices(),
               weights = context->GetVaryingStencilTablesWeights();

        clSetKernelArg(kernel, 0, sizeof(cl_mem), &_currentBindState.varyingBuffer);

        clSetKernelArg(kernel, 1, sizeof(cl_mem), &sizes);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &offsets);
        clSetKernelArg(kernel, 3, sizeof(cl_mem), &indices);
        clSetKernelArg(kernel, 4, sizeof(cl_mem), &weights);

        clSetKernelArg(kernel, 5, sizeof(int), &batch.start);
        clSetKernelArg(kernel, 6, sizeof(int), &batch.end);

        clSetKernelArg(kernel, 7, sizeof(int), &_currentBindState.varyingDesc.offset);
        clSetKernelArg(kernel, 8, sizeof(int), &ncvs);

        errNum = clEnqueueNDRangeKernel(
            _clQueue, kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
        if (errNum!=CL_SUCCESS) {
            Error(OSD_CL_RUNTIME_ERROR,
                "ApplyStencilTableKernel (%d)", errNum);
        }
    }
}


// ----------------------------------------------------------------------------

CLComputeController::KernelBundle const *
CLComputeController::getKernel(VertexBufferDescriptor const &desc) {

    KernelRegistry::iterator it =
        std::find_if(_kernelRegistry.begin(), _kernelRegistry.end(),
            KernelBundle::Match(desc));

    if (it != _kernelRegistry.end()) {
        return *it;
    } else {
        KernelBundle * kernelBundle = new KernelBundle();
        kernelBundle->Compile(_clContext, desc);
        _kernelRegistry.push_back(kernelBundle);
        return kernelBundle;
    }
}

// ----------------------------------------------------------------------------

CLComputeController::CLComputeController(
    cl_context clContext, cl_command_queue queue) :
         _clContext(clContext), _clQueue(queue) {
}

CLComputeController::~CLComputeController() {
    for (KernelRegistry::iterator it = _kernelRegistry.begin();
        it != _kernelRegistry.end(); ++it) {
        delete *it;
    }
}

// ----------------------------------------------------------------------------

void
CLComputeController::Synchronize() {

    clFinish(_clQueue);
}


// -----------------------------------------------------------------------------

}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
