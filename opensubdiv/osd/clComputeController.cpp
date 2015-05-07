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
#include "../far/error.h"

#if defined(_WIN32)
    #include <windows.h>
#endif

#include <algorithm>
#include <string.h>
#include <sstream>
#include <cassert>

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
        Far::Error(Far::FAR_RUNTIME_ERROR, "buildKernel '%s' (%d)\n", name, errNum);
    }
    return k;
}

// -----------------------------------------------------------------------------

class CLComputeController::KernelBundle :
    NonCopyable<CLComputeController::KernelBundle> {

public:

    bool Compile(cl_context clContext,
                 VertexBufferDescriptor const & srcDesc,
                 VertexBufferDescriptor const & dstDesc) {

        cl_int errNum;

        // XXX: only store srcDesc.
        //      this is ok since currently this kernel doesn't get called with
        //      different strides for src and dst. This function will be
        //      refactored soon.
        _desc = VertexBufferDescriptor(0, srcDesc.length, srcDesc.stride);

        std::ostringstream defines;
        defines << "#define LENGTH "     << srcDesc.length << "\n"
                << "#define SRC_STRIDE " << srcDesc.stride << "\n"
                << "#define DST_STRIDE " << dstDesc.stride << "\n";
        std::string defineStr = defines.str();

        const char *sources[] = { defineStr.c_str(), clSource };
        _program = clCreateProgramWithSource(clContext, 2, sources, 0, &errNum);
        if (errNum!=CL_SUCCESS) {
            Far::Error(Far::FAR_RUNTIME_ERROR,
                "clCreateProgramWithSource (%d)", errNum);
        }

        errNum = clBuildProgram(_program, 0, NULL, NULL, NULL, NULL);
        if (errNum != CL_SUCCESS) {
           Far::Error(Far::FAR_RUNTIME_ERROR, "clBuildProgram (%d) \n", errNum);

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
                Far::Error(Far::FAR_RUNTIME_ERROR, cBuildLog);
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
CLComputeController::ApplyStencilTableKernel(ComputeContext const *context) {

    assert(context);

    cl_int errNum;

    size_t globalWorkSize = 0;

    if (context->HasVertexStencilTables()) {
        int start = 0;
        int end = context->GetNumStencilsInVertexStencilTables();
        globalWorkSize = (size_t)(end - start);

        KernelBundle const * bundle = getKernel(_currentBindState.vertexDesc);

        cl_kernel kernel = bundle->GetStencilsKernel();

        cl_mem sizes = context->GetVertexStencilTablesSizes(),
               offsets = context->GetVertexStencilTablesOffsets(),
               indices = context->GetVertexStencilTablesIndices(),
               weights = context->GetVertexStencilTablesWeights();

        cl_mem src = _currentBindState.vertexBuffer;
        cl_mem dst = _currentBindState.vertexBuffer;

        VertexBufferDescriptor srcDesc = _currentBindState.vertexDesc;
        VertexBufferDescriptor dstDesc(srcDesc);
        dstDesc.offset += context->GetNumControlVertices() * dstDesc.stride;

        clSetKernelArg(kernel, 0, sizeof(cl_mem), &src);
        clSetKernelArg(kernel, 1, sizeof(int), &srcDesc.offset);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &dst);
        clSetKernelArg(kernel, 3, sizeof(int), &dstDesc.offset);
        clSetKernelArg(kernel, 4, sizeof(cl_mem), &sizes);
        clSetKernelArg(kernel, 5, sizeof(cl_mem), &offsets);
        clSetKernelArg(kernel, 6, sizeof(cl_mem), &indices);
        clSetKernelArg(kernel, 7, sizeof(cl_mem), &weights);
        clSetKernelArg(kernel, 8, sizeof(int), &start);
        clSetKernelArg(kernel, 9, sizeof(int), &end);

        errNum = clEnqueueNDRangeKernel(
            _clQueue, kernel, 1, NULL, &globalWorkSize, NULL, 0, NULL, NULL);
        if (errNum!=CL_SUCCESS) {
            Far::Error(Far::FAR_RUNTIME_ERROR,
                "ApplyStencilTableKernel (%d) ", errNum);
        }
    }

    if (context->HasVaryingStencilTables()) {
        int start = 0;
        int end = context->GetNumStencilsInVaryingStencilTables();
        globalWorkSize = (size_t)(end - start);

        KernelBundle const * bundle = getKernel(_currentBindState.varyingDesc);

        cl_kernel kernel = bundle->GetStencilsKernel();

        cl_mem sizes = context->GetVaryingStencilTablesSizes(),
               offsets = context->GetVaryingStencilTablesOffsets(),
               indices = context->GetVaryingStencilTablesIndices(),
               weights = context->GetVaryingStencilTablesWeights();

        cl_mem src = _currentBindState.varyingBuffer;
        cl_mem dst = _currentBindState.varyingBuffer;

        VertexBufferDescriptor srcDesc = _currentBindState.varyingDesc;
        VertexBufferDescriptor dstDesc(srcDesc);
        dstDesc.offset += context->GetNumControlVertices() * dstDesc.stride;

        clSetKernelArg(kernel, 0, sizeof(cl_mem), &src);
        clSetKernelArg(kernel, 1, sizeof(int), &srcDesc.offset);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &dst);
        clSetKernelArg(kernel, 3, sizeof(int), &dstDesc.offset);
        clSetKernelArg(kernel, 4, sizeof(cl_mem), &sizes);
        clSetKernelArg(kernel, 5, sizeof(cl_mem), &offsets);
        clSetKernelArg(kernel, 6, sizeof(cl_mem), &indices);
        clSetKernelArg(kernel, 7, sizeof(cl_mem), &weights);
        clSetKernelArg(kernel, 8, sizeof(int), &start);
        clSetKernelArg(kernel, 9, sizeof(int), &end);

        errNum = clEnqueueNDRangeKernel(
            _clQueue, kernel, 1, NULL, &globalWorkSize, NULL, 0, NULL, NULL);
        if (errNum!=CL_SUCCESS) {
            Far::Error(Far::FAR_RUNTIME_ERROR,
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
        kernelBundle->Compile(_clContext, desc, desc);
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
