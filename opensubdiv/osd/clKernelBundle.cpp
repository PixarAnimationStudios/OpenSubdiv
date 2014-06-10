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

#include "../osd/opengl.h"

#include "../osd/clKernelBundle.h"
#include "../osd/error.h"

#include <stdio.h>
#include <sstream>

#ifdef _MSC_VER
#define snprintf _snprintf
#endif

static const char *clSource =
#include "clKernel.gen.h"
    ;

#define CL_CHECK_ERROR(x, ...) {      \
        if (x != CL_SUCCESS)          \
        { printf("ERROR %d : ", x);   \
            printf(__VA_ARGS__);} }

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

OsdCLKernelBundle::OsdCLKernelBundle() :
    _clProgram(NULL),
    _clBilinearEdge(NULL),
    _clBilinearVertex(NULL),
    _clCatmarkFace(NULL),
    _clCatmarkQuadFace(NULL),
    _clCatmarkTriQuadFace(NULL),
    _clCatmarkEdge(NULL),
    _clCatmarkRestrictedEdge(NULL),
    _clCatmarkVertexA(NULL),
    _clCatmarkVertexB(NULL),
    _clCatmarkRestrictedVertexA(NULL),
    _clCatmarkRestrictedVertexB1(NULL),
    _clCatmarkRestrictedVertexB2(NULL),
    _clLoopEdge(NULL),
    _clLoopVertexA(NULL),
    _clLoopVertexB(NULL),
    _numVertexElements(0),
    _vertexStride(0),
    _numVaryingElements(0),
    _varyingStride(0) {
}

OsdCLKernelBundle::~OsdCLKernelBundle() {

    if (_clBilinearEdge)
        clReleaseKernel(_clBilinearEdge);
    if (_clBilinearVertex)
        clReleaseKernel(_clBilinearVertex);

    if (_clCatmarkFace)
        clReleaseKernel(_clCatmarkFace);
    if (_clCatmarkQuadFace)
        clReleaseKernel(_clCatmarkQuadFace);
    if (_clCatmarkTriQuadFace)
        clReleaseKernel(_clCatmarkTriQuadFace);
    if (_clCatmarkEdge)
        clReleaseKernel(_clCatmarkEdge);
    if (_clCatmarkRestrictedEdge)
        clReleaseKernel(_clCatmarkRestrictedEdge);
    if (_clCatmarkVertexA)
        clReleaseKernel(_clCatmarkVertexA);
    if (_clCatmarkVertexB)
        clReleaseKernel(_clCatmarkVertexB);
    if (_clCatmarkRestrictedVertexA)
        clReleaseKernel(_clCatmarkRestrictedVertexA);
    if (_clCatmarkRestrictedVertexB1)
        clReleaseKernel(_clCatmarkRestrictedVertexB1);
    if (_clCatmarkRestrictedVertexB2)
        clReleaseKernel(_clCatmarkRestrictedVertexB2);

    if (_clLoopEdge)
        clReleaseKernel(_clLoopEdge);
    if (_clLoopVertexA)
        clReleaseKernel(_clLoopVertexA);
    if (_clLoopVertexB)
        clReleaseKernel(_clLoopVertexB);

    if (_clProgram) clReleaseProgram(_clProgram);
}

static cl_kernel buildKernel(cl_program prog, const char * name) {

    cl_int ciErr;
    cl_kernel k = clCreateKernel(prog, name, &ciErr);

    if (ciErr != CL_SUCCESS) {
        OsdError(OSD_CL_KERNEL_CREATE_ERROR);
    }
    return k;
}

bool
OsdCLKernelBundle::Compile(cl_context clContext,
                           OsdVertexBufferDescriptor const &vertexDesc,
                           OsdVertexBufferDescriptor const &varyingDesc) {

    cl_int ciErrNum;

    _numVertexElements = vertexDesc.length;
    _vertexStride = vertexDesc.stride;
    _numVaryingElements = varyingDesc.length;
    _varyingStride = varyingDesc.stride;

    std::ostringstream defines;
    defines << "#define NUM_VERTEX_ELEMENTS "  << _numVertexElements << "\n"
            << "#define VERTEX_STRIDE "        << _vertexStride << "\n"
            << "#define NUM_VARYING_ELEMENTS " << _numVaryingElements << "\n"
            << "#define VARYING_STRIDE "       << _varyingStride << "\n";
    std::string defineStr = defines.str();

    const char *sources[] = { defineStr.c_str(), clSource };

    _clProgram = clCreateProgramWithSource(clContext, 2, sources, 0, &ciErrNum);
    CL_CHECK_ERROR(ciErrNum, "clCreateProgramWithSource\n");

    ciErrNum = clBuildProgram(_clProgram, 0, NULL, NULL, NULL, NULL);
    if (ciErrNum != CL_SUCCESS) {
        OsdError(OSD_CL_PROGRAM_BUILD_ERROR, "CLerr=%d", ciErrNum);

        cl_int numDevices = 0;
        clGetContextInfo(clContext, CL_CONTEXT_NUM_DEVICES,
                         sizeof(cl_uint), &numDevices, NULL);
        cl_device_id *devices = new cl_device_id[numDevices];
        clGetContextInfo(clContext, CL_CONTEXT_DEVICES,
                         sizeof(cl_device_id)*numDevices, devices, NULL);
        for (int i = 0; i < numDevices; ++i) {
            char cBuildLog[10240];
            clGetProgramBuildInfo(_clProgram, devices[i], CL_PROGRAM_BUILD_LOG,
                                  sizeof(cBuildLog), cBuildLog, NULL);
            OsdError(OSD_CL_PROGRAM_BUILD_ERROR, cBuildLog);
        }
        delete[] devices;

        return false;
    }

    _clBilinearEdge              = buildKernel(_clProgram, "computeBilinearEdge");
    _clBilinearVertex            = buildKernel(_clProgram, "computeBilinearVertex");
    _clCatmarkFace               = buildKernel(_clProgram, "computeFace");
    _clCatmarkQuadFace           = buildKernel(_clProgram, "computeQuadFace");
    _clCatmarkTriQuadFace        = buildKernel(_clProgram, "computeTriQuadFace");
    _clCatmarkEdge               = buildKernel(_clProgram, "computeEdge");
    _clCatmarkRestrictedEdge     = buildKernel(_clProgram, "computeRestrictedEdge");
    _clCatmarkVertexA            = buildKernel(_clProgram, "computeVertexA");
    _clCatmarkVertexB            = buildKernel(_clProgram, "computeVertexB");
    _clCatmarkRestrictedVertexA  = buildKernel(_clProgram, "computeRestrictedVertexA");
    _clCatmarkRestrictedVertexB1 = buildKernel(_clProgram, "computeRestrictedVertexB1");
    _clCatmarkRestrictedVertexB2 = buildKernel(_clProgram, "computeRestrictedVertexB2");
    _clLoopEdge                  = buildKernel(_clProgram, "computeEdge");
    _clLoopVertexA               = buildKernel(_clProgram, "computeVertexA");
    _clLoopVertexB               = buildKernel(_clProgram, "computeLoopVertexB");
    _clVertexEditAdd             = buildKernel(_clProgram, "editVertexAdd");

    return true;
}

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
