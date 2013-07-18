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

#if defined(_WIN32)
    #include <windows.h>
#elif defined(__APPLE__)
    #include <OpenGL/OpenGL.h>
#else
    #include <GL/gl.h>
#endif

#include "../osd/clKernelBundle.h"
#include "../osd/error.h"

#include <stdio.h>
#ifdef _MSC_VER
#define snprintf _snprintf
#endif

static const char *clSource =
#include "clKernel.inc"
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
    _clCatmarkEdge(NULL),
    _clCatmarkVertexA(NULL),
    _clCatmarkVertexB(NULL),
    _clLoopEdge(NULL),
    _clLoopVertexA(NULL),
    _clLoopVertexB(NULL)
 {
}

OsdCLKernelBundle::~OsdCLKernelBundle() {

    if (_clBilinearEdge)
        clReleaseKernel(_clBilinearEdge);
    if (_clBilinearVertex)
        clReleaseKernel(_clBilinearVertex);

    if (_clCatmarkFace)
        clReleaseKernel(_clCatmarkFace);
    if (_clCatmarkEdge)
        clReleaseKernel(_clCatmarkEdge);
    if (_clCatmarkVertexA)
        clReleaseKernel(_clCatmarkVertexA);
    if (_clCatmarkVertexB)
        clReleaseKernel(_clCatmarkVertexB);

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
                           int numVertexElements, int numVaryingElements) {

    cl_int ciErrNum;

    _vdesc.Set( numVertexElements, numVaryingElements );

    char constantDefine[256];
    snprintf(constantDefine, sizeof(constantDefine),
             "#define NUM_VERTEX_ELEMENTS %d\n"
             "#define NUM_VARYING_ELEMENTS %d\n",
             numVertexElements, numVaryingElements);

    const char *sources[] = { constantDefine, clSource };

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

    _clBilinearEdge   = buildKernel(_clProgram, "computeBilinearEdge");
    _clBilinearVertex = buildKernel(_clProgram, "computeBilinearVertex");
    _clCatmarkFace    = buildKernel(_clProgram, "computeFace");
    _clCatmarkEdge    = buildKernel(_clProgram, "computeEdge");
    _clCatmarkVertexA = buildKernel(_clProgram, "computeVertexA");
    _clCatmarkVertexB = buildKernel(_clProgram, "computeVertexB");
    _clLoopEdge       = buildKernel(_clProgram, "computeEdge");
    _clLoopVertexA    = buildKernel(_clProgram, "computeVertexA");
    _clLoopVertexB    = buildKernel(_clProgram, "computeLoopVertexB");
    _clVertexEditAdd  = buildKernel(_clProgram, "editVertexAdd");

    return true;
}

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
