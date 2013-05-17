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
