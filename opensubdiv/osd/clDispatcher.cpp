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
#include "../version.h"
#include "../osd/clDispatcher.h"
#include "../osd/local.h"

#if defined(_WIN32)
#include <windows.h>
#elif defined(__APPLE__)
#include <OpenCL/opencl.h>
#else
#include <GL/glx.h>
#include <CL/opencl.h>
#endif

#ifdef _MSC_VER
#define snprintf _snprintf
#endif

#include <stdio.h>
#include <string.h>

#define CL_CHECK_ERROR(x, ...) { if(x != CL_SUCCESS) { printf("ERROR %d : ", x); printf(__VA_ARGS__);} }

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

static const char *clSource =
#include "clKernel.inc"
    ;

std::vector<OsdClKernelDispatcher::ClKernel> OsdClKernelDispatcher::kernelRegistry;

// XXX: context and queue should be moved to client code
cl_context OsdClKernelDispatcher::_clContext = NULL;
cl_command_queue OsdClKernelDispatcher::_clQueue = NULL;

OsdClVertexBuffer::OsdClVertexBuffer(int numElements, int numVertices,
                                     cl_context clContext, cl_command_queue clQueue) :
    OsdGpuVertexBuffer(numElements, numVertices),
    _clVbo(NULL),
    _clQueue(clQueue) {

    // register vbo as cl resource
    cl_int ciErrNum;
    _clVbo = clCreateFromGLBuffer(clContext, CL_MEM_READ_WRITE, _vbo, &ciErrNum);
    CL_CHECK_ERROR(ciErrNum, "clCreateFromGLBuffer\n");
}

OsdClVertexBuffer::~OsdClVertexBuffer() {

    if (_clVbo)
        clReleaseMemObject(_clVbo);
}

void
OsdClVertexBuffer::UpdateData(const float *src, int numVertices) {

    size_t size = numVertices * _numElements * sizeof(float);
    Map();
    clEnqueueWriteBuffer(_clQueue, _clVbo, true, 0, size, src, 0, NULL, NULL);
    Unmap();
}

void *
OsdClVertexBuffer::Map() {

    clEnqueueAcquireGLObjects(_clQueue, 1, &_clVbo, 0, 0, 0);
}

void
OsdClVertexBuffer::Unmap() {

    clEnqueueReleaseGLObjects(_clQueue, 1, &_clVbo, 0, 0, 0);
}

// -------------------------------------------------------------------------------
OsdClKernelDispatcher::DeviceTable::~DeviceTable() {

    if (devicePtr) clReleaseMemObject(devicePtr);
}

void
OsdClKernelDispatcher::DeviceTable::Copy(cl_context context, int size, const void *table) {

    if (size > 0) {
        cl_int ciErrNum;
        if (devicePtr)
            clReleaseMemObject(devicePtr);
        devicePtr = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, size,
                                   const_cast<void*>(table), &ciErrNum);
        
        CL_CHECK_ERROR(ciErrNum, "Table copy %p\n", table);
    }
}

// -------------------------------------------------------------------------------------------

OsdClKernelDispatcher::OsdClKernelDispatcher(int levels) :
    OsdKernelDispatcher(levels) {

    _tables.resize(TABLE_MAX);

    if (_clContext == NULL) initCL();
}

OsdClKernelDispatcher::~OsdClKernelDispatcher() {
}

void
OsdClKernelDispatcher::CopyTable(int tableIndex, size_t size, const void *ptr) {

    _tables[tableIndex].Copy(_clContext, size, ptr);
}

OsdVertexBuffer *
OsdClKernelDispatcher::InitializeVertexBuffer(int numElements, int numVertices) {

    return new OsdClVertexBuffer(numElements, numVertices, _clContext, _clQueue);
}

void
OsdClKernelDispatcher::BindVertexBuffer(OsdVertexBuffer *vertex, OsdVertexBuffer *varying) {

    if (vertex)
        _currentVertexBuffer = dynamic_cast<OsdClVertexBuffer *>(vertex);
    else
        _currentVertexBuffer = NULL;

    if (varying)
        _currentVaryingBuffer = dynamic_cast<OsdClVertexBuffer *>(varying);
    else
        _currentVaryingBuffer = NULL;

    int numVertexElements = vertex ? vertex->GetNumElements() : 0;
    int numVaryingElements = varying ? varying->GetNumElements() : 0;

    if (_currentVertexBuffer) {
        _currentVertexBuffer->Map();
    }
    if (_currentVaryingBuffer) {
        _currentVaryingBuffer->Map();
    }

    // find cl kernel from registry (create it if needed)
    std::vector<ClKernel>::iterator it =
        std::find_if(kernelRegistry.begin(), kernelRegistry.end(),
                     ClKernel::Match(numVertexElements, numVaryingElements));

    if (it != kernelRegistry.end()) {
        _clKernel = &(*it);
    } else {
        kernelRegistry.push_back(ClKernel());
        _clKernel = &kernelRegistry.back();
        _clKernel->Compile(_clContext, numVertexElements, numVaryingElements);
    }
}

void
OsdClKernelDispatcher::UnbindVertexBuffer() {

    if (_currentVertexBuffer) {
        _currentVertexBuffer->Unmap();
    }
    if (_currentVaryingBuffer) {
        _currentVaryingBuffer->Unmap();
    }

    _currentVertexBuffer = NULL;
    _currentVaryingBuffer = NULL;
}

void
OsdClKernelDispatcher::Synchronize() {
    clFinish(_clQueue);
}

void
OsdClKernelDispatcher::ApplyCatmarkFaceVerticesKernel(FarMesh<OsdVertex> * mesh, int offset,
                                                    int level, int start, int end, void * data) const {

    cl_int ciErrNum;
    size_t globalWorkSize[1] = { end-start };
    cl_kernel kernel = _clKernel->GetCatmarkFaceKernel();

    clSetKernelArg(kernel, 0, sizeof(cl_mem), GetVertexBuffer());
    clSetKernelArg(kernel, 1, sizeof(cl_mem), GetVaryingBuffer());
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &_tables[F_IT].devicePtr);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &_tables[F_ITa].devicePtr);
    clSetKernelArg(kernel, 4, sizeof(int), &_tableOffsets[F_IT][level-1]);
    clSetKernelArg(kernel, 5, sizeof(int), &_tableOffsets[F_ITa][level-1]);
    clSetKernelArg(kernel, 6, sizeof(int), &offset);
    clSetKernelArg(kernel, 7, sizeof(int), &start);
    clSetKernelArg(kernel, 8, sizeof(int), &end);

    ciErrNum = clEnqueueNDRangeKernel(_clQueue, kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    CL_CHECK_ERROR(ciErrNum, "face kernel lv[%d] %d\n", level, ciErrNum);
}

void
OsdClKernelDispatcher::ApplyCatmarkEdgeVerticesKernel(FarMesh<OsdVertex> * mesh, int offset, 
                                                    int level, int start, int end, void * data) const {

    cl_int ciErrNum;
    size_t globalWorkSize[1] = { end-start };
    cl_kernel kernel = _clKernel->GetCatmarkEdgeKernel();

    clSetKernelArg(kernel, 0, sizeof(cl_mem), GetVertexBuffer());
    clSetKernelArg(kernel, 1, sizeof(cl_mem), GetVaryingBuffer());
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &_tables[E_IT].devicePtr);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &_tables[E_W].devicePtr);
    clSetKernelArg(kernel, 4, sizeof(int), &_tableOffsets[E_IT][level-1]);
    clSetKernelArg(kernel, 5, sizeof(int), &_tableOffsets[E_W][level-1]);
    clSetKernelArg(kernel, 6, sizeof(int), &offset);
    clSetKernelArg(kernel, 7, sizeof(int), &start);
    clSetKernelArg(kernel, 8, sizeof(int), &end);
            
    ciErrNum = clEnqueueNDRangeKernel(_clQueue, kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    CL_CHECK_ERROR(ciErrNum, "edge kernel %d\n", ciErrNum);
}

void
OsdClKernelDispatcher::ApplyCatmarkVertexVerticesKernelB(FarMesh<OsdVertex> * mesh, int offset,
                                                       int level, int start, int end, void * data) const {

    cl_int ciErrNum;
    size_t globalWorkSize[1] = { end-start };
    cl_kernel kernel = _clKernel->GetCatmarkVertexKernelB();

    clSetKernelArg(kernel, 0, sizeof(cl_mem), GetVertexBuffer());
    clSetKernelArg(kernel, 1, sizeof(cl_mem), GetVaryingBuffer());
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &_tables[V_ITa].devicePtr);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &_tables[V_IT].devicePtr);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &_tables[V_W].devicePtr);
    clSetKernelArg(kernel, 5, sizeof(int), &_tableOffsets[V_ITa][level-1]);
    clSetKernelArg(kernel, 6, sizeof(int), &_tableOffsets[V_IT][level-1]);
    clSetKernelArg(kernel, 7, sizeof(int), &_tableOffsets[V_W][level-1]);
    clSetKernelArg(kernel, 8, sizeof(int), (void*)&offset);
    clSetKernelArg(kernel, 9, sizeof(int), (void*)&start);
    clSetKernelArg(kernel, 10, sizeof(int), (void*)&end);
    ciErrNum = clEnqueueNDRangeKernel(_clQueue, kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    CL_CHECK_ERROR(ciErrNum, "vertex kernel 1 %d\n", ciErrNum);
}

void
OsdClKernelDispatcher::ApplyCatmarkVertexVerticesKernelA(FarMesh<OsdVertex> * mesh, int offset,
                                                       bool pass, int level, int start, int end, void * data) const {

    cl_int ciErrNum;
    size_t globalWorkSize[1] = { end-start };
    int ipass = pass;
    cl_kernel kernel = _clKernel->GetCatmarkVertexKernelA();

    clSetKernelArg(kernel, 0, sizeof(cl_mem), GetVertexBuffer());
    clSetKernelArg(kernel, 1, sizeof(cl_mem), GetVaryingBuffer());
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &_tables[V_ITa].devicePtr);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &_tables[V_W].devicePtr);
    clSetKernelArg(kernel, 4, sizeof(int), &_tableOffsets[V_ITa][level-1]);
    clSetKernelArg(kernel, 5, sizeof(int), &_tableOffsets[V_W][level-1]);
    clSetKernelArg(kernel, 6, sizeof(int), (void*)&offset);
    clSetKernelArg(kernel, 7, sizeof(int), (void*)&start);
    clSetKernelArg(kernel, 8, sizeof(int), (void*)&end);
    clSetKernelArg(kernel, 9, sizeof(int), (void*)&ipass);

    ciErrNum = clEnqueueNDRangeKernel(_clQueue, kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    CL_CHECK_ERROR(ciErrNum, "vertex kernel 2 %d\n", ciErrNum);
}

void
OsdClKernelDispatcher::ApplyLoopEdgeVerticesKernel(FarMesh<OsdVertex> * mesh, int offset,
                                                 int level, int start, int end, void * data) const {

    cl_int ciErrNum;
    size_t globalWorkSize[1] = { end-start };
    cl_kernel kernel = _clKernel->GetLoopEdgeKernel();

    clSetKernelArg(kernel, 0, sizeof(cl_mem), GetVertexBuffer());
    clSetKernelArg(kernel, 1, sizeof(cl_mem), GetVaryingBuffer());
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &_tables[E_IT].devicePtr);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &_tables[E_W].devicePtr);
    clSetKernelArg(kernel, 4, sizeof(int), &_tableOffsets[E_IT][level-1]);
    clSetKernelArg(kernel, 5, sizeof(int), &_tableOffsets[E_W][level-1]);
    clSetKernelArg(kernel, 6, sizeof(int), &offset);
    clSetKernelArg(kernel, 7, sizeof(int), &start);
    clSetKernelArg(kernel, 8, sizeof(int), &end);
    ciErrNum = clEnqueueNDRangeKernel(_clQueue, kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    CL_CHECK_ERROR(ciErrNum, "edge kernel %d\n", ciErrNum);
}

void
OsdClKernelDispatcher::ApplyLoopVertexVerticesKernelB(FarMesh<OsdVertex> * mesh, int offset,
                                                    int level, int start, int end, void * data) const {

    cl_int ciErrNum;
    size_t globalWorkSize[1] = { end-start };
    cl_kernel kernel = _clKernel->GetLoopVertexKernelB();

    clSetKernelArg(kernel, 0, sizeof(cl_mem), GetVertexBuffer());
    clSetKernelArg(kernel, 1, sizeof(cl_mem), GetVaryingBuffer());
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &_tables[V_ITa].devicePtr);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &_tables[V_IT].devicePtr);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &_tables[V_W].devicePtr);
    clSetKernelArg(kernel, 5, sizeof(int), &_tableOffsets[V_ITa][level-1]);
    clSetKernelArg(kernel, 6, sizeof(int), &_tableOffsets[V_IT][level-1]);
    clSetKernelArg(kernel, 7, sizeof(int), &_tableOffsets[V_W][level-1]);
    clSetKernelArg(kernel, 8, sizeof(int), &offset);
    clSetKernelArg(kernel, 9, sizeof(int), &start);
    clSetKernelArg(kernel, 10, sizeof(int), &end);
    ciErrNum = clEnqueueNDRangeKernel(_clQueue, kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    CL_CHECK_ERROR(ciErrNum, "vertex kernel 1 %d\n", ciErrNum);
}

void
OsdClKernelDispatcher::ApplyLoopVertexVerticesKernelA(FarMesh<OsdVertex> * mesh, int offset,
                                                    bool pass, int level, int start, int end, void * data) const {

    cl_int ciErrNum;
    size_t globalWorkSize[1] = { end-start };
    int ipass = pass;
    cl_kernel kernel = _clKernel->GetLoopVertexKernelA();

    clSetKernelArg(kernel, 0, sizeof(cl_mem), GetVertexBuffer());
    clSetKernelArg(kernel, 1, sizeof(cl_mem), GetVaryingBuffer());
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &_tables[V_ITa].devicePtr);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &_tables[V_W].devicePtr);
    clSetKernelArg(kernel, 4, sizeof(int), &_tableOffsets[V_ITa][level-1]);
    clSetKernelArg(kernel, 5, sizeof(int), &_tableOffsets[V_W][level-1]);
    clSetKernelArg(kernel, 6, sizeof(int), (void*)&offset);
    clSetKernelArg(kernel, 7, sizeof(int), (void*)&start);
    clSetKernelArg(kernel, 8, sizeof(int), (void*)&end);
    clSetKernelArg(kernel, 9, sizeof(int), (void*)&ipass);
    ciErrNum = clEnqueueNDRangeKernel(_clQueue, kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    CL_CHECK_ERROR(ciErrNum, "vertex kernel 2 %d\n", ciErrNum);
}

// XXX: initCL should be removed from libosd
void 
OsdClKernelDispatcher::initCL() {

    cl_int ciErrNum;

    cl_platform_id cpPlatform = 0;
    cl_uint num_platforms;
    ciErrNum = clGetPlatformIDs(0, NULL, &num_platforms);
    if (ciErrNum != CL_SUCCESS) {
        OSD_ERROR("Error %i in clGetPlatformIDs call.\n", ciErrNum);
        exit(1);
    }
    if (num_platforms == 0) {
        OSD_ERROR("No OpenCL platform found.\n");
        exit(1);
    }
    cl_platform_id *clPlatformIDs;
    clPlatformIDs = new cl_platform_id[num_platforms];
    ciErrNum = clGetPlatformIDs(num_platforms, clPlatformIDs, NULL);
    char chBuffer[1024];
    for (cl_uint i = 0; i < num_platforms; ++i) {
        ciErrNum = clGetPlatformInfo(clPlatformIDs[i], CL_PLATFORM_NAME, 1024, chBuffer,NULL);
        if (ciErrNum == CL_SUCCESS) {
            cpPlatform = clPlatformIDs[i];
        }
    }
    // -------------
    cl_device_id cdDevice;
    clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &cdDevice, NULL);
    
#if defined(_WIN32)
    cl_context_properties props[] = {
        CL_GL_CONTEXT_KHR, (cl_context_properties)wglGetCurrentContext(),
        CL_WGL_HDC_KHR, (cl_context_properties)wglGetCurrentDC(),
        CL_CONTEXT_PLATFORM, (cl_context_properties)cpPlatform,
        0
    };
#elif defined(__APPLE__)
    CGLContextObj kCGLContext = CGLGetCurrentContext();
    CGLShareGroupObj kCGLShareGroup = CGLGetShareGroup(kCGLContext);
    cl_context_properties props[] = {
        CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE, (cl_context_properties)kCGLShareGroup,
        0
    };
#else
    cl_context_properties props[] = {
        CL_GL_CONTEXT_KHR, (cl_context_properties)glXGetCurrentContext(),
        CL_GLX_DISPLAY_KHR, (cl_context_properties)glXGetCurrentDisplay(),
        CL_CONTEXT_PLATFORM, (cl_context_properties)cpPlatform,
        0
    };
#endif

    // XXX context creation should be moved to client code
    _clContext = clCreateContext(props, 1, &cdDevice, NULL, NULL, &ciErrNum);
    CL_CHECK_ERROR(ciErrNum, "clCreateContext\n");

    _clQueue = clCreateCommandQueue(_clContext, cdDevice, 0, &ciErrNum);
    CL_CHECK_ERROR(ciErrNum, "clCreateCommandQueue\n");
}

void
OsdClKernelDispatcher::uninitCL() {

    // XXX: who calls this function...
    clReleaseCommandQueue(_clQueue);
    clReleaseContext(_clContext);
}

// ------------------------------------------------------------------

OsdClKernelDispatcher::ClKernel::ClKernel() :
    _clCatmarkFace(NULL),
    _clCatmarkEdge(NULL),
    _clCatmarkVertexA(NULL),
    _clCatmarkVertexB(NULL),
    _clLoopEdge(NULL),
    _clLoopVertexA(NULL),
    _clLoopVertexB(NULL),
    _clProgram(NULL) {
}

OsdClKernelDispatcher::ClKernel::~ClKernel() {

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

bool
OsdClKernelDispatcher::ClKernel::Compile(cl_context clContext, int numVertexElements, int numVaryingElements) {

    cl_int ciErrNum;

    _numVertexElements = numVertexElements;
    _numVaryingElements = numVaryingElements;

    char constantDefine[256];
    snprintf(constantDefine, 256, "#define NUM_VERTEX_ELEMENTS %d\n"
             "#define NUM_VARYING_ELEMENTS %d\n", numVertexElements, numVaryingElements);

    const char *sources[] = { constantDefine, clSource };

    _clProgram = clCreateProgramWithSource(clContext, 2, sources, 0, &ciErrNum);
    CL_CHECK_ERROR(ciErrNum, "clCreateProgramWithSource\n");

    ciErrNum = clBuildProgram(_clProgram, 0, NULL, NULL, NULL, NULL);
    if (ciErrNum != CL_SUCCESS) {
        OSD_ERROR("ERROR in clBuildProgram %d\n", ciErrNum);
        //char cBuildLog[10240];
        //clGetProgramBuildInfo(_clProgram, cdDevice, CL_PROGRAM_BUILD_LOG,
        //                      sizeof(cBuildLog), cBuildLog, NULL);
        //OSD_ERROR(cBuildLog);
        return false;
    }

    // -------

    _clCatmarkFace = clCreateKernel(_clProgram, "computeFace", &ciErrNum);
    CL_CHECK_ERROR(ciErrNum, "clCreateKernel face\n");
    _clCatmarkEdge = clCreateKernel(_clProgram, "computeEdge", &ciErrNum);
    CL_CHECK_ERROR(ciErrNum, "clCreateKernel edge\n");
    _clCatmarkVertexA = clCreateKernel(_clProgram, "computeVertexA", &ciErrNum);
    CL_CHECK_ERROR(ciErrNum, "clCreateKernel vertex a\n");
    _clCatmarkVertexB = clCreateKernel(_clProgram, "computeVertexB", &ciErrNum);
    CL_CHECK_ERROR(ciErrNum, "clCreateKernel vertex b\n");
    _clLoopEdge = clCreateKernel(_clProgram, "computeEdge", &ciErrNum);
    CL_CHECK_ERROR(ciErrNum, "clCreateKernel edge\n");
    _clLoopVertexA = clCreateKernel(_clProgram, "computeVertexA", &ciErrNum);
    CL_CHECK_ERROR(ciErrNum, "clCreateKernel vertex a\n");
    _clLoopVertexB = clCreateKernel(_clProgram, "computeLoopVertexB", &ciErrNum);
    CL_CHECK_ERROR(ciErrNum, "clCreateKernel vertex b\n");

    return true;
}

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv
