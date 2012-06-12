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
#include "../osd/cudaDispatcher.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

extern "C" {

void OsdCudaComputeFace(float *vertex, float *varying, int numUserVertexElements, int numVaryingElements, int *F_IT, int *F_ITa, int offset, int start, int end);

void OsdCudaComputeEdge(float *vertex, float *varying, int numUserVertexElements, int numVaryingElements, int *E_IT, float *E_W, int offset, int start, int end);

void OsdCudaComputeVertexA(float *vertex, float *varying, int numUserVertexElements, int numVaryingElements, int *V_ITa, float *V_W, int offset, int start, int end, int pass);

void OsdCudaComputeVertexB(float *vertex, float *varying, int numUserVertexElements, int numVaryingElements, int *V_ITa, int *V_IT, float *V_W, int offset, int start, int end);

void OsdCudaComputeLoopVertexB(float *vertex, float *varying, int numUserVertexElements, int numVaryingElements, int *V_ITa, int *V_IT, float *V_W, int offset, int start, int end);

void OsdCudaComputeBilinearEdge(float *vertex, float *varying, int numUserVertexElements, int numVaryingElements, int *E_IT, int offset, int start, int end);

void OsdCudaComputeBilinearVertex(float *vertex, float *varying, int numUserVertexElements, int numVaryingElements, int *V_ITa, int offset, int start, int end);

}

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {


OsdCudaKernelDispatcher::DeviceTable::~DeviceTable() {

    if (devicePtr) cudaFree(devicePtr);
}

void
OsdCudaKernelDispatcher::DeviceTable::Copy(int size, const void *ptr) {

    if (devicePtr)
        cudaFree(devicePtr);
    cudaMalloc(&devicePtr, size);
    cudaMemcpy(devicePtr, ptr, size, cudaMemcpyHostToDevice);
}

OsdCudaKernelDispatcher::OsdCudaKernelDispatcher(int levels, int numVertexElements, int numVaryingElements)
    : OsdKernelDispatcher(levels),
      _cudaVertexResource(NULL),
      _cudaVaryingResource(NULL),
      _numVertexElements(numVertexElements),
      _numVaryingElements(numVaryingElements)
{
    _tables.resize(TABLE_MAX);
}


OsdCudaKernelDispatcher::~OsdCudaKernelDispatcher() {
    cudaDeviceReset(); // XXX: necessary?
}

void
OsdCudaKernelDispatcher::CopyTable(int tableIndex, size_t size, const void *ptr) {

    _tables[tableIndex].Copy(size, ptr);
}

void
OsdCudaKernelDispatcher::BeginLaunchKernel() { }

void
OsdCudaKernelDispatcher::EndLaunchKernel() { }

void
OsdCudaKernelDispatcher::BindVertexBuffer(GLuint vertexBuffer, GLuint varyingBuffer) {

    cudaGraphicsGLRegisterBuffer(&_cudaVertexResource, vertexBuffer, cudaGraphicsMapFlagsWriteDiscard);
    
    if (varyingBuffer)
        cudaGraphicsGLRegisterBuffer(&_cudaVaryingResource, varyingBuffer, cudaGraphicsMapFlagsWriteDiscard);
}

void
OsdCudaKernelDispatcher::UpdateVertexBuffer(size_t size, void *ptr) {

    cudaMemcpy(_deviceVertices, ptr, size, cudaMemcpyHostToDevice);
}

void
OsdCudaKernelDispatcher::UpdateVaryingBuffer(size_t size, void *ptr) {

    if(_cudaVaryingResource)
        cudaMemcpy(_deviceVaryings, ptr, size, cudaMemcpyHostToDevice);
}

void
OsdCudaKernelDispatcher::MapVertexBuffer() {

    size_t num_bytes;
    cudaGraphicsMapResources(1, &_cudaVertexResource, 0);
    cudaGraphicsResourceGetMappedPointer((void **)&_deviceVertices, &num_bytes, _cudaVertexResource);
}

void
OsdCudaKernelDispatcher::MapVaryingBuffer()
{
    if (_cudaVaryingResource) {
        size_t num_bytes;
        cudaGraphicsMapResources(1, &_cudaVaryingResource, 0);
        cudaGraphicsResourceGetMappedPointer((void **)&_deviceVaryings, &num_bytes, _cudaVaryingResource);
    }
}

void
OsdCudaKernelDispatcher::UnmapVertexBuffer() {

    cudaGraphicsUnmapResources(1, &_cudaVertexResource, 0);
}

void
OsdCudaKernelDispatcher::UnmapVaryingBuffer() {

    if (_cudaVaryingResource)
        cudaGraphicsUnmapResources(1, &_cudaVaryingResource, 0);
}

void
OsdCudaKernelDispatcher::Synchronize() {

    cudaThreadSynchronize();
}

void
OsdCudaKernelDispatcher::ApplyBilinearFaceVerticesKernel(FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const {
    OsdCudaComputeFace(_deviceVertices, _deviceVaryings,
                       _numVertexElements-3, _numVaryingElements,
                       (int*)_tables[F_IT].devicePtr + _tableOffsets[F_IT][level-1],
                       (int*)_tables[F_ITa].devicePtr + _tableOffsets[F_ITa][level-1],
                       offset, start, end);
}

void
OsdCudaKernelDispatcher::ApplyBilinearEdgeVerticesKernel(FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const {

    OsdCudaComputeBilinearEdge(_deviceVertices, _deviceVaryings,
                               _numVertexElements-3, _numVaryingElements,
                               (int*)_tables[E_IT].devicePtr + _tableOffsets[E_IT][level-1],
                               offset, start, end);
}

void
OsdCudaKernelDispatcher::ApplyBilinearVertexVerticesKernel(FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const {
    
    OsdCudaComputeBilinearVertex(_deviceVertices, _deviceVaryings,
                                 _numVertexElements-3, _numVaryingElements,
                                 (int*)_tables[V_ITa].devicePtr + _tableOffsets[V_ITa][level-1],
                                 offset, start, end);
}

void
OsdCudaKernelDispatcher::ApplyCatmarkFaceVerticesKernel(FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const {
    OsdCudaComputeFace(_deviceVertices, _deviceVaryings,
                       _numVertexElements-3, _numVaryingElements,
                       (int*)_tables[F_IT].devicePtr + _tableOffsets[F_IT][level-1],
                       (int*)_tables[F_ITa].devicePtr + _tableOffsets[F_ITa][level-1],
                       offset, start, end);
}

void
OsdCudaKernelDispatcher::ApplyCatmarkEdgeVerticesKernel(FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const {

    OsdCudaComputeEdge(_deviceVertices, _deviceVaryings,
                       _numVertexElements-3, _numVaryingElements,
                       (int*)_tables[E_IT].devicePtr + _tableOffsets[E_IT][level-1],
                       (float*)_tables[E_W].devicePtr + _tableOffsets[E_W][level-1],
                       offset, start, end);
}

void
OsdCudaKernelDispatcher::ApplyCatmarkVertexVerticesKernelB(FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const {

    OsdCudaComputeVertexB(_deviceVertices, _deviceVaryings,
                          _numVertexElements-3, _numVaryingElements,
                          (int*)_tables[V_ITa].devicePtr + _tableOffsets[V_ITa][level-1],
                          (int*)_tables[V_IT].devicePtr + _tableOffsets[V_IT][level-1],
                          (float*)_tables[V_W].devicePtr + _tableOffsets[V_W][level-1],
                          offset, start, end);
}

void
OsdCudaKernelDispatcher::ApplyCatmarkVertexVerticesKernelA(FarMesh<OsdVertex> * mesh, int offset, bool pass, int level, int start, int end, void * data) const {

    OsdCudaComputeVertexA(_deviceVertices, _deviceVaryings,
                          _numVertexElements-3, _numVaryingElements,
                          (int*)_tables[V_ITa].devicePtr + _tableOffsets[V_ITa][level-1],
                          (float*)_tables[V_W].devicePtr + _tableOffsets[V_W][level-1],
                          offset, start, end, pass);
}

void
OsdCudaKernelDispatcher::ApplyLoopEdgeVerticesKernel(FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const {

    OsdCudaComputeEdge(_deviceVertices, _deviceVaryings,
                       _numVertexElements-3, _numVaryingElements,
                       (int*)_tables[E_IT].devicePtr + _tableOffsets[E_IT][level-1],
                       (float*)_tables[E_W].devicePtr + _tableOffsets[E_W][level-1],
                       offset, start, end);
}

void
OsdCudaKernelDispatcher::ApplyLoopVertexVerticesKernelB(FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const {

    OsdCudaComputeLoopVertexB(_deviceVertices, _deviceVaryings,
                              _numVertexElements-3, _numVaryingElements,
                              (int*)_tables[V_ITa].devicePtr + _tableOffsets[V_ITa][level-1],
                              (int*)_tables[V_IT].devicePtr + _tableOffsets[V_IT][level-1],
                              (float*)_tables[V_W].devicePtr + _tableOffsets[V_W][level-1],
                              offset, start, end);
}

void
OsdCudaKernelDispatcher::ApplyLoopVertexVerticesKernelA(FarMesh<OsdVertex> * mesh, int offset, bool pass, int level, int start, int end, void * data) const {
    OsdCudaComputeVertexA(_deviceVertices, _deviceVaryings,
                          _numVertexElements-3, _numVaryingElements,
                          (int*)_tables[V_ITa].devicePtr + _tableOffsets[V_ITa][level-1],
                          (float*)_tables[V_W].devicePtr + _tableOffsets[V_W][level-1],
                          offset, start, end, pass);
}

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv
