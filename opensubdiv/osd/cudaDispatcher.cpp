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
#include "../osd/mutex.h"
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

void OsdCudaEditVertexAdd(float *vertex, int numUserVertexElements, int primVarOffset, int primVarWidth, int numVertices, int *editIndices, float *editValues);

}

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

OsdCudaVertexBuffer::OsdCudaVertexBuffer(int numElements, int numVertices) :
    OsdGpuVertexBuffer(numElements, numVertices) {

    // register vbo as cuda resource
    cudaGraphicsGLRegisterBuffer(&_cudaResource, _vbo, cudaGraphicsMapFlagsNone);
}

void
OsdCudaVertexBuffer::UpdateData(const float *src, int numVertices) {

    void *dst = Map();
    cudaMemcpy(dst, src, _numElements * numVertices * sizeof(float), cudaMemcpyHostToDevice);
    Unmap();
}

void *
OsdCudaVertexBuffer::Map() {

    size_t num_bytes;
    void *ptr;

    cudaGraphicsMapResources(1, &_cudaResource, 0);
    cudaGraphicsResourceGetMappedPointer(&ptr, &num_bytes, _cudaResource);
    return ptr;
}

void
OsdCudaVertexBuffer::Unmap() {
    cudaGraphicsUnmapResources(1, &_cudaResource, 0);
}

OsdCudaVertexBuffer::~OsdCudaVertexBuffer() {
    cudaGraphicsUnregisterResource(_cudaResource);
}

// -------------------------------------------------------------------------------
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
// -------------------------------------------------------------------------------

OsdCudaKernelDispatcher::OsdCudaKernelDispatcher(int levels)
    : OsdKernelDispatcher(levels)
{
    _tables.resize(TABLE_MAX);
}


OsdCudaKernelDispatcher::~OsdCudaKernelDispatcher() {
}

void
OsdCudaKernelDispatcher::CopyTable(int tableIndex, size_t size, const void *ptr) {

    _tables[tableIndex].Copy((int)size, ptr);
}

void
OsdCudaKernelDispatcher::AllocateEditTables(int n) {

    _editTables.resize(n*2);
    _edits.resize(n);
}

void
OsdCudaKernelDispatcher::UpdateEditTable(int tableIndex, const FarTable<unsigned int> &offsets, const FarTable<float> &values,
                                         int operation, int primVarOffset, int primVarWidth) {

    _editTables[tableIndex*2+0].Copy(offsets.GetMemoryUsed(), offsets[0]);
    _editTables[tableIndex*2+1].Copy(values.GetMemoryUsed(), values[0]);

    _edits[tableIndex].offsetOffsets.resize(_maxLevel);
    _edits[tableIndex].valueOffsets.resize(_maxLevel);
    _edits[tableIndex].numEdits.resize(_maxLevel);
    for (int i = 0; i < _maxLevel; ++i) {
        _edits[tableIndex].offsetOffsets[i] = (int)(offsets[i] - offsets[0]);
        _edits[tableIndex].valueOffsets[i] = (int)(values[i] - values[0]);
        _edits[tableIndex].numEdits[i] = offsets.GetNumElements(i);
    }
    _edits[tableIndex].operation = operation;
    _edits[tableIndex].primVarOffset = primVarOffset;
    _edits[tableIndex].primVarWidth = primVarWidth;
}

OsdVertexBuffer *
OsdCudaKernelDispatcher::InitializeVertexBuffer(int numElements, int numVertices)
{
    return new OsdCudaVertexBuffer(numElements, numVertices);
}

void
OsdCudaKernelDispatcher::BindVertexBuffer(OsdVertexBuffer *vertex, OsdVertexBuffer *varying) {

    if (vertex)
        _currentVertexBuffer = dynamic_cast<OsdCudaVertexBuffer *>(vertex);
    else
        _currentVertexBuffer = NULL;

    if (varying)
        _currentVaryingBuffer = dynamic_cast<OsdCudaVertexBuffer *>(varying);
    else
        _currentVaryingBuffer = NULL;

    if (_currentVertexBuffer) {
        _deviceVertices = (float*)_currentVertexBuffer->Map();
        // XXX todo remove _numVertexElements
        _numVertexElements = _currentVertexBuffer->GetNumElements();
    } else {
        _numVertexElements = 0;
    }

    if (_currentVaryingBuffer) {
        _deviceVaryings = (float*)_currentVaryingBuffer->Map();
        _numVaryingElements = _currentVaryingBuffer->GetNumElements();
    } else {
        _numVaryingElements = 0;
    }
}

void
OsdCudaKernelDispatcher::UnbindVertexBuffer()
{
    if (_currentVertexBuffer){
        _currentVertexBuffer->Unmap();
    }
    if (_currentVaryingBuffer)
        _currentVaryingBuffer->Unmap();

    _currentVertexBuffer = NULL;
    _currentVaryingBuffer = NULL;
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

void
OsdCudaKernelDispatcher::ApplyVertexEdits(FarMesh<OsdVertex> *mesh, int offset, int level, void * clientdata) const {

    for (int i=0; i<(int)_edits.size(); ++i) {
        const VertexEditArrayInfo &info = _edits[i];

        if (info.operation == FarVertexEdit::Add) {
            OsdCudaEditVertexAdd(_deviceVertices, _numVertexElements-3, info.primVarOffset, info.primVarWidth, info.numEdits[level-1],
                                 (int*)_editTables[i*2+0].devicePtr + info.offsetOffsets[level-1],
                                 (float*)_editTables[i*2+1].devicePtr + info.valueOffsets[level-1]);
        } else if (info.operation == FarVertexEdit::Set) {
            // XXXX TODO
        }
    }
}

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv
