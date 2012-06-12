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
#include "../osd/cpuDispatcher.h"
#include "../osd/cpuKernel.h"

#include <GL/glew.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

OsdCpuKernelDispatcher::DeviceTable::~DeviceTable() {

    if (devicePtr) 
        free(devicePtr);
}

void
OsdCpuKernelDispatcher::DeviceTable::Copy( int size, const void *table ) {

    if (size > 0) {
        if (devicePtr) 
	    free(devicePtr);
        devicePtr = malloc(size);
        memcpy(devicePtr, table, size);
    }
}

OsdCpuKernelDispatcher::OsdCpuKernelDispatcher( int levels )
    : OsdKernelDispatcher(levels) {
    _tables.resize(TABLE_MAX);
}

OsdCpuKernelDispatcher::~OsdCpuKernelDispatcher() { }

void
OsdCpuKernelDispatcher::CopyTable(int tableIndex, size_t size, const void *ptr) {

    _tables[tableIndex].Copy(size, ptr);
}

void
OsdCpuKernelDispatcher::BeginLaunchKernel() { }

void
OsdCpuKernelDispatcher::EndLaunchKernel() { }

OsdVertexBuffer *
OsdCpuKernelDispatcher::InitializeVertexBuffer(int numElements, int count)
{
    return new OsdCpuVertexBuffer(numElements, count);
}

void
OsdCpuKernelDispatcher::BindVertexBuffer(OsdVertexBuffer *vertex, OsdVertexBuffer *varying) {

    _vertexBuffer = dynamic_cast<OsdCpuVertexBuffer *>(vertex);
    _varyingBuffer = dynamic_cast<OsdCpuVertexBuffer *>(varying);
}

void
OsdCpuKernelDispatcher::UnbindVertexBuffer()
{
    _vertexBuffer = NULL;
    _varyingBuffer = NULL;
}

void
OsdCpuKernelDispatcher::Synchronize() { }

void
OsdCpuKernelDispatcher::ApplyCatmarkFaceVerticesKernel( FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const {
    
    VertexDescriptor vd(_vertexBuffer->GetNumElements(), _varyingBuffer->GetNumElements());

    computeFace(&vd, _vertexBuffer->GetCpuBuffer(), _varyingBuffer->GetCpuBuffer(),
                (int*)_tables[F_IT].devicePtr + _tableOffsets[F_IT][level-1],
                (int*)_tables[F_ITa].devicePtr + _tableOffsets[F_ITa][level-1],
                offset, start, end);

    float *p = _vertexBuffer->GetCpuBuffer();
    for(int i = 0; i < 150; i+=3){
        printf("%f %f %f\n", p[0], p[1], p[2]);
        p+=3;
    }
}

void
OsdCpuKernelDispatcher::ApplyCatmarkEdgeVerticesKernel( FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const {
    
    VertexDescriptor vd(_vertexBuffer->GetNumElements(), _varyingBuffer->GetNumElements());

    computeEdge(&vd, _vertexBuffer->GetCpuBuffer(), _varyingBuffer->GetCpuBuffer(),
                (int*)_tables[E_IT].devicePtr + _tableOffsets[E_IT][level-1],
                (float*)_tables[E_W].devicePtr + _tableOffsets[E_W][level-1],
                offset, 
                start, end);
}

void
OsdCpuKernelDispatcher::ApplyCatmarkVertexVerticesKernelB( FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const {
    
    VertexDescriptor vd(_vertexBuffer->GetNumElements(), _varyingBuffer->GetNumElements());

    computeVertexB(&vd, _vertexBuffer->GetCpuBuffer(), _varyingBuffer->GetCpuBuffer(),
                   (int*)_tables[V_ITa].devicePtr + _tableOffsets[V_ITa][level-1],
                   (int*)_tables[V_IT].devicePtr + _tableOffsets[V_IT][level-1],
                   (float*)_tables[V_W].devicePtr + _tableOffsets[V_W][level-1],
                   offset, start, end);
}

void
OsdCpuKernelDispatcher::ApplyCatmarkVertexVerticesKernelA( FarMesh<OsdVertex> * mesh, int offset, bool pass, int level, int start, int end, void * data) const {
    
    VertexDescriptor vd(_vertexBuffer->GetNumElements(), _varyingBuffer->GetNumElements());

    computeVertexA(&vd, _vertexBuffer->GetCpuBuffer(), _varyingBuffer->GetCpuBuffer(),
                   (int*)_tables[V_ITa].devicePtr + _tableOffsets[V_ITa][level-1],
                   (float*)_tables[V_W].devicePtr + _tableOffsets[V_W][level-1],
                   offset, start, end, pass);
}

void
OsdCpuKernelDispatcher::ApplyLoopEdgeVerticesKernel( FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const {
    
    VertexDescriptor vd(_vertexBuffer->GetNumElements(), _varyingBuffer->GetNumElements());

    computeEdge(&vd, _vertexBuffer->GetCpuBuffer(), _varyingBuffer->GetCpuBuffer(),
                (int*)_tables[E_IT].devicePtr + _tableOffsets[E_IT][level-1],
                (float*)_tables[E_W].devicePtr + _tableOffsets[E_W][level-1],
                offset, 
                start, end);
}

void
OsdCpuKernelDispatcher::ApplyLoopVertexVerticesKernelB( FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const {
    
    VertexDescriptor vd(_vertexBuffer->GetNumElements(), _varyingBuffer->GetNumElements());

    computeLoopVertexB(&vd, _vertexBuffer->GetCpuBuffer(), _varyingBuffer->GetCpuBuffer(),
                       (int*)_tables[V_ITa].devicePtr + _tableOffsets[V_ITa][level-1],
                       (int*)_tables[V_IT].devicePtr + _tableOffsets[V_IT][level-1],
                       (float*)_tables[V_W].devicePtr + _tableOffsets[V_W][level-1],
                       offset, start, end);
}

void
OsdCpuKernelDispatcher::ApplyLoopVertexVerticesKernelA( FarMesh<OsdVertex> * mesh, int offset, bool pass, int level, int start, int end, void * data) const {
    
    VertexDescriptor vd(_vertexBuffer->GetNumElements(), _varyingBuffer->GetNumElements());

    computeVertexA(&vd, _vertexBuffer->GetCpuBuffer(), _varyingBuffer->GetCpuBuffer(),
                   (int*)_tables[V_ITa].devicePtr + _tableOffsets[V_ITa][level-1],
                   (float*)_tables[V_W].devicePtr + _tableOffsets[V_W][level-1],
                   offset, start, end, pass);
}

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

