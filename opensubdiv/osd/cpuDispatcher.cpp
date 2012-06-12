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

OsdCpuKernelDispatcher::OsdCpuKernelDispatcher( int levels, int numVertexElements, int numVaryingElements ) 
    : OsdKernelDispatcher(levels), _vertexBuffer(0), _varyingBuffer(0), _vbo(NULL), _varyingVbo(NULL), _numVertexElements(numVertexElements), _numVaryingElements(numVaryingElements) {
    _tables.resize(TABLE_MAX);
}

OsdCpuKernelDispatcher::~OsdCpuKernelDispatcher() {

    if(_vbo) 
        delete[] _vbo;

    if(_varyingVbo) 
        delete[] _varyingVbo;
}

void
OsdCpuKernelDispatcher::CopyTable(int tableIndex, size_t size, const void *ptr) {

    _tables[tableIndex].Copy(size, ptr);
}

void
OsdCpuKernelDispatcher::BeginLaunchKernel() { }

void
OsdCpuKernelDispatcher::EndLaunchKernel() { }

void
OsdCpuKernelDispatcher::BindVertexBuffer(GLuint vertexBuffer, GLuint varyingBuffer) {

    _vertexBuffer = vertexBuffer;
    _varyingBuffer = varyingBuffer;
}

void
OsdCpuKernelDispatcher::UpdateVertexBuffer(size_t size, void *ptr) {

    memcpy(_vbo, ptr, size);
}

void
OsdCpuKernelDispatcher::UpdateVaryingBuffer(size_t size, void *ptr) {

    memcpy(_varyingVbo, ptr, size);
}

void
OsdCpuKernelDispatcher::MapVertexBuffer() {

    // XXX not efficient for CPU
    // copy vbo content to kernel-buffer
    glBindBuffer(GL_ARRAY_BUFFER, _vertexBuffer);
    glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &_vboSize);

    if (_vbo) 
        delete[] _vbo;
	
    _vbo = new float[_vboSize/sizeof(float)];

    // too slow...
    float *buffer = (float*)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);
    if (buffer) {
        memcpy(_vbo, buffer, _vboSize);
    }
    glUnmapBuffer(GL_ARRAY_BUFFER);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void
OsdCpuKernelDispatcher::MapVaryingBuffer() {

    if (_varyingBuffer) {
        glBindBuffer(GL_ARRAY_BUFFER, _varyingBuffer);
        glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &_varyingVboSize);
        
        if (_varyingVbo) 
	    delete[] _varyingVbo;
        _varyingVbo = new float[_varyingVboSize/sizeof(float)];
        
        float *buffer = (float*)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);
        if (buffer) 
            memcpy(_varyingVbo, buffer, _varyingVboSize);

        glUnmapBuffer(GL_ARRAY_BUFFER);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }
}

void
OsdCpuKernelDispatcher::UnmapVertexBuffer() {

    // write back kernel-buffer to vbo
    glBindBuffer(GL_ARRAY_BUFFER, _vertexBuffer);
    float *buffer = (float*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
    if (buffer)
        memcpy(buffer, _vbo, _vboSize);

    glUnmapBuffer(GL_ARRAY_BUFFER);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

}

void
OsdCpuKernelDispatcher::UnmapVaryingBuffer() { 

    if (_varyingBuffer) {
        // write back kernel-buffer to vbo
        glBindBuffer(GL_ARRAY_BUFFER, _varyingBuffer);
        float *buffer = (float*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
        if (buffer)
            memcpy(buffer, _varyingVbo, _varyingVboSize);

        glUnmapBuffer(GL_ARRAY_BUFFER);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }
}

void
OsdCpuKernelDispatcher::Synchronize() { }


void
OsdCpuKernelDispatcher::ApplyBilinearFaceVerticesKernel( FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const {
    
    VertexDescriptor vd(_numVertexElements, _numVaryingElements);

    computeFace(&vd, _vbo, _varyingVbo,
                (int*)_tables[F_IT].devicePtr + _tableOffsets[F_IT][level-1],
                (int*)_tables[F_ITa].devicePtr + _tableOffsets[F_ITa][level-1],
                offset, start, end);
}

void
OsdCpuKernelDispatcher::ApplyBilinearEdgeVerticesKernel( FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const {
    
    VertexDescriptor vd(_numVertexElements, _numVaryingElements);

    computeBilinearEdge(&vd, _vbo, _varyingVbo,
                        (int*)_tables[E_IT].devicePtr + _tableOffsets[E_IT][level-1],
                        offset, 
                        start, end);
}

void
OsdCpuKernelDispatcher::ApplyBilinearVertexVerticesKernel( FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const {
    
    VertexDescriptor vd(_numVertexElements, _numVaryingElements);

    computeBilinearVertex(&vd, _vbo, _varyingVbo,
                          (int*)_tables[V_ITa].devicePtr + _tableOffsets[V_ITa][level-1],
                          offset, start, end);
}

void
OsdCpuKernelDispatcher::ApplyCatmarkFaceVerticesKernel( FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const {
    
    VertexDescriptor vd(_numVertexElements, _numVaryingElements);

    computeFace(&vd, _vbo, _varyingVbo,
                (int*)_tables[F_IT].devicePtr + _tableOffsets[F_IT][level-1],
                (int*)_tables[F_ITa].devicePtr + _tableOffsets[F_ITa][level-1],
                offset, start, end);
}

void
OsdCpuKernelDispatcher::ApplyCatmarkEdgeVerticesKernel( FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const {
    
    VertexDescriptor vd(_numVertexElements, _numVaryingElements);

    computeEdge(&vd, _vbo, _varyingVbo,
                (int*)_tables[E_IT].devicePtr + _tableOffsets[E_IT][level-1],
                (float*)_tables[E_W].devicePtr + _tableOffsets[E_W][level-1],
                offset, 
                start, end);
}

void
OsdCpuKernelDispatcher::ApplyCatmarkVertexVerticesKernelB( FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const {
    
    VertexDescriptor vd(_numVertexElements, _numVaryingElements);

    computeVertexB(&vd, _vbo, _varyingVbo,
                   (int*)_tables[V_ITa].devicePtr + _tableOffsets[V_ITa][level-1],
                   (int*)_tables[V_IT].devicePtr + _tableOffsets[V_IT][level-1],
                   (float*)_tables[V_W].devicePtr + _tableOffsets[V_W][level-1],
                   offset, start, end);
}

void
OsdCpuKernelDispatcher::ApplyCatmarkVertexVerticesKernelA( FarMesh<OsdVertex> * mesh, int offset, bool pass, int level, int start, int end, void * data) const {
    
    VertexDescriptor vd(_numVertexElements, _numVaryingElements);

    computeVertexA(&vd, _vbo, _varyingVbo,
                   (int*)_tables[V_ITa].devicePtr + _tableOffsets[V_ITa][level-1],
                   (float*)_tables[V_W].devicePtr + _tableOffsets[V_W][level-1],
                   offset, start, end, pass);
}

void
OsdCpuKernelDispatcher::ApplyLoopEdgeVerticesKernel( FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const {
    
    VertexDescriptor vd(_numVertexElements, _numVaryingElements);

    computeEdge(&vd, _vbo, _varyingVbo,
                (int*)_tables[E_IT].devicePtr + _tableOffsets[E_IT][level-1],
                (float*)_tables[E_W].devicePtr + _tableOffsets[E_W][level-1],
                offset, 
                start, end);
}

void
OsdCpuKernelDispatcher::ApplyLoopVertexVerticesKernelB( FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const {
    
    VertexDescriptor vd(_numVertexElements, _numVaryingElements);

    computeLoopVertexB(&vd, _vbo, _varyingVbo,
                       (int*)_tables[V_ITa].devicePtr + _tableOffsets[V_ITa][level-1],
                       (int*)_tables[V_IT].devicePtr + _tableOffsets[V_IT][level-1],
                       (float*)_tables[V_W].devicePtr + _tableOffsets[V_W][level-1],
                       offset, start, end);
}

void
OsdCpuKernelDispatcher::ApplyLoopVertexVerticesKernelA( FarMesh<OsdVertex> * mesh, int offset, bool pass, int level, int start, int end, void * data) const {
    
    VertexDescriptor vd(_numVertexElements, _numVaryingElements);

    computeVertexA(&vd, _vbo, _varyingVbo,
                   (int*)_tables[V_ITa].devicePtr + _tableOffsets[V_ITa][level-1],
                   (float*)_tables[V_W].devicePtr + _tableOffsets[V_W][level-1],
                   offset, start, end, pass);
}

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

