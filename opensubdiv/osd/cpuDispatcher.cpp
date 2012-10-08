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
#include "../osd/cpuDispatcher.h"
#include "../osd/cpuKernel.h"

#include <stdlib.h>
#include <string.h>

#ifdef OPENSUBDIV_HAS_OPENMP
    #include <omp.h>
#endif

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

OsdCpuKernelDispatcher::Table::~Table() {

    if (ptr)
        free(ptr);
}

void
OsdCpuKernelDispatcher::Table::Copy( int size, const void *table ) {

    if (size > 0) {
        if (ptr)
            free(ptr);
        ptr = malloc(size);
        memcpy(ptr, table, size);
    }
}

OsdCpuKernelDispatcher::OsdCpuKernelDispatcher( int levels, int numOmpThreads )
    : OsdKernelDispatcher(levels), _currentVertexBuffer(NULL), _currentVaryingBuffer(NULL), _vdesc(NULL), _numOmpThreads(numOmpThreads) {
    _tables.resize(TABLE_MAX);
}

OsdCpuKernelDispatcher::~OsdCpuKernelDispatcher() {

    if (_vdesc)
        delete _vdesc;
}

static OsdCpuKernelDispatcher::OsdKernelDispatcher *
Create(int levels) {
    return new OsdCpuKernelDispatcher(levels);
}

#ifdef OPENSUBDIV_HAS_OPENMP
static OsdCpuKernelDispatcher::OsdKernelDispatcher *
CreateOmp(int levels) {
    return new OsdCpuKernelDispatcher(levels, omp_get_num_procs());
}
#endif

void
OsdCpuKernelDispatcher::Register() {

    Factory::GetInstance().Register(Create, kCPU);
#ifdef OPENSUBDIV_HAS_OPENMP
    Factory::GetInstance().Register(CreateOmp, kOPENMP);
#endif

}

void
OsdCpuKernelDispatcher::OnKernelLaunch() {
#ifdef OPENSUBDIV_HAS_OPENMP
    omp_set_num_threads(_numOmpThreads);
#endif
}

void
OsdCpuKernelDispatcher::CopyTable(int tableIndex, size_t size, const void *ptr) {

    _tables[tableIndex].Copy((int)size, ptr);
}

void
OsdCpuKernelDispatcher::AllocateEditTables(int n) {

    _editTables.resize(n*2);
    _edits.resize(n);
}

void
OsdCpuKernelDispatcher::UpdateEditTable(int tableIndex, const FarTable<unsigned int> &offsets, const FarTable<float> &values,
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
OsdCpuKernelDispatcher::InitializeVertexBuffer(int numElements, int numVertices)
{
    return new OsdCpuVertexBuffer(numElements, numVertices);
}

void
OsdCpuKernelDispatcher::BindVertexBuffer(OsdVertexBuffer *vertex, OsdVertexBuffer *varying) {

    if (vertex)
        _currentVertexBuffer = dynamic_cast<OsdCpuVertexBuffer *>(vertex);
    else
        _currentVertexBuffer = NULL;

    if (varying)
        _currentVaryingBuffer = dynamic_cast<OsdCpuVertexBuffer *>(varying);
    else
        _currentVaryingBuffer = NULL;

    _vdesc = new VertexDescriptor(_currentVertexBuffer ? _currentVertexBuffer->GetNumElements() : 0,
                                  _currentVaryingBuffer ? _currentVaryingBuffer->GetNumElements() : 0);
}

void
OsdCpuKernelDispatcher::UnbindVertexBuffer()
{
    delete _vdesc;
    _vdesc = NULL;

    _currentVertexBuffer = NULL;
    _currentVaryingBuffer = NULL;
}

void
OsdCpuKernelDispatcher::Synchronize() { }


void
OsdCpuKernelDispatcher::ApplyBilinearFaceVerticesKernel( FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const {

    computeFace(_vdesc, GetVertexBuffer(), GetVaryingBuffer(),
                (int*)_tables[F_IT].ptr + _tableOffsets[F_IT][level-1],
                (int*)_tables[F_ITa].ptr + _tableOffsets[F_ITa][level-1],
                offset, start, end);
}

void
OsdCpuKernelDispatcher::ApplyBilinearEdgeVerticesKernel( FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const {

    computeBilinearEdge(_vdesc, GetVertexBuffer(), GetVaryingBuffer(),
                        (int*)_tables[E_IT].ptr + _tableOffsets[E_IT][level-1],
                        offset,
                        start, end);
}

void
OsdCpuKernelDispatcher::ApplyBilinearVertexVerticesKernel( FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const {

    computeBilinearVertex(_vdesc, GetVertexBuffer(), GetVaryingBuffer(),
                          (int*)_tables[V_ITa].ptr + _tableOffsets[V_ITa][level-1],
                          offset, start, end);
}

void
OsdCpuKernelDispatcher::ApplyCatmarkFaceVerticesKernel( FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const {

    computeFace(_vdesc, GetVertexBuffer(), GetVaryingBuffer(),
                (int*)_tables[F_IT].ptr + _tableOffsets[F_IT][level-1],
                (int*)_tables[F_ITa].ptr + _tableOffsets[F_ITa][level-1],
                offset, start, end);
}

void
OsdCpuKernelDispatcher::ApplyCatmarkEdgeVerticesKernel( FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const {

    computeEdge(_vdesc, GetVertexBuffer(), GetVaryingBuffer(),
                (int*)_tables[E_IT].ptr + _tableOffsets[E_IT][level-1],
                (float*)_tables[E_W].ptr + _tableOffsets[E_W][level-1],
                offset,
                start, end);
}

void
OsdCpuKernelDispatcher::ApplyCatmarkVertexVerticesKernelB( FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const {

    computeVertexB(_vdesc, GetVertexBuffer(), GetVaryingBuffer(),
                   (int*)_tables[V_ITa].ptr + _tableOffsets[V_ITa][level-1],
                   (int*)_tables[V_IT].ptr + _tableOffsets[V_IT][level-1],
                   (float*)_tables[V_W].ptr + _tableOffsets[V_W][level-1],
                   offset, start, end);
}

void
OsdCpuKernelDispatcher::ApplyCatmarkVertexVerticesKernelA( FarMesh<OsdVertex> * mesh, int offset, bool pass, int level, int start, int end, void * data) const {

    computeVertexA(_vdesc, GetVertexBuffer(), GetVaryingBuffer(),
                   (int*)_tables[V_ITa].ptr + _tableOffsets[V_ITa][level-1],
                   (float*)_tables[V_W].ptr + _tableOffsets[V_W][level-1],
                   offset, start, end, pass);
}

void
OsdCpuKernelDispatcher::ApplyLoopEdgeVerticesKernel( FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const {

    computeEdge(_vdesc, GetVertexBuffer(), GetVaryingBuffer(),
                (int*)_tables[E_IT].ptr + _tableOffsets[E_IT][level-1],
                (float*)_tables[E_W].ptr + _tableOffsets[E_W][level-1],
                offset,
                start, end);
}

void
OsdCpuKernelDispatcher::ApplyLoopVertexVerticesKernelB( FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const {

    computeLoopVertexB(_vdesc, GetVertexBuffer(), GetVaryingBuffer(),
                       (int*)_tables[V_ITa].ptr + _tableOffsets[V_ITa][level-1],
                       (int*)_tables[V_IT].ptr + _tableOffsets[V_IT][level-1],
                       (float*)_tables[V_W].ptr + _tableOffsets[V_W][level-1],
                       offset, start, end);
}

void
OsdCpuKernelDispatcher::ApplyLoopVertexVerticesKernelA( FarMesh<OsdVertex> * mesh, int offset, bool pass, int level, int start, int end, void * data) const {

    computeVertexA(_vdesc, GetVertexBuffer(), GetVaryingBuffer(),
                   (int*)_tables[V_ITa].ptr + _tableOffsets[V_ITa][level-1],
                   (float*)_tables[V_W].ptr + _tableOffsets[V_W][level-1],
                   offset, start, end, pass);
}

void
OsdCpuKernelDispatcher::ApplyVertexEdits(FarMesh<OsdVertex> *mesh, int offset, int level, void * clientdata) const {
    for (int i=0; i<(int)_edits.size(); ++i) {
        const VertexEditArrayInfo &info = _edits[i];

        if (info.operation == FarVertexEdit::Add) {
            editVertexAdd(_vdesc, GetVertexBuffer(), info.primVarOffset, info.primVarWidth, info.numEdits[level-1],
                          (int*)_editTables[i*2+0].ptr + info.offsetOffsets[level-1],
                          (float*)_editTables[i*2+1].ptr + info.valueOffsets[level-1]);
        } else if (info.operation == FarVertexEdit::Set) {
            editVertexSet(_vdesc, GetVertexBuffer(), info.primVarOffset, info.primVarWidth, info.numEdits[level],
                          (int*)_editTables[i*2+0].ptr + info.offsetOffsets[level],
                          (float*)_editTables[i*2+1].ptr + info.valueOffsets[level]);
        }
    }
}

} // end namespace OPENSUBDIV_VERSION

} // end namespace OpenSubdiv

