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

#include "../far/mesh.h"
#include "../osd/clComputeContext.h"
#include "../osd/clKernelBundle.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

void
OsdCLTable::createCLBuffer(size_t size, const void *ptr, cl_context clContext)
{
    cl_int ciErrNum;
    _devicePtr = clCreateBuffer(
        clContext, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,
        size, const_cast<void*>(ptr), &ciErrNum);
}

OsdCLTable::~OsdCLTable() {

    if (_devicePtr) clReleaseMemObject(_devicePtr);
}

cl_mem
OsdCLTable::GetDevicePtr() const {

    return _devicePtr;
}

// -----------------------------------------------------------------------------

OsdCLHEditTable::OsdCLHEditTable(
    const FarVertexEditTables<OsdVertex>::VertexEditBatch &batch,
    cl_context clContext)
    : _primvarIndicesTable(new OsdCLTable(batch.GetVertexIndices(), clContext)),
      _editValuesTable(new OsdCLTable(batch.GetValues(), clContext)) {

    _operation = batch.GetOperation();
    _primvarOffset = batch.GetPrimvarIndex();
    _primvarWidth = batch.GetPrimvarWidth();
}

OsdCLHEditTable::~OsdCLHEditTable() {

    delete _primvarIndicesTable;
    delete _editValuesTable;
}

const OsdCLTable *
OsdCLHEditTable::GetPrimvarIndices() const {

    return _primvarIndicesTable;
}

const OsdCLTable *
OsdCLHEditTable::GetEditValues() const {

    return _editValuesTable;
}

int
OsdCLHEditTable::GetOperation() const {

    return _operation;
}

int
OsdCLHEditTable::GetPrimvarOffset() const {

    return _primvarOffset;
}

int
OsdCLHEditTable::GetPrimvarWidth() const {

    return _primvarWidth;
}

// ----------------------------------------------------------------------------

OsdCLComputeContext::OsdCLComputeContext(FarMesh<OsdVertex> *farMesh,
                                          cl_context clContext)
    : _clQueue(NULL), _kernelBundle(NULL) {

    FarSubdivisionTables<OsdVertex> const * farTables =
        farMesh->GetSubdivisionTables();

    // allocate 5 or 7 tables
    _tables.resize(farTables->GetNumTables(), 0);

    _tables[FarSubdivisionTables<OsdVertex>::E_IT]  = new OsdCLTable(farTables->Get_E_IT(), clContext);
    _tables[FarSubdivisionTables<OsdVertex>::V_IT]  = new OsdCLTable(farTables->Get_V_IT(), clContext);
    _tables[FarSubdivisionTables<OsdVertex>::V_ITa] = new OsdCLTable(farTables->Get_V_ITa(), clContext);
    _tables[FarSubdivisionTables<OsdVertex>::E_W]   = new OsdCLTable(farTables->Get_E_W(), clContext);
    _tables[FarSubdivisionTables<OsdVertex>::V_W]   = new OsdCLTable(farTables->Get_V_W(), clContext);

    if (farTables->GetNumTables() > 5) {
        _tables[FarSubdivisionTables<OsdVertex>::F_IT]  = new OsdCLTable(farTables->Get_F_IT(), clContext);
        _tables[FarSubdivisionTables<OsdVertex>::F_ITa] = new OsdCLTable(farTables->Get_F_ITa(), clContext);
    }

    // create hedit tables
    FarVertexEditTables<OsdVertex> const *editTables = farMesh->GetVertexEdit();
    if (editTables) {
        int numEditBatches = editTables->GetNumBatches();
        _editTables.reserve(numEditBatches);
        for (int i = 0; i < numEditBatches; ++i) {
            const FarVertexEditTables<OsdVertex>::VertexEditBatch & edit =
                editTables->GetBatch(i);
            _editTables.push_back(new OsdCLHEditTable(edit, clContext));
        }
    }
}

OsdCLComputeContext::~OsdCLComputeContext() {

    for (size_t i = 0; i < _tables.size(); ++i) {
        delete _tables[i];
    }
    for (size_t i = 0; i < _editTables.size(); ++i) {
        delete _editTables[i];
    }
}

const OsdCLTable *
OsdCLComputeContext::GetTable(int tableIndex) const {

    return _tables[tableIndex];
}

int
OsdCLComputeContext::GetNumEditTables() const {

    return static_cast<int>(_editTables.size());
}

const OsdCLHEditTable *
OsdCLComputeContext::GetEditTable(int tableIndex) const {

    return _editTables[tableIndex];
}

cl_mem
OsdCLComputeContext::GetCurrentVertexBuffer() const {

    return _currentVertexBuffer;
}

cl_mem
OsdCLComputeContext::GetCurrentVaryingBuffer() const {

    return _currentVaryingBuffer;
}

OsdCLKernelBundle *
OsdCLComputeContext::GetKernelBundle() const {

    return _kernelBundle;
}

void
OsdCLComputeContext::SetKernelBundle(OsdCLKernelBundle *kernelBundle) {

    _kernelBundle = kernelBundle;
}

void
OsdCLComputeContext::SetCommandQueue(cl_command_queue queue) {

    _clQueue = queue;
}

cl_command_queue
OsdCLComputeContext::GetCommandQueue() const {

    return _clQueue;
}

OsdCLComputeContext *
OsdCLComputeContext::Create(FarMesh<OsdVertex> *farmesh, cl_context clContext) {

    return new OsdCLComputeContext(farmesh, clContext);
}

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv

