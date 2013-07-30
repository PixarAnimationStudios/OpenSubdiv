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

OsdCLComputeContext::OsdCLComputeContext(FarMesh<OsdVertex> const *farMesh,
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
OsdCLComputeContext::Create(FarMesh<OsdVertex> const *farmesh, cl_context clContext) {

    return new OsdCLComputeContext(farmesh, clContext);
}

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv

