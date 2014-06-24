//
//   Copyright 2013 Pixar
//
//   Licensed under the Apache License, Version 2.0 (the "Apache License")
//   with the following modification; you may not use this file except in
//   compliance with the Apache License and the following modification to it:
//   Section 6. Trademarks. is deleted and replaced with:
//
//   6. Trademarks. This License does not grant permission to use the trade
//      names, trademarks, service marks, or product names of the Licensor
//      and its affiliates, except as required to comply with Section 4(c) of
//      the License and to reproduce the content of the NOTICE file.
//
//   You may obtain a copy of the Apache License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the Apache License with the above modification is
//   distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
//   KIND, either express or implied. See the Apache License for the specific
//   language governing permissions and limitations under the Apache License.
//

#include "../osd/clComputeContext.h"
#include "../osd/clKernelBundle.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

void
OsdCLTable::createCLBuffer(size_t size, const void *ptr, cl_context clContext)
{
    if (size == 0) {
        _devicePtr = NULL;
    } else {
        cl_int ciErrNum;
        _devicePtr = clCreateBuffer(
            clContext, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,
            size, const_cast<void*>(ptr), &ciErrNum);
    }
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
    const FarVertexEditTables::VertexEditBatch &batch,
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

OsdCLComputeContext::OsdCLComputeContext(FarSubdivisionTables const *subdivisionTables,
                                         FarVertexEditTables const *vertexEditTables,
                                         cl_context clContext) {

    // allocate 5 or 7 tables
    _tables.resize(subdivisionTables->GetNumTables(), 0);

    _tables[FarSubdivisionTables::E_IT]  = new OsdCLTable(subdivisionTables->Get_E_IT(), clContext);
    _tables[FarSubdivisionTables::V_IT]  = new OsdCLTable(subdivisionTables->Get_V_IT(), clContext);
    _tables[FarSubdivisionTables::V_ITa] = new OsdCLTable(subdivisionTables->Get_V_ITa(), clContext);
    _tables[FarSubdivisionTables::E_W]   = new OsdCLTable(subdivisionTables->Get_E_W(), clContext);
    _tables[FarSubdivisionTables::V_W]   = new OsdCLTable(subdivisionTables->Get_V_W(), clContext);

    if (subdivisionTables->GetNumTables() > 5) {
        _tables[FarSubdivisionTables::F_IT]  = new OsdCLTable(subdivisionTables->Get_F_IT(), clContext);
        _tables[FarSubdivisionTables::F_ITa] = new OsdCLTable(subdivisionTables->Get_F_ITa(), clContext);
    }

    // create hedit tables
    if (vertexEditTables) {
        int numEditBatches = vertexEditTables->GetNumBatches();
        _editTables.reserve(numEditBatches);
        for (int i = 0; i < numEditBatches; ++i) {
            const FarVertexEditTables::VertexEditBatch & edit =
                vertexEditTables->GetBatch(i);
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

OsdCLComputeContext *
OsdCLComputeContext::Create(FarSubdivisionTables const *subdivisionTables,
                            FarVertexEditTables const *vertexEditTables,
                            cl_context clContext) {

    return new OsdCLComputeContext(subdivisionTables, vertexEditTables, clContext);
}

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv

