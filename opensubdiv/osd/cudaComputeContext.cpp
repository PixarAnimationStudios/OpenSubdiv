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
#include "../osd/cudaComputeContext.h"

#include <cuda_runtime.h>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

bool
OsdCudaTable::createCudaBuffer(size_t size, const void *ptr) {

    cudaError_t err = cudaMalloc(&_devicePtr, size);
    if (err != cudaSuccess) {
        return false;
    }

    err = cudaMemcpy(_devicePtr, ptr, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(_devicePtr);
        _devicePtr = NULL;
        return false;
    }
    return true;
}

OsdCudaTable::~OsdCudaTable() {

    if (_devicePtr) cudaFree(_devicePtr);
}

void *
OsdCudaTable::GetCudaMemory() const {

    return _devicePtr;
}

// ----------------------------------------------------------------------------

OsdCudaHEditTable::OsdCudaHEditTable() 
    : _primvarIndicesTable(NULL), _editValuesTable(NULL) {
}

OsdCudaHEditTable::~OsdCudaHEditTable() {

    delete _primvarIndicesTable;
    delete _editValuesTable;
}

OsdCudaHEditTable *
OsdCudaHEditTable::Create(const FarVertexEditTables<OsdVertex>::
                          VertexEditBatch &batch) {

    OsdCudaHEditTable *result = new OsdCudaHEditTable();

    result->_operation = batch.GetOperation();
    result->_primvarOffset = batch.GetPrimvarIndex();
    result->_primvarWidth = batch.GetPrimvarWidth();
    result->_primvarIndicesTable = OsdCudaTable::Create(batch.GetVertexIndices());
    result->_editValuesTable = OsdCudaTable::Create(batch.GetValues());

    if (result->_primvarIndicesTable == NULL or
        result->_editValuesTable == NULL) {
        delete result;
        return NULL;
    }
    return result;
}

const OsdCudaTable *
OsdCudaHEditTable::GetPrimvarIndices() const {

    return _primvarIndicesTable;
}

const OsdCudaTable *
OsdCudaHEditTable::GetEditValues() const {

    return _editValuesTable;
}

int
OsdCudaHEditTable::GetOperation() const {

    return _operation;
}

int
OsdCudaHEditTable::GetPrimvarOffset() const {

    return _primvarOffset;
}

int
OsdCudaHEditTable::GetPrimvarWidth() const {

    return _primvarWidth;
}

// ----------------------------------------------------------------------------

OsdCudaComputeContext::OsdCudaComputeContext() :
    _currentVertexBuffer(NULL), _currentVaryingBuffer(NULL) {
}

OsdCudaComputeContext::~OsdCudaComputeContext() {

    for (size_t i = 0; i < _tables.size(); ++i) {
        delete _tables[i];
    }
    for (size_t i = 0; i < _editTables.size(); ++i) {
        delete _editTables[i];
    }
}

bool
OsdCudaComputeContext::initialize(FarMesh<OsdVertex> const *farMesh) {

    FarSubdivisionTables<OsdVertex> const * farTables =
        farMesh->GetSubdivisionTables();

    // allocate 5 or 7 tables
    _tables.resize(farTables->GetNumTables(), 0);

    _tables[FarSubdivisionTables<OsdVertex>::E_IT]  = OsdCudaTable::Create(farTables->Get_E_IT());
    _tables[FarSubdivisionTables<OsdVertex>::V_IT]  = OsdCudaTable::Create(farTables->Get_V_IT());
    _tables[FarSubdivisionTables<OsdVertex>::V_ITa] = OsdCudaTable::Create(farTables->Get_V_ITa());
    _tables[FarSubdivisionTables<OsdVertex>::E_W]   = OsdCudaTable::Create(farTables->Get_E_W());
    _tables[FarSubdivisionTables<OsdVertex>::V_W]   = OsdCudaTable::Create(farTables->Get_V_W());

    if (farTables->GetNumTables() > 5) {
        _tables[FarSubdivisionTables<OsdVertex>::F_IT]  = OsdCudaTable::Create(farTables->Get_F_IT());
        _tables[FarSubdivisionTables<OsdVertex>::F_ITa] = OsdCudaTable::Create(farTables->Get_F_ITa());
    }

    // error check
    for (size_t i = 0; i < _tables.size(); ++i) {
        if (_tables[i] == NULL) {
            return false;
        }
    }

    // create hedit tables
    FarVertexEditTables<OsdVertex> const *editTables = farMesh->GetVertexEdit();
    if (editTables) {
        int numEditBatches = editTables->GetNumBatches();
        _editTables.reserve(numEditBatches);
        for (int i = 0; i < numEditBatches; ++i) {
            const FarVertexEditTables<OsdVertex>::VertexEditBatch & edit =
                editTables->GetBatch(i);

            _editTables.push_back(OsdCudaHEditTable::Create(edit));
        }
    }

    // error check
    for (size_t i = 0; i < _editTables.size(); ++i) {
        if (_editTables[i] == NULL) return false;
    }

    return true;
}

const OsdCudaTable *
OsdCudaComputeContext::GetTable(int tableIndex) const {

    return _tables[tableIndex];
}

int
OsdCudaComputeContext::GetNumEditTables() const {

    return static_cast<int>(_editTables.size());
}

const OsdCudaHEditTable *
OsdCudaComputeContext::GetEditTable(int tableIndex) const {

    return _editTables[tableIndex];
}

float *
OsdCudaComputeContext::GetCurrentVertexBuffer() const {

    return _currentVertexBuffer;
}

float *
OsdCudaComputeContext::GetCurrentVaryingBuffer() const {

    return _currentVaryingBuffer;
}

OsdCudaComputeContext *
OsdCudaComputeContext::Create(FarMesh<OsdVertex> const *farmesh) {

    OsdCudaComputeContext *result = new OsdCudaComputeContext();

    if (result->initialize(farmesh) == false) {
        delete result;
        return NULL;
    }
    return result;
}

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
