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
