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
#include "../far/dispatcher.h"
#include "../far/catmarkSubdivisionTables.h"
#include "../far/bilinearSubdivisionTables.h"

#include "../osd/cpuComputeContext.h"
#include "../osd/cpuKernel.h"
#include "../osd/vertexDescriptor.h"
#include "../osd/error.h"

#include <cstring>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

void
OsdCpuTable::createCpuBuffer(size_t size, const void *ptr) {

    _devicePtr = new unsigned char[size];
    memcpy(_devicePtr, ptr, size);
}

OsdCpuTable::~OsdCpuTable() {

    if (_devicePtr) 
        delete [] (unsigned char *)_devicePtr;
}

void *
OsdCpuTable::GetBuffer() const {

    return _devicePtr;
}

// ----------------------------------------------------------------------------

OsdCpuHEditTable::OsdCpuHEditTable(
    const FarVertexEditTables<OsdVertex>::VertexEditBatch &batch)
    : _primvarIndicesTable(new OsdCpuTable(batch.GetVertexIndices())),
      _editValuesTable(new OsdCpuTable(batch.GetValues())) {

    _operation = batch.GetOperation();
    _primvarOffset = batch.GetPrimvarIndex();
    _primvarWidth = batch.GetPrimvarWidth();
}

OsdCpuHEditTable::~OsdCpuHEditTable() {

    delete _primvarIndicesTable;
    delete _editValuesTable;
}

const OsdCpuTable *
OsdCpuHEditTable::GetPrimvarIndices() const {

    return _primvarIndicesTable;
}

const OsdCpuTable *
OsdCpuHEditTable::GetEditValues() const {

    return _editValuesTable;
}

int
OsdCpuHEditTable::GetOperation() const {

    return _operation;
}

int
OsdCpuHEditTable::GetPrimvarOffset() const {

    return _primvarOffset;
}

int
OsdCpuHEditTable::GetPrimvarWidth() const {

    return _primvarWidth;
}

OsdCpuComputeContext::OsdCpuComputeContext(FarMesh<OsdVertex> const *farMesh) {

    FarSubdivisionTables<OsdVertex> const * farTables =
        farMesh->GetSubdivisionTables();

    // allocate 5 or 7 tables
    _tables.resize(farTables->GetNumTables(), 0);

    _tables[FarSubdivisionTables<OsdVertex>::E_IT]  = new OsdCpuTable(farTables->Get_E_IT());
    _tables[FarSubdivisionTables<OsdVertex>::V_IT]  = new OsdCpuTable(farTables->Get_V_IT());
    _tables[FarSubdivisionTables<OsdVertex>::V_ITa] = new OsdCpuTable(farTables->Get_V_ITa());
    _tables[FarSubdivisionTables<OsdVertex>::E_W]   = new OsdCpuTable(farTables->Get_E_W());
    _tables[FarSubdivisionTables<OsdVertex>::V_W]   = new OsdCpuTable(farTables->Get_V_W());

    if (farTables->GetNumTables() > 5) {
        _tables[FarSubdivisionTables<OsdVertex>::F_IT]  = new OsdCpuTable(farTables->Get_F_IT());
        _tables[FarSubdivisionTables<OsdVertex>::F_ITa] = new OsdCpuTable(farTables->Get_F_ITa());
    }

    // create hedit tables
    FarVertexEditTables<OsdVertex> const *editTables = farMesh->GetVertexEdit();
    if (editTables) {
        int numEditBatches = editTables->GetNumBatches();
        _editTables.reserve(numEditBatches);
        for (int i = 0; i < numEditBatches; ++i) {
            const FarVertexEditTables<OsdVertex>::VertexEditBatch & edit =
                editTables->GetBatch(i);

            _editTables.push_back(new OsdCpuHEditTable(edit));
        }
    }
    _currentVertexBuffer = 0;
    _currentVaryingBuffer = 0;
}

OsdCpuComputeContext::~OsdCpuComputeContext() {

    for (size_t i = 0; i < _tables.size(); ++i) {
        delete _tables[i];
    }
    for (size_t i = 0; i < _editTables.size(); ++i) {
        delete _editTables[i];
    }
}

const OsdCpuTable *
OsdCpuComputeContext::GetTable(int tableIndex) const {

    return _tables[tableIndex];
}

int
OsdCpuComputeContext::GetNumEditTables() const {

    return static_cast<int>(_editTables.size());
}

const OsdCpuHEditTable *
OsdCpuComputeContext::GetEditTable(int tableIndex) const {

    return _editTables[tableIndex];
}

float *
OsdCpuComputeContext::GetCurrentVertexBuffer() const {

    return _currentVertexBuffer;
}

float *
OsdCpuComputeContext::GetCurrentVaryingBuffer() const {

    return _currentVaryingBuffer;
}

OsdCpuComputeContext *
OsdCpuComputeContext::Create(FarMesh<OsdVertex> const *farmesh) {

    return new OsdCpuComputeContext(farmesh);
}

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
