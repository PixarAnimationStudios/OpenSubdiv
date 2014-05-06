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

#include "../far/dispatcher.h"
#include "../far/subdivisionTables.h"

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
    const FarVertexEditTables::VertexEditBatch &batch)
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

OsdCpuComputeContext::OsdCpuComputeContext(FarSubdivisionTables const *subdivisionTables,
                                           FarVertexEditTables const *vertexEditTables) {

    // allocate 5 or 7 tables
    _tables.resize(subdivisionTables->GetNumTables(), 0);

    _tables[FarSubdivisionTables::E_IT]  = new OsdCpuTable(subdivisionTables->Get_E_IT());
    _tables[FarSubdivisionTables::V_IT]  = new OsdCpuTable(subdivisionTables->Get_V_IT());
    _tables[FarSubdivisionTables::V_ITa] = new OsdCpuTable(subdivisionTables->Get_V_ITa());
    _tables[FarSubdivisionTables::E_W]   = new OsdCpuTable(subdivisionTables->Get_E_W());
    _tables[FarSubdivisionTables::V_W]   = new OsdCpuTable(subdivisionTables->Get_V_W());

    if (subdivisionTables->GetNumTables() > 5) {
        _tables[FarSubdivisionTables::F_IT]  = new OsdCpuTable(subdivisionTables->Get_F_IT());
        _tables[FarSubdivisionTables::F_ITa] = new OsdCpuTable(subdivisionTables->Get_F_ITa());
    }

    // create hedit tables
    if (vertexEditTables) {
        int numEditBatches = vertexEditTables->GetNumBatches();
        _editTables.reserve(numEditBatches);
        for (int i = 0; i < numEditBatches; ++i) {
            const FarVertexEditTables::VertexEditBatch & edit =
                vertexEditTables->GetBatch(i);

            _editTables.push_back(new OsdCpuHEditTable(edit));
        }
    }
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

OsdCpuComputeContext *
OsdCpuComputeContext::Create(FarSubdivisionTables const *subdivisionTables,
                             FarVertexEditTables const *vertexEditTables) {

    return new OsdCpuComputeContext(subdivisionTables, vertexEditTables);
}

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
