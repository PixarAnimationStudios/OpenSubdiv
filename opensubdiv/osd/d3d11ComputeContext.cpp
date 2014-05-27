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

#include "../far/subdivisionTables.h"
#include "../far/vertexEditTables.h"
#include "../osd/debug.h"
#include "../osd/error.h"
#include "../osd/d3d11ComputeContext.h"
#include "../osd/d3d11KernelBundle.h"

#include <D3D11.h>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

#define SAFE_RELEASE(p) { if(p) { (p)->Release(); (p)=NULL; } }

void
OsdD3D11ComputeTable::createBuffer(int size, const void *ptr, DXGI_FORMAT format, int numElements,
                                   ID3D11DeviceContext *deviceContext) {

    if (size == 0) {
        _buffer = NULL;
        _srv = NULL;
        return;
    }

    ID3D11Device *device = NULL;
    deviceContext->GetDevice(&device);
    assert(device);

    D3D11_BUFFER_DESC bd;
    bd.ByteWidth = size;
    bd.Usage = D3D11_USAGE_IMMUTABLE;
    bd.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    bd.CPUAccessFlags = 0;
    bd.MiscFlags = 0;
    bd.StructureByteStride = 0;
    D3D11_SUBRESOURCE_DATA initData;
    initData.pSysMem = ptr;
    HRESULT hr = device->CreateBuffer(&bd, &initData, &_buffer);
    if (FAILED(hr)) {
        OsdError(OSD_D3D11_COMPUTE_BUFFER_CREATE_ERROR,
                 "Error creating compute table buffer\n");
        return;
    }

    D3D11_SHADER_RESOURCE_VIEW_DESC srvd;
    ZeroMemory(&srvd, sizeof(srvd));
    srvd.Format = format;
    srvd.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
    srvd.Buffer.FirstElement = 0;
    srvd.Buffer.NumElements = numElements;
    hr = device->CreateShaderResourceView(_buffer, &srvd, &_srv);
    if (FAILED(hr)) {
        OsdError(OSD_D3D11_COMPUTE_BUFFER_CREATE_ERROR,
                 "Error creating compute table shader resource view\n");
        return;
    }
}

OsdD3D11ComputeTable::~OsdD3D11ComputeTable() {

    SAFE_RELEASE(_buffer);
    SAFE_RELEASE(_srv);
}

ID3D11Buffer *
OsdD3D11ComputeTable::GetBuffer() const {

    return _buffer;
}

ID3D11ShaderResourceView *
OsdD3D11ComputeTable::GetSRV() const {

    return _srv;
}

// ----------------------------------------------------------------------------

OsdD3D11ComputeHEditTable::OsdD3D11ComputeHEditTable(
    const FarVertexEditTables::VertexEditBatch &batch, ID3D11DeviceContext *deviceContext)
    : _primvarIndicesTable(new OsdD3D11ComputeTable(batch.GetVertexIndices(), deviceContext, DXGI_FORMAT_R32_UINT)),
      _editValuesTable(new OsdD3D11ComputeTable(batch.GetValues(), deviceContext, DXGI_FORMAT_R32_FLOAT)) {

    _operation = batch.GetOperation();
    _primvarOffset = batch.GetPrimvarIndex();
    _primvarWidth = batch.GetPrimvarWidth();
}

OsdD3D11ComputeHEditTable::~OsdD3D11ComputeHEditTable() {

    delete _primvarIndicesTable;
    delete _editValuesTable;
}

const OsdD3D11ComputeTable *
OsdD3D11ComputeHEditTable::GetPrimvarIndices() const {

    return _primvarIndicesTable;
}

const OsdD3D11ComputeTable *
OsdD3D11ComputeHEditTable::GetEditValues() const {

    return _editValuesTable;
}

int
OsdD3D11ComputeHEditTable::GetOperation() const {

    return _operation;
}

int
OsdD3D11ComputeHEditTable::GetPrimvarOffset() const {

    return _primvarOffset;
}

int
OsdD3D11ComputeHEditTable::GetPrimvarWidth() const {

    return _primvarWidth;
}

// ----------------------------------------------------------------------------

OsdD3D11ComputeContext::OsdD3D11ComputeContext(
    FarSubdivisionTables const *subdivisionTables,
    FarVertexEditTables const *vertexEditTables,
    ID3D11DeviceContext *deviceContext) {

    // allocate 5 or 7 tables
    // XXXtakahito: Although _tables size depends on table type, F_IT is set
    // to NULL even in loop case, to determine the condition in
    // bindShaderStorageBuffer()...
    _tables.resize(7, 0);

    _tables[FarSubdivisionTables::E_IT]  = new OsdD3D11ComputeTable(subdivisionTables->Get_E_IT(), deviceContext, DXGI_FORMAT_R32_SINT);
    _tables[FarSubdivisionTables::V_IT]  = new OsdD3D11ComputeTable(subdivisionTables->Get_V_IT(), deviceContext, DXGI_FORMAT_R32_UINT);
    _tables[FarSubdivisionTables::V_ITa] = new OsdD3D11ComputeTable(subdivisionTables->Get_V_ITa(), deviceContext, DXGI_FORMAT_R32_SINT);
    _tables[FarSubdivisionTables::E_W]   = new OsdD3D11ComputeTable(subdivisionTables->Get_E_W(), deviceContext, DXGI_FORMAT_R32_FLOAT);
    _tables[FarSubdivisionTables::V_W]   = new OsdD3D11ComputeTable(subdivisionTables->Get_V_W(), deviceContext, DXGI_FORMAT_R32_FLOAT);

    if (subdivisionTables->GetNumTables() > 5) {
        _tables[FarSubdivisionTables::F_IT]  = new OsdD3D11ComputeTable(subdivisionTables->Get_F_IT(), deviceContext, DXGI_FORMAT_R32_UINT);
        _tables[FarSubdivisionTables::F_ITa] = new OsdD3D11ComputeTable(subdivisionTables->Get_F_ITa(), deviceContext, DXGI_FORMAT_R32_SINT);
    } else {
        _tables[FarSubdivisionTables::F_IT] = NULL;
        _tables[FarSubdivisionTables::F_ITa] = NULL;
    }

    // create hedit tables
    if (vertexEditTables) {
        int numEditBatches = vertexEditTables->GetNumBatches();
        _editTables.reserve(numEditBatches);
        for (int i = 0; i < numEditBatches; ++i) {
            const FarVertexEditTables::VertexEditBatch & edit =
                vertexEditTables->GetBatch(i);
            _editTables.push_back(new OsdD3D11ComputeHEditTable(edit, deviceContext));
        }
    }
}

OsdD3D11ComputeContext::~OsdD3D11ComputeContext() {

    for (size_t i = 0; i < _tables.size(); ++i) {
        delete _tables[i];
    }
    for (size_t i = 0; i < _editTables.size(); ++i) {
        delete _editTables[i];
    }
}

const OsdD3D11ComputeTable *
OsdD3D11ComputeContext::GetTable(int tableIndex) const {

    return _tables[tableIndex];
}

int
OsdD3D11ComputeContext::GetNumEditTables() const {

    return static_cast<int>(_editTables.size());
}

const OsdD3D11ComputeHEditTable *
OsdD3D11ComputeContext::GetEditTable(int tableIndex) const {

    return _editTables[tableIndex];
}

OsdD3D11ComputeContext *
OsdD3D11ComputeContext::Create(FarSubdivisionTables const *subdivisionTables,
                               FarVertexEditTables const *vertexEditTables,
                               ID3D11DeviceContext *deviceContext) {

    return new OsdD3D11ComputeContext(subdivisionTables, vertexEditTables, deviceContext);
}

void
OsdD3D11ComputeContext::BindEditShaderStorageBuffers(int editIndex,
                                                     ID3D11DeviceContext *deviceContext) const {

    const OsdD3D11ComputeHEditTable * edit = _editTables[editIndex];
    const OsdD3D11ComputeTable * primvarIndices = edit->GetPrimvarIndices();
    const OsdD3D11ComputeTable * editValues = edit->GetEditValues();

    ID3D11ShaderResourceView *SRViews[] = {
        primvarIndices->GetSRV(),
        editValues->GetSRV(),
    };
    deviceContext->CSSetShaderResources(9, 2, SRViews); // t9-t10
}

void
OsdD3D11ComputeContext::UnbindEditShaderStorageBuffers(ID3D11DeviceContext *deviceContext) const {

    ID3D11ShaderResourceView *SRViews[] = { 0, 0 };
    deviceContext->CSSetShaderResources(9, 2, SRViews); // t9-t10
}

void
OsdD3D11ComputeContext::BindShaderStorageBuffers(ID3D11DeviceContext *deviceContext) const {

    // XXX: should be better handling for loop subdivision.
    if (_tables[FarSubdivisionTables::F_IT]) {
        ID3D11ShaderResourceView *SRViews[] = {
            _tables[FarSubdivisionTables::F_IT]->GetSRV(),
            _tables[FarSubdivisionTables::F_ITa]->GetSRV(),
        };
        deviceContext->CSSetShaderResources(2, 2, SRViews); // t2-t3
    }

    ID3D11ShaderResourceView *SRViews[] = {
        _tables[FarSubdivisionTables::E_IT]->GetSRV(),
        _tables[FarSubdivisionTables::V_IT]->GetSRV(),
        _tables[FarSubdivisionTables::V_ITa]->GetSRV(),
        _tables[FarSubdivisionTables::E_W]->GetSRV(),
        _tables[FarSubdivisionTables::V_W]->GetSRV(),
    };
    deviceContext->CSSetShaderResources(4, 5, SRViews); // t4-t8
}

void
OsdD3D11ComputeContext::UnbindShaderStorageBuffers(ID3D11DeviceContext *deviceContext) const {

    ID3D11ShaderResourceView *SRViews[] = { 0, 0, 0, 0, 0, 0, 0 };
    deviceContext->CSSetShaderResources(2, 7, SRViews); // t2-t8
}

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
