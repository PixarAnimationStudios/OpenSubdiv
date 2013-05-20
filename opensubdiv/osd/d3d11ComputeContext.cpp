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
#include "../far/subdivisionTables.h"
#include "../osd/debug.h"
#include "../osd/error.h"
#include "../osd/d3d11ComputeContext.h"
#include "../osd/d3d11KernelBundle.h"

#include <D3D11.h>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

#define SAFE_RELEASE(p) { if(p) { (p)->Release(); (p)=NULL; } }

void
OsdD3D11ComputeTable::createBuffer(int size, const void *ptr, DXGI_FORMAT format, int numElements, ID3D11DeviceContext *deviceContext) {

    if (size == 0)
        return;

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
    const FarVertexEditTables<OsdVertex>::VertexEditBatch &batch, ID3D11DeviceContext *deviceContext)
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
    FarMesh<OsdVertex> const *farMesh, ID3D11DeviceContext *deviceContext)
    : _deviceContext(deviceContext),
      _currentVertexBufferUAV(0), _currentVaryingBufferUAV(0) {

    FarSubdivisionTables<OsdVertex> const * farTables =
        farMesh->GetSubdivisionTables();

    // allocate 5 or 7 tables
    // XXXtakahito: Although _tables size depends on table type, F_IT is set
    // to NULL even in loop case, to determine the condition in
    // bindShaderStorageBuffer()...
    _tables.resize(7, 0);

    _tables[FarSubdivisionTables<OsdVertex>::E_IT]  = new OsdD3D11ComputeTable(farTables->Get_E_IT(), deviceContext, DXGI_FORMAT_R32_SINT);
    _tables[FarSubdivisionTables<OsdVertex>::V_IT]  = new OsdD3D11ComputeTable(farTables->Get_V_IT(), deviceContext, DXGI_FORMAT_R32_UINT);
    _tables[FarSubdivisionTables<OsdVertex>::V_ITa] = new OsdD3D11ComputeTable(farTables->Get_V_ITa(), deviceContext, DXGI_FORMAT_R32_SINT);
    _tables[FarSubdivisionTables<OsdVertex>::E_W]   = new OsdD3D11ComputeTable(farTables->Get_E_W(), deviceContext, DXGI_FORMAT_R32_FLOAT);
    _tables[FarSubdivisionTables<OsdVertex>::V_W]   = new OsdD3D11ComputeTable(farTables->Get_V_W(), deviceContext, DXGI_FORMAT_R32_FLOAT);

    if (farTables->GetNumTables() > 5) {
        _tables[FarSubdivisionTables<OsdVertex>::F_IT]  = new OsdD3D11ComputeTable(farTables->Get_F_IT(), deviceContext, DXGI_FORMAT_R32_UINT);
        _tables[FarSubdivisionTables<OsdVertex>::F_ITa] = new OsdD3D11ComputeTable(farTables->Get_F_ITa(), deviceContext, DXGI_FORMAT_R32_SINT);
    } else {
        _tables[FarSubdivisionTables<OsdVertex>::F_IT] = NULL;
        _tables[FarSubdivisionTables<OsdVertex>::F_ITa] = NULL;
    }

    // create hedit tables
    FarVertexEditTables<OsdVertex> const *editTables = farMesh->GetVertexEdit();
    if (editTables) {
        int numEditBatches = editTables->GetNumBatches();
        _editTables.reserve(numEditBatches);
        for (int i = 0; i < numEditBatches; ++i) {
            const FarVertexEditTables<OsdVertex>::VertexEditBatch & edit =
                editTables->GetBatch(i);
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

ID3D11UnorderedAccessView *
OsdD3D11ComputeContext::GetCurrentVertexBufferUAV() const {

    return _currentVertexBufferUAV;
}

ID3D11UnorderedAccessView *
OsdD3D11ComputeContext::GetCurrentVaryingBufferUAV() const {

    return _currentVaryingBufferUAV;
}

OsdD3D11ComputeKernelBundle *
OsdD3D11ComputeContext::GetKernelBundle() const {

    return _kernelBundle;
}

void
OsdD3D11ComputeContext::SetKernelBundle(
    OsdD3D11ComputeKernelBundle *kernelBundle) {

    _kernelBundle = kernelBundle;
}

ID3D11DeviceContext *
OsdD3D11ComputeContext::GetDeviceContext() const {
    return _deviceContext;
}

void
OsdD3D11ComputeContext::SetDeviceContext(ID3D11DeviceContext *deviceContext) {
    _deviceContext = deviceContext;
}

OsdD3D11ComputeContext *
OsdD3D11ComputeContext::Create(FarMesh<OsdVertex> const *farmesh, ID3D11DeviceContext *deviceContext) {

    return new OsdD3D11ComputeContext(farmesh, deviceContext);
}

void
OsdD3D11ComputeContext::BindEditShaderStorageBuffers(int editIndex) {

    const OsdD3D11ComputeHEditTable * edit = _editTables[editIndex];
    const OsdD3D11ComputeTable * primvarIndices = edit->GetPrimvarIndices();
    const OsdD3D11ComputeTable * editValues = edit->GetEditValues();

    ID3D11ShaderResourceView *SRViews[] = {
        primvarIndices->GetSRV(),
        editValues->GetSRV(),
    };
    _deviceContext->CSSetShaderResources(9, 2, SRViews); // t9-t10
}

void
OsdD3D11ComputeContext::UnbindEditShaderStorageBuffers() {

    ID3D11ShaderResourceView *SRViews[] = { 0, 0 };
    _deviceContext->CSSetShaderResources(9, 2, SRViews); // t9-t10
}

void
OsdD3D11ComputeContext::bindShaderStorageBuffers() {

    // Unbind the vertexBuffer from the input assembler
    ID3D11Buffer *NULLBuffer = 0;
    UINT voffset = 0;
    UINT vstride = 0;
    _deviceContext->IASetVertexBuffers(0, 1, &NULLBuffer, &voffset, &vstride);
    // Unbind the vertexBuffer from the vertex shader (gregory patch vertex srv)
    ID3D11ShaderResourceView *NULLSRV = 0;
    _deviceContext->VSSetShaderResources(0, 1, &NULLSRV);

    if (_currentVertexBufferUAV)
        _deviceContext->CSSetUnorderedAccessViews(0, 1, &_currentVertexBufferUAV, 0); // u0

    if (_currentVaryingBufferUAV)
        _deviceContext->CSSetUnorderedAccessViews(1, 1, &_currentVaryingBufferUAV, 0); // u1

    // XXX: should be better handling for loop subdivision.
    if (_tables[FarSubdivisionTables<OsdVertex>::F_IT]) {
        ID3D11ShaderResourceView *SRViews[] = {
            _tables[FarSubdivisionTables<OsdVertex>::F_IT]->GetSRV(),
            _tables[FarSubdivisionTables<OsdVertex>::F_ITa]->GetSRV(),
        };
        _deviceContext->CSSetShaderResources(2, 2, SRViews); // t2-t3
    }

    ID3D11ShaderResourceView *SRViews[] = {
        _tables[FarSubdivisionTables<OsdVertex>::E_IT]->GetSRV(),
        _tables[FarSubdivisionTables<OsdVertex>::V_IT]->GetSRV(),
        _tables[FarSubdivisionTables<OsdVertex>::V_ITa]->GetSRV(),
        _tables[FarSubdivisionTables<OsdVertex>::E_W]->GetSRV(),
        _tables[FarSubdivisionTables<OsdVertex>::V_W]->GetSRV(),
    };
    _deviceContext->CSSetShaderResources(4, 5, SRViews); // t4-t8
}

void
OsdD3D11ComputeContext::unbindShaderStorageBuffers() {

    ID3D11UnorderedAccessView *UAViews[] = { 0, 0 };
    _deviceContext->CSSetUnorderedAccessViews(0, 2, UAViews, 0); // u0-u2
    ID3D11ShaderResourceView *SRViews[] = { 0, 0, 0, 0, 0, 0, 0 };
    _deviceContext->CSSetShaderResources(2, 7, SRViews); // t2-t8
}

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
