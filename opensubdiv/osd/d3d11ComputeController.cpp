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

#include "../osd/d3d11ComputeController.h"
#include "../osd/d3d11ComputeContext.h"
#include "../osd/d3d11KernelBundle.h"

#include <D3D11.h>

#include <algorithm>
#include <cassert>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

#define SAFE_RELEASE(p) { if(p) { (p)->Release(); (p)=NULL; } }

OsdD3D11ComputeController::OsdD3D11ComputeController(
    ID3D11DeviceContext *deviceContext)
    : _deviceContext(deviceContext), _query(0) {
}

OsdD3D11ComputeController::~OsdD3D11ComputeController() {

    for (std::vector<OsdD3D11ComputeKernelBundle*>::iterator it =
             _kernelRegistry.begin();
         it != _kernelRegistry.end(); ++it) {
        delete *it;
    }
    SAFE_RELEASE(_query);
}

void
OsdD3D11ComputeController::Synchronize() {

    if (! _query) {
        ID3D11Device *device = NULL;
        _deviceContext->GetDevice(&device);
        assert(device);

        D3D11_QUERY_DESC desc;
        desc.Query = D3D11_QUERY_EVENT;
        desc.MiscFlags = 0;
        device->CreateQuery(&desc, &_query);
    }
    _deviceContext->Flush();
    _deviceContext->End(_query);
    while (S_OK != _deviceContext->GetData(_query, NULL, 0, 0));
}

OsdD3D11ComputeKernelBundle *
OsdD3D11ComputeController::getKernels(OsdVertexBufferDescriptor const &vertexDesc,
                                      OsdVertexBufferDescriptor const &varyingDesc) {

    std::vector<OsdD3D11ComputeKernelBundle*>::iterator it =
        std::find_if(_kernelRegistry.begin(), _kernelRegistry.end(),
                     OsdD3D11ComputeKernelBundle::Match(
                         vertexDesc, varyingDesc));

    if (it != _kernelRegistry.end()) {
        return *it;
    } else {
        OsdD3D11ComputeKernelBundle *kernelBundle =
            new OsdD3D11ComputeKernelBundle(_deviceContext);
        _kernelRegistry.push_back(kernelBundle);
        kernelBundle->Compile(vertexDesc, varyingDesc);
        return kernelBundle;
    }
}

void
OsdD3D11ComputeController::bindShaderResources()
{
    // Unbind the vertexBuffer from the input assembler
    ID3D11Buffer *NULLBuffer = 0;
    UINT voffset = 0;
    UINT vstride = 0;
    _deviceContext->IASetVertexBuffers(0, 1, &NULLBuffer, &voffset, &vstride);
    // Unbind the vertexBuffer from the vertex shader (gregory patch vertex srv)
    ID3D11ShaderResourceView *NULLSRV = 0;
    _deviceContext->VSSetShaderResources(0, 1, &NULLSRV);

    if (_currentBindState.vertexBuffer)
        _deviceContext->CSSetUnorderedAccessViews(0, 1, &_currentBindState.vertexBuffer, 0); // u0

    if (_currentBindState.varyingBuffer)
        _deviceContext->CSSetUnorderedAccessViews(1, 1, &_currentBindState.varyingBuffer, 0); // u1
}

void
OsdD3D11ComputeController::unbindShaderResources()
{
    ID3D11UnorderedAccessView *UAViews[] = { 0, 0 };
    _deviceContext->CSSetUnorderedAccessViews(0, 2, UAViews, 0); // u0-u2
}

void
OsdD3D11ComputeController::ApplyBilinearFaceVerticesKernel(
    FarKernelBatch const &batch, OsdD3D11ComputeContext const *context) const {

    assert(context);

    _currentBindState.kernelBundle->ApplyBilinearFaceVerticesKernel(
        batch.GetVertexOffset(), batch.GetTableOffset(),
        batch.GetStart(), batch.GetEnd(),
        _currentBindState.vertexDesc.offset, _currentBindState.varyingDesc.offset);
}

void
OsdD3D11ComputeController::ApplyBilinearEdgeVerticesKernel(
    FarKernelBatch const &batch, OsdD3D11ComputeContext const *context) const {

    assert(context);

    _currentBindState.kernelBundle->ApplyBilinearEdgeVerticesKernel(
        batch.GetVertexOffset(), batch.GetTableOffset(),
        batch.GetStart(), batch.GetEnd(),
        _currentBindState.vertexDesc.offset, _currentBindState.varyingDesc.offset);
}

void
OsdD3D11ComputeController::ApplyBilinearVertexVerticesKernel(
    FarKernelBatch const &batch, OsdD3D11ComputeContext const *context) const {

    assert(context);

    _currentBindState.kernelBundle->ApplyBilinearVertexVerticesKernel(
        batch.GetVertexOffset(), batch.GetTableOffset(),
        batch.GetStart(), batch.GetEnd(),
        _currentBindState.vertexDesc.offset, _currentBindState.varyingDesc.offset);
}

void
OsdD3D11ComputeController::ApplyCatmarkFaceVerticesKernel(
    FarKernelBatch const &batch, OsdD3D11ComputeContext const *context) const {

    assert(context);

    _currentBindState.kernelBundle->ApplyCatmarkFaceVerticesKernel(
        batch.GetVertexOffset(), batch.GetTableOffset(),
        batch.GetStart(), batch.GetEnd(),
        _currentBindState.vertexDesc.offset, _currentBindState.varyingDesc.offset);
}

void
OsdD3D11ComputeController::ApplyCatmarkQuadFaceVerticesKernel(
    FarKernelBatch const &batch, OsdD3D11ComputeContext const *context) const {

    assert(context);

    _currentBindState.kernelBundle->ApplyCatmarkQuadFaceVerticesKernel(
        batch.GetVertexOffset(), batch.GetTableOffset(),
        batch.GetStart(), batch.GetEnd(),
        _currentBindState.vertexDesc.offset, _currentBindState.varyingDesc.offset);
}

void
OsdD3D11ComputeController::ApplyCatmarkTriQuadFaceVerticesKernel(
    FarKernelBatch const &batch, OsdD3D11ComputeContext const *context) const {

    assert(context);

    _currentBindState.kernelBundle->ApplyCatmarkTriQuadFaceVerticesKernel(
        batch.GetVertexOffset(), batch.GetTableOffset(),
        batch.GetStart(), batch.GetEnd(),
        _currentBindState.vertexDesc.offset, _currentBindState.varyingDesc.offset);
}

void
OsdD3D11ComputeController::ApplyCatmarkRestrictedEdgeVerticesKernel(
    FarKernelBatch const &batch, OsdD3D11ComputeContext const *context) const {

    assert(context);

    _currentBindState.kernelBundle->ApplyCatmarkRestrictedEdgeVerticesKernel(
        batch.GetVertexOffset(), batch.GetTableOffset(),
        batch.GetStart(), batch.GetEnd(),
        _currentBindState.vertexDesc.offset, _currentBindState.varyingDesc.offset);
}

void
OsdD3D11ComputeController::ApplyCatmarkEdgeVerticesKernel(
    FarKernelBatch const &batch, OsdD3D11ComputeContext const *context) const {

    assert(context);

    _currentBindState.kernelBundle->ApplyCatmarkEdgeVerticesKernel(
        batch.GetVertexOffset(), batch.GetTableOffset(),
        batch.GetStart(), batch.GetEnd(),
        _currentBindState.vertexDesc.offset, _currentBindState.varyingDesc.offset);
}

void
OsdD3D11ComputeController::ApplyCatmarkVertexVerticesKernelB(
    FarKernelBatch const &batch, OsdD3D11ComputeContext const *context) const {

    assert(context);

    _currentBindState.kernelBundle->ApplyCatmarkVertexVerticesKernelB(
        batch.GetVertexOffset(), batch.GetTableOffset(),
        batch.GetStart(), batch.GetEnd(),
        _currentBindState.vertexDesc.offset, _currentBindState.varyingDesc.offset);
}

void
OsdD3D11ComputeController::ApplyCatmarkVertexVerticesKernelA1(
    FarKernelBatch const &batch, OsdD3D11ComputeContext const *context) const {

    assert(context);

    _currentBindState.kernelBundle->ApplyCatmarkVertexVerticesKernelA(
        batch.GetVertexOffset(), batch.GetTableOffset(),
        batch.GetStart(), batch.GetEnd(), false,
        _currentBindState.vertexDesc.offset, _currentBindState.varyingDesc.offset);
}

void
OsdD3D11ComputeController::ApplyCatmarkVertexVerticesKernelA2(
    FarKernelBatch const &batch, OsdD3D11ComputeContext const *context) const {

    assert(context);

    _currentBindState.kernelBundle->ApplyCatmarkVertexVerticesKernelA(
        batch.GetVertexOffset(), batch.GetTableOffset(),
        batch.GetStart(), batch.GetEnd(), true,
        _currentBindState.vertexDesc.offset, _currentBindState.varyingDesc.offset);
}

void
OsdD3D11ComputeController::ApplyCatmarkRestrictedVertexVerticesKernelB1(
    FarKernelBatch const &batch, OsdD3D11ComputeContext const *context) const {

    assert(context);

    _currentBindState.kernelBundle->ApplyCatmarkRestrictedVertexVerticesKernelB1(
        batch.GetVertexOffset(), batch.GetTableOffset(),
        batch.GetStart(), batch.GetEnd(),
        _currentBindState.vertexDesc.offset, _currentBindState.varyingDesc.offset);
}

void
OsdD3D11ComputeController::ApplyCatmarkRestrictedVertexVerticesKernelB2(
    FarKernelBatch const &batch, OsdD3D11ComputeContext const *context) const {

    assert(context);

    _currentBindState.kernelBundle->ApplyCatmarkRestrictedVertexVerticesKernelB2(
        batch.GetVertexOffset(), batch.GetTableOffset(),
        batch.GetStart(), batch.GetEnd(),
        _currentBindState.vertexDesc.offset, _currentBindState.varyingDesc.offset);
}

void
OsdD3D11ComputeController::ApplyCatmarkRestrictedVertexVerticesKernelA(
    FarKernelBatch const &batch, OsdD3D11ComputeContext const *context) const {

    assert(context);

    _currentBindState.kernelBundle->ApplyCatmarkRestrictedVertexVerticesKernelA(
        batch.GetVertexOffset(), batch.GetTableOffset(),
        batch.GetStart(), batch.GetEnd(),
        _currentBindState.vertexDesc.offset, _currentBindState.varyingDesc.offset);
}

void
OsdD3D11ComputeController::ApplyLoopEdgeVerticesKernel(
    FarKernelBatch const &batch, OsdD3D11ComputeContext const *context) const {

    assert(context);

    _currentBindState.kernelBundle->ApplyLoopEdgeVerticesKernel(
        batch.GetVertexOffset(), batch.GetTableOffset(),
        batch.GetStart(), batch.GetEnd(),
        _currentBindState.vertexDesc.offset, _currentBindState.varyingDesc.offset);
}

void
OsdD3D11ComputeController::ApplyLoopVertexVerticesKernelB(
    FarKernelBatch const &batch, OsdD3D11ComputeContext const *context) const {

    assert(context);

    _currentBindState.kernelBundle->ApplyLoopVertexVerticesKernelB(
        batch.GetVertexOffset(), batch.GetTableOffset(),
        batch.GetStart(), batch.GetEnd(),
        _currentBindState.vertexDesc.offset, _currentBindState.varyingDesc.offset);
}

void
OsdD3D11ComputeController::ApplyLoopVertexVerticesKernelA1(
    FarKernelBatch const &batch, OsdD3D11ComputeContext const *context) const {

    assert(context);

    _currentBindState.kernelBundle->ApplyLoopVertexVerticesKernelA(
        batch.GetVertexOffset(), batch.GetTableOffset(),
        batch.GetStart(), batch.GetEnd(), false,
        _currentBindState.vertexDesc.offset, _currentBindState.varyingDesc.offset);
}

void
OsdD3D11ComputeController::ApplyLoopVertexVerticesKernelA2(
    FarKernelBatch const &batch, OsdD3D11ComputeContext const *context) const {

    assert(context);

    _currentBindState.kernelBundle->ApplyLoopVertexVerticesKernelA(
        batch.GetVertexOffset(), batch.GetTableOffset(),
        batch.GetStart(), batch.GetEnd(), true,
        _currentBindState.vertexDesc.offset, _currentBindState.varyingDesc.offset);
}

void
OsdD3D11ComputeController::ApplyVertexEdits(
    FarKernelBatch const &batch, OsdD3D11ComputeContext const *context) const {

    assert(context);

    const OsdD3D11ComputeHEditTable * edit = context->GetEditTable(batch.GetTableIndex());
    assert(edit);

    context->BindEditShaderStorageBuffers(batch.GetTableIndex(), _deviceContext);

    int primvarOffset = edit->GetPrimvarOffset();
    int primvarWidth = edit->GetPrimvarWidth();

    if (edit->GetOperation() == FarVertexEdit::Add) {
        _currentBindState.kernelBundle->ApplyEditAdd(primvarOffset, primvarWidth,
                                           batch.GetVertexOffset(),
                                           batch.GetTableOffset(),
                                           batch.GetStart(),
                                           batch.GetEnd(),
                                           _currentBindState.vertexDesc.offset,
                                           _currentBindState.varyingDesc.offset);
    } else {
        // XXX: edit SET is not implemented yet.
    }

    context->UnbindEditShaderStorageBuffers(_deviceContext);
}
}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
