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
OsdD3D11ComputeController::getKernels(int numVertexElements,
                                     int numVaryingElements) {

    std::vector<OsdD3D11ComputeKernelBundle*>::iterator it =
        std::find_if(_kernelRegistry.begin(), _kernelRegistry.end(),
                     OsdD3D11ComputeKernelBundle::Match(numVertexElements,
                                                numVaryingElements));
    if (it != _kernelRegistry.end()) {
        return *it;
    } else {
        OsdD3D11ComputeKernelBundle *kernelBundle =
            new OsdD3D11ComputeKernelBundle(_deviceContext);
        _kernelRegistry.push_back(kernelBundle);
        kernelBundle->Compile(numVertexElements, numVaryingElements);
        return kernelBundle;
    }
}


void
OsdD3D11ComputeController::ApplyBilinearFaceVerticesKernel(
    FarKernelBatch const &batch, void * clientdata) const {

    OsdD3D11ComputeContext * context =
        static_cast<OsdD3D11ComputeContext*>(clientdata);
    assert(context);

    OsdD3D11ComputeKernelBundle * kernelBundle = context->GetKernelBundle();

    kernelBundle->ApplyBilinearFaceVerticesKernel(
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdD3D11ComputeController::ApplyBilinearEdgeVerticesKernel(
    FarKernelBatch const &batch, void * clientdata) const {

    OsdD3D11ComputeContext * context =
        static_cast<OsdD3D11ComputeContext*>(clientdata);
    assert(context);

    OsdD3D11ComputeKernelBundle * kernelBundle = context->GetKernelBundle();

    kernelBundle->ApplyBilinearEdgeVerticesKernel(
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdD3D11ComputeController::ApplyBilinearVertexVerticesKernel(
    FarKernelBatch const &batch, void * clientdata) const {

    OsdD3D11ComputeContext * context =
        static_cast<OsdD3D11ComputeContext*>(clientdata);
    assert(context);

    OsdD3D11ComputeKernelBundle * kernelBundle = context->GetKernelBundle();

    kernelBundle->ApplyBilinearVertexVerticesKernel(
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdD3D11ComputeController::ApplyCatmarkFaceVerticesKernel(
    FarKernelBatch const &batch, void * clientdata) const {

    OsdD3D11ComputeContext * context =
        static_cast<OsdD3D11ComputeContext*>(clientdata);
    assert(context);

    OsdD3D11ComputeKernelBundle * kernelBundle = context->GetKernelBundle();

    kernelBundle->ApplyCatmarkFaceVerticesKernel(
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}



void
OsdD3D11ComputeController::ApplyCatmarkEdgeVerticesKernel(
    FarKernelBatch const &batch, void * clientdata) const {

    OsdD3D11ComputeContext * context =
        static_cast<OsdD3D11ComputeContext*>(clientdata);
    assert(context);

    OsdD3D11ComputeKernelBundle * kernelBundle = context->GetKernelBundle();

    kernelBundle->ApplyCatmarkEdgeVerticesKernel(
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdD3D11ComputeController::ApplyCatmarkVertexVerticesKernelB(
    FarKernelBatch const &batch, void * clientdata) const {

    OsdD3D11ComputeContext * context =
        static_cast<OsdD3D11ComputeContext*>(clientdata);
    assert(context);

    OsdD3D11ComputeKernelBundle * kernelBundle = context->GetKernelBundle();

    kernelBundle->ApplyCatmarkVertexVerticesKernelB(
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdD3D11ComputeController::ApplyCatmarkVertexVerticesKernelA1(
    FarKernelBatch const &batch, void * clientdata) const {

    OsdD3D11ComputeContext * context =
        static_cast<OsdD3D11ComputeContext*>(clientdata);
    assert(context);

    OsdD3D11ComputeKernelBundle * kernelBundle = context->GetKernelBundle();

    kernelBundle->ApplyCatmarkVertexVerticesKernelA(
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd(), false);
}

void
OsdD3D11ComputeController::ApplyCatmarkVertexVerticesKernelA2(
    FarKernelBatch const &batch, void * clientdata) const {

    OsdD3D11ComputeContext * context =
        static_cast<OsdD3D11ComputeContext*>(clientdata);
    assert(context);

    OsdD3D11ComputeKernelBundle * kernelBundle = context->GetKernelBundle();

    kernelBundle->ApplyCatmarkVertexVerticesKernelA(
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd(), true);
}

void
OsdD3D11ComputeController::ApplyLoopEdgeVerticesKernel(
    FarKernelBatch const &batch, void * clientdata) const {

    OsdD3D11ComputeContext * context =
        static_cast<OsdD3D11ComputeContext*>(clientdata);
    assert(context);

    OsdD3D11ComputeKernelBundle * kernelBundle = context->GetKernelBundle();

    kernelBundle->ApplyLoopEdgeVerticesKernel(
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdD3D11ComputeController::ApplyLoopVertexVerticesKernelB(
    FarKernelBatch const &batch, void * clientdata) const {

    OsdD3D11ComputeContext * context =
        static_cast<OsdD3D11ComputeContext*>(clientdata);
    assert(context);

    OsdD3D11ComputeKernelBundle * kernelBundle = context->GetKernelBundle();

    kernelBundle->ApplyLoopVertexVerticesKernelB(
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdD3D11ComputeController::ApplyLoopVertexVerticesKernelA1(
    FarKernelBatch const &batch, void * clientdata) const {

    OsdD3D11ComputeContext * context =
        static_cast<OsdD3D11ComputeContext*>(clientdata);
    assert(context);

    OsdD3D11ComputeKernelBundle * kernelBundle = context->GetKernelBundle();

    kernelBundle->ApplyLoopVertexVerticesKernelA(
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd(), false);
}

void
OsdD3D11ComputeController::ApplyLoopVertexVerticesKernelA2(
    FarKernelBatch const &batch, void * clientdata) const {

    OsdD3D11ComputeContext * context =
        static_cast<OsdD3D11ComputeContext*>(clientdata);
    assert(context);

    OsdD3D11ComputeKernelBundle * kernelBundle = context->GetKernelBundle();

    kernelBundle->ApplyLoopVertexVerticesKernelA(
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd(), true);
}

void
OsdD3D11ComputeController::ApplyVertexEdits(
    FarKernelBatch const &batch, void * clientdata) const {

    OsdD3D11ComputeContext * context =
        static_cast<OsdD3D11ComputeContext*>(clientdata);
    assert(context);

    OsdD3D11ComputeKernelBundle * kernelBundle = context->GetKernelBundle();

    const OsdD3D11ComputeHEditTable * edit = context->GetEditTable(batch.GetTableIndex());
    assert(edit);

    context->BindEditShaderStorageBuffers(batch.GetTableIndex());

    int primvarOffset = edit->GetPrimvarOffset();
    int primvarWidth = edit->GetPrimvarWidth();
    
    if (edit->GetOperation() == FarVertexEdit::Add) {
        kernelBundle->ApplyEditAdd(primvarOffset, primvarWidth,
                                   batch.GetVertexOffset(), batch.GetTableOffset(),
                                   batch.GetStart(), batch.GetEnd());
    } else {
        // XXX: edit SET is not implemented yet.
    }
    
    context->UnbindEditShaderStorageBuffers();
}
}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
