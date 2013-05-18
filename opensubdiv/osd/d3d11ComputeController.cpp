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
