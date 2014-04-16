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

#include "../osd/glslComputeController.h"
#include "../osd/glslComputeContext.h"
#include "../osd/glslKernelBundle.h"

#include "../osd/opengl.h"

#include <algorithm>
#include <cassert>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

OsdGLSLComputeController::OsdGLSLComputeController() {
}

OsdGLSLComputeController::~OsdGLSLComputeController() {

    for (std::vector<OsdGLSLComputeKernelBundle*>::iterator it =
             _kernelRegistry.begin();
         it != _kernelRegistry.end(); ++it) {
        delete *it;
    }
}

void
OsdGLSLComputeController::Synchronize() {

    glFinish();
}

OsdGLSLComputeKernelBundle *
OsdGLSLComputeController::getKernels(int numVertexElements,
                                     int numVaryingElements) {

    std::vector<OsdGLSLComputeKernelBundle*>::iterator it =
        std::find_if(_kernelRegistry.begin(), _kernelRegistry.end(),
                     OsdGLSLComputeKernelBundle::Match(numVertexElements,
                                                numVaryingElements));
    if (it != _kernelRegistry.end()) {
        return *it;
    } else {
        OsdGLSLComputeKernelBundle *kernelBundle =
            new OsdGLSLComputeKernelBundle();
        _kernelRegistry.push_back(kernelBundle);
        kernelBundle->Compile(numVertexElements, numVaryingElements);
        return kernelBundle;
    }
}

void
OsdGLSLComputeController::ApplyBilinearFaceVerticesKernel(
    FarKernelBatch const &batch, OsdGLSLComputeContext *context) const {

    assert(context);

    OsdGLSLComputeKernelBundle * kernelBundle = context->GetKernelBundle();

    kernelBundle->ApplyBilinearFaceVerticesKernel(
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdGLSLComputeController::ApplyBilinearEdgeVerticesKernel(
    FarKernelBatch const &batch, OsdGLSLComputeContext *context) const {

    assert(context);

    OsdGLSLComputeKernelBundle * kernelBundle = context->GetKernelBundle();

    kernelBundle->ApplyBilinearEdgeVerticesKernel(
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdGLSLComputeController::ApplyBilinearVertexVerticesKernel(
    FarKernelBatch const &batch, OsdGLSLComputeContext *context) const {

    assert(context);

    OsdGLSLComputeKernelBundle * kernelBundle = context->GetKernelBundle();

    kernelBundle->ApplyBilinearVertexVerticesKernel(
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdGLSLComputeController::ApplyCatmarkFaceVerticesKernel(
    FarKernelBatch const &batch, OsdGLSLComputeContext *context) const {

    assert(context);

    OsdGLSLComputeKernelBundle * kernelBundle = context->GetKernelBundle();

    kernelBundle->ApplyCatmarkFaceVerticesKernel(
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}



void
OsdGLSLComputeController::ApplyCatmarkEdgeVerticesKernel(
    FarKernelBatch const &batch, OsdGLSLComputeContext *context) const {

    assert(context);

    OsdGLSLComputeKernelBundle * kernelBundle = context->GetKernelBundle();

    kernelBundle->ApplyCatmarkEdgeVerticesKernel(
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdGLSLComputeController::ApplyCatmarkVertexVerticesKernelB(
    FarKernelBatch const &batch, OsdGLSLComputeContext *context) const {

    assert(context);

    OsdGLSLComputeKernelBundle * kernelBundle = context->GetKernelBundle();

    kernelBundle->ApplyCatmarkVertexVerticesKernelB(
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdGLSLComputeController::ApplyCatmarkVertexVerticesKernelA1(
    FarKernelBatch const &batch, OsdGLSLComputeContext *context) const {

    assert(context);

    OsdGLSLComputeKernelBundle * kernelBundle = context->GetKernelBundle();

    kernelBundle->ApplyCatmarkVertexVerticesKernelA(
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd(), false);
}

void
OsdGLSLComputeController::ApplyCatmarkVertexVerticesKernelA2(
    FarKernelBatch const &batch, OsdGLSLComputeContext *context) const {

    assert(context);

    OsdGLSLComputeKernelBundle * kernelBundle = context->GetKernelBundle();

    kernelBundle->ApplyCatmarkVertexVerticesKernelA(
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd(), true);
}

void
OsdGLSLComputeController::ApplyLoopEdgeVerticesKernel(
    FarKernelBatch const &batch, OsdGLSLComputeContext *context) const {

    assert(context);

    OsdGLSLComputeKernelBundle * kernelBundle = context->GetKernelBundle();

    kernelBundle->ApplyLoopEdgeVerticesKernel(
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdGLSLComputeController::ApplyLoopVertexVerticesKernelB(
    FarKernelBatch const &batch, OsdGLSLComputeContext *context) const {

    assert(context);

    OsdGLSLComputeKernelBundle * kernelBundle = context->GetKernelBundle();

    kernelBundle->ApplyLoopVertexVerticesKernelB(
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdGLSLComputeController::ApplyLoopVertexVerticesKernelA1(
    FarKernelBatch const &batch, OsdGLSLComputeContext *context) const {

    assert(context);

    OsdGLSLComputeKernelBundle * kernelBundle = context->GetKernelBundle();

    kernelBundle->ApplyLoopVertexVerticesKernelA(
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd(), false);
}

void
OsdGLSLComputeController::ApplyLoopVertexVerticesKernelA2(
    FarKernelBatch const &batch, OsdGLSLComputeContext *context) const {

    assert(context);

    OsdGLSLComputeKernelBundle * kernelBundle = context->GetKernelBundle();

    kernelBundle->ApplyLoopVertexVerticesKernelA(
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd(), true);
}

void
OsdGLSLComputeController::ApplyVertexEdits(
    FarKernelBatch const &batch, OsdGLSLComputeContext *context) const {

    assert(context);

    OsdGLSLComputeKernelBundle * kernelBundle = context->GetKernelBundle();

    const OsdGLSLComputeHEditTable * edit = context->GetEditTable(batch.GetTableIndex());
    assert(edit);

    context->BindEditShaderStorageBuffers(batch.GetTableIndex());

    int primvarOffset = edit->GetPrimvarOffset();
    int primvarWidth = edit->GetPrimvarWidth();
    
    if (edit->GetOperation() == FarVertexEdit::Add) {
        kernelBundle->ApplyEditAdd( primvarOffset, 
                                    primvarWidth,
                                    batch.GetVertexOffset(), 
                                    batch.GetTableOffset(), 
                                    batch.GetStart(), 
                                    batch.GetEnd());
    } else {
        // XXX: edit SET is not implemented yet.
    }
    
    context->UnbindEditShaderStorageBuffers();
}

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
