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

#include "../osd/glslTransformFeedbackComputeController.h"
#include "../osd/glslTransformFeedbackComputeContext.h"
#include "../osd/glslTransformFeedbackKernelBundle.h"

#include "../osd/opengl.h"

#include <algorithm>
#include <cassert>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

OsdGLSLTransformFeedbackComputeController::OsdGLSLTransformFeedbackComputeController() {
}

OsdGLSLTransformFeedbackComputeController::~OsdGLSLTransformFeedbackComputeController() {

    for (std::vector<OsdGLSLTransformFeedbackKernelBundle*>::iterator it =
             _kernelRegistry.begin();
         it != _kernelRegistry.end(); ++it) {
        delete *it;
    }
}

void
OsdGLSLTransformFeedbackComputeController::Synchronize() {

    glFinish();
}

OsdGLSLTransformFeedbackKernelBundle *
OsdGLSLTransformFeedbackComputeController::getKernels(int numVertexElements,
                                     int numVaryingElements) {

    std::vector<OsdGLSLTransformFeedbackKernelBundle*>::iterator it =
        std::find_if(_kernelRegistry.begin(), _kernelRegistry.end(),
                     OsdGLSLTransformFeedbackKernelBundle::Match(numVertexElements,
                                                numVaryingElements));
    if (it != _kernelRegistry.end()) {
        return *it;
    } else {
        OsdGLSLTransformFeedbackKernelBundle *kernelBundle = new OsdGLSLTransformFeedbackKernelBundle();
        _kernelRegistry.push_back(kernelBundle);
        kernelBundle->Compile(numVertexElements, numVaryingElements);
        return kernelBundle;
    }
}

void
OsdGLSLTransformFeedbackComputeController::ApplyBilinearFaceVerticesKernel(
    FarKernelBatch const &batch, void * clientdata) const {

    OsdGLSLTransformFeedbackComputeContext * context =
        static_cast<OsdGLSLTransformFeedbackComputeContext*>(clientdata);
    assert(context);

    OsdGLSLTransformFeedbackKernelBundle * kernelBundle = context->GetKernelBundle();

    kernelBundle->ApplyBilinearFaceVerticesKernel(
        context->GetCurrentVertexBuffer(),
        context->GetVertexDescriptor().numVertexElements,
        context->GetCurrentVaryingBuffer(),
        context->GetVertexDescriptor().numVaryingElements,
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdGLSLTransformFeedbackComputeController::ApplyBilinearEdgeVerticesKernel(
    FarKernelBatch const &batch, void * clientdata) const {

    OsdGLSLTransformFeedbackComputeContext * context =
        static_cast<OsdGLSLTransformFeedbackComputeContext*>(clientdata);
    assert(context);

    OsdGLSLTransformFeedbackKernelBundle * kernelBundle = context->GetKernelBundle();

    kernelBundle->ApplyBilinearEdgeVerticesKernel(
        context->GetCurrentVertexBuffer(),
        context->GetVertexDescriptor().numVertexElements,
        context->GetCurrentVaryingBuffer(),
        context->GetVertexDescriptor().numVaryingElements,
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdGLSLTransformFeedbackComputeController::ApplyBilinearVertexVerticesKernel(
    FarKernelBatch const &batch, void * clientdata) const {

    OsdGLSLTransformFeedbackComputeContext * context =
        static_cast<OsdGLSLTransformFeedbackComputeContext*>(clientdata);
    assert(context);

    OsdGLSLTransformFeedbackKernelBundle * kernelBundle = context->GetKernelBundle();

    kernelBundle->ApplyBilinearVertexVerticesKernel(
        context->GetCurrentVertexBuffer(),
        context->GetVertexDescriptor().numVertexElements,
        context->GetCurrentVaryingBuffer(),
        context->GetVertexDescriptor().numVaryingElements,
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdGLSLTransformFeedbackComputeController::ApplyCatmarkFaceVerticesKernel(
    FarKernelBatch const &batch, void * clientdata) const {

    OsdGLSLTransformFeedbackComputeContext * context =
        static_cast<OsdGLSLTransformFeedbackComputeContext*>(clientdata);
    assert(context);

    OsdGLSLTransformFeedbackKernelBundle * kernelBundle = context->GetKernelBundle();

    kernelBundle->ApplyCatmarkFaceVerticesKernel(
        context->GetCurrentVertexBuffer(),
        context->GetVertexDescriptor().numVertexElements,
        context->GetCurrentVaryingBuffer(),
        context->GetVertexDescriptor().numVaryingElements,
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}



void
OsdGLSLTransformFeedbackComputeController::ApplyCatmarkEdgeVerticesKernel(
    FarKernelBatch const &batch, void * clientdata) const {

    OsdGLSLTransformFeedbackComputeContext * context =
        static_cast<OsdGLSLTransformFeedbackComputeContext*>(clientdata);
    assert(context);

    OsdGLSLTransformFeedbackKernelBundle * kernelBundle = context->GetKernelBundle();

    kernelBundle->ApplyCatmarkEdgeVerticesKernel(
        context->GetCurrentVertexBuffer(),
        context->GetVertexDescriptor().numVertexElements,
        context->GetCurrentVaryingBuffer(),
        context->GetVertexDescriptor().numVaryingElements,
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdGLSLTransformFeedbackComputeController::ApplyCatmarkVertexVerticesKernelB(
    FarKernelBatch const &batch, void * clientdata) const {

    OsdGLSLTransformFeedbackComputeContext * context =
        static_cast<OsdGLSLTransformFeedbackComputeContext*>(clientdata);
    assert(context);

    OsdGLSLTransformFeedbackKernelBundle * kernelBundle = context->GetKernelBundle();

    kernelBundle->ApplyCatmarkVertexVerticesKernelB(
        context->GetCurrentVertexBuffer(),
        context->GetVertexDescriptor().numVertexElements,
        context->GetCurrentVaryingBuffer(),
        context->GetVertexDescriptor().numVaryingElements,
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdGLSLTransformFeedbackComputeController::ApplyCatmarkVertexVerticesKernelA1(
    FarKernelBatch const &batch, void * clientdata) const {

    OsdGLSLTransformFeedbackComputeContext * context =
        static_cast<OsdGLSLTransformFeedbackComputeContext*>(clientdata);
    assert(context);

    OsdGLSLTransformFeedbackKernelBundle * kernelBundle = context->GetKernelBundle();

    kernelBundle->ApplyCatmarkVertexVerticesKernelA(
        context->GetCurrentVertexBuffer(),
        context->GetVertexDescriptor().numVertexElements,
        context->GetCurrentVaryingBuffer(),
        context->GetVertexDescriptor().numVaryingElements,
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd(), false);
}

void
OsdGLSLTransformFeedbackComputeController::ApplyCatmarkVertexVerticesKernelA2(
    FarKernelBatch const &batch, void * clientdata) const {

    OsdGLSLTransformFeedbackComputeContext * context =
        static_cast<OsdGLSLTransformFeedbackComputeContext*>(clientdata);
    assert(context);

    OsdGLSLTransformFeedbackKernelBundle * kernelBundle = context->GetKernelBundle();

    kernelBundle->ApplyCatmarkVertexVerticesKernelA(
        context->GetCurrentVertexBuffer(),
        context->GetVertexDescriptor().numVertexElements,
        context->GetCurrentVaryingBuffer(),
        context->GetVertexDescriptor().numVaryingElements,
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd(), true);
}

void
OsdGLSLTransformFeedbackComputeController::ApplyLoopEdgeVerticesKernel(
    FarKernelBatch const &batch, void * clientdata) const {

    OsdGLSLTransformFeedbackComputeContext * context =
        static_cast<OsdGLSLTransformFeedbackComputeContext*>(clientdata);
    assert(context);

    OsdGLSLTransformFeedbackKernelBundle * kernelBundle = context->GetKernelBundle();

    kernelBundle->ApplyLoopEdgeVerticesKernel(
        context->GetCurrentVertexBuffer(),
        context->GetVertexDescriptor().numVertexElements,
        context->GetCurrentVaryingBuffer(),
        context->GetVertexDescriptor().numVaryingElements,
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdGLSLTransformFeedbackComputeController::ApplyLoopVertexVerticesKernelB(
    FarKernelBatch const &batch, void * clientdata) const {

    OsdGLSLTransformFeedbackComputeContext * context =
        static_cast<OsdGLSLTransformFeedbackComputeContext*>(clientdata);
    assert(context);

    OsdGLSLTransformFeedbackKernelBundle * kernelBundle = context->GetKernelBundle();

    kernelBundle->ApplyLoopVertexVerticesKernelB(
        context->GetCurrentVertexBuffer(),
        context->GetVertexDescriptor().numVertexElements,
        context->GetCurrentVaryingBuffer(),
        context->GetVertexDescriptor().numVaryingElements,
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
}

void
OsdGLSLTransformFeedbackComputeController::ApplyLoopVertexVerticesKernelA1(
    FarKernelBatch const &batch, void * clientdata) const {

    OsdGLSLTransformFeedbackComputeContext * context =
        static_cast<OsdGLSLTransformFeedbackComputeContext*>(clientdata);
    assert(context);

    OsdGLSLTransformFeedbackKernelBundle * kernelBundle = context->GetKernelBundle();

    kernelBundle->ApplyLoopVertexVerticesKernelA(
        context->GetCurrentVertexBuffer(),
        context->GetVertexDescriptor().numVertexElements,
        context->GetCurrentVaryingBuffer(),
        context->GetVertexDescriptor().numVaryingElements,
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd(), false);
}

void
OsdGLSLTransformFeedbackComputeController::ApplyLoopVertexVerticesKernelA2(
    FarKernelBatch const &batch, void * clientdata) const {

    OsdGLSLTransformFeedbackComputeContext * context =
        static_cast<OsdGLSLTransformFeedbackComputeContext*>(clientdata);
    assert(context);

    OsdGLSLTransformFeedbackKernelBundle * kernelBundle = context->GetKernelBundle();

    kernelBundle->ApplyLoopVertexVerticesKernelA(
        context->GetCurrentVertexBuffer(),
        context->GetVertexDescriptor().numVertexElements,
        context->GetCurrentVaryingBuffer(),
        context->GetVertexDescriptor().numVaryingElements,
        batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd(), true);
}

void
OsdGLSLTransformFeedbackComputeController::ApplyVertexEdits(
    FarKernelBatch const &batch, void * clientdata) const {

    OsdGLSLTransformFeedbackComputeContext * context =
        static_cast<OsdGLSLTransformFeedbackComputeContext*>(clientdata);
    assert(context);

    OsdGLSLTransformFeedbackKernelBundle * kernelBundle = context->GetKernelBundle();

    const OsdGLSLTransformFeedbackHEditTable * edit = context->GetEditTable(batch.GetTableIndex());
    assert(edit);

    context->BindEditTextures(batch.GetTableIndex());

    int primvarOffset = edit->GetPrimvarOffset();
    int primvarWidth = edit->GetPrimvarWidth();

    if (edit->GetOperation() == FarVertexEdit::Add) {
        kernelBundle->ApplyEditAdd(
            context->GetCurrentVertexBuffer(),
            context->GetVertexDescriptor().numVertexElements,
            context->GetCurrentVaryingBuffer(),
            context->GetVertexDescriptor().numVaryingElements,
            primvarOffset, primvarWidth,
            batch.GetVertexOffset(), batch.GetTableOffset(), batch.GetStart(), batch.GetEnd());
    } else {
        // XXX: edit SET is not implemented yet.
    }
    
    context->UnbindEditTextures();
}

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
