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
#ifndef OSDUTIL_MESH_BATCH_CL_H
#define OSDUTIL_MESH_BATCH_CL_H

#include "../version.h"
#include "../osdutil/batch.h"
#include "../osd/clComputeController.h"

#include <vector>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

// ----------------------------------------------------------------------------
// OsdUtilMeshBatch OpenCL specialized class
//
template <typename VERTEX_BUFFER, typename DRAW_CONTEXT>
class OsdUtilMeshBatch<VERTEX_BUFFER, DRAW_CONTEXT, OsdCLComputeController> : public OsdUtilMeshBatchBase<DRAW_CONTEXT> {
public:
    typedef VERTEX_BUFFER VertexBuffer;
    typedef OsdCLComputeController::ComputeContext ComputeContext;
    typedef DRAW_CONTEXT DrawContext;
    typedef OsdCLComputeController ComputeController;
    typedef OsdUtilMeshBatchBase<DRAW_CONTEXT> Base;
    
    // XXX: not happy with retaining compute controller..
    static OsdUtilMeshBatch* Create(ComputeController *computeController,
                                    const std::vector<FarMesh<OsdVertex> const * > &meshVector,
                                    int numVertexElements,
                                    int numVaryingElements,
                                    int batchIndex,
                                    bool requireFVarData=false);

    virtual ~OsdUtilMeshBatch();

    virtual typename DrawContext::VertexBufferBinding BindVertexBuffer() {
        if (not _vertexBuffer) return 0;
        return _vertexBuffer->BindVBO();
    }

    virtual typename DrawContext::VertexBufferBinding BindVaryingBuffer() {
        if (not _varyingBuffer) return 0;
        return _varyingBuffer->BindVBO();
    }

    virtual DrawContext * GetDrawContext() const { return _drawContext; }

    // update API
    virtual void UpdateCoarseVertices(int meshIndex, const float *data, int numVertices) {

        if (not _vertexBuffer)
            return;

        Base::setMeshDirty(meshIndex);

        _vertexBuffer->UpdateData(data, Base::GetVertexOffset(meshIndex), numVertices,
                                  _computeController->GetCommandQueue());
    }

    virtual void UpdateCoarseVaryings(int meshIndex, const float *data, int numVertices) {

        if (not _varyingBuffer)
            return;

        Base::setMeshDirty(meshIndex);

        _varyingBuffer->UpdateData(data, Base::GetVertexOffset(meshIndex), numVertices,
                                   _computeController->GetCommandQueue());
    }

    // utility function
    virtual void FinalizeUpdate() {
        // create kernel batch for dirty handles
        
        FarKernelBatchVector batches;
        Base::populateDirtyKernelBatches(batches);
        Base::resetMeshDirty();

        _computeController->Refine(_computeContext, batches, _vertexBuffer);
    }

    VertexBuffer *GetVertexBuffer() const { return _vertexBuffer; }
    VertexBuffer *GetVaryingBuffer() const { return _varyingBuffer; }

private:
    OsdUtilMeshBatch();
    
    bool initialize(ComputeController *computeController,
                    OsdUtilMeshBatchEntryVector const &entries,
                    FarMesh<OsdVertex> const *farMultiMesh,
                    int numVertexElements,
                    int numVaryingElements,
                    int batchIndex,
                    bool requireFVarData);

    bool initialize(FarPatchTables const *patchTables,
                    int numVertices,
                    int numPtexFaces,
                    int numVertexElements,
                    int numVaryingElements,
                    int batchIndex,
                    bool requireFVarData);

    ComputeController *_computeController;
    ComputeContext *_computeContext;

    VertexBuffer *_vertexBuffer, *_varyingBuffer;
    DrawContext *_drawContext;
};

// -----------------------------------------------------------------------------

template <typename VERTEX_BUFFER, typename DRAW_CONTEXT>
OsdUtilMeshBatch<VERTEX_BUFFER, DRAW_CONTEXT, OsdCLComputeController>::OsdUtilMeshBatch() :
    _computeController(NULL), _computeContext(NULL), _vertexBuffer(NULL), _varyingBuffer(NULL), _drawContext(NULL) {
}

template <typename VERTEX_BUFFER, typename DRAW_CONTEXT> bool
OsdUtilMeshBatch<VERTEX_BUFFER, DRAW_CONTEXT, OsdCLComputeController>::initialize(ComputeController *computeController,
                                                                                  OsdUtilMeshBatchEntryVector const &entries,
                                                                                  FarMesh<OsdVertex> const *farMultiMesh,
                                                                                  int numVertexElements,
                                                                                  int numVaryingElements,
                                                                                  int batchIndex,
                                                                                  bool requireFVarData) {
    Base::initialize(entries, farMultiMesh->GetNumVertices(), farMultiMesh->GetNumPtexFaces(),
                     numVertexElements, batchIndex);
    
    _computeController = computeController;

    // copy batches
    Base::setKernelBatches(farMultiMesh->GetKernelBatches());

    // create compute contexts
    _computeContext = ComputeContext::Create(farMultiMesh, _computeController->GetContext());

    if (not _computeContext) return false;

    FarPatchTables const * patchTables = farMultiMesh->GetPatchTables();

    _vertexBuffer = numVertexElements ? VertexBuffer::Create(numVertexElements, Base::GetNumVertices(), _computeController->GetContext()) : NULL;
    _varyingBuffer = numVaryingElements ? VertexBuffer::Create(numVaryingElements, Base::GetNumVertices(), _computeController->GetContext()) : NULL;

    _drawContext = DrawContext::Create(patchTables, requireFVarData);

    if (not _drawContext) return false;

    _drawContext->UpdateVertexTexture(_vertexBuffer);

    return true;
}

template <typename VERTEX_BUFFER, typename DRAW_CONTEXT> bool
OsdUtilMeshBatch<VERTEX_BUFFER, DRAW_CONTEXT, OsdCLComputeController>::initialize(FarPatchTables const *patchTables,
                                                                                  int numVertices,
                                                                                  int numPtexFaces,
                                                                                  int numVertexElements,
                                                                                  int numVaryingElements,
                                                                                  int batchIndex,
                                                                                  bool requireFVarData) {
    Base::initialize(numVertices, numPtexFaces, numVertexElements, batchIndex);

    _vertexBuffer = numVertexElements ? VertexBuffer::Create(numVertexElements, Base::GetNumVertices(), _computeController->GetContext()) : NULL;
    _varyingBuffer = numVaryingElements ? VertexBuffer::Create(numVaryingElements, Base::GetNumVertices(), _computeController->GetContext()) : NULL;

    _drawContext = DrawContext::Create(patchTables, requireFVarData);

    if (not _drawContext) return false;

    _drawContext->UpdateVertexTexture(_vertexBuffer);

    return true;
}

template <typename VERTEX_BUFFER, typename DRAW_CONTEXT>
OsdUtilMeshBatch<VERTEX_BUFFER, DRAW_CONTEXT, OsdCLComputeController>::~OsdUtilMeshBatch()
{
    delete _computeContext;
    delete _drawContext;

    delete _vertexBuffer;
    delete _varyingBuffer;
}

template <typename VERTEX_BUFFER, typename DRAW_CONTEXT>
OsdUtilMeshBatch<VERTEX_BUFFER, DRAW_CONTEXT, OsdCLComputeController> *
OsdUtilMeshBatch<VERTEX_BUFFER, DRAW_CONTEXT, OsdCLComputeController>::Create(ComputeController *computeController,
                                                                              const std::vector<FarMesh<OsdVertex> const * > &meshVector,
                                                                              int numVertexElements,
                                                                              int numVaryingElements,
                                                                              int batchIndex,
                                                                              bool requireFVarData) {
    std::vector<FarPatchTables::PatchArrayVector> multiFarPatchArray;
    FarMesh <OsdVertex> *farMultiMesh = createMultiMesh(meshVector, multiFarPatchArray);

    FarPatchTables const * patchTables = farMultiMesh->GetPatchTables();

    // create entries
    OsdUtilMeshBatchEntryVector entries;
    createEntries(entries, multiFarPatchArray, patchTables->GetMaxValence(), meshVector);

    OsdUtilMeshBatch<VERTEX_BUFFER, DRAW_CONTEXT, OsdCLComputeController> *batch =
        new OsdUtilMeshBatch<VERTEX_BUFFER, DRAW_CONTEXT, OsdCLComputeController>();

    batch->initialize(computeController, entries, farMultiMesh, numVertexElements, numVaryingElements, batchIndex, requireFVarData);

    delete farMultiMesh;

    return batch;
}

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  /* OSDUTIL_MESH_BATCH_H */
