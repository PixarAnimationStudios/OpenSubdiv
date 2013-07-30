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
#ifndef OSDUTIL_MESH_BATCH_H
#define OSDUTIL_MESH_BATCH_H

#include "../version.h"
#include "../far/multiMeshFactory.h"
#include "../far/patchTables.h"
#include "../osd/vertexDescriptor.h"

#include <vector>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

template <class U> class FarMesh;
class OsdVertex;


// ----------------------------------------------------------------------------
// OsdUtilMeshBatchEntry
//
//
//
struct OsdUtilMeshBatchEntry {
    OsdDrawContext::PatchArrayVector patchArrays;
    int vertexOffset;       // relative offset to first coarse vertex in vbo
    int ptexFaceOffset;     // relative offset to first ptex face ID
};

typedef std::vector<OsdUtilMeshBatchEntry> OsdUtilMeshBatchEntryVector;

// ----------------------------------------------------------------------------
// OsdUtilMeshBatchBase
//
//  base batching class which holds kernel batches and patch arrays.
// 
template <typename DRAW_CONTEXT>
class OsdUtilMeshBatchBase {
public:
    typedef DRAW_CONTEXT DrawContext;

    virtual ~OsdUtilMeshBatchBase();

    virtual typename DrawContext::VertexBufferBinding BindVertexBuffer() = 0;
    virtual typename DrawContext::VertexBufferBinding BindVaryingBuffer() = 0;
    virtual DrawContext * GetDrawContext() const = 0;

    // entry accessors
    int GetVertexOffset(int meshIndex) const
        { return _entries[meshIndex].vertexOffset; }
    int GetPtexFaceOffset(int meshIndex) const
        { return _entries[meshIndex].ptexFaceOffset; }
    OsdDrawContext::PatchArrayVector const & GetPatchArrays(int meshIndex) const
        { return _entries[meshIndex].patchArrays; }

    // update APIs
    virtual void UpdateCoarseVertices(int meshIndex, const float *data, int numVertices) = 0;
    virtual void UpdateCoarseVaryings(int meshIndex, const float *data, int numVertices) = 0;
    virtual void FinalizeUpdate() = 0;

    int GetBatchIndex() const { return _batchIndex; }

    int GetNumVertices() const { return _numVertices; }

    int GetNumPtexFaces() const { return _numPtexFaces; }

protected:
    OsdUtilMeshBatchBase() {}

    bool initialize(OsdUtilMeshBatchEntryVector const & entries,
                    int numVertices,
                    int numPtexFaces,
                    int numVertexElements,
                    int batchIndex);

    void setKernelBatches(FarKernelBatchVector const &batches);
    void setMeshDirty(int meshIndex);
    void resetMeshDirty();

    void populateDirtyKernelBatches(FarKernelBatchVector &result);

private:
    // compute batch
    FarKernelBatchVector _allKernelBatches;

    // drawing batch entries
    OsdUtilMeshBatchEntryVector _entries;

    // update flags
    std::vector<bool>          _dirtyFlags;   // same size as _entries

    int _numVertices;
    int _numPtexFaces;
    int _batchIndex;
};

// ----------------------------------------------------------------------------
// OsdUtilMeshBatch
//
//  derived batching class which contains vertexbuffer, computecontext, drawcontext
// 
template <typename VERTEX_BUFFER, typename DRAW_CONTEXT, typename COMPUTE_CONTROLLER>
class OsdUtilMeshBatch : public OsdUtilMeshBatchBase<DRAW_CONTEXT> {
public:
    typedef VERTEX_BUFFER VertexBuffer;
    typedef typename COMPUTE_CONTROLLER::ComputeContext ComputeContext;
    typedef DRAW_CONTEXT DrawContext;
    typedef COMPUTE_CONTROLLER ComputeController;
    typedef OsdUtilMeshBatchBase<DRAW_CONTEXT> Base;

    // constructor from far mesh vector
    // XXX: not happy with retaining compute controller..
    static OsdUtilMeshBatch *Create(ComputeController *computeController,
                                    std::vector<FarMesh<OsdVertex> const * > const &meshVector,
                                    int numVertexElements,
                                    int numVaryingElements,
                                    int batchIndex,
                                    bool requireFVarData=false);

    // constructor (for client defined arbitrary patches)
    static OsdUtilMeshBatch *Create(FarPatchTables const *patchTables,
                                    OsdUtilMeshBatchEntryVector const &entries,
                                    int numVertices,
                                    int numPtexFaces,
                                    int numVertexElements,
                                    int numVaryingElements,
                                    int batchIndex,
                                    bool requireFVarData=false);

    virtual ~OsdUtilMeshBatch();

    virtual typename DrawContext::VertexBufferBinding BindVertexBuffer() {
        if (not _vertexBuffer)
            return 0;
        return _vertexBuffer->BindVBO();
    }

    virtual typename DrawContext::VertexBufferBinding BindVaryingBuffer() {
        if (not _varyingBuffer)
            return 0;
        return _varyingBuffer->BindVBO();
    }

    virtual DrawContext * GetDrawContext() const { return _drawContext; }

    // update APIs
    virtual void UpdateCoarseVertices(int meshIndex, const float *data, int numVertices) {

        if (not _vertexBuffer)
            return;

        Base::setMeshDirty(meshIndex);

        _vertexBuffer->UpdateData(data, Base::GetVertexOffset(meshIndex), numVertices);
    }

    virtual void UpdateCoarseVaryings(int meshIndex, const float *data, int numVertices) {

        if (not _varyingBuffer)
            return;

        Base::setMeshDirty(meshIndex);

        _varyingBuffer->UpdateData(data, Base::GetVertexOffset(meshIndex), numVertices);
    }

    virtual void FinalizeUpdate() {
        // create kernel batch for dirty handles

        if (not _computeController)
            return;

        FarKernelBatchVector batches;
        Base::populateDirtyKernelBatches(batches);
        Base::resetMeshDirty();

        _computeController->Refine(_computeContext, batches, _vertexBuffer, _varyingBuffer);
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
                    OsdUtilMeshBatchEntryVector const &entries,
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
template <typename DRAW_CONTEXT> bool
OsdUtilMeshBatchBase<DRAW_CONTEXT>::initialize(OsdUtilMeshBatchEntryVector const & entries,
                                               int numVertices,
                                               int numPtexFaces,
                                               int numVertexElements,
                                               int batchIndex) {
    _entries = entries;
    _numVertices = numVertices;
    _numPtexFaces = numPtexFaces;
    _batchIndex = batchIndex;

    // update patcharrays in entries
    for (int i = 0; i < (int)_entries.size(); ++i) {
        for (int j = 0; j < (int)_entries[i].patchArrays.size(); ++j) {
            OsdDrawContext::PatchDescriptor desc = _entries[i].patchArrays[j].GetDescriptor();
            desc.SetNumElements(numVertexElements);
            _entries[i].patchArrays[j].SetDescriptor(desc);
        }
    }

    // init dirty flags
    _dirtyFlags.resize(entries.size());
    resetMeshDirty();

    return true;
}

template <typename DRAW_CONTEXT>
OsdUtilMeshBatchBase<DRAW_CONTEXT>::~OsdUtilMeshBatchBase() {
}

template <typename DRAW_CONTEXT> void
OsdUtilMeshBatchBase<DRAW_CONTEXT>::setMeshDirty(int meshIndex) {

    assert(meshIndex < (int)_dirtyFlags.size());
    _dirtyFlags[meshIndex] = true;
}

template <typename DRAW_CONTEXT> void
OsdUtilMeshBatchBase<DRAW_CONTEXT>::resetMeshDirty() {
    std::fill(_dirtyFlags.begin(), _dirtyFlags.end(), false);
}

template <typename DRAW_CONTEXT> void
OsdUtilMeshBatchBase<DRAW_CONTEXT>::setKernelBatches(FarKernelBatchVector const &batches) {
    _allKernelBatches = batches;
}

template <typename DRAW_CONTEXT> void
OsdUtilMeshBatchBase<DRAW_CONTEXT>::populateDirtyKernelBatches(FarKernelBatchVector &result) {

    result.clear();
    for (FarKernelBatchVector::const_iterator it = _allKernelBatches.begin();
         it != _allKernelBatches.end(); ++it) {
        if (_dirtyFlags[it->GetMeshIndex()]) {
            result.push_back(*it);
        }
    }
}

// -----------------------------------------------------------------------------
inline FarMesh<OsdVertex> *
createMultiMesh(std::vector<FarMesh<OsdVertex> const * > const & meshVector,
                std::vector<FarPatchTables::PatchArrayVector> & multiFarPatchArray) {

    // create multimesh
    FarMultiMeshFactory<OsdVertex> multiMeshFactory;
    FarMesh <OsdVertex> *farMultiMesh = multiMeshFactory.Create(meshVector);

    // return patch arrays
    multiFarPatchArray = multiMeshFactory.GetMultiPatchArrays();

    return farMultiMesh;
}

inline void
createEntries(OsdUtilMeshBatchEntryVector &result,
              std::vector<FarPatchTables::PatchArrayVector> const & multiFarPatchArray,
              int maxValence,
              std::vector<FarMesh<OsdVertex> const * > const & meshVector) {

    // create osd patch array per mesh (note: numVertexElements will be updated later)
    int numEntries = (int)multiFarPatchArray.size();
    result.resize(numEntries);
    for (int i = 0; i < numEntries; ++i) {
        OsdDrawContext::ConvertPatchArrays(multiFarPatchArray[i],
                                           result[i].patchArrays,
                                           maxValence, 0);
    }

    // set entries
    int vertexOffset = 0, ptexFaceOffset = 0;
    for (size_t i = 0; i < meshVector.size(); ++i) {
        int numVertices = meshVector[i]->GetNumVertices();
        int numPtexFaces = meshVector[i]->GetNumPtexFaces();

        result[i].vertexOffset = vertexOffset;
        result[i].ptexFaceOffset = ptexFaceOffset;

        vertexOffset += numVertices;
        ptexFaceOffset += numPtexFaces;
    }
}

template <typename VERTEX_BUFFER, typename DRAW_CONTEXT, typename COMPUTE_CONTROLLER>
OsdUtilMeshBatch<VERTEX_BUFFER, DRAW_CONTEXT, COMPUTE_CONTROLLER>::OsdUtilMeshBatch() :
    _computeController(NULL), _computeContext(NULL), _vertexBuffer(NULL), _varyingBuffer(NULL), _drawContext(NULL) {
}

template <typename VERTEX_BUFFER, typename DRAW_CONTEXT, typename COMPUTE_CONTROLLER> bool
OsdUtilMeshBatch<VERTEX_BUFFER, DRAW_CONTEXT, COMPUTE_CONTROLLER>::initialize(ComputeController *computeController,
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
    _computeContext = ComputeContext::Create(farMultiMesh);

    if (not _computeContext) return false;

    FarPatchTables const * patchTables = farMultiMesh->GetPatchTables();

    _vertexBuffer = numVertexElements ? VertexBuffer::Create(numVertexElements, Base::GetNumVertices()) : NULL;
    _varyingBuffer = numVaryingElements ? VertexBuffer::Create(numVaryingElements, Base::GetNumVertices()) : NULL;

    _drawContext = DrawContext::Create(patchTables, requireFVarData);
    if (not _drawContext) return false;

    _drawContext->UpdateVertexTexture(_vertexBuffer);

    return true;
}

// Constructor from patch table
template <typename VERTEX_BUFFER, typename DRAW_CONTEXT, typename COMPUTE_CONTROLLER> bool
OsdUtilMeshBatch<VERTEX_BUFFER, DRAW_CONTEXT, COMPUTE_CONTROLLER>::initialize(FarPatchTables const *patchTables,
                                                                              OsdUtilMeshBatchEntryVector const &entries,
                                                                              int numVertices, int numPtexFaces,
                                                                              int numVertexElements,
                                                                              int numVaryingElements,
                                                                              int batchIndex,
                                                                              bool requireFVarData) {

    Base::initialize(entries, numVertices, numPtexFaces, numVertexElements, batchIndex);

    _vertexBuffer = numVertexElements ? VertexBuffer::Create(numVertexElements, numVertices) : NULL;
    _varyingBuffer = numVaryingElements ? VertexBuffer::Create(numVaryingElements, numVertices) : NULL;

    _drawContext = DrawContext::Create(patchTables, requireFVarData);
    if (not _drawContext) return false;

    _drawContext->UpdateVertexTexture(_vertexBuffer);

    return true;
}

template <typename VERTEX_BUFFER, typename DRAW_CONTEXT, typename COMPUTE_CONTROLLER>
OsdUtilMeshBatch<VERTEX_BUFFER, DRAW_CONTEXT, COMPUTE_CONTROLLER>::~OsdUtilMeshBatch() {
    delete _computeContext;
    delete _drawContext;

    delete _vertexBuffer;
    delete _varyingBuffer;
}

template <typename VERTEX_BUFFER, typename DRAW_CONTEXT, typename COMPUTE_CONTROLLER>
OsdUtilMeshBatch<VERTEX_BUFFER, DRAW_CONTEXT, COMPUTE_CONTROLLER> *
OsdUtilMeshBatch<VERTEX_BUFFER, DRAW_CONTEXT, COMPUTE_CONTROLLER>::Create(ComputeController *computeController,
                                                                          const std::vector<FarMesh<OsdVertex> const * > &meshVector,
                                                                          int numVertexElements,
                                                                          int numVaryingElements,
                                                                          int batchIndex,
                                                                          bool requireFVarData)
{
    std::vector<FarPatchTables::PatchArrayVector> multiFarPatchArray;
    FarMesh <OsdVertex> *farMultiMesh = createMultiMesh(meshVector, multiFarPatchArray);

    FarPatchTables const * patchTables = farMultiMesh->GetPatchTables();

    // create entries
    OsdUtilMeshBatchEntryVector entries;
    createEntries(entries, multiFarPatchArray, patchTables->GetMaxValence(), meshVector);

    OsdUtilMeshBatch<VERTEX_BUFFER, DRAW_CONTEXT, COMPUTE_CONTROLLER> *batch =
        new OsdUtilMeshBatch<VERTEX_BUFFER, DRAW_CONTEXT, COMPUTE_CONTROLLER>();

    batch->initialize(computeController, entries, farMultiMesh, numVertexElements, numVaryingElements, batchIndex, requireFVarData);

    delete farMultiMesh;

    return batch;
}

template <typename VERTEX_BUFFER, typename DRAW_CONTEXT, typename COMPUTE_CONTROLLER>
OsdUtilMeshBatch<VERTEX_BUFFER, DRAW_CONTEXT, COMPUTE_CONTROLLER> *
OsdUtilMeshBatch<VERTEX_BUFFER, DRAW_CONTEXT, COMPUTE_CONTROLLER>::Create(FarPatchTables const *patchTables,
                                                                          OsdUtilMeshBatchEntryVector const &entries,
                                                                          int numVertices,
                                                                          int numPtexFaces,
                                                                          int numVertexElements,
                                                                          int numVaryingElements,
                                                                          int batchIndex,
                                                                          bool requireFVarData)
{
    OsdUtilMeshBatch<VERTEX_BUFFER, DRAW_CONTEXT, COMPUTE_CONTROLLER> *batch =
        new OsdUtilMeshBatch<VERTEX_BUFFER, DRAW_CONTEXT, COMPUTE_CONTROLLER>();

    batch->initialize(patchTables, entries, numVertices, numPtexFaces, numVertexElements, numVaryingElements, batchIndex, requireFVarData);

    return batch;
}

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  /* OSDUTIL_MESH_BATCH_H */
