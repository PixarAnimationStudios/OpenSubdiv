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
#ifndef OSD_UTIL_MESH_BATCH_H
#define OSD_UTIL_MESH_BATCH_H

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
    virtual DrawContext * GetDrawContext() const = 0;

    // entry accessors
    int GetVertexOffset(int meshIndex) const
        { return _entries[meshIndex].vertexOffset; }
    int GetPtexFaceOffset(int meshIndex) const
        { return _entries[meshIndex].ptexFaceOffset; }
    OsdDrawContext::PatchArrayVector const & GetPatchArrays(int meshIndex) const
        { return _entries[meshIndex].patchArrays; }

    // vertex buffer initializer
    virtual void InitializeVertexBuffers(int numVertexElements, int numVaryingElements) = 0;

    // update APIs
    virtual void UpdateCoarseVertices(int meshIndex, const float *ptrs, int numVertices) = 0;
    virtual void FinalizeUpdate() = 0;

    int GetBatchIndex() const { return _batchIndex; }

    int GetNumVertices() const { return _numVertices; }

    int GetNumPtexFaces() const { return _numPtexFaces; }

protected:
    OsdUtilMeshBatchBase(OsdUtilMeshBatchEntryVector const & entries, int numVertices, int numPtexFaces, int batchIndex);

    void setKernelBatches(FarKernelBatchVector const &batches);
    void setMeshDirty(int meshIndex);
    void resetMeshDirty();
    void updatePatchArrays(int numVertexElements);

    void populateKernelBatches(FarKernelBatchVector &result);

private:
    // compute batch
    FarKernelBatchVector _allKernelBatches;

    // drawing batch entries
    OsdUtilMeshBatchEntryVector _entries;

    // update flags
    std::vector<bool>          _dirtyFlags;

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
                                    int batchIndex);

    // constructor (for client defined arbitrary patches)
    static OsdUtilMeshBatch *Create(FarPatchTables const *patchTables,
                                    OsdUtilMeshBatchEntryVector const &entries,
                                    int numVertices,
                                    int numPtexFaces,
                                    int batchIndex);

    virtual ~OsdUtilMeshBatch();

    virtual void InitializeVertexBuffers(int numVertexElements, int numVaryingElements);

    virtual typename DrawContext::VertexBufferBinding BindVertexBuffer() { return _vertexBuffer->BindVBO(); }

    virtual DrawContext * GetDrawContext() const { return _drawContext; }

    // update APIs
    virtual void UpdateCoarseVertices(int meshIndex, const float *ptrs, int numVertices) {

        Base::setMeshDirty(meshIndex);

        _vertexBuffer->UpdateData(ptrs, Base::GetVertexOffset(meshIndex), numVertices);
    }

    virtual void FinalizeUpdate() {
        // create kernel batch for dirty handles

        if (not _computeController)
            return;

        FarKernelBatchVector batches;
        Base::populateKernelBatches(batches);
        Base::resetMeshDirty();

        _computeController->Refine(_computeContext, batches, _vertexBuffer);
    }

    VertexBuffer *GetVertexBuffer() const { return _vertexBuffer; }

    VertexBuffer *GetVaryingBuffer() const { return _varyingBuffer; }

private:
    OsdUtilMeshBatch(ComputeController *computeController,
                     OsdUtilMeshBatchEntryVector const &entries,
                     FarMesh<OsdVertex> const *farMultiMesh,
                     int batchIndex);

    OsdUtilMeshBatch(FarPatchTables const *patchTables,
                     OsdUtilMeshBatchEntryVector const &entries,
                     int numVertices, int numPtexFaces, int batchIndex);

    ComputeController *_computeController;
    ComputeContext *_computeContext;

    VertexBuffer *_vertexBuffer, *_varyingBuffer;
    DrawContext *_drawContext;
};

// -----------------------------------------------------------------------------
template <typename DRAW_CONTEXT>
    OsdUtilMeshBatchBase<DRAW_CONTEXT>::OsdUtilMeshBatchBase(OsdUtilMeshBatchEntryVector const & entries, int numVertices, int numPtexFaces, int batchIndex) :
    _entries(entries), _numVertices(numVertices), _numPtexFaces(numPtexFaces), _batchIndex(batchIndex) {

    // init dirty flags
    _dirtyFlags.resize(entries.size());
    resetMeshDirty();
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
OsdUtilMeshBatchBase<DRAW_CONTEXT>::updatePatchArrays(int numVertexElements) {
    for (int i = 0; i < (int)_entries.size(); ++i) {
        for (int j = 0; j < (int)_entries[i].patchArrays.size(); ++j) {
            OsdDrawContext::PatchDescriptor desc = _entries[i].patchArrays[j].GetDescriptor();
            desc.SetNumElements(numVertexElements);
            _entries[i].patchArrays[j].SetDescriptor(desc);
        }
    }
}

template <typename DRAW_CONTEXT> void
OsdUtilMeshBatchBase<DRAW_CONTEXT>::populateKernelBatches(FarKernelBatchVector &result) {

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

    // create osd patch array per mesh (note: numVertexElements will be updated later on InitializeVertexBuffers)
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

// Constructor from farMesh
template <typename VERTEX_BUFFER, typename DRAW_CONTEXT, typename COMPUTE_CONTROLLER>
OsdUtilMeshBatch<VERTEX_BUFFER, DRAW_CONTEXT, COMPUTE_CONTROLLER>::
OsdUtilMeshBatch(ComputeController *computeController,
                 OsdUtilMeshBatchEntryVector const &entries,
                 FarMesh<OsdVertex> const *farMultiMesh,
                 int batchIndex) :
    OsdUtilMeshBatchBase<DRAW_CONTEXT>(entries,
                                       farMultiMesh->GetNumVertices(),
                                       farMultiMesh->GetNumPtexFaces(),
                                       batchIndex),
             _computeController(computeController),
             _computeContext(NULL), _vertexBuffer(NULL), _varyingBuffer(NULL), _drawContext(NULL) {

    // copy batches
    Base::setKernelBatches(farMultiMesh->GetKernelBatches());

    // create compute contexts
    _computeContext = ComputeContext::Create(farMultiMesh);

    FarPatchTables const * patchTables = farMultiMesh->GetPatchTables();

    // create drawcontext
    _drawContext = DrawContext::Create(patchTables, /*fvar=*/false);
}

// Constructor from patch table
template <typename VERTEX_BUFFER, typename DRAW_CONTEXT, typename COMPUTE_CONTROLLER>
OsdUtilMeshBatch<VERTEX_BUFFER, DRAW_CONTEXT, COMPUTE_CONTROLLER>::
OsdUtilMeshBatch(FarPatchTables const *patchTables,
                 OsdUtilMeshBatchEntryVector const &entries,
                 int numVertices, int numPtexFaces, int batchIndex) :
    OsdUtilMeshBatchBase<DRAW_CONTEXT>(entries,
                                       numVertices,
                                       numPtexFaces,
                                       batchIndex),
        _computeController(NULL), _computeContext(NULL),
        _vertexBuffer(NULL), _varyingBuffer(NULL), _drawContext(NULL) {

    // create drawcontext
    _drawContext = DrawContext::Create(patchTables, /*fvar=*/false);
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
                     const std::vector<FarMesh<OsdVertex> const * > &meshVector, int batchIndex)
{
    std::vector<FarPatchTables::PatchArrayVector> multiFarPatchArray;
    FarMesh <OsdVertex> *farMultiMesh = createMultiMesh(meshVector, multiFarPatchArray);

    FarPatchTables const * patchTables = farMultiMesh->GetPatchTables();

    // create entries
    OsdUtilMeshBatchEntryVector entries;
    createEntries(entries, multiFarPatchArray, patchTables->GetMaxValence(), meshVector);

    OsdUtilMeshBatch<VERTEX_BUFFER, DRAW_CONTEXT, COMPUTE_CONTROLLER> *batch =
        new OsdUtilMeshBatch<VERTEX_BUFFER, DRAW_CONTEXT, COMPUTE_CONTROLLER>(
            computeController, entries, farMultiMesh, batchIndex);

    delete farMultiMesh;

    return batch;
}

template <typename VERTEX_BUFFER, typename DRAW_CONTEXT, typename COMPUTE_CONTROLLER>
OsdUtilMeshBatch<VERTEX_BUFFER, DRAW_CONTEXT, COMPUTE_CONTROLLER> *
OsdUtilMeshBatch<VERTEX_BUFFER, DRAW_CONTEXT, COMPUTE_CONTROLLER>::Create(FarPatchTables const *patchTables,
                     OsdUtilMeshBatchEntryVector const &entries, int numVertices, int numPtexFaces, int batchIndex)
{
    OsdUtilMeshBatch<VERTEX_BUFFER, DRAW_CONTEXT, COMPUTE_CONTROLLER> *batch =
        new OsdUtilMeshBatch<VERTEX_BUFFER, DRAW_CONTEXT, COMPUTE_CONTROLLER>(
            patchTables, entries, numVertices, numPtexFaces, batchIndex);

    return batch;
}

template <typename VERTEX_BUFFER, typename DRAW_CONTEXT, typename COMPUTE_CONTROLLER> void
OsdUtilMeshBatch<VERTEX_BUFFER, DRAW_CONTEXT, COMPUTE_CONTROLLER>::InitializeVertexBuffers(int numVertexElements, int numVaryingElements) {

    delete _vertexBuffer;
    delete _varyingBuffer;

    _vertexBuffer = numVertexElements ? VertexBuffer::Create(numVertexElements, Base::GetNumVertices()) : NULL;
    _varyingBuffer = numVaryingElements ? VertexBuffer::Create(numVaryingElements, Base::GetNumVertices()) : NULL;

    _drawContext->UpdateVertexTexture(_vertexBuffer);

    // update patch arrays
    Base::updatePatchArrays(numVertexElements);
}

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  /* OSD_UTIL_MESH_BATCH_H */
