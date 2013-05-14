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
#include "../osd_util/meshHandle.h"

#include <vector>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

template <class U> class FarMesh;
class OsdVertex;

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

    int GetVertexOffset(int meshIndex) const { return _vertexOffsets[meshIndex]; }
    int GetPtexFaceOffset(int meshIndex) const { return _ptexFaceOffsets[meshIndex]; }
    int GetQuadOffset(int meshIndex) const { return _quadOffsets[meshIndex]; }
    int GetQuadCount(int meshIndex) const { return _quadCounts[meshIndex]; }

    virtual void InitializeVertexBuffers(int numVertexElements, int numVaryingElements) = 0;

    // similar methods to OsdGLMesh
    virtual void UpdateCoarseVertices(const OsdUtilMeshHandle &handle, const float *ptrs, int numVertices) = 0;
    virtual void FinalizeUpdate() = 0;
 
    OsdDrawContext::PatchArrayVector const & GetPatchArrays(const OsdUtilMeshHandle &meshHandle) const;

    const typename OsdUtilMeshHandle::Collection & GetMeshHandles() const { return _meshHandles; }

    int GetBatchIndex() const { return _batchIndex; }

    int GetNumVertices() const { return _numVertices; }

    int GetNumPtexFaces() const { return _numPtexFaces; }

protected:
    OsdUtilMeshBatchBase(const std::vector<FarMesh<OsdVertex> const * > &meshVector, int batchIndex);

    FarMesh<OsdVertex> * createMultiMesh(std::vector<FarMesh<OsdVertex> const * > const & meshVector,
                                         std::vector<FarPatchTables::PatchArrayVector> & multiFarPatchArray);

    void createPatchArrays(std::vector<FarPatchTables::PatchArrayVector> const & multiFarPatchArray,
                           int maxValence, bool adaptive,
                           std::vector<FarMesh<OsdVertex> const * > const & meshVector);

    void setMeshDirty(int meshIndex);
    void resetMeshDirty();
    void updatePatchArrays(int numVertexElements);

    FarKernelBatchVector getKernelBatches();

private:
    typename OsdUtilMeshHandle::Collection _meshHandles;
    std::vector<OsdDrawContext::PatchArrayVector> _multiPatchArrays;
    FarKernelBatchVector _allKernelBatches;

    std::vector<int>           _vertexOffsets;
    std::vector<int>           _ptexFaceOffsets;
    std::vector<bool>          _dirtyFlags;

    std::vector<int>           _quadOffsets;  // for uniform subdivision
    std::vector<int>           _quadCounts;   // for uniform subdivision

    int _batchIndex;
    int _numVertices;
    int _numPtexFaces;
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
    
    // XXX: not happy with retaining compute controller..
    OsdUtilMeshBatch(ComputeController *computeController,
                     const std::vector<FarMesh<OsdVertex> const * > &meshVector, int batchIndex);

    virtual ~OsdUtilMeshBatch();

    virtual void InitializeVertexBuffers(int numVertexElements, int numVaryingElements);

    virtual typename DrawContext::VertexBufferBinding BindVertexBuffer() { return _vertexBuffer->BindVBO(); }
    virtual DrawContext * GetDrawContext() const { return _drawContext; }

    // update API
    virtual void UpdateCoarseVertices(const OsdUtilMeshHandle &handle,
                                      const float *ptrs, int numVertices) {

        assert(handle.batchIndex == Base::GetBatchIndex());

        Base::setMeshDirty(handle.meshIndex);

        _vertexBuffer->UpdateData(ptrs, Base::GetVertexOffset(handle.meshIndex), numVertices);
    }

    // utility function
    virtual void FinalizeUpdate() {
        // create kernel batch for dirty handles
        
        FarKernelBatchVector batches = Base::getKernelBatches();
        Base::resetMeshDirty();

        _computeController->Refine(_computeContext, batches, _vertexBuffer);
    }

    VertexBuffer *GetVertexBuffer() const { return _vertexBuffer; }
    VertexBuffer *GetVaryingBuffer() const { return _varyingBuffer; }

private:
    ComputeController *_computeController;
    ComputeContext *_computeContext;

    VertexBuffer *_vertexBuffer, *_varyingBuffer;
    DrawContext *_drawContext;
};

// -----------------------------------------------------------------------------
template <typename DRAW_CONTEXT>
OsdUtilMeshBatchBase<DRAW_CONTEXT>::
    OsdUtilMeshBatchBase(std::vector<FarMesh<OsdVertex> const * > const &meshVector, int batchIndex) :
        _batchIndex(batchIndex) {

}
template <typename DRAW_CONTEXT>
OsdUtilMeshBatchBase<DRAW_CONTEXT>::~OsdUtilMeshBatchBase() {
}

template <typename DRAW_CONTEXT> FarMesh<OsdVertex> *
OsdUtilMeshBatchBase<DRAW_CONTEXT>::createMultiMesh(std::vector<FarMesh<OsdVertex> const * > const & meshVector,
                                                    std::vector<FarPatchTables::PatchArrayVector> & multiFarPatchArray) {

    // create multimesh
    FarMultiMeshFactory<OsdVertex> multiMeshFactory;

    FarMesh <OsdVertex> *farMultiMesh = multiMeshFactory.Create(meshVector);

    // copy batches
    _allKernelBatches = farMultiMesh->GetKernelBatches();
    
    _numVertices = farMultiMesh->GetNumVertices();
    _numPtexFaces = farMultiMesh->GetNumPtexFaces();

    // return patch arrays
    multiFarPatchArray = multiMeshFactory.GetMultiPatchArrays();

    return farMultiMesh;
}

template <typename DRAW_CONTEXT> void
OsdUtilMeshBatchBase<DRAW_CONTEXT>::createPatchArrays(std::vector<FarPatchTables::PatchArrayVector> const & multiFarPatchArray,
                                                      int maxValence, bool adaptive,
                                                      std::vector<FarMesh<OsdVertex> const * > const & meshVector) {

    // create osd patch array per mesh (note: numVertexElements will be updated later on InitializeVertexBuffers)
    _multiPatchArrays.resize(multiFarPatchArray.size());
    for (int i = 0; i < (int)multiFarPatchArray.size(); ++i) {
        OsdDrawContext::ConvertPatchArrays(multiFarPatchArray[i], _multiPatchArrays[i], maxValence, 0);
    }

    // create mesh handles and prepare patch offsets
    _meshHandles.reserve(meshVector.size());
    int vertexOffset = 0, ptexFaceOffset = 0, quadOffset = 0;
    for (size_t i = 0; i < meshVector.size(); ++i) {
        int numVertices = meshVector[i]->GetNumVertices();
        int numPtexFaces = meshVector[i]->GetNumPtexFaces();

        _meshHandles.push_back(OsdUtilMeshHandle(_batchIndex, (int)i));

        _vertexOffsets.push_back(vertexOffset);
        _ptexFaceOffsets.push_back(ptexFaceOffset);
        _quadOffsets.push_back(quadOffset);

        // XXX: temporary for uniform
        int maxLevel = meshVector[i]->GetSubdivisionTables()->GetMaxLevel();
        int numQuads = adaptive ? 0 : (int)meshVector[i]->GetFaceVertices(maxLevel-1).size()/4;
        _quadCounts.push_back(numQuads);

        vertexOffset += numVertices;
        ptexFaceOffset += numPtexFaces;

        quadOffset += numQuads;
    }
    _dirtyFlags.resize(meshVector.size());
    resetMeshDirty();
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
OsdUtilMeshBatchBase<DRAW_CONTEXT>::updatePatchArrays(int numVertexElements) {
    for (int i = 0; i < (int)_multiPatchArrays.size(); ++i) {
        for (int j = 0; j < (int)_multiPatchArrays[i].size(); ++j) {
            OsdDrawContext::PatchDescriptor desc = _multiPatchArrays[i][j].GetDescriptor();
            desc.SetNumElements(numVertexElements);
            _multiPatchArrays[i][j].SetDescriptor(desc);
        }
    }
}

template <typename DRAW_CONTEXT> FarKernelBatchVector
OsdUtilMeshBatchBase<DRAW_CONTEXT>::getKernelBatches() {
    FarKernelBatchVector result;
    for (FarKernelBatchVector::const_iterator it = _allKernelBatches.begin();
         it != _allKernelBatches.end(); ++it) {
        if (_dirtyFlags[it->GetMeshIndex()]) {
            result.push_back(*it);
        }
    }
    return result;
}

template <typename DRAW_CONTEXT> OsdDrawContext::PatchArrayVector const &
OsdUtilMeshBatchBase<DRAW_CONTEXT>::GetPatchArrays(const OsdUtilMeshHandle &meshHandle) const {
    return _multiPatchArrays[meshHandle.meshIndex];
}

// -----------------------------------------------------------------------------
template <typename VERTEX_BUFFER, typename DRAW_CONTEXT, typename COMPUTE_CONTROLLER>
OsdUtilMeshBatch<VERTEX_BUFFER, DRAW_CONTEXT, COMPUTE_CONTROLLER>::
OsdUtilMeshBatch(ComputeController *computeController,
                     const std::vector<FarMesh<OsdVertex> const * > &meshVector, int batchIndex) :
         OsdUtilMeshBatchBase<DRAW_CONTEXT>(meshVector, batchIndex),
             _computeController(computeController),
             _computeContext(NULL), _vertexBuffer(NULL), _varyingBuffer(NULL), _drawContext(NULL) {

    std::vector<FarPatchTables::PatchArrayVector> multiFarPatchArray;
    FarMesh <OsdVertex> *farMultiMesh = Base::createMultiMesh(meshVector, multiFarPatchArray);

    FarPatchTables const * patchTables = farMultiMesh->GetPatchTables();
    bool adaptive = (patchTables != 0);

    // create compute contexts
    _computeContext = ComputeContext::Create(farMultiMesh);

    // create drawcontext
    if (patchTables)
        _drawContext = DrawContext::Create(patchTables, /*fvar=*/false);
    else
        _drawContext = DrawContext::Create(farMultiMesh, /*fvar=*/false);  // XXX: will be retired

    // temporary function. will be refactored
    Base::createPatchArrays(multiFarPatchArray, patchTables->GetMaxValence(), adaptive, meshVector);

    delete farMultiMesh;
}

template <typename VERTEX_BUFFER, typename DRAW_CONTEXT, typename COMPUTE_CONTROLLER>
OsdUtilMeshBatch<VERTEX_BUFFER, DRAW_CONTEXT, COMPUTE_CONTROLLER>::~OsdUtilMeshBatch() {
    delete _computeContext;
    delete _drawContext;

    delete _vertexBuffer;
    delete _varyingBuffer;
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
