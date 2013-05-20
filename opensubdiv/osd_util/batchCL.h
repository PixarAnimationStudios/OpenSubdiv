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
#ifndef OSD_UTIL_MESH_BATCH_CL_H
#define OSD_UTIL_MESH_BATCH_CL_H

#include "../version.h"
#include "../osd_util/batch.h"
#include "../osd/clComputeController.h"

#include <vector>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

#ifdef OPENSUBDIV_HAS_OPENCL

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
                                    int batchIndex);

    virtual ~OsdUtilMeshBatch();

    virtual void InitializeVertexBuffers(int numVertexElements, int numVaryingElements);

    virtual typename DrawContext::VertexBufferBinding BindVertexBuffer() { return _vertexBuffer->BindVBO(); }
    virtual DrawContext * GetDrawContext() const { return _drawContext; }

    // update API
    virtual void UpdateCoarseVertices(int meshIndex, const float *ptrs, int numVertices) {

        Base::setMeshDirty(meshIndex);

        _vertexBuffer->UpdateData(ptrs, Base::GetVertexOffset(meshIndex), numVertices, _computeController->GetCommandQueue());
    }

    // utility function
    virtual void FinalizeUpdate() {
        // create kernel batch for dirty handles
        
        FarKernelBatchVector batches;
        Base::populateKernelBatches(batches);
        Base::resetMeshDirty();

        _computeController->Refine(_computeContext, batches, _vertexBuffer);
    }

    VertexBuffer *GetVertexBuffer() const { return _vertexBuffer; }
    VertexBuffer *GetVaryingBuffer() const { return _varyingBuffer; }

private:
    // constructor from far mesh vector
    // XXX: not happy with retaining compute controller..
    OsdUtilMeshBatch(ComputeController *computeController,
                     OsdUtilMeshBatchEntryVector const &entries,
                     FarMesh<OsdVertex> const *farMultiMesh,
                     int batchIndex);

    OsdUtilMeshBatch(FarPatchTables const *patchTables,
                     int numVertices, int numPtexFaces, int batchIndex);

    ComputeController *_computeController;
    ComputeContext *_computeContext;

    VertexBuffer *_vertexBuffer, *_varyingBuffer;
    DrawContext *_drawContext;
};

// -----------------------------------------------------------------------------

// Constructor from farMesh
template <typename VERTEX_BUFFER, typename DRAW_CONTEXT>
OsdUtilMeshBatch<VERTEX_BUFFER, DRAW_CONTEXT, OsdCLComputeController>::
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
    _computeContext = ComputeContext::Create(farMultiMesh, _computeController->GetContext());

    FarPatchTables const * patchTables = farMultiMesh->GetPatchTables();

    // create drawcontext
    _drawContext = DrawContext::Create(patchTables, /*fvar=*/false);
}

// Constructor from patch table
template <typename VERTEX_BUFFER, typename DRAW_CONTEXT>
OsdUtilMeshBatch<VERTEX_BUFFER, DRAW_CONTEXT, OsdCLComputeController>::
OsdUtilMeshBatch(FarPatchTables const *patchTables,
                 int numVertices, int numPtexFaces, int batchIndex) :
    OsdUtilMeshBatchBase<DRAW_CONTEXT>(numVertices,
                                       numPtexFaces,
                                       batchIndex),
        _computeController(NULL), _computeContext(NULL),
        _vertexBuffer(NULL), _varyingBuffer(NULL), _drawContext(NULL) {

    // create drawcontext
    _drawContext = DrawContext::Create(patchTables, /*fvar=*/false);
}

template <typename VERTEX_BUFFER, typename DRAW_CONTEXT>
OsdUtilMeshBatch<VERTEX_BUFFER, DRAW_CONTEXT, OsdCLComputeController>::~OsdUtilMeshBatch() {
    delete _computeContext;
    delete _drawContext;

    delete _vertexBuffer;
    delete _varyingBuffer;
}

template <typename VERTEX_BUFFER, typename DRAW_CONTEXT>
OsdUtilMeshBatch<VERTEX_BUFFER, DRAW_CONTEXT, OsdCLComputeController> *
OsdUtilMeshBatch<VERTEX_BUFFER, DRAW_CONTEXT, OsdCLComputeController>::Create(ComputeController *computeController,
                     const std::vector<FarMesh<OsdVertex> const * > &meshVector, int batchIndex)
{
    std::vector<FarPatchTables::PatchArrayVector> multiFarPatchArray;
    FarMesh <OsdVertex> *farMultiMesh = createMultiMesh(meshVector, multiFarPatchArray);

    FarPatchTables const * patchTables = farMultiMesh->GetPatchTables();

    // create entries
    OsdUtilMeshBatchEntryVector entries;
    createEntries(entries, multiFarPatchArray, patchTables->GetMaxValence(), meshVector);

    OsdUtilMeshBatch<VERTEX_BUFFER, DRAW_CONTEXT, OsdCLComputeController> *batch =
        new OsdUtilMeshBatch<VERTEX_BUFFER, DRAW_CONTEXT, OsdCLComputeController>(
            computeController, entries, farMultiMesh, batchIndex);

    delete farMultiMesh;

    return batch;
}

template <typename VERTEX_BUFFER, typename DRAW_CONTEXT> void
OsdUtilMeshBatch<VERTEX_BUFFER, DRAW_CONTEXT, OsdCLComputeController>::InitializeVertexBuffers(int numVertexElements, int numVaryingElements) {

    delete _vertexBuffer;
    delete _varyingBuffer;

    _vertexBuffer = numVertexElements ? VertexBuffer::Create(numVertexElements, Base::GetNumVertices(), _computeController->GetContext()) : NULL;
    _varyingBuffer = numVaryingElements ? VertexBuffer::Create(numVaryingElements, Base::GetNumVertices(), _computeController->GetContext()) : NULL;

    _drawContext->UpdateVertexTexture(_vertexBuffer);

    // update patch arrays
    Base::updatePatchArrays(numVertexElements);
}

#endif  /* OPENSUBDIV_HAS_OPENCL */

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  /* OSD_UTIL_MESH_BATCH_H */
