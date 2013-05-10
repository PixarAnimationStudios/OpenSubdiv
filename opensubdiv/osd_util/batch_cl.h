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

        _vertexBuffer->UpdateData(ptrs, Base::GetVertexOffset(handle.meshIndex), numVertices, _computeController->GetCommandQueue());
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
template <typename VERTEX_BUFFER, typename DRAW_CONTEXT>
OsdUtilMeshBatch<VERTEX_BUFFER, DRAW_CONTEXT, OsdCLComputeController>::
    OsdUtilMeshBatch(OsdCLComputeController *computeController,
                     const std::vector<FarMesh<OsdVertex> const * > &meshVector, int batchIndex) :
         OsdUtilMeshBatchBase<DRAW_CONTEXT>(meshVector, batchIndex),
             _computeController(computeController),
             _computeContext(NULL), _vertexBuffer(NULL), _varyingBuffer(NULL), _drawContext(NULL) {
        
    // create compute contexts
    _computeContext = ComputeContext::Create(Base::_GetFarMesh(), _computeController->GetContext());
}

template <typename VERTEX_BUFFER, typename DRAW_CONTEXT>
OsdUtilMeshBatch<VERTEX_BUFFER, DRAW_CONTEXT, OsdCLComputeController>::~OsdUtilMeshBatch() {
    delete _computeContext;
    delete _drawContext;

    delete _vertexBuffer;
    delete _varyingBuffer;
}

template <typename VERTEX_BUFFER, typename DRAW_CONTEXT> void
OsdUtilMeshBatch<VERTEX_BUFFER, DRAW_CONTEXT, OsdCLComputeController>::InitializeVertexBuffers(int numVertexElements, int numVaryingElements) {

    delete _vertexBuffer;
    delete _varyingBuffer;
    delete _drawContext;

    int numVerts = Base::_GetFarMesh()->GetNumVertices();
    _vertexBuffer = numVertexElements ? VertexBuffer::Create(numVertexElements, numVerts, _computeController->GetContext()) : NULL;
    _varyingBuffer = numVaryingElements ? VertexBuffer::Create(numVaryingElements, numVerts, _computeController->GetContext()) : NULL;
    _drawContext = DrawContext::Create(Base::_GetFarMesh(), _vertexBuffer, /*fvar=*/false);
    assert(_drawContext);

    // update patch arrays
    Base::updatePatchArrays(numVertexElements);
}

#endif  /* OPENSUBDIV_HAS_OPENCL */

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  /* OSD_UTIL_MESH_BATCH_H */
