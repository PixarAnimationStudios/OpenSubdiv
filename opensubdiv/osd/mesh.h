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

#ifndef OSD_MESH_H
#define OSD_MESH_H

#include "../version.h"

#include "../far/kernelBatch.h"
#include "../far/topologyRefiner.h"
#include "../far/patchTablesFactory.h"
#include "../far/stencilTables.h"
#include "../far/stencilTablesFactory.h"

#include "../osd/error.h"
#include "../osd/vertex.h"
#include "../osd/vertexDescriptor.h"

#include <bitset>
#include <cassert>
#include <cstring>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

enum MeshBits {
    MeshAdaptive          = 0,
    MeshInterleaveVarying = 1,
    MeshPtexData          = 2,
    MeshFVarData          = 3,
    NUM_MESH_BITS         = 4,
};
typedef std::bitset<NUM_MESH_BITS> MeshBitset;

template <class DRAW_CONTEXT>
class MeshInterface {
public:
    typedef DRAW_CONTEXT DrawContext;
    typedef typename DrawContext::VertexBufferBinding VertexBufferBinding;

public:
    MeshInterface() { }

    virtual ~MeshInterface() { }

    virtual int GetNumVertices() const = 0;

    virtual void UpdateVertexBuffer(float const *vertexData, int startVertex, int numVerts) = 0;

    virtual void UpdateVaryingBuffer(float const *varyingData, int startVertex, int numVerts) = 0;

    virtual void Refine() = 0;

    virtual void Refine(VertexBufferDescriptor const *vertexDesc,
                        VertexBufferDescriptor const *varyingDesc,
                        bool interleaved) = 0;

    virtual void Synchronize() = 0;

    virtual DrawContext * GetDrawContext() = 0;

    virtual VertexBufferBinding BindVertexBuffer() = 0;

    virtual VertexBufferBinding BindVaryingBuffer() = 0;

    virtual void SetFVarDataChannel(int fvarWidth, std::vector<float> const & fvarData) = 0;

protected:

    static inline int getNumVertices(Far::TopologyRefiner const & refiner) {
        return refiner.IsUniform() ?
            refiner.GetNumVertices(0) + refiner.GetNumVertices(refiner.GetMaxLevel()) :
                refiner.GetNumVerticesTotal();
    }

    static inline void refineMesh(Far::TopologyRefiner & refiner, int level, bool adaptive) {

        bool fullTopologyInLastLevel = refiner.GetNumFVarChannels()>0;

        if (adaptive) {
            refiner.RefineAdaptive(level, fullTopologyInLastLevel);
        } else {
            refiner.RefineUniform(level, fullTopologyInLastLevel);
        }
    }
};



template <class VERTEX_BUFFER, class COMPUTE_CONTROLLER, class DRAW_CONTEXT>
class Mesh : public MeshInterface<DRAW_CONTEXT> {
public:
    typedef VERTEX_BUFFER VertexBuffer;
    typedef COMPUTE_CONTROLLER ComputeController;
    typedef typename ComputeController::ComputeContext ComputeContext;
    typedef DRAW_CONTEXT DrawContext;
    typedef typename DrawContext::VertexBufferBinding VertexBufferBinding;

    Mesh(ComputeController * computeController,
            Far::TopologyRefiner * refiner,
            int numVertexElements,
            int numVaryingElements,
            int level,
            MeshBitset bits = MeshBitset()) :

            _refiner(refiner),
            _vertexBuffer(0),
            _varyingBuffer(0),
            _computeContext(0),
            _computeController(computeController),
            _drawContext(0) {

        assert(_refiner);

        MeshInterface<DRAW_CONTEXT>::refineMesh(*_refiner, level, bits.test(MeshAdaptive));

        initializeVertexBuffers(numVertexElements, numVaryingElements, bits);

        initializeComputeContext(numVertexElements, numVaryingElements);

        initializeDrawContext(numVertexElements, bits);
    }

    Mesh(ComputeController * computeController,
            Far::TopologyRefiner * refiner,
            VertexBuffer * vertexBuffer,
            VertexBuffer * varyingBuffer,
            ComputeContext * computeContext,
            DrawContext * drawContext) :

            _refiner(refiner),
            _vertexBuffer(vertexBuffer),
            _varyingBuffer(varyingBuffer),
            _computeContext(computeContext),
            _computeController(computeController),
            _drawContext(drawContext) { }

    virtual ~Mesh() {
        delete _refiner;
        delete _patchTables;
        delete _vertexBuffer;
        delete _varyingBuffer;
        delete _computeContext;
        delete _drawContext;
    }

    virtual int GetNumVertices() const {
        assert(_refiner);
        return MeshInterface<DRAW_CONTEXT>::getNumVertices(*_refiner);
    }

    virtual void UpdateVertexBuffer(float const *vertexData, int startVertex, int numVerts) {
        _vertexBuffer->UpdateData(vertexData, startVertex, numVerts);
    }

    virtual void UpdateVaryingBuffer(float const *varyingData, int startVertex, int numVerts) {
        _varyingBuffer->UpdateData(varyingData, startVertex, numVerts);
    }

    virtual void Refine() {
        _computeController->Compute(_computeContext, _kernelBatches, _vertexBuffer, _varyingBuffer);
    }

    virtual void Refine(VertexBufferDescriptor const *vertexDesc, VertexBufferDescriptor const *varyingDesc) {
        _computeController->Refine(_computeContext, _kernelBatches, _vertexBuffer, _varyingBuffer, vertexDesc, varyingDesc);
    }

    virtual void Synchronize() {
        _computeController->Synchronize();
    }

    virtual VertexBufferBinding BindVertexBuffer() {
        return VertexBufferBinding(0);
    }

    virtual VertexBufferBinding BindVaryingBuffer() {
        return VertexBufferBinding(0);
    }

    virtual DrawContext * GetDrawContext() {
        return _drawContext;
    }

    virtual void SetFVarDataChannel(int fvarWidth, std::vector<float> const & fvarData) {
        if (_patchTables and _drawContext and fvarWidth and (not fvarData.empty())) {
            _drawContext->SetFVarDataTexture(*_patchTables, fvarWidth, fvarData);
        }
    }

private:

    void initializeComputeContext(int numVertexElements,
        int numVaryingElements ) {

        assert(_refiner);

        Far::StencilTablesFactory::Options options;
        options.generateOffsets=true;
        options.generateAllLevels=_refiner->IsUniform() ? false : true;

        Far::StencilTables const * vertexStencils=0, * varyingStencils=0;

        if (numVertexElements>0) {

            vertexStencils = Far::StencilTablesFactory::Create(*_refiner, options);

            _kernelBatches.push_back(Far::StencilTablesFactory::Create(*vertexStencils));
        }

        if (numVaryingElements>0) {

            options.interpolationMode = Far::StencilTablesFactory::INTERPOLATE_VARYING;

            varyingStencils = Far::StencilTablesFactory::Create(*_refiner, options);
        }

        _computeContext = ComputeContext::Create(vertexStencils, varyingStencils);

        delete vertexStencils;
        delete varyingStencils;
    }

    void initializeDrawContext(int numElements, MeshBitset bits) {

        assert(_refiner and _vertexBuffer);

        Far::PatchTablesFactory::Options options;
        options.generateFVarTables = bits.test(MeshFVarData);

        _patchTables = Far::PatchTablesFactory::Create(*_refiner);

        _drawContext = DrawContext::Create(
            _patchTables, numElements, bits.test(MeshFVarData));

        _drawContext->UpdateVertexTexture(_vertexBuffer);
    }

    int initializeVertexBuffers(int numVertexElements,
        int numVaryingElements, MeshBitset bits) {

        int numVertices = MeshInterface<DRAW_CONTEXT>::getNumVertices(*_refiner);

        int numElements = numVertexElements +
            (bits.test(MeshInterleaveVarying) ? numVaryingElements : 0);

        if (numVertexElements) {

            _vertexBuffer = VertexBuffer::Create(numElements, numVertices);
        }

        if (numVaryingElements>0 and (not bits.test(MeshInterleaveVarying))) {
            _varyingBuffer = VertexBuffer::Create(numVaryingElements, numVertices);
        }
        return numElements;
   }

    Far::TopologyRefiner * _refiner;
    Far::PatchTables * _patchTables;
    Far::KernelBatchVector _kernelBatches;

    VertexBuffer * _vertexBuffer,
                 * _varyingBuffer;

    ComputeContext    * _computeContext;
    ComputeController * _computeController;

    DrawContext *_drawContext;
};

} // end namespace Osd

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OSD_MESH_H
