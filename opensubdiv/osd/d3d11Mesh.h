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

#ifndef OSD_D3D11MESH_H
#define OSD_D3D11MESH_H

#include "../version.h"

#include "../osd/mesh.h"
#include "../osd/d3d11ComputeController.h"
#include "../osd/d3d11DrawContext.h"
#include "../osd/d3d11VertexBuffer.h"
#include "../far/endCapLegacyGregoryPatchFactory.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

typedef MeshInterface<D3D11DrawContext> D3D11MeshInterface;

template <class VERTEX_BUFFER, class COMPUTE_CONTROLLER>
class Mesh<VERTEX_BUFFER, COMPUTE_CONTROLLER, D3D11DrawContext> : public D3D11MeshInterface {
public:
    typedef VERTEX_BUFFER VertexBuffer;
    typedef COMPUTE_CONTROLLER ComputeController;
    typedef typename ComputeController::ComputeContext ComputeContext;
    typedef D3D11DrawContext DrawContext;
    typedef typename DrawContext::VertexBufferBinding VertexBufferBinding;

    Mesh(ComputeController * computeController,
            Far::TopologyRefiner * refiner,
            int numVertexElements,
            int numVaryingElements,
            int level,
            MeshBitset bits,
            ID3D11DeviceContext *d3d11DeviceContext) :

            _refiner(refiner),
            _patchTables(0),
            _vertexBuffer(0),
            _varyingBuffer(0),
            _computeContext(0),
            _computeController(computeController),
            _drawContext(0),
            _numVertices(0),
            _d3d11DeviceContext(d3d11DeviceContext)
    {
        D3D11MeshInterface::refineMesh(*_refiner, level, bits.test(MeshAdaptive), bits.test(MeshUseSingleCreasePatch));

        initializeContext(numVertexElements,
                          numVaryingElements,
                          numVertexElements, /* interleaved stride */
                          level, bits);

        initializeVertexBuffers(_numVertices,
                                numVertexElements,
                                numVaryingElements);

        // for classic gregory patch -- will retire
        _drawContext->UpdateVertexTexture(_vertexBuffer, _d3d11DeviceContext);
    }

    Mesh(ComputeController * computeController,
            Far::TopologyRefiner * refiner,
            Far::PatchTables * patchTables,
            VertexBuffer * vertexBuffer,
            VertexBuffer * varyingBuffer,
            ComputeContext * computeContext,
            DrawContext * drawContext,
            ID3D11DeviceContext *d3d11DeviceContext) :

            _refiner(refiner),
            _patchTables(patchTables),
            _vertexBuffer(vertexBuffer),
            _varyingBuffer(varyingBuffer),
            _computeContext(computeContext),
            _computeController(computeController),
            _drawContext(drawContext),
            _numVertices(0),
            _d3d11DeviceContext(d3d11DeviceContext)
    {
        _drawContext->UpdateVertexTexture(_vertexBuffer, _d3d11DeviceContext);
    }

    virtual ~Mesh() {
        delete _refiner;
        delete _patchTables;
        delete _vertexBuffer;
        delete _varyingBuffer;
        delete _computeContext;
        delete _drawContext;
    }

    virtual int GetNumVertices() const { return _numVertices; }

    virtual void UpdateVertexBuffer(float const *vertexData, int startVertex, int numVerts) {
        _vertexBuffer->UpdateData(vertexData, startVertex, numVerts, _d3d11DeviceContext);
    }
    virtual void UpdateVaryingBuffer(float const *varyingData, int startVertex, int numVerts) {
        _varyingBuffer->UpdateData(varyingData, startVertex, numVerts, _d3d11DeviceContext);
    }
    virtual void Refine() {
        _computeController->Compute(_computeContext, _vertexBuffer, _varyingBuffer);
    }
    virtual void Refine(VertexBufferDescriptor const *vertexDesc,
                        VertexBufferDescriptor const *varyingDesc,
                        bool interleaved) {
        _computeController->Compute(_computeContext,
                                    _vertexBuffer,
                                    (interleaved ? _vertexBuffer : _varyingBuffer),
                                    vertexDesc, varyingDesc);
    }
    virtual void Synchronize() {
        _computeController->Synchronize();
    }
    virtual VertexBufferBinding BindVertexBuffer() {
        return _vertexBuffer->BindD3D11Buffer(_d3d11DeviceContext);
    }
    virtual VertexBufferBinding BindVaryingBuffer() {
        return _varyingBuffer->BindD3D11Buffer(_d3d11DeviceContext);
    }
    virtual DrawContext * GetDrawContext() {
        return _drawContext;
    }
    virtual VertexBuffer * GetVertexBuffer() {
        return _vertexBuffer;
    }
    virtual VertexBuffer * GetVaryingBuffer() {
        return _varyingBuffer;
    }

    virtual Far::TopologyRefiner const * GetTopologyRefiner() const {
        return _refiner;
    }

    virtual void SetFVarDataChannel(int fvarWidth,
                                    std::vector<float> const & fvarData) {

        if (_patchTables and _drawContext and fvarWidth and (not fvarData.empty())) {
            _drawContext->SetFVarDataTexture(*_patchTables,
                _d3d11DeviceContext, fvarWidth, fvarData);
        }
    }

private:

    void initializeContext(int numVertexElements,
                           int numVaryingElements,
                           int numElements, int level, MeshBitset bits) {

        assert(_refiner);

        Far::StencilTablesFactory::Options options;
        options.generateOffsets=true;
        options.generateIntermediateLevels=_refiner->IsUniform() ? false : true;

        Far::StencilTables const * vertexStencils=0, * varyingStencils=0;

        if (numVertexElements>0) {

            vertexStencils = Far::StencilTablesFactory::Create(*_refiner, options);
        }

        if (numVaryingElements>0) {

            options.interpolationMode = Far::StencilTablesFactory::INTERPOLATE_VARYING;

            varyingStencils = Far::StencilTablesFactory::Create(*_refiner, options);
        }

        Far::PatchTablesFactory::Options poptions(level);
        poptions.generateFVarTables = bits.test(MeshFVarData);
        poptions.useSingleCreasePatch = bits.test(MeshUseSingleCreasePatch);

        // TODO: add gregory basis shader for DX11
		Far::EndCapLegacyGregoryPatchFactory *gregoryPatchFactory = new Far::EndCapLegacyGregoryPatchFactory(*_refiner);
		_patchTables = Far::PatchTablesFactoryT<Far::EndCapLegacyGregoryPatchFactory>::Create(*_refiner, poptions, gregoryPatchFactory);
        gregoryPatchFactory->AddGregoryPatchTables(_patchTables);

        _drawContext = DrawContext::Create(_patchTables,
                                           _d3d11DeviceContext,
                                           numElements);

        _computeContext = ComputeContext::Create(vertexStencils,
                                                 varyingStencils);

        // numvertices = coarse verts + refined verts + gregory basis verts
        _numVertices = vertexStencils->GetNumControlVertices()
            + vertexStencils->GetNumStencils();

        delete gregoryPatchFactory;
        delete vertexStencils;
        delete varyingStencils;
    }

    void initializeVertexBuffers(int numVertices,
                                 int numVertexElements,
                                 int numVaryingElements) {

        ID3D11Device * pd3d11Device;
        _d3d11DeviceContext->GetDevice(&pd3d11Device);

        if (numVertexElements) {
            _vertexBuffer =
                VertexBuffer::Create(numVertexElements, numVertices, pd3d11Device);
        }

        if (numVaryingElements) {
            _varyingBuffer =
                VertexBuffer::Create(numVaryingElements, numVertices, pd3d11Device);
        }
    }

    Far::TopologyRefiner * _refiner;
    Far::PatchTables * _patchTables;

    VertexBuffer *_vertexBuffer;
    VertexBuffer *_varyingBuffer;

    ComputeContext *_computeContext;
    ComputeController *_computeController;
    DrawContext *_drawContext;
    int _numVertices;

    ID3D11DeviceContext *_d3d11DeviceContext;
};

template <>
class Mesh<D3D11VertexBuffer, D3D11ComputeController, D3D11DrawContext> : public D3D11MeshInterface {
public:
    typedef D3D11VertexBuffer VertexBuffer;
    typedef D3D11ComputeController ComputeController;
    typedef ComputeController::ComputeContext ComputeContext;
    typedef D3D11DrawContext DrawContext;
    typedef DrawContext::VertexBufferBinding VertexBufferBinding;

    Mesh(ComputeController * computeController,
            Far::TopologyRefiner * refiner,
            int numVertexElements,
            int numVaryingElements,
            int level,
            MeshBitset bits,
            ID3D11DeviceContext *d3d11DeviceContext) :

            _refiner(refiner),
            _patchTables(0),
            _vertexBuffer(0),
            _varyingBuffer(0),
            _computeContext(0),
            _computeController(computeController),
            _drawContext(0),
            _numVertices(0),
            _d3d11DeviceContext(d3d11DeviceContext)
    {
        D3D11MeshInterface::refineMesh(*_refiner, level,
                                       bits.test(MeshAdaptive),
                                       bits.test(MeshUseSingleCreasePatch));

        initializeContext(numVertexElements,
                          numVaryingElements,
                          numVertexElements, /* interleaved stride */
                          level, bits);

        initializeVertexBuffers(_numVertices,
                                numVertexElements,
                                numVaryingElements);

        // for classic gregory patch -- will retire
        _drawContext->UpdateVertexTexture(_vertexBuffer, _d3d11DeviceContext);
    }

    Mesh(ComputeController * computeController,
            Far::TopologyRefiner * refiner,
            Far::PatchTables * patchTables,
            VertexBuffer * vertexBuffer,
            VertexBuffer * varyingBuffer,
            ComputeContext * computeContext,
            DrawContext * drawContext,
            ID3D11DeviceContext *d3d11DeviceContext) :

            _refiner(refiner),
            _patchTables(patchTables),
            _vertexBuffer(vertexBuffer),
            _varyingBuffer(varyingBuffer),
            _computeContext(computeContext),
            _computeController(computeController),
            _drawContext(drawContext),
            _numVertices(0),
            _d3d11DeviceContext(d3d11DeviceContext)
    {
        _drawContext->UpdateVertexTexture(_vertexBuffer, _d3d11DeviceContext);
    }

    virtual ~Mesh() {
        delete _refiner;
        delete _patchTables;
        delete _vertexBuffer;
        delete _varyingBuffer;
        delete _computeContext;
        delete _drawContext;
    }

    virtual int GetNumVertices() const { return _numVertices; }

    virtual void UpdateVertexBuffer(float const *vertexData, int startVertex, int numVerts) {
        _vertexBuffer->UpdateData(vertexData, startVertex, numVerts, _d3d11DeviceContext);
    }
    virtual void UpdateVaryingBuffer(float const *varyingData, int startVertex, int numVerts) {
        _varyingBuffer->UpdateData(varyingData, startVertex, numVerts, _d3d11DeviceContext);
    }
    virtual void Refine() {
        _computeController->Compute(_computeContext, _vertexBuffer, _varyingBuffer);
    }
    virtual void Refine(VertexBufferDescriptor const *vertexDesc,
                        VertexBufferDescriptor const *varyingDesc,
                        bool interleaved) {
        _computeController->Compute(_computeContext,
                                    _vertexBuffer, (interleaved ? _vertexBuffer : _varyingBuffer),
                                    vertexDesc, varyingDesc);
    }
    virtual void Synchronize() {
        _computeController->Synchronize();
    }
    virtual VertexBufferBinding BindVertexBuffer() {
        return _vertexBuffer->BindD3D11Buffer(_d3d11DeviceContext);
    }
    virtual VertexBufferBinding BindVaryingBuffer() {
        return _varyingBuffer->BindD3D11Buffer(_d3d11DeviceContext);
    }
    virtual DrawContext * GetDrawContext() {
        return _drawContext;
    }
    virtual VertexBuffer * GetVertexBuffer() {
        return _vertexBuffer;
    }
    virtual VertexBuffer * GetVaryingBuffer() {
        return _varyingBuffer;
    }

    virtual Far::TopologyRefiner const * GetTopologyRefiner() const {
        return _refiner;
    }

    virtual void SetFVarDataChannel(int fvarWidth,
                                    std::vector<float> const & fvarData) {

        if (_patchTables and _drawContext and fvarWidth and (not fvarData.empty())) {
            _drawContext->SetFVarDataTexture(*_patchTables,
                _d3d11DeviceContext, fvarWidth, fvarData);
        }
    }

private:


    void initializeContext(int numVertexElements,
                           int numVaryingElements,
                           int numElements, int level, MeshBitset bits) {

        assert(_refiner);

        Far::StencilTablesFactory::Options options;
        options.generateOffsets=true;
        options.generateIntermediateLevels=_refiner->IsUniform() ? false : true;

        Far::StencilTables const * vertexStencils=0, * varyingStencils=0;

        if (numVertexElements>0) {

            vertexStencils = Far::StencilTablesFactory::Create(*_refiner, options);
        }

        if (numVaryingElements>0) {

            options.interpolationMode = Far::StencilTablesFactory::INTERPOLATE_VARYING;

            varyingStencils = Far::StencilTablesFactory::Create(*_refiner, options);
        }

        Far::PatchTablesFactory::Options poptions(level);
        poptions.generateFVarTables = bits.test(MeshFVarData);
        poptions.useSingleCreasePatch = bits.test(MeshUseSingleCreasePatch);

		Far::EndCapLegacyGregoryPatchFactory *gregoryPatchFactory = new Far::EndCapLegacyGregoryPatchFactory(*_refiner);
		_patchTables = Far::PatchTablesFactoryT<Far::EndCapLegacyGregoryPatchFactory>::Create(*_refiner, poptions, gregoryPatchFactory);
        gregoryPatchFactory->AddGregoryPatchTables(_patchTables);

        _drawContext = DrawContext::Create(_patchTables,
                                           _d3d11DeviceContext,
                                           numElements);

        _computeContext =
            ComputeContext::Create(_d3d11DeviceContext, vertexStencils, varyingStencils);

        // numvertices = coarse verts + refined verts + gregory basis verts
        if (vertexStencils) {
            _numVertices = vertexStencils->GetNumControlVertices()
                + vertexStencils->GetNumStencils();
        } else {
            // if no stencils (torus), ask refiner.
            _numVertices = _refiner->GetNumVertices(0);
        }

        delete vertexStencils;
        delete varyingStencils;
    }

    void initializeVertexBuffers(int numVertices,
                                 int numVertexElements,
                                 int numVaryingElements) {

        ID3D11Device * pd3d11Device;
        _d3d11DeviceContext->GetDevice(&pd3d11Device);

        if (numVertexElements) {
            _vertexBuffer =
                VertexBuffer::Create(numVertexElements, numVertices, pd3d11Device);
        }

        if (numVaryingElements) {
            _varyingBuffer =
                VertexBuffer::Create(numVaryingElements, numVertices, pd3d11Device);
        }
    }

    Far::TopologyRefiner * _refiner;
    Far::PatchTables * _patchTables;

    VertexBuffer *_vertexBuffer;
    VertexBuffer *_varyingBuffer;

    ComputeContext *_computeContext;
    ComputeController *_computeController;
    DrawContext *_drawContext;
    int _numVertices;

    ID3D11DeviceContext *_d3d11DeviceContext;
};


} // end namespace Osd

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OSD_D3D11MESH_H
