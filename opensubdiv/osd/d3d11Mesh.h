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

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

typedef OsdMeshInterface<OsdD3D11DrawContext> OsdD3D11MeshInterface;

template <class VERTEX_BUFFER, class COMPUTE_CONTROLLER>
class OsdMesh<VERTEX_BUFFER, COMPUTE_CONTROLLER, OsdD3D11DrawContext> : public OsdD3D11MeshInterface {
public:
    typedef VERTEX_BUFFER VertexBuffer;
    typedef COMPUTE_CONTROLLER ComputeController;
    typedef typename ComputeController::ComputeContext ComputeContext; 
    typedef OsdD3D11DrawContext DrawContext; 
    typedef typename DrawContext::VertexBufferBinding VertexBufferBinding;

    OsdMesh(ComputeController * computeController,
            HbrMesh<OsdVertex> * hmesh,
            int numVertexElements,
            int numVaryingElements,
            int level,
            OsdMeshBitset bits,
            ID3D11DeviceContext *d3d11DeviceContext) :

            _farMesh(0),
            _vertexBuffer(0),
            _varyingBuffer(0),
            _computeContext(0),
            _computeController(computeController),
            _drawContext(0),
            _pd3d11DeviceContext(d3d11DeviceContext)
    {
        FarMeshFactory<OsdVertex> meshFactory(hmesh, level, bits.test(MeshAdaptive));
        _farMesh = meshFactory.Create(bits.test(MeshFVarData));

        _initialize(numVertexElements, numVaryingElements, bits);
    }

    OsdMesh(ComputeController * computeController,
            FarMesh<OsdVertex> * fmesh,
            int numVertexElements,
            int numVaryingElements,
            OsdMeshBitset bits,
            ID3D11DeviceContext *d3d11DeviceContext) :

            _farMesh(fmesh),
            _vertexBuffer(0),
            _varyingBuffer(0),
            _computeContext(0),
            _computeController(computeController),
            _drawContext(0),
            _pd3d11DeviceContext(d3d11DeviceContext)
    {
        _initialize(numVertexElements, numVaryingElements, bits);
    }

    OsdMesh(ComputeController * computeController,
            FarMesh<OsdVertex> * fmesh,
            VertexBuffer * vertexBuffer,
            VertexBuffer * varyingBuffer,
            ComputeContext * computeContext,
            DrawContext * drawContext,
            ID3D11DeviceContext *d3d11DeviceContext) :

            _farMesh(fmesh),
            _vertexBuffer(vertexBuffer),
            _varyingBuffer(varyingBuffer),
            _computeContext(computeContext),
            _computeController(computeController),
            _drawContext(drawContext),
            _pd3d11DeviceContext(d3d11DeviceContext)
    {
        _drawContext->UpdateVertexTexture(_vertexBuffer, _pd3d11DeviceContext);
    }

    virtual ~OsdMesh() {
        delete _farMesh;
        delete _vertexBuffer;
        delete _varyingBuffer;
        delete _computeContext;
        delete _drawContext;
    }

    virtual int GetNumVertices() const { return _farMesh->GetNumVertices(); }

    virtual void UpdateVertexBuffer(float const *vertexData, int startVertex, int numVerts) {
        _vertexBuffer->UpdateData(vertexData, startVertex, numVerts, _pd3d11DeviceContext);
    }
    virtual void UpdateVaryingBuffer(float const *varyingData, int startVertex, int numVerts) {
        _varyingBuffer->UpdateData(varyingData, startVertex, numVerts, _pd3d11DeviceContext);
    }
    virtual void Refine() {
        _computeController->Refine(_computeContext, _farMesh->GetKernelBatches(), _vertexBuffer, _varyingBuffer);
    }
    virtual void Refine(OsdVertexBufferDescriptor const *vertexDesc,
                        OsdVertexBufferDescriptor const *varyingDesc,
                        bool interleaved) {
        _computeController->Refine(_computeContext, _farMesh->GetKernelBatches(),
                                   _vertexBuffer, (interleaved ? _vertexBuffer : _varyingBuffer),
                                    vertexDesc, varyingDesc);
    }
    virtual void Synchronize() {
        _computeController->Synchronize();
    }
    virtual VertexBufferBinding BindVertexBuffer() {
        return _vertexBuffer->BindD3D11Buffer(_pd3d11DeviceContext);
    }
    virtual VertexBufferBinding BindVaryingBuffer() {
        return _varyingBuffer->BindD3D11Buffer(_pd3d11DeviceContext);
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
    virtual FarMesh<OsdVertex> const * GetFarMesh() const {
        return _farMesh;
    }

private:

    void _initialize( int numVertexElements,
                      int numVaryingElements,
                      OsdMeshBitset bits)
    {
        ID3D11Device * pd3d11Device;
        _pd3d11DeviceContext->GetDevice(&pd3d11Device);

        int numVertices = _farMesh->GetNumVertices();
        if (numVertexElements)
            _vertexBuffer = VertexBuffer::Create(numVertexElements, numVertices, pd3d11Device);
        if (numVaryingElements)
            _varyingBuffer = VertexBuffer::Create(numVaryingElements, numVertices, pd3d11Device);
        _computeContext = ComputeContext::Create(_farMesh->GetSubdivisionTables(), _farMesh->GetVertexEditTables());
        _drawContext = DrawContext::Create(_farMesh->GetPatchTables(),
                                           _pd3d11DeviceContext,
                                           numVertexElements,
                                           bits.test(MeshFVarData));
        _drawContext->UpdateVertexTexture(_vertexBuffer, _pd3d11DeviceContext);
    }

    FarMesh<OsdVertex> *_farMesh;
    VertexBuffer *_vertexBuffer;
    VertexBuffer *_varyingBuffer;
    ComputeContext *_computeContext;
    ComputeController *_computeController;
    DrawContext *_drawContext;

    ID3D11DeviceContext *_pd3d11DeviceContext;
};

template <>
class OsdMesh<OsdD3D11VertexBuffer, OsdD3D11ComputeController, OsdD3D11DrawContext> : public OsdD3D11MeshInterface {
public:
    typedef OsdD3D11VertexBuffer VertexBuffer;
    typedef OsdD3D11ComputeController ComputeController;
    typedef ComputeController::ComputeContext ComputeContext; 
    typedef OsdD3D11DrawContext DrawContext; 
    typedef DrawContext::VertexBufferBinding VertexBufferBinding;

    OsdMesh(ComputeController * computeController,
            HbrMesh<OsdVertex> * hmesh,
            int numVertexElements,
            int numVaryingElements,
            int level,
            OsdMeshBitset bits,
            ID3D11DeviceContext *d3d11DeviceContext) :

            _farMesh(0),
            _vertexBuffer(0),
            _varyingBuffer(0),
            _computeContext(0),
            _computeController(computeController),
            _drawContext(0),
            _pd3d11DeviceContext(d3d11DeviceContext)
    {
        FarMeshFactory<OsdVertex> meshFactory(hmesh, level, bits.test(MeshAdaptive));
        _farMesh = meshFactory.Create(bits.test(MeshFVarData));

        _initialize(numVertexElements, numVaryingElements, bits);
    }

    OsdMesh(ComputeController * computeController,
            FarMesh<OsdVertex> * fmesh,
            int numVertexElements,
            int numVaryingElements,
            OsdMeshBitset bits,
            ID3D11DeviceContext *d3d11DeviceContext) :

            _farMesh(fmesh),
            _vertexBuffer(0),
            _varyingBuffer(0),
            _computeContext(0),
            _computeController(computeController),
            _drawContext(0),
            _pd3d11DeviceContext(d3d11DeviceContext)
    {
        _initialize(numVertexElements, numVaryingElements, bits);
    }

    OsdMesh(ComputeController * computeController,
            FarMesh<OsdVertex> * fmesh,
            VertexBuffer * vertexBuffer,
            VertexBuffer * varyingBuffer,
            ComputeContext * computeContext,
            DrawContext * drawContext,
            ID3D11DeviceContext *d3d11DeviceContext) :

            _farMesh(fmesh),
            _vertexBuffer(vertexBuffer),
            _varyingBuffer(varyingBuffer),
            _computeContext(computeContext),
            _computeController(computeController),
            _drawContext(drawContext),
            _pd3d11DeviceContext(d3d11DeviceContext)
    {
        _drawContext->UpdateVertexTexture(_vertexBuffer, _pd3d11DeviceContext);
    }

    virtual ~OsdMesh() {
        delete _farMesh;
        delete _vertexBuffer;
        delete _varyingBuffer;
        delete _computeContext;
        delete _drawContext;
    }

    virtual int GetNumVertices() const { return _farMesh->GetNumVertices(); }

    virtual void UpdateVertexBuffer(float const *vertexData, int startVertex, int numVerts) {
        _vertexBuffer->UpdateData(vertexData, startVertex, numVerts, _pd3d11DeviceContext);
    }
    virtual void UpdateVaryingBuffer(float const *varyingData, int startVertex, int numVerts) {
        _varyingBuffer->UpdateData(varyingData, startVertex, numVerts, _pd3d11DeviceContext);
    }
    virtual void Refine() {
        _computeController->Refine(_computeContext, _farMesh->GetKernelBatches(), _vertexBuffer, _varyingBuffer);
    }
    virtual void Refine(OsdVertexBufferDescriptor const *vertexDesc,
                        OsdVertexBufferDescriptor const *varyingDesc,
                        bool interleaved) {
        _computeController->Refine(_computeContext, _farMesh->GetKernelBatches(),
                                   _vertexBuffer, (interleaved ? _vertexBuffer : _varyingBuffer),
                                    vertexDesc, varyingDesc);
    }
    virtual void Synchronize() {
        _computeController->Synchronize();
    }
    virtual VertexBufferBinding BindVertexBuffer() {
        return _vertexBuffer->BindD3D11Buffer(_pd3d11DeviceContext);
    }
    virtual VertexBufferBinding BindVaryingBuffer() {
        return _varyingBuffer->BindD3D11Buffer(_pd3d11DeviceContext);
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
    virtual FarMesh<OsdVertex> const * GetFarMesh() const {
        return _farMesh;
    }

private:

    void _initialize( int numVertexElements,
                      int numVaryingElements,
                      OsdMeshBitset bits)
    {
        ID3D11Device * pd3d11Device;
        _pd3d11DeviceContext->GetDevice(&pd3d11Device);

        int numVertices = _farMesh->GetNumVertices();
        if (numVertexElements)
            _vertexBuffer = VertexBuffer::Create(numVertexElements, numVertices, pd3d11Device);
        if (numVaryingElements)
            _varyingBuffer = VertexBuffer::Create(numVaryingElements, numVertices, pd3d11Device);
        _computeContext = ComputeContext::Create(_farMesh->GetSubdivisionTables(),
                                                 _farMesh->GetVertexEditTables(),
                                                 _pd3d11DeviceContext);
        _drawContext = DrawContext::Create(_farMesh->GetPatchTables(),
                                           _pd3d11DeviceContext,
                                           numVertexElements,
                                           bits.test(MeshFVarData));
        _drawContext->UpdateVertexTexture(_vertexBuffer, _pd3d11DeviceContext);
    }

    FarMesh<OsdVertex> *_farMesh;
    VertexBuffer *_vertexBuffer;
    VertexBuffer *_varyingBuffer;
    ComputeContext *_computeContext;
    ComputeController *_computeController;
    DrawContext *_drawContext;

    ID3D11DeviceContext *_pd3d11DeviceContext;
};

#ifdef OPENSUBDIV_HAS_OPENCL

#include "../osd/opencl.h"

class OsdCLComputeController;

template <class VERTEX_BUFFER>
class OsdMesh<VERTEX_BUFFER, OsdCLComputeController, OsdD3D11DrawContext> : public OsdD3D11MeshInterface {
public:
    typedef VERTEX_BUFFER VertexBuffer;
    typedef OsdCLComputeController ComputeController; 
    typedef typename ComputeController::ComputeContext ComputeContext; 
    typedef OsdD3D11DrawContext DrawContext; 
    typedef typename DrawContext::VertexBufferBinding VertexBufferBinding; 

    OsdMesh(ComputeController * computeController,
            HbrMesh<OsdVertex> * hmesh,
            int numVertexElements,
            int numVaryingElements,
            int level,
            OsdMeshBitset bits,
            cl_context clContext,
            cl_command_queue clQueue,
            ID3D11DeviceContext *d3d11DeviceContext) :

            _farMesh(0),
            _vertexBuffer(0),
            _varyingBuffer(0),
            _computeContext(0),
            _computeController(computeController),
            _drawContext(0),
            _clContext(clContext),
            _clQueue(clQueue),
            _pd3d11DeviceContext(d3d11DeviceContext)
    {
        FarMeshFactory<OsdVertex> meshFactory(hmesh, level, bits.test(MeshAdaptive));
        _farMesh = meshFactory.Create(bits.test(MeshFVarData));

        _initialize(numVertexElements, numVaryingElements, bits);
    }

    OsdMesh(ComputeController * computeController,
            FarMesh<OsdVertex> * fmesh,
            int numVertexElements,
            int numVaryingElements,
            OsdMeshBitset bits,
            cl_context clContext,
            cl_command_queue clQueue,
            ID3D11DeviceContext *d3d11DeviceContext) :

            _farMesh(fmesh),
            _vertexBuffer(0),
            _varyingBuffer(0),
            _computeContext(0),
            _computeController(computeController),
            _drawContext(0),
            _clContext(clContext),
            _clQueue(clQueue),
            _pd3d11DeviceContext(d3d11DeviceContext)
    {
        _initialize(numVertexElements, numVaryingElements, bits);
    }

    OsdMesh(ComputeController * computeController,
            FarMesh<OsdVertex> * fmesh,
            VertexBuffer * vertexBuffer,
            VertexBuffer * varyingBuffer,
            ComputeContext * computeContext,
            DrawContext * drawContext,
            cl_context clContext,
            cl_command_queue clQueue,
            ID3D11DeviceContext *d3d11DeviceContext) :

            _farMesh(fmesh),
            _vertexBuffer(vertexBuffer),
            _varyingBuffer(varyingBuffer),
            _computeContext(computeContext),
            _computeController(computeController),
            _drawContext(drawContext),
            _clContext(clContext),
            _clQueue(clQueue),
            _pd3d11DeviceContext(d3d11DeviceContext)
    {
        _drawContext->UpdateVertexTexture(_vertexBuffer, _pd3d11DeviceContext);
    }

    virtual ~OsdMesh() {
        delete _farMesh;
        delete _vertexBuffer;
        delete _varyingBuffer;
        delete _computeContext;
        delete _drawContext;
    }

    virtual int GetNumVertices() const { return _farMesh->GetNumVertices(); }

    virtual void UpdateVertexBuffer(float const *vertexData, int startVertex, int numVerts) {
        ID3D11Device * pd3d11Device;
        _pd3d11DeviceContext->GetDevice(&pd3d11Device);
        _vertexBuffer->UpdateData(vertexData, startVertex, numVerts, _clQueue);
    }
    virtual void UpdateVaryingBuffer(float const *varyingData, int startVertex, int numVerts) {
        ID3D11Device * pd3d11Device;
        _pd3d11DeviceContext->GetDevice(&pd3d11Device);
        _varyingBuffer->UpdateData(varyingData, startVertex, numVerts, _clQueue);
    }
    virtual void Refine() {
        _computeController->Refine(_computeContext, _farMesh->GetKernelBatches(), _vertexBuffer, _varyingBuffer);
    }
    virtual void Synchronize() {
        _computeController->Synchronize();
    }
    virtual VertexBufferBinding BindVertexBuffer() {
        return _vertexBuffer->BindD3D11Buffer(_pd3d11DeviceContext);
    }
    virtual VertexBufferBinding BindVaryingBuffer() {
        return _varyingBuffer->BindD3D11Buffer(_pd3d11DeviceContext);
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
    virtual FarMesh<OsdVertex> const * GetFarMesh() const {
        return _farMesh;
    }

private:

    void _initialize( int numVertexElements,
                      int numVaryingElements,
                      OsdMeshBitset bits)
    {
        ID3D11Device * pd3d11Device;
        _pd3d11DeviceContext->GetDevice(&pd3d11Device);

        int numVertices = _farMesh->GetNumVertices();
        _vertexBuffer = typename VertexBuffer::Create(numVertexElements, numVertices, _clContext, pd3d11Device);
        if (numVaryingElements)
            _varyingBuffer = typename VertexBuffer::Create(numVaryingElements, numVertices, _clContext, pd3d11Device);
        _computeContext = ComputeContext::Create(_farMesh->GetSubdivisionTables(),
                                                 _farMesh->GetVertexEditTables(), _clContext);
        _drawContext = DrawContext::Create(_farMesh->GetPatchTables(),
                                           _pd3d11DeviceContext,
                                           numVertexElements,
                                           bits.test(MeshFVarData));
        _drawContext->UpdateVertexTexture(_vertexBuffer, _pd3d11DeviceContext);
    }

    FarMesh<OsdVertex> *_farMesh;
    VertexBuffer *_vertexBuffer;
    VertexBuffer *_varyingBuffer;
    ComputeContext *_computeContext;
    ComputeController *_computeController;
    DrawContext *_drawContext;

    cl_context _clContext;
    cl_command_queue _clQueue;

    ID3D11DeviceContext *_pd3d11DeviceContext;
};
#endif

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OSD_D3D11MESH_H
