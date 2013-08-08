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
#ifndef OSD_GL_MESH_H
#define OSD_GL_MESH_H

#include "../version.h"

#include "../osd/mesh.h"
#include "../osd/glDrawContext.h"

#ifdef OPENSUBDIV_HAS_OPENCL
#if defined(__APPLE__)
    #include <OpenCL/opencl.h>
#else
    #include <CL/opencl.h>
#endif
#include "../osd/clComputeController.h"
#endif

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

typedef OsdMeshInterface<OsdGLDrawContext> OsdGLMeshInterface;

template <class VERTEX_BUFFER, class COMPUTE_CONTROLLER>
class OsdMesh<VERTEX_BUFFER, COMPUTE_CONTROLLER, OsdGLDrawContext> : public OsdGLMeshInterface {
public:
    typedef VERTEX_BUFFER VertexBuffer;
    typedef COMPUTE_CONTROLLER ComputeController;
    typedef typename ComputeController::ComputeContext ComputeContext; 
    typedef OsdGLDrawContext DrawContext; 
    typedef typename DrawContext::VertexBufferBinding VertexBufferBinding;

    OsdMesh(ComputeController * computeController,
            HbrMesh<OsdVertex> * hmesh,
            int numVertexElements,
            int numVaryingElements,
            int level,
            OsdMeshBitset bits) :

            _farMesh(0),
            _vertexBuffer(0),
            _varyingBuffer(0),
            _computeContext(0),
            _computeController(computeController),
            _drawContext(0)
    {
        FarMeshFactory<OsdVertex> meshFactory(hmesh, level, bits.test(MeshAdaptive));
        _farMesh = meshFactory.Create(bits.test(MeshFVarData));

        _initialize(numVertexElements, numVaryingElements, level, bits);
    }

    OsdMesh(ComputeController * computeController,
            FarMesh<OsdVertex> * fmesh,
            int numVertexElements,
            int numVaryingElements,
            int level,
            OsdMeshBitset bits) :

            _farMesh(fmesh),
            _vertexBuffer(0),
            _varyingBuffer(0),
            _computeContext(0),
            _computeController(computeController),
            _drawContext(0)
    {
        _initialize(numVertexElements, numVaryingElements, level, bits);
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
        _vertexBuffer->UpdateData(vertexData, startVertex, numVerts);
    }
    virtual void UpdateVaryingBuffer(float const *varyingData, int startVertex, int numVerts) {
        _varyingBuffer->UpdateData(varyingData, startVertex, numVerts);
    }
    virtual void Refine() {
        _computeController->Refine(_computeContext, _farMesh->GetKernelBatches(), _vertexBuffer, _varyingBuffer);
    }
    virtual void Synchronize() {
        _computeController->Synchronize();
    }
    virtual VertexBufferBinding BindVertexBuffer() {
        return _vertexBuffer->BindVBO();
    }
    virtual VertexBufferBinding BindVaryingBuffer() {
        return _varyingBuffer->BindVBO();
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
                      int level,
                      OsdMeshBitset bits) 
    {
        int numVertices = _farMesh->GetNumVertices();
        if (numVertexElements)
            _vertexBuffer = VertexBuffer::Create(numVertexElements, numVertices);
        if (numVaryingElements)
            _varyingBuffer = VertexBuffer::Create(numVaryingElements, numVertices);
        _computeContext = ComputeContext::Create(_farMesh);
        _drawContext = DrawContext::Create(_farMesh->GetPatchTables(), bits.test(MeshFVarData));
        _drawContext->UpdateVertexTexture(_vertexBuffer);
    }

    FarMesh<OsdVertex> *_farMesh;
    VertexBuffer *_vertexBuffer;
    VertexBuffer *_varyingBuffer;
    ComputeContext *_computeContext;
    ComputeController *_computeController;
    DrawContext *_drawContext;
};

#ifdef OPENSUBDIV_HAS_OPENCL

template <class VERTEX_BUFFER>
class OsdMesh<VERTEX_BUFFER, OsdCLComputeController, OsdGLDrawContext> : public OsdGLMeshInterface {
public:
    typedef VERTEX_BUFFER VertexBuffer;
    typedef OsdCLComputeController ComputeController; 
    typedef typename ComputeController::ComputeContext ComputeContext; 
    typedef OsdGLDrawContext DrawContext; 
    typedef typename DrawContext::VertexBufferBinding VertexBufferBinding; 

    OsdMesh(ComputeController * computeController,
            HbrMesh<OsdVertex> * hmesh,
            int numVertexElements,
            int numVaryingElements,
            int level,
            OsdMeshBitset bits,
            cl_context clContext,
            cl_command_queue clQueue) :

            _farMesh(0),
            _vertexBuffer(0),
            _varyingBuffer(0),
            _computeContext(0),
            _computeController(computeController),
            _drawContext(0),
            _clContext(clContext),
            _clQueue(clQueue)
    {
        FarMeshFactory<OsdVertex> meshFactory(hmesh, level, bits.test(MeshAdaptive));
        _farMesh = meshFactory.Create(bits.test(MeshFVarData));

        _initialize(numVertexElements, numVaryingElements, level, bits);
    }

    OsdMesh(ComputeController * computeController,
            FarMesh<OsdVertex> * fmesh,
            int numVertexElements,
            int numVaryingElements,
            int level,
            OsdMeshBitset bits,
            cl_context clContext,
            cl_command_queue clQueue) :

            _farMesh(fmesh),
            _vertexBuffer(0),
            _varyingBuffer(0),
            _computeContext(0),
            _computeController(computeController),
            _drawContext(0),
            _clContext(clContext),
            _clQueue(clQueue)
    {
        _initialize(numVertexElements, numVaryingElements, level, bits);
    }

    virtual ~OsdMesh() {
        delete _farMesh;
        delete _vertexBuffer;
        delete _computeContext;
        delete _drawContext;
    }

    virtual int GetNumVertices() const { return _farMesh->GetNumVertices(); }

    virtual void UpdateVertexBuffer(float const *vertexData, int startVertex, int numVerts) {
        _vertexBuffer->UpdateData(vertexData, startVertex, numVerts, _clQueue);
    }
    virtual void UpdateVaryingBuffer(float const *varyingData, int startVertex, int numVerts) {
        _varyingBuffer->UpdateData(varyingData, startVertex, numVerts, _clQueue);
    }
    virtual void Refine() {
        _computeController->Refine(_computeContext, _farMesh->GetKernelBatches(), _vertexBuffer, _varyingBuffer);
    }
    virtual void Synchronize() {
        _computeController->Synchronize();
    }
    virtual VertexBufferBinding BindVertexBuffer() {
        return _vertexBuffer->BindVBO();
    }
    virtual VertexBufferBinding BindVaryingBuffer() {
        return _varyingBuffer->BindVBO();
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
                      int level,
                      OsdMeshBitset bits) 
    {
        int numVertices = _farMesh->GetNumVertices();
        if (numVertexElements)
            _vertexBuffer = VertexBuffer::Create(numVertexElements, numVertices, _clContext);
        if (numVaryingElements)
            _varyingBuffer = VertexBuffer::Create(numVaryingElements, numVertices, _clContext);
        _computeContext = ComputeContext::Create(_farMesh, _clContext);
        _drawContext = DrawContext::Create(_farMesh->GetPatchTables(), bits.test(MeshFVarData));
        _drawContext->UpdateVertexTexture(_vertexBuffer);
    }

    FarMesh<OsdVertex> *_farMesh;
    VertexBuffer *_vertexBuffer;
    VertexBuffer *_varyingBuffer;
    ComputeContext *_computeContext;
    ComputeController *_computeController;
    DrawContext *_drawContext;

    cl_context _clContext;
    cl_command_queue _clQueue;
};
#endif

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OSD_GL_MESH_H
