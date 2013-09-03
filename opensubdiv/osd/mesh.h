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
#ifndef OSD_MESH_H
#define OSD_MESH_H

#include "../version.h"

#include "../far/mesh.h"
#include "../far/meshFactory.h"

#include "../hbr/mesh.h"

#include "../osd/vertex.h"

#include <bitset>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

enum OsdMeshBits {
    MeshAdaptive    = 0,

    MeshPtexData    = 1,
    MeshFVarData    = 2,

    NUM_MESH_BITS   = 3,
};
typedef std::bitset<NUM_MESH_BITS> OsdMeshBitset;

template <class DRAW_CONTEXT>
class OsdMeshInterface {
public:
    typedef DRAW_CONTEXT DrawContext;
    typedef typename DrawContext::VertexBufferBinding VertexBufferBinding;

public:
    OsdMeshInterface() { }

    virtual ~OsdMeshInterface() { }

    virtual int GetNumVertices() const = 0;

    virtual void UpdateVertexBuffer(float const *vertexData, int startVertex, int numVerts) = 0;

    virtual void UpdateVaryingBuffer(float const *varyingData, int startVertex, int numVerts) = 0;

    virtual void Refine() = 0;

    virtual void Synchronize() = 0;

    virtual DrawContext * GetDrawContext() = 0;

    virtual VertexBufferBinding BindVertexBuffer() = 0;

    virtual VertexBufferBinding BindVaryingBuffer() = 0;
};

template <class VERTEX_BUFFER, class COMPUTE_CONTROLLER, class DRAW_CONTEXT>
class OsdMesh : public OsdMeshInterface<DRAW_CONTEXT> {
public:
    typedef VERTEX_BUFFER VertexBuffer;
    typedef COMPUTE_CONTROLLER ComputeController;
    typedef typename ComputeController::ComputeContext ComputeContext; 
    typedef DRAW_CONTEXT DrawContext; 
    typedef typename DrawContext::VertexBufferBinding VertexBufferBinding;

    OsdMesh(ComputeController * computeController,
            HbrMesh<OsdVertex> * hmesh,
            int numVertexElements,
            int numVaryingElements,
            int level,
            OsdMeshBitset bits = OsdMeshBitset()) :

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
            OsdMeshBitset bits = OsdMeshBitset()) :

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
        return VertexBufferBinding(0);
    }
    virtual VertexBufferBinding BindVaryingBuffer() {
        return VertexBufferBinding(0);
    }
    virtual DrawContext * GetDrawContext() {
        return _drawContext;
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
    }

    FarMesh<OsdVertex> *_farMesh;
    VertexBuffer *_vertexBuffer;
    VertexBuffer *_varyingBuffer;
    ComputeContext *_computeContext;
    ComputeController *_computeController;
    DrawContext *_drawContext;
};

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OSD_MESH_H
