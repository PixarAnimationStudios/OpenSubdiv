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

#ifndef OSD_GL_MESH_H
#define OSD_GL_MESH_H

#include "../version.h"

#include "../osd/mesh.h"
#include "../osd/glDrawContext.h"
#include "../osd/vertexDescriptor.h"
#include "../far/gregoryBasis.h"
#include "../far/gregoryPatch.h"

#ifdef OPENSUBDIV_HAS_OPENCL
#  include "../osd/clComputeController.h"
#  include "../osd/opencl.h"
#endif

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

typedef MeshInterface<GLDrawContext> GLMeshInterface;

template <class VERTEX_BUFFER, class COMPUTE_CONTROLLER>
class Mesh<VERTEX_BUFFER, COMPUTE_CONTROLLER, GLDrawContext> : public GLMeshInterface {
public:
    typedef VERTEX_BUFFER VertexBuffer;
    typedef COMPUTE_CONTROLLER ComputeController;
    typedef typename ComputeController::ComputeContext ComputeContext;
    typedef GLDrawContext DrawContext;
    typedef typename DrawContext::VertexBufferBinding VertexBufferBinding;

    Mesh(ComputeController * computeController,
            Far::TopologyRefiner * refiner,
            int numVertexElements,
            int numVaryingElements,
            int level,
            MeshBitset bits) :

            _refiner(refiner),
            _patchTables(0),
            _vertexBuffer(0),
            _varyingBuffer(0),
            _computeContext(0),
            _computeController(computeController),
            _drawContext(0),
            _numVertices(0)
    {

        GLMeshInterface::refineMesh(*_refiner, level,
                                    bits.test(MeshAdaptive),
                                    bits.test(MeshUseSingleCreasePatch));

        int numVertexElementsInterleaved = numVertexElements +
            (bits.test(MeshInterleaveVarying) ? numVaryingElements : 0);
        int numVaryingElementsNonInterleaved = 
            (bits.test(MeshInterleaveVarying) ? 0 : numVaryingElements);

        // numVertices is computed from stencilTables (may or may not includes gregory basis vertices)
        initializeContext(numVertexElements,
                          numVaryingElements,
                          numVertexElementsInterleaved, level, bits);

        initializeVertexBuffers(_numVertices,
                                numVertexElementsInterleaved,
                                numVaryingElementsNonInterleaved);

        // will retire soon
        _drawContext->UpdateVertexTexture(_vertexBuffer);
    }

    Mesh(ComputeController * computeController,
            Far::TopologyRefiner * refiner,
            Far::PatchTables * patchTables,
            VertexBuffer * vertexBuffer,
            VertexBuffer * varyingBuffer,
            ComputeContext * computeContext,
            DrawContext * drawContext) :

            _refiner(refiner),
            _patchTables(patchTables),
            _vertexBuffer(vertexBuffer),
            _varyingBuffer(varyingBuffer),
            _computeContext(computeContext),
            _computeController(computeController),
            _drawContext(drawContext),
            _numVertices(0)
    {
        _drawContext->UpdateVertexTexture(_vertexBuffer);
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
        _vertexBuffer->UpdateData(vertexData, startVertex, numVerts);
    }

    virtual void UpdateVaryingBuffer(float const *varyingData, int startVertex, int numVerts) {
        _varyingBuffer->UpdateData(varyingData, startVertex, numVerts);
    }

    virtual void Refine() {
        _computeController->Compute(_computeContext, _vertexBuffer, _varyingBuffer);
    }

    virtual void Refine(VertexBufferDescriptor const * vertexDesc,
                        VertexBufferDescriptor const * varyingDesc,
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

    virtual Far::TopologyRefiner const * GetTopologyRefiner() const {
        return _refiner;
    }

    virtual void SetFVarDataChannel(int fvarWidth, std::vector<float> const & fvarData) {
        if (_patchTables and _drawContext and fvarWidth and (not fvarData.empty())) {
            _drawContext->SetFVarDataTexture(*_patchTables, fvarWidth, fvarData);
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

        if (bits.test(MeshUseGregoryBasis) and bits.test(MeshAdaptive)) {
            // use gregory stencils
            Far::GregoryBasisFactory *gregoryBasisFactory = new Far::GregoryBasisFactory(*_refiner, true);
            _patchTables = Far::PatchTablesFactory::Create(*_refiner, poptions, gregoryBasisFactory);
            if (gregoryBasisFactory) {
                if (Far::StencilTables const *vertexStencilsWithGregoryBasis =
                    gregoryBasisFactory->CreateVertexStencilTables(vertexStencils, true)) {
                    delete vertexStencils;
                    vertexStencils = vertexStencilsWithGregoryBasis;
                }

                if (varyingStencils) {
                    if (Far::StencilTables const *varyingStencilsWithGregoryBasis =
                        gregoryBasisFactory->CreateVaryingStencilTables(varyingStencils, true)) {
                        delete varyingStencils;
                        varyingStencils = varyingStencilsWithGregoryBasis;
                    }
                }
            }
            delete gregoryBasisFactory;
        } else if (bits.test(MeshAdaptive)) {
            // use legacy gregory
            Far::GregoryPatchFactory *gregoryPatchFactory = new Far::GregoryPatchFactory(*_refiner);
            _patchTables = Far::PatchTablesFactory::Create(*_refiner, poptions, gregoryPatchFactory);

            gregoryPatchFactory->AddGregoryPatchTables(_patchTables);
            delete gregoryPatchFactory;
        } else {
            // uniform
            _patchTables = Far::PatchTablesFactory::Create(*_refiner, poptions);
        }

        _drawContext = DrawContext::Create(_patchTables, numElements);
        _computeContext = ComputeContext::Create(vertexStencils,
                                                 varyingStencils);

        // numvertices = coarse verts + refined verts + gregory basis verts
        _numVertices = vertexStencils->GetNumControlVertices()
            + vertexStencils->GetNumStencils();

        delete vertexStencils;
        delete varyingStencils;
    }

    void initializeVertexBuffers(int numVertices,
                                 int numVertexElements,
                                 int numVaryingElements) {

        if (numVertexElements) {
            _vertexBuffer = VertexBuffer::Create(numVertexElements, numVertices);
        }

        if (numVaryingElements) {
            _varyingBuffer = VertexBuffer::Create(numVaryingElements, numVertices);
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
};

#ifdef OPENSUBDIV_HAS_OPENCL

template <class VERTEX_BUFFER>
class Mesh<VERTEX_BUFFER, CLComputeController, GLDrawContext> : public GLMeshInterface {
public:
    typedef VERTEX_BUFFER VertexBuffer;
    typedef CLComputeController ComputeController;
    typedef typename ComputeController::ComputeContext ComputeContext;
    typedef GLDrawContext DrawContext;
    typedef typename DrawContext::VertexBufferBinding VertexBufferBinding;

    Mesh(ComputeController * computeController,
            Far::TopologyRefiner * refiner,
            int numVertexElements,
            int numVaryingElements,
            int level,
            MeshBitset bits,
            cl_context clContext,
            cl_command_queue clQueue) :

            _refiner(refiner),
            _patchTables(0),
            _vertexBuffer(0),
            _varyingBuffer(0),
            _computeContext(0),
            _computeController(computeController),
            _drawContext(0),
            _numVertices(0),
            _clContext(clContext),
            _clQueue(clQueue)
    {
        assert(_refiner);

        GLMeshInterface::refineMesh(*_refiner, level,
                                    bits.test(MeshAdaptive),
                                    bits.test(MeshUseSingleCreasePatch));


        int numVertexElementsInterleaved = numVertexElements +
            (bits.test(MeshInterleaveVarying) ? numVaryingElements : 0);
        int numVaryingElementsNonInterleaved = 
            (bits.test(MeshInterleaveVarying) ? 0 : numVaryingElements);

        initializeContext(numVertexElements,
                          numVaryingElements,
                          numVertexElementsInterleaved, level, bits);

        initializeVertexBuffers(_numVertices,
                                numVertexElementsInterleaved,
                                numVaryingElementsNonInterleaved);

        // will retire
        _drawContext->UpdateVertexTexture(_vertexBuffer);
    }

    Mesh(ComputeController * computeController,
            Far::TopologyRefiner * refiner,
            Far::PatchTables * patchTables,
            VertexBuffer * vertexBuffer,
            VertexBuffer * varyingBuffer,
            ComputeContext * computeContext,
            DrawContext * drawContext,
            cl_context clContext,
            cl_command_queue clQueue) :

            _refiner(refiner),
            _patchTables(patchTables),
            _vertexBuffer(vertexBuffer),
            _varyingBuffer(varyingBuffer),
            _computeContext(computeContext),
            _computeController(computeController),
            _drawContext(drawContext),
            _numVertices(0),
            _clContext(clContext),
            _clQueue(clQueue)
    {
        _drawContext->UpdateVertexTexture(_vertexBuffer);
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
        _vertexBuffer->UpdateData(vertexData, startVertex, numVerts, _clQueue);
    }

    virtual void UpdateVaryingBuffer(float const *varyingData, int startVertex, int numVerts) {
        _varyingBuffer->UpdateData(varyingData, startVertex, numVerts, _clQueue);
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

    virtual Far::TopologyRefiner const * GetTopologyRefiner() const {
        return _refiner;
    }

    virtual void SetFVarDataChannel(int fvarWidth, std::vector<float> const & fvarData) {
        if (_patchTables and _drawContext and fvarWidth and (not fvarData.empty())) {
            _drawContext->SetFVarDataTexture(*_patchTables, fvarWidth, fvarData);
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

        _computeContext = ComputeContext::Create(_clContext, vertexStencils, varyingStencils);

        Far::PatchTablesFactory::Options poptions(level);
        poptions.generateFVarTables = bits.test(MeshFVarData);
        poptions.useSingleCreasePatch = bits.test(MeshUseSingleCreasePatch);

        // use gregory stencils

        if (bits.test(MeshUseGregoryBasis) and bits.test(MeshAdaptive)) {
            Far::GregoryBasisFactory *gregoryBasisFactory = new Far::GregoryBasisFactory(*_refiner, true);
            _patchTables = Far::PatchTablesFactory::Create(*_refiner, poptions, gregoryBasisFactory);
            if (gregoryBasisFactory) {
                if (Far::StencilTables const *vertexStencilsWithGregoryBasis =
                    gregoryBasisFactory->CreateVertexStencilTables(vertexStencils, true)) {
                    delete vertexStencils;
                    vertexStencils = vertexStencilsWithGregoryBasis;
                }

                if (varyingStencils) {
                    if (Far::StencilTables const *varyingStencilsWithGregoryBasis =
                        gregoryBasisFactory->CreateVaryingStencilTables(varyingStencils, true)) {
                        delete varyingStencils;
                        varyingStencils = varyingStencilsWithGregoryBasis;
                    }
                }
            }
            delete gregoryBasisFactory;
        } else if (bits.test(MeshAdaptive)) {
            Far::GregoryPatchFactory *gregoryPatchFactory = new Far::GregoryPatchFactory(*_refiner);
            _patchTables = Far::PatchTablesFactory::Create(*_refiner, poptions, gregoryPatchFactory);
        } else {
            // uniform
            _patchTables = Far::PatchTablesFactory::Create(*_refiner, poptions);
        }

        _drawContext = DrawContext::Create(_patchTables, numElements);
        _computeContext = ComputeContext::Create(_clContext,
                                                 vertexStencils,
                                                 varyingStencils);

        // numvertices = coarse verts + refined verts + gregory basis verts
        _numVertices = vertexStencils->GetNumControlVertices()
            + vertexStencils->GetNumStencils();

        delete vertexStencils;
        delete varyingStencils;
    }

    void initializeVertexBuffers(int numVertices,
                                 int numVertexElements,
                                 int numVaryingElements) {

        if (numVertexElements) {
            _vertexBuffer = VertexBuffer::Create(numVertexElements, numVertices, _clContext);
        }

        if (numVaryingElements) {
            _varyingBuffer = VertexBuffer::Create(numVaryingElements, numVertices, _clContext);
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

    cl_context _clContext;
    cl_command_queue _clQueue;
};
#endif

} // end namespace Osd

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OSD_GL_MESH_H
