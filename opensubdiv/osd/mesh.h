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

#include <bitset>
#include <cassert>
#include <cstring>
#include <vector>

#include "../far/topologyRefiner.h"
#include "../far/patchTablesFactory.h"
#include "../far/stencilTables.h"
#include "../far/stencilTablesFactory.h"

#include "../osd/vertexDescriptor.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

enum MeshBits {
    MeshAdaptive             = 0,
    MeshInterleaveVarying    = 1,
    MeshPtexData             = 2,
    MeshFVarData             = 3,
    MeshUseSingleCreasePatch = 4,
    MeshEndCapBSplineBasis   = 5,  // exclusive
    MeshEndCapGregoryBasis   = 6,  // exclusive
    MeshEndCapLegacyGregory  = 7,  // exclusive
    NUM_MESH_BITS            = 8,
};
typedef std::bitset<NUM_MESH_BITS> MeshBitset;

// ---------------------------------------------------------------------------

template <class DRAW_CONTEXT>
class MeshInterface {
public:
    typedef DRAW_CONTEXT DrawContext;
    typedef typename DrawContext::VertexBufferBinding VertexBufferBinding;

public:
    MeshInterface() { }

    virtual ~MeshInterface() { }

    virtual int GetNumVertices() const = 0;

    virtual void UpdateVertexBuffer(float const *vertexData,
                                    int startVertex, int numVerts) = 0;

    virtual void UpdateVaryingBuffer(float const *varyingData,
                                     int startVertex, int numVerts) = 0;

    virtual void Refine() = 0;

    virtual void Refine(VertexBufferDescriptor const *vertexDesc,
                        VertexBufferDescriptor const *varyingDesc) = 0;

    virtual void Refine(VertexBufferDescriptor const *vertexDesc,
                        VertexBufferDescriptor const *varyingDesc,
                        bool interleaved) = 0;

    virtual void Synchronize() = 0;

    virtual DrawContext * GetDrawContext() = 0;

    virtual VertexBufferBinding BindVertexBuffer() = 0;

    virtual VertexBufferBinding BindVaryingBuffer() = 0;

    virtual void SetFVarDataChannel(int fvarWidth,
                                    std::vector<float> const & fvarData) = 0;

protected:
    static inline void refineMesh(Far::TopologyRefiner & refiner,
                                  int level, bool adaptive,
                                  bool singleCreasePatch) {
        if (adaptive) {
            Far::TopologyRefiner::AdaptiveOptions options(level);
            options.useSingleCreasePatch = singleCreasePatch;
            refiner.RefineAdaptive(options);
        } else {
            //  This dependency on FVar channels should not be necessary
            bool fullTopologyInLastLevel = refiner.GetNumFVarChannels()>0;

            Far::TopologyRefiner::UniformOptions options(level);
            options.fullTopologyInLastLevel = fullTopologyInLastLevel;
            refiner.RefineUniform(options);
        }
    }
};

// ---------------------------------------------------------------------------

template <class VERTEX_BUFFER,
          class COMPUTE_CONTROLLER,
          class DRAW_CONTEXT,
          class DEVICE_CONTEXT = void>
class Mesh : public MeshInterface<DRAW_CONTEXT> {
public:
    typedef VERTEX_BUFFER VertexBuffer;
    typedef COMPUTE_CONTROLLER ComputeController;
    typedef DRAW_CONTEXT DrawContext;
    typedef DEVICE_CONTEXT DeviceContext;
    typedef typename ComputeController::ComputeContext ComputeContext;
    typedef typename DrawContext::VertexBufferBinding VertexBufferBinding;

    Mesh(ComputeController * computeController,
         Far::TopologyRefiner * refiner,
         int numVertexElements,
         int numVaryingElements,
         int level,
         MeshBitset bits = MeshBitset(),
         DeviceContext * deviceContext = NULL) :

            _refiner(refiner),
            _patchTables(NULL),
            _numVertices(0),
            _vertexBuffer(NULL),
            _varyingBuffer(NULL),
            _computeContext(NULL),
            _computeController(computeController),
            _drawContext(NULL),
            _deviceContext(deviceContext) {

        assert(_refiner);

        MeshInterface<DRAW_CONTEXT>::refineMesh(
            *_refiner, level,
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
        _drawContext->UpdateVertexTexture(_vertexBuffer, _deviceContext);
    }

    virtual ~Mesh() {
        delete _refiner;
        delete _patchTables;
        delete _vertexBuffer;
        delete _varyingBuffer;
        delete _computeContext;
        delete _drawContext;
        // devicecontext and computecontroller are not owned by this class.
    }

    virtual void UpdateVertexBuffer(float const *vertexData,
                                    int startVertex, int numVerts) {
        _vertexBuffer->UpdateData(vertexData, startVertex, numVerts,
                                  _deviceContext);
    }

    virtual void UpdateVaryingBuffer(float const *varyingData,
                                     int startVertex, int numVerts) {
        _varyingBuffer->UpdateData(varyingData, startVertex, numVerts,
                                   _deviceContext);
    }

    virtual void Refine() {
        _computeController->Compute(_computeContext,
                                    _vertexBuffer, _varyingBuffer);
    }

    virtual void Refine(VertexBufferDescriptor const *vertexDesc,
                        VertexBufferDescriptor const *varyingDesc) {
        _computeController->Compute(_computeContext,
                                    _vertexBuffer, _varyingBuffer,
                                    vertexDesc, varyingDesc);
    }

    virtual void Refine(VertexBufferDescriptor const *vertexDesc,
                        VertexBufferDescriptor const *varyingDesc,
                        bool interleaved) {
        _computeController->Compute(_computeContext,
                                    _vertexBuffer,
                                    (interleaved ?
                                     _vertexBuffer : _varyingBuffer),
                                    vertexDesc, varyingDesc);
    }

    virtual void Synchronize() {
        _computeController->Synchronize();
    }

    virtual DrawContext * GetDrawContext() {
        return _drawContext;
    }

    virtual void SetFVarDataChannel(int fvarWidth,
                                    std::vector<float> const & fvarData) {
        if (_patchTables and
            _drawContext and
            fvarWidth and
            (not fvarData.empty())) {
            _drawContext->SetFVarDataTexture(*_patchTables, fvarWidth, fvarData,
                                             _deviceContext);
        }
    }

    virtual int GetNumVertices() const { return _numVertices; }

    virtual VertexBufferBinding BindVertexBuffer() {
        return _vertexBuffer->BindVBO(_deviceContext);
    }

    virtual VertexBufferBinding BindVaryingBuffer() {
        return _varyingBuffer->BindVBO(_deviceContext);
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

private:
    void initializeContext(int numVertexElements,
                           int numVaryingElements,
                           int numElements, int level, MeshBitset bits) {
        assert(_refiner);

        Far::StencilTablesFactory::Options options;
        options.generateOffsets = true;
        options.generateIntermediateLevels =
            _refiner->IsUniform() ? false : true;

        Far::StencilTables const * vertexStencils = NULL;
        Far::StencilTables const * varyingStencils = NULL;

        if (numVertexElements>0) {

            vertexStencils = Far::StencilTablesFactory::Create(*_refiner,
                                                               options);
        }

        if (numVaryingElements>0) {

            options.interpolationMode =
                Far::StencilTablesFactory::INTERPOLATE_VARYING;

            varyingStencils = Far::StencilTablesFactory::Create(*_refiner,
                                                                options);
        }

        Far::PatchTablesFactory::Options poptions(level);
        poptions.generateFVarTables = bits.test(MeshFVarData);
        poptions.useSingleCreasePatch = bits.test(MeshUseSingleCreasePatch);

        if (bits.test(MeshEndCapBSplineBasis)) {
            poptions.SetEndCapType(
                Far::PatchTablesFactory::Options::ENDCAP_BSPLINE_BASIS);
        } else if (bits.test(MeshEndCapGregoryBasis)) {
            poptions.SetEndCapType(
                Far::PatchTablesFactory::Options::ENDCAP_GREGORY_BASIS);
            // points on gregory basis endcap boundary can be shared among
            // adjacent patches to save some stencils.
            poptions.shareEndCapPatchPoints = true;
        } else if (bits.test(MeshEndCapLegacyGregory)) {
            poptions.SetEndCapType(
                Far::PatchTablesFactory::Options::ENDCAP_LEGACY_GREGORY);
        }

        _patchTables = Far::PatchTablesFactory::Create(*_refiner, poptions);

        // if there's endcap stencils, merge it into regular stencils.
        if (_patchTables->GetEndCapVertexStencilTables()) {
            // append stencils
            if (Far::StencilTables const *vertexStencilsWithEndCap =
                Far::StencilTablesFactory::AppendEndCapStencilTables(
                    *_refiner,
                    vertexStencils,
                    _patchTables->GetEndCapVertexStencilTables())) {
                delete vertexStencils;
                vertexStencils = vertexStencilsWithEndCap;
            }
            if (varyingStencils) {
                if (Far::StencilTables const *varyingStencilsWithEndCap =
                    Far::StencilTablesFactory::AppendEndCapStencilTables(
                        *_refiner,
                        varyingStencils,
                        _patchTables->GetEndCapVaryingStencilTables())) {
                    delete varyingStencils;
                    varyingStencils = varyingStencilsWithEndCap;
                }
            }
        }

        _drawContext = DrawContext::Create(_patchTables, numElements,
                                           _deviceContext);
        _computeContext = ComputeContext::Create(vertexStencils,
                                                 varyingStencils,
                                                 _deviceContext);

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
            _vertexBuffer = VertexBuffer::Create(numVertexElements,
                                                 numVertices, _deviceContext);
        }

        if (numVaryingElements) {
            _varyingBuffer = VertexBuffer::Create(numVaryingElements,
                                                  numVertices, _deviceContext);
        }
    }

    Far::TopologyRefiner * _refiner;
    Far::PatchTables * _patchTables;

    int _numVertices;

    VertexBuffer * _vertexBuffer,
                 * _varyingBuffer;

    ComputeContext    * _computeContext;
    ComputeController * _computeController;

    DrawContext *_drawContext;

    DeviceContext *_deviceContext;
};

} // end namespace Osd

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OSD_MESH_H
