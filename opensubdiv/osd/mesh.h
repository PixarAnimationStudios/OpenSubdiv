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

struct ID3D11DeviceContext;

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

enum MeshBits {
    MeshAdaptive             = 0,
    MeshInterleaveVarying    = 1,
    MeshFVarData             = 2,
    MeshUseSingleCreasePatch = 3,
    MeshEndCapBSplineBasis   = 4,  // exclusive
    MeshEndCapGregoryBasis   = 5,  // exclusive
    MeshEndCapLegacyGregory  = 6,  // exclusive
    NUM_MESH_BITS            = 7,
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

template <typename STENCIL_TABLES, typename DEVICE_CONTEXT>
STENCIL_TABLES const *
convertToCompatibleStencilTables(
    Far::StencilTables const *table, DEVICE_CONTEXT *context) {
    if (not table) return NULL;
    return STENCIL_TABLES::Create(table, context);
}

template <>
Far::StencilTables const *
convertToCompatibleStencilTables<Far::StencilTables, void>(
    Far::StencilTables const *table, void *  /*context*/) {
    // no need for conversion
    // XXX: We don't want to even copy.
    if (not table) return NULL;
    return new Far::StencilTables(*table);
}

template <>
Far::StencilTables const *
convertToCompatibleStencilTables<Far::StencilTables, ID3D11DeviceContext>(
    Far::StencilTables const *table, ID3D11DeviceContext *  /*context*/) {
    // no need for conversion
    // XXX: We don't want to even copy.
    if (not table) return NULL;
    return new Far::StencilTables(*table);
}

// ---------------------------------------------------------------------------

template <typename EVALUATOR>
class EvaluatorCacheT {
public:
    ~EvaluatorCacheT() {
        for(typename Evaluators::iterator it = _evaluators.begin();
            it != _evaluators.end(); ++it) {
            delete it->evaluator;
        }
    }

    // XXX: FIXME, linear search
    struct Entry {
        Entry(VertexBufferDescriptor const &sd,
              VertexBufferDescriptor const &dd,
              EVALUATOR *e) : srcDesc(sd), dstDesc(dd), evaluator(e) {}
        VertexBufferDescriptor srcDesc, dstDesc;
        EVALUATOR *evaluator;
    };
    typedef std::vector<Entry> Evaluators;

    template <typename DEVICE_CONTEXT>
    EVALUATOR *GetEvaluator(VertexBufferDescriptor const &srcDesc,
                            VertexBufferDescriptor const &dstDesc,
                            DEVICE_CONTEXT *deviceContext) {

        for(typename Evaluators::iterator it = _evaluators.begin();
            it != _evaluators.end(); ++it) {
            if (it->srcDesc.length == srcDesc.length and
                it->srcDesc.stride == srcDesc.stride and
                it->dstDesc.length == dstDesc.length and
                it->dstDesc.stride == dstDesc.stride) {
                return it->evaluator;
            }
        }
        EVALUATOR *e = EVALUATOR::Create(srcDesc, dstDesc, deviceContext);
        _evaluators.push_back(Entry(srcDesc, dstDesc, e));
        return e;
    }

private:
    Evaluators _evaluators;
};


// template helpers to see if the evaluator is instantiatable or not.
template <typename EVALUATOR>
struct instantiatable
{
    typedef char yes[1];
    typedef char no[2];
    template <typename C> static yes &chk(typename C::Instantiatable *t=0);
    template <typename C> static no  &chk(...);
    static bool const value = sizeof(chk<EVALUATOR>(0)) == sizeof(yes);
};
template <bool C, typename T=void>
struct enable_if { typedef T type; };
template <typename T>
struct enable_if<false, T> { };

// extract a kernel from cache if available
template <typename EVALUATOR, typename DEVICE_CONTEXT>
static EVALUATOR *GetEvaluator(
    EvaluatorCacheT<EVALUATOR> *cache,
    VertexBufferDescriptor const &srcDesc,
    VertexBufferDescriptor const &dstDesc,
    DEVICE_CONTEXT deviceContext,
    typename enable_if<instantiatable<EVALUATOR>::value, void>::type*t=0) {
    (void)t;
    if (cache == NULL) return NULL;
    return cache->GetEvaluator(srcDesc, dstDesc, deviceContext);
}

// fallback
template <typename EVALUATOR, typename DEVICE_CONTEXT>
static EVALUATOR *GetEvaluator(
    EvaluatorCacheT<EVALUATOR> *,
    VertexBufferDescriptor const &,
    VertexBufferDescriptor const &,
    DEVICE_CONTEXT,
    typename enable_if<!instantiatable<EVALUATOR>::value, void>::type*t=0) {
    (void)t;
    return NULL;
}

// ---------------------------------------------------------------------------

template <typename VERTEX_BUFFER,
          typename STENCIL_TABLES,
          typename EVALUATOR,
          typename DRAW_CONTEXT,
          typename DEVICE_CONTEXT = void>
class Mesh : public MeshInterface<DRAW_CONTEXT> {
public:
    typedef VERTEX_BUFFER VertexBuffer;
    typedef EVALUATOR Evaluator;
    typedef STENCIL_TABLES StencilTables;
    typedef DRAW_CONTEXT DrawContext;
    typedef DEVICE_CONTEXT DeviceContext;
    typedef EvaluatorCacheT<Evaluator> EvaluatorCache;
    typedef typename DrawContext::VertexBufferBinding VertexBufferBinding;

    Mesh(Far::TopologyRefiner * refiner,
         int numVertexElements,
         int numVaryingElements,
         int level,
         MeshBitset bits = MeshBitset(),
         EvaluatorCache * evaluatorCache = NULL,
         DeviceContext * deviceContext = NULL) :

            _refiner(refiner),
            _patchTables(NULL),
            _numVertices(0),
            _vertexBuffer(NULL),
            _varyingBuffer(NULL),
            _vertexStencilTables(NULL),
            _varyingStencilTables(NULL),
            _evaluatorCache(evaluatorCache),
            _drawContext(NULL),
            _deviceContext(deviceContext) {

        assert(_refiner);

        MeshInterface<DRAW_CONTEXT>::refineMesh(
            *_refiner, level,
            bits.test(MeshAdaptive),
            bits.test(MeshUseSingleCreasePatch));

        int vertexBufferStride = numVertexElements +
            (bits.test(MeshInterleaveVarying) ? numVaryingElements : 0);
        int varyingBufferStride =
            (bits.test(MeshInterleaveVarying) ? 0 : numVaryingElements);

        initializeContext(numVertexElements,
                          numVaryingElements,
                          level, bits);

        initializeVertexBuffers(_numVertices,
                                vertexBufferStride,
                                varyingBufferStride);

        // configure vertex buffer descriptor
        _vertexDesc = VertexBufferDescriptor(0,
                                             numVertexElements,
                                             vertexBufferStride);
        if (bits.test(MeshInterleaveVarying)) {
            _varyingDesc = VertexBufferDescriptor(numVertexElements,
                                                  numVaryingElements,
                                                  vertexBufferStride);
        } else {
            _varyingDesc = VertexBufferDescriptor(0,
                                                  numVaryingElements,
                                                  varyingBufferStride);
        }



        // will retire
        _drawContext->UpdateVertexTexture(_vertexBuffer, _deviceContext);
    }

    virtual ~Mesh() {
        delete _refiner;
        delete _patchTables;
        delete _vertexBuffer;
        delete _varyingBuffer;
        delete _vertexStencilTables;
        delete _varyingStencilTables;
        delete _drawContext;
        // deviceContext and evaluatorCache are not owned by this class.
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

        int numControlVertices = _refiner->GetNumVertices(0);

        VertexBufferDescriptor srcDesc = _vertexDesc;
        VertexBufferDescriptor dstDesc(srcDesc);
        dstDesc.offset += numControlVertices * dstDesc.stride;

        // note that the _evaluatorCache can be NULL and thus
        // the evaluatorInstance can be NULL
        //  (for uninstantiatable kernels CPU,TBB etc)
        Evaluator const *instance = GetEvaluator<Evaluator>(
            _evaluatorCache, srcDesc, dstDesc, _deviceContext);

        Evaluator::EvalStencils(_vertexBuffer, srcDesc,
                                _vertexBuffer, dstDesc,
                                _vertexStencilTables,
                                instance, _deviceContext);

        if (_varyingDesc.length > 0) {
            VertexBufferDescriptor srcDesc = _varyingDesc;
            VertexBufferDescriptor dstDesc(srcDesc);
            dstDesc.offset += numControlVertices * dstDesc.stride;

            instance = GetEvaluator<Evaluator>(
                _evaluatorCache, srcDesc, dstDesc, _deviceContext);

            if (_varyingBuffer) {
                // non-interleaved
                Evaluator::EvalStencils(_varyingBuffer, srcDesc,
                                        _varyingBuffer, dstDesc,
                                        _varyingStencilTables,
                                        instance, _deviceContext);
            } else {
                // interleaved
                Evaluator::EvalStencils(_vertexBuffer, srcDesc,
                                        _vertexBuffer, dstDesc,
                                        _varyingStencilTables,
                                        instance, _deviceContext);
            }
        }
    }

    virtual void Synchronize() {
        Evaluator::Synchronize(_deviceContext);
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
                           int level, MeshBitset bits) {
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

        _drawContext = DrawContext::Create(_patchTables, _deviceContext);

        // numvertices = coarse verts + refined verts + gregory basis verts
        _numVertices = vertexStencils->GetNumControlVertices()
            + vertexStencils->GetNumStencils();

        // convert to device stenciltables if necessary.
        _vertexStencilTables =
            convertToCompatibleStencilTables<StencilTables>(
            vertexStencils, _deviceContext);
        _varyingStencilTables =
            convertToCompatibleStencilTables<StencilTables>(
            varyingStencils, _deviceContext);

        // FIXME: we do extra copyings for Far::Stencils.
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

    VertexBuffer * _vertexBuffer;
    VertexBuffer * _varyingBuffer;

    VertexBufferDescriptor _vertexDesc;
    VertexBufferDescriptor _varyingDesc;

    StencilTables const * _vertexStencilTables;
    StencilTables const * _varyingStencilTables;
    EvaluatorCache * _evaluatorCache;

    DrawContext *_drawContext;
    DeviceContext *_deviceContext;
};

} // end namespace Osd

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OSD_MESH_H
