//
//   Copyright 2015 Pixar
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

#ifndef OPENSUBDIV_EXAMPLES_GL_SHARE_TOPOLOGY_MESH_REFINER_H
#define OPENSUBDIV_EXAMPLES_GL_SHARE_TOPOLOGY_MESH_REFINER_H

#include <opensubdiv/osd/mesh.h>   // for evaluator cache

template <class EVALUATOR,
          class VERTEX_BUFFER,
          class STENCIL_TABLE,
          class DEVICE_CONTEXT=void>
class MeshRefiner {
public:
    typedef EVALUATOR Evaluator;
    typedef STENCIL_TABLE StencilTable;
    typedef DEVICE_CONTEXT DeviceContext;
    typedef OpenSubdiv::Osd::EvaluatorCacheT<Evaluator> EvaluatorCache;

    MeshRefiner(OpenSubdiv::Far::StencilTable const * vertexStencils, //XXX: takes ownership
                OpenSubdiv::Far::StencilTable const * varyingStencils,
                int numControlVertices,
                EvaluatorCache * evaluatorCache = NULL,
                DeviceContext * deviceContext = NULL)
        : _evaluatorCache(evaluatorCache),
          _deviceContext(deviceContext) {

        _numControlVertices = numControlVertices;
        _numVertices = numControlVertices + vertexStencils->GetNumStencils();

        _vertexStencils = OpenSubdiv::Osd::convertToCompatibleStencilTable<StencilTable>(
            vertexStencils, deviceContext);
        _varyingStencils = OpenSubdiv::Osd::convertToCompatibleStencilTable<StencilTable>(
            varyingStencils, deviceContext);
    }

    ~MeshRefiner() {
        delete _vertexStencils;
        delete _varyingStencils;
    }

    template <typename VBO>
    void Refine(VBO *vbo, int vertsOffset) {
        OpenSubdiv::Osd::BufferDescriptor const &globalVertexDesc =
            vbo->GetVertexDesc();
        OpenSubdiv::Osd::BufferDescriptor const &globalVaryingDesc =
            vbo->GetVaryingDesc();

        OpenSubdiv::Osd::BufferDescriptor vertexSrcDesc(
            globalVertexDesc.offset + vertsOffset * globalVertexDesc.stride,
            globalVertexDesc.length,
            globalVertexDesc.stride);
        OpenSubdiv::Osd::BufferDescriptor vertexDstDesc(
            vertexSrcDesc.offset + (_numControlVertices * vertexSrcDesc.stride),
            vertexSrcDesc.length,
            vertexSrcDesc.stride);

        // vertex
        Evaluator const *evalInstance = OpenSubdiv::Osd::GetEvaluator<Evaluator>(
            _evaluatorCache, vertexSrcDesc, vertexDstDesc, _deviceContext);

        Evaluator::EvalStencils(
            vbo->GetVertexBuffer(), vertexSrcDesc,
            vbo->GetVertexBuffer(), vertexDstDesc,
            _vertexStencils,
            evalInstance,
            _deviceContext);

        // varying
        if (_varyingStencils) {
            OpenSubdiv::Osd::BufferDescriptor varyingSrcDesc(
                globalVaryingDesc.offset + vertsOffset * globalVaryingDesc.stride,
                globalVaryingDesc.length,
                globalVaryingDesc.stride);

            OpenSubdiv::Osd::BufferDescriptor varyingDstDesc(
                varyingSrcDesc.offset + (_numControlVertices * varyingSrcDesc.stride),
                varyingSrcDesc.length,
                varyingSrcDesc.stride);

            evalInstance = OpenSubdiv::Osd::GetEvaluator<Evaluator>(
                _evaluatorCache, varyingSrcDesc, varyingDstDesc, _deviceContext);

            if (vbo->GetVaryingBuffer()) {
                // non interleaved
                Evaluator::EvalStencils(
                    vbo->GetVaryingBuffer(), varyingSrcDesc,
                    vbo->GetVaryingBuffer(), varyingDstDesc,
                    _varyingStencils,
                    evalInstance,
                    _deviceContext);
            } else {
                // interleaved
                Evaluator::EvalStencils(
                    vbo->GetVertexBuffer(), varyingSrcDesc,
                    vbo->GetVertexBuffer(), varyingDstDesc,
                    _varyingStencils,
                    evalInstance,
                    _deviceContext);
            }
        }
    }

    void Synchronize() {
        Evaluator::Synchronize(_deviceContext);
    }

    int GetNumVertices() const {  // total (control + refined)
        return _numVertices;
    }
    int GetNumControlVertices() const {
        return _numControlVertices;
    }

private:
    int _numVertices;
    int _numControlVertices;

    StencilTable const *_vertexStencils;
    StencilTable const *_varyingStencils;
    EvaluatorCache * _evaluatorCache;
    DeviceContext *_deviceContext;
};


#endif   // OPENSUBDIV_EXAMPLES_GL_SHARE_TOPOLOGY_TOPOLOGY_H
