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

#ifndef OPENSUBDIV_EXAMPLES_GL_SHARE_TOPOLOGY_SCENE_H
#define OPENSUBDIV_EXAMPLES_GL_SHARE_TOPOLOGY_SCENE_H

#include "meshRefiner.h"
#include "vbo.h"
#include "sceneBase.h"

template <class EVALUATOR,
          class VERTEX_BUFFER,
          class STENCIL_TABLES,
          class DEVICE_CONTEXT=void>
class Scene : public SceneBase {
public:
    typedef EVALUATOR Evaluator;
    typedef STENCIL_TABLES StencilTables;
    typedef DEVICE_CONTEXT DeviceContext;
    typedef OpenSubdiv::Osd::EvaluatorCacheT<Evaluator> EvaluatorCache;
    typedef MeshRefiner<EVALUATOR, VERTEX_BUFFER,
                        STENCIL_TABLES, DEVICE_CONTEXT> MeshRefinerType;
    typedef VBO<VERTEX_BUFFER, DEVICE_CONTEXT> VBOType;

    Scene(Options const &options,
          EvaluatorCache * evaluatorCache = NULL,
          DeviceContext * deviceContext = NULL)
        : SceneBase(options),
          _vbo(NULL),
          _evaluatorCache(evaluatorCache),
          _deviceContext(deviceContext) {
    }

    virtual ~Scene() {
        Clear();
    }

    virtual void Refine(int object) {
        int vertsOffset = GetVertsOffset(object);
        _refiners[_objects[object].topologyIndex]->Refine(_vbo, vertsOffset);
    }

    virtual void Synchronize() {
        Evaluator::Synchronize(_deviceContext);
    }

    virtual void UpdateVertexBuffer(int vertsOffset, std::vector<float> const &src) {
        _vbo->UpdateVertexBuffer(vertsOffset, src);
    }
    virtual void UpdateVaryingBuffer(int vertsOffset, std::vector<float> const &src) {
        _vbo->UpdateVaryingBuffer(vertsOffset, src);
    }

    virtual GLuint BindVertexBuffer() {
        return _vbo->BindVertexBuffer();
    }

    virtual GLuint BindVaryingBuffer() {
        return _vbo->BindVaryingBuffer();
    }

    virtual size_t AllocateVBO(int numVerts,
                               OpenSubdiv::Osd::BufferDescriptor const &vertexDesc,
                               OpenSubdiv::Osd::BufferDescriptor const &varyingDesc,
                               bool interleaved) {

        if (_vbo) delete _vbo;
        _vbo = new VBOType(vertexDesc, varyingDesc, interleaved, numVerts, _deviceContext);
        return _vbo->GetSize();
    }

    virtual size_t createMeshRefiner(
        OpenSubdiv::Far::StencilTable const * vertexStencils,
        OpenSubdiv::Far::StencilTable const * varyingStencils,
        int numControlVertices) {

        MeshRefinerType *meshRefiner =
            new MeshRefinerType(vertexStencils, varyingStencils,
                                numControlVertices,
                                _evaluatorCache,
                                _deviceContext);
        _refiners.push_back(meshRefiner);

        size_t size = 0;
        if (vertexStencils) {
            size += vertexStencils->GetSizes().size() * sizeof(vertexStencils->GetSizes()[0])
            + vertexStencils->GetOffsets().size() * sizeof(vertexStencils->GetOffsets()[0])
            + vertexStencils->GetControlIndices().size() * sizeof(vertexStencils->GetControlIndices()[0])
            + vertexStencils->GetWeights().size() * sizeof(vertexStencils->GetWeights()[0]);
        }
        if (varyingStencils) {
            size += varyingStencils->GetSizes().size() * sizeof(varyingStencils->GetSizes()[0])
            + varyingStencils->GetOffsets().size() * sizeof(varyingStencils->GetOffsets()[0])
            + varyingStencils->GetControlIndices().size() * sizeof(varyingStencils->GetControlIndices()[0])
            + varyingStencils->GetWeights().size() * sizeof(varyingStencils->GetWeights()[0]);
        }
        return size;
    }

    void Clear() {
        for (typename std::vector<MeshRefinerType*>::iterator it = _refiners.begin();
             it != _refiners.end(); ++it) {
            delete *it;
        }

        _refiners.clear();
        _objects.clear();
        delete _vbo;
        _vbo = NULL;
    }

private:
    std::vector<MeshRefinerType*> _refiners;
    VBOType *_vbo;
    EvaluatorCache * _evaluatorCache;
    DeviceContext *_deviceContext;

};

#endif  // OPENSUBDIV_EXAMPLES_GL_SHARE_TOPOLOGY_SCENE_H
