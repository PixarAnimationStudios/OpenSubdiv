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

#include "glLoader.h"

#include "sceneBase.h"

#include "../../regression/common/far_utils.h"

#include <opensubdiv/far/patchTableFactory.h>
#include <opensubdiv/far/stencilTableFactory.h>

#include <limits>

using namespace OpenSubdiv;


SceneBase::SceneBase(Options const &options)
    : _options(options),
      _indexBuffer(0), _patchParamTexture(0) {
}

SceneBase::~SceneBase() {
    if (_indexBuffer) glDeleteBuffers(1, &_indexBuffer);
    if (_patchParamTexture) glDeleteTextures(1, &_patchParamTexture);

    for (int i = 0; i < (int)_patchTables.size(); ++i) {
        delete _patchTables[i];
    }
}

void
SceneBase::AddTopology(Shape const *shape, int level, bool varying) {
    Far::PatchTable const * patchTable = NULL;
    int numVerts = createStencilTable(shape, level, varying, &patchTable);

    // centering rest position
    float pmin[3] = { std::numeric_limits<float>::max(),
                      std::numeric_limits<float>::max(),
                      std::numeric_limits<float>::max() };
    float pmax[3] = { -std::numeric_limits<float>::max(),
                      -std::numeric_limits<float>::max(),
                      -std::numeric_limits<float>::max() };
    int nverts = shape->GetNumVertices();
    for (int i = 0; i < nverts; ++i) {
        for (int j = 0; j < 3; ++j) {
            float v = shape->verts[i*3+j];
            pmin[j] = std::min(v, pmin[j]);
            pmax[j] = std::max(v, pmax[j]);
        }
    }
    float center[3] = { (pmax[0]+pmin[0])*0.5f,
                        (pmax[1]+pmin[1])*0.5f,
                        pmin[2] };
    float radius = sqrt((pmax[0]-pmin[0])*(pmax[0]-pmin[0]) +
                        (pmax[1]-pmin[1])*(pmax[1]-pmin[1]) +
                        (pmax[2]-pmin[2])*(pmax[2]-pmin[2]));

    std::vector<float> restPosition(shape->verts);
    for (size_t i=0; i < restPosition.size()/3; ++i) {
        for (int j = 0; j < 3; ++j) {
            restPosition[i*3+j] = (shape->verts[i*3+j] - center[j])/radius;
        }
    }

    // store topology
    Topology topology;
    topology.numVerts = numVerts;
    topology.restPosition = restPosition;
    _topologies.push_back(topology);

    // store patch.
    // PatchTables is used later to be spliced into the index buffer.
    _patchTables.push_back(patchTable);
}

int
SceneBase::createStencilTable(Shape const *shape, int level, bool varying,
                              OpenSubdiv::Far::PatchTable const **patchTableOut) {

    Far::TopologyRefiner * refiner = 0;
    {
        Sdc::SchemeType type = GetSdcType(*shape);
        Sdc::Options options = GetSdcOptions(*shape);

        refiner = Far::TopologyRefinerFactory<Shape>::Create(
            *shape, Far::TopologyRefinerFactory<Shape>::Options(type, options));
        assert(refiner);
    }

    // Adaptive refinement currently supported only for catmull-clark scheme

    if (_options.adaptive) {
        Far::TopologyRefiner::AdaptiveOptions options(level);
        refiner->RefineAdaptive(options);
    } else {
        Far::TopologyRefiner::UniformOptions options(level);
        options.fullTopologyInLastLevel = true;
        refiner->RefineUniform(options);
    }

    Far::StencilTable const * vertexStencils=0, * varyingStencils=0;
    {
        Far::StencilTableFactory::Options options;
        options.generateOffsets = true;
        options.generateIntermediateLevels = _options.adaptive;

        vertexStencils = Far::StencilTableFactory::Create(*refiner, options);

        if (varying) {
            varyingStencils = Far::StencilTableFactory::Create(*refiner, options);
        }

        assert(vertexStencils);
    }

    Far::PatchTable const * patchTable = NULL;
    {
        Far::PatchTableFactory::Options poptions(level);
        if (_options.endCap == kEndCapBSplineBasis) {
            poptions.SetEndCapType(
                Far::PatchTableFactory::Options::ENDCAP_BSPLINE_BASIS);
        } else {
            poptions.SetEndCapType(
                Far::PatchTableFactory::Options::ENDCAP_GREGORY_BASIS);
        }
        patchTable = Far::PatchTableFactory::Create(*refiner, poptions);
    }
    *patchTableOut = patchTable;

    // append local points to stencils
    {
        if (Far::StencilTable const *vertexStencilsWithLocalPoints =
            Far::StencilTableFactory::AppendLocalPointStencilTable(
                *refiner,
                vertexStencils,
                patchTable->GetLocalPointStencilTable())) {
            delete vertexStencils;
            vertexStencils = vertexStencilsWithLocalPoints;
        }
        if (varyingStencils) {
            if (Far::StencilTable const *varyingStencilsWithLocalPoints =
                Far::StencilTableFactory::AppendLocalPointStencilTable(
                    *refiner,
                    varyingStencils,
                    patchTable->GetLocalPointVaryingStencilTable())) {
                delete varyingStencils;
                varyingStencils = varyingStencilsWithLocalPoints;
            }
        }
    }
    int numControlVertices = refiner->GetLevel(0).GetNumVertices();

    _stencilTableSize = createMeshRefiner(vertexStencils, varyingStencils,
                                          numControlVertices);
    // note: refiner takes ownership of vertexStencils, varyingStencils, patchTable

    delete refiner;
    return numControlVertices + vertexStencils->GetNumStencils();
}

int
SceneBase::AddObjects(int numObjects) {

    _objects.clear();

    int numTopologies = (int)_topologies.size();
    int vertsOffset = 0;

    for (int i = 0; i < numObjects; ++i) {

        Object obj;
        obj.topologyIndex = i % numTopologies;
        obj.vertsOffset = vertsOffset;
        _objects.push_back(obj);

        vertsOffset += _topologies[obj.topologyIndex].numVerts;
    }

    // invalidate batch
    for (int i = 0; i < (int)_batches.size(); ++i) {
        glDeleteBuffers(1, &_batches[i].dispatchBuffer);
    }
    _batches.clear();

    return vertsOffset;
}

size_t
SceneBase::CreateIndexBuffer() {
    if (_indexBuffer == 0) {
        glGenBuffers(1, &_indexBuffer);
    }

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _indexBuffer);
    std::vector<int> buffer;
    std::vector<unsigned int> ppBuffer;

    int numTopologies = (int)_topologies.size();
    for (int i = 0; i < numTopologies; ++i) {
        Far::PatchTable const *patchTable = _patchTables[i];

        int nPatchArrays = patchTable->GetNumPatchArrays();

        _topologies[i].patchArrays.clear();

        // for each patchArray
        for (int j = 0; j < nPatchArrays; ++j) {

            SceneBase::PatchArray patchArray;
            patchArray.desc = patchTable->GetPatchArrayDescriptor(j);
            patchArray.numPatches = patchTable->GetNumPatches(j);
            patchArray.indexOffset = (int)buffer.size();
            patchArray.primitiveIDOffset = (int)ppBuffer.size()/3;

            _topologies[i].patchArrays.push_back(patchArray);

            // indices
            Far::ConstIndexArray indices = patchTable->GetPatchArrayVertices(j);
            for (int k = 0; k < indices.size(); ++k) {
                buffer.push_back(indices[k]);
            }

            // patchParams
            Far::ConstPatchParamArray patchParams = patchTable->GetPatchParams(j);
            // XXX: needs sharpness interface for patcharray or put sharpness into patchParam.
            for (int k = 0; k < patchParams.size(); ++k) {
                float sharpness = 0.0;
                ppBuffer.push_back(patchParams[k].field0);
                ppBuffer.push_back(patchParams[k].field1);
                ppBuffer.push_back(*((unsigned int *)&sharpness));
            }
        }
#if 0
        // XXX: we'll remove below APIs from Far::PatchTable.
        //      use GetPatchParams(patchArray) instead as above.

        // patch param (all in one)
        Far::PatchParamTable const &patchParamTable =
            patchTable->GetPatchParamTable();
        std::vector<int> const &sharpnessIndexTable =
            patchTable->GetSharpnessIndexTable();
        std::vector<float> const &sharpnessValues =
            patchTable->GetSharpnessValues();

        int npatches = (int)patchParamTable.size();
        for (int i = 0; i < npatches; ++i) {
            float sharpness = 0.0;
            if (i < (int)sharpnessIndexTable.size()) {
                sharpness = sharpnessIndexTable[i] >= 0 ?
                    sharpnessValues[sharpnessIndexTable[i]] : 0.0f;
            }
            ppBuffer.push_back(patchParamTable[i].faceIndex);
            ppBuffer.push_back(patchParamTable[i].bitField.field);
            ppBuffer.push_back(*((unsigned int *)&sharpness));
        }
#endif
    }

    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 (int)buffer.size()*sizeof(int), &buffer[0], GL_STATIC_DRAW);

    // patchParam is currently expected to be texture (it can be SSBO)
    GLuint texBuffer = 0;
    glGenBuffers(1, &texBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, texBuffer);
    glBufferData(GL_ARRAY_BUFFER, ppBuffer.size()*sizeof(unsigned int),
                 &ppBuffer[0], GL_STATIC_DRAW);

    if (_patchParamTexture == 0) {
        glGenTextures(1, &_patchParamTexture);
    }
    glBindTexture(GL_TEXTURE_BUFFER, _patchParamTexture);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_RGB32I, texBuffer);
    glBindTexture(GL_TEXTURE_BUFFER, 0);

    glDeleteBuffers(1, &texBuffer);

    return buffer.size()*sizeof(int) + ppBuffer.size()*sizeof(int);
}

void
SceneBase::buildBatches() {

    int numObjects = (int)_objects.size();
    for (int i = 0; i < (int)_batches.size(); ++i) {
        glDeleteBuffers(1, &_batches[i].dispatchBuffer);
    }
    _batches.clear();

    typedef std::map<Far::PatchDescriptor, std::vector<int> > StagingBatches;
    StagingBatches stagingBatches;

    for (int i = 0; i < numObjects; ++i) {
        // get patchArrays from topology
        SceneBase::PatchArrayVector const &patchArrays =
            _topologies[_objects[i].topologyIndex].patchArrays;

        // for each patchArray:
        for (int j = 0; j < (int)patchArrays.size(); ++j) {
            SceneBase::PatchArray const &patchArray = patchArrays[j];

            // find batch for the descriptor
            std::vector<int> &command = stagingBatches[patchArray.desc];

            int nPatch = patchArray.numPatches;
            int baseVertex = GetVertsOffset(i);

            command.push_back(nPatch * patchArray.desc.GetNumControlVertices());
            command.push_back(1);
            command.push_back(patchArray.indexOffset);
            command.push_back(baseVertex);
            command.push_back(0);

            command.push_back(patchArray.primitiveIDOffset);
        }
    }
    int stride = sizeof(int)*6; // not 5, since we interleave primitiveIDOffset

    for (StagingBatches::iterator it = stagingBatches.begin();
         it != stagingBatches.end(); ++it) {

        Batch batch;
        glGenBuffers(1, &batch.dispatchBuffer);

        glBindBuffer(GL_DRAW_INDIRECT_BUFFER, batch.dispatchBuffer);
        glBufferData(GL_DRAW_INDIRECT_BUFFER,
                     it->second.size()*sizeof(int),
                     &it->second[0], GL_STATIC_DRAW);

        batch.desc = it->first;
        batch.count = (int)it->second.size()/6;
        batch.stride = stride;

        _batches.push_back(batch);
    }
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);
}
