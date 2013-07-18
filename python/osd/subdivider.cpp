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

#include "internal.h"
#include "subdivider.h"
#include "topology.h"

using namespace std;
using namespace shim;

OpenSubdiv::OsdCpuComputeController* g_osdComputeController = 0;

shim::Subdivider::Subdivider(
    const Topology& topo,
    Layout refinedLayout,
    DataType refinedIndexType,
    int subdivisionLevels)
{
    self = new SubdividerImpl();

    if (!g_osdComputeController) {
        g_osdComputeController = new OpenSubdiv::OsdCpuComputeController();
    }

    int numFloatsPerVertex = 0;
    Layout::const_iterator it;
    for (it = refinedLayout.begin(); it != refinedLayout.end(); ++it) {
        if (*it != float32) {
            cerr << "Unsupported vertex type." << endl;
            break;
        }
        ++numFloatsPerVertex;
    }
    OpenSubdiv::FarMeshFactory<OpenSubdiv::OsdVertex> meshFactory(
        topo.self->hmesh,
        subdivisionLevels);
    self->farMesh = meshFactory.Create();
    self->computeContext = OpenSubdiv::OsdCpuComputeContext::Create(
        self->farMesh);

    self->vertexBuffer = OpenSubdiv::OsdCpuVertexBuffer::Create(
        numFloatsPerVertex, self->farMesh->GetNumVertices());
}

shim::Subdivider::~Subdivider()
{
    delete self->computeContext;
    delete self->farMesh;
    delete self->vertexBuffer;
    delete self;
}

void
shim::Subdivider::setCoarseVertices(const HeterogeneousBuffer& cage)
{
    float* pFloats = (float*) &cage.Buffer[0];
    int numFloats = cage.Buffer.size() / sizeof(float);
    self->vertexBuffer->UpdateData(pFloats, /*start vertex*/ 0, numFloats);
}

void
shim::Subdivider::refine()
{
    g_osdComputeController->Refine(self->computeContext,
                                   self->farMesh->GetKernelBatches(),
                                   self->vertexBuffer);
}

void
shim::Subdivider::getRefinedVertices(Buffer* refinedVertices)
{
    float* pFloats = self->vertexBuffer->BindCpuBuffer();

    int numFloats = self->vertexBuffer->GetNumElements() * 
        self->vertexBuffer->GetNumVertices();

    unsigned char* srcBegin = (unsigned char*) pFloats;
    unsigned char* srcEnd = srcBegin + numFloats * 4;
    refinedVertices->assign(srcBegin, srcEnd);
}

void
shim::Subdivider::getRefinedQuads(Buffer* refinedQuads)
{
    OpenSubdiv::FarPatchTables const * patchTables =
        self->farMesh->GetPatchTables();

    if (patchTables) {
        cerr << "Feature adaptive not supported" << endl;
        return;
    }

    const OpenSubdiv::FarSubdivisionTables<OpenSubdiv::OsdVertex> *tables =
        self->farMesh->GetSubdivisionTables();

    bool loop = dynamic_cast<const OpenSubdiv::FarLoopSubdivisionTables<
        OpenSubdiv::OsdVertex>*>(tables);

    if (loop) {
        cerr << "loop subdivision not supported" << endl;
        return;
    }

    int level = tables->GetMaxLevel();
    unsigned int const * indices = self->farMesh->GetPatchTables()->GetFaceVertices(level-1);
    int numInts = self->farMesh->GetPatchTables()->GetNumFaces(level-1)*4;

    unsigned char const * srcBegin = reinterpret_cast<unsigned char const *>(indices);
    unsigned char const * srcEnd = srcBegin + numInts * 4;
    refinedQuads->assign(srcBegin, srcEnd);
}
