//
//     Copyright (C) Pixar. All rights reserved.
//
//     This license governs use of the accompanying software. If you
//     use the software, you accept this license. If you do not accept
//     the license, do not use the software.
//
//     1. Definitions
//     The terms "reproduce," "reproduction," "derivative works," and
//     "distribution" have the same meaning here as under U.S.
//     copyright law.  A "contribution" is the original software, or
//     any additions or changes to the software.
//     A "contributor" is any person or entity that distributes its
//     contribution under this license.
//     "Licensed patents" are a contributor's patent claims that read
//     directly on its contribution.
//
//     2. Grant of Rights
//     (A) Copyright Grant- Subject to the terms of this license,
//     including the license conditions and limitations in section 3,
//     each contributor grants you a non-exclusive, worldwide,
//     royalty-free copyright license to reproduce its contribution,
//     prepare derivative works of its contribution, and distribute
//     its contribution or any derivative works that you create.
//     (B) Patent Grant- Subject to the terms of this license,
//     including the license conditions and limitations in section 3,
//     each contributor grants you a non-exclusive, worldwide,
//     royalty-free license under its licensed patents to make, have
//     made, use, sell, offer for sale, import, and/or otherwise
//     dispose of its contribution in the software or derivative works
//     of the contribution in the software.
//
//     3. Conditions and Limitations
//     (A) No Trademark License- This license does not grant you
//     rights to use any contributor's name, logo, or trademarks.
//     (B) If you bring a patent claim against any contributor over
//     patents that you claim are infringed by the software, your
//     patent license from such contributor to the software ends
//     automatically.
//     (C) If you distribute any portion of the software, you must
//     retain all copyright, patent, trademark, and attribution
//     notices that are present in the software.
//     (D) If you distribute any portion of the software in source
//     code form, you may do so only under this license by including a
//     complete copy of this license with your distribution. If you
//     distribute any portion of the software in compiled or object
//     code form, you may only do so under a license that complies
//     with this license.
//     (E) The software is licensed "as-is." You bear the risk of
//     using it. The contributors give no express warranties,
//     guarantees or conditions. You may have additional consumer
//     rights under your local laws which this license cannot change.
//     To the extent permitted under your local laws, the contributors
//     exclude the implied warranties of merchantability, fitness for
//     a particular purpose and non-infringement.
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
    self->vertexBuffer->UpdateData(pFloats, numFloats);
}

void
shim::Subdivider::refine()
{
    g_osdComputeController->Refine(self->computeContext, self->vertexBuffer);
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
    const std::vector<int> &indices = self->farMesh->GetFaceVertices(level-1);
    int numInts = (int) indices.size();

    unsigned char* srcBegin = (unsigned char*) &indices[0];
    unsigned char* srcEnd = srcBegin + numInts * 4;
    refinedQuads->assign(srcBegin, srcEnd);
}
