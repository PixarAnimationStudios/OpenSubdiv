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

    size_t numFloatsPerVertex = 0;
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
