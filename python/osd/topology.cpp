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
#include "topology.h"

using namespace shim;
using namespace std;

static OpenSubdiv::HbrCatmarkSubdivision<OpenSubdiv::OsdVertex>  _catmark;

shim::Topology::Topology(
    const HomogeneousBuffer& indices, const HomogeneousBuffer& valences)
{
    self = new TopologyImpl();
    self->hmesh = new OsdHbrMesh(&_catmark);

    size_t maxIndex = 0;
    size_t byteCount = indices.Buffer.size();

    switch (indices.Type) {
    case int32: {
        const int *d = (const int*) &indices.Buffer[0];
        maxIndex = (size_t) *max_element(d, d + byteCount / 4);
        break;
    }
    default:
        cerr << "Unsupported index type " << indices.Type << endl;
    };

    size_t maxValence = 0;
    switch (valences.Type) {
    case uint8: {
        const unsigned char *d = (const unsigned char*) &valences.Buffer[0];
        maxValence = (size_t) *max_element(d, d + byteCount);
        break;
    }
    default:
        cerr << "Unsupported valence type " << valences.Type << endl;
    };

    self->numVertices = 1 + (int) maxIndex;
    OpenSubdiv::OsdVertex vert;
    for (size_t i = 0; i < self->numVertices; ++i) {
        OpenSubdiv::HbrVertex<OpenSubdiv::OsdVertex>* pVert =
            self->hmesh->NewVertex((int) i, vert); 
        if (!pVert) {
            cerr << "Error: Unable to create vertex " << i << endl;
        }
    }

    int* pIndices = (int*) &indices.Buffer[0];
    unsigned char* pValence = (unsigned char*) &valences.Buffer[0];
    size_t valenceCount = valences.Buffer.size();
    while (valenceCount--) {
        int vertsPerFace = *pValence;
        OpenSubdiv::HbrFace<OpenSubdiv::OsdVertex>* pFace =
            self->hmesh->NewFace(vertsPerFace, pIndices, 0);
        if (!pFace) {
            cerr << "Error: Unable to create face (valence = "
                 << vertsPerFace << ")\n";
        }
        pIndices += vertsPerFace;
        ++pValence;
    }

    self->hmesh->GetFaces(back_inserter(self->faces));
}

shim::Topology::~Topology()
{
    delete self->hmesh;
    delete self;
}

void
shim::Topology::copyAnnotationsFrom(const Topology& topo)
{
    int vertexCount = getNumVertices();
    for (int i = 0; i < vertexCount; ++i) {
        float s = topo.getVertexSharpness(i);
        setVertexSharpness(i, s);
    }

    int faceCount = getNumFaces();
    for (int faceIndex = 0; faceIndex < faceCount; ++faceIndex) {
        int numEdges = topo.getNumEdges(faceIndex);
        for (int edgeIndex = 0; edgeIndex < numEdges; ++edgeIndex) {
            float s = topo.getEdgeSharpness(faceIndex, edgeIndex);
            setEdgeSharpness(faceIndex, edgeIndex, s);
        }
    }
}

void
shim::Topology::finalize()
{
    self->hmesh->Finish();
}

BoundaryMode::e
shim::Topology::getBoundaryMode() const
{
    OsdHbrMesh::InterpolateBoundaryMethod bm = 
        self->hmesh->GetInterpolateBoundaryMethod();
    switch (bm) {
    case OsdHbrMesh::k_InterpolateBoundaryNone:
        return BoundaryMode::NONE;
    case OsdHbrMesh::k_InterpolateBoundaryEdgeOnly:
        return BoundaryMode::EDGE_ONLY;
    case OsdHbrMesh::k_InterpolateBoundaryEdgeAndCorner:
        return BoundaryMode::EDGE_AND_CORNER;
    case OsdHbrMesh::k_InterpolateBoundaryAlwaysSharp:
        return BoundaryMode::ALWAYS_SHARP;
    }
    throw("Bad interpolation method.");
}

void
shim::Topology::setBoundaryMode(BoundaryMode::e bm)
{
    switch (bm) {
    case BoundaryMode::NONE:
        self->hmesh->SetInterpolateBoundaryMethod(
            OsdHbrMesh::k_InterpolateBoundaryNone);
        break;
    case BoundaryMode::EDGE_ONLY:
        self->hmesh->SetInterpolateBoundaryMethod(
            OsdHbrMesh::k_InterpolateBoundaryEdgeOnly);
        break;
    case BoundaryMode::EDGE_AND_CORNER:
        self->hmesh->SetInterpolateBoundaryMethod(
            OsdHbrMesh::k_InterpolateBoundaryEdgeAndCorner);
        break;
    case BoundaryMode::ALWAYS_SHARP:
        self->hmesh->SetInterpolateBoundaryMethod(
            OsdHbrMesh::k_InterpolateBoundaryAlwaysSharp);
        break;
    }
}

int
shim::Topology::getNumVertices() const
{
    return (int) self->numVertices;
}

float
shim::Topology::getVertexSharpness(int vertex) const
{
    return self->hmesh->GetVertex(vertex)->GetSharpness();
}

void
shim::Topology::setVertexSharpness(int vertex, float sharpness)
{
    self->hmesh->GetVertex(vertex)->SetSharpness(sharpness);
}

int
shim::Topology::getNumFaces() const
{
    return self->hmesh->GetNumFaces();
}

bool
shim::Topology::getFaceHole(int faceIndex) const
{
    return self->faces[faceIndex]->IsHole();
}

void
shim::Topology::setFaceHole(int faceIndex, bool isHole)
{
    self->faces[faceIndex]->SetHole(isHole);
}

int
shim::Topology::getNumEdges(int faceIndex) const
{
    return self->faces[faceIndex]->GetNumVertices();
}

float
shim::Topology::getEdgeSharpness(int faceIndex, int edgeIndex) const
{
    return self->faces[faceIndex]->GetEdge(edgeIndex)->GetSharpness();
}

void
shim::Topology::setEdgeSharpness(int faceIndex, int edgeIndex, float sharpness)
{
    self->faces[faceIndex]->GetEdge(edgeIndex)->SetSharpness(sharpness);
}
