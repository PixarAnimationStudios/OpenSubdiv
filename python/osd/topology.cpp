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
