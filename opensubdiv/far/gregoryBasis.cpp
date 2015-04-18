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

#include "../far/gregoryBasis.h"
#include "../far/error.h"
#include "../far/stencilTablesFactory.h"
#include "../far/topologyRefiner.h"

#include <cassert>
#include <cmath>
#include <cstring>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

// Builds a table of local indices pairs for each vertex of the patch.
//
//            o
//         N0 |
//            |              ....
//            |              .... : Gregory patch
//   o ------ o ------ o     ....
// N1       V | .... M3
//            | .......
//            | .......
//            o .......
//          N2
//
// [...] [N2 - N3] [...]
//
// Each value pair is composed of 2 index values in range [0-4[ pointing
// to the 2 neighbor vertices of the vertex 'V' belonging to the Gregory patch.
// Neighbor ordering is valence CCW and must match the winding of the 1-ring
// vertices.
//
static void
getQuadOffsets(Vtr::Level const & level, Vtr::Index fIndex,
    Vtr::Index offsets[], int fvarChannel=-1) {

    Far::ConstIndexArray fPoints = (fvarChannel<0) ?
        level.getFaceVertices(fIndex) :
            level.getFVarFaceValues(fIndex, fvarChannel);
    assert(fPoints.size()==4);

    for (int i = 0; i < 4; ++i) {

        Vtr::Index      vIndex = fPoints[i];
        Vtr::ConstIndexArray vFaces = level.getVertexFaces(vIndex),
                             vEdges = level.getVertexEdges(vIndex);

        int thisFaceInVFaces = -1;
        for (int j = 0; j < vFaces.size(); ++j) {
            if (fIndex == vFaces[j]) {
                thisFaceInVFaces = j;
                break;
            }
        }
        assert(thisFaceInVFaces != -1);

        // we have to use the number of incident edges to modulo the local index
        // because there could be 2 consecutive edges in the face belonging to
        // the Gregory patch.
        offsets[i*2+0] = thisFaceInVFaces;
        offsets[i*2+1] = (thisFaceInVFaces + 1)%vEdges.size();
    }
}

namespace Far {

void
GregoryBasisFactory::Point::Copy(int ** size, Index ** indices, float ** weights) const {
    memcpy(*indices, _indices, _size*sizeof(Index));
    memcpy(*weights, _weights, _size*sizeof(float));
    **size = _size;
    *indices += _size;
    *weights += _size;
    ++(*size);
}

//
// ProtoBasis
//
// Given a Vtr::Level and a face index, gathers all the influences of the 1-ring
// that supports the 20 CVs of a Gregory patch basis.
//
struct GregoryBasisFactory::ProtoBasis {

    ProtoBasis(Vtr::Level const & level, Index faceIndex, int fvarChannel=-1);

    int GetNumElements() const;

    void Copy(int * sizes, Index * indices, float * weights) const;

    // Control Vertices based on :
    // "Approximating Subdivision Surfaces with Gregory Patches for Hardware Tessellation"
    // Loop, Schaefer, Ni, Castafio (ACM ToG Siggraph Asia 2009)
    //
    //  P3         e3-      e2+         P2
    //     O--------O--------O--------O
    //     |        |        |        |
    //     |        |        |        |
    //     |        | f3-    | f2+    |
    //     |        O        O        |
    // e3+ O------O            O------O e2-
    //     |     f3+          f2-     |
    //     |                          |
    //     |                          |
    //     |      f0-         f1+     |
    // e0- O------O            O------O e1+
    //     |        O        O        |
    //     |        | f0+    | f1-    |
    //     |        |        |        |
    //     |        |        |        |
    //     O--------O--------O--------O
    //  P0         e0+      e1-         P1
    //

    Point P[4], Ep[4], Em[4], Fp[4], Fm[4];

    // for varying interpolation
    Point V[4];
};

static inline bool
checkMaxValence(Vtr::Level const & level) {
    if (level.getMaxValence()>GregoryBasisFactory::GetMaxValence()) {
        // The proto-basis closed-form table limits valence to 'MAX_VALENCE'
        Error(FAR_RUNTIME_ERROR,
            "Vertex valence %d exceeds maximum %d supported",
                level.getMaxValence(), GregoryBasisFactory::GetMaxValence());
        return false;
    }
    return true;
}


int
GregoryBasisFactory::ProtoBasis::GetNumElements() const {
    int nelems=0;
    for (int vid=0; vid<4; ++vid) {
        nelems += P[vid].GetSize();
        nelems += Ep[vid].GetSize();
        nelems += Em[vid].GetSize();
        nelems += Fp[vid].GetSize();
        nelems += Fm[vid].GetSize();
    }
    return nelems;
}
void
GregoryBasisFactory::ProtoBasis::Copy(int * sizes, Index * indices, float * weights) const {
    for (int vid=0; vid<4; ++vid) {
        P[vid].Copy(&sizes, &indices, &weights);
        Ep[vid].Copy(&sizes, &indices, &weights);
        Em[vid].Copy(&sizes, &indices, &weights);
        Fp[vid].Copy(&sizes, &indices, &weights);
        Fm[vid].Copy(&sizes, &indices, &weights);
    }
}
inline float csf(Index n, Index j) {
    if (j%2 == 0) {
        return cosf((2.0f * float(M_PI) * float(float(j-0)/2.0f))/(float(n)+3.0f));
    } else {
        return sinf((2.0f * float(M_PI) * float(float(j-1)/2.0f))/(float(n)+3.0f));
    }
}
GregoryBasisFactory::ProtoBasis::ProtoBasis(Vtr::Level const & level, Index faceIndex, int fvarChannel) {

    static float ef[MAX_VALENCE-3] = {
        0.812816f, 0.500000f, 0.363644f, 0.287514f,
        0.238688f, 0.204544f, 0.179229f, 0.159657f,
        0.144042f, 0.131276f, 0.120632f, 0.111614f,
        0.103872f, 0.09715f, 0.0912559f, 0.0860444f,
        0.0814022f, 0.0772401f, 0.0734867f, 0.0700842f,
        0.0669851f, 0.0641504f, 0.0615475f, 0.0591488f,
        0.0569311f, 0.0548745f, 0.0529621f
    };

    Vtr::ConstIndexArray facePoints = (fvarChannel<0) ?
        level.getFaceVertices(faceIndex) :
            level.getFVarFaceValues(faceIndex, fvarChannel);
    assert(facePoints.size()==4);

    int maxvalence = level.getMaxValence(),
        valences[4],
        zerothNeighbors[4];

    Index * manifoldRing = (int *)alloca((maxvalence+2)*2 * sizeof(int));

// Because MSVC does not support VLAs, we have to run alloca() in a macro and
// call in-place constructors - it's only been standardized for 15 years after
// all...
#define AllocaPointsArrays(variable, npoints) \
    Point * variable = (Point *)alloca(npoints*sizeof(Point)); \
    { for (int i=0; i<npoints; ++i) { new (&variable[i]) Point; } }

    AllocaPointsArrays(f, maxvalence);
    AllocaPointsArrays(r, maxvalence*4);
    Point e0[4], e1[4], org[4];

    for (int vid=0; vid<4; ++vid) {

        org[vid] = facePoints[vid];
        // save for varying stencils
        V[vid] = facePoints[vid];

        int ringSize =
            level.gatherQuadRegularRingAroundVertex(
                facePoints[vid], manifoldRing, fvarChannel);

        int valence;
        if (ringSize & 1) {
            // boundary vertex
            manifoldRing[ringSize] = manifoldRing[ringSize-1];
            ++ringSize;
            valence = -ringSize/2;
        } else {
            valence = ringSize/2;
        }
        int ivalence = abs(valence);
        valences[vid] = valence;

        Index boundaryEdgeNeighbors[2],
              currentNeighbor = 0,
              zerothNeighbor=0,
              ibefore=0;

        Point pos(facePoints[vid]);

        for (int i=0; i<ivalence; ++i) {

            Index im = (i+ivalence-1)%ivalence,
                  ip = (i+1)%ivalence;

            Index idx_neighbor = (manifoldRing[2*i + 0]),
                  idx_diagonal = (manifoldRing[2*i + 1]),
                  idx_neighbor_p = (manifoldRing[2*ip + 0]),
                  idx_neighbor_m = (manifoldRing[2*im + 0]),
                  idx_diagonal_m = (manifoldRing[2*im + 1]);

            bool boundaryNeighbor = (level.getVertexEdges(idx_neighbor).size() >
                level.getVertexFaces(idx_neighbor).size());

            if (fvarChannel>=0) {
                // XXXX manuelk need logic to check for boundary in fvar
                boundaryNeighbor = false;
            }

            if (boundaryNeighbor) {
                if (currentNeighbor<2) {
                    boundaryEdgeNeighbors[currentNeighbor] = idx_neighbor;
                }
                ++currentNeighbor;
                if (currentNeighbor==1) {
                    ibefore = zerothNeighbor = i;
                } else {
                    if (i-ibefore==1) {
                        std::swap(boundaryEdgeNeighbors[0], boundaryEdgeNeighbors[1]);
                        zerothNeighbor = i;
                    }
                }
            }

            Point neighbor(idx_neighbor),
                  diagonal(idx_diagonal),
                  neighbor_p(idx_neighbor_p),
                  neighbor_m(idx_neighbor_m),
                  diagonal_m(idx_diagonal_m);

            f[i] = (pos*float(ivalence) + (neighbor_p+neighbor)*2.0f + diagonal) / (float(ivalence)+5.0f);

            P[vid] += f[i];

            r[vid*maxvalence+i] = (neighbor_p-neighbor_m)/3.0f + (diagonal-diagonal_m)/6.0f;
        }

        P[vid] /= float(ivalence);

        zerothNeighbors[vid] = zerothNeighbor;
        if (currentNeighbor == 1) {
            boundaryEdgeNeighbors[1] = boundaryEdgeNeighbors[0];
        }

        for (int i=0; i<ivalence; ++i) {
            int im = (i+ivalence-1)%ivalence;
            Point e = (f[i]+f[im])*0.5f;
            e0[vid] += e * csf(ivalence-3, 2*i);
            e1[vid] += e * csf(ivalence-3, 2*i+1);
        }

        e0[vid] *= ef[ivalence-3];
        e1[vid] *= ef[ivalence-3];

        if (valence<0) {

            Point b0(boundaryEdgeNeighbors[0]),
                  b1(boundaryEdgeNeighbors[1]);

            if (ivalence>2) {
                P[vid] = (b0 + b1 + pos*4.0f)/6.0f;
            } else {
                P[vid] = pos;
            }
            float k = float(float(ivalence) - 1.0f);    //k is the number of faces
            float c = cosf(float(M_PI)/k);
            float s = sinf(float(M_PI)/k);
            float gamma = -(4.0f*s)/(3.0f*k+c);
            float alpha_0k = -((1.0f+2.0f*c)*sqrtf(1.0f+c))/((3.0f*k+c)*sqrtf(1.0f-c));
            float beta_0 = s/(3.0f*k + c);

            Point diagonal(manifoldRing[2*zerothNeighbor + 1]);

            e0[vid] = (b0 - b1)/6.0f;
            e1[vid] = pos*gamma + diagonal*beta_0 + (b0 + b1)*alpha_0k;

            for (int x=1; x<ivalence-1; ++x) {

                Index curri = ((x + zerothNeighbor)%ivalence);

                float alpha = (4.0f*sinf((float(M_PI) * float(x))/k))/(3.0f*k+c),
                      beta = (sinf((float(M_PI) * float(x))/k) + sinf((float(M_PI) * float(x+1))/k))/(3.0f*k+c);

                Index idx_neighbor = manifoldRing[2*curri + 0],
                      idx_diagonal = manifoldRing[2*curri + 1];

                Point p_neighbor(idx_neighbor),
                      p_diagonal(idx_diagonal);

                e1[vid] += p_neighbor*alpha + p_diagonal*beta;
            }
            e1[vid] /= 3.0f;
        }
    }

    Index quadOffsets[8];
    getQuadOffsets(level, faceIndex, quadOffsets, fvarChannel);

    for (int vid=0; vid<4; ++vid) {

        int n = abs(valences[vid]),
            ivalence = n;

        int ip = (vid+1)%4,
            im = (vid+3)%4,
            np = abs(valences[ip]),
            nm = abs(valences[im]);

        Index start = quadOffsets[vid*2+0],
              prev = quadOffsets[vid*2+1],
              start_m = quadOffsets[im*2],
              prev_p = quadOffsets[ip*2+1];

        Point Em_ip, Ep_im;

        if (valences[ip]<-2) {
            Index j = (np + prev_p - zerothNeighbors[ip]) % np;
            Em_ip = P[ip] + e0[ip]*cosf((float(M_PI)*j)/float(np-1)) + e1[ip]*sinf((float(M_PI)*j)/float(np-1));
        } else {
            Em_ip = P[ip] + e0[ip]*csf(np-3,2*prev_p) + e1[ip]*csf(np-3,2*prev_p+1);
        }

        if (valences[im]<-2) {
            Index j = (nm + start_m - zerothNeighbors[im]) % nm;
            Ep_im = P[im] + e0[im]*cosf((float(M_PI)*j)/float(nm-1)) + e1[im]*sinf((float(M_PI)*j)/float(nm-1));
        } else {
            Ep_im = P[im] + e0[im]*csf(nm-3,2*start_m) + e1[im]*csf(nm-3,2*start_m+1);
        }

        if (valences[vid] < 0) {
            n = (n-1)*2;
        }
        if (valences[im] < 0) {
            nm = (nm-1)*2;
        }
        if (valences[ip] < 0) {
            np = (np-1)*2;
        }

        Point const * rp = &r[vid*maxvalence];

        if (valences[vid] > 2) {

            float s1 = 3.0f - 2.0f*csf(n-3,2)-csf(np-3,2),
                  s2 = 2.0f*csf(n-3,2),
                  s3 = 3.0f -2.0f*cosf(2.0f*float(M_PI)/float(n)) - cosf(2.0f*float(M_PI)/float(nm));

            Ep[vid] = P[vid] + e0[vid]*csf(n-3, 2*start) + e1[vid]*csf(n-3, 2*start +1);
            Em[vid] = P[vid] + e0[vid]*csf(n-3, 2*prev ) + e1[vid]*csf(n-3, 2*prev + 1);
            Fp[vid] = (P[vid]*csf(np-3,2) + Ep[vid]*s1 + Em_ip*s2 + rp[start])/3.0f;
            Fm[vid] = (P[vid]*csf(nm-3,2) + Em[vid]*s3 + Ep_im*s2 - rp[prev])/3.0f;

        } else if (valences[vid] < -2) {

            Index jp = (ivalence + start - zerothNeighbors[vid]) % ivalence,
                  jm = (ivalence + prev  - zerothNeighbors[vid]) % ivalence;

            float s1 = 3-2*csf(n-3,2)-csf(np-3,2),
                  s2 = 2*csf(n-3,2),
                  s3 = 3.0f-2.0f*cosf(2.0f*float(M_PI)/n)-cosf(2.0f*float(M_PI)/nm);

            Ep[vid] = P[vid] + e0[vid]*cosf((float(M_PI)*jp)/float(ivalence-1)) + e1[vid]*sinf((float(M_PI)*jp)/float(ivalence-1));
            Em[vid] = P[vid] + e0[vid]*cosf((float(M_PI)*jm)/float(ivalence-1)) + e1[vid]*sinf((float(M_PI)*jm)/float(ivalence-1));
            Fp[vid] = (P[vid]*csf(np-3,2) + Ep[vid]*s1 + Em_ip*s2 + rp[start])/3.0f;
            Fm[vid] = (P[vid]*csf(nm-3,2) + Em[vid]*s3 + Ep_im*s2 - rp[prev])/3.0f;

            if (valences[im]<0) {
                s1=3-2*csf(n-3,2)-csf(np-3,2);
                Fp[vid] = Fm[vid] = (P[vid]*csf(np-3,2) + Ep[vid]*s1 + Em_ip*s2 + rp[start])/3.0f;
            } else if (valences[ip]<0) {
                s1 = 3.0f-2.0f*cosf(2.0f*float(M_PI)/n)-cosf(2.0f*float(M_PI)/nm);
                Fm[vid] = Fp[vid] = (P[vid]*csf(nm-3,2) + Em[vid]*s1 + Ep_im*s2 - rp[prev])/3.0f;
            }

        } else if (valences[vid]==-2) {

            Ep[vid] = (org[vid]*2.0f + org[ip])/3.0f;
            Em[vid] = (org[vid]*2.0f + org[im])/3.0f;
            Fp[vid] = Fm[vid] = (org[vid]*4.0f + org[((vid+2)%n)] + org[ip]*2.0f + org[im]*2.0f)/9.0f;
        }
    }
}

int GregoryBasisFactory::GetMaxValence() {
    return MAX_VALENCE;
}

//
// Stateless GregoryBasisFactory
//
GregoryBasis const *
GregoryBasisFactory::Create(TopologyRefiner const & refiner,
    Index faceIndex, int fvarChannel) {

    // Gregory patches are end-caps: they only exist on max-level
    Vtr::Level const & level = refiner.getLevel(refiner.GetMaxLevel());

    if (not checkMaxValence(level)) {
        return 0;
    }

    ProtoBasis basis(level, faceIndex, fvarChannel);

    int nelems= basis.GetNumElements();

    GregoryBasis * result = new GregoryBasis;

    result->_indices.resize(nelems);
    result->_weights.resize(nelems);

    basis.Copy(result->_sizes, &result->_indices[0], &result->_weights[0]);

    // note: this function doesn't create varying stencils.

    for (int i=0, offset=0; i<20; ++i) {
        result->_offsets[i] = offset;
        offset += result->_sizes[i];
    }

    return result;
}

//
// GregoryBasisFactory for Vertex StencilTables
//
GregoryBasisFactory::GregoryBasisFactory(TopologyRefiner const & refiner,
                                         bool shareBoundaryVertices) :
    _refiner(refiner), _maxvalence(0), _shareBoundaryVertices(shareBoundaryVertices),
    numGregoryBasisVertices(0), numGregoryBasisPatches(0) {

    // Sanity check: the mesh must be adaptively refined
    assert(not _refiner.IsUniform());

    _maxvalence = _refiner.GetMaxValence();

    // would be great to reserve _pointArray here to something
}

void
GregoryBasisFactory::factorizeBasisVertex(StencilTables const * stencils,
                                          GregoryBasisFactory::Point const & p,
                                          ProtoStencil dst) {
    // Use the Allocator to factorize the Gregory patch influence CVs with the
    // supporting CVs from the stencil tables.
    if (!stencils) return;

    dst.Clear();
    for (int j=0; j<p.GetSize(); ++j) {
        dst.AddWithWeight(*stencils,
            p.GetIndices()[j], p.GetWeights()[j]);
    }
}

bool
GregoryBasisFactory::addPatchBasis(Index faceIndex,
                                   bool verticesMask[4][5]) {

    // Gregory patches only exist on the hight
    Vtr::Level const & level = _refiner.getLevel(_refiner.GetMaxLevel());

    if (not checkMaxValence(level)) {
        return false;
    }

    // Gather the CVs that influence the Gregory patch and their relative
    // weights in a basis
    ProtoBasis basis(level, faceIndex);

    for (int i = 0; i < 4; ++i) {
        if (verticesMask[i][0]) {
            _vertexStencils.push_back(basis.P[i]);
            _varyingStencils.push_back(basis.V[i]);
        }
        if (verticesMask[i][1]) {
            _vertexStencils.push_back(basis.Ep[i]);
            _varyingStencils.push_back(basis.V[i]);
        }
        if (verticesMask[i][2]) {
            _vertexStencils.push_back(basis.Em[i]);
            _varyingStencils.push_back(basis.V[i]);
        }
        if (verticesMask[i][3]) {
            _vertexStencils.push_back(basis.Fp[i]);
            _varyingStencils.push_back(basis.V[i]);
        }
        if (verticesMask[i][4]) {
            _vertexStencils.push_back(basis.Fm[i]);
            _varyingStencils.push_back(basis.V[i]);
        }
    }
    return true;
}
StencilTables const *
GregoryBasisFactory::createStencilTables(StencilTables const *baseStencils,
                                         bool append,
                                         std::vector<Point> const &gregoryStencils,
                                         int const permute[20]) {

    // Gregory limit stencils have indices that are relative to the level
    // (maxlevel) of subdivision. These indices need to be offset to match
    // the indices from the multi-level adaptive stencil tables.
    // In addition: stencil tables can be built with singular stencils
    // (single weight of 1.0f) as place-holders for coarse mesh vertices,
    // which also needs to be accounted for.
    int stencilsIndexOffset = 0;
    {
        int maxlevel = _refiner.GetMaxLevel();
        int nverts = _refiner.GetNumVerticesTotal();
        int nBaseStencils = baseStencils->GetNumStencils();
        if (nBaseStencils == nverts) {

            // the table contain stencils for the control vertices
            stencilsIndexOffset = nverts - _refiner.GetNumVertices(maxlevel);

        } else if (nBaseStencils == (nverts -_refiner.GetNumVertices(0))) {

            // the table does not contain stencils for the control vertices
            stencilsIndexOffset = nverts - _refiner.GetNumVertices(maxlevel)
                - _refiner.GetNumVertices(0);

        } else {
            // these are not the stencils you are looking for...
            assert(0);
            return NULL;
        }
    }

    int nStencils = (int)gregoryStencils.size();

    // Factorize the basis CVs with the stencil tables: the basis is now
    // expressed as a linear combination of vertices from the coarse control
    // mesh with no data dependencies
    StencilAllocator alloc(getNumMaxElements(_maxvalence));

    alloc.Resize(nStencils);
    for (int i = 0; i < nStencils; ++i) {
        Point p = gregoryStencils[i];
        p.OffsetIndices(stencilsIndexOffset);

        factorizeBasisVertex(baseStencils, p, alloc[i]);
    }

    int nelems = alloc.GetNumVerticesTotal();

    if (nStencils==0 or nelems==0) {
        return NULL;
    }

    // Finalize the stencil tables from the temporary pool allocator
    StencilTables * result = new StencilTables;

    result->_numControlVertices = _refiner.GetNumVertices(0);

    result->resize(nStencils, nelems);

    Stencil dst(&result->_sizes.at(0),
        &result->_indices.at(0), &result->_weights.at(0));

    for (int i = 0; i < nStencils; ++i) {

        Index index = i;
        if (permute) {
            int localIndex = i % 20,
                baseIndex = i - localIndex;
            index = baseIndex + permute[localIndex];
        }

        *dst._size = alloc.CopyStencil(index, dst._indices, dst._weights);

        dst.Next();
    }

    result->generateOffsets();

    if (append) {
        StencilTables const *inStencilTables[] = {
            baseStencils, result
        };
        StencilTables const *concatStencilTables =
            StencilTablesFactory::Create(2, inStencilTables);
        delete result;
        return concatStencilTables;
    } else {
        return result;
    }
}

PatchDescriptor::Type
GregoryBasisFactory::GetPatchType(
    PatchTablesFactory::PatchFaceTag const &) const {

    return PatchDescriptor::GREGORY_BASIS;
}

//
// Populates the topology table used by Gregory-basis patches
//
// Note  : 'faceIndex' values are expected to be sorted in ascending order !!!
// Note 2: this code attempts to identify basis vertices shared along
//         gregory patch edges
ConstIndexArray
GregoryBasisFactory::GetTopology(Vtr::Level const& level, Index faceIndex,
                                 PatchTablesFactory::PatchFaceTag const * levelPatchTags,
                                 int /*not used: levelVertsOffset*/)
{
    // allocate indices (awkward)
    // assert(Vtr::INDEX_INVALID==0xFFFFFFFF);
    for (int i = 0; i < 20; ++i) {
        topology.push_back(Vtr::INDEX_INVALID);
    }
    Index * dest = &topology[numGregoryBasisPatches*20];

    ConstIndexArray fedges = level.getFaceEdges(faceIndex);
    assert(fedges.size()==4);

    int gregoryVertexOffset = _refiner.GetNumVerticesTotal();

    if (_shareBoundaryVertices) {
        for (int i=0; i<4; ++i) {
            Index edge = fedges[i], adjface = 0;

            { // Gather adjacent faces
                ConstIndexArray adjfaces = level.getEdgeFaces(edge);
                for (int i=0; i<adjfaces.size(); ++i) {
                    if (adjfaces[i]==faceIndex) {
                        // XXXX manuelk if 'edge' is non-manifold, arbitrarily pick the
                        // next face in the list of adjacent faces
                        adjface = (adjfaces[(i+1)%adjfaces.size()]);
                        break;
                    }
                }
            }
            // We are looking for adjacent faces that:
            // - exist (no boundary)
            // - have already been processed (known CV indices)
            // - are also Gregory basis patches
            if (adjface!=Vtr::INDEX_INVALID and (adjface < faceIndex) and
                (not levelPatchTags[adjface]._isRegular)) {

                ConstIndexArray aedges = level.getFaceEdges(adjface);
                int aedge = aedges.FindIndexIn4Tuple(edge);
                assert(aedge!=Vtr::INDEX_INVALID);

                // Find index of basis in the list of basis already generated
                struct compare {
                    static int op(void const * a, void const * b) {
                        return *(Index *)a - *(Index *)b;
                    }
                };

                Index * ptr = (Index *)std::bsearch( &adjface, &basisIndices[0],
                                                     basisIndices.size(), sizeof(Index), compare::op);

                int srcBasisIdx = (int)(ptr - &basisIndices[0]);

                if (!ptr) {
                    // if the adjface is hole, it won't be found
                    break;
                }
                assert(ptr and srcBasisIdx>=0 and srcBasisIdx<(int)basisIndices.size());

                // Copy the indices of CVs from the face on the other side of the
                // shared edge
                static int const gregoryEdgeVerts[4][4] = { { 0,  1,  7,  5},
                                                            { 5,  6, 12, 10},
                                                            {10, 11, 17, 15},
                                                            {15, 16,  2,  0} };
                Index * src = &topology[srcBasisIdx*20];
                for (int j=0; j<4; ++j) {
                    // invert direction
                    // note that src  indices have already been offsetted.
                    dest[gregoryEdgeVerts[i][3-j]] = src[gregoryEdgeVerts[aedge][j]];
                }
            }
        }
    }

    bool newVerticesMask[4][5];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 5; ++j) {
            if (dest[i*5+j]==Vtr::INDEX_INVALID) {
                // assign new vertex
                dest[i*5+j] = numGregoryBasisVertices+ gregoryVertexOffset;
                ++numGregoryBasisVertices;
                newVerticesMask[i][j] = true;
            } else {
                // share vertex
                newVerticesMask[i][j] = false;
            }
        }
    }
    basisIndices.push_back(faceIndex);

    // add basis
    addPatchBasis(faceIndex, newVerticesMask);

    ++numGregoryBasisPatches;

    // return cvs;
    return ConstIndexArray(dest, 20);
}

} // end namespace Far

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv
