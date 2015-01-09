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
getQuadOffsets(Vtr::Level const& level, Vtr::Index fIndex, Vtr::Index offsets[]) {

    Far::ConstIndexArray fVerts = level.getFaceVertices(fIndex);
    assert(fVerts.size()==4);

    for (int i = 0; i < 4; ++i) {

        Vtr::Index      vIndex = fVerts[i];
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

#define GetNumMaxElems( maxvalence ) \
    16 + maxvalence - 3

// limit valence of 30 because we use a pre-computed closed-form 'ef' table
static const int MAX_VALENCE=30,
                 MAX_ELEMS = GetNumMaxElems(MAX_VALENCE);

namespace Far {

//
// Basis point
//
// Implements arithmetic operators to manipulate the influence of the
// 1-ring control vertices supporting the patch basis
//
class Point {

public:

    Point() : _size(0) { }

    Point(Index idx, float weight = 1.0f) {
        _size = 1;
        _indices[0] = idx;
        _weights[0] = weight;
    }

    Point(Point const & other) {
        *this = other;
    }

    int GetSize() const {
        return _size;
    }

    Index const * GetIndices() const {
        return _indices;
    }

    float const * GetWeights() const {
        return _weights;
    }

    Point & operator = (Point const & other) {
        _size = other._size;
        memcpy(_indices, other._indices, other._size*sizeof(Index));
        memcpy(_weights, other._weights, other._size*sizeof(float));
        return *this;
    }

    Point & operator += (Point const & other) {
        for (int i=0; i<other._size; ++i) {
            Index idx = findIndex(other._indices[i]);
            _weights[idx] += other._weights[i];
        }
        return *this;
    }

    Point & operator -= (Point const & other) {
        for (int i=0; i<other._size; ++i) {
            Index idx = findIndex(other._indices[i]);
            _weights[idx] -= other._weights[i];
        }
        return *this;
    }

    Point & operator *= (float f) {
        for (int i=0; i<_size; ++i) {
            _weights[i] *= f;
        }
        return *this;
    }

    Point & operator /= (float f) {
       return (*this)*=(1.0f/f);
    }

    friend Point operator * (Point const & src, float f) {
        Point p( src ); return p*=f;
    }

    friend Point operator / (Point const & src, float f) {
        Point p( src ); return p*= (1.0f/f);
    }

    Point operator + (Point const & other) {
        Point p(*this); return p+=other;
    }

    Point operator - (Point const & other) {
        Point p(*this); return p-=other;
    }

    void OffsetIndices(Index offset) {
        for (int i=0; i<_size; ++i) {
            _indices[i] += offset;
        }
    }

    void Copy(int ** size, Index ** indices, float ** weights) const;

private:

    int findIndex(Index idx) {
        for (int i=0; i<_size; ++i) {
            if (_indices[i]==idx) {
                return i;
            }
        }
        _indices[_size]=idx;
        _weights[_size]=0.0f;
        ++_size;
        return _size-1;
    }

    int _size;
    // XXXX this would really be better with VLA where we only allocate
    // space based on the max vertex valence in the mesh, not the
    // absolute maximum supported by the closed-form tangents table.
    Index _indices[MAX_ELEMS];
    float _weights[MAX_ELEMS];
};

void
Point::Copy(int ** size, Index ** indices, float ** weights) const {
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
struct ProtoBasis {

    ProtoBasis(Vtr::Level const & level, Index faceIndex);

    int GetNumElements() const;

    void OffsetIndices(Index offset);

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
};

int
ProtoBasis::GetNumElements() const {
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
ProtoBasis::OffsetIndices(Index offset) {
    for (int vid=0; vid<4; ++vid) {
        P[vid].OffsetIndices(offset);
        Ep[vid].OffsetIndices(offset);
        Em[vid].OffsetIndices(offset);
        Fp[vid].OffsetIndices(offset);
        Fm[vid].OffsetIndices(offset);
    }
}
void
ProtoBasis::Copy(int * sizes, Index * indices, float * weights) const {
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
ProtoBasis::ProtoBasis(Vtr::Level const & level, Index faceIndex) {

    static float ef[MAX_VALENCE-3] = {
        0.812816f, 0.500000f, 0.363644f, 0.287514f,
        0.238688f, 0.204544f, 0.179229f, 0.159657f,
        0.144042f, 0.131276f, 0.120632f, 0.111614f,
        0.103872f, 0.09715f, 0.0912559f, 0.0860444f,
        0.0814022f, 0.0772401f, 0.0734867f, 0.0700842f,
        0.0669851f, 0.0641504f, 0.0615475f, 0.0591488f,
        0.0569311f, 0.0548745f, 0.0529621f
    };

    Vtr::ConstIndexArray faceVerts = level.getFaceVertices(faceIndex);
    assert(faceVerts.size()==4);

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

        org[vid] = faceVerts[vid];

        int ringSize = level.gatherManifoldVertexRingFromIncidentQuads(faceVerts[vid], 0, manifoldRing),
            valence;
        if (ringSize & 1) {
            // boundary vertex
            ++ringSize;
            manifoldRing[ringSize] = manifoldRing[ringSize-1];
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

        Point pos(faceVerts[vid]);

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
    getQuadOffsets(level, faceIndex, quadOffsets);

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
GregoryBasisFactory::Create(TopologyRefiner const & refiner, Index faceIndex) {

    // Gregory patches are end-caps: they only exist on max-level
    Vtr::Level const & level = refiner.getLevel(refiner.GetMaxLevel());

    if (level.getMaxValence()>GetMaxValence()) {
        // The proto-basis closed-form table limits valence to 'MAX_VALENCE'
        return 0;
    }

    ProtoBasis basis(level, faceIndex);

    int nelems= basis.GetNumElements();

    GregoryBasis * result = new GregoryBasis;

    result->_indices.resize(nelems);
    result->_weights.resize(nelems);

    basis.Copy(result->_sizes, &result->_indices[0], &result->_weights[0]);

    for (int i=0, offset=0; i<20; ++i) {
        result->_offsets[i] = offset;
        offset += result->_sizes[i];
    }

    return result;
}

//
// GregoryBasisFactory for StencilTables
//
GregoryBasisFactory::GregoryBasisFactory(TopologyRefiner const & refiner,
    StencilTables const & stencils, int numpatches, int maxvalence) :
        _currentStencil(0), _refiner(refiner),
            _stencils(stencils), _alloc(GetNumMaxElems(maxvalence)) {

    // Sanity check: the mesh must be adaptively refined
    assert(not _refiner.IsUniform());

    _alloc.Resize(numpatches * 20);

    // Gregory limit stencils have indices that are relative to the level
    // (maxlevel) of subdivision. These indices need to be offset to match
    // the indices from the multi-level adaptive stencil tables.
    // In addition: stencil tables can be built with singular stencils
    // (single weight of 1.0f) as place-holders for coarse mesh vertices,
    // which also needs to be accounted for.
    _stencilsOffset=-1;
    {         int maxlevel = _refiner.GetMaxLevel(),
            nverts = _refiner.GetNumVerticesTotal(),
            nstencils = _stencils.GetNumStencils();
        if (nstencils==nverts) {

            // the table contain stencils for the control vertices
            _stencilsOffset = nverts - _refiner.GetNumVertices(maxlevel);

        } else if (nstencils==(nverts-_refiner.GetNumVertices(0))) {

            // the table does not contain stencils for the control vertices
            _stencilsOffset = nverts - _refiner.GetNumVertices(maxlevel)
                - _refiner.GetNumVertices(0);

        } else {
            // these are not the stencils you are looking for...
            assert(0);
        }
    }
}
static inline void
factorizeBasisVertex(StencilTables const & stencils, Point const & p, ProtoStencil dst) {
    // Use the Allocator to factorize the Gregory patch influence CVs with the
    // supporting CVs from the stencil tables.
    dst.Clear();
    for (int j=0; j<p.GetSize(); ++j) {
        dst.AddWithWeight(stencils,
            p.GetIndices()[j], p.GetWeights()[j]);
    }
}
bool
GregoryBasisFactory::AddPatchBasis(Index faceIndex) {

    // Gregory patches only exist on the hight
    Vtr::Level const & level = _refiner.getLevel(_refiner.GetMaxLevel());

    if (level.getMaxValence()>GetMaxValence()) {
        // The proto-basis closed-form table limits valence to 'MAX_VALENCE'
        return false;
    }

    // Gather the CVs that influence the Gregory patch and their relative
    // weights in a basis
    ProtoBasis basis(level, faceIndex);

    // The basis vertex indices are currently local to maxlevel: need to offset
    // to match layout of adaptive StencilTables (see factory constructor above)
    assert(_stencilsOffset>=0);
    basis.OffsetIndices(_stencilsOffset);

    // Factorize the basis CVs with the stencil tables: the basis is now
    // expressed as a linear combination of vertices from the coarse control
    // mesh with no data dependencies
    for (int i=0; i<4; ++i) {
        int offset = _currentStencil + i * 5;
        factorizeBasisVertex(_stencils, basis.P[i],  _alloc[offset]);
        factorizeBasisVertex(_stencils, basis.Ep[i], _alloc[offset+1]);
        factorizeBasisVertex(_stencils, basis.Em[i], _alloc[offset+2]);
        factorizeBasisVertex(_stencils, basis.Fp[i], _alloc[offset+3]);
        factorizeBasisVertex(_stencils, basis.Fm[i], _alloc[offset+4]);
    }
    _currentStencil += 20;
    return true;
}
StencilTables const *
GregoryBasisFactory::CreateStencilTables(int const permute[20]) {

    int nstencils = (int)_alloc.GetNumStencils(),
           nelems = _alloc.GetNumVerticesTotal();

    if (nstencils==0 or nelems==0) {
        return 0;
    }

    // Finalize the stencil tables from the temporary pool allocator
    StencilTables * result = new StencilTables;

    result->_numControlVertices = _refiner.GetNumVertices(0);

    result->resize(nstencils, nelems);

    Stencil dst(&result->_sizes.at(0),
        &result->_indices.at(0), &result->_weights.at(0));

    for (int i=0; i<nstencils; ++i) {

        Index index = i;
        if (permute) {
            int localIndex = i % 20,
                baseIndex = i - localIndex;
            index = baseIndex + permute[localIndex];
        }

        *dst._size = _alloc.CopyStencil(index, dst._indices, dst._weights);

        dst.Next();
    }

    result->generateOffsets();

    return result;
}


} // end namespace Far

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv
