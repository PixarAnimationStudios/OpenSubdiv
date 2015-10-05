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
#include "../far/stencilTableFactory.h"
#include "../far/topologyRefiner.h"
#include "../vtr/stackBuffer.h"

#include <cassert>
#include <cmath>
#include <cstring>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Far {

int
GregoryBasis::ProtoBasis::GetNumElements() const {
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
GregoryBasis::ProtoBasis::Copy(int * sizes, Index * indices, float * weights) const {
    for (int vid=0; vid<4; ++vid) {
        P[vid].Copy(&sizes, &indices, &weights);
        Ep[vid].Copy(&sizes, &indices, &weights);
        Em[vid].Copy(&sizes, &indices, &weights);
        Fp[vid].Copy(&sizes, &indices, &weights);
        Fm[vid].Copy(&sizes, &indices, &weights);
    }
}

void
GregoryBasis::ProtoBasis::Copy(GregoryBasis * dest) const {
    int nelems = GetNumElements();

    dest->_indices.resize(nelems);
    dest->_weights.resize(nelems);

    Copy(dest->_sizes, &dest->_indices[0], &dest->_weights[0]);
}

inline float csf(Index n, Index j) {
    if (j%2 == 0) {
        return cosf((2.0f * float(M_PI) * float(float(j-0)/2.0f))/(float(n)+3.0f));
    } else {
        return sinf((2.0f * float(M_PI) * float(float(j-1)/2.0f))/(float(n)+3.0f));
    }
}

inline float computeCoefficient(int valence) {
    // precomputed coefficient table up to valence 29
    static float efTable[] = {
        0, 0, 0,
        0.812816f, 0.500000f, 0.363644f, 0.287514f,
        0.238688f, 0.204544f, 0.179229f, 0.159657f,
        0.144042f, 0.131276f, 0.120632f, 0.111614f,
        0.103872f, 0.09715f, 0.0912559f, 0.0860444f,
        0.0814022f, 0.0772401f, 0.0734867f, 0.0700842f,
        0.0669851f, 0.0641504f, 0.0615475f, 0.0591488f,
        0.0569311f, 0.0548745f, 0.0529621f
    };
    assert(valence > 0);
    if (valence < 30) return efTable[valence];

    float t = 2.0f * float(M_PI) / float(valence);
    return 1.0f / (valence * (cosf(t) + 5.0f +
                              sqrtf((cosf(t) + 9) * (cosf(t) + 1)))/16.0f);
}

GregoryBasis::ProtoBasis::ProtoBasis(
    Vtr::internal::Level const & level, Index faceIndex,
    int levelVertOffset, int fvarChannel) {

    // XXX: This function is subject to refactor in 3.1

    Vtr::ConstIndexArray facePoints = (fvarChannel<0) ?
        level.getFaceVertices(faceIndex) :
            level.getFaceFVarValues(faceIndex, fvarChannel);
    assert(facePoints.size()==4);

    int maxvalence = level.getMaxValence(),
        valences[4],
        zerothNeighbors[4];

    // XXX: a temporary hack for the performance issue
    // ensure Point has a capacity for the neighborhood of
    // 2 extraordinary verts + 2 regular verts
    // worse case: n-valence verts at a corner of n-gon.
    int stencilCapacity =
        4/*0-ring*/ + 2*(2*(maxvalence-2)/*1-ring around extraordinaries*/
                         + 2/*1-ring around regulars, excluding shared ones*/);

    Point e0[4], e1[4];
    for (int i = 0; i < 4; ++i) {
        P[i].Clear(stencilCapacity);
        e0[i].Clear(stencilCapacity);
        e1[i].Clear(stencilCapacity);
        V[i].Clear(1);
    }

    Vtr::internal::StackBuffer<Index, 40> manifoldRings[4];
    manifoldRings[0].SetSize(maxvalence*2);
    manifoldRings[1].SetSize(maxvalence*2);
    manifoldRings[2].SetSize(maxvalence*2);
    manifoldRings[3].SetSize(maxvalence*2);

    Vtr::internal::StackBuffer<Point, 10> f(maxvalence);
    Vtr::internal::StackBuffer<Point, 40> r(maxvalence*4);

    // the first phase

    for (int vid=0; vid<4; ++vid) {
        // save for varying stencils
        V[vid].AddWithWeight(facePoints[vid], 1.0f);

        int ringSize =
            level.gatherQuadRegularRingAroundVertex(
                facePoints[vid], manifoldRings[vid], fvarChannel);

        int valence;
        if (ringSize & 1) {
            // boundary vertex
            manifoldRings[vid][ringSize] = manifoldRings[vid][ringSize-1];
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

        for (int i=0; i<ivalence; ++i) {

            Index im = (i+ivalence-1)%ivalence,
                  ip = (i+1)%ivalence;

            Index idx_neighbor = (manifoldRings[vid][2*i + 0]),
                  idx_diagonal = (manifoldRings[vid][2*i + 1]),
                  idx_neighbor_p = (manifoldRings[vid][2*ip + 0]),
                  idx_neighbor_m = (manifoldRings[vid][2*im + 0]),
                  idx_diagonal_m = (manifoldRings[vid][2*im + 1]);

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

            float d = float(ivalence)+5.0f;
            f[i].Clear(4);
            f[i].AddWithWeight(facePoints[vid], float(ivalence)/d);
            f[i].AddWithWeight(idx_neighbor_p,  2.0f/d);
            f[i].AddWithWeight(idx_neighbor,    2.0f/d);
            f[i].AddWithWeight(idx_diagonal,    1.0f/d);
            P[vid].AddWithWeight(f[i], 1.0f/float(ivalence));

            int rid = vid * maxvalence + i;
            r[rid].Clear(4);
            r[rid].AddWithWeight(idx_neighbor_p,  1.0f/3.0f);
            r[rid].AddWithWeight(idx_neighbor_m, -1.0f/3.0f);
            r[rid].AddWithWeight(idx_diagonal,    1.0f/6.0f);
            r[rid].AddWithWeight(idx_diagonal_m, -1.0f/6.0f);
        }

        zerothNeighbors[vid] = zerothNeighbor;
        if (currentNeighbor == 1) {
            boundaryEdgeNeighbors[1] = boundaryEdgeNeighbors[0];
        }

        for (int i=0; i<ivalence; ++i) {
            int im = (i+ivalence-1)%ivalence;
            float c0 = 0.5f * csf(ivalence-3, 2*i);
            float c1 = 0.5f * csf(ivalence-3, 2*i+1);
            e0[vid].AddWithWeight(f[i ], c0);
            e0[vid].AddWithWeight(f[im], c0);
            e1[vid].AddWithWeight(f[i ], c1);
            e1[vid].AddWithWeight(f[im], c1);
        }

        float ef = computeCoefficient(ivalence);
        e0[vid] *= ef;
        e1[vid] *= ef;

        // Boundary gregory case:
        if (valence < 0) {
            P[vid].Clear(stencilCapacity);
            if (ivalence>2) {
                P[vid].AddWithWeight(boundaryEdgeNeighbors[0], 1.0f/6.0f);
                P[vid].AddWithWeight(boundaryEdgeNeighbors[1], 1.0f/6.0f);
                P[vid].AddWithWeight(facePoints[vid], 4.0f/6.0f);
            } else {
                P[vid].AddWithWeight(facePoints[vid], 1.0f);
            }
            float k = float(float(ivalence) - 1.0f);    //k is the number of faces
            float c = cosf(float(M_PI)/k);
            float s = sinf(float(M_PI)/k);
            float gamma = -(4.0f*s)/(3.0f*k+c);
            float alpha_0k = -((1.0f+2.0f*c)*sqrtf(1.0f+c))/((3.0f*k+c)*sqrtf(1.0f-c));
            float beta_0 = s/(3.0f*k + c);

            int idx_diagonal = manifoldRings[vid][2*zerothNeighbor + 1];

            e0[vid].Clear(stencilCapacity);
            e0[vid].AddWithWeight(boundaryEdgeNeighbors[0],  1.0f/6.0f);
            e0[vid].AddWithWeight(boundaryEdgeNeighbors[1], -1.0f/6.0f);

            e1[vid].Clear(stencilCapacity);
            e1[vid].AddWithWeight(facePoints[vid],           gamma);
            e1[vid].AddWithWeight(idx_diagonal,              beta_0);
            e1[vid].AddWithWeight(boundaryEdgeNeighbors[0],  alpha_0k);
            e1[vid].AddWithWeight(boundaryEdgeNeighbors[1],  alpha_0k);

            for (int x=1; x<ivalence-1; ++x) {

                Index curri = ((x + zerothNeighbor)%ivalence);

                float alpha = (4.0f*sinf((float(M_PI) * float(x))/k))/(3.0f*k+c),
                      beta = (sinf((float(M_PI) * float(x))/k) + sinf((float(M_PI) * float(x+1))/k))/(3.0f*k+c);

                Index idx_neighbor = manifoldRings[vid][2*curri + 0],
                      idx_diagonal = manifoldRings[vid][2*curri + 1];

                e1[vid].AddWithWeight(idx_neighbor, alpha);
                e1[vid].AddWithWeight(idx_diagonal, beta);
            }
            e1[vid] *= 1.0f/3.0f;
        }
    }

    // the second phase

    for (int vid=0; vid<4; ++vid) {

        int n = abs(valences[vid]);
        int ivalence = n;

        int ip = (vid+1)%4,
            im = (vid+3)%4,
            np = abs(valences[ip]),
            nm = abs(valences[im]);

        Index start = -1, prev = -1, start_m = -1, prev_p = -1;
        for (int i = 0; i < n; ++i) {
            if (manifoldRings[vid][i*2] == facePoints[ip])
                start = i;
            if (manifoldRings[vid][i*2] == facePoints[im])
                prev = i;
        }
        for (int i = 0; i < np; ++i) {
            if (manifoldRings[ip][i*2] == facePoints[vid]) {
                prev_p = i;
                break;
            }
        }
        for (int i = 0; i < nm; ++i) {
            if (manifoldRings[im][i*2] == facePoints[vid]) {
                start_m = i;
                break;
            }
        }
        assert(start != -1 && prev != -1 && start_m != -1 && prev_p != -1);

        Point Em_ip = P[ip];
        Point Ep_im = P[im];

        if (valences[ip]<-2) {
            Index j = (np + prev_p - zerothNeighbors[ip]) % np;
            Em_ip.AddWithWeight(e0[ip], cosf((float(M_PI)*j)/float(np-1)));
            Em_ip.AddWithWeight(e1[ip], sinf((float(M_PI)*j)/float(np-1)));
        } else {
            Em_ip.AddWithWeight(e0[ip], csf(np-3, 2*prev_p));
            Em_ip.AddWithWeight(e1[ip], csf(np-3, 2*prev_p+1));
        }

        if (valences[im]<-2) {
            Index j = (nm + start_m - zerothNeighbors[im]) % nm;
            Ep_im.AddWithWeight(e0[im], cosf((float(M_PI)*j)/float(nm-1)));
            Ep_im.AddWithWeight(e1[im], sinf((float(M_PI)*j)/float(nm-1)));
        } else {
            Ep_im.AddWithWeight(e0[im], csf(nm-3, 2*start_m));
            Ep_im.AddWithWeight(e1[im], csf(nm-3, 2*start_m+1));
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

        if (valences[vid] >= 2) {

            float s1 = 3.0f - 2.0f*csf(n-3,2)-csf(np-3,2),
                  s2 = 2.0f*csf(n-3,2),
                  s3 = 3.0f -2.0f*cosf(2.0f*float(M_PI)/float(n)) - cosf(2.0f*float(M_PI)/float(nm));
            Ep[vid] = P[vid];
            Ep[vid].AddWithWeight(e0[vid], csf(n-3, 2*start));
            Ep[vid].AddWithWeight(e1[vid], csf(n-3, 2*start +1));

            Em[vid] = P[vid];
            Em[vid].AddWithWeight(e0[vid], csf(n-3, 2*prev ));
            Em[vid].AddWithWeight(e1[vid], csf(n-3, 2*prev + 1));

            Fp[vid].Clear(stencilCapacity);
            Fp[vid].AddWithWeight(P[vid],    csf(np-3, 2)/3.0f);
            Fp[vid].AddWithWeight(Ep[vid],   s1/3.0f);
            Fp[vid].AddWithWeight(Em_ip,     s2/3.0f);
            Fp[vid].AddWithWeight(rp[start], 1.0f/3.0f);

            Fm[vid].Clear(stencilCapacity);
            Fm[vid].AddWithWeight(P[vid],   csf(nm-3, 2)/3.0f);
            Fm[vid].AddWithWeight(Em[vid],  s3/3.0f);
            Fm[vid].AddWithWeight(Ep_im,    s2/3.0f);
            Fm[vid].AddWithWeight(rp[prev], -1.0f/3.0f);
        } else if (valences[vid] < -2) {

            Index jp = (ivalence + start - zerothNeighbors[vid]) % ivalence,
                  jm = (ivalence + prev  - zerothNeighbors[vid]) % ivalence;

            float s1 = 3-2*csf(n-3,2)-csf(np-3,2),
                  s2 = 2*csf(n-3,2),
                  s3 = 3.0f-2.0f*cosf(2.0f*float(M_PI)/n)-cosf(2.0f*float(M_PI)/nm);

            Ep[vid] = P[vid];
            Ep[vid].AddWithWeight(e0[vid], cosf((float(M_PI)*jp)/float(ivalence-1)));
            Ep[vid].AddWithWeight(e1[vid], sinf((float(M_PI)*jp)/float(ivalence-1)));

            Em[vid] = P[vid];
            Em[vid].AddWithWeight(e0[vid], cosf((float(M_PI)*jm)/float(ivalence-1)));
            Em[vid].AddWithWeight(e1[vid], sinf((float(M_PI)*jm)/float(ivalence-1)));

            Fp[vid].Clear(stencilCapacity);
            Fp[vid].AddWithWeight(P[vid],    csf(np-3,2)/3.0f);
            Fp[vid].AddWithWeight(Ep[vid],   s1/3.0f);
            Fp[vid].AddWithWeight(Em_ip,     s2/3.0f);
            Fp[vid].AddWithWeight(rp[start], 1.0f/3.0f);

            Fm[vid].Clear(stencilCapacity);
            Fm[vid].AddWithWeight(P[vid],   csf(nm-3,2)/3.0f);
            Fm[vid].AddWithWeight(Em[vid],  s3/3.0f);
            Fm[vid].AddWithWeight(Ep_im,    s2/3.0f);
            Fm[vid].AddWithWeight(rp[prev], -1.0f/3.0f);

            if (valences[im]<0) {
                s1=3-2*csf(n-3,2)-csf(np-3,2);
                Fp[vid].Clear(stencilCapacity);
                Fp[vid].AddWithWeight(P[vid],    csf(np-3,2)/3.0f);
                Fp[vid].AddWithWeight(Ep[vid],   s1/3.0f);
                Fp[vid].AddWithWeight(Em_ip,     s2/3.0f);
                Fp[vid].AddWithWeight(rp[start], 1.0f/3.0f);
                Fm[vid] = Fp[vid];
            } else if (valences[ip]<0) {
                s1 = 3.0f-2.0f*cosf(2.0f*float(M_PI)/n)-cosf(2.0f*float(M_PI)/nm);
                Fm[vid].Clear(stencilCapacity);
                Fm[vid].AddWithWeight(P[vid],   csf(nm-3,2)/3.0f);
                Fm[vid].AddWithWeight(Em[vid],  s1/3.0f);
                Fm[vid].AddWithWeight(Ep_im,    s2/3.0f);
                Fm[vid].AddWithWeight(rp[prev], -1.0f/3.0f);
                Fp[vid] = Fm[vid];
            }

        } else if (valences[vid]==-2) {
            Ep[vid].Clear(stencilCapacity);
            Ep[vid].AddWithWeight(facePoints[vid], 2.0f/3.0f);
            Ep[vid].AddWithWeight(facePoints[ip],  1.0f/3.0f);

            Em[vid].Clear(stencilCapacity);
            Em[vid].AddWithWeight(facePoints[vid], 2.0f/3.0f);
            Em[vid].AddWithWeight(facePoints[im],  1.0f/3.0f);

            Fp[vid].Clear(stencilCapacity);
            Fp[vid].AddWithWeight(facePoints[vid],         4.0f/9.0f);
            Fp[vid].AddWithWeight(facePoints[((vid+2)%n)], 1.0f/9.0f);
            Fp[vid].AddWithWeight(facePoints[ip],          2.0f/9.0f);
            Fp[vid].AddWithWeight(facePoints[im],          2.0f/9.0f);
            Fm[vid] = Fp[vid];
        }
    }

    // offset stencil indices.
    // These stencils are created relative to the level. Adding levelVertOffset,
    // we get stencils with absolute indices
    // (starts from the coarse level if the leveVertOffset includes level 0)
    for (int i = 0; i < 4; ++i) {
        P[i].OffsetIndices(levelVertOffset);
        Ep[i].OffsetIndices(levelVertOffset);
        Em[i].OffsetIndices(levelVertOffset);
        Fp[i].OffsetIndices(levelVertOffset);
        Fm[i].OffsetIndices(levelVertOffset);
        V[i].OffsetIndices(levelVertOffset);
    }
}

/*static*/
StencilTable *
GregoryBasis::CreateStencilTable(PointsVector const &stencils) {

    int nStencils = (int)stencils.size();
    if (nStencils == 0) return NULL;

    int nElements = 0;
    for (int i = 0; i < nStencils; ++i) {
        nElements += stencils[i].GetSize();
    }

    // allocate destination
    StencilTable *stencilTable = new StencilTable();

    // XXX: do we need numControlVertices in stencilTable?
    stencilTable->_numControlVertices = 0;
    stencilTable->resize(nStencils, nElements);

    int * sizes = &stencilTable->_sizes[0];
    Index * indices = &stencilTable->_indices[0];
    float * weights = &stencilTable->_weights[0];

    for (int i = 0; i < nStencils; ++i) {
        stencils[i].Copy(&sizes, &indices, &weights);
    }
    stencilTable->generateOffsets();

    return stencilTable;
}

} // end namespace Far

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv
