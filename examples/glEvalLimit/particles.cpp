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

#include "particles.h"

#include <far/ptexIndices.h>
#include <far/patchMap.h>

#include <cmath>

#ifdef OPENSUBDIV_HAS_TBB
#include <tbb/parallel_for.h>
#include <tbb/atomic.h>
tbb::atomic<int> g_tbbCounter;
class TbbUpdateKernel {
public:
    TbbUpdateKernel(float speed,
                    STParticles::Position *positions,
                    float *velocities,
                    std::vector<STParticles::FaceInfo> const &adjacency,
                    OpenSubdiv::Osd::PatchCoord *patchCoords,
                    OpenSubdiv::Far::PatchMap const *patchMap) :
        _speed(speed), _positions(positions), _velocities(velocities),
        _adjacency(adjacency), _patchCoords(patchCoords), _patchMap(patchMap) {
    }

    void operator () (tbb::blocked_range<int> const &r) const {
        for (int i = r.begin(); i < r.end(); ++i) {
            STParticles::Position * p = _positions + i;
            float    *dp = _velocities + i*2;

            // apply velocity
            p->s += dp[0] * _speed;
            p->t += dp[1] * _speed;

            // make sure particles can't skip more than 1 face boundary at a time
            assert((p->s>-2.0f) and (p->s<2.0f) and (p->t>-2.0f) and (p->t<2.0f));

            // check if the particle is jumping a boundary
            // note: a particle can jump 2 edges at a time (a "diagonal" jump)
            //       this is not treated here.
            int edge = -1;
            if (p->s >= 1.0f) edge = 1;
            if (p->s <= 0.0f) edge = 3;
            if (p->t >= 1.0f) edge = 2;
            if (p->t <= 0.0f) edge = 0;

            if (edge>=0) {
                // warp the particle to the other side of the boundary
                STParticles::WarpParticle(_adjacency, edge, p, dp);
            }
            assert((p->s>=0.0f) and (p->s<=1.0f) and (p->t>=0.0f) and (p->t<=1.0f));

            // resolve particle positions into patch handles
            OpenSubdiv::Far::PatchTable::PatchHandle const *handle =
                _patchMap->FindPatch(p->ptexIndex, p->s, p->t);
            if (handle) {
                int index = g_tbbCounter.fetch_and_add(1);
                _patchCoords[index] =
                    OpenSubdiv::Osd::PatchCoord(*handle, p->s, p->t);
            }
        }
    }
private:
    float _speed;
    STParticles::Position *_positions;
    float *_velocities;
    std::vector<STParticles::FaceInfo> const &_adjacency;
    OpenSubdiv::Osd::PatchCoord *_patchCoords;
    OpenSubdiv::Far::PatchMap const *_patchMap;
};
#endif

#include <cassert>

STParticles::STParticles(Refiner const & refiner,
                         PatchTable const *patchTable,
                         int nParticles, bool centered) :
    _speed(1.0f) {

    OpenSubdiv::Far::PtexIndices ptexIndices(refiner);

    // Create a far patch map
    _patchMap = new OpenSubdiv::Far::PatchMap(*patchTable);

    int nPtexFaces = ptexIndices.GetNumFaces();

    srand(static_cast<int>(2147483647));

    {   // initialize positions

        _positions.resize(nParticles);
        Position * pos = &_positions[0];

        for (int i = 0; i < nParticles; ++i) {
            pos->ptexIndex = std::min(
                (int)(((float)rand()/(float)RAND_MAX) * nPtexFaces), nPtexFaces-1);
            pos->s = centered ? 0.5f : (float)rand()/(float)RAND_MAX;
            pos->t = centered ? 0.5f : (float)rand()/(float)RAND_MAX;
            ++pos;
        }
    }

    {   // initialize velocities
        _velocities.resize(nParticles * 2);

        for (int i = 0; i < nParticles; ++i) {
            // initialize normalized random directions
            float s = 2.0f*(float)rand()/(float)RAND_MAX - 1.0f,
                  t = 2.0f*(float)rand()/(float)RAND_MAX - 1.0f,
                  l = sqrtf(s*s+t*t);

            _velocities[2*i  ] = s / l;
            _velocities[2*i+1] = t / l;
        }
    }

    {   // initialize topology adjacency
        _adjacency.resize(nPtexFaces);

        OpenSubdiv::Far::TopologyLevel const & refBaseLevel = refiner.GetLevel(0);

        int nfaces = refBaseLevel.GetNumFaces(),
           adjfaces[4],
           adjedges[4];

        for (int face=0, ptexface=0; face<nfaces; ++face) {

            OpenSubdiv::Far::ConstIndexArray fverts = refBaseLevel.GetFaceVertices(face);

            if (fverts.size()==4) {
                ptexIndices.GetAdjacency(refiner, face, 0, adjfaces, adjedges);
                _adjacency[ptexface] = FaceInfo(adjfaces, adjedges, false);
                ++ptexface;
            } else {
                for (int vert=0; vert<fverts.size(); ++vert) {
                    ptexIndices.GetAdjacency(refiner, face, vert, adjfaces, adjedges);
                    _adjacency[ptexface+vert] =
                        FaceInfo(adjfaces, adjedges, true);
                }
                ptexface+=fverts.size();
            }
        }
    }
    //std::cout << *this;
}

inline void
FlipS(STParticles::Position * p, float * dp) {
    p->s = 1.0f-p->s;
    dp[0] = - dp[0];
}

inline void
FlipT(STParticles::Position * p, float * dp) {
    p->t = 1.0f-p->t;
    dp[1] = - dp[1];
}

inline void
SwapST(STParticles::Position * p, float * dp) {
    std::swap(p->s, p->t);
    std::swap(dp[0], dp[1]);
}

inline void
Rotate(int rot, STParticles::Position * p, float * dp) {

    switch (rot & 3) {
        default: return;
        case 1: FlipS(p, dp); SwapST(p, dp); break;
        case 2: FlipS(p, dp); FlipT(p, dp);  break;
        case 3: FlipT(p, dp); SwapST(p, dp); break;
    }
    assert((p->s>=0.0f) and (p->s<=1.0f) and (p->t>=0.0f) and (p->t<=1.0f));
}

inline void
Trim(STParticles::Position * p) {
    if (p->s <0.0f) p->s = 1.0f + p->s;
    if (p->s>=1.0f) p->s = p->s - 1.0f;
    if (p->t <0.0f) p->t = 1.0f + p->t;
    if (p->t>=1.0f) p->t = p->t - 1.0f;
    assert((p->s>=0.0f) and (p->s<=1.0f) and (p->t>=0.0f) and (p->t<=1.0f));
}

inline void
Clamp(STParticles::Position * p) {
         if (p->s<0.0f) {
        p->s=0.0f; 
    } else if (p->s>1.0f) {
        p->s=1.0f;
    }
         if (p->t<0.0f) {
        p->t=0.0f; 
    } else if (p->t>1.0f) {
        p->t=1.0f;
    }
}

inline void
Bounce(int edge, STParticles::Position * p, float * dp) {
    switch (edge) {
        case 0: assert(p->t<=0.0f); p->t = -p->t;       dp[1] = -dp[1]; break;
        case 1: assert(p->s>=1.0f); p->s = 2.0f - p->s; dp[0] = -dp[0]; break;
        case 2: assert(p->t>=1.0f); p->t = 2.0f - p->t; dp[1] = -dp[1]; break;
        case 3: assert(p->s<=0.0f); p->s = -p->s;       dp[0] = -dp[0]; break;
    }
    
    // because 'diagonal' cases aren't handled, stick particles to edges when
    // if they cross 2 boundaries
    Clamp(p);
    assert((p->s>=0.0f) and (p->s<=1.0f) and (p->t>=0.0f) and (p->t<=1.0f));
}

void
STParticles::WarpParticle(std::vector<FaceInfo> const &adjacency,
                          int edge, Position * p, float * dp) {
    assert(p->ptexIndex<(int)adjacency.size() and (edge>=0 and edge<4));
    
    FaceInfo const & f = adjacency[p->ptexIndex];

    int afid = f.adjface(edge),
        aeid = f.adjedge(edge);

    if (afid==-1) {
        // boundary detected: bounce the particle
        Bounce(edge, p, dp);
    } else {
        FaceInfo const & af = adjacency[afid];
        int rot = edge - aeid + 2;

        bool fIsSubface = f.isSubface(),
             afIsSubface = af.isSubface();

        if (fIsSubface != afIsSubface) {
            // XXXX manuelk domain should be split properly
            Bounce(edge, p, dp);
        } else {
            Trim(p);
            Rotate(rot, p, dp);
            p->ptexIndex = afid; // move particle to adjacent face
        }
    }
    assert((p->s>=0.0f) and (p->s<=1.0f) and (p->t>=0.0f) and (p->t<=1.0f));
}

STParticles::~STParticles() {
    delete _patchMap;
}

void
STParticles::Update(float deltaTime) {

    if (deltaTime == 0) return;
    float speed = GetSpeed() * std::max(0.001f, std::min(deltaTime, 0.5f));

    _patchCoords.clear();

    // XXX: this process should be parallelized.
#ifdef OPENSUBDIV_HAS_TBB

    _patchCoords.resize((int)GetNumParticles());
    TbbUpdateKernel kernel(speed, &_positions[0], &_velocities[0],
                           _adjacency, &_patchCoords[0], _patchMap);;
    g_tbbCounter = 0;
    tbb::blocked_range<int> range(0, GetNumParticles(), 256);
    tbb::parallel_for(range, kernel);
    _patchCoords.resize(g_tbbCounter);
#else
    Position *  p = &_positions[0];
    float    * dp = &_velocities[0];
    for (int i=0; i<GetNumParticles(); ++i, ++p, dp+=2) {
        // apply velocity
        p->s += dp[0] * speed;
        p->t += dp[1] * speed;

        // make sure particles can't skip more than 1 face boundary at a time
        assert((p->s>-2.0f) and (p->s<2.0f) and (p->t>-2.0f) and (p->t<2.0f));

        // check if the particle is jumping a boundary
        // note: a particle can jump 2 edges at a time (a "diagonal" jump)
        //       this is not treated here.
        int edge = -1;

        if (p->s >= 1.0f) edge = 1;
        if (p->s <= 0.0f) edge = 3;
        if (p->t >= 1.0f) edge = 2;
        if (p->t <= 0.0f) edge = 0;

        if (edge>=0) {
            // warp the particle to the other side of the boundary
            WarpParticle(_adjacency, edge, p, dp);
        }
        assert((p->s>=0.0f) and (p->s<=1.0f) and (p->t>=0.0f) and (p->t<=1.0f));

        // resolve particle positions into patch handles
        OpenSubdiv::Far::PatchTable::PatchHandle const *handle =
            _patchMap->FindPatch(p->ptexIndex, p->s, p->t);
        if (handle) {
            _patchCoords.push_back(
                OpenSubdiv::Osd::PatchCoord(*handle, p->s, p->t));
        }
    }
#endif
}

// Dump adjacency info
std::ostream & operator << (std::ostream & os,
    STParticles::FaceInfo const & f) {

    os << "  adjface: " << f.adjfaces[0] << ' '
                        << f.adjfaces[1] << ' '
                        << f.adjfaces[2] << ' '
                        << f.adjfaces[3]
       << "  adjedge: " << f.adjedge(0) << ' '
                        << f.adjedge(1) << ' '
                        << f.adjedge(2) << ' '
                        << f.adjedge(3)
       << "  flags:";

    if (f.flags == 0) {
        os << " (none)";
    } else {
        if (f.isSubface()) {
            std::cout << " subface";
        }
    }
    os << std::endl;

    return os;
}

std::ostream & operator << (std::ostream & os,
    STParticles const & particles) {

    for (int i=0; i<(int)particles._adjacency.size(); ++i) {
        os << particles._adjacency[i];
    }

    return os;
}


