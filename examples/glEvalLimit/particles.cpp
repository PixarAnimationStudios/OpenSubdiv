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

#include <cassert>

STParticles::STParticles(Refiner const & refiner, int nparticles, bool centered) :
    _speed(1.0f) {

    int nptexfaces = refiner.GetNumPtexFaces(),
        nsamples = nptexfaces * nparticles;

    srand(static_cast<int>(2147483647));

    {   // initialize positions

        _positions.resize(nsamples);
        Position * pos = &_positions[0];

        for (int i=0; i<nptexfaces; ++i) {
            for (int j=0; j<nparticles; ++j) {
                pos->ptexIndex = i;
                pos->s = centered ? 0.5f : (float)rand()/(float)RAND_MAX;
                pos->t = centered ? 0.5f : (float)rand()/(float)RAND_MAX;
                ++pos;
            }
        }
    }

    {   // initialize velocities
        _velocities.resize(nsamples*2);

        for (int i=0; i<nsamples; ++i) {
            // initialize normalized random directions
            float s = 2.0f*(float)rand()/(float)RAND_MAX - 1.0f,
                  t = 2.0f*(float)rand()/(float)RAND_MAX - 1.0f,
                  l = sqrtf(s*s+t*t);

            _velocities[2*i  ] = s / l;
            _velocities[2*i+1] = t / l;
        }
    }

    {   // initialize topology adjacency
        _adjacency.resize(nptexfaces);

        int nfaces = refiner.GetNumFaces(0),
           adjfaces[4],
           adjedges[4];

        for (int face=0, ptexface=0; face<nfaces; ++face) {

            OpenSubdiv::Far::ConstIndexArray fverts =
                refiner.GetFaceVertices(0, face);

            if (fverts.size()==4) {
                refiner.GetPtexAdjacency(face, 0, adjfaces, adjedges);
                _adjacency[ptexface] = FaceInfo(adjfaces, adjedges, false);
                ++ptexface;
            } else {
                for (int vert=0; vert<fverts.size(); ++vert) {
                    refiner.GetPtexAdjacency(face, vert, adjfaces, adjedges);
                    _adjacency[ptexface+vert] =
                        FaceInfo(adjfaces, adjedges, true);
                }
                ptexface+=fverts.size();
            }
        }
    }
    //std::cout << *this;
}

void
STParticles::Update(float deltaTime) {

    float speed = GetSpeed() * std::max(0.001f, std::min(deltaTime, 0.5f));

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
            warpParticle(edge, p, dp);
        }
        assert((p->s>=0.0f) and (p->s<=1.0f) and (p->t>=0.0f) and (p->t<=1.0f));
    }
    
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

inline void
Trim(STParticles::Position * p) {
    if (p->s <0.0f) p->s = 1.0f + p->s;
    if (p->s>=1.0f) p->s = p->s - 1.0f;
    if (p->t <0.0f) p->t = 1.0f + p->t;
    if (p->t>=1.0f) p->t = p->t - 1.0f;
    assert((p->s>=0.0f) and (p->s<=1.0f) and (p->t>=0.0f) and (p->t<=1.0f));
}

void
STParticles::warpParticle(int edge, Position * p, float * dp) {

    assert(p->ptexIndex<(int)_adjacency.size() and (edge>=0 and edge<4));
    
    FaceInfo const & f = _adjacency[p->ptexIndex];

    int afid = f.adjface(edge),
        aeid = f.adjedge(edge);

    if (afid==-1) {
        // boundary detected: bounce the particle
        Bounce(edge, p, dp);
    } else {
        FaceInfo const & af = _adjacency[afid];
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


