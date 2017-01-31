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

#ifndef ST_PARTICLES_H
#define ST_PARTICLES_H

#include <far/topologyRefiner.h>
#include <far/patchMap.h>
#include <osd/types.h>
#include <iostream>

//
// In order to emphasize the dynamic nature of the EvalLimit API, where the
// locations can be arbitrarily updated before each evaluation, the glEvalLimit
// example treats each sample as a 'ST particle'.
//
// ST Particles are a simplified parametric-space particle dynamics simulation: each
// particle is assigned a location on the subdivision surface limit that is
// composed of a unique ptex face index, with a local (s,t) parametric pair.
//
// The system also generates an array of parametric velocities (ds, dt) for each
// particle. An Update() function then applies the velocities to the locations and
// moves the points along the parametric space.
//
// Face boundaries are managed using a ptex adjacency table obtained from the
// Far::TopologyRefiner. Every time a particle moves outside of the [0.0f, 1.0f]
// parametric range, a 'warp' function moves it to the neighboring face, or
// bounces it, if the edge happens to be a boundary.
// 
// Note: currently the adjacency code does not handle 'diagonal' crossings, nor
// crossings between quad and non-quad faces.
//
class STParticles {

public:
    /// \brief Coordinates set on a limit surface
    ///
    struct Position {
        Position() { }

        /// \brief Constructor
        ///
        /// @param f Ptex face id
        ///
        /// @param x parametric location on face
        ///
        /// @param y parametric location on face
        ///
        Position(int f, float x, float y) : ptexIndex(f), s(x), t(y) { }

        int ptexIndex;      ///< ptex face index
        float s, t;         ///< parametric location on face
    };

    //
    // Topology adjacency (borrowed from Ptexture.h)
    //
    struct FaceInfo {

        enum { flag_subface = 8 };

        FaceInfo() : adjedges(0), flags(0) {
            adjfaces[0] = adjfaces[1] = adjfaces[2] = adjfaces[3] = -1;
        }

        FaceInfo(int adjfaces_[4], int adjedges_[4], bool isSubface=false) :
            flags(isSubface ? flag_subface : 0) {
            setadjfaces(adjfaces_[0], adjfaces_[1], adjfaces_[2], adjfaces_[3]);
            setadjedges(adjedges_[0], adjedges_[1], adjedges_[2], adjedges_[3]);
        }

        void setadjfaces(int f0, int f1, int f2, int f3) {
            adjfaces[0] = f0;
            adjfaces[1] = f1;
            adjfaces[2] = f2;
            adjfaces[3] = f3;
        }

        void setadjedges(int e0, int e1, int e2, int e3) {
            adjedges = (e0&3) | ((e1&3)<<2) | ((e2&3)<<4) | ((e3&3)<<6);
        }

        int adjface(int eid) const { return adjfaces[eid]; }

        int adjedge(int eid) const { return int((adjedges >> (2*eid)) & 3); }

        bool isSubface() const { return (flags & flag_subface) != 0; }

        unsigned int adjedges :8,
                     flags    :8;
                 int adjfaces[4];
    };

    typedef OpenSubdiv::Far::TopologyRefiner Refiner;
    typedef OpenSubdiv::Far::PatchTable PatchTable;

    STParticles(Refiner const & refiner, PatchTable const *patchTable,
                int nparticles, bool centered=false);

    ~STParticles();

    void Update(float deltaTime);

    int GetNumParticles() const {
        return (int)_positions.size();
    }

    void SetSpeed(float speed) {
        _speed = std::max(-1.0f, std::min(1.0f, speed));
    }
    
    float GetSpeed() const {
        return _speed;
    }
    
    std::vector<Position> & GetPositions() {
        return _positions;
    }

    std::vector<float> & GetVelocities() {
        return _velocities;
    }

    std::vector<OpenSubdiv::Osd::PatchCoord> GetPatchCoords() const {
        return _patchCoords;
    }

    friend std::ostream & operator << (std::ostream & os, STParticles const & f);

    static void WarpParticle(std::vector<FaceInfo> const &adjacency,
                             int edge, Position * p, float * dp);

private:

    //
    // Particle "Dynamics"
    //
    std::vector<Position> _positions;

    std::vector<float> _velocities;

    std::vector<OpenSubdiv::Osd::PatchCoord> _patchCoords;

    float _speed;  // velocity multiplier

    friend std::ostream & operator << (std::ostream & os, FaceInfo const & f);


    std::vector<FaceInfo> _adjacency;
    OpenSubdiv::Far::PatchMap const *_patchMap;
};

#endif // ST_PARTICLES_H
