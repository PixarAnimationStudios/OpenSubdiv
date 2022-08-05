//
//   Copyright 2021 Pixar
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
#ifndef OPENSUBDIV3_REGRESSION_BFR_EVALUATE_TYPES_H
#define OPENSUBDIV3_REGRESSION_BFR_EVALUATE_TYPES_H

#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cassert>

//
//  Simple interpolatable struct for (x,y,z) positions and normals:
//
template <typename REAL>
struct Vec3 {
    Vec3<REAL>() { }
    Vec3<REAL>(REAL x, REAL y, REAL z) { p[0] = x, p[1] = y, p[2] = z; }

    //  Clear() and AddWithWeight() required for interpolation:
    void Clear( void * =0 ) { p[0] = p[1] = p[2] = 0.0f; }

    void AddWithWeight(Vec3<REAL> const & src, REAL weight) {
        p[0] += weight * src.p[0];
        p[1] += weight * src.p[1];
        p[2] += weight * src.p[2];
    }

    //  Element access via []:
    REAL const & operator[](int i) const { return p[i]; }
    REAL       & operator[](int i)       { return p[i]; }

    //  Element access via []:
    REAL const * Coords() const { return p; }
    REAL       * Coords()       { return p; }

    //  Additional useful mathematical operations:
    Vec3<REAL> operator-(Vec3<REAL> const & x) const {
        return Vec3<REAL>(p[0] - x.p[0], p[1] - x.p[1], p[2] - x.p[2]);
    }
    Vec3<REAL> operator+(Vec3<REAL> const & x) const {
        return Vec3<REAL>(p[0] + x.p[0], p[1] + x.p[1], p[2] + x.p[2]);
    }
    Vec3<REAL> operator*(REAL s) const {
        return Vec3<REAL>(p[0] * s, p[1] * s, p[2] * s);
    }
    Vec3<REAL> Cross(Vec3<REAL> const & x) const {
        return Vec3<REAL>(p[1]*x.p[2] - p[2]*x.p[1],
                          p[2]*x.p[0] - p[0]*x.p[2],
                          p[0]*x.p[1] - p[1]*x.p[0]);
    }
    REAL Dot(Vec3<REAL> const & x) const {
        return p[0]*x.p[0] + p[1]*x.p[1] + p[2]*x.p[2];
    }
    REAL Length() const {
        return std::sqrt(this->Dot(*this));
    }

    //  Static method to compute normal vector:
    static
    Vec3<REAL> ComputeNormal(Vec3<REAL> const & Du, Vec3<REAL> const & Dv,
                             REAL eps = 0.0f) {
        Vec3<REAL> N = Du.Cross(Dv);
        REAL lenSqrd = N.Dot(N);
        if (lenSqrd <= eps) return Vec3<REAL>(0.0f, 0.0f, 0.0f);
        return N * (1.0f / std::sqrt(lenSqrd));
    }

    //  Member variables (XYZ coordinates):
    REAL p[3];
};

typedef Vec3<float>  Vec3f;
typedef Vec3<double> Vec3d;


//
//  Simple struct to hold the results of a face evaluation:
//
template <typename REAL>
struct EvalResults {
    EvalResults() : evalPosition(true),
                    eval1stDeriv(true),
                    eval2ndDeriv(false),
                    evalUV(false),
                    useStencils(false) { }

    bool evalPosition;
    bool eval1stDeriv;
    bool eval2ndDeriv;
    bool evalUV;
    bool useStencils;

    std::vector< Vec3<REAL> > p;
    std::vector< Vec3<REAL> > du;
    std::vector< Vec3<REAL> > dv;
    std::vector< Vec3<REAL> > duu;
    std::vector< Vec3<REAL> > duv;
    std::vector< Vec3<REAL> > dvv;

    std::vector< Vec3<REAL> > uv;

    void Resize(int size) {
       if (evalPosition) {
            p.resize(size);
            if (eval1stDeriv) {
                du.resize(size);
                dv.resize(size);
                if (eval2ndDeriv) {
                    duu.resize(size);
                    duv.resize(size);
                    dvv.resize(size);
                }
            }
        }
        if (evalUV) {
            uv.resize(size);
        }
    }
};


//
//  Simple struct to hold the differences between two vectors:
//
template <typename REAL>
class VectorDelta {
public:
    typedef std::vector< Vec3<REAL> >  VectorVec3;

public:
    //  Member variables:
    std::vector< Vec3<REAL> > const * vectorA;
    std::vector< Vec3<REAL> > const * vectorB;

    int  numDeltas;
    REAL maxDelta;
    REAL tolerance;

public:
    VectorDelta(REAL epsilon = 0.0f) :
            vectorA(0), vectorB(0),
            numDeltas(0), maxDelta(0.0f),
            tolerance(epsilon) { }

    void
    Compare(VectorVec3 const & a, VectorVec3 const & b) {

        assert(a.size() == b.size());

        vectorA = &a;
        vectorB = &b;

        numDeltas = 0;
        maxDelta = 0.0f;

        for (size_t i = 0; i < a.size(); ++i) {
            REAL const * ai = a[i].Coords();
            REAL const * bi = b[i].Coords();

            REAL dx = std::abs(ai[0] - bi[0]);
            REAL dy = std::abs(ai[1] - bi[1]);
            REAL dz = std::abs(ai[2] - bi[2]);
            if ((dx > tolerance) || (dy > tolerance) || (dz > tolerance)) {
                ++ numDeltas;

                if (maxDelta < dx) maxDelta = dx;
                if (maxDelta < dy) maxDelta = dy;
                if (maxDelta < dz) maxDelta = dz;
            }
        }
    }
};

template <typename REAL>
class FaceDelta {
public:
    //  Member variables:
    bool hasDeltas;
    bool hasGeomDeltas;
    bool hasUVDeltas;

    int numPDeltas;
    int numD1Deltas;
    int numD2Deltas;
    int numUVDeltas;

    REAL maxPDelta;
    REAL maxD1Delta;
    REAL maxD2Delta;
    REAL maxUVDelta;

public:
    FaceDelta() { Clear(); }

    void Clear() {
        std::memset(this, 0, sizeof(*this));
    }

    void AddPDelta(VectorDelta<REAL> const & pDelta) {
        if (pDelta.numDeltas) {
            numPDeltas = pDelta.numDeltas;
            maxPDelta  = pDelta.maxDelta;
            hasDeltas = hasGeomDeltas = true;
        }
    }
    void AddDuDelta(VectorDelta<REAL> const & duDelta) {
        if (duDelta.numDeltas) {
            numD1Deltas += duDelta.numDeltas;
            maxD1Delta   = std::max(maxD1Delta, duDelta.maxDelta);
            hasDeltas = hasGeomDeltas = true;
        }
    }
    void AddDvDelta(VectorDelta<REAL> const & dvDelta) {
        if (dvDelta.numDeltas) {
            numD1Deltas += dvDelta.numDeltas;
            maxD1Delta   = std::max(maxD1Delta, dvDelta.maxDelta);
            hasDeltas = hasGeomDeltas = true;
        }
    }
    void AddDuuDelta(VectorDelta<REAL> const & duuDelta) {
        if (duuDelta.numDeltas) {
            numD2Deltas += duuDelta.numDeltas;
            maxD2Delta   = std::max(maxD2Delta, duuDelta.maxDelta);
            hasDeltas = hasGeomDeltas = true;
        }
    }
    void AddDuvDelta(VectorDelta<REAL> const & duvDelta) {
        if (duvDelta.numDeltas) {
            numD2Deltas += duvDelta.numDeltas;
            maxD2Delta   = std::max(maxD2Delta, duvDelta.maxDelta);
            hasDeltas = hasGeomDeltas = true;
        }
    }
    void AddDvvDelta(VectorDelta<REAL> const & dvvDelta) {
        if (dvvDelta.numDeltas) {
            numD2Deltas += dvvDelta.numDeltas;
            maxD2Delta   = std::max(maxD2Delta, dvvDelta.maxDelta);
            hasDeltas = hasGeomDeltas = true;
        }
    }
    void AddUVDelta(VectorDelta<REAL> const & uvDelta) {
        if (uvDelta.numDeltas) {
            numUVDeltas = uvDelta.numDeltas;
            maxUVDelta  = uvDelta.maxDelta;
            hasDeltas = hasUVDeltas = true;
        }
    }
};

template <typename REAL>
class MeshDelta {
public:
    //  Member variables:
    int numFacesWithDeltas;
    int numFacesWithGeomDeltas;
    int numFacesWithUVDeltas;

    int numFacesWithPDeltas;
    int numFacesWithD1Deltas;
    int numFacesWithD2Deltas;

    REAL maxPDelta;
    REAL maxD1Delta;
    REAL maxD2Delta;
    REAL maxUVDelta;

public:
    MeshDelta() { Clear(); }

    void Clear() {
        std::memset(this, 0, sizeof(*this));
    }

    void AddFace(FaceDelta<REAL> const & faceDelta) {

        numFacesWithDeltas     += faceDelta.hasDeltas;
        numFacesWithGeomDeltas += faceDelta.hasGeomDeltas;
        numFacesWithUVDeltas   += faceDelta.hasUVDeltas;

        numFacesWithPDeltas  += (faceDelta.numPDeltas  > 0);
        numFacesWithD1Deltas += (faceDelta.numD1Deltas > 0);
        numFacesWithD2Deltas += (faceDelta.numD2Deltas > 0);

        maxPDelta  = std::max(maxPDelta,  faceDelta.maxPDelta);
        maxD1Delta = std::max(maxD1Delta, faceDelta.maxD1Delta);
        maxD2Delta = std::max(maxD2Delta, faceDelta.maxD2Delta);
        maxUVDelta = std::max(maxUVDelta, faceDelta.maxUVDelta);
    }
};

#endif /* OPENSUBDIV3_REGRESSION_BFR_EVALUATE_TYPES_H */
