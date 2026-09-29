//
//   Copyright 2021 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#include "../bfr/parameterization.h"
#include "../bfr/limits.h"
#include "../sdc/types.h"

#include <cmath>
#include <cassert>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Bfr {

//
//  Construction:
//
namespace {
    //
    //  Simple internal utilities supporting sub-face parameterizations:
    //
    inline bool
    isQuadSubFaceSizeValid(int faceSize) {
        return (faceSize > 2) && (faceSize <= Limits::MaxFaceSize()) &&
               (faceSize != 4);
    }

    //
    //  The quad sub-face parameterization uses the integer square root
    //  of the face size for its tiling to preserve accuracy. Computation
    //  here avoids use of sqrt() for common lower face sizes:
    //
    inline int
    computeQuadSubFaceUDim(int faceSize) {
        return (faceSize < 10) ? (2 + (faceSize > 4)) :
                                 (1 + (int) std::sqrt((float)(faceSize - 1)));
    }
}

Parameterization::Parameterization(Sdc::SchemeType scheme, int faceSize) :
        _type(0), _uDim(0), _faceSize(0) {

    int regFaceSize = Sdc::SchemeTypeTraits::GetRegularFaceSize(scheme);

    if (regFaceSize == 3) {
        //  Triangular schemes currently require triangular faces:
        _type     = (unsigned char) TRI;
        _faceSize = (faceSize == 3) ? 3 : 0;
    } else if (faceSize == 4) {
        //  Quad schemes with a quad face:
        _type     = (unsigned char) QUAD;
        _faceSize = 4;
    } else if (isQuadSubFaceSizeValid(faceSize)) {
        //  Quad schemes with a non-quad face (of valid face size):
        _type     = (unsigned char)  QUAD_SUBFACES;
        _faceSize = (unsigned short) faceSize;
        _uDim     = (unsigned char)  computeQuadSubFaceUDim(faceSize);
    }
}

Parameterization::Parameterization(Type type, int numSubFaces) :
        _type((unsigned char) type), _uDim(0), _faceSize(0) {

    switch (type) {
    case TRI:
        //  Any optionally specified sub-face count is ignored:
        _faceSize = 3;
        break;
    case QUAD:
        //  Any optionally specified sub-face count is ignored:
        _faceSize = 4;
        break;
    case QUAD_SUBFACES:
        //  Number of sub-faces is required here and must be valid:
        if (isQuadSubFaceSizeValid(numSubFaces)) {
            _faceSize = (unsigned short) numSubFaces;
            _uDim     = (unsigned char)  computeQuadSubFaceUDim(numSubFaces);
        }
        break;
    }
}

//
//  Simple coordinate queries:
//
template <typename REAL>
void
Parameterization::GetVertexCoord(int vertex, REAL uv[2]) const {

    switch (GetType()) {
    case QUAD:
        uv[0] = (REAL) (vertex && (vertex < 3));
        uv[1] = (REAL) (vertex > 1);
        break;
    case TRI:
        uv[0] = (REAL) (vertex == 1);
        uv[1] = (REAL) (vertex == 2);
        break;
    case QUAD_SUBFACES:
        uv[0] = (REAL) (vertex % _uDim);
        uv[1] = (REAL) (vertex / _uDim);
        break;
    default:
        uv[0] = -1.0f;
        uv[1] = -1.0f;
        break;
    }
}

template <typename REAL>
void
Parameterization::GetEdgeCoord(int edge, REAL t, REAL uv[2]) const {

    switch (GetType()) {
    case QUAD:
        switch (edge) {
        case 0: uv[0] = t;        uv[1] = 0.0f;     break;
        case 1: uv[0] = 1.0f;     uv[1] = t;        break;
        case 2: uv[0] = 1.0f - t; uv[1] = 1.0f;     break;
        case 3: uv[0] = 0.0f;     uv[1] = 1.0f - t; break;
        }
        break;

    case TRI:
        switch (edge) {
        case 0: uv[0] = t;        uv[1] = 0.0f;     break;
        case 1: uv[0] = 1.0f - t; uv[1] = t;        break;
        case 2: uv[0] = 0.0f;     uv[1] = 1.0f - t; break;
        }
        break;

    case QUAD_SUBFACES:
        if (t < 0.5f) {
            GetVertexCoord(edge, uv);
            uv[0] += t;
        } else {
            GetVertexCoord((edge + 1) % _faceSize, uv);
            uv[1] += 1.0f - t;
        }
        break;
    default:
        uv[0] = -1.0f;
        uv[1] = -1.0f;
        break;
    }
}

template <typename REAL>
void
Parameterization::GetCenterCoord(REAL uv[2]) const {

    if (GetType() == TRI) {
        uv[0] = 1.0f / 3.0f;
        uv[1] = 1.0f / 3.0f;
    } else {
        uv[0] = 0.5f;
        uv[1] = 0.5f;
    }
}

//
//  Private sub-face coordinate conversion methods used externally:
//
template <typename REAL>
int
Parameterization::convertCoordToSubFace(bool normalized,
        REAL const uvCoord[2], REAL subCoord[2]) const {

    assert(HasSubFaces());

    //  Include an interval around the domain when identifying the subface:
    int uTile = (int) (uvCoord[0] + 0.25f);
    int vTile = (int) (uvCoord[1] + 0.25f);

    //  Clamp tiles in the U and V direction to valid sub-face tiles on
    //  the boundaries of the rectangular subset. This is trivial in the
    //  U direction but less so in V as the top row is usually not full
    //  (so max in V is greater for the first colums in U):
    uTile = std::max(0, std::min(uTile, _uDim - 1));

    vTile = std::max(0, vTile);
    if ((vTile * _uDim + uTile) >= _faceSize) {
        //  Only clamp upper bound in V when tile exceeds face size:
        vTile = (_faceSize / _uDim) - 1 + (uTile < (_faceSize % _uDim));
    }

    //  Be sure to support in-place conversion (i.e. uvCoord == subCoord):
    subCoord[0] = uvCoord[0] - (REAL) uTile;
    subCoord[1] = uvCoord[1] - (REAL) vTile;
    if (normalized) {
        subCoord[0] *= 2.0f;
        subCoord[1] *= 2.0f;
    }
    return vTile * _uDim + uTile;
}

template <typename REAL>
void
Parameterization::convertSubFaceToCoord(bool normalized, int subFace,
        REAL const subCoord[2], REAL uvCoord[2]) const {

    assert(HasSubFaces());

    int uTile = subFace % _uDim;
    int vTile = subFace / _uDim;

    //  Be sure this assignment always supports in-place conversion:
    if (normalized) {
        uvCoord[0] = (REAL) uTile + subCoord[0] * 0.5f;
        uvCoord[1] = (REAL) vTile + subCoord[1] * 0.5f;
    } else {
        uvCoord[0] = (REAL) uTile + subCoord[0];
        uvCoord[1] = (REAL) vTile + subCoord[1];
    }
}

//
//  Explicit instantiation of template methods for <REAL>:
//
//  Coordinate queries:
template void
Parameterization::GetVertexCoord<float>(int, float uv[2]) const;
template void
Parameterization::GetEdgeCoord<float>(int, float, float uv[2]) const;
template void
Parameterization::GetCenterCoord<float>(float uv[2]) const;

template void
Parameterization::GetVertexCoord<double>(int, double uv[2]) const;
template void
Parameterization::GetEdgeCoord<double>(int, double, double uv[2]) const;
template void
Parameterization::GetCenterCoord<double>(double uv[2]) const;

//  Sub-face conversions:
template int
Parameterization::convertCoordToSubFace<float>(bool,
                        float const uvIn[2], float uvOut[2]) const;
template void
Parameterization::convertSubFaceToCoord<float>(bool, int,
                        float const uvIn[2], float uvOut[2]) const;

template int
Parameterization::convertCoordToSubFace<double>(bool,
                        double const uvIn[2], double uvOut[2]) const;
template void
Parameterization::convertSubFaceToCoord<double>(bool, int,
                        double const uvIn[2], double uvOut[2]) const;

} // end namespace Bfr

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv
