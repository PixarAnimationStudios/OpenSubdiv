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

#include "../far/patchTables.h"
#include "../far/stencilTables.h"

#include <cstring>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Far {

static void
getBezierWeights(float t, float point[4], float deriv[3]) {

    // The weights for the four uniform cubic Bezier basis functions are:
    // (1 - t)^3
    // 3 * t * (1-t)
    // 3 * t^2 * (1-t)
    // t^3
    float t2 = t*t,
          w0 = 1.0f - t,
          w2 = w0 * w0;

    assert(point);
    point[0] = w0*w2;
    point[1] = 3.0f * t * w2;
    point[2] = 3.0f * t2 * w0;
    point[3] = t * t2;

    // The weights for the three uniform quadratic basis functions are:
    // (1-t)^2
    // 2 * t * (1-t)
    // t^2
    if (deriv) {
        deriv[0] = w2;
        deriv[1] = 2.0f * t * w0;
        deriv[2] = t2;
    }
}

static void
getBSplineWeights(float t, float point[4], float deriv[3]) {

    // The weights for the four uniform cubic B-Spline basis functions are:
    // (1/6)(1 - t)^3
    // (1/6)(3t^3 - 6t^2 + 4)
    // (1/6)(-3t^3 + 3t^2 + 3t + 1)
    // (1/6)t^3
    float t2 = t*t,
          t3 = 3.0f*t2*t,
          w0 = 1.0f-t;

    assert(point);
    point[0] = (w0*w0*w0) / 6.0f;
    point[1] = (t3 - 6.0f*t2 + 4.0f) / 6.0f;
    point[2] = (3.0f*t2 - t3 + 3.0f*t + 1.0f) / 6.0f;
    point[3] = t3 / 18.0f;


    // The weights for the three uniform quadratic basis functions are:
    // (1/2)(1-t)^2
    // (1/2)(1 + 2t - 2t^2)
    // (1/2)t^2
    if (deriv) {
        deriv[0] = 0.5f * w0 * w0;
        deriv[1] = 0.5f + t - t2;
        deriv[2] = 0.5f * t2;
    }
}

void
getBoxSplineWeights(float v, float w, float B[12]) {

    float u = 1.0f - v - w;

    //
    //  The 12 basis functions of the quartic box spline (unscaled by their common
    //  factor of 1/12 until later, and formatted to make it easy to spot any
    //  typing errors):
    //
    //      15 terms for the 3 points above the triangle corners
    //       9 terms for the 3 points on faces opposite the triangle edges
    //       2 terms for the 6 points on faces opposite the triangle corners
    //
    //  Powers of each variable for notational convenience:
    float u2 = u*u;
    float u3 = u*u2;
    float u4 = u*u3;
    float v2 = v*v;
    float v3 = v*v2;
    float v4 = v*v3;
    float w2 = w*w;
    float w3 = w*w2;
    float w4 = w*w3;

    //  And now the basis functions:
    B[ 0] = u4 + 2.0f*u3*v;
    B[ 1] = u4 + 2.0f*u3*w;
    B[ 8] = w4 + 2.0f*w3*u;
    B[11] = w4 + 2.0f*w3*v;
    B[ 9] = v4 + 2.0f*v3*w;
    B[ 5] = v4 + 2.0f*v3*u;

    B[ 2] = u4 + 2.0f*u3*w + 6.0f*u3*v + 6.0f*u2*v*w + 12.0f*u2*v2 +
            v4 + 2.0f*v3*w + 6.0f*v3*u + 6.0f*v2*u*w;
    B[ 4] = w4 + 2.0f*w3*v + 6.0f*w3*u + 6.0f*w2*u*v + 12.0f*w2*u2 +
            u4 + 2.0f*u3*v + 6.0f*u3*w + 6.0f*u2*v*w;
    B[10] = v4 + 2.0f*v3*u + 6.0f*v3*w + 6.0f*v2*w*u + 12.0f*v2*w2 +
            w4 + 2.0f*w3*u + 6.0f*w3*v + 6.0f*w3*u*v;

    B[ 3] = v4 + 6*v3*w + 8*v3*u + 36*v2*w*u + 24*v2*u2 + 24*v*u3 +
            w4 + 6*w3*v + 8*w3*u + 36*w2*v*u + 24*w2*u2 + 24*w*u3 + 6*u4 + 60*u2*v*w + 12*v2*w2;
    B[ 6] = w4 + 6*w3*u + 8*w3*v + 36*w2*u*v + 24*w2*v2 + 24*w*v3 +
            u4 + 6*u3*w + 8*u3*v + 36*u2*v*w + 24*u2*v2 + 24*u*v3 + 6*v4 + 60*v2*w*u + 12*w2*u2;
    B[ 7] = u4 + 6*u3*v + 8*u3*w + 36*u2*v*w + 24*u2*w2 + 24*u*w3 +
            v4 + 6*v3*u + 8*v3*w + 36*v2*u*w + 24*v2*w2 + 24*v*w3 + 6*w4 + 60*w2*u*v + 12*u2*v2;

    for (int i = 0; i < 12; ++i) {
        B[i] *= 1.0f / 12.0f;
    }
}

void
PatchTables::GetBasisWeights(TensorBasis basis, PatchParam::BitField bits,
    float s, float t, float point[16], float deriv1[16], float deriv2[16]) {

    static int const rots[4][16] =
        { { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
          { 12, 8, 4, 0, 13, 9, 5, 1, 14, 10, 6, 2, 15, 11, 7, 3 },
          { 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 },
          { 3, 7, 11, 15, 2, 6, 10, 14, 1, 5, 9, 13, 0, 4, 8, 12 } };

    assert(bits.GetRotation()<4);
    int const * rot = rots[bits.GetRotation()];

    float sWeights[4], tWeights[4], d1Weights[3], d2Weights[3];

    if (basis==BASIS_BSPLINE) {
        getBSplineWeights(s, point ? sWeights : 0, deriv1 ? d1Weights : 0);
        getBSplineWeights(t, point ? tWeights : 0, deriv2 ? d2Weights : 0);
    } else if (basis==BASIS_BEZIER) {
        getBezierWeights(s, point ? sWeights : 0, deriv1 ? d1Weights : 0);
        getBezierWeights(t, point ? tWeights : 0, deriv2 ? d2Weights : 0);
    } else {
        assert(0);
    }

    if (point) {
        // Compute the tensor product weight corresponding to each control
        // vertex
        memset(point,  0, 16*sizeof(float));
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                point[rot[4*i+j]] += sWeights[j] * tWeights[i];
            }
        }
    }

    if (deriv1 and deriv2) {
        // Compute the tangent stencil. This is done by taking the tensor
        // product between the quadratic weights computed for s and the cubic
        // weights computed for t. The stencil is constructed using
        // differences between consecutive vertices in each row (i.e.
        // in the s direction).
        memset(deriv1, 0, 16*sizeof(float));
        for (int i = 0, k = 0; i < 4; ++i) {
            float prevWeight = 0.0f;
            for (int j = 0; j < 3; ++j) {
                float weight = d1Weights[j]*tWeights[i];
                deriv1[rot[k++]] += prevWeight - weight;
                prevWeight = weight;
            }
            deriv1[rot[k++]]+=prevWeight;
        }

        memset(deriv2, 0, 16*sizeof(float));
#define FASTER_TENSOR
#ifdef FASTER_TENSOR
        // XXXX manuelk this might be slightly more efficient ?
        float dW[4];
        dW[0] = - d2Weights[0];
        dW[1] = d2Weights[0] - d2Weights[1];
        dW[2] = d2Weights[1] - d2Weights[2];
        dW[3] = d2Weights[2];
        for (int i = 0, k = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                deriv2[rot[k++]] = sWeights[j] * dW[i];
            }
        }
#else
        for (int j = 0; j < 4; ++j) {
            float prevWeight = 0.0f;
            for (int i = 0; i < 3; ++i) {
                float weight = sWeights[j]*d2Weights[i];
                deriv2[rot[4*i+j]]+=prevWeight - weight;
                prevWeight = weight;
            }
            deriv2[rot[12+j]] += prevWeight;
        }
#endif
        // Scale derivatives up based on level of subdivision
        float scale = float(1 << bits.GetDepth());
        for (int k=0; k<16; ++k) {
            deriv1[k] *= scale;
            deriv2[k] *= scale;
        }
    }
}

PatchTables::PatchTables(int maxvalence) :
    _maxValence(maxvalence), _endcapStencilTables(0), _fvarPatchTables(0) { }

// Copy constructor
// XXXX manuelk we need to eliminate this constructor (C++11 smart pointers)
PatchTables::PatchTables(PatchTables const & src) :
    _maxValence(src._maxValence),
    _numPtexFaces(src._numPtexFaces),
    _patchArrays(src._patchArrays),
    _patchVerts(src._patchVerts),
    _paramTable(src._paramTable),
#ifdef ENDCAP_TOPOPOLGY
    _endcapTopology(src._endcapTopology),
#endif
    _quadOffsetsTable(src._quadOffsetsTable),
    _vertexValenceTable(src._vertexValenceTable),
    _sharpnessIndices(src._sharpnessIndices),
    _sharpnessValues(src._sharpnessValues) {

    _endcapStencilTables = src._endcapStencilTables ?
        new StencilTables(*src._endcapStencilTables) : 0;

    _fvarPatchTables = src._fvarPatchTables ?
        new FVarPatchTables(*src._fvarPatchTables) : 0;
}

PatchTables::~PatchTables() {
    delete _endcapStencilTables;
    delete _fvarPatchTables;
}

//
// PatchArrays
//

struct PatchTables::PatchArray {

    PatchArray(PatchDescriptor d, int np, Index v, Index p, Index qo) :
            desc(d), numPatches(np), vertIndex(v),
                patchIndex(p), quadOffsetIndex (qo) { }

    PatchDescriptor desc;  // type of patches in the array

    int numPatches;          // number of patches in the array

    Index vertIndex,       // index to the first control vertex
          patchIndex,      // index of the first patch in the array
          quadOffsetIndex; // index of the first quad offset entry
};

inline PatchTables::PatchArray &
PatchTables::getPatchArray(Index arrayIndex) {
    assert(arrayIndex<(Index)GetNumPatchArrays());
    return _patchArrays[arrayIndex];
}

inline PatchTables::PatchArray const &
PatchTables::getPatchArray(Index arrayIndex) const {
    assert(arrayIndex<(Index)GetNumPatchArrays());
    return _patchArrays[arrayIndex];
}

void
PatchTables::reservePatchArrays(int numPatchArrays) {
    _patchArrays.reserve(numPatchArrays);
}

inline int
getPatchSize(PatchDescriptor desc) {
    int size = desc.GetNumControlVertices();
    // XXXX manuelk we do not store the topology for Gregory Basis
    // patch types yet - so point to the 4 corners of the 0-ring
    if (desc.GetType() == PatchDescriptor::GREGORY_BASIS) {
        size = 4;
    }
    return size;
}

void
PatchTables::pushPatchArray(PatchDescriptor desc, int npatches,
    Index * vidx, Index * pidx, Index * qoidx) {

    if (npatches>0) {
        _patchArrays.push_back(PatchArray(
            desc, npatches, *vidx, *pidx, qoidx ? *qoidx : 0));
        int nverts = getPatchSize(desc);
        *vidx += npatches * nverts;
        *pidx += npatches;
        if (qoidx) {
            *qoidx += (desc.GetType() == PatchDescriptor::GREGORY) ?
                npatches*nverts  : 0;
        }
    }
}

Index *
PatchTables::getSharpnessIndices(int arrayIndex) {
    return &_sharpnessIndices[getPatchArray(arrayIndex).patchIndex];
}

float *
PatchTables::getSharpnessValues(int arrayIndex) {
    return &_sharpnessValues[getPatchArray(arrayIndex).patchIndex];
}

PatchDescriptor
PatchTables::GetPatchDescriptor(PatchHandle const & handle) const {
    return getPatchArray(handle.arrayIndex).desc;
}

PatchDescriptor
PatchTables::GetPatchArrayDescriptor(int arrayIndex) const {
    return getPatchArray(arrayIndex).desc;
}

int
PatchTables::GetNumPatchArrays() const {
    return (int)_patchArrays.size();
}
int
PatchTables::GetNumPatches(int arrayIndex) const {
    return getPatchArray(arrayIndex).numPatches;
}
int
PatchTables::GetNumControlVertices(int arrayIndex) const {
    PatchArray const & pa = getPatchArray(arrayIndex);
    return pa.numPatches * getPatchSize(pa.desc);
}

IndexArray
PatchTables::getPatchArrayVertices(int arrayIndex) {
    PatchArray const & pa = getPatchArray(arrayIndex);
    int size = getPatchSize(pa.desc);
    assert(pa.vertIndex<(Index)_patchVerts.size());
    return IndexArray(&_patchVerts[pa.vertIndex], pa.numPatches * size);
}
ConstIndexArray
PatchTables::GetPatchArrayVertices(int arrayIndex) const {
    PatchArray const & pa = getPatchArray(arrayIndex);
    int size = getPatchSize(pa.desc);
    assert(pa.vertIndex<(Index)_patchVerts.size());
    return ConstIndexArray(&_patchVerts[pa.vertIndex], pa.numPatches * size);
}

ConstIndexArray
PatchTables::GetPatchVertices(PatchHandle const & handle) const {
    PatchArray const & pa = getPatchArray(handle.arrayIndex);

    Index vert = pa.vertIndex;
    // XXXX manuelk we do not store the topology for Gregory Basis
    // patch types yet - so point to the 4 corners of the 0-ring
    vert += (pa.desc.GetType() == PatchDescriptor::GREGORY_BASIS) ?
        handle.vertIndex / 5 : handle.vertIndex;
    assert(vert<(Index)_patchVerts.size());
    return ConstIndexArray(&_patchVerts[vert], getPatchSize(pa.desc));
}
ConstIndexArray
PatchTables::GetPatchVertices(int arrayIndex, int patchIndex) const {
    PatchArray const & pa = getPatchArray(arrayIndex);
    int size = getPatchSize(pa.desc);
    assert((pa.vertIndex + patchIndex*size)<(Index)_patchVerts.size());
    return ConstIndexArray(&_patchVerts[pa.vertIndex + patchIndex*size], size);
}

PatchParam
PatchTables::GetPatchParam(PatchHandle const & handle) const {
    assert(handle.patchIndex < (Index)_paramTable.size());
    return _paramTable[handle.patchIndex];
}
PatchParam
PatchTables::GetPatchParam(int arrayIndex, int patchIndex) const {
    PatchArray const & pa = getPatchArray(arrayIndex);
    assert((pa.patchIndex + patchIndex) < (int)_paramTable.size());
    return _paramTable[pa.patchIndex + patchIndex];
}
PatchParamArray
PatchTables::getPatchParams(int arrayIndex) {
    PatchArray const & pa = getPatchArray(arrayIndex);
    return PatchParamArray(&_paramTable[pa.patchIndex], pa.numPatches);
}
ConstPatchParamArray const
PatchTables::GetPatchParams(int arrayIndex) const {
    PatchArray const & pa = getPatchArray(arrayIndex);
    return ConstPatchParamArray(&_paramTable[pa.patchIndex], pa.numPatches);
}

float
PatchTables::GetSingleCreasePatchSharpnessValue(PatchHandle const & handle) const {
    assert((handle.patchIndex) < (int)_sharpnessIndices.size());
    Index index = _sharpnessIndices[handle.patchIndex];
    if (index == Vtr::INDEX_INVALID) {
        return 0.0f;
    }
    assert(index < (Index)_sharpnessValues.size());
    return _sharpnessValues[index];
}
float
PatchTables::GetSingleCreasePatchSharpnessValue(int arrayIndex, int patchIndex) const {
    PatchArray const & pa = getPatchArray(arrayIndex);
    assert((pa.patchIndex + patchIndex) < (int)_sharpnessIndices.size());
    Index index = _sharpnessIndices[pa.patchIndex + patchIndex];
    if (index == Vtr::INDEX_INVALID) {
        return 0.0f;
    }
    assert(index < (Index)_sharpnessValues.size());
    return _sharpnessValues[index];
}

PatchTables::ConstQuadOffsetsArray
PatchTables::GetPatchQuadOffsets(PatchHandle const & handle) const {
    PatchArray const & pa = getPatchArray(handle.arrayIndex);
    return Vtr::ConstArray<unsigned int>(&_quadOffsetsTable[pa.quadOffsetIndex + handle.vertIndex], 4);
}

IndexArray
PatchTables::getFVarVerts(int arrayIndex, int channel) {
    PatchArray const & pa = getPatchArray(arrayIndex);
    assert(_fvarPatchTables and (channel<(int)_fvarPatchTables->_channels.size()));
    std::vector<Index> & verts = _fvarPatchTables->_channels[channel].patchVertIndices;
    int ofs = pa.patchIndex * pa.desc.GetNumFVarControlVertices();
    return IndexArray(&verts[ofs],pa.numPatches * pa.desc.GetNumFVarControlVertices());
}

bool
PatchTables::IsFeatureAdaptive() const {

    // check for presence of tables only used by adaptive patches
    if (not _vertexValenceTable.empty() or _endcapStencilTables)
        return true;

    // otherwise, we have to check each patch array
    for (int i=0; i<GetNumPatchArrays(); ++i) {
        PatchDescriptor const & desc = _patchArrays[i].desc;
        if (desc.GetType()>=PatchDescriptor::REGULAR and
            desc.GetType()<=PatchDescriptor::GREGORY_BASIS) {
            return true;
        }
    }
    return false;
}

int
PatchTables::GetNumPatchesTotal() const {
    // there is one PatchParam record for each patch in the mesh
    return (int)_paramTable.size();
}

// Returns the first array of patches matching the descriptor
Index
PatchTables::findPatchArray(PatchDescriptor desc) {
    for (int i=0; i<(int)_patchArrays.size(); ++i) {
        if (_patchArrays[i].desc==desc)
            return i;
    }
    return Vtr::INDEX_INVALID;
}

} // end namespace Far

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv
