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

#include "../far/patchTable.h"
#include "../far/patchBasis.h"

#include <cstring>
#include <cstdio>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Far {

PatchTable::PatchTable(int maxvalence) :
    _maxValence(maxvalence),
    _localPointStencils(NULL),
    _localPointVaryingStencils(NULL) {
}

// Copy constructor
// XXXX manuelk we need to eliminate this constructor (C++11 smart pointers)
PatchTable::PatchTable(PatchTable const & src) :
    _maxValence(src._maxValence),
    _numPtexFaces(src._numPtexFaces),
    _patchArrays(src._patchArrays),
    _patchVerts(src._patchVerts),
    _paramTable(src._paramTable),
    _quadOffsetsTable(src._quadOffsetsTable),
    _vertexValenceTable(src._vertexValenceTable),
    _localPointStencils(NULL),
    _localPointVaryingStencils(NULL),
    _fvarChannels(src._fvarChannels),
    _sharpnessIndices(src._sharpnessIndices),
    _sharpnessValues(src._sharpnessValues) {

    if (src._localPointStencils) {
        _localPointStencils =
            new StencilTable(*src._localPointStencils);
    }
    if (src._localPointVaryingStencils) {
        _localPointVaryingStencils =
            new StencilTable(*src._localPointVaryingStencils);
    }
    if (! src._localPointFaceVaryingStencils.empty()) {
        _localPointFaceVaryingStencils.resize(src._localPointFaceVaryingStencils.size());
        for (int fvc=0; fvc<(int)_localPointFaceVaryingStencils.size(); ++fvc) {
            _localPointFaceVaryingStencils[fvc] =
                new StencilTable(*src._localPointFaceVaryingStencils[fvc]);
}
    }
}

PatchTable::~PatchTable() {
    delete _localPointStencils;
    delete _localPointVaryingStencils;
    for (int fvc=0; fvc<(int)_localPointFaceVaryingStencils.size(); ++fvc) {
        delete _localPointFaceVaryingStencils[fvc];
}
}

//
// PatchArrays
//
struct PatchTable::PatchArray {

    PatchArray(PatchDescriptor d, int np, Index v, Index p, Index qo) :
            desc(d), numPatches(np), vertIndex(v),
                patchIndex(p), quadOffsetIndex (qo) { }

    void print() const;

    PatchDescriptor desc;  // type of patches in the array

    int numPatches;        // number of patches in the array

    Index vertIndex,       // index to the first control vertex
          patchIndex,      // absolute index of the first patch in the array
          quadOffsetIndex; // index of the first quad offset entry

};

// debug helper
void
PatchTable::PatchArray::print() const {
    desc.print();
    printf("    numPatches=%d vertIndex=%d patchIndex=%d "
        "quadOffsetIndex=%d\n", numPatches, vertIndex, patchIndex,
            quadOffsetIndex);
}
inline PatchTable::PatchArray &
PatchTable::getPatchArray(Index arrayIndex) {
    assert(arrayIndex<(Index)GetNumPatchArrays());
    return _patchArrays[arrayIndex];
}
inline PatchTable::PatchArray const &
PatchTable::getPatchArray(Index arrayIndex) const {
    assert(arrayIndex<(Index)GetNumPatchArrays());
    return _patchArrays[arrayIndex];
}
void
PatchTable::reservePatchArrays(int numPatchArrays) {
    _patchArrays.reserve(numPatchArrays);
}

//
// FVarPatchChannel
//
// Stores a record for each patch in the primitive :
//
//  - Each patch in the PatchTable has a corresponding patch in each
//    face-varying patch channel. Patch vertex indices are sorted in the same
//    patch-type order as PatchTable::PTables. Face-varying data for a patch
//    can therefore be quickly accessed by using the patch primitive ID as
//    index into patchValueOffsets to locate the face-varying control vertex
//    indices.
//
//  - Face-varying channels can have a different interpolation modes
//
//  - Unlike "vertex" patches, there are no transition masks required
//    for face-varying patches.
//
//  - Face-varying patches still require boundary edge masks.
//
//  - currently most patches with sharp boundaries but smooth interiors have
//    to be isolated to level 10 : we need a special type of bicubic patch
//    similar to single-crease to resolve this condition without requiring
//    isolation if possible
//
struct PatchTable::FVarPatchChannel {

    Sdc::Options::FVarLinearInterpolation interpolation;

    PatchDescriptor desc;

    std::vector<Index> patchValues;
    std::vector<PatchParam> patchParam;
};

void
PatchTable::allocateVaryingVertices(
        PatchDescriptor desc, int numPatches) {
    _varyingDesc = desc;
    _varyingVerts.resize(numPatches*desc.GetNumControlVertices());
}

inline PatchTable::FVarPatchChannel &
PatchTable::getFVarPatchChannel(int channel) {
    assert(channel>=0 && channel<(int)_fvarChannels.size());
    return _fvarChannels[channel];
}
inline PatchTable::FVarPatchChannel const &
PatchTable::getFVarPatchChannel(int channel) const {
    assert(channel>=0 && channel<(int)_fvarChannels.size());
    return _fvarChannels[channel];
}
void
PatchTable::allocateFVarPatchChannels(int numChannels) {
    _fvarChannels.resize(numChannels);
}
void
PatchTable::allocateFVarPatchChannelValues(
        PatchDescriptor desc, int numPatches, int channel) {
    FVarPatchChannel & c = getFVarPatchChannel(channel);
    c.desc = desc;
    c.patchValues.resize(numPatches*desc.GetNumControlVertices());
    c.patchParam.resize(numPatches);
}
void
PatchTable::setFVarPatchChannelLinearInterpolation(
        Sdc::Options::FVarLinearInterpolation interpolation, int channel) {
    FVarPatchChannel & c = getFVarPatchChannel(channel);
    c.interpolation = interpolation;
}

//
// PatchTable
//

inline int
getPatchSize(PatchDescriptor desc) {
    return desc.GetNumControlVertices();
}

void
PatchTable::pushPatchArray(PatchDescriptor desc, int npatches,
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

int
PatchTable::getPatchIndex(int arrayIndex, int patchIndex) const {
    PatchArray const & pa = getPatchArray(arrayIndex);
    assert(patchIndex<pa.numPatches);
    return pa.patchIndex + patchIndex;
}
Index *
PatchTable::getSharpnessIndices(int arrayIndex) {
    return &_sharpnessIndices[getPatchArray(arrayIndex).patchIndex];
}

float *
PatchTable::getSharpnessValues(int arrayIndex) {
    return &_sharpnessValues[getPatchArray(arrayIndex).patchIndex];
}

PatchDescriptor
PatchTable::GetPatchDescriptor(PatchHandle const & handle) const {
    return getPatchArray(handle.arrayIndex).desc;
}

PatchDescriptor
PatchTable::GetPatchArrayDescriptor(int arrayIndex) const {
    return getPatchArray(arrayIndex).desc;
}

int
PatchTable::GetNumPatchArrays() const {
    return (int)_patchArrays.size();
}
int
PatchTable::GetNumPatches(int arrayIndex) const {
    return getPatchArray(arrayIndex).numPatches;
}
int
PatchTable::GetNumPatchesTotal() const {
    // there is one PatchParam record for each patch in the mesh
    return (int)_paramTable.size();
}
int
PatchTable::GetNumControlVertices(int arrayIndex) const {
    PatchArray const & pa = getPatchArray(arrayIndex);
    return pa.numPatches * getPatchSize(pa.desc);
}

Index
PatchTable::findPatchArray(PatchDescriptor desc) {
    for (int i=0; i<(int)_patchArrays.size(); ++i) {
        if (_patchArrays[i].desc==desc)
            return i;
    }
    return Vtr::INDEX_INVALID;
}
IndexArray
PatchTable::getPatchArrayVertices(int arrayIndex) {
    PatchArray const & pa = getPatchArray(arrayIndex);
    int size = getPatchSize(pa.desc);
    assert(pa.vertIndex<(Index)_patchVerts.size());
    return IndexArray(&_patchVerts[pa.vertIndex], pa.numPatches * size);
}
ConstIndexArray
PatchTable::GetPatchArrayVertices(int arrayIndex) const {
    PatchArray const & pa = getPatchArray(arrayIndex);
    int size = getPatchSize(pa.desc);
    assert(pa.vertIndex<(Index)_patchVerts.size());
    return ConstIndexArray(&_patchVerts[pa.vertIndex], pa.numPatches * size);
}

ConstIndexArray
PatchTable::GetPatchVertices(PatchHandle const & handle) const {
    PatchArray const & pa = getPatchArray(handle.arrayIndex);
    Index vert = pa.vertIndex + handle.vertIndex;
    return ConstIndexArray(&_patchVerts[vert], getPatchSize(pa.desc));
}
ConstIndexArray
PatchTable::GetPatchVertices(int arrayIndex, int patchIndex) const {
    PatchArray const & pa = getPatchArray(arrayIndex);
    int size = getPatchSize(pa.desc);
    assert((pa.vertIndex + patchIndex*size)<(Index)_patchVerts.size());
    return ConstIndexArray(&_patchVerts[pa.vertIndex + patchIndex*size], size);
}

PatchParam
PatchTable::GetPatchParam(PatchHandle const & handle) const {
    assert(handle.patchIndex < (Index)_paramTable.size());
    return _paramTable[handle.patchIndex];
}
PatchParam
PatchTable::GetPatchParam(int arrayIndex, int patchIndex) const {
    PatchArray const & pa = getPatchArray(arrayIndex);
    assert((pa.patchIndex + patchIndex) < (int)_paramTable.size());
    return _paramTable[pa.patchIndex + patchIndex];
}
PatchParamArray
PatchTable::getPatchParams(int arrayIndex) {
    PatchArray const & pa = getPatchArray(arrayIndex);
    return PatchParamArray(&_paramTable[pa.patchIndex], pa.numPatches);
}
ConstPatchParamArray const
PatchTable::GetPatchParams(int arrayIndex) const {
    PatchArray const & pa = getPatchArray(arrayIndex);
    return ConstPatchParamArray(&_paramTable[pa.patchIndex], pa.numPatches);
}

float
PatchTable::GetSingleCreasePatchSharpnessValue(PatchHandle const & handle) const {
    assert((handle.patchIndex) < (int)_sharpnessIndices.size());
    Index index = _sharpnessIndices[handle.patchIndex];
    if (index == Vtr::INDEX_INVALID) {
        return 0.0f;
    }
    assert(index < (Index)_sharpnessValues.size());
    return _sharpnessValues[index];
}
float
PatchTable::GetSingleCreasePatchSharpnessValue(int arrayIndex, int patchIndex) const {
    PatchArray const & pa = getPatchArray(arrayIndex);
    assert((pa.patchIndex + patchIndex) < (int)_sharpnessIndices.size());
    Index index = _sharpnessIndices[pa.patchIndex + patchIndex];
    if (index == Vtr::INDEX_INVALID) {
        return 0.0f;
    }
    assert(index < (Index)_sharpnessValues.size());
    return _sharpnessValues[index];
}

int
PatchTable::GetNumLocalPoints() const {
    return _localPointStencils ? _localPointStencils->GetNumStencils() : 0;
}
int
PatchTable::GetNumLocalPointsVarying() const {
    return _localPointVaryingStencils ? _localPointVaryingStencils->GetNumStencils() : 0;
}
int
PatchTable::GetNumLocalPointsFaceVarying(int channel) const {
    if (channel>=0 && channel<(int)_localPointFaceVaryingStencils.size() &&
        _localPointFaceVaryingStencils[channel]) {
        return _localPointFaceVaryingStencils[channel]->GetNumStencils();
    }
    return 0;
}

PatchTable::ConstQuadOffsetsArray
PatchTable::GetPatchQuadOffsets(PatchHandle const & handle) const {
    PatchArray const & pa = getPatchArray(handle.arrayIndex);
    return Vtr::ConstArray<unsigned int>(&_quadOffsetsTable[pa.quadOffsetIndex + handle.vertIndex], 4);
}
bool
PatchTable::IsFeatureAdaptive() const {

    // XXX:
    // revisit this function, since we'll add uniform cubic patches later.

    for (int i=0; i<GetNumPatchArrays(); ++i) {
        PatchDescriptor const & desc = _patchArrays[i].desc;
        if (desc.GetType()>=PatchDescriptor::REGULAR &&
            desc.GetType()<=PatchDescriptor::GREGORY_BASIS) {
            return true;
        }
    }
    return false;
}

ConstIndexArray
PatchTable::GetPatchVaryingVertices(PatchHandle const & handle) const {
    int numVaryingCVs = _varyingDesc.GetNumControlVertices();
    Index start = handle.patchIndex * numVaryingCVs;
    return ConstIndexArray(&_varyingVerts[start], numVaryingCVs);
}
ConstIndexArray
PatchTable::GetPatchVaryingVertices(int array, int patch) const {
    PatchArray const & pa = getPatchArray(array);
    int numVaryingCVs = _varyingDesc.GetNumControlVertices();
    Index start = (pa.patchIndex + patch) * numVaryingCVs;
    return ConstIndexArray(&_varyingVerts[start], numVaryingCVs);
}
ConstIndexArray
PatchTable::GetPatchArrayVaryingVertices(int array) const {
    PatchArray const & pa = getPatchArray(array);
    int numVaryingCVs = _varyingDesc.GetNumControlVertices();
    Index start = pa.patchIndex * numVaryingCVs;
    Index count = pa.numPatches * numVaryingCVs;
    return ConstIndexArray(&_varyingVerts[start], count);
}
ConstIndexArray
PatchTable::GetVaryingVertices() const {
    return ConstIndexArray(&_varyingVerts[0], (int)_varyingVerts.size());
}
IndexArray
PatchTable::getPatchArrayVaryingVertices(int arrayIndex) {
    PatchArray const & pa = getPatchArray(arrayIndex);
    int numVaryingCVs = _varyingDesc.GetNumControlVertices();
    Index start = pa.patchIndex * numVaryingCVs;
    return IndexArray(&_varyingVerts[start], pa.numPatches * numVaryingCVs);
}
void
PatchTable::populateVaryingVertices() {
    // In order to support evaluation of varying data we need to access
    // the varying values indexed by the zero ring vertices of the vertex
    // patch. This indexing is redundant for triangles and quads and
    // could be made redunant for other patch types if we reorganized
    // the vertex patch indices so that the zero ring indices always occured
    // first. This will also need to be updated when we add support for
    // triangle patches.
    int numVaryingCVs = _varyingDesc.GetNumControlVertices();
    for (int arrayIndex=0; arrayIndex<(int)_patchArrays.size(); ++arrayIndex) {
        PatchArray const & pa = getPatchArray(arrayIndex);
        PatchDescriptor::Type patchType = pa.desc.GetType();
        for (int patch=0; patch<pa.numPatches; ++patch) {
            ConstIndexArray vertexCVs = GetPatchVertices(arrayIndex, patch);
            int start = (pa.patchIndex + patch) * numVaryingCVs;
            if (patchType == PatchDescriptor::REGULAR) {
                _varyingVerts[start+0] = vertexCVs[5];
                _varyingVerts[start+1] = vertexCVs[6];
                _varyingVerts[start+2] = vertexCVs[10];
                _varyingVerts[start+3] = vertexCVs[9];
            } else if (patchType == PatchDescriptor::GREGORY_BASIS) {
                _varyingVerts[start+0] = vertexCVs[0];
                _varyingVerts[start+1] = vertexCVs[5];
                _varyingVerts[start+2] = vertexCVs[10];
                _varyingVerts[start+3] = vertexCVs[15];
            } else if (patchType == PatchDescriptor::QUADS) {
                _varyingVerts[start+0] = vertexCVs[0];
                _varyingVerts[start+1] = vertexCVs[1];
                _varyingVerts[start+2] = vertexCVs[2];
                _varyingVerts[start+3] = vertexCVs[3];
            } else if (patchType == PatchDescriptor::TRIANGLES) {
                _varyingVerts[start+0] = vertexCVs[0];
                _varyingVerts[start+1] = vertexCVs[1];
                _varyingVerts[start+2] = vertexCVs[2];
            }
        }
    }
}

int
PatchTable::GetNumFVarChannels() const {
    return (int)_fvarChannels.size();
}
Sdc::Options::FVarLinearInterpolation
PatchTable::GetFVarChannelLinearInterpolation(int channel) const {
    FVarPatchChannel const & c = getFVarPatchChannel(channel);
    return c.interpolation;
}
PatchDescriptor
PatchTable::GetFVarChannelPatchDescriptor(int channel) const {
    FVarPatchChannel const & c = getFVarPatchChannel(channel);
    return c.desc;
}
ConstIndexArray
PatchTable::GetFVarValues(int channel) const {
    FVarPatchChannel const & c = getFVarPatchChannel(channel);
    return ConstIndexArray(&c.patchValues[0], (int)c.patchValues.size());
}
IndexArray
PatchTable::getFVarValues(int channel) {
    FVarPatchChannel & c = getFVarPatchChannel(channel);
    return IndexArray(&c.patchValues[0], (int)c.patchValues.size());
}
ConstIndexArray
PatchTable::getPatchFVarValues(int patch, int channel) const {
    FVarPatchChannel const & c = getFVarPatchChannel(channel);
    int ncvs = c.desc.GetNumControlVertices();
    return ConstIndexArray(&c.patchValues[patch * ncvs], ncvs);
}
ConstIndexArray
PatchTable::GetPatchFVarValues(PatchHandle const & handle, int channel) const {
    return getPatchFVarValues(handle.patchIndex, channel);
}
ConstIndexArray
PatchTable::GetPatchFVarValues(int arrayIndex, int patchIndex, int channel) const {
    return getPatchFVarValues(getPatchIndex(arrayIndex, patchIndex), channel);
}
ConstIndexArray
PatchTable::GetPatchArrayFVarValues(int array, int channel) const {
    PatchArray const & pa = getPatchArray(array);
    FVarPatchChannel const & c = getFVarPatchChannel(channel);
    int ncvs = c.desc.GetNumControlVertices();
    int start = pa.patchIndex * ncvs;
    int count = pa.numPatches * ncvs;
    return ConstIndexArray(&c.patchValues[start], count);
}
PatchParam
PatchTable::getPatchFVarPatchParam(int patch, int channel) const {

    FVarPatchChannel const & c = getFVarPatchChannel(channel);
    return c.patchParam[patch];
}
PatchParam
PatchTable::GetPatchFVarPatchParam(PatchHandle const & handle, int channel) const {
    return getPatchFVarPatchParam(handle.patchIndex, channel);
}
PatchParam
PatchTable::GetPatchFVarPatchParam(int arrayIndex, int patchIndex, int channel) const {
    return getPatchFVarPatchParam(getPatchIndex(arrayIndex, patchIndex), channel);
}
ConstPatchParamArray
PatchTable::GetPatchArrayFVarPatchParam(int array, int channel) const {
    PatchArray const & pa = getPatchArray(array);
    FVarPatchChannel const & c = getFVarPatchChannel(channel);
    return ConstPatchParamArray(&c.patchParam[pa.patchIndex], pa.numPatches);
}
ConstPatchParamArray
PatchTable::GetFVarPatchParam(int channel) const {
    FVarPatchChannel const & c = getFVarPatchChannel(channel);
    return ConstPatchParamArray(&c.patchParam[0], (int)c.patchParam.size());
}
PatchParamArray
PatchTable::getFVarPatchParam(int channel) {
    FVarPatchChannel & c = getFVarPatchChannel(channel);
    return PatchParamArray(&c.patchParam[0], (int)c.patchParam.size());
}

void
PatchTable::print() const {
    printf("patchTable (0x%p)\n", this);
    printf("  numPatches = %d\n", GetNumPatchesTotal());
    for (int i=0; i<GetNumPatchArrays(); ++i) {
        printf("  patchArray %d:\n", i);
        PatchArray const & pa = getPatchArray(i);
        pa.print();
    }
}

//
//  Evaluate basis functions for vertex and derivatives at (s,t):
//
void
PatchTable::EvaluateBasis(
    PatchHandle const & handle, float s, float t,
    float wP[], float wDs[], float wDt[],
    float wDss[], float wDst[], float wDtt[]) const {

    PatchDescriptor::Type patchType = GetPatchArrayDescriptor(handle.arrayIndex).GetType();
    PatchParam const & param = _paramTable[handle.patchIndex];

    if (patchType == PatchDescriptor::REGULAR) {
        internal::GetBSplineWeights(param, s, t, wP, wDs, wDt, wDss, wDst, wDtt);
    } else if (patchType == PatchDescriptor::GREGORY_BASIS) {
        internal::GetGregoryWeights(param, s, t, wP, wDs, wDt, wDss, wDst, wDtt);
    } else if (patchType == PatchDescriptor::QUADS) {
        internal::GetBilinearWeights(param, s, t, wP, wDs, wDt, wDss, wDst, wDtt);
    } else {
        assert(0);
    }
}

//
//  Evaluate basis functions for varying and derivatives at (s,t):
//
void
PatchTable::EvaluateBasisVarying(
    PatchHandle const & handle, float s, float t,
    float wP[], float wDs[], float wDt[],
    float wDss[], float wDst[], float wDtt[]) const {

    PatchParam const & param = _paramTable[handle.patchIndex];

    internal::GetBilinearWeights(param, s, t, wP, wDs, wDt, wDss, wDst, wDtt);
}

//
//  Evaluate basis functions for face-varying and derivatives at (s,t):
//
void
PatchTable::EvaluateBasisFaceVarying(
    PatchHandle const & handle, float s, float t,
    float wP[], float wDs[], float wDt[],
    float wDss[], float wDst[], float wDtt[],
    int channel) const {

    PatchParam param = GetPatchFVarPatchParam(handle.arrayIndex, handle.patchIndex, channel);
    PatchDescriptor::Type patchType = param.IsRegular()
            ? PatchDescriptor::REGULAR
            : GetFVarChannelPatchDescriptor(channel).GetType();

    if (patchType == PatchDescriptor::REGULAR) {
        internal::GetBSplineWeights(param, s, t, wP, wDs, wDt, wDss, wDst, wDtt);
    } else if (patchType == PatchDescriptor::GREGORY_BASIS) {
        internal::GetGregoryWeights(param, s, t, wP, wDs, wDt, wDss, wDst, wDtt);
    } else if (patchType == PatchDescriptor::QUADS) {
        internal::GetBilinearWeights(param, s, t, wP, wDs, wDt, wDss, wDst, wDtt);
    } else {
        assert(0);
    }
}


} // end namespace Far

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv
