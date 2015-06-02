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
}

PatchTable::~PatchTable() {
    delete _localPointStencils;
    delete _localPointVaryingStencils;
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
          patchIndex,      // index of the first patch in the array
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
//  - Unlike "vertex" PatchTable, there are no "transition" patterns required
//    for face-varying patches.
//
//  - No transition patterns means vertex indices of face-varying patches can
//    be pre-rotated in the factory, so we do not store patch rotation
//
//  - Face-varying patches still special variants for boundary and corner cases
//
//  - currently most patches with sharp boundaries but smooth interiors have
//    to be isolated to level 10 : we need a special type of bicubic patch
//    similar to single-crease to resolve this condition without requiring
//    isolation if possible
//
struct PatchTable::FVarPatchChannel {

    // Channel interpolation mode
    Sdc::Options::FVarLinearInterpolation interpolation;

    // Patch type
    //
    // Note : in bilinear interpolation modes, all patches are of the same type,
    // so we only need a single type (patchesType). In bi-cubic modes, each
    // patch requires its own type (patchTypes).
    PatchDescriptor::Type              patchesType;
    std::vector<PatchDescriptor::Type> patchTypes;

    // Patch points values
    std::vector<Index> patchValuesOffsets; // offset to the first value of each patch
    std::vector<Index> patchValues; // point values for each patch
};

inline PatchTable::FVarPatchChannel &
PatchTable::getFVarPatchChannel(int channel) {
    assert(channel<(int)_fvarChannels.size());
    return _fvarChannels[channel];
}
inline PatchTable::FVarPatchChannel const &
PatchTable::getFVarPatchChannel(int channel) const {
    assert(channel<(int)_fvarChannels.size());
    return _fvarChannels[channel];
}
void
PatchTable::allocateFVarPatchChannels(int numChannels) {
    _fvarChannels.resize(numChannels);
}
void
PatchTable::allocateChannelValues(int channel,
    int numPatches, int numVerticesTotal) {

    FVarPatchChannel & c = getFVarPatchChannel(channel);
    if (c.interpolation==Sdc::Options::FVAR_LINEAR_ALL) {
        // Allocate bi-linear channels (allows uniform topology to be populated
        // in a single traversal)
        c.patchValues.resize(numVerticesTotal);
    } else {
        // Allocate per-patch type and offset vectors for bi-cubic patches
        //
        // Note : c.patchValues cannot be allocated pre-emptively since we do
        // not know the type (and size) of each patch yet. These channels
        // require an extra step to compact the value indices and generate
        // offsets
        c.patchesType = PatchDescriptor::NON_PATCH;
        c.patchTypes.resize(numPatches);
    }
}
void
PatchTable::setFVarPatchChannelLinearInterpolation(int channel,
        Sdc::Options::FVarLinearInterpolation interpolation) {
    FVarPatchChannel & c = getFVarPatchChannel(channel);
    c.interpolation = interpolation;
}
void
PatchTable::setFVarPatchChannelPatchesType(int channel, PatchDescriptor::Type type) {
    FVarPatchChannel & c = getFVarPatchChannel(channel);
    c.patchesType = type;
}
void
PatchTable::setBicubicFVarPatchChannelValues(int channel, int patchSize,
    std::vector<Index> const & values) {

    // This method populates the sparse array of values held in the patch
    // table from a non-sparse array of value indices generated during
    // the second traversal of an adaptive TopologyRefiner.
    // It is assumed that the patch types have been stored in the channel's
    // 'patchTypes' vector during the first traversal.

    FVarPatchChannel & c = getFVarPatchChannel(channel);
    assert(c.interpolation!=Sdc::Options::FVAR_LINEAR_ALL and
           c.patchTypes.size()*patchSize==values.size());

    int npatches = (int)c.patchTypes.size(),
        nverts = 0;

    // Generate offsets and count vertices
    c.patchValuesOffsets.resize(npatches);
    for (int patch=0; patch<npatches; ++patch) {
        int nv = PatchDescriptor::GetNumFVarControlVertices(c.patchTypes[patch]);
        c.patchValuesOffsets[patch] = nverts;
        nverts += nv;
    }

    // Populate values
    Index const * srcValues = &values[0];

    c.patchValues.resize(nverts);
    Index * dstValues = &c.patchValues[0];

    for (int patch=0; patch<npatches; ++patch) {

        int nv = PatchDescriptor::GetNumFVarControlVertices(c.patchTypes[patch]);

        memcpy(dstValues, srcValues, nv * sizeof(Index));

        srcValues += patchSize;
        dstValues += nv;
    }
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
        if (desc.GetType()>=PatchDescriptor::REGULAR and
            desc.GetType()<=PatchDescriptor::GREGORY_BASIS) {
            return true;
        }
    }
    return false;
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
Vtr::Array<PatchDescriptor::Type>
PatchTable::getFVarPatchTypes(int channel) {
    FVarPatchChannel & c = getFVarPatchChannel(channel);
    return Vtr::Array<PatchDescriptor::Type>(&c.patchTypes[0],
        (int)c.patchTypes.size());
}
Vtr::ConstArray<PatchDescriptor::Type>
PatchTable::GetFVarPatchTypes(int channel) const {
    FVarPatchChannel const & c = getFVarPatchChannel(channel);
    if (c.patchesType!=PatchDescriptor::NON_PATCH) {
        return Vtr::ConstArray<PatchDescriptor::Type>(&c.patchesType, 1);
    } else {
        return Vtr::ConstArray<PatchDescriptor::Type>(&c.patchTypes[0],
            (int)c.patchTypes.size());
    }
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
PatchDescriptor::Type
PatchTable::getFVarPatchType(int channel, int patch) const {
    FVarPatchChannel const & c = getFVarPatchChannel(channel);
    PatchDescriptor::Type type;
    if (c.patchesType!=PatchDescriptor::NON_PATCH) {
        assert(c.patchTypes.empty());
        type = c.patchesType;
    } else {
        assert(patch<(int)c.patchTypes.size());
        type = c.patchTypes[patch];
    }
    return type;
}
PatchDescriptor::Type
PatchTable::GetFVarPatchType(int channel, PatchHandle const & handle) const {
    return getFVarPatchType(channel, handle.patchIndex);
}
PatchDescriptor::Type
PatchTable::GetFVarPatchType(int channel, int arrayIndex, int patchIndex) const {
    return getFVarPatchType(channel, getPatchIndex(arrayIndex, patchIndex));
}
ConstIndexArray
PatchTable::getPatchFVarValues(int channel, int patch) const {

    FVarPatchChannel const & c = getFVarPatchChannel(channel);

    if (c.patchValuesOffsets.empty()) {
        int ncvs = PatchDescriptor::GetNumFVarControlVertices(c.patchesType);
        return ConstIndexArray(&c.patchValues[patch * ncvs], ncvs);
    } else {
        assert(patch<(int)c.patchValuesOffsets.size() and
            patch<(int)c.patchTypes.size());
        return ConstIndexArray(&c.patchValues[c.patchValuesOffsets[patch]],
            PatchDescriptor::GetNumFVarControlVertices(c.patchTypes[patch]));
   }
}
ConstIndexArray
PatchTable::GetPatchFVarValues(int channel, PatchHandle const & handle) const {
    return getPatchFVarValues(channel, handle.patchIndex);
}
ConstIndexArray
PatchTable::GetPatchFVarValues(int channel, int arrayIndex, int patchIndex) const {
    return getPatchFVarValues(channel, getPatchIndex(arrayIndex, patchIndex));
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
//  Evaluate basis functions for position and first derivatives at (s,t):
//
void
PatchTable::EvaluateBasis(PatchHandle const & handle, float s, float t,
    float wP[], float wDs[], float wDt[]) const {

    PatchDescriptor::Type patchType = GetPatchArrayDescriptor(handle.arrayIndex).GetType();
    PatchParam::BitField const & patchBits = _paramTable[handle.patchIndex].bitField;

    if (patchType == PatchDescriptor::REGULAR) {
        internal::GetBSplineWeights(patchBits, s, t, wP, wDs, wDt);
    } else if (patchType == PatchDescriptor::GREGORY_BASIS) {
        internal::GetGregoryWeights(patchBits, s, t, wP, wDs, wDt);
    } else if (patchType == PatchDescriptor::QUADS) {
        internal::GetBilinearWeights(patchBits, s, t, wP, wDs, wDt);
    } else {
        assert(0);
    }
}


} // end namespace Far

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv
