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

#include <cstring>
#include <cstdio>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Far {

PatchTables::PatchTables(int maxvalence) :
    _maxValence(maxvalence) {
}

// Copy constructor
// XXXX manuelk we need to eliminate this constructor (C++11 smart pointers)
PatchTables::PatchTables(PatchTables const & src) :
    _maxValence(src._maxValence),
    _numPtexFaces(src._numPtexFaces),
    _patchArrays(src._patchArrays),
    _patchVerts(src._patchVerts),
    _paramTable(src._paramTable),
    _quadOffsetsTable(src._quadOffsetsTable),
    _vertexValenceTable(src._vertexValenceTable),
    _fvarChannels(src._fvarChannels),
    _sharpnessIndices(src._sharpnessIndices),
    _sharpnessValues(src._sharpnessValues) {
}

PatchTables::~PatchTables() {
}

//
// PatchArrays
//
struct PatchTables::PatchArray {

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
PatchTables::PatchArray::print() const {
    desc.print();
    printf("    numPatches=%d vertIndex=%d patchIndex=%d "
        "quadOffsetIndex=%d\n", numPatches, vertIndex, patchIndex,
            quadOffsetIndex);
}
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

//
// FVarPatchChannel
//
// Stores a record for each patch in the primitive :
//
//  - Each patch in the PatchTables has a corresponding patch in each
//    face-varying patch channel. Patch vertex indices are sorted in the same
//    patch-type order as PatchTables::PTables. Face-varying data for a patch
//    can therefore be quickly accessed by using the patch primitive ID as
//    index into patchValueOffsets to locate the face-varying control vertex
//    indices.
//
//  - Face-varying channels can have a different interpolation modes
//
//  - Unlike "vertex" PatchTables, there are no "transition" patterns required
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
struct PatchTables::FVarPatchChannel {

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

inline PatchTables::FVarPatchChannel &
PatchTables::getFVarPatchChannel(int channel) {
    assert(channel<(int)_fvarChannels.size());
    return _fvarChannels[channel];
}
inline PatchTables::FVarPatchChannel const &
PatchTables::getFVarPatchChannel(int channel) const {
    assert(channel<(int)_fvarChannels.size());
    return _fvarChannels[channel];
}
void
PatchTables::allocateFVarPatchChannels(int numChannels) {
    _fvarChannels.resize(numChannels);
}
void
PatchTables::allocateChannelValues(int channel,
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
PatchTables::setFVarPatchChannelLinearInterpolation(int channel,
        Sdc::Options::FVarLinearInterpolation interpolation) {
    FVarPatchChannel & c = getFVarPatchChannel(channel);
    c.interpolation = interpolation;
}
void
PatchTables::setFVarPatchChannelPatchesType(int channel, PatchDescriptor::Type type) {
    FVarPatchChannel & c = getFVarPatchChannel(channel);
    c.patchesType = type;
}
void
PatchTables::setBicubicFVarPatchChannelValues(int channel, int patchSize,
    std::vector<Index> const & values) {

    // This method populates the sparse array of values held in the patch
    // tables from a non-sparse array of value indices generated during
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
// PatchTables
//

inline int
getPatchSize(PatchDescriptor desc) {
    return desc.GetNumControlVertices();
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

int
PatchTables::getPatchIndex(int arrayIndex, int patchIndex) const {
    PatchArray const & pa = getPatchArray(arrayIndex);
    assert(patchIndex<pa.numPatches);
    return pa.patchIndex + patchIndex;
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
PatchTables::GetNumPatchesTotal() const {
    // there is one PatchParam record for each patch in the mesh
    return (int)_paramTable.size();
}
int
PatchTables::GetNumControlVertices(int arrayIndex) const {
    PatchArray const & pa = getPatchArray(arrayIndex);
    return pa.numPatches * getPatchSize(pa.desc);
}

Index
PatchTables::findPatchArray(PatchDescriptor desc) {
    for (int i=0; i<(int)_patchArrays.size(); ++i) {
        if (_patchArrays[i].desc==desc)
            return i;
    }
    return Vtr::INDEX_INVALID;
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
    Index vert = pa.vertIndex + handle.vertIndex;
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
bool
PatchTables::IsFeatureAdaptive() const {

    // check for presence of tables only used by adaptive patches
    if (not _vertexValenceTable.empty())
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
PatchTables::GetNumFVarChannels() const {
    return (int)_fvarChannels.size();
}
Sdc::Options::FVarLinearInterpolation
PatchTables::GetFVarChannelLinearInterpolation(int channel) const {
    FVarPatchChannel const & c = getFVarPatchChannel(channel);
    return c.interpolation;
}
Vtr::Array<PatchDescriptor::Type>
PatchTables::getFVarPatchTypes(int channel) {
    FVarPatchChannel & c = getFVarPatchChannel(channel);
    return Vtr::Array<PatchDescriptor::Type>(&c.patchTypes[0],
        (int)c.patchTypes.size());
}
Vtr::ConstArray<PatchDescriptor::Type>
PatchTables::GetFVarPatchTypes(int channel) const {
    FVarPatchChannel const & c = getFVarPatchChannel(channel);
    if (c.patchesType!=PatchDescriptor::NON_PATCH) {
        return Vtr::ConstArray<PatchDescriptor::Type>(&c.patchesType, 1);
    } else {
        return Vtr::ConstArray<PatchDescriptor::Type>(&c.patchTypes[0],
            (int)c.patchTypes.size());
    }
}
ConstIndexArray
PatchTables::GetFVarPatchesValues(int channel) const {
    FVarPatchChannel const & c = getFVarPatchChannel(channel);
    return ConstIndexArray(&c.patchValues[0], (int)c.patchValues.size());
}
IndexArray
PatchTables::getFVarPatchesValues(int channel) {
    FVarPatchChannel & c = getFVarPatchChannel(channel);
    return IndexArray(&c.patchValues[0], (int)c.patchValues.size());
}
PatchDescriptor::Type
PatchTables::getFVarPatchType(int channel, int patch) const {
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
PatchTables::GetFVarPatchType(int channel, PatchHandle const & handle) const {
    return getFVarPatchType(channel, handle.patchIndex);
}
PatchDescriptor::Type
PatchTables::GetFVarPatchType(int channel, int arrayIndex, int patchIndex) const {
    return getFVarPatchType(channel, getPatchIndex(arrayIndex, patchIndex));
}
ConstIndexArray
PatchTables::getFVarPatchValues(int channel, int patch) const {

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
PatchTables::GetFVarPatchValues(int channel, PatchHandle const & handle) const {
    return getFVarPatchValues(channel, handle.patchIndex);
}
ConstIndexArray
PatchTables::GetFVarPatchValues(int channel, int arrayIndex, int patchIndex) const {
    return getFVarPatchValues(channel, getPatchIndex(arrayIndex, patchIndex));
}

void
PatchTables::print() const {
    printf("patchTables (0x%p)\n", this);
    printf("  numPatches = %d\n", GetNumPatchesTotal());
    for (int i=0; i<GetNumPatchArrays(); ++i) {
        printf("  patchArray %d:\n", i);
        PatchArray const & pa = getPatchArray(i);
        pa.print();
    }
}

} // end namespace Far

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv
