//
//   Copyright 2016 Nvidia
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
#include "../far/endCapBilinearBasisPatchFactory.h"
#include "../far/error.h"
#include "../far/stencilTableFactory.h"
#include "../far/topologyRefiner.h"

#include "../vtr/componentInterfaces.h"
#include "../vtr/refinement.h"

#include "../sdc/bilinearScheme.h"
#include "../sdc/catmarkScheme.h"
#include "../sdc/loopScheme.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Far {

//
//  Local class to fulfil interface for <typename MASK> in the Scheme mask queries:
//
class Mask {
public:
    typedef float Weight;  //  Also part of the expected interface

public:
    Mask(Weight* v, Weight* e, Weight* f) :
        _vertWeights(v), _edgeWeights(e), _faceWeights(f),
        _vertCount(0), _edgeCount(0), _faceCount(0),
        _faceWeightsForFaceCenters(false)
    { }

    ~Mask() { }

public:  //  Generic interface expected of <typename MASK>:
    int GetNumVertexWeights() const { return _vertCount; }
    int GetNumEdgeWeights()   const { return _edgeCount; }
    int GetNumFaceWeights()   const { return _faceCount; }

    void SetNumVertexWeights(int count) { _vertCount = count; }
    void SetNumEdgeWeights(  int count) { _edgeCount = count; }
    void SetNumFaceWeights(  int count) { _faceCount = count; }

    Weight const& VertexWeight(int index) const { return _vertWeights[index]; }
    Weight const& EdgeWeight(  int index) const { return _edgeWeights[index]; }
    Weight const& FaceWeight(  int index) const { return _faceWeights[index]; }

    Weight& VertexWeight(int index) { return _vertWeights[index]; }
    Weight& EdgeWeight(  int index) { return _edgeWeights[index]; }
    Weight& FaceWeight(  int index) { return _faceWeights[index]; }

    bool AreFaceWeightsForFaceCenters() const  { return _faceWeightsForFaceCenters; }
    void SetFaceWeightsForFaceCenters(bool on) { _faceWeightsForFaceCenters = on; }

private:
    Weight* _vertWeights;
    Weight* _edgeWeights;
    Weight* _faceWeights;

    int _vertCount;
    int _edgeCount;
    int _faceCount;

    bool _faceWeightsForFaceCenters;
};


EndCapBilinearBasisPatchFactory::EndCapBilinearBasisPatchFactory(
    TopologyRefiner const & refiner,
    StencilTable * vertexStencils,
    StencilTable * varyingStencils,
    bool shareBoundaryVertices) :
        _vertexStencils(vertexStencils),
        _varyingStencils(varyingStencils),
        _refiner(refiner),
        _shareBoundaryVertices(shareBoundaryVertices),
        _numBilinearBasisPatches(0),
        _numBilinearBasisVertices(0) {

    // Sanity check: the mesh must be adaptively refined
    assert(! refiner.IsUniform());

    // Reserve the patch point stencils. Ideally topology refiner
    // would have an API to return how many endcap patches will be required.
    // Instead we conservatively estimate by the number of patches at the
    // finest level.
    int numMaxLevelFaces = refiner.GetLevel(refiner.GetMaxLevel()).GetNumFaces();

    int numPatchPointsExpected = numMaxLevelFaces * 4;

    // limits to 100M (=800M bytes) entries for the reserved size.
    int numStencilsExpected = std::min(numPatchPointsExpected * 16,
                                       100*1024*1024);
    _vertexStencils->reserve(numPatchPointsExpected, numStencilsExpected);
    if (_varyingStencils) {
        // varying stencils use only 1 index with weight=1.0
        _varyingStencils->reserve(numPatchPointsExpected, numPatchPointsExpected);
    }
}

ConstIndexArray
EndCapBilinearBasisPatchFactory::GetPatchPoints(
    Vtr::internal::Level const * level, Index faceIndex,
    Vtr::internal::Level::VSpan const /* cornerSpans */[],
    int levelVertOffset) {

    Sdc::Scheme<Sdc::SCHEME_CATMARK> scheme(_refiner.GetSchemeOptions());

    // allocate indices (awkward)
    // assert(Vtr::INDEX_INVALID==0xFFFFFFFF);
    for (int i = 0; i < 4; ++i) {
        _patchPoints.push_back(Vtr::INDEX_INVALID);
    }
    Index * dest = &_patchPoints[_numBilinearBasisPatches * 4];

    int  maxValence = level->getMaxValence(),
         maxWeightsPerMask = 1 + 2 * maxValence,
         stencilCapacity = maxValence*4,
         numMasks = 1;

    Vtr::internal::StackBuffer<Index,33> indexBuffer(maxWeightsPerMask);
    Vtr::internal::StackBuffer<float,99> weightBuffer(numMasks * maxWeightsPerMask);

    float * vPosWeights = weightBuffer,
          * ePosWeights = vPosWeights + 1,
          * fPosWeights = ePosWeights + maxValence;

    Mask posMask(vPosWeights, ePosWeights, fPosWeights);

    Vtr::internal::VertexInterface vHood(*level, *level);

    ConstIndexArray faceVerts = level->getFaceVertices(faceIndex);

    GregoryBasis::Point dstPoint;

    for (int vert=0; vert<faceVerts.size(); ++vert) {

        Index srcVertIndex = faceVerts[vert];

        ConstIndexArray vEdges =
            level->getVertexEdges(srcVertIndex);

        //  Incomplete vertices (present in sparse refinement) do not have their full
        //  topological neighborhood to determine a proper limit -- just leave the
        //  vertex at the refined location and continue to the next:
        if (level->getVertexTag(srcVertIndex)._incomplete ||
            (vEdges.size() == 0)) {
            dstPoint.Clear(stencilCapacity);
            dstPoint.AddWithWeight(srcVertIndex, 1.0);
            continue;
        }

        //
        //  Limit masks require the subdivision Rule for the vertex in order to deal
        //  with infinitely sharp features correctly -- including boundaries and corners.
        //  The vertex neighborhood is minimally defined with vertex and edge counts.
        //
        Sdc::Crease::Rule vRule = level->getVertexRule(srcVertIndex);

        //  This is a bit obscure -- child vertex index will be ignored here
        vHood.SetIndex(srcVertIndex, srcVertIndex);

        scheme.ComputeVertexLimitMask(vHood, posMask, vRule);

        //
        //  Gather the neighboring vertices of this vertex -- the vertices opposite its
        //  incident edges, and the opposite vertices of its incident faces:
        //
        Index * eIndices = indexBuffer;
        Index * fIndices = indexBuffer + vEdges.size();

        for (int i = 0; i < vEdges.size(); ++i) {
            ConstIndexArray eVerts = level->getEdgeVertices(vEdges[i]);

            eIndices[i] = (eVerts[0] == srcVertIndex) ? eVerts[1] : eVerts[0];
        }
        if (posMask.GetNumFaceWeights()) {

            ConstIndexArray      vFaces =
                level->getVertexFaces(srcVertIndex);

            ConstLocalIndexArray vInFace =
                level->getVertexFaceLocalIndices(srcVertIndex);

            for (int i = 0; i < vFaces.size(); ++i) {
                ConstIndexArray fVerts = level->getFaceVertices(vFaces[i]);

                LocalIndex vOppInFace = (vInFace[i] + 2);
                if (vOppInFace >= fVerts.size()) vOppInFace -= (LocalIndex)fVerts.size();

                fIndices[i] = level->getFaceVertices(vFaces[i])[vOppInFace];
            }
        }

        //
        //  Combine the weights and indices for position and tangents.  As with applying
        //  refinment masks to vertex data, in order to improve numerical precision, its
        //  better to apply smaller weights first, so begin with the face-weights followed
        //  by the edge-weights and the vertex weight last.
        //
        dstPoint.Clear(stencilCapacity);
        for (int i = 0; i < posMask.GetNumFaceWeights(); ++i) {
            dstPoint.AddWithWeight(fIndices[i], fPosWeights[i]);
        }
        for (int i = 0; i < posMask.GetNumEdgeWeights(); ++i) {
            dstPoint.AddWithWeight(eIndices[i], ePosWeights[i]);
        }
        dstPoint.AddWithWeight(srcVertIndex, vPosWeights[0]);

        dstPoint.OffsetIndices(levelVertOffset);

        GregoryBasis::AppendToStencilTable(dstPoint, _vertexStencils);
    }

    // XXXX TODO implement vertex sharing

    int vertexOffset = _refiner.GetNumVerticesTotal();
    for (int i = 0; i < 4; ++i) {
        dest[i] = _numBilinearBasisVertices + vertexOffset;
        ++_numBilinearBasisVertices;
    }

    ++_numBilinearBasisPatches;

    // return cvs;
    return ConstIndexArray(dest, 4);
}

} // end namespace Far

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv
