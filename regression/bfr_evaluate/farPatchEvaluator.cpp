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

#include "farPatchEvaluator.h"

#include <opensubdiv/far/topologyRefinerFactory.h>
#include <opensubdiv/far/topologyDescriptor.h>
#include <opensubdiv/far/primvarRefiner.h>
#include <opensubdiv/far/stencilTable.h>


template <typename REAL>
FarPatchEvaluator<REAL>::FarPatchEvaluator(
        Far::TopologyRefiner const & baseMesh,
        Vec3RealVector       const & basePos,
        Vec3RealVector       const & baseUVs,
        BfrSurfaceOptions    const & bfrSurfaceOptions) :
            _baseMesh(baseMesh),
            _baseMeshPos(basePos),
            _baseMeshUVs(baseUVs) {

    //
    //  Initialize simple members first:
    //
    _regFaceSize = Sdc::SchemeTypeTraits::GetRegularFaceSize(
                        baseMesh.GetSchemeType());

    //
    //  Declare options to use in construction of PatchTable et al:
    //
    int primaryLevel   = bfrSurfaceOptions.GetApproxLevelSharp();
    int secondaryLevel = bfrSurfaceOptions.GetApproxLevelSmooth();

    Far::PatchTableFactory::Options patchOptions(primaryLevel);
    patchOptions.SetPatchPrecision<REAL>();
    patchOptions.SetFVarPatchPrecision<REAL>();
    patchOptions.useInfSharpPatch = true;
    patchOptions.generateLegacySharpCornerPatches = false;
    patchOptions.shareEndCapPatchPoints = false;
    patchOptions.endCapType =
        Far::PatchTableFactory::Options::ENDCAP_GREGORY_BASIS;

    bool hasUVs = !baseUVs.empty();
    int  fvarChannel = 0;

    patchOptions.generateFVarTables = hasUVs;
    patchOptions.numFVarChannels = hasUVs ? 1 : 0;
    patchOptions.fvarChannelIndices = &fvarChannel;
    patchOptions.generateFVarLegacyLinearPatches = false;

    patchOptions.generateVaryingTables = false;

    Far::TopologyRefiner::AdaptiveOptions refineOptions =
            patchOptions.GetRefineAdaptiveOptions();
    refineOptions.SetIsolationLevel(primaryLevel);
    refineOptions.SetSecondaryLevel(secondaryLevel);

    //
    //  Create a TopologyRefiner (sharing the base) to adaptively refine
    //  and create the associated PatchTable:
    //
    Far::TopologyRefiner *patchRefiner =
        Far::TopologyRefinerFactory<Far::TopologyDescriptor>::Create(
            baseMesh);

    patchRefiner->RefineAdaptive(refineOptions);

    _patchTable = Far::PatchTableFactory::Create(*patchRefiner, patchOptions);

    _patchFaces = new Far::PtexIndices(baseMesh);

    _patchMap = new Far::PatchMap(*_patchTable);


    //
    //  Declare buffers/vectors for refined/patch points:
    //
    Far::TopologyLevel const & baseLevel = baseMesh.GetLevel(0);

    int numBasePoints    = baseLevel.GetNumVertices();
    int numRefinedPoints = patchRefiner->GetNumVerticesTotal() - numBasePoints;
    int numLocalPoints   = _patchTable->GetNumLocalPoints();

    _patchPos.resize(numBasePoints + numRefinedPoints + numLocalPoints);

    std::memcpy(&_patchPos[0], &basePos[0], numBasePoints * sizeof(Vec3Real));

    //
    //  Similarly declare buffers/vectors for refined/patch UVs:
    //
    int numBaseUVs    = 0;
    int numRefinedUVs = 0;
    int numLocalUVs   = 0;

    if (hasUVs) {
        numBaseUVs    = baseLevel.GetNumFVarValues();
        numRefinedUVs = patchRefiner->GetNumFVarValuesTotal() - numBaseUVs;
        numLocalUVs   = _patchTable->GetNumLocalPointsFaceVarying();

        _patchUVs.resize(numBaseUVs + numRefinedUVs + numLocalUVs);

        std::memcpy(&_patchUVs[0], &baseUVs[0], numBaseUVs * sizeof(Vec3Real));
    }

    //
    //  Compute refined and local patch points and UVs:
    //
    if (numRefinedPoints) {
        Far::PrimvarRefinerReal<REAL> primvarRefiner(*patchRefiner);

        Vec3Real const * srcP = &_patchPos[0];
        Vec3Real       * dstP = &_patchPos[numBasePoints];

        Vec3Real const * srcUV = hasUVs ? &_patchUVs[0] : 0;
        Vec3Real       * dstUV = hasUVs ? &_patchUVs[numBaseUVs] : 0;

        for (int level = 1; level < patchRefiner->GetNumLevels(); ++level) {
            primvarRefiner.Interpolate(level, srcP, dstP);
            srcP  = dstP;
            dstP += patchRefiner->GetLevel(level).GetNumVertices();

            if (hasUVs) {
                primvarRefiner.InterpolateFaceVarying(level, srcUV, dstUV);
                srcUV  = dstUV;
                dstUV += patchRefiner->GetLevel(level).GetNumFVarValues();
            }
        }
    }
    if (numLocalPoints) {
        _patchTable->GetLocalPointStencilTable<REAL>()->UpdateValues(
            &_patchPos[0], &_patchPos[numBasePoints + numRefinedPoints]);
    }
    if (hasUVs && numLocalUVs) {
        _patchTable->GetLocalPointFaceVaryingStencilTable<REAL>()->UpdateValues(
            &_patchUVs[0], &_patchUVs[numBaseUVs + numRefinedUVs]);
    }

    delete patchRefiner;
}

template <typename REAL>
bool
FarPatchEvaluator<REAL>::FaceHasLimit(Far::Index baseFace) const {

    return ! _baseMesh.GetLevel(0).IsFaceHole(baseFace);
}

template <typename REAL>
void
FarPatchEvaluator<REAL>::Evaluate(Far::Index                baseFace,
                                  TessCoordVector   const & tessCoords,
                                  EvalResults<REAL>       & results) const {

    assert(FaceHasLimit(baseFace));

    int numCoords = (int) tessCoords.size() / 2;

    //  Allocate vectors for the properties to be evaluated:
    results.Resize(numCoords);

    //
    //  Identify the patch face and see if it needs to be re-parameterized:
    //
    int patchFace = _patchFaces->GetFaceId(baseFace);

    int faceSize = _baseMesh.GetLevel(0).GetFaceVertices(baseFace).size();

    Bfr::Parameterization faceParam(_baseMesh.GetSchemeType(), faceSize);

    bool reparameterize = faceParam.HasSubFaces();

    //
    //  Evaluate at each of the given coordinates:
    //
    REAL const * stPair = &tessCoords[0];
    for (int i = 0; i < numCoords; ++i, stPair += 2) {
        REAL st[2] = { stPair[0], stPair[1] };

        int patchIndex = patchFace;
        if (reparameterize) {
            patchIndex += faceParam.ConvertCoordToNormalizedSubFace(st, st);
        }

        REAL s = st[0];
        REAL t = st[1];

        Far::PatchTable::PatchHandle const * patchHandle =
                _patchMap->FindPatch(patchIndex, s, t);
        assert(patchHandle);

        //  Evaluate position and derivatives:
        if (results.evalPosition) {
            REAL wP[20], wDu[20], wDv[20], wDuu[20], wDuv[20], wDvv[20];

            if (!results.eval1stDeriv) {
                _patchTable->EvaluateBasis(*patchHandle, s, t, wP);
            } else if (!results.eval2ndDeriv) {
                _patchTable->EvaluateBasis(*patchHandle, s, t, wP,
                                           wDu, wDv);
            } else {
                _patchTable->EvaluateBasis(*patchHandle, s, t, wP,
                                           wDu, wDv, wDuu, wDuv, wDvv);
            }

            Vec3Real * P   = results.evalPosition ? &results.p[i]   : 0;
            Vec3Real * Du  = results.eval1stDeriv ? &results.du[i]  : 0;
            Vec3Real * Dv  = results.eval1stDeriv ? &results.dv[i]  : 0;
            Vec3Real * Duu = results.eval2ndDeriv ? &results.duu[i] : 0;
            Vec3Real * Duv = results.eval2ndDeriv ? &results.duv[i] : 0;
            Vec3Real * Dvv = results.eval2ndDeriv ? &results.dvv[i] : 0;

            Far::ConstIndexArray cvIndices =
                    _patchTable->GetPatchVertices(*patchHandle);

            P->Clear();
            if (results.eval1stDeriv) {
                Du->Clear();
                Dv->Clear();
                if (results.eval2ndDeriv) {
                    Duu->Clear();
                    Duv->Clear();
                    Dvv->Clear();
                }
            }

            for (int cv = 0; cv < cvIndices.size(); ++cv) {
                P->AddWithWeight(_patchPos[cvIndices[cv]], wP[cv]);
                if (results.eval1stDeriv) {
                    Du->AddWithWeight(_patchPos[cvIndices[cv]], wDu[cv]);
                    Dv->AddWithWeight(_patchPos[cvIndices[cv]], wDv[cv]);
                    if (results.eval2ndDeriv) {
                        Duu->AddWithWeight(_patchPos[cvIndices[cv]], wDuu[cv]);
                        Duv->AddWithWeight(_patchPos[cvIndices[cv]], wDuv[cv]);
                        Dvv->AddWithWeight(_patchPos[cvIndices[cv]], wDvv[cv]);
                    }
                }
            }
        }
        if (results.evalUV) {
            REAL wUV[20];
            _patchTable->EvaluateBasisFaceVarying(*patchHandle, s, t, wUV);

            Vec3Real & UV = results.uv[i];

            UV.Clear();

            Far::ConstIndexArray cvIndices =
                    _patchTable->GetPatchFVarValues(*patchHandle);

            for (int cv = 0; cv < cvIndices.size(); ++cv) {
                UV.AddWithWeight(_patchUVs[cvIndices[cv]], wUV[cv]);
            }
        }
    }
}

template <typename REAL>
FarPatchEvaluator<REAL>::~FarPatchEvaluator() {

    delete _patchTable;
    delete _patchFaces;
    delete _patchMap;
}

//
//  Explicit instantiation for float and double:
//
template class FarPatchEvaluator<float>;
template class FarPatchEvaluator<double>;

