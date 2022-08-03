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

#include "bfrSurfaceEvaluator.h"


template <typename REAL>
BfrSurfaceEvaluator<REAL>::BfrSurfaceEvaluator(
        Far::TopologyRefiner const & baseMesh,
        Vec3Vector           const & basePos,
        Vec3Vector           const & baseUVs,
        FactoryOptions       const & factoryOptions) :
            _baseMesh(baseMesh),
            _baseMeshPos(basePos),
            _baseMeshUVs(baseUVs),
            _factory(baseMesh, factoryOptions) {

}

template <typename REAL>
bool
BfrSurfaceEvaluator<REAL>::FaceHasLimit(IndexType baseFace) const {

    return _factory.FaceHasLimitSurface(baseFace);
}

template <typename REAL>
void
BfrSurfaceEvaluator<REAL>::Evaluate(IndexType                 baseFace,
                                    TessCoordVector   const & tessCoords,
                                    EvalResults<REAL>       & results) const {

    //  Allocate vectors for the properties to be evaluated:
    int numCoords = (int) tessCoords.size() / 2;

    results.Resize(numCoords);

    //  Create the Surfaces for position and UV (optional) and assert if
    //  not valid, since a limit surface is expected here. (Note we may
    //  create the position surface but not actually evaluate it.)
    SurfaceType pSurface;
    SurfaceType uvSurface;

    //  Figure out how to get a command line arg here to run both
    bool initSeparate = false;
    if (initSeparate || !results.evalUV) {
        _factory.InitVertexSurface(baseFace, &pSurface);
        if (results.evalUV) {
            _factory.InitFaceVaryingSurface(baseFace, &uvSurface);
        }
    } else {
        _factory.InitSurfaces(baseFace, &pSurface, &uvSurface);
    }

    assert(pSurface.IsValid());
    assert(uvSurface.IsValid() == results.evalUV);

    //  Evaluate directly or using stencils:
    if (results.useStencils) {
        evaluateByStencils(pSurface, uvSurface, tessCoords, results);
    } else {
        evaluateDirectly(pSurface, uvSurface, tessCoords, results);
    }
}

template <typename REAL>
void
BfrSurfaceEvaluator<REAL>::evaluateDirectly(
        SurfaceType const & pSurface, SurfaceType const & uvSurface,
        TessCoordVector const & tessCoords, EvalResults<REAL> & results) const {

    int numCoords = (int) tessCoords.size() / 2;

    if (results.evalPosition) {
        Vec3Vector baseFacePos(pSurface.GetNumPatchPoints());

        REAL const * meshPoints  = &_baseMeshPos[0][0];
        REAL       * patchPoints = &baseFacePos[0][0];

        pSurface.PreparePatchPoints(meshPoints, 3, patchPoints, 3);

        REAL const * st = &tessCoords[0];
        for (int i = 0; i < numCoords; ++i, st += 2) {
            if (!results.eval1stDeriv) {
                pSurface.Evaluate(st, patchPoints, 3, &results.p[i][0]);
            } else if (!results.eval2ndDeriv) {
                pSurface.Evaluate(st, patchPoints, 3,
                    &results.p[i][0], &results.du[i][0], &results.dv[i][0]);
            } else {
                pSurface.Evaluate(st, patchPoints, 3,
                    &results.p[i][0], &results.du[i][0], &results.dv[i][0],
                    &results.duu[i][0], &results.duv[i][0], &results.dvv[i][0]);
            }
        }
    }
    if (results.evalUV) {
        Vec3Vector baseFaceUVs(uvSurface.GetNumPatchPoints());

        REAL const * meshPoints  = &_baseMeshUVs[0][0];
        REAL       * patchPoints = &baseFaceUVs[0][0];

        uvSurface.PreparePatchPoints(meshPoints, 3, patchPoints, 3);

        REAL const * st = &tessCoords[0];
        for (int i = 0; i < numCoords; ++i, st += 2) {
            uvSurface.Evaluate(st, patchPoints, 3, &results.uv[i][0]);
        }
    }
}

template <typename REAL>
void
BfrSurfaceEvaluator<REAL>::evaluateByStencils(
        SurfaceType const & pSurface, SurfaceType const & uvSurface,
        TessCoordVector const & tessCoords, EvalResults<REAL> & results) const {

    std::vector<REAL> stencilWeights;

    int numCoords = (int) tessCoords.size() / 2;

    if (results.evalPosition) {
        stencilWeights.resize(6 * pSurface.GetNumControlPoints());

        REAL * sP   = &stencilWeights[0];
        REAL * sDu  = sP   + pSurface.GetNumControlPoints();
        REAL * sDv  = sDu  + pSurface.GetNumControlPoints();
        REAL * sDuu = sDv  + pSurface.GetNumControlPoints();
        REAL * sDuv = sDuu + pSurface.GetNumControlPoints();
        REAL * sDvv = sDuv + pSurface.GetNumControlPoints();

        REAL const * st = &tessCoords[0];
        for (int i = 0; i < numCoords; ++i, st += 2) {
            REAL const * meshPos = &_baseMeshPos[0][0];

            if (!results.eval1stDeriv) {
                pSurface.EvaluateStencil(st, &sP[0]);
            } else if (!results.eval2ndDeriv) {
                pSurface.EvaluateStencil(st, &sP[0], &sDu[0], &sDv[0]);
            } else {
                pSurface.EvaluateStencil(st, &sP[0], &sDu[0], &sDv[0],
                                             &sDuu[0], &sDuv[0], &sDvv[0]);
            }

            if (results.evalPosition) {
                pSurface.ApplyStencilFromMesh(&sP[0],  meshPos, 3,
                                              &results.p[i][0]);
            }
            if (results.eval1stDeriv) {
                pSurface.ApplyStencilFromMesh(&sDu[0], meshPos, 3,
                                              &results.du[i][0]);
                pSurface.ApplyStencilFromMesh(&sDv[0], meshPos, 3,
                                              &results.dv[i][0]);
            }
            if (results.eval2ndDeriv) {
                pSurface.ApplyStencilFromMesh(&sDuu[0], meshPos, 3,
                                              &results.duu[i][0]);
                pSurface.ApplyStencilFromMesh(&sDuv[0], meshPos, 3,
                                              &results.duv[i][0]);
                pSurface.ApplyStencilFromMesh(&sDvv[0], meshPos, 3,
                                              &results.dvv[i][0]);
            }
        }
    }
    if (results.evalUV) {
        REAL const * meshUVs = &_baseMeshUVs[0][0];

        stencilWeights.resize(uvSurface.GetNumControlPoints());

        REAL * sUV = &stencilWeights[0];

        REAL const * st = &tessCoords[0];
        for (int i = 0; i < numCoords; ++i, st += 2) {
            uvSurface.EvaluateStencil(st, &sUV[0]);

            uvSurface.ApplyStencilFromMesh(&sUV[0], meshUVs, 3,
                                           &results.uv[i][0]);
        }
    }
}


//
//  Explicit instantiation for float and double:
//
template class BfrSurfaceEvaluator<float>;
template class BfrSurfaceEvaluator<double>;

