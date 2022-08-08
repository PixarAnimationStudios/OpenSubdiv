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

#include "./types.h"

#include <opensubdiv/far/topologyRefiner.h>

#include <opensubdiv/bfr/refinerSurfaceFactory.h>
#include <opensubdiv/bfr/tessellation.h>


using namespace OpenSubdiv;
using namespace OpenSubdiv::OPENSUBDIV_VERSION;



template <typename REAL>
class BfrSurfaceEvaluator {
public:
    typedef Bfr::Surface<REAL>           SurfaceType;

    typedef Bfr::RefinerSurfaceFactory<> SurfaceFactory;
    typedef Bfr::SurfaceFactory::Options FactoryOptions;
    typedef Bfr::SurfaceFactory::Index   IndexType;

    typedef std::vector<REAL>            TessCoordVector;

    typedef std::vector< Vec3<REAL> > Vec3Vector;

public:
    BfrSurfaceEvaluator(Far::TopologyRefiner const & baseMesh,
                        Vec3Vector           const & basePos,
                        Vec3Vector           const & baseUVs,
                        FactoryOptions       const & factoryOptions);
    ~BfrSurfaceEvaluator() { }

public:
    bool FaceHasLimit(IndexType baseFace) const;

    void Evaluate(IndexType               baseface,
                  TessCoordVector const & tessCoords,
                  EvalResults<REAL>     & results) const;

private:
    void evaluateDirectly(SurfaceType     const & posSurface,
                          SurfaceType     const & uvSurface,
                          TessCoordVector const & tessCoords,
                          EvalResults<REAL>     & results) const;

    void evaluateByStencils(SurfaceType     const & posSurface,
                            SurfaceType     const & uvSurface,
                            TessCoordVector const & tessCoords,
                            EvalResults<REAL>     & results) const;

private:
    Far::TopologyRefiner const & _baseMesh;
    Vec3Vector           const & _baseMeshPos;
    Vec3Vector           const & _baseMeshUVs;

    SurfaceFactory _factory;
};
