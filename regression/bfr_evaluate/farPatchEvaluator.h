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
#include <opensubdiv/far/patchTable.h>
#include <opensubdiv/far/patchTableFactory.h>
#include <opensubdiv/far/patchMap.h>
#include <opensubdiv/far/ptexIndices.h>

#include <opensubdiv/bfr/refinerSurfaceFactory.h>
#include <opensubdiv/bfr/tessellation.h>

using namespace OpenSubdiv;
using namespace OpenSubdiv::OPENSUBDIV_VERSION;


//
//  FarPatchEvaluator bundles the Far::PatchTable and its ecosystem of
//  related class to provide an evaluation interface targeted towards
//  evaluation of the base faces of a mesh.
//
template <typename REAL>
class FarPatchEvaluator {
public:
    typedef Far::PatchTableFactory       PatchFactory;
    typedef Bfr::SurfaceFactory::Options BfrSurfaceOptions;

    typedef std::vector<REAL>            TessCoordVector;

    typedef Vec3<REAL>                   Vec3Real;
    typedef std::vector<Vec3Real>        Vec3RealVector;

public:
    FarPatchEvaluator(Far::TopologyRefiner const & baseMesh,
                      Vec3RealVector       const & basePos,
                      Vec3RealVector       const & baseUVs,
                      BfrSurfaceOptions    const & bfrSurfaceOptions);
    ~FarPatchEvaluator();

public:
    bool FaceHasLimit(Far::Index baseFace) const;

    void Evaluate(Far::Index              baseface,
                  TessCoordVector const & tessCoords,
                  EvalResults<REAL>     & results) const;

private:
    Far::TopologyRefiner const & _baseMesh;
    Vec3RealVector       const & _baseMeshPos;
    Vec3RealVector       const & _baseMeshUVs;

    Far::PatchTable  * _patchTable;
    Far::PatchMap    * _patchMap;
    Far::PtexIndices * _patchFaces;

    Vec3RealVector     _patchPos;
    Vec3RealVector     _patchUVs;

    int  _regFaceSize;
};
