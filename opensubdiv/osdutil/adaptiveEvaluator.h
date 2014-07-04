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

#ifndef OSDUTIL_ADAPTIVE_EVALUATOR_H
#define OSDUTIL_ADAPTIVE_EVALUATOR_H

#include "../version.h"

#include "mesh.h"
#include "refiner.h"

#include <string>
#include <vector>

#define HBR_ADAPTIVE 

#include "../osd/cpuVertexBuffer.h"
#include "../osd/cpuComputeContext.h"
#include "../osd/cpuEvalLimitController.h"
#include "../osd/cpuEvalLimitContext.h"
#include "../far/mesh.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

// This class takes a mesh that has undergone adaptive refinement to
// create bspline and gregory patches to a fixed subdivision level,
// and creates required run time OpenSubdiv data structures used to
// call the eval API to sample values on subdivision surfaces on the
// limit surface..
//
class OsdUtilAdaptiveEvaluator {
  public:
    OsdUtilAdaptiveEvaluator();

    ~OsdUtilAdaptiveEvaluator();    

    // Initialize returns false on error.  If errorMessage is non-NULL it'll
    // be populated upon error.
    //
    // If successful vertex buffers and compute contexts will have been
    // created and are ready to SetCoarse* methods, call Refine, and then
    // sample refined values with the Get* methods
    // 
    // Note that calling this method assumes that the evaluator isn't
    // responsible for the refiner's lifetime, someone else needs to
    // hold onto the refiner pointer.  This allows for lightweight sharing
    // of refiners among evaluators.
    //
    bool Initialize(
        OsdUtilRefiner* refiner,
        std::string *errorMessage = NULL);
    
    bool Initialize(
        const OsdUtilSubdivTopology &topology,
        std::string *errorMessage = NULL,
        OsdUtilMesh<OsdVertex>::Scheme scheme = OsdUtilMesh<OsdVertex>::SCHEME_CATMARK);

    // Set new coarse-mesh CV positions, need to call Refine
    // before calling Get* methods.  Three floats per
    // point packed.
    void SetCoarsePositions(
        const float *coords, int numFloats,
        std::string *errorMessage = NULL
        );

    // Refine the coarse mesh, needed before calling GetPositions, GetQuads etc
    // If numThreads is 1, use single cpu.  If numThreads > 1 use Omp and set
    // number of omp threads.
    //
    bool Refine(int numThreads,
                std::string *errorMessage = NULL);
    

    void EvaluateLimit(
        const OpenSubdiv::OsdEvalCoords &coords,
	float P[3], float dPdu[3], float dPdv[3]);

    bool GetRefinedTopology(
        OsdUtilSubdivTopology *t,
        //positions will have three floats * t->numVertices
	std::vector<float> *positions,
        std::string *errorMessage = NULL);    
    
    // Forward these calls through to the refiner, which may forward
    // to the mesh.  Make these top level API calls on the evaluator
    // so clients can talk to a single API
    //
    const std::string &GetName() const { return _refiner->GetName();}

    const OpenSubdiv::HbrMesh<OpenSubdiv::OsdVertex> *GetHbrMesh() {
        return _refiner->GetHbrMesh();
    }

    const OsdUtilSubdivTopology &GetTopology() const {
        return _refiner->GetTopology();
    }

    const OpenSubdiv::FarMesh<OpenSubdiv::OsdVertex>*  GetFarMesh() {
        return _refiner->GetFarMesh();
    }
    
  private:    
    
    // A pointer to the shared refiner used.  Note that this class may
    // own the refiner pointer (if _ownsRefiner is true), or it may
    // assume that someone else is responsible for managing that pointer
    // if _ownsRefiner is false.
    OsdUtilRefiner *_refiner;
    bool _ownsRefiner;

    OpenSubdiv::OsdCpuComputeContext *_computeContext;
    OpenSubdiv::OsdCpuEvalLimitContext *_evalLimitContext;
    OpenSubdiv::OsdCpuVertexBuffer *_vertexBuffer;
    OpenSubdiv::OsdCpuVertexBuffer *_vvBuffer; // not yet used
};


}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv


#endif /* OSDUTIL_ADAPTIVE_EVALUATOR_H */
