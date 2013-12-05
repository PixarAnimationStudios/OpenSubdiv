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

#ifndef PXOSDUTIL_UNIFORM_EVALUATOR_H
#define PXOSDUTIL_UNIFORM_EVALUATOR_H

#include "refiner.h"

#include <string>
#include <vector>

#define HBR_ADAPTIVE 

#include "../osd/cpuVertexBuffer.h"
#include "../osd/cpuComputeContext.h"
#include "../far/mesh.h"

// This class takes a mesh that has undergone uniform refinement to
// a fixed subdivision level, and creates required run time OpenSubdiv
// data structures used to sample values on subdivision surfaces.
//
// An important note here is that refined positions and vertex varying
// attributes are sampled at the n-th subdivision level, not at the
// exact limit surface.  Use PxOsdUtilAdaptiveEvaluator for true limits.
//
class PxOsdUtilUniformEvaluator {
  public:
    PxOsdUtilUniformEvaluator();

    ~PxOsdUtilUniformEvaluator();    

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
        PxOsdUtilRefiner* refiner,
        std::string *errorMessage = NULL);
    
    bool Initialize(
        const PxOsdUtilSubdivTopology &topology,
        std::string *errorMessage = NULL);    

    // Set new coarse-mesh CV positions, need to call Refine
    // before calling Get* methods
    void SetCoarsePositions(
        const std::vector<float>& coords,
        std::string *errorMessage = NULL
        );

    // Set new coarse-mesh vertex varying values, need to call Refine
    // before calling Get* methods    
    void SetCoarseVVData(
        const std::vector<float>& data,
        std::string *errorMessage = NULL
        );

    // Refine the coarse mesh, needed before calling GetPositions, GetQuads etc.
    // If numThreads is 1, use single cpu.  If numThreads > 1 use Omp and set
    // number of omp threads.
    //
    bool Refine(int numThreads,
                std::string *errorMessage = NULL);
    
    // Grab position results of calling Refine, return by reference a pointer
    // into _vertexBuffer memory with the refined points positions,
    // packed as 3 floats per point.  This doesn't involve a copy, the
    // pointer will be valid as long as _vertexBuffer exists.
    //
    bool GetRefinedPositions(const float **positions, int *numFloats,
                             std::string *errorMessage = NULL) const;

    // Grab vertex varying results of calling Refine, return by reference
    // a pointer into _vvBuffer memory with the refined data, 
    // packed as (numElements) floats per point.  This doesn't involve
    // a copy, the pointer will be valid as long as _vertexBuffer exists.
    //
    bool GetRefinedVVData(float **data, int *numFloats,
                          int *numElements = NULL,
                          std::string *errorMessage = NULL) const;    

    // Fetch the face varying attribute values on refined quads, call
    // through to the refiner but keep in evaluator API for 
    // one-stop-service for user API
    void GetRefinedFVData( int level,
                   const std::vector<std::string>& names,
                   std::vector<float>* fvdata) {
        _refiner->GetRefinedFVData(level, names, fvdata);
    }


    // Fetch the topology of the post-refined mesh. The "quads" vector
    // will be filled with 4 ints per quad which index into a vector
    // of positions.
    //
    // Calls through to the refiner.
    //
    bool GetRefinedQuads(std::vector<int>* quads,
                         std::string *errorMessage = NULL) const {
        return _refiner->GetRefinedQuads(quads, errorMessage);
    }
        
    // Fetch the U/V coordinates of the refined quads in the U/V space
    // of their parent coarse face
    bool GetRefinedPtexUvs(std::vector<float>* subfaceUvs,
                    std::vector<int>* ptexIndices,
                    std::string *errorMessage = NULL) const {
        return _refiner->GetRefinedPtexUvs(subfaceUvs, ptexIndices, errorMessage);
    }

    bool GetRefinedTopology(
        PxOsdUtilSubdivTopology *t,
        //positions will have three floats * t->numVertices
        const float **positions,    
        std::string *errorMessage = NULL);
    
    // Forward these calls through to the refiner, which may forward
    // to the mesh.  Make these top level API calls on the evaluator
    // so clients can talk to a single API
    //
    const std::string &GetName() const { return _refiner->GetName();}

    const OpenSubdiv::HbrMesh<OpenSubdiv::OsdVertex> *GetHbrMesh() {
        return _refiner->GetHbrMesh();
    }

    const PxOsdUtilSubdivTopology &GetTopology() const {
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
    PxOsdUtilRefiner *_refiner;
    bool _ownsRefiner;

    // responsible for performing uniform catmull/clark subdivision
    // on the incoming polygonal topology.  This only stores topology and
    // face varying data and can be shared among threads.  Doesn't store
    // per-vertex information being refined in different threads, those
    // are in vertex buffers that can't be shared.
    OpenSubdiv::OsdCpuComputeContext *_computeContext;
    OpenSubdiv::OsdCpuVertexBuffer *_vertexBuffer;
    OpenSubdiv::OsdCpuVertexBuffer *_vvBuffer;
};



#endif /* PXOSDUTIL_UNIFORM_EVALUATOR_H */
