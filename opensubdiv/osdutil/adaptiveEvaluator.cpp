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

#include "adaptiveEvaluator.h"

#define HBR_ADAPTIVE
#include "../hbr/mesh.h"

#include "../osd/vertex.h"

#ifdef OPENSUBDIV_HAS_OPENMP
#include <omp.h>
#include "../osd/ompComputeController.h"
#endif


#include "../osd/cpuComputeController.h"

#include <fstream>
#include <iostream>

using namespace std;

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

OsdUtilAdaptiveEvaluator::OsdUtilAdaptiveEvaluator():
    _refiner(NULL),
    _ownsRefiner(false),
    _computeContext(NULL),
    _evalLimitContext(NULL),
    _vertexBuffer(NULL),
    _vvBuffer(NULL)
{
}

OsdUtilAdaptiveEvaluator::~OsdUtilAdaptiveEvaluator()
{
    if (_ownsRefiner and _refiner) {
        delete _refiner;
    }
    if (_computeContext)
        delete _computeContext;
    if (_evalLimitContext)
        delete _evalLimitContext;
    if (_vertexBuffer)
        delete _vertexBuffer;
    if (_vvBuffer)
        delete _vvBuffer;
}


bool
OsdUtilAdaptiveEvaluator::Initialize(
    const OsdUtilSubdivTopology &t,
    string *errorMessage,
    OsdUtilMesh<OsdVertex>::Scheme scheme)
{

    // create and initialize a refiner, passing "true" for adaptive
    // to indicate we wish for adaptive refinement rather than uniform
    OsdUtilRefiner *refiner = new OsdUtilRefiner();

    if (not refiner->Initialize(t, true, errorMessage, scheme)) {
        delete refiner;
        return false;
    }

    bool result = Initialize(refiner, errorMessage);
    _ownsRefiner = true;
    return result;
}

bool
OsdUtilAdaptiveEvaluator::Initialize(
    OsdUtilRefiner *refiner,
    string *errorMessage)
{

    if (refiner->GetAdaptive()) {
        if (errorMessage)
            *errorMessage = "Adaptive evaluator requires adaptive refiner";
        return false;
    }

    // Note we assume someone else keeps this pointer alive
    _refiner = refiner;
    _ownsRefiner = false;

    const FarMesh<OsdVertex> *fmesh = _refiner->GetFarMesh();
    const HbrMesh<OsdVertex> *hmesh = _refiner->GetHbrMesh();

    if (not (fmesh and hmesh)) {
        if (errorMessage)
            *errorMessage = "No valid adaptive far/hbr mesh";
        return false;
    }

    // Three elements (x/y/z) per refined point at every subdivision level
    // defined by the farMesh.  The coarse vertices seed the beginning of
    // this buffer, and Refine populates the rest based on subdivision
    _vertexBuffer = OsdCpuVertexBuffer::Create(
        3, fmesh->GetNumVertices());

    // zeros
    memset( _vertexBuffer->BindCpuBuffer(), 0,
            3 * fmesh->GetNumVertices() * sizeof(float));

    /*
    const vector<string> &vvNames   = _refiner->GetTopology().vvNames;
    // If needed, allocate vertex buffer for other vertex varying
    // values, like UVs or gprim data.
    if (vvNames.size()) {

        // One element in the vertex buffer for each
        // named vertex varying attribute in the refined mesh
        _vvBuffer = OsdCpuVertexBuffer::Create(
            (int)vvNames.size(), fmesh->GetNumVertices());

        // zeros
        memset( _vvBuffer->BindCpuBuffer(), 0,
                vvNames.size() * fmesh->GetNumVertices() * sizeof(float));
    }
    */

    // A context object used to store data used in refinement
    _computeContext = OsdCpuComputeContext::Create(fmesh->GetSubdivisionTables(),
                                                   fmesh->GetVertexEditTables());

    // A context object used to store data used in fast limit surface
    // evaluation.  This contains vectors of patches and associated
    // tables pulled and computed from the adaptive farMesh.
    // It also holds onto vertex buffer data through binds
    _evalLimitContext = OsdCpuEvalLimitContext::Create(
        fmesh->GetPatchTables(), /*requierFVarData*/ false);

    return true;
}


void
OsdUtilAdaptiveEvaluator::SetCoarsePositions(
        const float *coords, int numFloats, string *errorMessage )
{
    //XXX: should be >= num coarse vertices
    if (numFloats/3 >= _refiner->GetFarMesh()->GetNumVertices()) {
        if (errorMessage)
            *errorMessage = "Indexing error in tesselator";
    } else {
        _vertexBuffer->UpdateData(coords, 0, numFloats / 3);
    }
}

bool
OsdUtilAdaptiveEvaluator::Refine(
    int numThreads, string * /* errorMessage */)
{
    const FarMesh<OsdVertex> *fmesh = _refiner->GetFarMesh();


    if (numThreads > 1) {
#ifdef OPENSUBDIV_HAS_OPENMP
        OsdOmpComputeController ompComputeController(numThreads);
        ompComputeController.Refine(_computeContext,
                                    fmesh->GetKernelBatches(),
                                    _vertexBuffer, _vvBuffer);
        return true;
#endif
    }


    OsdCpuComputeController cpuComputeController;
    cpuComputeController.Refine(_computeContext,
                                fmesh->GetKernelBatches(),
                                _vertexBuffer, _vvBuffer);

    return true;
}

void
OsdUtilAdaptiveEvaluator::EvaluateLimit(
    const OsdEvalCoords &coords, float P[3], float dPdu[3], float dPdv[3])
{

    // This controller is an empty object, essentially a namespace.
    OsdCpuEvalLimitController cpuEvalLimitController;
    
    static OsdVertexBufferDescriptor desc(0,3,3);

    // Setup evaluation controller. Values are offset, length, stride */
    OsdVertexBufferDescriptor in_desc(0, 3, 3), out_desc(0, 0, 0);
    
    cpuEvalLimitController.BindVertexBuffers<OsdCpuVertexBuffer,OsdCpuVertexBuffer>(in_desc, _vertexBuffer, out_desc, NULL);
    
    cpuEvalLimitController.EvalLimitSample(coords, _evalLimitContext, desc, P, dPdu, dPdv);
}


void ccgSubSurf__mapGridToFace(int S, float grid_u, float grid_v,
                                      float *face_u, float *face_v)
{
    float u, v;

    /* - Each grid covers half of the face along the edges.
     * - Grid's (0, 0) starts from the middle of the face.
     */
    u = 0.5f - 0.5f * grid_u;
    v = 0.5f - 0.5f * grid_v;

    if (S == 0) {
        *face_u = v;
        *face_v = u;
    }
    else if (S == 1) {
        *face_u = 1.0f - u;
        *face_v = v;
    }
    else if (S == 2) {
        *face_u = 1.0f - v;
        *face_v = 1.0f - u;
    }
    else {
        *face_u = v;
        *face_v = 1.0f - u;
    }
}


bool
OsdUtilAdaptiveEvaluator::GetRefinedTopology(
    OsdUtilSubdivTopology *out,
    //positions will have three floats * t->numVertices
    std::vector<float> *positions,
    std::string *errorMessage)
{

    const OsdUtilSubdivTopology &t = GetTopology();

    positions->clear();

    // XXX: dirk
    // What are correct values for subfaceGridSize?
    // These look good.
    // Power of 2 + 1?
    int gridSize = 5;
    int subfaceGridSize = 3;

    int coarseFaceIndex = 0;
    int ptexFaceIndex = 0;
    int numSubfacesToProcess = 0;

    bool done = false;

    while (not done ) {

        // iterate through faces in coarse topology.  Four sided
        // faces are traversed in one pass, sub-faces of non quads
        // are traversed in order, i.e. three subfaces for each triangle
        // These are the indices used by ptex and the eval API
        bool subface = false;

        if (numSubfacesToProcess > 0) {
            // We are iterating over sub-faces of a non-quad
            numSubfacesToProcess--;
            subface = true;
        } else {
            int vertsInCoarseFace = t.nverts[coarseFaceIndex++];

            if (vertsInCoarseFace != 4) {
                // Non quads are subdivided by putting a point
                // in the middle of the face and creating
                // subfaces.
                numSubfacesToProcess = vertsInCoarseFace-1;
                subface = true;
            }
        }

        int startingPositionIndex = (int) positions->size();
        int currentGridSize = gridSize;

        // Subfaces have a smaller gridsize so tessellation lines up.
        if (subface)
            currentGridSize = subfaceGridSize;

        OsdEvalCoords coords;
        coords.face = ptexFaceIndex;

        for (int x = 0; x < currentGridSize; x++) {
            for (int y = 0; y < currentGridSize; y++) {
                float grid_u = (float) x / (currentGridSize - 1),
                      grid_v = (float) y / (currentGridSize - 1);

                float P[3];

                coords.u = grid_u;
                coords.v = grid_v;

                EvaluateLimit(coords, P, NULL, NULL);
                positions->push_back(P[0]);
                positions->push_back(P[1]);
                positions->push_back(P[2]);

                // If not on edges, add a quad
                if ( (x<currentGridSize-1) and (y<currentGridSize-1)) {
                    int v[4];
                    int vBase = startingPositionIndex/3;
                    v[0] = vBase + x*currentGridSize + y;
                    v[1] = vBase + x*currentGridSize + y+1;
                    v[2] = vBase + (x+1)*currentGridSize+y+1;
                    v[3] = vBase + (x+1)*currentGridSize+y;
                    out->AddFace(4, v);
                }
            }
        }

        ptexFaceIndex++;

        if ((numSubfacesToProcess == 0) and
            (coarseFaceIndex == (int)t.nverts.size())) {
            done = true; // last face
        }

    } // while (not done)

    out->name = GetTopology().name + "_refined";
    out->numVertices = (int) positions->size()/3;
    out->refinementLevel = GetTopology().refinementLevel;

    return out->IsValid(errorMessage);
}

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
