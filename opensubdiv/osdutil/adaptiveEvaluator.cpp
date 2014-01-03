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

using namespace OpenSubdiv;
using namespace std;


PxOsdUtilAdaptiveEvaluator::PxOsdUtilAdaptiveEvaluator():
    _refiner(NULL),
    _ownsRefiner(false),
    _computeContext(NULL),
    _evalLimitContext(NULL),
    _vertexBuffer(NULL),
    _vvBuffer(NULL),    
    _vbufP(NULL),
    _vbufdPdu(NULL),
    _vbufdPdv(NULL),
    _pOutput(NULL),
    _dPduOutput(NULL),
    _dPdvOutput(NULL)        
{
}

PxOsdUtilAdaptiveEvaluator::~PxOsdUtilAdaptiveEvaluator()
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
    if (_vbufP)
	delete _vbufP;
    if (_vbufdPdu)
	delete _vbufdPdu;
    if (_vbufdPdv)
	delete _vbufdPdv;
}


bool
PxOsdUtilAdaptiveEvaluator::Initialize(
    const PxOsdUtilSubdivTopology &t,
    string *errorMessage)    
{

    // create and initialize a refiner, passing "true" for adaptive
    // to indicate we wish for adaptive refinement rather than uniform
    PxOsdUtilRefiner *refiner = new PxOsdUtilRefiner();
    _ownsRefiner = true;

    if (not refiner->Initialize(t, true, errorMessage)) {
        return false;   
    }

    return Initialize(refiner, errorMessage);
}

bool
PxOsdUtilAdaptiveEvaluator::Initialize(
    PxOsdUtilRefiner *refiner,
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


    _computeContext = OsdCpuComputeContext::Create(fmesh);
    
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
    _computeContext = OsdCpuComputeContext::Create(fmesh);

    // A context object used to store data used in fast limit surface
    // evaluation.  This contains vectors of patches and associated
    // tables pulled and computed from the adaptive farMesh.
    // It also holds onto vertex buffer data through binds    
    _evalLimitContext = OsdCpuEvalLimitContext::Create(
        fmesh, /*requierFVarData*/ false);
    
    // A buffer with one float per target point to use when
    // evaluating interpolated weights
    OsdCpuVertexBuffer* _vbufP = OsdCpuVertexBuffer::Create(3, 1);
    OsdCpuVertexBuffer* _vbufdPdu = OsdCpuVertexBuffer::Create(3, 1);
    OsdCpuVertexBuffer* _vbufdPdv = OsdCpuVertexBuffer::Create(3, 1);
    _pOutput = _vbufP->BindCpuBuffer();
    _dPduOutput = _vbufdPdu->BindCpuBuffer();
    _dPdvOutput = _vbufdPdv->BindCpuBuffer();
    
    memset( (void*)_pOutput, 0, 3 * sizeof(float));
    memset( (void*)_dPduOutput, 0, 3 * sizeof(float));
    memset( (void*)_dPdvOutput, 0, 3 * sizeof(float));    

    // Setup evaluation context. Values are offset, length, stride */
    OsdVertexBufferDescriptor in_desc(0, 3, 3), out_desc(0, 3, 3); 
    _evalLimitContext->GetVertexData().Bind(in_desc, _vertexBuffer, out_desc,
					    _vbufP, _vbufdPdu, _vbufdPdv);

    std::cout << "Initialized adaptive evaluator\n";
    return true;
}


void
PxOsdUtilAdaptiveEvaluator::SetCoarsePositions(
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
PxOsdUtilAdaptiveEvaluator::Refine(
    int numThreads, string *errorMessage)
{
    const FarMesh<OsdVertex> *fmesh = _refiner->GetFarMesh();
    
#ifdef OPENSUBDIV_HAS_OPENMP
    
    if (numThreads > 1) {
        OsdOmpComputeController ompComputeController(numThreads);
        ompComputeController.Refine(_computeContext,
                                    fmesh->GetKernelBatches(),
                                    _vertexBuffer, _vvBuffer);
        return true;
    }
    
#endif

    OsdCpuComputeController cpuComputeController;
    cpuComputeController.Refine(_computeContext,
                                fmesh->GetKernelBatches(),
                                _vertexBuffer, _vvBuffer);        

    return true;
}

void
PxOsdUtilAdaptiveEvaluator::EvaluateLimit(
    const OsdEvalCoords &coords, float P[3], float dPdu[3], float dPdv[3])
{

    // This controller is an empty object, essentially a namespace.
    OsdCpuEvalLimitController cpuEvalLimitController;
    
    cpuEvalLimitController.
	EvalLimitSample<OsdCpuVertexBuffer, OsdCpuVertexBuffer>(
            coords, _evalLimitContext, 0 /*index*/);

    // Copy results from vertex buffers into return parameters
    memcpy(P, _pOutput, sizeof(float) * 3);
    if (dPdu) {
	memcpy(dPdu, _dPduOutput, sizeof(float) * 3);
    }
    if (dPdv) {
	memcpy(dPdv, _dPdvOutput, sizeof(float) * 3);
    }
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
PxOsdUtilAdaptiveEvaluator::GetRefinedTopology(
    PxOsdUtilSubdivTopology *out,
    //positions will have three floats * t->numVertices
    std::vector<float> *positions,
    std::string *errorMessage)
{

    const PxOsdUtilSubdivTopology &t = GetTopology();

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

     




