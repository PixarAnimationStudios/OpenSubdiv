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

#include "uniformEvaluator.h"

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


PxOsdUtilUniformEvaluator::PxOsdUtilUniformEvaluator():
    _refiner(NULL),
    _ownsRefiner(false),
    _computeContext(NULL),
    _vertexBuffer(NULL),
    _vvBuffer(NULL)
{
}

PxOsdUtilUniformEvaluator::~PxOsdUtilUniformEvaluator()
{
    if (_ownsRefiner and _refiner) {
        delete _refiner;
    }
    if (_computeContext)
        delete _computeContext;
    if (_vertexBuffer)
        delete _vertexBuffer;
    if (_vvBuffer)
        delete _vvBuffer;
}


bool
PxOsdUtilUniformEvaluator::Initialize(
    const PxOsdUtilSubdivTopology &t,
    string *errorMessage)    
{

    // create and initialize a refiner, passing "false" for adaptive
    // to indicate we wish for uniform refinement
    PxOsdUtilRefiner *refiner = new PxOsdUtilRefiner();
    _ownsRefiner = true;

    if (not refiner->Initialize(t, false, errorMessage)) {
        return false;   
    }

    return Initialize(refiner, errorMessage);
}

bool
PxOsdUtilUniformEvaluator::Initialize(
    PxOsdUtilRefiner *refiner,
    string *errorMessage)    
{    

    if (refiner->GetAdaptive()) {
        if (errorMessage)
            *errorMessage = "Uniform evaluator requires uniform refiner";
        return false;
    }

    // Note we assume someone else keeps this pointer alive
    _refiner = refiner;
    _ownsRefiner = false; 
    
    const FarMesh<OsdVertex> *fmesh = _refiner->GetFarMesh();
    const HbrMesh<OsdVertex> *hmesh = _refiner->GetHbrMesh();
    const vector<string> &vvNames   = _refiner->GetTopology().vvNames;
    
    if (not (fmesh and hmesh)) {
        if (errorMessage)
            *errorMessage = "No valid uniform far/hbr mesh";
        return false;
    }

    // No need to create a far mesh if no subdivision is required.
    if (_refiner->GetTopology().refinementLevel == 0) {
        
        // Three elements per unrefined point
        _vertexBuffer = OsdCpuVertexBuffer::Create(
            3, hmesh->GetNumVertices());

        // zeros
        memset( _vertexBuffer->BindCpuBuffer(), 0,
                3 * hmesh->GetNumVertices() * sizeof(float));        
        
        if (vvNames.size()) {

            // One element in the vertex buffer for each
            // named vertex varying attribute in the unrefined mesh
            _vvBuffer = OsdCpuVertexBuffer::Create(
                (int)vvNames.size(), hmesh->GetNumVertices());

            // zeros
            memset( _vvBuffer->BindCpuBuffer(), 0,
                    vvNames.size() * hmesh->GetNumVertices() * sizeof(float));
        }
        return true;
    }

    _computeContext = OsdCpuComputeContext::Create(fmesh);
    
    // Three elements per refined point    
    _vertexBuffer = OsdCpuVertexBuffer::Create(
        3, fmesh->GetNumVertices());
    
    // zeros
    memset( _vertexBuffer->BindCpuBuffer(), 0,
            3 * fmesh->GetNumVertices() * sizeof(float));

    
    if (vvNames.size()) {

        // One element in the vertex buffer for each
        // named vertex varying attribute in the refined mesh
        _vvBuffer = OsdCpuVertexBuffer::Create(
            (int)vvNames.size(), fmesh->GetNumVertices());
        
        // zeros
        memset( _vvBuffer->BindCpuBuffer(), 0,
                vvNames.size() * fmesh->GetNumVertices() * sizeof(float));
    }

    return true;
}


void
PxOsdUtilUniformEvaluator::SetCoarsePositions(
    const vector<float>& coords, string *errorMessage ) 
{
    const float* pFloats = &coords.front();
    int numFloats = (int) coords.size();

    //XXX: should be >= num coarse vertices
    if (numFloats/3 >= _refiner->GetFarMesh()->GetNumVertices()) {
        if (errorMessage)
            *errorMessage = "Indexing error in tesselator";
    } else {
        _vertexBuffer->UpdateData(pFloats, 0, numFloats / 3);
    }
}


void
PxOsdUtilUniformEvaluator::SetCoarseVVData(
    const vector<float>& data, string *errorMessage
    ) 
{
    if (!_vvBuffer) {
        if (!data.empty() and errorMessage)
            *errorMessage = 
                "Mesh was not constructed with VV variables.";
        return;                 
    }
    
    int numElements = _vvBuffer->GetNumElements();
    int numVertices = (int) data.size() / numElements;
    const float* pFloats = &data.front();
    _vvBuffer->UpdateData(pFloats, 0, numVertices);
}


bool
PxOsdUtilUniformEvaluator::Refine(
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

bool
PxOsdUtilUniformEvaluator::GetRefinedPositions(
    const float **positions, int *numFloats,
    string *errorMessage) const 
{

    if (not (positions and numFloats)) {
        if (errorMessage) {
            *errorMessage =
                "GetRefinedPositions: positions and/or numFloats was NULL";
        }
        return false;
    }
    
    if (!_refiner->IsRefined()) {
        if (errorMessage) {
            *errorMessage = "GetRefinedPositions: Mesh has not been refined.";
        }
        return false;
    }           

    int numRefinedVerts = _refiner->GetNumRefinedVertices();
    int firstVertexOffset = _refiner->GetFirstVertexOffset();
        
    if (numRefinedVerts == 0) {
        if (errorMessage) {
            *errorMessage = "GetRefinedPositions: not refined.";
        }        
        return false;
    }


    // The vertexBuffer has all subdivision levels, here we are skipping
    // past the vertices on lower subdivision levels and returning
    // a pointer to the start of the most refined level
    *positions = _vertexBuffer->BindCpuBuffer() + (3*firstVertexOffset);
    *numFloats = numRefinedVerts*3;

    return true;
}


bool
PxOsdUtilUniformEvaluator::GetRefinedVVData(
    float **data, int *numFloats, int *numElementsRetVal,
    std::string *errorMessage ) const
{


    if (not (data and numFloats)) {
        if (errorMessage) {
            *errorMessage =
                "GetRefinedVVData: data and/or numFloats was NULL";
        }
        return false;
    }
    
    if (!_refiner->IsRefined()) {
        if (errorMessage) {
            *errorMessage = "GetRefinedVVData: Mesh has not been refined.";
        }
        return false;
    }           

    int numRefinedVerts = _refiner->GetNumRefinedVertices();
    int firstVertexOffset = _refiner->GetFirstVertexOffset();
        
    if (numRefinedVerts == 0) {
        if (errorMessage) {
            *errorMessage = "GetRefinedVVData: not refined.";
        }        
        return false;
    }


    int numElements = (int)GetTopology().vvNames.size();
    if (numElementsRetVal)
        *numElementsRetVal = numElements;
            
    // The vertexBuffer has all subdivision levels, here we are skipping
    // past the vertices on lower subdivision levels and returning
    // a pointer to the start of the most refined level
    *data =
            _vvBuffer->BindCpuBuffer() + (numElements * firstVertexOffset);
    *numFloats = numElements * numRefinedVerts;

    return true;
}




bool
PxOsdUtilUniformEvaluator::GetRefinedTopology(
    PxOsdUtilSubdivTopology *t,
    //positions will have three floats * t->numVertices
    const float **positions,
    std::string *errorMessage)
{
    
    if (not GetRefinedQuads(&t->indices, errorMessage)) {
        return false;
    }

    int numQuads = (int)t->indices.size()/4;
    t->nverts.resize(numQuads);
    for (int i=0; i<numQuads; ++i) {
        t->nverts[i] = 4;
    }

    int numFloats = 0;
    if (not GetRefinedPositions(positions, &numFloats, errorMessage)) {
        return false;           
    }

    t->name = GetTopology().name + "_refined";
    t->numVertices = numFloats/3;
    t->refinementLevel = GetTopology().refinementLevel;

    return t->IsValid(errorMessage);
}
