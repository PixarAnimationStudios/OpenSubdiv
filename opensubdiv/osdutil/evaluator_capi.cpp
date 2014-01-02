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

#include "evaluator_capi.h"

#include <iostream>

#include "adaptiveEvaluator.h"
#include "topology.h"

#if !defined(WITH_ASSERT_ABORT)
#  define OSD_abort()
#else
#  include <stdlib.h>
#  define OSD_abort() abort()
#endif

#if defined(_MSC_VER)
#  define __func__ __FUNCTION__
#endif

/* Define this when you want to have additional verbosity
 * about what's being passed to OpenSubdiv.
 */
#undef DEBUG_PRINT

using namespace OpenSubdiv;

/* **************** Types declaration **************** */

typedef struct OpenSubdiv_EvaluatorDescr {
    PxOsdUtilSubdivTopology topology;
    PxOsdUtilAdaptiveEvaluator evaluator;
    std::vector<float> coarsePositions;
} OpenSubdiv_EvaluatorDescr;


/* **************** Mesh descriptor functions **************** */

OpenSubdiv_EvaluatorDescr *openSubdiv_createEvaluatorDescr(int numVertices)
{
    OpenSubdiv_EvaluatorDescr *evaluator_descr =
	new OpenSubdiv_EvaluatorDescr();
    
    evaluator_descr->topology.numVertices = numVertices;
    
    return evaluator_descr;
}

void openSubdiv_deleteEvaluatorDescr(
    OpenSubdiv_EvaluatorDescr *evaluator_descr)
{
    delete(evaluator_descr);
}

void openSubdiv_createEvaluatorDescrFace(OpenSubdiv_EvaluatorDescr *evaluator_descr, int num_vertices, int *indices)
{
#ifdef DEBUG_PRINT
        printf("Adding face index:%d vertices:", (int)evaluator_descr->topology.nverts.size());
	for (int i = 0; i < num_vertices; i++) {
		printf("%d ", indices[i]);
	}
	printf("\n");
#endif

        evaluator_descr->topology.nverts.push_back(num_vertices);
	for (int i = 0; i < num_vertices; i++) {
            evaluator_descr->topology.indices.push_back(indices[i]);
        }
}

void openSubdiv_finishEvaluatorDescr(OpenSubdiv_EvaluatorDescr *evaluator_descr,
                                int refinementLevel)
{
    std::string errorMessage;
    evaluator_descr->topology.refinementLevel = refinementLevel;
    
    if (not evaluator_descr->topology.IsValid(&errorMessage)) {
        std::cout <<"OpenSubdiv topology is not valid due to " << errorMessage << std::endl;
    } else {
        if (not evaluator_descr->evaluator.Initialize(
                evaluator_descr->topology, &errorMessage)) {
            std::cout <<"OpenSubdiv uniform evaluator initialization failed due to " << errorMessage << std::endl;
        }
    }
}


int openSubdiv_setEvaluatorCoarsePositions(
    struct OpenSubdiv_EvaluatorDescr *evaluator_descr,
    const float *positions, int numVertices)
{
    std::string errorMessage;

    // TODO: returns void, need error check on length of positions?
    evaluator_descr->evaluator.SetCoarsePositions(
        &evaluator_descr->coarsePositions[0],
        (int)evaluator_descr->coarsePositions.size(),
	&errorMessage);
    
    // Refine with 1 thread for now
    if (not evaluator_descr->evaluator.Refine(1, &errorMessage)) {
        std::cout << "OpenSubdiv refinement failed due to " << errorMessage << std::endl;
        return 0;
    }

    return 1;
}


void openSubdiv_evaluateLimit(
    struct OpenSubdiv_EvaluatorDescr *evaluation_descr,
    int face_id, float u, float v,
    float P[3], float dPdu[3], float dPdv[3])
{
    OsdEvalCoords coords;
    coords.u = u;
    coords.v = v;
    coords.face = face_id;

    evaluation_descr->evaluator.EvaluateLimit(coords, P, dPdu, dPdv);
}
   
