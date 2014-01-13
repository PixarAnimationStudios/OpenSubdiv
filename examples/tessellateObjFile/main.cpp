//
//     Copyright (C) Pixar. All rights reserved.
//
//     This license governs use of the accompanying software. If you
//     use the software, you accept this license. If you do not accept
//     the license, do not use the software.
//
//     1. Definitions
//     The terms "reproduce," "reproduction," "derivative works," and
//     "distribution" have the same meaning here as under U.S.
//     copyright law.  A "contribution" is the original software, or
//     any additions or changes to the software.
//     A "contributor" is any person or entity that distributes its
//     contribution under this license.
//     "Licensed patents" are a contributor's patent claims that read
//     directly on its contribution.
//
//     2. Grant of Rights
//     (A) Copyright Grant- Subject to the terms of this license,
//     including the license conditions and limitations in section 3,
//     each contributor grants you a non-exclusive, worldwide,
//     royalty-free copyright license to reproduce its contribution,
//     prepare derivative works of its contribution, and distribute
//     its contribution or any derivative works that you create.
//     (B) Patent Grant- Subject to the terms of this license,
//     including the license conditions and limitations in section 3,
//     each contributor grants you a non-exclusive, worldwide,
//     royalty-free license under its licensed patents to make, have
//     made, use, sell, offer for sale, import, and/or otherwise
//     dispose of its contribution in the software or derivative works
//     of the contribution in the software.
//
//     3. Conditions and Limitations
//     (A) No Trademark License- This license does not grant you
//     rights to use any contributor's name, logo, or trademarks.
//     (B) If you bring a patent claim against any contributor over
//     patents that you claim are infringed by the software, your
//     patent license from such contributor to the software ends
//     automatically.
//     (C) If you distribute any portion of the software, you must
//     retain all copyright, patent, trademark, and attribution
//     notices that are present in the software.
//     (D) If you distribute any portion of the software in source
//     code form, you may do so only under this license by including a
//     complete copy of this license with your distribution. If you
//     distribute any portion of the software in compiled or object
//     code form, you may only do so under a license that complies
//     with this license.
//     (E) The software is licensed "as-is." You bear the risk of
//     using it. The contributors give no express warranties,
//     guarantees or conditions. You may have additional consumer
//     rights under your local laws which this license cannot change.
//     To the extent permitted under your local laws, the contributors
//     exclude the implied warranties of merchantability, fitness for
//     a particular purpose and non-infringement.
//

#if defined( _WIN32)
#include <windows.h>
#endif

#include <osdutil/adaptiveEvaluator.h>
#include <osdutil/uniformEvaluator.h>
#include <osdutil/topology.h>

#include <osd/error.h>

#include <vector>
#include <iostream>
#include <stdio.h>

#ifdef OPENSUBDIV_HAS_OPENMP
    #include <omp.h>
#endif

using namespace OpenSubdiv;


//------------------------------------------------------------------------------
static bool
uniformTessellate(char *inputFile, char *outputFile, std::string *errorMessage)
{

    PxOsdUtilSubdivTopology topology;
    std::vector<float> pointPositions;
    
    if (not topology.ReadFromObjFile(inputFile, &pointPositions, errorMessage)) {
        return false;
    }

    topology.refinementLevel = 2;

    PxOsdUtilUniformEvaluator uniformEvaluator;

    // Create uniformEvaluator
    if (not uniformEvaluator.Initialize(topology, errorMessage)) {
        std::cout << "Initialize failed with " << *errorMessage << "\n";        
        return false;
    }

    // Push the vertex data
    uniformEvaluator.SetCoarsePositions(pointPositions, errorMessage);

    // Refine with one thread
    if (not uniformEvaluator.Refine(1, errorMessage)) {
        std::cout << "Refine failed with " << *errorMessage << "\n";
        return false;
    }

    PxOsdUtilSubdivTopology refinedTopology;
    const float *positions = NULL;
   
    if (not uniformEvaluator.GetRefinedTopology(
            &refinedTopology, &positions, errorMessage)) {
        std::cout << "GetRefinedTopology failed with " << *errorMessage <<"\n";
        return false;
    }
    
    if (not refinedTopology.WriteObjFile(
            outputFile, positions, errorMessage)) {
        std::cout << errorMessage << std::endl;             
    }

    return true;
}


static bool
blenderStyleTessellate(char *inputFile, char *outputFile, std::string *errorMessage)
{

    PxOsdUtilSubdivTopology topology;
    std::vector<float> pointPositions;
    
    if (not topology.ReadFromObjFile(inputFile, &pointPositions, errorMessage)) {
        return false;
    }

    topology.refinementLevel = 5;

    PxOsdUtilAdaptiveEvaluator adaptiveEvaluator;

    // Create adaptiveEvaluator
    if (not adaptiveEvaluator.Initialize(topology, errorMessage)) {
        std::cout << "Initialize failed with " << *errorMessage << "\n";        
        return false;
    }

    // Push the vertex data
    adaptiveEvaluator.SetCoarsePositions(
        &(pointPositions[0]), (int) pointPositions.size(), errorMessage);

    // Refine with one thread
    if (not adaptiveEvaluator.Refine(1, errorMessage)) {
        std::cout << "Refine failed with " << *errorMessage << "\n";
        return false;
    }

    PxOsdUtilSubdivTopology refinedTopology;
    std::vector<float> positions;
   
    if (not adaptiveEvaluator.GetRefinedTopology(
            &refinedTopology, &positions, errorMessage)) {
        std::cout << "GetRefinedTopology failed with " << *errorMessage <<"\n";
        return false;
    }
    
    if (not refinedTopology.WriteObjFile(
            outputFile, &(positions[0]), errorMessage)) {
        std::cout << errorMessage << std::endl;             
    }

    return true;
}

//------------------------------------------------------------------------------
static void
callbackError(OpenSubdiv::OsdErrorType err, const char *message)
{
    printf("OsdError: %d\n", err);
    printf("%s", message);
}


//------------------------------------------------------------------------------
int main(int argc, char** argv) {



    OsdSetErrorCallback(callbackError);

    std::string errorMessage;

    if ((argc == 4) and (std::string(argv[1]) == std::string("-blender"))) {
	std::cout << "About to call blender-style tessellate\n";
	if (not blenderStyleTessellate(argv[2], argv[3], &errorMessage)) {
	    std::cout << "Failed with error: " << errorMessage << std::endl;
	    return -1;
	}
    } else if (argc != 3) {
        std::cout << "Usage: tessellate [-blender] input.obj output.obj\n";
        return false;
    } else {
	if (not uniformTessellate(argv[1], argv[2], &errorMessage)) {
	    std::cout << "Failed with error: " << errorMessage << std::endl;
	}
    }
}
