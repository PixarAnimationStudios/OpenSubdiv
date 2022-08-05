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

//------------------------------------------------------------------------------
//  Tutorial description:
//
//      This tutorial builds on others using the SurfaceFactory, Surface
//      and Tessellation classes by using more of the functionality of the
//      Tessellation class to construct a tessellation of the mesh that is
//      topologically watertight, i.e. resulting points evaluated along
//      shared edges or vertices are shared and not duplicated.
//
//      Since Tessellation provides points around its boundary first, the
//      evaluated points for shared vertices and edges are identified when
//      constructed and reused when shared later. The boundary of the
//      tessellation of a face is therefore a collection of shared points
//      and methods of Tessellation help to remap the faces generated to
//      the shared set of points.
//

#include <opensubdiv/far/topologyRefiner.h>
#include <opensubdiv/bfr/refinerSurfaceFactory.h>
#include <opensubdiv/bfr/surface.h>
#include <opensubdiv/bfr/tessellation.h>

#include <vector>
#include <string>
#include <cstring>
#include <cstdio>

//  Local headers with support for this tutorial in "namespace tutorial"
#include "./meshLoader.h"
#include "./objWriter.h"

using namespace OpenSubdiv;

using Far::Index;
using Far::IndexArray;
using Far::ConstIndexArray;

//
//  Simple command line arguments to provide input and run-time options:
//
class Args {
public:
    std::string     inputObjFile;
    std::string     outputObjFile;
    Sdc::SchemeType schemeType;
    int             tessUniformRate;
    bool            tessQuadsFlag;

public:
    Args(int argc, char * argv[]) :
        inputObjFile(),
        outputObjFile(),
        schemeType(Sdc::SCHEME_CATMARK),
        tessUniformRate(5),
        tessQuadsFlag(false) {

        for (int i = 1; i < argc; ++i) {
            if (strstr(argv[i], ".obj")) {
                if (inputObjFile.empty()) {
                    inputObjFile = std::string(argv[i]);
                } else {
                    fprintf(stderr,
                        "Warning: Extra Obj file '%s' ignored\n", argv[i]);
                }
            } else if (!strcmp(argv[i], "-o")) {
                if (++i < argc) outputObjFile = std::string(argv[i]);
            } else if (!strcmp(argv[i], "-bilinear")) {
                schemeType = Sdc::SCHEME_BILINEAR;
            } else if (!strcmp(argv[i], "-catmark")) {
                schemeType = Sdc::SCHEME_CATMARK;
            } else if (!strcmp(argv[i], "-loop")) {
                schemeType = Sdc::SCHEME_LOOP;
            } else if (!strcmp(argv[i], "-res")) {
                if (++i < argc) tessUniformRate = atoi(argv[i]);
            } else if (!strcmp(argv[i], "-quads")) {
                tessQuadsFlag = true;
            } else {
                fprintf(stderr,
                    "Warning: Unrecognized argument '%s' ignored\n", argv[i]);
            }
        }
    }

private:
    Args() { }
};

//
//  Local helpers for the main tessellation function that follows:
//
namespace {
    inline bool
    DoEdgeVertexIndicesIncrease(int edgeInFace, ConstIndexArray & faceVerts) {

        int v0InFace = edgeInFace;
        int v1InFace = (v0InFace == (faceVerts.size()-1)) ? 0 : (v0InFace+1);

        return (faceVerts[v0InFace] < faceVerts[v1InFace]);
    }
} // end namespace

//
//  The main tessellation function:  given a mesh and vertex positions,
//  tessellate each face -- writing results in Obj format.
//
//  This tessellation function differs from earlier tutorials in that it
//  computes and used shared points at vertices and edges of the mesh.
//  These are computed and used as encountered by the faces -- rather than
//  computing all shared vertex and edge points at once (which is more
//  amenable to threading).
//
//  This method has the advantage of only constructing face Surfaces once
//  per face, but requires additional book-keeping, and accesses memory
//  less coherently (making threading more difficult).
//
void
tessellateToObj(Far::TopologyRefiner const & meshTopology,
                std::vector<float>   const & meshVertexPositions,
                Args                 const & options) {

    //
    //  Use simpler local type names for the Surface and its factory:
    //
    typedef Bfr::RefinerSurfaceFactory<> SurfaceFactory;
    typedef Bfr::Surface<float>          Surface;

    //
    //  Initialize the SurfaceFactory for the given base mesh (very low
    //  cost in terms of both time and space) and tessellate each face
    //  independently (i.e. no shared vertices):
    //
    //  Note that the SurfaceFactory is not thread-safe by default due to
    //  use of an internal cache.  Creating a separate instance of the
    //  SurfaceFactory for each thread is one way to safely parallelize
    //  this loop.  Another (preferred) is to assign a thread-safe cache
    //  to the single instance.
    //
    //  First declare any evaluation options when initializing (though
    //  none are used in this simple case):
    //
    SurfaceFactory::Options surfaceOptions;

    SurfaceFactory meshSurfaceFactory(meshTopology, surfaceOptions);

    //
    //  The Surface to be constructed and evaluated for each face -- as
    //  well as the intermediate and output data associated with it -- can
    //  be declared in the scope local to each face. But since dynamic
    //  memory is involved with these variables, it is preferred to declare
    //  them outside that loop to preserve and reuse that dynamic memory.
    //
    Surface posSurface;

    std::vector<float> facePatchPoints;

    std::vector<float> outCoords;
    std::vector<float> outPos, outDu, outDv;
    std::vector<int>   outFacets;

    //
    //  Assign Tessellation Options applied for all faces.  Tessellations
    //  allow the creating of either 3- or 4-sided faces -- both of which
    //  are supported here via a command line option:
    //
    int const tessFacetSize = 3 + options.tessQuadsFlag;

    Bfr::Tessellation::Options tessOptions;
    tessOptions.SetFacetSize(tessFacetSize);
    tessOptions.PreserveQuads(options.tessQuadsFlag);

    //
    //  Vectors to identify shared tessellation points at vertices and
    //  edges and their indices around the boundary of a face:
    //
    Far::TopologyLevel const & baseLevel = meshTopology.GetLevel(0);

    std::vector<int> sharedVertexPointIndex(baseLevel.GetNumVertices(), -1);
    std::vector<int> sharedEdgePointIndex(baseLevel.GetNumEdges(), -1);

    std::vector<int> tessBoundaryIndices;

    //
    //  Process each face, writing the output of each in Obj format:
    //
    tutorial::ObjWriter objWriter(options.outputObjFile);

    int numMeshPointsEvaluated = 0;

    int numFaces = meshSurfaceFactory.GetNumFaces();
    for (int faceIndex = 0; faceIndex < numFaces; ++faceIndex) {
        //
        //  Initialize the Surface for this face -- if valid (skipping
        //  holes and boundary faces in some rare cases):
        //
        if (!meshSurfaceFactory.InitVertexSurface(faceIndex, &posSurface)) {
            continue;
        }

        //
        //  Declare a simple uniform Tessellation for the Parameterization
        //  of this face and identify coordinates of the points to evaluate:
        //
        Bfr::Tessellation tessPattern(posSurface.GetParameterization(),
                                      options.tessUniformRate, tessOptions);

        int numTessCoords = tessPattern.GetNumCoords();

        outCoords.resize(numTessCoords * 2);

        tessPattern.GetCoords(outCoords.data());

        //
        //  Prepare the patch points for the Surface, then use them to
        //  evaluate output points for all identified coordinates:
        //
        //  Resize patch point and output arrays:
        int pointSize = 3;

        facePatchPoints.resize(posSurface.GetNumPatchPoints() * pointSize);

        outPos.resize(numTessCoords * pointSize);
        outDu.resize(numTessCoords * pointSize);
        outDv.resize(numTessCoords * pointSize);

        posSurface.PreparePatchPoints(meshVertexPositions.data(), pointSize,
                                      facePatchPoints.data(), pointSize);

        //
        //  Evaluate the sample points of the Tessellation:
        //
        //  First we traverse the boundary of the face to determine whether
        //  to evaluate or share points on vertices and edges of the face.
        //  Both pre-existing and new boundary points are identified by
        //  index in an index buffer for later use.  The interior points
        //  are trivially computed after the boundary is dealt with.
        //
        //  Identify the boundary and interior coords and initialize the
        //  buffer for the potentially shared boundary points:
        //
        int numBoundaryCoords = tessPattern.GetNumBoundaryCoords();
        int numInteriorCoords = numTessCoords - numBoundaryCoords;

        float const * tessBoundaryCoords = &outCoords[0];
        float const * tessInteriorCoords = &outCoords[numBoundaryCoords*2];

        ConstIndexArray fVerts = baseLevel.GetFaceVertices(faceIndex);
        ConstIndexArray fEdges = baseLevel.GetFaceEdges(faceIndex);

        tessBoundaryIndices.resize(numBoundaryCoords);

        //
        //  Walk around the face, inspecting each vertex and outgoing edge,
        //  and populating the boundary index buffer in the process:
        //
        float * patchPointData = facePatchPoints.data();

        int boundaryIndex = 0;
        int numFacePointsEvaluated = 0;
        for (int i = 0; i < fVerts.size(); ++i) {
            //  Evaluate and/or retrieve the shared point for the vertex:
            {
                int & vertPointIndex = sharedVertexPointIndex[fVerts[i]];
                if (vertPointIndex < 0) {
                    vertPointIndex = numMeshPointsEvaluated ++;

                    float const * uv = &tessBoundaryCoords[boundaryIndex*2];

                    int k = (numFacePointsEvaluated ++) * pointSize;
                    posSurface.Evaluate(uv, patchPointData, pointSize,
                                        &outPos[k], &outDu[k], &outDv[k]);
                }
                tessBoundaryIndices[boundaryIndex++] = vertPointIndex;
            }

            //  Evaluate and/or retrieve all shared points for the edge:
            int N = options.tessUniformRate - 1;
            if (N) {
                //  Be careful to respect ordering of the edge and its
                //  points when both evaluating and identifying indices:
                bool edgeIsNotReversed = DoEdgeVertexIndicesIncrease(i, fVerts);

                int iOffset = edgeIsNotReversed ? 0 : (N - 1);
                int iDelta  = edgeIsNotReversed ? 1 : -1;

                int & edgePointIndex = sharedEdgePointIndex[fEdges[i]];
                if (edgePointIndex < 0) {
                    edgePointIndex = numMeshPointsEvaluated;

                    float const * uv = &tessBoundaryCoords[boundaryIndex*2];

                    int iNext = numFacePointsEvaluated + iOffset;
                    for (int j = 0; j < N; ++j, iNext += iDelta, uv += 2) {
                        int k = iNext * pointSize;
                        posSurface.Evaluate(uv, patchPointData, pointSize,
                                            &outPos[k], &outDu[k], &outDv[k]);
                    }
                    numFacePointsEvaluated += N;
                    numMeshPointsEvaluated += N;
                }
                int iNext = edgePointIndex + iOffset;
                for (int j = 0; j < N; ++j, iNext += iDelta) {
                    tessBoundaryIndices[boundaryIndex++] = iNext;
                }
            }
        }

        //
        //  Evaluate any interior points unique to this face -- appending
        //  them to those shared points computed above for the boundary:
        //
        if (numInteriorCoords) {
            float const * uv = tessInteriorCoords;

            int iLast = numFacePointsEvaluated + numInteriorCoords;
            for (int i = numFacePointsEvaluated; i < iLast; ++i, uv += 2) {
                int k = i * pointSize;
                posSurface.Evaluate(uv, patchPointData, pointSize,
                                    &outPos[k], &outDu[k], &outDv[k]);
            }
            numFacePointsEvaluated += numInteriorCoords;
            numMeshPointsEvaluated += numInteriorCoords;
        }

        //
        //  Remember to trim/resize the buffers storing evaluation results
        //  for new points to reflect the size actually populated.
        //
        outPos.resize(numFacePointsEvaluated * pointSize);
        outDu.resize(numFacePointsEvaluated * pointSize);
        outDv.resize(numFacePointsEvaluated * pointSize);

        //
        //  Identify the faces of the Tessellation:
        //
        //  Note that the coordinate indices used by the facets are local
        //  to the face (i.e. they range from [0..N-1], where N is the
        //  number of coordinates in the pattern) and so need to be offset
        //  when writing to Obj format.
        //
        //  For more advanced use, the coordinates associated with the
        //  boundary and interior of the pattern are distinguishable so
        //  that those on the boundary can be easily remapped to refer to
        //  shared edge or corner points, while those in the interior can
        //  be separately offset or similarly remapped.
        //
        //  So transform the indices of the facets here as needed using
        //  the indices of shared boundary points assembled above and a
        //  suitable offset for the new interior points added:
        //
        int tessInteriorOffset = numMeshPointsEvaluated - numTessCoords;

        int numFacets = tessPattern.GetNumFacets();
        outFacets.resize(numFacets * tessFacetSize);
        tessPattern.GetFacets(outFacets.data());

        tessPattern.TransformFacetCoordIndices(outFacets.data(),
                        tessBoundaryIndices.data(), tessInteriorOffset);

        //
        //  Write the evaluated points and faces connecting them as Obj:
        //
        objWriter.WriteGroupName("baseFace_", faceIndex);

        objWriter.WriteVertexPositions(outPos);
        objWriter.WriteVertexNormals(outDu, outDv);

        objWriter.WriteFaces(outFacets, tessFacetSize, true, false);
    }
}

//
//  Load command line arguments, specified or default geometry and process:
//
int
main(int argc, char * argv[]) {

    Args args(argc, argv);

    Far::TopologyRefiner * meshTopology = 0;
    std::vector<float>     meshVtxPositions;
    std::vector<float>     meshFVarUVs;

    meshTopology = tutorial::createTopologyRefiner(
            args.inputObjFile, args.schemeType, meshVtxPositions, meshFVarUVs);
    if (meshTopology == 0) {
        return EXIT_FAILURE;
    }

    tessellateToObj(*meshTopology, meshVtxPositions, args);

    delete meshTopology;
    return EXIT_SUCCESS;
}

//------------------------------------------------------------------------------
