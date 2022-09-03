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
#include <cassert>

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
//  Simple local structs supporting shared points for vertices and edges:
//
namespace {
    struct SharedVertex {
        SharedVertex() : pointIndex(-1) { }

        bool IsSet() const { return pointIndex >= 0; }
        void Set(int index) { pointIndex = index; }

        int pointIndex;
    };

    struct SharedEdge {
        SharedEdge() : pointIndex(-1), numPoints(0) { }

        bool IsSet() const { return pointIndex >= 0; }
        void Set(int index, int n) { pointIndex = index, numPoints = n; }

        int pointIndex;
        int numPoints;
    };
} // end namespace

//
//  The main tessellation function:  given a mesh and vertex positions,
//  tessellate each face -- writing results in Obj format.
//
//  This tessellation function differs from earlier tutorials in that it
//  computes and reuses shared points at vertices and edges of the mesh.
//  There are several ways to compute these shared points, and which is
//  best depends on context.
//
//  Dealing with shared data poses complications for threading in general,
//  so computing all points for the vertices and edges up front may be
//  preferred -- despite the fact that faces will be visited more than once
//  (first when generating potentially shared vertex or edge points, and
//  later when generating any interior points). The loops for vertices and
//  edges can be threaded and the indexing of the shared points is simpler.
//
//  For the single-threaded case here, the faces are each processed in
//  order and any shared points will be computed and used as needed. So
//  each face is visited once (and so each Surface initialized once) but
//  the bookkeeping to deal with indices of shared points becomes more
//  complicated.
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
    Surface faceSurface;

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
    //  Declare vectors to identify shared tessellation points at vertices
    //  and edges and their indices around the boundary of a face:
    //
    Far::TopologyLevel const & baseLevel = meshTopology.GetLevel(0);

    std::vector<SharedVertex> sharedVerts(baseLevel.GetNumVertices());
    std::vector<SharedEdge>   sharedEdges(baseLevel.GetNumEdges());

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
        if (!meshSurfaceFactory.InitVertexSurface(faceIndex, &faceSurface)) {
            continue;
        }

        //
        //  Declare a simple uniform Tessellation for the Parameterization
        //  of this face and identify coordinates of the points to evaluate:
        //
        Bfr::Tessellation tessPattern(faceSurface.GetParameterization(),
                                      options.tessUniformRate, tessOptions);

        int numOutCoords = tessPattern.GetNumCoords();

        outCoords.resize(numOutCoords * 2);

        tessPattern.GetCoords(outCoords.data());

        //
        //  Prepare the patch points for the Surface, then use them to
        //  evaluate output points for all identified coordinates:
        //
        //  Resize patch point and output arrays:
        int pointSize = 3;

        facePatchPoints.resize(faceSurface.GetNumPatchPoints() * pointSize);

        outPos.resize(numOutCoords * pointSize);
        outDu.resize(numOutCoords * pointSize);
        outDv.resize(numOutCoords * pointSize);

        //  Populate the patch point array:
        faceSurface.PreparePatchPoints(meshVertexPositions.data(), pointSize,
                                       facePatchPoints.data(), pointSize);

        //
        //  Evaluate the sample points of the Tessellation:
        //
        //  First traverse the boundary of the face to determine whether
        //  to evaluate or share points on vertices and edges of the face.
        //  Both pre-existing and new boundary points are identified by
        //  index in an array for later use.  The interior points are all
        //  trivially computed after the boundary is dealt with.
        //
        //  Identify the boundary and interior coords and initialize the
        //  index array for the potentially shared boundary points:
        //
        int numBoundaryCoords = tessPattern.GetNumBoundaryCoords();
        int numInteriorCoords = numOutCoords - numBoundaryCoords;

        float const * tessBoundaryCoords = &outCoords[0];
        float const * tessInteriorCoords = &outCoords[numBoundaryCoords*2];

        ConstIndexArray fVerts = baseLevel.GetFaceVertices(faceIndex);
        ConstIndexArray fEdges = baseLevel.GetFaceEdges(faceIndex);

        tessBoundaryIndices.resize(numBoundaryCoords);

        //
        //  Walk around the face, inspecting each vertex and outgoing edge,
        //  and populating the index array of boundary points:
        //
        float * patchPointData = facePatchPoints.data();

        int boundaryIndex = 0;
        int numFacePointsEvaluated = 0;
        for (int i = 0; i < fVerts.size(); ++i) {
            Index vertIndex = fVerts[i];
            Index edgeIndex = fEdges[i];
            int   edgeRate  = options.tessUniformRate;

            //
            //  Evaluate/assign or retrieve the shared point for the vertex:
            //
            SharedVertex & sharedVertex = sharedVerts[vertIndex];
            if (!sharedVertex.IsSet()) {
                //  Identify indices of the new shared point in both the
                //  mesh and face and increment their inventory:
                int indexInMesh = numMeshPointsEvaluated++;
                int indexInFace = numFacePointsEvaluated++;

                sharedVertex.Set(indexInMesh);

                //  Evaluate new shared point and assign index to boundary:
                float const * uv = &tessBoundaryCoords[boundaryIndex*2];

                int pIndex = indexInFace * pointSize;
                faceSurface.Evaluate(uv, patchPointData, pointSize,
                        &outPos[pIndex], &outDu[pIndex], &outDv[pIndex]);

                tessBoundaryIndices[boundaryIndex++] = indexInMesh;
            } else {
                //  Assign shared vertex point index to boundary:
                tessBoundaryIndices[boundaryIndex++] = sharedVertex.pointIndex;
            }

            //
            //  Evaluate/assign or retrieve all shared points for the edge:
            //
            //  To keep this simple, assume the edge is manifold. So the
            //  second face sharing the edge has that edge in the opposite
            //  direction in its boundary relative to the first face --
            //  making it necessary to reverse the order of shared points
            //  for the boundary of the second face.
            //
            //  To support a non-manifold edge, all subsequent faces that
            //  share the assigned shared edge must determine if their
            //  orientation of that edge is reversed relative to the first
            //  face for which the shared edge points were evaluated. So a
            //  little more book-keeping and/or inspection is required.
            //
            if (edgeRate > 1) {
                int pointsPerEdge = edgeRate - 1;

                SharedEdge & sharedEdge = sharedEdges[edgeIndex];
                if (!sharedEdge.IsSet()) {
                    //  Identify indices of the new shared points in both the
                    //  mesh and face and increment their inventory:
                    int nextInMesh = numMeshPointsEvaluated;
                    int nextInFace = numFacePointsEvaluated;

                    numFacePointsEvaluated += pointsPerEdge;
                    numMeshPointsEvaluated += pointsPerEdge;

                    sharedEdge.Set(nextInMesh, pointsPerEdge);

                    //  Evaluate shared points and assign indices to boundary:
                    float const * uv = &tessBoundaryCoords[boundaryIndex*2];

                    for (int j = 0; j < pointsPerEdge; ++j, uv += 2) {
                        int pIndex = (nextInFace++) * pointSize;
                        faceSurface.Evaluate(uv, patchPointData, pointSize,
                            &outPos[pIndex], &outDu[pIndex], &outDv[pIndex]);

                        tessBoundaryIndices[boundaryIndex++] = nextInMesh++;
                    }
                } else {
                    //  See note above on simplification for manifold edges
                    assert(!baseLevel.IsEdgeNonManifold(edgeIndex));

                    //  Assign shared points to boundary in reverse order:
                    int nextInMesh = sharedEdge.pointIndex + pointsPerEdge - 1;
                    for (int j = 0; j < pointsPerEdge; ++j) {
                        tessBoundaryIndices[boundaryIndex++] = nextInMesh--;
                    }
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
                int pIndex = i * pointSize;
                faceSurface.Evaluate(uv, patchPointData, pointSize,
                         &outPos[pIndex], &outDu[pIndex], &outDv[pIndex]);
            }
            numFacePointsEvaluated += numInteriorCoords;
            numMeshPointsEvaluated += numInteriorCoords;
        }

        //
        //  Remember to trim/resize the arrays storing evaluation results
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
        int tessInteriorOffset = numMeshPointsEvaluated - numOutCoords;

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
