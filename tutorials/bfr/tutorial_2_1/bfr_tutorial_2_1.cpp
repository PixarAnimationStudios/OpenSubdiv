//
//   Copyright 2022 Pixar
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
//      This tutorial builds on the previous tutorial that makes use of the
//      SurfaceFactory, Surface and Tessellation classes by illustrating the
//      use of non-uniform tessellation parameters with Tessellation.
//
//      Tessellation rates for the edges of a face are determined by a
//      length associated with each edge. That length may be computed using
//      either the control hull or the limit surface. The length of a
//      tessellation interval is required and will be inferred if not
//      explicitly specified (as a command line option).
//
//      The tessellation rate for an edge is computed as its length divided
//      by the length of the tessellation interval. A maximum tessellation
//      rate is imposed to prevent accidental unbounded tessellation, but
//      can easily be raised as needed.
//

#include <opensubdiv/far/topologyRefiner.h>
#include <opensubdiv/bfr/refinerSurfaceFactory.h>
#include <opensubdiv/bfr/surface.h>
#include <opensubdiv/bfr/tessellation.h>

#include <vector>
#include <string>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <algorithm>

//  Local headers with support for this tutorial in "namespace tutorial"
#include "./meshLoader.h"
#include "./objWriter.h"

using namespace OpenSubdiv;

//
//  Simple command line arguments to provide input and run-time options:
//
class Args {
public:
    std::string     inputObjFile;
    std::string     outputObjFile;
    Sdc::SchemeType schemeType;
    float           tessInterval;
    int             tessRateMax;
    bool            useHullFlag;
    bool            tessQuadsFlag;

public:
    Args(int argc, char * argv[]) :
        inputObjFile(),
        outputObjFile(),
        schemeType(Sdc::SCHEME_CATMARK),
        tessInterval(0.0f),
        tessRateMax(10),
        useHullFlag(false),
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
            } else if (!strcmp(argv[i], "-length")) {
                if (++i < argc) tessInterval = (float) atof(argv[i]);
            } else if (!strcmp(argv[i], "-max")) {
                if (++i < argc) tessRateMax = atoi(argv[i]);
            } else if (!strcmp(argv[i], "-hull")) {
                useHullFlag = true;
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
//  Local trivial functions for simple edge length calculations and the
//  determination of associated tessellation rates:
//
inline float
EdgeLength(float const * v0, float const * v1) {

    float dv[3];
    dv[0] = std::abs(v0[0] - v1[0]);
    dv[1] = std::abs(v0[1] - v1[1]);
    dv[2] = std::abs(v0[2] - v1[2]);
    return std::sqrt(dv[0]*dv[0] + dv[1]*dv[1] + dv[2]*dv[2]);
}

float
FindLongestEdge(Far::TopologyRefiner const & mesh,
                std::vector<float>   const & vertPos, int pointSize) {

    float maxLength = 0.0f;

    int numEdges = mesh.GetLevel(0).GetNumEdges();
    for (int i = 0; i < numEdges; ++i) {
        Far::ConstIndexArray edgeVerts = mesh.GetLevel(0).GetEdgeVertices(i);

        float edgeLength = EdgeLength(&vertPos[edgeVerts[0] * pointSize],
                                      &vertPos[edgeVerts[1] * pointSize]);

        maxLength = std::max(maxLength, edgeLength);
    }
    return maxLength;
}

void
GetEdgeTessRates(std::vector<float> const & vertPos, int pointSize,
                 Args               const & options,
                 int                      * edgeRates) {

    int numEdges = (int) vertPos.size() / pointSize;
    for (int i = 0; i < numEdges; ++i) {
        int j = (i + 1) % numEdges;

        float edgeLength = EdgeLength(&vertPos[i * pointSize],
                                      &vertPos[j * pointSize]);

        edgeRates[i] = 1 + (int)(edgeLength / options.tessInterval);
        edgeRates[i] = std::min(edgeRates[i], options.tessRateMax);
    }
}

//
//  The main tessellation function:  given a mesh and vertex positions,
//  tessellate each face -- writing results in Obj format.
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
    std::vector<int>   faceTessRates;

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
    //  Process each face, writing the output of each in Obj format:
    //
    tutorial::ObjWriter objWriter(options.outputObjFile);

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
        //  Prepare the Surface patch points first as it may be evaluated
        //  to determine suitable edge-rates for Tessellation:
        //
        int pointSize = 3;

        facePatchPoints.resize(faceSurface.GetNumPatchPoints() * pointSize);

        faceSurface.PreparePatchPoints(meshVertexPositions.data(), pointSize,
                                       facePatchPoints.data(), pointSize);

        //
        //  For each of the N edges of the face, a tessellation rate is
        //  determined to initialize a non-uniform Tessellation pattern.
        //
        //  Many metrics are possible -- some based on the geometry itself
        //  (size, curvature), others dependent on viewpoint (screen space
        //  size, center of view, etc.) and many more. Simple techniques
        //  are chosen here for illustration and can easily be replaced.
        //
        //  Here two methods are shown using lengths between the corners of
        //  the face -- the first using the vertex positions of the face and
        //  the second using points evaluated at the corners of its limit
        //  surface. Use of the control hull is more efficient (avoiding the
        //  evaluation) but may prove less effective in some cases (though
        //  both estimates have their limitations).
        //
        int N = faceSurface.GetFaceSize();

        //  Use the output array temporarily to hold the N positions:
        outPos.resize(N * pointSize);

        if (options.useHullFlag) {
            Far::ConstIndexArray verts =
                    meshTopology.GetLevel(0).GetFaceVertices(faceIndex);

            for (int i = 0, j = 0; i < N; ++i, j += pointSize) {
                float const * vPos = &meshVertexPositions[verts[i] * pointSize];
                outPos[j  ] = vPos[0];
                outPos[j+1] = vPos[1];
                outPos[j+2] = vPos[2];
            }
        } else {
            Bfr::Parameterization faceParam = faceSurface.GetParameterization();

            for (int i = 0, j = 0; i < N; ++i, j += pointSize) {
                float uv[2];
                faceParam.GetVertexCoord(i, uv);
                faceSurface.Evaluate(uv, facePatchPoints.data(), pointSize,
                                     &outPos[j]);
            }
        }

        faceTessRates.resize(N);
        GetEdgeTessRates(outPos, pointSize, options, faceTessRates.data());

        //
        //  Declare a non-uniform Tessellation using the rates for each
        //  edge and identify coordinates of the points to evaluate:
        //
        //  Additional interior rates can be optionally provided (2 for
        //  quads, 1 for others) but will be inferred in their absence.
        //
        Bfr::Tessellation tessPattern(faceSurface.GetParameterization(),
                                      N, faceTessRates.data(), tessOptions);

        int numOutCoords = tessPattern.GetNumCoords();

        outCoords.resize(numOutCoords * 2);

        tessPattern.GetCoords(outCoords.data());

        //
        //  Resize the output arrays and evaluate:
        //
        outPos.resize(numOutCoords * pointSize);
        outDu.resize(numOutCoords * pointSize);
        outDv.resize(numOutCoords * pointSize);

        for (int i = 0, j = 0; i < numOutCoords; ++i, j += pointSize) {
            faceSurface.Evaluate(&outCoords[i*2],
                                 facePatchPoints.data(), pointSize,
                                 &outPos[j], &outDu[j], &outDv[j]);
        }

        //
        //  Identify the faces of the Tessellation:
        //
        //  Note the need to offset vertex indices for the output faces --
        //  using the number of vertices generated prior to this face. One
        //  of several Tessellation methods to transform the facet indices
        //  simply translates all indices by the desired offset.
        //
        int objVertexIndexOffset = objWriter.GetNumVertices();

        int numFacets = tessPattern.GetNumFacets();
        outFacets.resize(numFacets * tessFacetSize);
        tessPattern.GetFacets(outFacets.data());

        tessPattern.TransformFacetCoordIndices(outFacets.data(),
                                               objVertexIndexOffset);

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

    //
    //  If no interval length was specified, set one by finding the longest
    //  edge of the mesh and dividing it by the maximum tessellation rate:
    //
    if (args.tessInterval <= 0.0f) {
        args.tessInterval = FindLongestEdge(*meshTopology, meshVtxPositions, 3)
                          / (float) args.tessRateMax;
    }

    tessellateToObj(*meshTopology, meshVtxPositions, args);

    delete meshTopology;
    return EXIT_SUCCESS;
}

//------------------------------------------------------------------------------
