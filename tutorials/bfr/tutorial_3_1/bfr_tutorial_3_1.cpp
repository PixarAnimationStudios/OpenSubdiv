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
//      This tutorial illustrates the definition of a custom subclass of
//      Bfr::SurfaceFactory -- providing a class with the SurfaceFactory
//      interface adapted to a connected mesh representation.
//
//      The bulk of this code is therefore identical to a previous tutorial
//      (1.3) which illustrates simple use of a Bfr::Surface factory. The
//      only difference here lies in the explicit local definition of the
//      subclass of Bfr::SurfaceFactory for Far::TopologyRefiner -- named
//      CustomSurfaceFactory in this case.
//

#include "./customSurfaceFactory.h"

#include <opensubdiv/far/topologyRefiner.h>
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
    bool            uv2xyzFlag;

public:
    Args(int argc, char * argv[]) :
        inputObjFile(),
        outputObjFile(),
        schemeType(Sdc::SCHEME_CATMARK),
        tessUniformRate(5),
        tessQuadsFlag(false),
        uv2xyzFlag(false) {

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
            } else if (!strcmp(argv[i], "-uv2xyz")) {
                uv2xyzFlag = true;
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
//  The main tessellation function:  given a mesh and vertex positions,
//  tessellate each face -- writing results in Obj format.
//
void
tessellateToObj(Far::TopologyRefiner const & meshTopology,
                std::vector<float>   const & meshVertexPositions,
                std::vector<float>   const & meshFaceVaryingUVs,
                Args                 const & options) {

    //
    //  Use simpler local type names for the Surface and its factory:
    //
    typedef CustomSurfaceFactory SurfaceFactory;
    typedef Bfr::Surface<float>  Surface;

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
    //  First declare any evaluation options when initializing:
    //
    //  When dealing with face-varying data, an identifier is necessary
    //  when constructing Surfaces in order to distinguish the different 
    //  face-varying data channels. To avoid repeatedly specifying that
    //  identifier when only one is present (or of interest), it can be
    //  specified via the Options.
    //
    bool meshHasUVs = (meshTopology.GetNumFVarChannels() > 0);

    SurfaceFactory::Options surfaceOptions;
    if (meshHasUVs) {
        surfaceOptions.SetDefaultFVarID(0);
    }

    SurfaceFactory surfaceFactory(meshTopology, surfaceOptions);

    //
    //  The Surface to be constructed and evaluated for each face -- as
    //  well as the intermediate and output data associated with it -- can
    //  be declared in the scope local to each face. But since dynamic
    //  memory is involved with these variables, it is preferred to declare
    //  them outside that loop to preserve and reuse that dynamic memory.
    //
    Surface posSurface;
    Surface uvSurface;

    std::vector<float> facePatchPoints;

    std::vector<float> outCoords;
    std::vector<float> outPos, outDu, outDv;
    std::vector<float> outUV;
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

    int numFaces = surfaceFactory.GetNumFaces();
    for (int faceIndex = 0; faceIndex < numFaces; ++faceIndex) {
        //
        //  Initialize the Surfaces for position and UVs of this face.
        //  There are two ways to do this -- both illustrated here:
        //
        //  Creating Surfaces for the different data interpolation types
        //  independently is clear and convenient, but considerable work
        //  may be duplicated in the construction process in the case of
        //  non-linear face-varying Surfaces. So unless it is known that
        //  face-varying interpolation is linear, use of InitSurfaces()
        //  is generally preferred.
        //
        //  Remember also that the face-varying identifier is omitted from
        //  the initialization methods here as it was previously assigned
        //  to the SurfaceFactory::Options. In the absence of an assignment
        //  of the default FVarID to the Options, a failure to specify the
        //  FVarID here will result in failure.
        //
        //  The cases below are expanded for illustration purposes, and
        //  validity of the resulting Surface is tested here, rather than
        //  the return value of initialization methods.
        //
        bool createSurfacesTogether = true;
        if (!meshHasUVs) {
            surfaceFactory.InitVertexSurface(faceIndex, &posSurface);
        } else if (createSurfacesTogether) {
            surfaceFactory.InitSurfaces(faceIndex, &posSurface, &uvSurface);
        } else {
            if (surfaceFactory.InitVertexSurface(faceIndex, &posSurface)) {
                surfaceFactory.InitFaceVaryingSurface(faceIndex, &uvSurface);
            }
        }
        if (!posSurface.IsValid()) continue;

        //
        //  Declare a simple uniform Tessellation for the Parameterization
        //  of this face and identify coordinates of the points to evaluate:
        //
        Bfr::Tessellation tessPattern(posSurface.GetParameterization(),
                                      options.tessUniformRate, tessOptions);

        int numOutCoords = tessPattern.GetNumCoords();

        outCoords.resize(numOutCoords * 2);

        tessPattern.GetCoords(outCoords.data());

        //
        //  Prepare the patch points for the Surface, then use them to
        //  evaluate output points for all identified coordinates:
        //
        //  Evaluate vertex positions:
        {
            //  Resize patch point and output arrays:
            int pointSize = 3;

            facePatchPoints.resize(posSurface.GetNumPatchPoints() * pointSize);

            outPos.resize(numOutCoords * pointSize);
            outDu.resize(numOutCoords * pointSize);
            outDv.resize(numOutCoords * pointSize);

            //  Populate patch point and output arrays:
            posSurface.PreparePatchPoints(meshVertexPositions.data(), pointSize,
                                          facePatchPoints.data(), pointSize);

            for (int i = 0, j = 0; i < numOutCoords; ++i, j += pointSize) {
                posSurface.Evaluate(&outCoords[i*2],
                                    facePatchPoints.data(), pointSize,
                                    &outPos[j], &outDu[j], &outDv[j]);
            }
        }

        //  Evaluate face-varying UVs (when present):
        if (meshHasUVs) {
            //  Resize patch point and output arrays:
            //      - note reuse of the same patch point array as position
            int pointSize = 2;

            facePatchPoints.resize(uvSurface.GetNumPatchPoints() * pointSize);

            outUV.resize(numOutCoords * pointSize);

            //  Populate patch point and output arrays:
            uvSurface.PreparePatchPoints(meshFaceVaryingUVs.data(), pointSize,
                                         facePatchPoints.data(), pointSize);

            for (int i = 0, j = 0; i < numOutCoords; ++i, j += pointSize) {
                uvSurface.Evaluate(&outCoords[i*2],
                                   facePatchPoints.data(), pointSize,
                                   &outUV[j]);
            }
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

        if (meshHasUVs && options.uv2xyzFlag) {
            objWriter.WriteVertexPositions(outUV, 2);
            objWriter.WriteFaces(outFacets, tessFacetSize, false, false);
        } else {
            objWriter.WriteVertexPositions(outPos);
            objWriter.WriteVertexNormals(outDu, outDv);
            if (meshHasUVs) {
                objWriter.WriteVertexUVs(outUV);
            }
            objWriter.WriteFaces(outFacets, tessFacetSize, true, meshHasUVs);
        }
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

    tessellateToObj(*meshTopology, meshVtxPositions, meshFVarUVs, args);

    delete meshTopology;
    return EXIT_SUCCESS;
}

//------------------------------------------------------------------------------
