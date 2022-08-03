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
//      This tutorial illustrates the use of the SurfaceFactory, Surface
//      and Parameterization classes for creating and evaluating the limit
//      surface associated with each base face of a mesh.
//
//      Following the creation of a connected mesh for a shape (using a
//      Far::TopologyRefiner, as illustrated in Far tutorials), an instance
//      of a SurfaceFactory is declared to process its faces.  Each face of
//      the mesh is evaluated and tessellated independently (with a simple
//      triangle fan), with results written out in Obj format for inspection.
//
//      These classes make it simple to evaluate and tessellate all faces
//      (quads, tris or others) while supporting the full set of subdivision
//      options. While a triangle fan may be a trivial tessellation (and so
//      not very useful) later examples using the Tessellation class provide
//      more useful results with the same simplicity.
//

#include <opensubdiv/far/topologyRefiner.h>
#include <opensubdiv/bfr/refinerSurfaceFactory.h>
#include <opensubdiv/bfr/surface.h>

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

public:
    Args(int argc, char * argv[]) :
        inputObjFile(),
        outputObjFile(),
        schemeType(Sdc::SCHEME_CATMARK) {

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
    std::vector<int>   outTriangles;

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
        //  Get the Parameterization of the Surface and use it to identify
        //  coordinates for evaluation -- in this case, at the vertices
        //  and center of the face to create a fan of triangles:
        //
        Bfr::Parameterization faceParam = faceSurface.GetParameterization();

        int faceSize = faceParam.GetFaceSize();

        int numOutCoords = faceSize + 1;
        outCoords.resize(numOutCoords * 2);

        for (int i = 0; i < faceSize; ++i) {
            faceParam.GetVertexCoord(i, &outCoords[i*2]);
        }
        faceParam.GetCenterCoord(&outCoords[faceSize*2]);

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

        //  Populate patch point and output arrays:
        faceSurface.PreparePatchPoints(meshVertexPositions.data(), pointSize,
                                       facePatchPoints.data(), pointSize);

        for (int i = 0, j = 0; i < numOutCoords; ++i, j += pointSize) {
            faceSurface.Evaluate(&outCoords[i*2],
                                 facePatchPoints.data(), pointSize,
                                 &outPos[j], &outDu[j], &outDv[j]);
        }

        //
        //  Identify the faces of the tessellation, i.e. the triangle fan
        //  connecting points at the vertices to the center (last) point:
        //
        //  Note the need to offset vertex indices for the output faces --
        //  using the number of vertices generated prior to this face.
        //
        int objVertexIndexOffset = objWriter.GetNumVertices();

        outTriangles.resize(faceSize * 3);

        int * outTriangle = outTriangles.data();
        for (int i = 0; i < faceSize; ++i, outTriangle += 3) {
            outTriangle[0] = objVertexIndexOffset + i;
            outTriangle[1] = objVertexIndexOffset + (i + 1) % faceSize;
            outTriangle[2] = objVertexIndexOffset + faceSize;
        }

        //
        //  Write the evaluated points and faces connecting them as Obj:
        //
        objWriter.WriteGroupName("baseFace_", faceIndex);

        objWriter.WriteVertexPositions(outPos);
        objWriter.WriteVertexNormals(outDu, outDv);

        objWriter.WriteFaces(outTriangles, 3, true, false);
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
