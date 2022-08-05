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

#include "./types.h"
#include "./bfrSurfaceEvaluator.h"
#include "./farPatchEvaluator.h"

#include "../../regression/common/far_utils.h"

#include "init_shapes.h"
#include "init_shapes_all.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>
#include <cstdio>

using namespace OpenSubdiv;
using namespace OpenSubdiv::OPENSUBDIV_VERSION;

//
//  Global set of shapes -- populated by variants of initShapes() that
//  include explicit lists:
//
std::vector<ShapeDesc> g_shapes;


//
//  Command line arguments and their parsing:
//
class Args {
public:
    //  options related to testing and reporting:
    unsigned int posEvaluate : 1;
    unsigned int d1Evaluate : 1;
    unsigned int d2Evaluate : 1;
    unsigned int uvEvaluate : 1;
    unsigned int posIgnore : 1;
    unsigned int d1Ignore : 1;
    unsigned int d2Ignore : 1;
    unsigned int uvIgnore : 1;
    unsigned int printArgs : 1;
    unsigned int printProgress : 1;
    unsigned int printFaceDiffs : 1;
    unsigned int printSummary : 1;
    unsigned int printWarnings : 1;
    unsigned int ptexConvert : 1;

    //  options affecting configuration and execution:
    unsigned int evalByStencils : 1;
    unsigned int doublePrecision : 1;
    unsigned int noCacheFlag : 1;

    //  options affecting the shape of the limit surface:
    int  depthSharp;
    int  depthSmooth;
    int  bndInterp;
    int  uvInterp;

    //  options related to tessellation and comparison:
    int   uniformRes;
    float relTolerance;
    float absTolerance;
    float uvTolerance;

    //  options affecting the list of shapes to be tested:
    int    shapeCount;
    Scheme shapeScheme;
    bool   shapesCat2Loop;
    bool   shapesAll;

    std::vector<ShapeDesc> shapes;

    //  options determining overall success/failure:
    int passCount;

public:
    Args(int argc, char **argv) :
        posEvaluate(true),
        d1Evaluate(false),
        d2Evaluate(false),
        uvEvaluate(false),
        posIgnore(false),
        d1Ignore(false),
        d2Ignore(false),
        uvIgnore(false),
        printArgs(true),
        printProgress(true),
        printFaceDiffs(false),
        printSummary(true),
        printWarnings(true),
        ptexConvert(false),
        evalByStencils(false),
        doublePrecision(false),
        noCacheFlag(false),
        depthSharp(-1),
        depthSmooth(-1),
        bndInterp(-1),
        uvInterp(-1),
        uniformRes(3),
        relTolerance(0.00005f),
        absTolerance(0.0f),
        uvTolerance(0.0001f),
        shapeCount(0),
        shapeScheme(kCatmark),
        shapesCat2Loop(false),
        shapesAll(false),
        shapes(),
        passCount(0) {

        std::string fileString;

        std::vector<std::string> shapeNames;

        for (int i = 1; i < argc; ++i) {
            char * arg = argv[i];

            //  Options related to input .obj files:
            if (strstr(arg, ".obj")) {
                if (readString(arg, fileString)) {
                    //  Use the scheme declared at the time so that multiple
                    //  shape/scheme pairs can be specified
                    shapes.push_back(
                            ShapeDesc(arg, fileString.c_str(), shapeScheme));
                } else {
                    fprintf(stderr,
                            "Error: Unable to open/read .obj file '%s'\n", arg);
                    exit(0);
                }

            //  Options affecting the limit surface shapes:
            } else if (!strcmp(arg, "-l")) {
                if (++i < argc) {
                    int maxLevel = atoi(argv[i]);
                    depthSharp  = maxLevel;
                    depthSmooth = maxLevel;
                }
            } else if (!strcmp(arg, "-lsharp")) {
                if (++i < argc) depthSharp = atoi(argv[i]);
            } else if (!strcmp(arg, "-lsmooth")) {
                if (++i < argc) depthSmooth = atoi(argv[i]);
            } else if (!strcmp(argv[i], "-bint")) {
                if (++i < argc) bndInterp = atoi(argv[i]);
            } else if (!strcmp(argv[i], "-uvint")) {
                if (++i < argc) uvInterp = atoi(argv[i]);

            //  Options affecting what gets evaluated:
            } else if (!strcmp(arg, "-res")) {
                if (++i < argc) uniformRes = atoi(argv[i]);
            } else if (!strcmp(arg, "-pos")) {
                posEvaluate = true;
            } else if (!strcmp(arg, "-nopos")) {
                posEvaluate = false;
            } else if (!strcmp(arg, "-d1")) {
                d1Evaluate = true;
            } else if (!strcmp(arg, "-nod1")) {
                d1Evaluate = false;
            } else if (!strcmp(arg, "-d2")) {
                d2Evaluate = true;
            } else if (!strcmp(arg, "-nod2")) {
                d2Evaluate = false;
            } else if (!strcmp(arg, "-uv")) {
                uvEvaluate = true;
            } else if (!strcmp(arg, "-nouv")) {
                uvEvaluate = false;
            } else if (!strcmp(arg, "-ptex")) {
                ptexConvert = true;
            } else if (!strcmp(arg, "-noptex")) {
                ptexConvert = false;

            //  Options affecting what gets compared and reported:
            } else if (!strcmp(arg, "-skippos")) {
                posIgnore = true;
            } else if (!strcmp(arg, "-skipd1")) {
                d1Ignore = true;
            } else if (!strcmp(arg, "-skipd2")) {
                d2Ignore = true;
            } else if (!strcmp(arg, "-skipuv")) {
                uvIgnore = true;
            } else if (!strcmp(arg, "-faces")) {
                printFaceDiffs = true;

            //  Options affecting comparison tolerances:
            } else if (!strcmp(argv[i], "-reltol")) {
                if (++i < argc) relTolerance = (float)atof(argv[i]);
            } else if (!strcmp(argv[i], "-abstol")) {
                if (++i < argc) absTolerance = (float)atof(argv[i]);
            } else if (!strcmp(argv[i], "-uvtol")) {
                if (++i < argc) uvTolerance = (float)atof(argv[i]);

            //  Options controlling other internal processing:
            } else if (!strcmp(arg, "-stencils")) {
                evalByStencils = true;
            } else if (!strcmp(arg, "-double")) {
                doublePrecision = true;
            } else if (!strcmp(arg, "-nocache")) {
                noCacheFlag = true;

            //  Options affecting the shapes to be included:
            } else if (!strcmp(arg, "-bilinear")) {
                shapeScheme = kBilinear;
            } else if (!strcmp(arg, "-catmark")) {
                shapeScheme = kCatmark;
            } else if (!strcmp(arg, "-loop")) {
                shapeScheme = kLoop;
            } else if (!strcmp(arg, "-cat2loop")) {
                shapesCat2Loop = true;
            } else if (!strcmp(arg, "-count")) {
                if (++i < argc) shapeCount = atoi(argv[i]);
            } else if (!strcmp(arg, "-shape")) {
                if (++i < argc) {
                    shapeNames.push_back(std::string(argv[i]));
                }
            } else if (!strcmp(arg, "-all")) {
                shapesAll = true;

            //  Printing and reporting:
            } else if (!strcmp(arg, "-args")) {
                printArgs = true;
            } else if (!strcmp(arg, "-noargs")) {
                printArgs = false;
            } else if (!strcmp(arg, "-prog")) {
                printProgress = true;
            } else if (!strcmp(arg, "-noprog")) {
                printProgress = false;
            } else if (!strcmp(arg, "-sum")) {
                printSummary = true;
            } else if (!strcmp(arg, "-nosum")) {
                printSummary = false;
            } else if (!strcmp(arg, "-quiet")) {
                printWarnings = false;
            } else if (!strcmp(arg, "-silent")) {
                printArgs     = false;
                printProgress = false;
                printSummary  = false;
                printWarnings = false;

            //  Success/failure of the entire test:
            } else if (!strcmp(argv[i], "-pass")) {
                if (++i < argc) passCount = atoi(argv[i]);

            //  Unrecognized...
            } else {
                fprintf(stderr, "Error: Unrecognized argument '%s'\n", arg);
                exit(0);
            }
        }

        //  Validation -- possible conflicting options, values, etc.
        if (bndInterp > 2) {
            fprintf(stderr, "Warning: Ignoring bad value to -bint (%d)\n",
                    bndInterp);
            bndInterp = -1;
        }
        if (uvInterp > 5) {
            fprintf(stderr, "Warning: Ignoring bad value to -uvint (%d)\n",
                    uvInterp);
            uvInterp = -1;
        }

        if (d2Evaluate) {
            if (!d1Evaluate) {
                fprintf(stderr, "Warning: 2nd deriv evaluation forces 1st.\n");
                d1Evaluate = true;
            }
            if (!posEvaluate) {
                fprintf(stderr, "Warning: 2nd deriv evaluation forces pos.\n");
                posEvaluate = true;
            }
        } else if (d1Evaluate) {
            if (!posEvaluate) {
                fprintf(stderr, "Warning: 1st deriv evaluation forces pos.\n");
                posEvaluate = true;
            }
        }
        if (!posEvaluate && !uvEvaluate) {
            fprintf(stderr, "Error: All pos and UV evaluation disabled.\n");
            exit(0);
        }
        if (posIgnore && d1Ignore && d2Ignore && uvIgnore) {
            fprintf(stderr, "Error: All pos and UV comparisons disabled.\n");
            exit(0);
        }

        if ((depthSmooth == 0) || (depthSharp == 0)) {
            fprintf(stderr,
                "Warning: Far evaluation unstable with refinement level 0.\n");
        }

        //  Managing the list of shapes:
        assert(g_shapes.empty());
        if (!shapeNames.empty()) {
            if (shapesAll) {
                initShapesAll(g_shapes);
            } else {
                initShapes(g_shapes);
            }
            //  Maybe worth building a map -- for this and more...
            for (size_t i = 0; i < shapeNames.size(); ++i) {
                std::string & shapeName = shapeNames[i];
                bool found = false;
                for (size_t j = 0; !found && (j < g_shapes.size()); ++j) {
                    if (g_shapes[j].name == shapeName) {
                        shapes.push_back(g_shapes[j]);
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    fprintf(stderr,
                        "Error: Specified shape '%s' not found.\n",
                        shapeName.c_str());
                    exit(0);
                }
            }
        }
    }
    ~Args() { }

    void
    Print() const {

        char const * boolStrings[2] = { "false", "true" };
        char const * bIntStrings[3] = { "BOUNDARY_NONE",
                                        "BOUNDARY_EDGE_ONLY",
                                        "BOUNDARY_EDGE_AND_CORNER" };
        char const * fvIntStrings[6] = { "LINEAR_NONE",
                                         "LINEAR_CORNERS_ONLY",
                                         "LINEAR_CORNERS_PLUS1",
                                         "LINEAR_CORNERS_PLUS2",
                                         "LINEAR_BOUNDARIES",
                                         "LINEAR_ALL" };

        printf("\n");
        printf("Shape options:\n");
        if (depthSharp >= 0) {
            printf("  - max level sharp  = %d\n",  depthSharp);
        } else {
            printf("  - max level sharp  = %d (dflt)\n",
                (Bfr::SurfaceFactory::Options()).GetApproxLevelSharp());
        }
        if (depthSmooth >= 0) {
            printf("  - max level smooth = %d\n",  depthSmooth);
        } else {
            printf("  - max level smooth = %d (dflt)\n",
                (Bfr::SurfaceFactory::Options()).GetApproxLevelSmooth());
        }
        if (bndInterp < 0) {
            printf("  - boundary interp  = (as assigned)\n");
        } else {
            printf("  - boundary interp  = %s\n", bIntStrings[bndInterp]);
        }
        if (uvEvaluate) {
            if (uvInterp < 0) {
                printf("  - UV linear interp = (as assigned)\n");
            } else {
                printf("  - UV linear interp = %s\n", fvIntStrings[uvInterp]);
            }
        }

        printf("Evaluation options:\n");
        printf("  - tessellation res = %d\n",  uniformRes);
        printf("  - position         = %s\n",  boolStrings[posEvaluate]);
        printf("  - 1st derivative   = %s\n",  boolStrings[d1Evaluate]);
        printf("  - 2nd derivative   = %s\n",  boolStrings[d2Evaluate]);
        printf("  - UV               = %s\n",  boolStrings[uvEvaluate]);

        printf("Comparison options:\n");
        if (absTolerance > 0.0f) {
            printf("  - tolerance (abs)  = %g\n",  absTolerance);
        } else {
            printf("  - tolerance (rel)  = %g\n",  relTolerance);
        }
        if (uvEvaluate) {
            printf("  - tolerance UV     = %g\n",  uvTolerance);
        }
        if (posEvaluate && posIgnore) {
            printf("  - ignore pos       = %s\n",  boolStrings[posIgnore]);
        }
        if (d1Evaluate && d1Ignore) {
            printf("  - ignore 1st deriv = %s\n",  boolStrings[d1Ignore]);
        }
        if (d2Evaluate && d2Ignore) {
            printf("  - ignore 2nd deriv = %s\n",  boolStrings[d2Ignore]);
        }
        if (uvEvaluate && uvIgnore) {
            printf("  - ignore UV        = %s\n",  boolStrings[uvIgnore]);
        }
        printf("\n");
    }

private:
    Args() { }

    bool
    readString(const char *fileName, std::string& fileString) {
        std::ifstream ifs(fileName);
        if (ifs) {
            std::stringstream ss;
            ss << ifs.rdbuf();
            ifs.close();

            fileString = ss.str();
            return true;
        }
        return false;
    }
};


//
//  Create a TopologyRefiner from a Shape:
//
template <typename REAL>
Far::TopologyRefiner *
createTopologyRefiner(ShapeDesc const           & shapeDesc,
                      std::vector< Vec3<REAL> > & shapePos,
                      std::vector< Vec3<REAL> > & shapeUVs,
                      Args const                & args) {

    typedef Vec3<REAL> Vec3Real;

    //
    //  Load the Shape -- skip with a warning on failure:
    //
    Shape * shape = Shape::parseObj(shapeDesc.data.c_str(),
                                    shapeDesc.scheme,
                                    shapeDesc.isLeftHanded);
    if (shape == 0) {
        if (args.printWarnings) {
            fprintf(stderr, "Warning: Failed to parse shape '%s'\n",
                    shapeDesc.name.c_str());
        }
        return 0;
    }

    //  Verify UVs before continuing:
    if (args.uvEvaluate) {
        if (shape->uvs.empty() != shape->faceuvs.empty()) {
            if (args.printWarnings) {
                fprintf(stderr,
                        "Warning: Incomplete UVs assigned to Shape '%s'\n",
                        shapeDesc.name.c_str());
            }
            delete shape;
            return 0;
        }
    }

    //
    //  Create a TopologyRefiner and load position and UVs:
    //
    Sdc::SchemeType sdcType = GetSdcType(*shape);

    if (args.shapesCat2Loop && (sdcType == Sdc::SCHEME_LOOP)) {
        if (args.printWarnings) {
            fprintf(stderr,
                    "\t\tWarning: Applying Catmark to Loop shape '%s'\n",
                    shapeDesc.name.c_str());
        }
        sdcType = Sdc::SCHEME_CATMARK;
    }

    Sdc::Options sdcOptions = GetSdcOptions(*shape);
    if (args.bndInterp >= 0) {
        sdcOptions.SetVtxBoundaryInterpolation(
            (Sdc::Options::VtxBoundaryInterpolation) args.bndInterp);
    }
    if (args.uvInterp >= 0) {
        sdcOptions.SetFVarLinearInterpolation(
            (Sdc::Options::FVarLinearInterpolation) args.uvInterp);
    }

    Far::TopologyRefiner * refiner =
        Far::TopologyRefinerFactory<Shape>::Create(*shape,
        Far::TopologyRefinerFactory<Shape>::Options(sdcType, sdcOptions));

    if (refiner == 0) {
        if (args.printWarnings) {
            fprintf(stderr, "Warning: Unable to interpret Shape '%s'\n",
                    shapeDesc.name.c_str());
        }
        delete shape;
        return 0;
    }

    int numVertices = refiner->GetNumVerticesTotal();
    shapePos.resize(numVertices);
    for (int i = 0; i < numVertices; ++i) {
        shapePos[i] = Vec3Real(shape->verts[i*3],
                               shape->verts[i*3+1],
                               shape->verts[i*3+2]);
    }

    shapeUVs.resize(0);
    if (args.uvEvaluate) {
        if (refiner->GetNumFVarChannels()) {
            int numUVs = refiner->GetNumFVarValuesTotal(0);
            shapeUVs.resize(numUVs);
            for (int i = 0; i < numUVs; ++i) {
                shapeUVs[i] = Vec3Real(shape->uvs[i*2],
                                       shape->uvs[i*2+1],
                                       0.0f);
            }
        }
    }

    delete shape;
    return refiner;
}


//
//  Compute the bounding box of a Vec3 vector and a relative tolerance:
//
template <typename REAL>
REAL
GetRelativeTolerance(std::vector< Vec3<REAL> > const & p, REAL fraction) {

    Vec3<REAL> pMin = p[0];
    Vec3<REAL> pMax = p[0];

    for (size_t i = 1; i < p.size(); ++i) {
        Vec3<REAL> const & pi = p[i];

        pMin[0] = std::min(pMin[0], pi[0]);
        pMin[1] = std::min(pMin[1], pi[1]);
        pMin[2] = std::min(pMin[2], pi[2]);

        pMax[0] = std::max(pMax[0], pi[0]);
        pMax[1] = std::max(pMax[1], pi[1]);
        pMax[2] = std::max(pMax[2], pi[2]);
    }

    Vec3<REAL> pDelta = pMax - pMin;

    REAL maxSize = std::max(std::abs(pDelta[0]), std::abs(pDelta[1]));
    maxSize = std::max(maxSize, std::abs(pDelta[2]));

    return fraction * maxSize;
}


//
//  An independent test from limit surface evaluation:  comparing the
//  conversion of (u,v) coordinates for Bfr::Parameterization to Ptex
//  and back (subject to a given tolerance):
//
template <typename REAL>
void
ValidatePtexConversion(Bfr::Parameterization const & param,
                       REAL const givenCoord[2], REAL tol = 0.0001f) {

    if (!param.HasSubFaces()) return;

    //
    //  Convert the given (u,v) coordinate to Ptex and back and
    //  compare the final result to the original:
    //
    REAL ptexCoord[2];
    REAL finalCoord[2];

    int ptexFace = param.ConvertCoordToNormalizedSubFace(givenCoord, ptexCoord);
    param.ConvertNormalizedSubFaceToCoord(ptexFace, ptexCoord, finalCoord);

    bool subFaceDiff = (ptexFace != param.GetSubFace(givenCoord));
    bool uCoordDiff  = (std::abs(finalCoord[0] - givenCoord[0]) > tol);
    bool vCoordDiff  = (std::abs(finalCoord[1] - givenCoord[1]) > tol);

    if (subFaceDiff || uCoordDiff || vCoordDiff) {
        fprintf(stderr,
                "Warning: Mismatch in sub-face Parameterization conversion:\n");
        if (subFaceDiff ) {
            fprintf(stderr,
                "    converted sub-face (%d) != original (%d)\n",
                ptexFace, param.GetSubFace(givenCoord));
        }
        if (uCoordDiff || vCoordDiff) {
            fprintf(stderr,
                "    converted coord (%f,%f) != original (%f,%f)\n",
                finalCoord[0], finalCoord[1], givenCoord[0], givenCoord[1]);
        }
    }
}


//
//  Compare two meshes using Bfr::Surfaces and a Far::PatchTable:
//
template <typename REAL>
int
testMesh(Far::TopologyRefiner      const & mesh,
         std::string               const & meshName,
         std::vector< Vec3<REAL> > const & meshPos,
         std::vector< Vec3<REAL> > const & meshUVs,
         Args                      const & args) {

    //
    //  Determine what to evaluate/compare based on args and mesh content
    //  (remember that these are not completely independent -- position
    //  evaluation will have been set if evaluating any derivatives):
    //
    bool evalPos = args.posEvaluate;
    bool evalD1  = args.d1Evaluate;
    bool evalD2  = args.d2Evaluate;
    bool evalUV  = args.uvEvaluate && (meshUVs.size() > 0);

    bool comparePos = evalPos && !args.posIgnore;
    bool compareD1  = evalD1  && !args.d1Ignore;
    bool compareD2  = evalD2  && !args.d2Ignore;
    bool compareUV  = evalUV  && !args.uvIgnore;

    //  If nothing to compare, return 0 failures:
    if ((comparePos + compareD1 + compareD2 + compareUV) == 0) {
        return 0;
    }

    //  Declare/allocate output evaluation buffers for both Bfr and Far:
    std::vector<REAL> evalCoords;

    EvalResults<REAL> bfrResults;
    bfrResults.evalPosition = evalPos;
    bfrResults.eval1stDeriv = evalD1;
    bfrResults.eval2ndDeriv = evalD2;
    bfrResults.evalUV       = evalUV;
    bfrResults.useStencils  = args.evalByStencils;

    EvalResults<REAL> farResults;
    farResults.evalPosition = evalPos;
    farResults.eval1stDeriv = evalD1;
    farResults.eval2ndDeriv = evalD2;
    farResults.evalUV       = evalUV;

    //
    //  Create evaluators for Bfr and Far -- using the same set of Bfr
    //  options to ensure consistency (the Far evaluator needs to interpret
    //  them appropriate to Far::PatchTable and associated refinement)
    //
    Bfr::SurfaceFactory::Options surfaceOptions;

    //  Leave approximation defaults in place unless explicitly overridden:
    if (args.depthSharp >= 0) {
        surfaceOptions.SetApproxLevelSharp(args.depthSharp);
    }
    if (args.depthSmooth >= 0) {
        surfaceOptions.SetApproxLevelSmooth(args.depthSmooth);
    }
    surfaceOptions.SetDefaultFVarID(0);
    surfaceOptions.EnableCaching(!args.noCacheFlag);

    BfrSurfaceEvaluator<REAL> bfrEval(mesh, meshPos, meshUVs, surfaceOptions);
    FarPatchEvaluator<REAL>   farEval(mesh, meshPos, meshUVs, surfaceOptions);

    //
    //  Initialize tolerances and variables to track differences:
    //
    REAL pTol  = (args.absTolerance > 0.0f) ? args.absTolerance :
                  GetRelativeTolerance<REAL>(meshPos, args.relTolerance);
    REAL d1Tol = pTol  * 5.0f;
    REAL d2Tol = d1Tol * 5.0f;
    REAL uvTol = args.uvTolerance;

    VectorDelta<REAL> pDelta(pTol);
    VectorDelta<REAL> duDelta(d1Tol);
    VectorDelta<REAL> dvDelta(d1Tol);
    VectorDelta<REAL> duuDelta(d2Tol);
    VectorDelta<REAL> duvDelta(d2Tol);
    VectorDelta<REAL> dvvDelta(d2Tol);
    VectorDelta<REAL> uvDelta(uvTol);

    FaceDelta<REAL> faceDelta;

    MeshDelta<REAL> meshDelta;

    bool meshHasBeenLabeled = false;

    int numFaces = mesh.GetNumFacesTotal();
    for (int faceIndex = 0; faceIndex < numFaces; ++faceIndex) {
        //
        //  Make sure both match in terms of identifying a limit surface:
        //
        assert(bfrEval.FaceHasLimit(faceIndex) ==
               farEval.FaceHasLimit(faceIndex));

        if (!farEval.FaceHasLimit(faceIndex)) continue;

        //
        //  Declare/define a Tessellation to generate a consistent set of
        //  (u,v) locations to compare and evaluate:
        //
        int faceSize = mesh.GetLevel(0).GetFaceVertices(faceIndex).size();

        Bfr::Parameterization faceParam(mesh.GetSchemeType(), faceSize);
        assert(faceParam.IsValid());

        Bfr::Tessellation faceTess(faceParam, args.uniformRes);
        assert(faceTess.IsValid());

        evalCoords.resize(2 * faceTess.GetNumCoords());
        faceTess.GetCoords(&evalCoords[0]);

        //
        //  Before evaluating and comparing, run the test to convert the
        //  parametric coords to Ptex and back:
        //
        if (args.ptexConvert) {
            for (int i = 0; i < faceTess.GetNumCoords(); ++i) {
                ValidatePtexConversion<REAL>(faceParam, &evalCoords[2*i]);
            }
        }

        //
        //  Evaluate and capture results of comparisons between results:
        //
        bfrEval.Evaluate(faceIndex, evalCoords, bfrResults);
        farEval.Evaluate(faceIndex, evalCoords, farResults);

        if (comparePos) {
            pDelta.Compare(bfrResults.p, farResults.p);
        }
        if (compareD1) {
            duDelta.Compare(bfrResults.du, farResults.du);
            dvDelta.Compare(bfrResults.dv, farResults.dv);
        }
        if (compareD2) {
            duuDelta.Compare(bfrResults.duu, farResults.duu);
            duvDelta.Compare(bfrResults.duv, farResults.duv);
            dvvDelta.Compare(bfrResults.dvv, farResults.dvv);
        }
        if (compareUV) {
            uvDelta.Compare(bfrResults.uv, farResults.uv);
        }

        //
        //  Note collective differences for this face and report:
        //
        faceDelta.Clear();
        faceDelta.AddPDelta(pDelta);
        faceDelta.AddDuDelta(duDelta);
        faceDelta.AddDvDelta(dvDelta);
        faceDelta.AddDuuDelta(duuDelta);
        faceDelta.AddDuvDelta(duvDelta);
        faceDelta.AddDvvDelta(dvvDelta);
        faceDelta.AddUVDelta(uvDelta);

        if (args.printFaceDiffs && faceDelta.hasDeltas) {
            if (!meshHasBeenLabeled) {
                meshHasBeenLabeled = true;
                printf("'%s':\n", meshName.c_str());
            }
            printf("\t    Face %d:\n", faceIndex);

            if (comparePos && faceDelta.numPDeltas) {
                printf("\t\t      POS:%6d diffs, max delta P  = %g\n",
                        faceDelta.numPDeltas, (float) faceDelta.maxPDelta);
            }
            if (compareD1 && faceDelta.numD1Deltas) {
                printf("\t\t       D1:%6d diffs, max delta D1 = %g\n",
                        faceDelta.numD1Deltas, (float) faceDelta.maxD1Delta);
            }
            if (compareD2 && faceDelta.numD2Deltas) {
                printf("\t\t       D2:%6d diffs, max delta D2 = %g\n",
                        faceDelta.numD2Deltas, (float) faceDelta.maxD2Delta);
            }
            if (compareUV && faceDelta.hasUVDeltas) {
                printf("\t\t       UV:%6d diffs, max delta UV = %g\n",
                        uvDelta.numDeltas, (float) uvDelta.maxDelta);
            }
        }

        //  Add the results for this face to the collective mesh delta:
        meshDelta.AddFace(faceDelta);
    }

    //
    //  Report the differences for this mesh:
    //
    if (meshDelta.numFacesWithDeltas) {
        if (args.printFaceDiffs) {
            printf("\t    Total:\n");
        } else {
            printf("'%s':\n", meshName.c_str());
        }
    }

    if (comparePos && meshDelta.numFacesWithPDeltas) {
        printf("\t\tPOS diffs:%6d faces, max delta P  = %g\n",
                meshDelta.numFacesWithPDeltas, (float) meshDelta.maxPDelta);
    }
    if (compareD1 && meshDelta.numFacesWithD1Deltas) {
        printf("\t\t D1 diffs:%6d faces, max delta D1 = %g\n",
                meshDelta.numFacesWithD1Deltas, (float) meshDelta.maxD1Delta);
    }
    if (compareD2 && meshDelta.numFacesWithD2Deltas) {
        printf("\t\t D2 diffs:%6d faces, max delta D2 = %g\n",
                meshDelta.numFacesWithD2Deltas, (float) meshDelta.maxD2Delta);
    }
    if (compareUV && meshDelta.numFacesWithUVDeltas) {
        printf("\t\t UV diffs:%6d faces, max delta UV = %g\n",
                meshDelta.numFacesWithUVDeltas, (float) meshDelta.maxUVDelta);
    }
    return meshDelta.numFacesWithDeltas;
}


//
//  Run the comparison for a given Shape in single or double precision:
//
template <typename REAL>
int
testShape(ShapeDesc const & shapeDesc, Args const & args) {

    //
    //  Get the TopologyRefiner, positions and UVs for the Shape, report
    //  failure to generate the shape, and run the test:
    //
    std::string const & meshName = shapeDesc.name;

    std::vector< Vec3<REAL> > basePos;
    std::vector< Vec3<REAL> > baseUV;

    Far::TopologyRefiner * refiner =
            createTopologyRefiner<REAL>(shapeDesc, basePos,  baseUV, args);

    if (refiner == 0) {
        if (args.printWarnings) {
            fprintf(stderr,
                "Warning: Shape '%s' ignored (unable to construct refiner)\n",
                meshName.c_str());
        }
        return -1;
    }

    int nFailures = testMesh<REAL>(*refiner, meshName, basePos, baseUV, args);

    delete refiner;

    return nFailures;
}


//
//  Run comparison tests on a list of shapes using command line options:
//
int
main(int argc, char **argv) {

    Args args(argc, argv);

    //  Capture relevant command line options used here:
    if (args.printArgs) {
        args.Print();
    }

    //
    //  Initialize the list of shapes and test each (or only the first):
    //
    //    - currently the internal list can be overridden on the command
    //      line (so use of wildcards is possible)
    //
    //    - still exploring additional command line options, e.g. hoping
    //      to specify a list of shape names from the internal list...
    //
    //  So a bit more to be done here...
    //
    std::vector<ShapeDesc>& shapeList = g_shapes;

    if (!args.shapes.empty()) {
        shapeList.swap(args.shapes);
    }
    if (shapeList.empty()) {
        if (args.shapesAll) {
            initShapesAll(shapeList);
        } else {
            initShapes(shapeList);
        }
    }

    int shapesToTest  = (int) shapeList.size();
    int shapesIgnored = 0;
    if ((args.shapeCount > 0) && (args.shapeCount < shapesToTest)) {
        shapesIgnored = shapesToTest - args.shapeCount;
        shapesToTest = args.shapeCount;
    }

    if (args.printProgress) {
        printf("Testing %d shapes", shapesToTest);
        if (shapesIgnored) {
            printf(" (%d ignored)", shapesIgnored);
        }
        printf(":\n");
    }

    //
    //  Run the comparison test for each shape (ShapeDesc) in the
    //  specified precision and report results:
    //
    int shapesFailed = 0;

    for (int shapeIndex = 0; shapeIndex < shapesToTest; ++shapeIndex) {
        ShapeDesc  & shapeDesc = shapeList[shapeIndex];

        if (args.printProgress) {
            printf("%4d of %d:  '%s'\n", 1 + shapeIndex, shapesToTest,
                    shapeDesc.name.c_str());
        }

        int nFailures = args.doublePrecision ?
                        testShape<double>(shapeDesc, args) :
                        testShape<float>(shapeDesc, args);

        if (nFailures < 0) {
            //  Possible error/warning...?
            ++ shapesFailed;
        }
        if (nFailures > 0) {
            ++ shapesFailed;
        }
    }

    if (args.printSummary) {
        printf("\n");
        if (shapesFailed == 0) {
            printf("All tests passed for %d shapes\n", shapesToTest);
        } else {
            printf("Total failures: %d of %d shapes\n", shapesFailed,
                                                        shapesToTest);
        }
    }

    return (shapesFailed == args.passCount) ? EXIT_SUCCESS : EXIT_FAILURE;
}
