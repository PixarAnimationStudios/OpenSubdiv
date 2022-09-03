//
//   Copyright 2015 Pixar
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

#include <cassert>
#include <cstdio>
#include <fstream>
#include <sstream>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <opensubdiv/far/primvarRefiner.h>
#include <opensubdiv/far/stencilTableFactory.h>
#include <opensubdiv/far/patchTableFactory.h>

#include "../../regression/common/far_utils.h"
#include "../../examples/common/stopwatch.h"

#include "init_shapes.h"

//------------------------------------------------------------------------------

using namespace OpenSubdiv;

struct TestOptions {
    TestOptions() :
        refineLevel(2),
        refineAdaptive(true),
        createPatches(true),
        createStencils(true),
        endCapType(Far::PatchTableFactory::Options::ENDCAP_GREGORY_BASIS) { }

    int  refineLevel;
    bool refineAdaptive;
    bool createPatches;
    bool createStencils;

    Far::PatchTableFactory::Options::EndCapType endCapType;
};

struct TestResult {
    TestResult() :
        level(-1),
        timeTotal(0),
        timeRefine(0),
        timePatchFactory(0),
        timeStencilFactory(0),
        timeAppendStencil(0) { }

    std::string name;
    int level;
    double timeTotal;
    double timeRefine;
    double timePatchFactory;
    double timeStencilFactory;
    double timeAppendStencil;
};

template <typename REAL>
static TestResult
RunPerfTest(Shape const & shape, TestOptions const & options) {

    typedef Far::StencilTableReal<REAL>        FarStencilTable;
    typedef Far::StencilTableFactoryReal<REAL> FarStencilTableFactory;

    Sdc::SchemeType sdcType = GetSdcType(shape);
    Sdc::Options sdcOptions = GetSdcOptions(shape);

    TestResult result;
    result.level = options.refineLevel;

    Stopwatch s; 

    // ----------------------------------------------------------------------
    // Configure the patch table factory options
    Far::PatchTableFactory::Options poptions(options.refineLevel);
    poptions.SetEndCapType(options.endCapType);
    poptions.SetPatchPrecision<REAL>();

    // ----------------------------------------------------------------------
    // Instantiate a FarTopologyRefiner from the descriptor and refine
    Far::TopologyRefiner * refiner = Far::TopologyRefinerFactory<Shape>::Create(
        shape, Far::TopologyRefinerFactory<Shape>::Options(sdcType, sdcOptions));
    assert(refiner);

    s.Start();
    if (options.refineAdaptive) {
        Far::TopologyRefiner::AdaptiveOptions rOptions =
            poptions.GetRefineAdaptiveOptions();
        refiner->RefineAdaptive(rOptions);
    } else {
        Far::TopologyRefiner::UniformOptions rOptions(options.refineLevel);
        refiner->RefineUniform(rOptions);
    }
    s.Stop();
    result.timeRefine = s.GetElapsed();

    // ----------------------------------------------------------------------
    // Create patch table
    Far::PatchTable const * patchTable = NULL;
    if (options.createPatches) {
        s.Start();
        patchTable = Far::PatchTableFactory::Create(*refiner, poptions);
        s.Stop();
        result.timePatchFactory = s.GetElapsed();
    } else {
        result.timePatchFactory = 0;
    }

    // ----------------------------------------------------------------------
    // Create stencil table
    FarStencilTable const * vertexStencils = NULL;
    if (options.createStencils) {
        s.Start();
        vertexStencils = FarStencilTableFactory::Create(*refiner);
        s.Stop();
        result.timeStencilFactory = s.GetElapsed();
    } else {
        result.timeStencilFactory = 0;
    }

    // ----------------------------------------------------------------------
    // append local points to stencils
    if (options.createPatches && options.createStencils) {
        s.Start();

        if (FarStencilTable const *vertexStencilsWithLocalPoints =
            FarStencilTableFactory::AppendLocalPointStencilTable(
                *refiner, vertexStencils,
                patchTable->GetLocalPointStencilTable<REAL>())) {
            delete vertexStencils;
            vertexStencils = vertexStencilsWithLocalPoints;
        }

        s.Stop();
        result.timeAppendStencil = s.GetElapsed();
    } else {
        result.timeAppendStencil = 0;
    }

    // ---------------------------------------------------------------------
    result.timeTotal = s.GetTotalElapsed();

    delete vertexStencils;
    delete patchTable;
    delete refiner;

    return result;
}

//------------------------------------------------------------------------------

struct PrintOptions {
    PrintOptions() :
        csvFormat(false),
        refineTime(true),
        patchTime(true),
        stencilTime(true),
        appendTime(true),
        totalTime(true) { }

    bool csvFormat;
    bool refineTime;
    bool patchTime;
    bool stencilTime;
    bool appendTime;
    bool totalTime;
};

static void
PrintShape(ShapeDesc const & shapeDesc, PrintOptions const & ) {

    static char const * g_schemeNames[3] = { "bilinear", "catmark", "loop" };

    char const * shapeName   = shapeDesc.name.c_str();
    Scheme       shapeScheme = shapeDesc.scheme;

    printf("%s (%s):\n", shapeName, g_schemeNames[shapeScheme]);
}

static void
PrintResult(TestResult const & result, PrintOptions const & options) {

    //  If only printing the total, combine on same line as level:
    if (!options.refineTime  && !options.patchTime &&
        !options.stencilTime && !options.appendTime) {
        printf("  level %d:  %f\n", result.level, result.timeTotal);
        return;
    }

    printf("  level %d:\n", result.level);

    if (options.refineTime) {
        printf("    TopologyRefiner::Refine     %f %5.2f%%\n",
               result.timeRefine,
               result.timeRefine/result.timeTotal*100);
    }
    if (options.patchTime) {
        printf("    PatchTableFactory::Create   %f %5.2f%%\n",
               result.timePatchFactory,
               result.timePatchFactory/result.timeTotal*100);
    }
    if (options.stencilTime) {
        printf("    StencilTableFactory::Create %f %5.2f%%\n",
               result.timeStencilFactory,
               result.timeStencilFactory/result.timeTotal*100);
    }
    if (options.appendTime) {
        printf("    StencilTableFactory::Append %f %5.2f%%\n",
               result.timeAppendStencil,
               result.timeAppendStencil/result.timeTotal*100);
    }
    if (options.totalTime) {
        printf("    Total                       %f\n",
               result.timeTotal);
    }
}

static void
PrintHeaderCSV(PrintOptions const & options) {

    // spreadsheet header row
    printf("shape");
    printf(",level");
    if (options.refineTime)  printf(",refine");
    if (options.patchTime)   printf(",patch");
    if (options.stencilTime) printf(",stencilFactory");
    if (options.appendTime)  printf(",stencilAppend");
    if (options.totalTime)   printf(",total");
    printf("\n");
}
static void
PrintResultCSV(TestResult const & result, PrintOptions const & options) {

    // spreadsheet data row
    printf("%s",  result.name.c_str());
    printf(",%d", result.level);
    if (options.refineTime)  printf(",%f", result.timeRefine);
    if (options.patchTime)   printf(",%f", result.timePatchFactory);
    if (options.stencilTime) printf(",%f", result.timeStencilFactory);
    if (options.appendTime)  printf(",%f", result.timeAppendStencil);
    if (options.totalTime)   printf(",%f", result.timeTotal);
    printf("\n");
}

//------------------------------------------------------------------------------

static int
parseIntArg(char const * argString, int dfltValue = 0) {
    char *argEndptr;
    int argValue = (int) strtol(argString, &argEndptr, 10);
    if (*argEndptr != 0) {
        fprintf(stderr,
                "Warning: non-integer option parameter '%s' ignored\n",
                argString);
        argValue = dfltValue;
    }
    return argValue;
}

int main(int argc, char **argv)
{
    TestOptions testOptions;
    PrintOptions printOptions;
    std::vector<std::string> objFiles;
    Scheme defaultScheme = kCatmark;
    int minLevel = 1;
    int maxLevel = 2;
    bool runDouble = false;

    for (int i = 1; i < argc; ++i) {
        if (strstr(argv[i], ".obj")) {
            objFiles.push_back(std::string(argv[i]));
        } else if (!strcmp(argv[i], "-a")) {
            testOptions.refineAdaptive = true;
        } else if (!strcmp(argv[i], "-u")) {
            testOptions.refineAdaptive = false;
        } else if (!strcmp(argv[i], "-l")) {
            if (++i < argc) maxLevel = parseIntArg(argv[i], maxLevel);
        } else if (!strcmp(argv[i], "-bilinear")) {
            defaultScheme = kBilinear;
        } else if (!strcmp(argv[i], "-catmark")) {
            defaultScheme = kCatmark;
        } else if (!strcmp(argv[i], "-loop")) {
            defaultScheme = kLoop;
        } else if (!strcmp(argv[i], "-e")) {
            char const * type = argv[++i];
            if (!strcmp(type, "linear")) {
                testOptions.endCapType =
                        Far::PatchTableFactory::Options::ENDCAP_BILINEAR_BASIS;
            } else if (!strcmp(type, "regular")) {
                testOptions.endCapType =
                        Far::PatchTableFactory::Options::ENDCAP_BSPLINE_BASIS;
            } else if (!strcmp(type, "gregory")) {
                testOptions.endCapType =
                        Far::PatchTableFactory::Options::ENDCAP_GREGORY_BASIS;
            } else {
                fprintf(stderr, "Error: Unknown endcap type %s\n", type);
                return 1;
            }
        } else if (!strcmp(argv[i], "-double")) {
            runDouble = true;
        } else if (!strcmp(argv[i], "-nopatches")) {
            testOptions.createPatches = false;

            printOptions.patchTime  = false;
            printOptions.appendTime = false;
        } else if (!strcmp(argv[i], "-nostencils")) {
            testOptions.createStencils = false;

            printOptions.stencilTime = false;
            printOptions.appendTime  = false;
        } else if (!strcmp(argv[i], "-total")) {
            printOptions.refineTime  = false;
            printOptions.patchTime   = false;
            printOptions.stencilTime = false;
            printOptions.appendTime  = false;
        } else if (!strcmp(argv[i], "-csv")) {
            printOptions.csvFormat = true;
        } else {
            fprintf(stderr,
                "Warning: unrecognized argument '%s' ignored\n", argv[i]);
        }
    }
    assert(minLevel <= maxLevel);

    if (!objFiles.empty()) {
        for (size_t i = 0; i < objFiles.size(); ++i) {
            char const * objFile = objFiles[i].c_str();
            std::ifstream ifs(objFile);
            if (ifs) {
                std::stringstream ss;
                ss << ifs.rdbuf();
                ifs.close();
                g_shapes.push_back(ShapeDesc(objFile, ss.str(), defaultScheme));
            } else {
                fprintf(stderr,
                    "Warning: cannot open shape file '%s'\n", objFile);
            }
        }
    }

    if (g_shapes.empty()) {
        initShapes();
    }

    //  For each shape, run tests for all specified levels -- printing the
    //  results in the specified format:
    //
    if (printOptions.csvFormat) {
        PrintHeaderCSV(printOptions);
    }
    for (size_t i = 0; i < g_shapes.size(); ++i) {
        ShapeDesc const & shapeDesc = g_shapes[i];
        Shape const * shape = Shape::parseObj(shapeDesc);

        if (!printOptions.csvFormat) {
            PrintShape(shapeDesc, printOptions);
        }

        for (int levelIndex = minLevel; levelIndex <= maxLevel; ++levelIndex) {
            testOptions.refineLevel = levelIndex;

            TestResult result;
            if (runDouble) {
                result = RunPerfTest<double>(*shape, testOptions);
            } else {
                result = RunPerfTest<float>(*shape, testOptions);
            }
            result.name = shapeDesc.name;

            if (printOptions.csvFormat) {
                PrintResultCSV(result, printOptions);
            } else {
                PrintResult(result, printOptions);
            }
        }
        delete shape;
    }
}

//------------------------------------------------------------------------------
