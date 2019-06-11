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
// XXX: revisit the directory structure for examples/tests
#include "../../examples/common/stopwatch.h"

#include "init_shapes.h"

//------------------------------------------------------------------------------

using namespace OpenSubdiv;

struct Result {
    std::string name;
    double timeTotal;
    double timeRefine;
    double timePatchFactory;
    double timeStencilFactory;
    double timeAppendStencil;
};

template <typename REAL>
static Result
doPerf(std::string const & name,
       Shape const * shape,
       int level,
       bool adaptive,
       Far::PatchTableFactory::Options::EndCapType endCapType)
{

    typedef Far::StencilTableReal<REAL>        FarStencilTable;
    typedef Far::StencilTableFactoryReal<REAL> FarStencilTableFactory;

    Sdc::SchemeType sdcType = GetSdcType(*shape);
    Sdc::Options sdcOptions = GetSdcOptions(*shape);

    Result result;
    result.name = name;

    Stopwatch s; 

    // ----------------------------------------------------------------------
    // Configure the patch table factory options
    Far::PatchTableFactory::Options poptions(level);
    poptions.SetEndCapType(endCapType);
    poptions.SetPatchPrecision<REAL>();

    // ----------------------------------------------------------------------
    // Instantiate a FarTopologyRefiner from the descriptor and refine
    s.Start();
    Far::TopologyRefiner * refiner = Far::TopologyRefinerFactory<Shape>::Create(
        *shape, Far::TopologyRefinerFactory<Shape>::Options(sdcType, sdcOptions));
    {
        if (adaptive) {
            Far::TopologyRefiner::AdaptiveOptions options =
                poptions.GetRefineAdaptiveOptions();
            refiner->RefineAdaptive(options);
        } else {
            Far::TopologyRefiner::UniformOptions options(level);
            refiner->RefineUniform(options);
        }
    }
    s.Stop();
    result.timeRefine = s.GetElapsed();

    // ----------------------------------------------------------------------
    // Create patch table
    s.Start();
    Far::PatchTable const * patchTable = NULL;
    {
        patchTable = Far::PatchTableFactory::Create(*refiner, poptions);
    }
    s.Stop();
    result.timePatchFactory = s.GetElapsed();

    // ----------------------------------------------------------------------
    // Create stencil table
    s.Start();
    FarStencilTable const * vertexStencils = NULL;
    {
        typename FarStencilTableFactory::Options options;
        vertexStencils = FarStencilTableFactory::Create(*refiner, options);
    }
    s.Stop();
    result.timeStencilFactory = s.GetElapsed();

    // ----------------------------------------------------------------------
    // append local points to stencils
    s.Start();
    {
        if (FarStencilTable const *vertexStencilsWithLocalPoints =
            FarStencilTableFactory::AppendLocalPointStencilTable(
                *refiner, vertexStencils,
                patchTable->GetLocalPointStencilTable<REAL>())) {
            delete vertexStencils;
            vertexStencils = vertexStencilsWithLocalPoints;
        }
    }
    s.Stop();
    result.timeAppendStencil = s.GetElapsed();

    // ---------------------------------------------------------------------
    result.timeTotal = s.GetTotalElapsed();

    return result;
}

//------------------------------------------------------------------------------

static int
parseIntArg(char const * argString, int dfltValue = 0) {
    char *argEndptr;
    int argValue = strtol(argString, &argEndptr, 10);
    if (*argEndptr != 0) {
        printf("Warning: non-integer option parameter '%s' ignored\n",
                           argString);
        argValue = dfltValue;
    }
    return argValue;
}

int main(int argc, char **argv)
{
    bool adaptive = true;
    int level = 8;
    Scheme defaultScheme = kCatmark;
    Far::PatchTableFactory::Options::EndCapType endCapType =
        Far::PatchTableFactory::Options::ENDCAP_GREGORY_BASIS;
    bool runDouble = false;
    bool spreadsheet = false;

    for (int i = 1; i < argc; ++i) {
        if (strstr(argv[i], ".obj")) {
            std::ifstream ifs(argv[i]);
            if (ifs) {
                std::stringstream ss;
                ss << ifs.rdbuf();
                ifs.close();
                g_shapes.push_back(
                        ShapeDesc(argv[i], ss.str(), defaultScheme));
            }
        } else if (!strcmp(argv[i], "-a")) {
            adaptive = true;
        } else if (!strcmp(argv[i], "-u")) {
            adaptive = false;
        } else if (!strcmp(argv[i], "-l")) {
            if (++i < argc) level = parseIntArg(argv[i], 8);
        } else if (!strcmp(argv[i], "-bilinear")) {
            defaultScheme = kBilinear;
        } else if (!strcmp(argv[i], "-catmark")) {
            defaultScheme = kCatmark;
        } else if (!strcmp(argv[i], "-loop")) {
            defaultScheme = kLoop;
        } else if (!strcmp(argv[i], "-e")) {
            char const * type = argv[++i];
            if (!strcmp(type, "linear")) {
                endCapType =
                        Far::PatchTableFactory::Options::ENDCAP_BILINEAR_BASIS;
            } else if (!strcmp(type, "regular")) {
                endCapType =
                        Far::PatchTableFactory::Options::ENDCAP_BSPLINE_BASIS;
            } else if (!strcmp(type, "gregory")) {
                endCapType =
                        Far::PatchTableFactory::Options::ENDCAP_GREGORY_BASIS;
            } else {
                printf("Unknown endcap type %s\n", type);
                return 1;
            }
        } else if (!strcmp(argv[i], "-double")) {
            runDouble = true;
        } else if (!strcmp(argv[i], "-spreadsheet")) {
            spreadsheet = true;
        }
    }

    if (g_shapes.empty()) {
        initShapes();
    }

    std::vector< std::vector<Result> > resultsByLevel(level+1);

    for (int i = 0; i < (int)g_shapes.size(); ++i) {
        std::string const & name = g_shapes[i].name;
        Shape const * shape = Shape::parseObj(g_shapes[i]);

        for (int lv = 1; lv <= level; ++lv) {
            Result result;
            if (runDouble) {
                result = doPerf<double>(name, shape, lv, adaptive, endCapType);
            } else {
                result = doPerf<float>(name, shape, lv, adaptive, endCapType);
            }
            printf("---- %s, level %d ----\n", result.name.c_str(), lv);
            printf("TopologyRefiner::Refine     %f %5.2f%%\n",
                   result.timeRefine,
                   result.timeRefine/result.timeTotal*100);
            printf("StencilTableFactory::Create %f %5.2f%%\n",
                   result.timeStencilFactory,
                   result.timeStencilFactory/result.timeTotal*100);
            printf("PatchTableFactory::Create   %f %5.2f%%\n",
                   result.timePatchFactory,
                   result.timePatchFactory/result.timeTotal*100);
            printf("StencilTableFactory::Append %f %5.2f%%\n",
                   result.timeAppendStencil,
                   result.timeAppendStencil/result.timeTotal*100);
            printf("Total                       %f\n",
                   result.timeTotal);
            if (spreadsheet) {
                resultsByLevel[lv].push_back(result);
            }
        }
    }
    if (spreadsheet) {
        for (int lv=1; lv<(int)resultsByLevel.size(); ++lv) {
            std::vector<Result> const & results = resultsByLevel[lv];
            if (lv == 1) {
                // spreadsheet header row
                printf("level,");
                for (int s=0; s<(int)results.size(); ++s) {
                    Result const & result = results[s];
                    printf("%s total,", result.name.c_str());
                    printf("%s refine,", result.name.c_str());
                    printf("%s patchFactory,", result.name.c_str());
                    printf("%s stencilFactory,", result.name.c_str());
                    printf("%s stencilAppend,", result.name.c_str());
                }
                printf("\n");
            }
            // spreadsheet data row
            printf("%d,", lv);
            for (int s=0; s<(int)results.size(); ++s) {
                Result const & result = results[s];
                printf("%f,", result.timeTotal);
                printf("%f,", result.timeRefine);
                printf("%f,", result.timePatchFactory);
                printf("%f,", result.timeStencilFactory);
                printf("%f,", result.timeAppendStencil);
            }
            printf("\n");
        }
    }
}

//------------------------------------------------------------------------------
