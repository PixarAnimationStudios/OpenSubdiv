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

#include <opensubdiv/far/primvarRefiner.h>
#include <opensubdiv/far/stencilTableFactory.h>
#include <opensubdiv/far/patchTableFactory.h>
#include "../../regression/common/far_utils.h"
// XXX: revisit the directory structure for examples/tests
#include "../../examples/common/stopwatch.h"

#include "init_shapes.h"

//------------------------------------------------------------------------------
static void
doPerf(const Shape *shape, int maxlevel, int endCapType)
{
    using namespace OpenSubdiv;

    Sdc::SchemeType type = OpenSubdiv::Sdc::SCHEME_CATMARK;

    Sdc::Options sdcOptions;
    sdcOptions.SetVtxBoundaryInterpolation(Sdc::Options::VTX_BOUNDARY_EDGE_ONLY);

    Stopwatch s;

    // ----------------------------------------------------------------------
    // Instantiate a FarTopologyRefiner from the descriptor and refine
    s.Start();
    Far::TopologyRefiner * refiner = Far::TopologyRefinerFactory<Shape>::Create(
        *shape, Far::TopologyRefinerFactory<Shape>::Options(type, sdcOptions));
    {
        Far::TopologyRefiner::AdaptiveOptions options(maxlevel);
        refiner->RefineAdaptive(options);
    }

    s.Stop();
    double timeRefine = s.GetElapsed();

    // ----------------------------------------------------------------------
    // Create stencil table
    s.Start();
    Far::StencilTable const * vertexStencils = NULL;
    {
        Far::StencilTableFactory::Options options;
        vertexStencils = Far::StencilTableFactory::Create(*refiner, options);
    }
    s.Stop();
    double timeCreateStencil = s.GetElapsed();

    // ----------------------------------------------------------------------
    // Create patch table
    s.Start();
    Far::PatchTable const * patchTable = NULL;
    {
        Far::PatchTableFactory::Options poptions(maxlevel);
        poptions.SetEndCapType((Far::PatchTableFactory::Options::EndCapType)endCapType);
        patchTable = Far::PatchTableFactory::Create(*refiner, poptions);
    }

    s.Stop();
    double timeCreatePatch = s.GetElapsed();

    // ----------------------------------------------------------------------
    // append local points to stencils
    s.Start();
    {
        if (Far::StencilTable const *vertexStencilsWithLocalPoints =
            Far::StencilTableFactory::AppendLocalPointStencilTable(
                *refiner, vertexStencils,
                patchTable->GetLocalPointStencilTable())) {
            delete vertexStencils;
            vertexStencils = vertexStencilsWithLocalPoints;
        }
    }
    s.Stop();
    double timeAppendStencil = s.GetElapsed();

    // ---------------------------------------------------------------------
    double timeTotal = s.GetTotalElapsed();

    printf("TopologyRefiner::Refine     %f %5.2f%%\n",
           timeRefine, timeRefine/timeTotal*100);
    printf("StencilTableFactory::Create %f %5.2f%%\n",
           timeCreateStencil, timeCreateStencil/timeTotal*100);
    printf("PatchTableFactory::Create   %f %5.2f%%\n",
           timeCreatePatch, timeCreatePatch/timeTotal*100);
    printf("StencilTableFactory::Append %f %5.2f%%\n",
           timeAppendStencil, timeAppendStencil/timeTotal*100);
    printf("Total                       %f\n", timeTotal);
}

//------------------------------------------------------------------------------
int main(int argc, char **argv)
{
    using namespace OpenSubdiv;

    int maxlevel = 8;
    std::string str;
    int endCapType = Far::PatchTableFactory::Options::ENDCAP_GREGORY_BASIS;

    for (int i = 1; i < argc; ++i) {
        if (strstr(argv[i], ".obj")) {
            std::ifstream ifs(argv[i]);
            if (ifs) {
                std::stringstream ss;
                ss << ifs.rdbuf();
                ifs.close();
                str = ss.str();
                g_shapes.push_back(ShapeDesc(argv[i], str.c_str(), kCatmark));
            }
        }
        else if (!strcmp(argv[i], "-l")) {
            maxlevel = atoi(argv[++i]);
        }
        else if (!strcmp(argv[i], "-e")) {
            const char *type = argv[++i];
            if (!strcmp(type, "bspline")) {
                endCapType = Far::PatchTableFactory::Options::ENDCAP_BSPLINE_BASIS;
            } else if (!strcmp(type, "gregory")) {
                endCapType = Far::PatchTableFactory::Options::ENDCAP_GREGORY_BASIS;
            } else {
                printf("Unknown endcap type %s\n", type);
                return 1;
            }
        }
    }

    if (g_shapes.empty()) {
        initShapes();
    }

    for (int i = 0; i < (int)g_shapes.size(); ++i) {
        Shape const * shape = Shape::parseObj(
            g_shapes[i].data.c_str(),
            g_shapes[i].scheme,
            g_shapes[i].isLeftHanded);

        for (int lv = 1; lv <= maxlevel; ++lv) {
            printf("---- %s, level %d ----\n", g_shapes[i].name.c_str(), lv);
            doPerf(shape, lv, endCapType);
        }
    }
}

//------------------------------------------------------------------------------
