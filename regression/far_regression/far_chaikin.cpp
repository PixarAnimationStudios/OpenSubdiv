//
//   Copyright 2026 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#include "../../opensubdiv/far/topologyDescriptor.h"
#include "../../opensubdiv/far/primvarRefiner.h"
#include "../../opensubdiv/far/stencilTableFactory.h"
#include "../../opensubdiv/far/stencilTable.h"

#include <cstdio>
#include <memory>
#include <vector>

using namespace OpenSubdiv;

struct Vertex {
    double value = 0.0;

    void Clear() { value = 0.0; }
    void AddWithWeight(Vertex const & src, double weight) {
        value += src.value * weight;
    }
};

static int
checkEdgeMask(Sdc::SchemeType scheme, Sdc::Options::CreasingMethod method,
              bool adaptive, float sharpness, double expected) {

    // A periodic 4x4 grid avoids boundary rules, with triangles for Loop.
    std::vector<int> sizes, corners;
    for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 4; ++x) {
            int a = y * 4 + x,
                b = y * 4 + (x + 1) % 4,
                c = ((y + 1) % 4) * 4 + (x + 1) % 4,
                d = ((y + 1) % 4) * 4 + x;
            if (scheme == Sdc::SCHEME_CATMARK) {
                sizes.push_back(4);
                corners.insert(corners.end(), {a, b, c, d});
            } else {
                sizes.insert(sizes.end(), {3, 3});
                corners.insert(corners.end(), {a, b, c, a, c, d});
            }
        }
    }
    int creases[] = {0, 1, 0, 4};
    float weights[] = {sharpness, 0.375f};
    Far::TopologyDescriptor desc;
    desc.numVertices = 16;
    desc.numFaces = (int)sizes.size();
    desc.numVertsPerFace = sizes.data();
    desc.vertIndicesPerFace = corners.data();
    desc.numCreases = 2;
    desc.creaseVertexIndexPairs = creases;
    desc.creaseWeights = weights;

    Sdc::Options options;
    options.SetCreasingMethod(method);
    typedef Far::TopologyRefinerFactory<Far::TopologyDescriptor> Factory;
    std::unique_ptr<Far::TopologyRefiner> refiner(
        Factory::Create(desc, Factory::Options(scheme, options)));
    if (!refiner || !refiner->GetLevel(0).ValidateTopology()) return 1;

    if (adaptive) {
        refiner->RefineAdaptive(Far::TopologyRefiner::AdaptiveOptions(1));
    } else {
        refiner->RefineUniform(Far::TopologyRefiner::UniformOptions(1));
    }

    Far::TopologyLevel const & base = refiner->GetLevel(0);
    int midpoint = base.GetEdgeChildVertex(base.FindEdge(0, 1));
    std::vector<Vertex> input(16), refined(refiner->GetLevel(1).GetNumVertices());
    input[5].value = 1.0;
    Far::PrimvarRefinerReal<double>(*refiner).Interpolate(1, input, refined);

    std::unique_ptr<Far::StencilTableReal<double> const> stencils(
        Far::StencilTableFactoryReal<double>::Create(*refiner));
    std::vector<Vertex> evaluated(stencils->GetNumStencils());
    stencils->UpdateValues(input, evaluated);

    // These dyadic values are exact. Sharpness >= 1 must give the midpoint,
    // even when Chaikin sharpness decays to zero on only one child edge.
    if ((refined[midpoint].value != expected) ||
        (evaluated[midpoint].value != expected)) {
        printf("Edge mask failed: scheme %d, method %d, adaptive %d, sharpness %g: "
               "expected %g, refined %g, stencil %g\n",
               scheme, method, adaptive, sharpness, expected,
               refined[midpoint].value, evaluated[midpoint].value);
        return 1;
    }
    return 0;
}

int main() {
    int failures = 0;
    for (Sdc::SchemeType scheme : {Sdc::SCHEME_CATMARK, Sdc::SCHEME_LOOP}) {
        for (Sdc::Options::CreasingMethod method :
                {Sdc::Options::CREASE_UNIFORM, Sdc::Options::CREASE_CHAIKIN}) {
            for (bool adaptive : {false, true}) {
                double fractional = (scheme == Sdc::SCHEME_CATMARK) ? 1.0/64.0 : 1.0/32.0;
                failures += checkEdgeMask(scheme, method, adaptive, 0.75f, fractional);
                failures += checkEdgeMask(scheme, method, adaptive, 1.0f, 0.0);
                failures += checkEdgeMask(scheme, method, adaptive, 1.125f, 0.0);
            }
        }
    }
    if (failures == 0) printf("All Chaikin edge mask tests passed.\n");
    return failures ? 1 : 0;
}
