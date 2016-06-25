//
//   Copyright 2013 Pixar
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

#include "far_utils.h"

struct FVarVertex {

    float u,v;

    void Clear() {
        u=v=0.0f;
    }

    void AddWithWeight(FVarVertex const & src, float weight) {
        u += weight * src.u;
        v += weight * src.v;
    }
};

void
InterpolateFVarData(OpenSubdiv::Far::TopologyRefiner & refiner,
    Shape const & shape, std::vector<float> & fvarData) {

    int channel = 0,    // shapes only have 1 UV channel
        fvarWidth = 2;

    int maxlevel = refiner.GetMaxLevel(),
        numValuesM = refiner.GetLevel(maxlevel).GetNumFVarValues(channel),
        numValuesTotal = refiner.GetNumFVarValuesTotal(channel);

    if (shape.uvs.empty() || numValuesTotal<=0) {
        return;
    }

    OpenSubdiv::Far::PrimvarRefiner primvarRefiner(refiner);

    if (refiner.IsUniform()) {

        // For uniform we only keep the highest level of refinement:
        fvarData.resize(numValuesM * fvarWidth);

        std::vector<FVarVertex> buffer(numValuesTotal - numValuesM);

        FVarVertex * src = &buffer[0];
        memcpy(src, &shape.uvs[0], shape.uvs.size()*sizeof(float));

        //  Defer the last level to treat separately with its alternate destination:
        for (int level = 1; level < maxlevel; ++level) {
            FVarVertex * dst = src + refiner.GetLevel(level-1).GetNumFVarValues(channel);

            primvarRefiner.InterpolateFaceVarying(level, src, dst, channel);

            src = dst;
        }

        FVarVertex * dst = reinterpret_cast<FVarVertex *>(&fvarData[0]);
        primvarRefiner.InterpolateFaceVarying(maxlevel, src, dst, channel);

    } else {

        // For adaptive we keep all levels:
        fvarData.resize(numValuesTotal * fvarWidth);

        FVarVertex * src = reinterpret_cast<FVarVertex *>(&fvarData[0]);
        memcpy(src, &shape.uvs[0], shape.uvs.size()*sizeof(float));

        for (int level = 1; level <= maxlevel; ++level) {
            FVarVertex * dst = src + refiner.GetLevel(level-1).GetNumFVarValues(channel);

            primvarRefiner.InterpolateFaceVarying(level, src, dst, channel);

            src = dst;
        }
    }
}
