// Copyright 2014 Google Inc. All rights reserved.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "../osd/neonKernel.h"

#include <cassert>
#include <cstring>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

static inline void clear(float *dst)
{
    memset(dst, 0, 6 * sizeof(float));
}

static inline void addWithWeight(float *dst, const float *srcOrigin,
    int srcIndex, float weight)
{
    const float *src = srcOrigin + srcIndex * 8;
    for (int i = 0; i < 6; ++i)
        dst[i] += src[i] * weight;
}

static inline void copy(float *dstOrigin, const float *src, int dstIndex)
{
    float *dst = dstOrigin + dstIndex * 8;
    memcpy(dst, src, 6 * sizeof(float));
}

void OsdNeonComputeQuadFace(float *vertex, const int *F_IT, int vertexOffset,
    int tableOffset, int batchSize)
{
    float result[6];

    for (int i = 0; i < batchSize; ++i) {
        int fidx0 = F_IT[tableOffset + 4 * i + 0];
        int fidx1 = F_IT[tableOffset + 4 * i + 1];
        int fidx2 = F_IT[tableOffset + 4 * i + 2];
        int fidx3 = F_IT[tableOffset + 4 * i + 3];

        clear(result);
        addWithWeight(result, vertex, fidx0, 0.25f);
        addWithWeight(result, vertex, fidx1, 0.25f);
        addWithWeight(result, vertex, fidx2, 0.25f);
        addWithWeight(result, vertex, fidx3, 0.25f);

        int dstIndex = i + vertexOffset;
        copy(vertex, result, dstIndex);
    }
}

void OsdNeonComputeTriQuadFace(float *vertex, const int *F_IT, int vertexOffset,
    int tableOffset, int batchSize)
{
    float result[6];

    for (int i = 0; i < batchSize; ++i) {
        int fidx0 = F_IT[tableOffset + 4 * i + 0];
        int fidx1 = F_IT[tableOffset + 4 * i + 1];
        int fidx2 = F_IT[tableOffset + 4 * i + 2];
        int fidx3 = F_IT[tableOffset + 4 * i + 3];

        const float fw[2][2] = {
            { 1.0f / 3.0f, 0.0f },
            { 1.0f / 4.0f, 1.0f / 4.0f }
        };

        int widx = (fidx2 == fidx3 ? 0 : 1);

        clear(result);
        addWithWeight(result, vertex, fidx0, fw[widx][0]);
        addWithWeight(result, vertex, fidx1, fw[widx][0]);
        addWithWeight(result, vertex, fidx2, fw[widx][0]);
        addWithWeight(result, vertex, fidx3, fw[widx][1]);

        int dstIndex = i + vertexOffset;
        copy(vertex, result, dstIndex);
    }
}

void OsdNeonComputeRestrictedEdge(float *vertex, const int *E_IT,
    int vertexOffset, int tableOffset, int batchSize)
{
    float result[6];

    for (int i = tableOffset; i < batchSize + tableOffset; ++i) {
        int eidx0 = E_IT[4 * i + 0];
        int eidx1 = E_IT[4 * i + 1];
        int eidx2 = E_IT[4 * i + 2];
        int eidx3 = E_IT[4 * i + 3];

        clear(result);
        addWithWeight(result, vertex, eidx0, 0.25f);
        addWithWeight(result, vertex, eidx1, 0.25f);
        addWithWeight(result, vertex, eidx2, 0.25f);
        addWithWeight(result, vertex, eidx3, 0.25f);

        int dstIndex = i + vertexOffset - tableOffset;
        copy(vertex, result, dstIndex);
    }
}

void OsdNeonComputeRestrictedVertexB1(float *vertex, const int *V_ITa,
    const int *V_IT, int vertexOffset, int tableOffset, int start, int end)
{
    float result[6];

    for (int i = start + tableOffset; i < end + tableOffset; ++i) {
        int h = V_ITa[5 * i];
        int p = V_ITa[5 * i + 2];

        clear(result);
        addWithWeight(result, vertex, p, 0.5f);
        for (int j = 0; j < 8; ++j, ++h)
            addWithWeight(result, vertex, V_IT[h], 0.0625f);

        int dstIndex = i + vertexOffset - tableOffset;
        copy(vertex, result, dstIndex);
    }
}

void OsdNeonComputeRestrictedVertexB2(float *vertex, const int *V_ITa,
    const int *V_IT, int vertexOffset, int tableOffset, int start, int end)
{
    float result[6];

    for (int i = start + tableOffset; i < end + tableOffset; ++i) {
        int h = V_ITa[5 * i];
        int n = V_ITa[5 * i + 1];
        int p = V_ITa[5 * i + 2];
        assert(n >= 3 && n <= 10);

        const float vw[8][2] = {
            { 1.0f - 2.0f / 3.0f, 1.0f / (3.0f * 3.0f) },
            { 1.0f - 2.0f / 4.0f, 1.0f / (4.0f * 4.0f) },
            { 1.0f - 2.0f / 5.0f, 1.0f / (5.0f * 5.0f) },
            { 1.0f - 2.0f / 6.0f, 1.0f / (6.0f * 6.0f) },
            { 1.0f - 2.0f / 7.0f, 1.0f / (7.0f * 7.0f) },
            { 1.0f - 2.0f / 8.0f, 1.0f / (8.0f * 8.0f) },
            { 1.0f - 2.0f / 9.0f, 1.0f / (9.0f * 9.0f) },
            { 1.0f - 2.0f / 10.0f, 1.0f / (10.0f * 10.0f) }
        };

        int vwidx = n - 3;

        clear(result);
        addWithWeight(result, vertex, p, vw[vwidx][0]);
        for (int j = 0; j < 2 * n; ++j, ++h)
            addWithWeight(result, vertex, V_IT[h], vw[vwidx][1]);

        int dstIndex = i + vertexOffset - tableOffset;
        copy(vertex, result, dstIndex);
    }
}

void OsdNeonComputeRestrictedVertexA(float *vertex, const int *V_ITa,
    int vertexOffset, int tableOffset, int start, int end)
{
    float result[6];

    for (int i = start + tableOffset; i < end + tableOffset; ++i) {
        int p = V_ITa[5 * i + 2];
        int eidx0 = V_ITa[5 * i + 3];
        int eidx1 = V_ITa[5 * i + 4];

        clear(result);
        addWithWeight(result, vertex, p, 0.75f);
        addWithWeight(result, vertex, eidx0, 0.125f);
        addWithWeight(result, vertex, eidx1, 0.125f);

        int dstIndex = i + vertexOffset - tableOffset;
        copy(vertex, result, dstIndex);
    }
}

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
