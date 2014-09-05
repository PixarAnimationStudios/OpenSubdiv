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

struct Vertex {
    float v[LENGTH];
};

static void clear(struct Vertex *vertex) {
    for (int i = 0; i < LENGTH; i++) {
        vertex->v[i] = 0.0f;
    }
}

static void addWithWeight(struct Vertex *dst,
                          __global float *srcOrigin,
                          int index, float weight) {

    __global float *src = srcOrigin + index * STRIDE;
    for (int i = 0; i < LENGTH; ++i) {
        dst->v[i] += src[i] * weight;
    }
}

static void writeVertex(__global float *dstOrigin,
                        int index,
                        struct Vertex *src) {

    __global float *dst = dstOrigin + index * STRIDE;
    for (int i = 0; i < LENGTH; ++i) {
        dst[i] = src->v[i];
    }
}


__kernel void computeStencils( __global float * vertexBuffer,
                               __global unsigned char * sizes,
                               __global int * offsets,
                               __global int * indices,
                               __global float * weights,
                               int batchStart,
                               int batchEnd,
                               int primvarOffset,
                               int numCVs ) {

    int current = get_global_id(0) + batchStart;

    struct Vertex dst;
    clear(&dst);

    int size = (int)sizes[current],
        offset = offsets[current];

    vertexBuffer += primvarOffset;

    for (int i=0; i<size; ++i) {
        addWithWeight(&dst, vertexBuffer, indices[offset+i], weights[offset+i]);
    }

    writeVertex(vertexBuffer, numCVs+current, &dst);
}
