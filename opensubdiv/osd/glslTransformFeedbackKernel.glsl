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

#version 420

subroutine void computeKernelType();
subroutine uniform computeKernelType computeKernel;

//------------------------------------------------------------------------------

uniform samplerBuffer vertexBuffer;

out float outVertexBuffer[LENGTH];

uniform isamplerBuffer sizes;
uniform isamplerBuffer offsets;
uniform isamplerBuffer indices;
uniform samplerBuffer  weights;

uniform int batchStart = 0;
uniform int batchEnd = 0;

uniform int primvarOffset = 0;

//------------------------------------------------------------------------------

struct Vertex {
    float vertexData[LENGTH];
};

void clear(out Vertex v) {
    for (int i = 0; i < LENGTH; i++) {
        v.vertexData[i] = 0;
    }
}

void addWithWeight(inout Vertex v, Vertex src, float weight) {
    for(int i = 0; i < LENGTH; i++) {
        v.vertexData[i] += weight * src.vertexData[i];
    }
}

Vertex readVertex(int index) {
    Vertex v;
    int vertexIndex = primvarOffset + index * STRIDE;
    for(int i = 0; i < LENGTH; i++) {
        v.vertexData[i] = texelFetch(vertexBuffer, vertexIndex+i).x;
    }
    return v;
}

void copyVertex(out Vertex dst, int index) {
    for(int i = 0; i < LENGTH; i++) {
        dst.vertexData[i] = texelFetch(vertexBuffer, index*STRIDE+i).x;
    }
}

void writeVertex(Vertex v) {
    for(int i = 0; i < LENGTH; i++) {
        outVertexBuffer[i] = v.vertexData[i];
    }
}

//------------------------------------------------------------------------------
subroutine(computeKernelType)
void computeStencil() {

    int current = gl_VertexID + batchStart;

    if (current>batchEnd) {
        return;
    }

    Vertex dst;
    clear(dst);

    int offset = texelFetch(offsets, current).x,
        size = texelFetch(sizes, current).x;

    for (int i=0; i<size; ++i) {
        int index = texelFetch(indices, offset+i).x;
        float weight = texelFetch(weights, offset+i).x;
        addWithWeight(dst, readVertex( index ), weight);
    }

    // the vertex buffer contains our control vertices at the beginning: don't
    // stomp on those !
    writeVertex(dst);
}

//------------------------------------------------------------------------------

void main() {

    // call subroutine
    computeKernel();
}

//------------------------------------------------------------------------------
