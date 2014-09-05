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

#version 430

subroutine void computeKernelType();
subroutine uniform computeKernelType computeKernel;

//------------------------------------------------------------------------------

uniform int batchStart = 0;
uniform int batchEnd = 0;

uniform int primvarOffset = 0;
uniform int numCVs = 0;

layout(binding=0) buffer vertex_buffer    { float         vertexBuffer[]; };
layout(binding=1) buffer sterncilSizes    { unsigned char _sizes[];   };
layout(binding=2) buffer sterncilOffsets  { int           _offsets[]; };
layout(binding=3) buffer sterncilIndices  { int           _indices[]; };
layout(binding=4) buffer sterncilWeights  { float         _weights[]; };

layout(local_size_x=WORK_GROUP_SIZE, local_size_y=1, local_size_z=1) in;

//------------------------------------------------------------------------------

struct Vertex {
    float vertexData[LENGTH];
};

void clear(out Vertex v) {
    for (int i = 0; i < LENGTH; ++i) {
        v.vertexData[i] = 0;
    }
}

Vertex readVertex(int index) {
    Vertex v;
    int vertexIndex = primvarOffset + index * STRIDE;
    for (int i = 0; i < LENGTH; ++i) {
        v.vertexData[i] = vertexBuffer[vertexIndex + i];
    }
    return v;
}

void writeVertex(int index, Vertex v) {
    int vertexIndex = primvarOffset + index * STRIDE;
    for (int i = 0; i < LENGTH; ++i) {
        vertexBuffer[vertexIndex + i] = v.vertexData[i];
    }
}

void addWithWeight(inout Vertex v, const Vertex src, float weight) {
    for (int i = 0; i < LENGTH; ++i) {
        v.vertexData[i] += weight * src.vertexData[i];
    }
}

//------------------------------------------------------------------------------
subroutine(computeKernelType)
void computeStencil() {

    int current = int(gl_GlobalInvocationID.x) + batchStart;

    if (current>batchEnd) {
        return;
    }

    Vertex dst;
    clear(dst);

    int offset = _offsets[current],
        size = int(_sizes[current]);
    
    for (int i=0; i<size; ++i) {
        addWithWeight(dst, readVertex( _indices[offset+i] ), _weights[offset+i]);
    }

    // the vertex buffer contains our control vertices at the beginning: don't
    // stomp on those !
    writeVertex(numCVs+current, dst);
}

//------------------------------------------------------------------------------

void main()
{
    // call subroutine
    computeKernel();
}

//------------------------------------------------------------------------------
