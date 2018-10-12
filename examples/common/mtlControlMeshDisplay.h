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

#ifndef OPENSUBDIV_EXAMPLES_MTL_CONTROL_MESH_DISPLAY_H
#define OPENSUBDIV_EXAMPLES_MTL_CONTROL_MESH_DISPLAY_H

#include <Metal/Metal.h>
#include <opensubdiv/far/topologyLevel.h>

class MTLControlMeshDisplay {
public:
    MTLControlMeshDisplay(id<MTLDevice> device, MTLRenderPipelineDescriptor* pipelineDescriptor);
    ~MTLControlMeshDisplay() = default;

    void Draw(id<MTLRenderCommandEncoder> encoder, 
    		  id<MTLBuffer> vertexBuffer,
              const float *modelViewProjectionMatrix);

    void SetTopology(OpenSubdiv::Far::TopologyLevel const &level);

    bool GetEdgesDisplay() const { return _displayEdges; }
    void SetEdgesDisplay(bool display) { _displayEdges = display; }
    bool GetVerticesDisplay() const { return _displayVertices; }
    void SetVerticesDisplay(bool display) { _displayVertices = display; }

private:
    bool createProgram(MTLRenderPipelineDescriptor* pipelineDescriptor);

    bool _displayEdges;
    bool _displayVertices;

    id<MTLDevice> _device;

    int _numEdges, _numPoints;
    id<MTLRenderPipelineState> _renderPipelineState;
    id<MTLBuffer> _vertexSharpnessBuffer;
    id<MTLBuffer> _edgeSharpnessBuffer;
    id<MTLBuffer> _edgeIndicesBuffer;
};

#endif  // OPENSUBDIV_EXAMPLES_MTL_CONTROL_MESH_DISPLAY_H
