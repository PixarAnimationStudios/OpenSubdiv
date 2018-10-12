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

#ifndef OPENSUBDIV_EXAMPLES_D3D11_CONTROL_MESH_DISPLAY_H
#define OPENSUBDIV_EXAMPLES_D3D11_CONTROL_MESH_DISPLAY_H

#include <d3d11.h>
#include <opensubdiv/far/topologyLevel.h>

class D3D11ControlMeshDisplay {
public:
    D3D11ControlMeshDisplay(ID3D11DeviceContext *deviceContext);
    ~D3D11ControlMeshDisplay();

    void Draw(ID3D11Buffer *buffer, int stride,
              const float *modelViewProjectionMatrix);

    void SetTopology(OpenSubdiv::Far::TopologyLevel const &level);

    bool GetEdgesDisplay() const { return _displayEdges; }
    void SetEdgesDisplay(bool display) { _displayEdges = display; }
    bool GetVerticesDisplay() const { return _displayVertices; }
    void SetVerticesDisplay(bool display) { _displayVertices = display; }

private:
    bool createProgram();

    bool _displayEdges;
    bool _displayVertices;

    ID3D11DeviceContext *_deviceContext;
    ID3D11InputLayout *_inputLayout;
    ID3D11VertexShader *_vertexShader;
    ID3D11PixelShader *_pixelShader;
    ID3D11RasterizerState *_rasterizerState;
    ID3D11Buffer *_constantBuffer;
    ID3D11ShaderResourceView *_edgeSharpnessSRV;
    ID3D11Buffer *_edgeSharpness;
    ID3D11Buffer *_edgeIndices;

    int _numEdges, _numPoints;
};

#endif  // OPENSUBDIV_EXAMPLES_D3D11_CONTROL_MESH_DISPLAY_H
