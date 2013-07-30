//
//     Copyright 2013 Pixar
//
//     Licensed under the Apache License, Version 2.0 (the "License");
//     you may not use this file except in compliance with the License
//     and the following modification to it: Section 6 Trademarks.
//     deleted and replaced with:
//
//     6. Trademarks. This License does not grant permission to use the
//     trade names, trademarks, service marks, or product names of the
//     Licensor and its affiliates, except as required for reproducing
//     the content of the NOTICE file.
//
//     You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
//     Unless required by applicable law or agreed to in writing,
//     software distributed under the License is distributed on an
//     "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
//     either express or implied.  See the License for the specific
//     language governing permissions and limitations under the
//     License.
//
#ifndef OSD_D3D11L_DRAW_CONTEXT_H
#define OSD_D3D11L_DRAW_CONTEXT_H

#include "../version.h"

#include "../far/mesh.h"
#include "../osd/drawContext.h"
#include "../osd/vertex.h"

#include <map>

struct ID3D11VertexShader;
struct ID3D11HullShader;
struct ID3D11DomainShader;
struct ID3D11GeometryShader;
struct ID3D11PixelShader;

struct ID3D11Buffer;
struct ID3D11ShaderResourceView;
struct ID3D11Device;
struct ID3D11DeviceContext;

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

/// \brief D3D11 specialized DrawContext class
///
/// OsdD3D11DrawContext implements the OSD drawing interface with Microsoft(c)
/// DirectX D3D11 API.
/// Some functionality may be disabled depending on compile and run-time driver
/// support.
///
/// Contexts interface the serialized topological data pertaining to the 
/// geometric primitives with the capabilities of the selected discrete 
/// compute device.
///
class OsdD3D11DrawContext : public OsdDrawContext {
public:
    typedef ID3D11Buffer * VertexBufferBinding;

    virtual ~OsdD3D11DrawContext();

    /// \brief Create an OsdD3D11DrawContext from FarPatchTables
    ///
    /// @param patchTables          a valid set of FarPatchTables
    ///
    /// @param pd3d11DeviceContext  a device context
    ///
    /// @param requireFVarData      set to true to enable face-varying data to be 
    ///                             carried over from the Far data structures.
    ///
    ///
    static OsdD3D11DrawContext *Create(FarPatchTables const *patchTables,
                                       ID3D11DeviceContext *pd3d11DeviceContext,
                                       bool requireFVarData=false);

    /// Set vbo as a vertex texture (for gregory patch drawing)
    ///
    /// @param vbo                  the vertex buffer object to update
    ///
    /// @param pd3d11DeviceContext  a device context
    ///
    template<class VERTEX_BUFFER>
    void UpdateVertexTexture(VERTEX_BUFFER *vbo, ID3D11DeviceContext *pd3d11DeviceContext) {
        updateVertexTexture(vbo->BindD3D11Buffer(pd3d11DeviceContext),
                            pd3d11DeviceContext,
                            vbo->GetNumVertices(),
                            vbo->GetNumElements());
    }

    ID3D11Buffer             *patchIndexBuffer;

    ID3D11Buffer             *ptexCoordinateBuffer;
    ID3D11ShaderResourceView *ptexCoordinateBufferSRV;
    ID3D11Buffer             *fvarDataBuffer;
    ID3D11Buffer             *fvarDataBufferSRV;

    ID3D11ShaderResourceView *vertexBufferSRV;
    ID3D11Buffer             *vertexValenceBuffer;
    ID3D11ShaderResourceView *vertexValenceBufferSRV;
    ID3D11Buffer             *quadOffsetBuffer;
    ID3D11ShaderResourceView *quadOffsetBufferSRV;

private:
    OsdD3D11DrawContext();

    // allocate buffers from patchTables
    bool create(FarPatchTables const *patchTables,
                ID3D11DeviceContext *pd3d11DeviceContext,
                bool requireFVarData);

    void updateVertexTexture(ID3D11Buffer *vbo,
                             ID3D11DeviceContext *pd3d11DeviceContext,
                             int numVertices,
                             int numElements);

    int _numVertices;
};

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* OSD_D3D11L_DRAW_CONTEXT_H */
