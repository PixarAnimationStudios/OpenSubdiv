//
//     Copyright (C) Pixar. All rights reserved.
//
//     This license governs use of the accompanying software. If you
//     use the software, you accept this license. If you do not accept
//     the license, do not use the software.
//
//     1. Definitions
//     The terms "reproduce," "reproduction," "derivative works," and
//     "distribution" have the same meaning here as under U.S.
//     copyright law.  A "contribution" is the original software, or
//     any additions or changes to the software.
//     A "contributor" is any person or entity that distributes its
//     contribution under this license.
//     "Licensed patents" are a contributor's patent claims that read
//     directly on its contribution.
//
//     2. Grant of Rights
//     (A) Copyright Grant- Subject to the terms of this license,
//     including the license conditions and limitations in section 3,
//     each contributor grants you a non-exclusive, worldwide,
//     royalty-free copyright license to reproduce its contribution,
//     prepare derivative works of its contribution, and distribute
//     its contribution or any derivative works that you create.
//     (B) Patent Grant- Subject to the terms of this license,
//     including the license conditions and limitations in section 3,
//     each contributor grants you a non-exclusive, worldwide,
//     royalty-free license under its licensed patents to make, have
//     made, use, sell, offer for sale, import, and/or otherwise
//     dispose of its contribution in the software or derivative works
//     of the contribution in the software.
//
//     3. Conditions and Limitations
//     (A) No Trademark License- This license does not grant you
//     rights to use any contributor's name, logo, or trademarks.
//     (B) If you bring a patent claim against any contributor over
//     patents that you claim are infringed by the software, your
//     patent license from such contributor to the software ends
//     automatically.
//     (C) If you distribute any portion of the software, you must
//     retain all copyright, patent, trademark, and attribution
//     notices that are present in the software.
//     (D) If you distribute any portion of the software in source
//     code form, you may do so only under this license by including a
//     complete copy of this license with your distribution. If you
//     distribute any portion of the software in compiled or object
//     code form, you may only do so under a license that complies
//     with this license.
//     (E) The software is licensed "as-is." You bear the risk of
//     using it. The contributors give no express warranties,
//     guarantees or conditions. You may have additional consumer
//     rights under your local laws which this license cannot change.
//     To the extent permitted under your local laws, the contributors
//     exclude the implied warranties of merchantability, fitness for
//     a particular purpose and non-infringement.
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
