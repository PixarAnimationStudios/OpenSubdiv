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

class OsdD3D11DrawContext : public OsdDrawContext {
public:
    typedef ID3D11Buffer * VertexBufferBinding;

    OsdD3D11DrawContext();
    virtual ~OsdD3D11DrawContext();

    template<class VERTEX_BUFFER>
    static OsdD3D11DrawContext *
    Create(FarMesh<OsdVertex> *farMesh,
           VERTEX_BUFFER *vertexBuffer,
           ID3D11DeviceContext *pd3d11DeviceContext,
           bool requirePtexCoordinates = false,
           bool requireFVarData = false) {

        OsdD3D11DrawContext * instance = new OsdD3D11DrawContext();
        if (instance->allocate(farMesh,
                        vertexBuffer->BindD3D11Buffer(pd3d11DeviceContext),
                        vertexBuffer->GetNumElements(),
                        pd3d11DeviceContext,
                        requirePtexCoordinates,
                        requireFVarData)) return instance;
        delete instance;
        return NULL;
    }

    ID3D11Buffer             *patchIndexBuffer;

    ID3D11Buffer             *ptexCoordinateBuffer;
    ID3D11Buffer             *ptexCoordinateBufferSRV;
    ID3D11Buffer             *fvarDataBuffer;
    ID3D11Buffer             *fvarDataBufferSRV;

    ID3D11ShaderResourceView *vertexBufferSRV;
    ID3D11Buffer             *vertexValenceBuffer;
    ID3D11ShaderResourceView *vertexValenceBufferSRV;
    ID3D11Buffer             *quadOffsetBuffer;
    ID3D11ShaderResourceView *quadOffsetBufferSRV;
    ID3D11Buffer             *patchLevelBuffer;
    ID3D11ShaderResourceView *patchLevelBufferSRV;

private:
    bool allocate(FarMesh<OsdVertex> *farmesh,
                  ID3D11Buffer *vertexBuffer,
                  int numElements,
                  ID3D11DeviceContext *pd3d11DeviceContext,
                  bool requirePtexCoordinates,
                  bool requireFVarData);

    void _AppendPatchArray(
            unsigned int *indexBuffer, int *indexBase,
            unsigned int *levelBuffer, int *levelBase,
            FarPatchTables::PTable const & ptable, int patchSize,
            FarPatchTables::PtexCoordinateTable const & ptexTable,
            FarPatchTables::FVarDataTable const & fvarTable, int fvarDataWidth,
            OsdPatchDescriptor const & desc,
            int gregoryQuadOffsetBase);
};

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* OSD_D3D11L_DRAW_CONTEXT_H */
