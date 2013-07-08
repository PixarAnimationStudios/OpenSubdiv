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

#include "../far/dispatcher.h"
#include "../far/loopSubdivisionTables.h"
#include "../osd/d3d11DrawContext.h"

#include <D3D11.h>

#include <string.h>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

OsdD3D11DrawContext::OsdD3D11DrawContext() :
    patchIndexBuffer(NULL),
    ptexCoordinateBuffer(NULL),
    ptexCoordinateBufferSRV(NULL),
    fvarDataBuffer(NULL),
    fvarDataBufferSRV(NULL),
    vertexBufferSRV(NULL),
    vertexValenceBuffer(NULL),
    vertexValenceBufferSRV(NULL),
    quadOffsetBuffer(NULL),
    quadOffsetBufferSRV(NULL)
{
}

OsdD3D11DrawContext::~OsdD3D11DrawContext()
{
    if (patchIndexBuffer) patchIndexBuffer->Release();
    if (ptexCoordinateBuffer) ptexCoordinateBuffer->Release();
    if (ptexCoordinateBufferSRV) ptexCoordinateBufferSRV->Release();
    if (fvarDataBuffer) fvarDataBuffer->Release();
    if (fvarDataBufferSRV) fvarDataBufferSRV->Release();
    if (vertexBufferSRV) vertexBufferSRV->Release();
    if (vertexValenceBuffer) vertexValenceBuffer->Release();
    if (vertexValenceBufferSRV) vertexValenceBufferSRV->Release();
    if (quadOffsetBuffer) quadOffsetBuffer->Release();
    if (quadOffsetBufferSRV) quadOffsetBufferSRV->Release();
};

OsdD3D11DrawContext *
OsdD3D11DrawContext::Create(FarPatchTables const *patchTables,
                            ID3D11DeviceContext *pd3d11DeviceContext,
                            bool requireFVarData)
{
    OsdD3D11DrawContext * result = new OsdD3D11DrawContext();
    if (result->create(patchTables, pd3d11DeviceContext, requireFVarData))
        return result;

    delete result;
    return NULL;
}

bool
OsdD3D11DrawContext::create(FarPatchTables const *patchTables,
                            ID3D11DeviceContext *pd3d11DeviceContext,
                            bool requireFVarData)
{
    // adaptive patches
    _isAdaptive = true;

    ID3D11Device *pd3d11Device = NULL;
    pd3d11DeviceContext->GetDevice(&pd3d11Device);
    assert(pd3d11Device);

    ConvertPatchArrays(patchTables->GetPatchArrayVector(), patchArrays, patchTables->GetMaxValence(), 0);

    FarPatchTables::PTable const & ptables = patchTables->GetPatchTable();
    FarPatchTables::PatchParamTable const & ptexCoordTables = patchTables->GetPatchParamTable();
    int totalPatchIndices = (int)ptables.size();
    int totalPatches = (int)ptexCoordTables.size();

    // Allocate and fill index buffer.
    D3D11_BUFFER_DESC bd;
    bd.ByteWidth = totalPatchIndices * sizeof(int);
    bd.Usage = D3D11_USAGE_DYNAMIC;
    bd.BindFlags = D3D11_BIND_INDEX_BUFFER;
    bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    bd.MiscFlags = 0;
    bd.StructureByteStride = sizeof(int);
    HRESULT hr = pd3d11Device->CreateBuffer(&bd, NULL, &patchIndexBuffer);
    if (FAILED(hr)) {
        return false;
    }

    D3D11_MAPPED_SUBRESOURCE mappedResource;
    hr = pd3d11DeviceContext->Map(patchIndexBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
    if (FAILED(hr)) {
        return false;
    }
    unsigned int * indexBuffer = (unsigned int *) mappedResource.pData;
    memcpy(indexBuffer, &ptables[0], totalPatchIndices * sizeof(unsigned int));

    pd3d11DeviceContext->Unmap(patchIndexBuffer, 0);

    // create ptex coordinate buffer
    bd.ByteWidth = totalPatches * sizeof(unsigned int);
    bd.Usage = D3D11_USAGE_DYNAMIC;
    bd.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    bd.MiscFlags = 0;
    bd.StructureByteStride = sizeof(unsigned int);
    hr = pd3d11Device->CreateBuffer(&bd, NULL, &ptexCoordinateBuffer);
    if (FAILED(hr)) {
        return false;
    }

    D3D11_SHADER_RESOURCE_VIEW_DESC srvd;
    ZeroMemory(&srvd, sizeof(srvd));
    srvd.Format = DXGI_FORMAT_R32_UINT;   // XXX: this should be DXGI_FORMAT_R32G32_UINT?
    srvd.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
    srvd.Buffer.FirstElement = 0;
    srvd.Buffer.NumElements = totalPatches;
    hr = pd3d11Device->CreateShaderResourceView(ptexCoordinateBuffer, &srvd, &ptexCoordinateBufferSRV);
    if (FAILED(hr)) {
        return false;
    }

    hr = pd3d11DeviceContext->Map(ptexCoordinateBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
    if (FAILED(hr)) {
        return false;
    }
    unsigned int * ptexBuffer = (unsigned int *) mappedResource.pData;
    memcpy(ptexBuffer, &ptexCoordTables[0], totalPatches * sizeof(FarPatchParam));
    pd3d11DeviceContext->Unmap(ptexCoordinateBuffer, 0);

    // create vertex valence buffer and vertex texture
    FarPatchTables::VertexValenceTable const &
        valenceTable = patchTables->GetVertexValenceTable();

    if (not valenceTable.empty()) {
        D3D11_BUFFER_DESC bd;
        bd.ByteWidth = UINT(valenceTable.size() * sizeof(unsigned int));
        bd.Usage = D3D11_USAGE_DEFAULT;
        bd.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        bd.CPUAccessFlags = 0;
        bd.MiscFlags = 0;
        bd.StructureByteStride = sizeof(unsigned int);
        D3D11_SUBRESOURCE_DATA initData;
        initData.pSysMem = &valenceTable[0];
        HRESULT hr = pd3d11Device->CreateBuffer(&bd, &initData, &vertexValenceBuffer);
        if (FAILED(hr)) {
            return false;
        }

        D3D11_SHADER_RESOURCE_VIEW_DESC srvd;
        ZeroMemory(&srvd, sizeof(srvd));
        srvd.Format = DXGI_FORMAT_R32_SINT;
        srvd.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
        srvd.Buffer.FirstElement = 0;
        srvd.Buffer.NumElements = UINT(valenceTable.size());
        hr = pd3d11Device->CreateShaderResourceView(vertexValenceBuffer, &srvd, &vertexValenceBufferSRV);
        if (FAILED(hr)) {
            return false;
        }
    }

    FarPatchTables::QuadOffsetTable const &
        quadOffsetTable = patchTables->GetQuadOffsetTable();

    if (not quadOffsetTable.empty()) {
        D3D11_BUFFER_DESC bd;
        bd.ByteWidth = UINT(quadOffsetTable.size() * sizeof(unsigned int));
        bd.Usage = D3D11_USAGE_DEFAULT;
        bd.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        bd.CPUAccessFlags = 0;
        bd.MiscFlags = 0;
        bd.StructureByteStride = sizeof(unsigned int);
        D3D11_SUBRESOURCE_DATA initData;
        initData.pSysMem = &quadOffsetTable[0];
        HRESULT hr = pd3d11Device->CreateBuffer(&bd, &initData, &quadOffsetBuffer);
        if (FAILED(hr)) {
            return false;
        }

        D3D11_SHADER_RESOURCE_VIEW_DESC srvd;
        ZeroMemory(&srvd, sizeof(srvd));
        srvd.Format = DXGI_FORMAT_R32_SINT;
        srvd.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
        srvd.Buffer.FirstElement = 0;
        srvd.Buffer.NumElements = UINT(quadOffsetTable.size());
        hr = pd3d11Device->CreateShaderResourceView(quadOffsetBuffer, &srvd, &quadOffsetBufferSRV);
        if (FAILED(hr)) {
            return false;
        }
    }

    return true;
}


void
OsdD3D11DrawContext::updateVertexTexture(ID3D11Buffer *vbo,
                                         ID3D11DeviceContext *pd3d11DeviceContext,
                                         int numVertices,
                                         int numVertexElements)
{
    ID3D11Device *pd3d11Device = NULL;
    pd3d11DeviceContext->GetDevice(&pd3d11Device);
    assert(pd3d11Device);

    D3D11_SHADER_RESOURCE_VIEW_DESC srvd;
    ZeroMemory(&srvd, sizeof(srvd));
    srvd.Format = DXGI_FORMAT_R32_FLOAT;
    srvd.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
    srvd.Buffer.FirstElement = 0;
    srvd.Buffer.NumElements = numVertices * numVertexElements;
    HRESULT hr = pd3d11Device->CreateShaderResourceView(vbo, &srvd, &vertexBufferSRV);
    if (FAILED(hr)) {
        return;
    }

    // XXX: consider moving this proc to base class
    // updating num elements in descriptor with new vbo specs
    for (int i = 0; i < (int)patchArrays.size(); ++i) {
        PatchArray &parray = patchArrays[i];
        PatchDescriptor desc = parray.GetDescriptor();
        desc.SetNumElements(numVertexElements);
        parray.SetDescriptor(desc);
    }
}

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv
