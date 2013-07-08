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

#include "../osd/d3d11DrawRegistry.h"
#include "../osd/error.h"

#include <D3D11.h>
#include <D3Dcompiler.h>

#include <sstream>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

OsdD3D11DrawConfig::~OsdD3D11DrawConfig()
{
    if (vertexShader) vertexShader->Release();
    if (hullShader) hullShader->Release();
    if (domainShader) domainShader->Release();
    if (geometryShader) geometryShader->Release();
    if (pixelShader) pixelShader->Release();
}

static const char *commonShaderSource =
#include "hlslPatchCommon.inc"
;
static const char *bsplineShaderSource =
#include "hlslPatchBSpline.inc"
;
static const char *gregoryShaderSource =
#include "hlslPatchGregory.inc"
;
static const char *transitionShaderSource =
#include "hlslPatchTransition.inc"
;

OsdD3D11DrawRegistryBase::~OsdD3D11DrawRegistryBase() {}

OsdD3D11DrawSourceConfig *
OsdD3D11DrawRegistryBase::_CreateDrawSourceConfig(
    OsdDrawContext::PatchDescriptor const & desc, ID3D11Device * pd3dDevice)
{
    OsdD3D11DrawSourceConfig * sconfig = _NewDrawSourceConfig();

    sconfig->commonShader.source = commonShaderSource;
    {
        std::ostringstream ss;
        ss << (int)desc.GetMaxValence();
        sconfig->commonShader.AddDefine("OSD_MAX_VALENCE", ss.str());
        ss.str("");
        ss << (int)desc.GetNumElements();
        sconfig->commonShader.AddDefine("OSD_NUM_ELEMENTS", ss.str());
    }

    if (desc.GetPattern() == FarPatchTables::NON_TRANSITION) {
        switch (desc.GetType()) {
        case FarPatchTables::QUADS:
        case FarPatchTables::TRIANGLES:
            // do nothing
            break;
        case FarPatchTables::REGULAR:
            sconfig->vertexShader.source = bsplineShaderSource;
            sconfig->vertexShader.target = "vs_5_0";
            sconfig->vertexShader.entry = "vs_main_patches";
            sconfig->hullShader.source = bsplineShaderSource;
            sconfig->hullShader.target = "hs_5_0";
            sconfig->hullShader.entry = "hs_main_patches";
            sconfig->domainShader.source = bsplineShaderSource;
            sconfig->domainShader.target = "ds_5_0";
            sconfig->domainShader.entry = "ds_main_patches";
            break;
        case FarPatchTables::BOUNDARY:
            sconfig->vertexShader.source = bsplineShaderSource;
            sconfig->vertexShader.target = "vs_5_0";
            sconfig->vertexShader.entry = "vs_main_patches";
            sconfig->hullShader.source = bsplineShaderSource;
            sconfig->hullShader.target = "hs_5_0";
            sconfig->hullShader.entry = "hs_main_patches";
            sconfig->hullShader.AddDefine("OSD_PATCH_BOUNDARY");
            sconfig->domainShader.source = bsplineShaderSource;
            sconfig->domainShader.target = "ds_5_0";
            sconfig->domainShader.entry = "ds_main_patches";
            break;
        case FarPatchTables::CORNER:
            sconfig->vertexShader.source = bsplineShaderSource;
            sconfig->vertexShader.target = "vs_5_0";
            sconfig->vertexShader.entry = "vs_main_patches";
            sconfig->hullShader.source = bsplineShaderSource;
            sconfig->hullShader.target = "hs_5_0";
            sconfig->hullShader.entry = "hs_main_patches";
            sconfig->hullShader.AddDefine("OSD_PATCH_CORNER");
            sconfig->domainShader.source = bsplineShaderSource;
            sconfig->domainShader.target = "ds_5_0";
            sconfig->domainShader.entry = "ds_main_patches";
            break;
        case FarPatchTables::GREGORY:
            sconfig->vertexShader.source = gregoryShaderSource;
            sconfig->vertexShader.target = "vs_5_0";
            sconfig->vertexShader.entry = "vs_main_patches";
            sconfig->hullShader.source = gregoryShaderSource;
            sconfig->hullShader.target = "hs_5_0";
            sconfig->hullShader.entry = "hs_main_patches";
            sconfig->domainShader.source = gregoryShaderSource;
            sconfig->domainShader.target = "ds_5_0";
            sconfig->domainShader.entry = "ds_main_patches";
            break;
        case FarPatchTables::GREGORY_BOUNDARY:
            sconfig->vertexShader.source = gregoryShaderSource;
            sconfig->vertexShader.target = "vs_5_0";
            sconfig->vertexShader.entry = "vs_main_patches";
            sconfig->vertexShader.AddDefine("OSD_PATCH_GREGORY_BOUNDARY");
            sconfig->hullShader.source = gregoryShaderSource;
            sconfig->hullShader.target = "hs_5_0";
            sconfig->hullShader.entry = "hs_main_patches";
            sconfig->hullShader.AddDefine("OSD_PATCH_GREGORY_BOUNDARY");
            sconfig->domainShader.source = gregoryShaderSource;
            sconfig->domainShader.target = "ds_5_0";
            sconfig->domainShader.entry = "ds_main_patches";
            sconfig->domainShader.AddDefine("OSD_PATCH_GREGORY_BOUNDARY");
            break;
        default:
            delete sconfig;
            sconfig = NULL;
            break;
        }
    } else { // pattern != NON_TRANSITION
        sconfig->vertexShader.source = bsplineShaderSource;
        sconfig->vertexShader.target = "vs_5_0";
        sconfig->vertexShader.entry = "vs_main_patches";
        sconfig->hullShader.source =
            std::string(transitionShaderSource) + bsplineShaderSource;
        sconfig->hullShader.target = "hs_5_0";
        sconfig->hullShader.entry = "hs_main_patches";
        sconfig->hullShader.AddDefine("OSD_PATCH_TRANSITION");
        sconfig->domainShader.source =
            std::string(transitionShaderSource) + bsplineShaderSource;
        sconfig->domainShader.target = "ds_5_0";
        sconfig->domainShader.entry = "ds_main_patches";
        sconfig->domainShader.AddDefine("OSD_PATCH_TRANSITION");

        int pattern = desc.GetPattern() - 1;
        int rotation = desc.GetRotation();
        int subpatch = desc.GetSubPatch();

        std::ostringstream ss;
        ss << "OSD_TRANSITION_PATTERN" << pattern << subpatch;
        sconfig->hullShader.AddDefine(ss.str());
        sconfig->domainShader.AddDefine(ss.str());

        ss.str("");
        ss << rotation;
        sconfig->hullShader.AddDefine("OSD_TRANSITION_ROTATE", ss.str());
        sconfig->domainShader.AddDefine("OSD_TRANSITION_ROTATE", ss.str());

        if (desc.GetType() == FarPatchTables::BOUNDARY) {
            sconfig->hullShader.AddDefine("OSD_PATCH_BOUNDARY");
        } else if (desc.GetType() == FarPatchTables::CORNER) {
            sconfig->hullShader.AddDefine("OSD_PATCH_CORNER");
        }
    }

    return sconfig;
}

static ID3DBlob *
_CompileShader(
        OsdDrawShaderSource const & common,
        OsdDrawShaderSource const & source)
{
    DWORD dwShaderFlags = D3DCOMPILE_ENABLE_STRICTNESS;
#ifdef _DEBUG
    dwShaderFlags |= D3DCOMPILE_DEBUG;
#endif

    ID3DBlob* pBlob = NULL;
    ID3DBlob* pBlobError = NULL;

    std::vector<D3D_SHADER_MACRO> shaderDefines;
    for (int i=0; i<(int)common.defines.size(); ++i) {
        const D3D_SHADER_MACRO def = {
            common.defines[i].first.c_str(),
            common.defines[i].second.c_str(),
        };
        shaderDefines.push_back(def);
    }
    for (int i=0; i<(int)source.defines.size(); ++i) {
        const D3D_SHADER_MACRO def = {
            source.defines[i].first.c_str(),
            source.defines[i].second.c_str(),
        };
        shaderDefines.push_back(def);
    }
    const D3D_SHADER_MACRO def = { 0, 0 };
    shaderDefines.push_back(def);

    std::string shaderSource = common.source + source.source;

    HRESULT hr = D3DCompile(shaderSource.c_str(), shaderSource.size(),
                            NULL, &shaderDefines[0], NULL,
                            source.entry.c_str(), source.target.c_str(),
                            dwShaderFlags, 0, &pBlob, &pBlobError);
    if (FAILED(hr)) {
        if ( pBlobError != NULL ) {
            OsdError(OSD_D3D11_COMPILE_ERROR,
                     "Error compiling HLSL shader: %s\n",
                     (CHAR*)pBlobError->GetBufferPointer());
            pBlobError->Release();
            return NULL;
        }
    }

    return pBlob;
}

#define SAFE_RELEASE(p) { if(p) { (p)->Release(); (p)=NULL; } }

OsdD3D11DrawConfig*
OsdD3D11DrawRegistryBase::_CreateDrawConfig(
        DescType const & desc,
        SourceConfigType const * sconfig,
        ID3D11Device * pd3dDevice,
        ID3D11InputLayout ** ppInputLayout,
        D3D11_INPUT_ELEMENT_DESC const * pInputElementDescs,
        int numInputElements)
{
    assert(sconfig);

    ID3DBlob * pBlob;

    ID3D11VertexShader *vertexShader = NULL;
    if (! sconfig->vertexShader.source.empty()) {
        pBlob = _CompileShader(sconfig->commonShader, sconfig->vertexShader);
        pd3dDevice->CreateVertexShader(pBlob->GetBufferPointer(),
                                         pBlob->GetBufferSize(),
                                         NULL,
                                         &vertexShader);
        assert(vertexShader);

        if (ppInputLayout and !*ppInputLayout) {
            pd3dDevice->CreateInputLayout(pInputElementDescs, numInputElements,
                                          pBlob->GetBufferPointer(),
                                          pBlob->GetBufferSize(), ppInputLayout);
            assert(ppInputLayout);
        }

        SAFE_RELEASE(pBlob);
    }


    ID3D11HullShader *hullShader = NULL;
    if (! sconfig->hullShader.source.empty()) {
        pBlob = _CompileShader(sconfig->commonShader, sconfig->hullShader);
        pd3dDevice->CreateHullShader(pBlob->GetBufferPointer(),
                                       pBlob->GetBufferSize(),
                                       NULL,
                                       &hullShader);
        assert(hullShader);
        SAFE_RELEASE(pBlob);
    }

    ID3D11DomainShader *domainShader = NULL;
    if (! sconfig->domainShader.source.empty()) {
        pBlob = _CompileShader(sconfig->commonShader, sconfig->domainShader);
        pd3dDevice->CreateDomainShader(pBlob->GetBufferPointer(),
                                         pBlob->GetBufferSize(),
                                         NULL,
                                         &domainShader);
        assert(domainShader);
        SAFE_RELEASE(pBlob);
    }

    ID3D11GeometryShader *geometryShader = NULL;
    if (! sconfig->geometryShader.source.empty()) {
        pBlob = _CompileShader(sconfig->commonShader, sconfig->geometryShader);
        pd3dDevice->CreateGeometryShader(pBlob->GetBufferPointer(),
                                           pBlob->GetBufferSize(),
                                           NULL,
                                           &geometryShader);
        assert(geometryShader);
        SAFE_RELEASE(pBlob);
    }

    ID3D11PixelShader *pixelShader = NULL;
    if (! sconfig->pixelShader.source.empty()) {
        pBlob = _CompileShader(sconfig->commonShader, sconfig->pixelShader);
        pd3dDevice->CreatePixelShader(pBlob->GetBufferPointer(),
                                        pBlob->GetBufferSize(),
                                        NULL,
                                        &pixelShader);
        assert(pixelShader);
        SAFE_RELEASE(pBlob);
    }

    OsdD3D11DrawConfig * config = _NewDrawConfig();

    config->vertexShader = vertexShader;
    config->hullShader = hullShader;
    config->domainShader = domainShader;
    config->geometryShader = geometryShader;
    config->pixelShader = pixelShader;

    return config;
}

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv
