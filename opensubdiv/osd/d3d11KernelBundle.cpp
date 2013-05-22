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

#include "../osd/debug.h"
#include "../osd/error.h"
#include "../osd/d3d11KernelBundle.h"

#define INITGUID        // for IID_ID3D11ShaderReflection
#include <D3D11.h>
#include <D3D11shader.h>
#include <D3Dcompiler.h>

#include <cassert>
#include <cstring>
#include <sstream>
#include <string>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

static const char *shaderSource =
#include "../osd/hlslComputeKernel.inc"
;

#define SAFE_RELEASE(p) { if(p) { (p)->Release(); (p)=NULL; } }

OsdD3D11ComputeKernelBundle::OsdD3D11ComputeKernelBundle(
        ID3D11DeviceContext * deviceContext) :
    _deviceContext(deviceContext),
    _computeShader(0),
    _classLinkage(0),
    _kernelCB(0),
    _kernelComputeFace(0),
    _kernelComputeEdge(0),
    _kernelComputeBilinearEdge(0),
    _kernelComputeVertex(0),
    _kernelComputeVertexA(0),
    _kernelComputeCatmarkVertexB(0),
    _kernelComputeLoopVertexB(0),
    _kernelEditAdd(0) {

    // XXX: too rough!
    _workGroupSize = 64;
}

OsdD3D11ComputeKernelBundle::~OsdD3D11ComputeKernelBundle() {
    SAFE_RELEASE(_computeShader);
    SAFE_RELEASE(_classLinkage);
    SAFE_RELEASE(_kernelCB);
    SAFE_RELEASE(_kernelComputeFace);
    SAFE_RELEASE(_kernelComputeEdge);
    SAFE_RELEASE(_kernelComputeBilinearEdge);
    SAFE_RELEASE(_kernelComputeVertex);
    SAFE_RELEASE(_kernelComputeVertexA);
    SAFE_RELEASE(_kernelComputeCatmarkVertexB);
    SAFE_RELEASE(_kernelComputeLoopVertexB);
    SAFE_RELEASE(_kernelEditAdd);
}

bool
OsdD3D11ComputeKernelBundle::Compile(int numVertexElements,
                                     int numVaryingElements) {

    _vdesc.Set( numVertexElements, numVaryingElements );

    DWORD dwShaderFlags = D3DCOMPILE_ENABLE_STRICTNESS;
#ifdef _DEBUG
    dwShaderFlags |= D3DCOMPILE_DEBUG;
#endif

    std::ostringstream ss;
    ss << numVertexElements;
    std::string numVertexElementsStr(ss.str());
    ss.str("");
    ss << numVaryingElements;
    std::string numVaryingElementsStr(ss.str());
    ss.str("");
    ss << _workGroupSize;
    std::string workGroupSizeStr(ss.str());

    D3D_SHADER_MACRO shaderDefines[] = {
        "NUM_VERTEX_ELEMENTS", numVertexElementsStr.c_str(),
        "NUM_VARYING_ELEMENTS", numVaryingElementsStr.c_str(),
        "WORK_GROUP_SIZE", workGroupSizeStr.c_str(),
        0, 0
    };

    ID3DBlob* pComputeShaderBuffer = NULL;
    ID3DBlob* pErrorBuffer = NULL;

    HRESULT hr = D3DCompile(shaderSource, strlen(shaderSource),
                            NULL, &shaderDefines[0], NULL,
                            "cs_main", "cs_5_0",
                            dwShaderFlags, 0,
                            &pComputeShaderBuffer, &pErrorBuffer);
    if (FAILED(hr)) {
        if (pErrorBuffer != NULL) {
            OsdError(OSD_D3D11_COMPILE_ERROR,
                     "Error compiling HLSL shader: %s\n",
                     (CHAR*)pErrorBuffer->GetBufferPointer());
            pErrorBuffer->Release();
            return false;
        }
    }

    ID3D11Device *device = NULL;
    _deviceContext->GetDevice(&device);
    assert(device);

    device->CreateClassLinkage(&_classLinkage);
    assert(_classLinkage);

    device->CreateComputeShader(pComputeShaderBuffer->GetBufferPointer(),
                                pComputeShaderBuffer->GetBufferSize(),
                                _classLinkage,
                                &_computeShader);
    assert(_computeShader);

    ID3D11ShaderReflection *reflector;
    D3DReflect(pComputeShaderBuffer->GetBufferPointer(),
               pComputeShaderBuffer->GetBufferSize(),
               IID_ID3D11ShaderReflection, (void**) &reflector);
    assert(reflector);

    assert(reflector->GetNumInterfaceSlots() == 1);
    reflector->Release();

    pComputeShaderBuffer->Release();

    _classLinkage->GetClassInstance(
        "catmarkComputeFace", 0, &_kernelComputeFace);
    assert(_kernelComputeFace);
    _classLinkage->GetClassInstance(
        "catmarkComputeEdge", 0, &_kernelComputeEdge);
    assert(_kernelComputeEdge);
    _classLinkage->GetClassInstance(
        "bilinearComputeEdge", 0, &_kernelComputeBilinearEdge);
    assert(_kernelComputeBilinearEdge);
    _classLinkage->GetClassInstance(
        "bilinearComputeVertex", 0, &_kernelComputeVertex);
    assert(_kernelComputeVertex);
    _classLinkage->GetClassInstance(
        "catmarkComputeVertexA", 0, &_kernelComputeVertexA);
    assert(_kernelComputeVertexA);
    _classLinkage->GetClassInstance(
        "catmarkComputeVertexB", 0, &_kernelComputeCatmarkVertexB);
    assert(_kernelComputeCatmarkVertexB);
    _classLinkage->GetClassInstance(
        "loopComputeVertexB", 0, &_kernelComputeLoopVertexB);
    assert(_kernelComputeLoopVertexB);
    _classLinkage->GetClassInstance(
        "editAdd", 0, &_kernelEditAdd);
    assert(_kernelEditAdd);

    return true;
}

// must match constant buffer declaration in hlslComputeKernel.hlsl
__declspec(align(16))
struct OsdD3D11ComputeKernelBundle::KernelCB {
    int vertexOffset;   // vertex index offset for the batch
    int tableOffset;    // offset of subdivision table
    int indexStart;     // start index relative to tableOffset
    int indexEnd;       // end index relative to tableOffset
    BOOL vertexPass;    // 4-byte bool

// vertex edit kernel
    int editPrimVarOffset;
    int editPrimVarWidth;
};

void
OsdD3D11ComputeKernelBundle::dispatchCompute(
        ID3D11ClassInstance * kernel, KernelCB const & args) {

    int count = args.indexEnd - args.indexStart;
    if (count <= 0) return;

    if (! _kernelCB) {
        ID3D11Device *device = NULL;
        _deviceContext->GetDevice(&device);
        assert(device);

        D3D11_BUFFER_DESC cbDesc;
        ZeroMemory(&cbDesc, sizeof(cbDesc));
        cbDesc.Usage = D3D11_USAGE_DYNAMIC;
        cbDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        cbDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        cbDesc.MiscFlags = 0;
        cbDesc.ByteWidth = sizeof(KernelCB);
        device->CreateBuffer(&cbDesc, NULL, &_kernelCB);
    }
    assert(_kernelCB);

    D3D11_MAPPED_SUBRESOURCE MappedResource;
    _deviceContext->Map(_kernelCB, 0, D3D11_MAP_WRITE_DISCARD, 0, &MappedResource);
    CopyMemory(MappedResource.pData, &args, sizeof(KernelCB));
    _deviceContext->Unmap(_kernelCB, 0);
    _deviceContext->CSSetConstantBuffers(0, 1, &_kernelCB); // b0

    _deviceContext->CSSetShader(_computeShader, &kernel, 1);
    _deviceContext->Dispatch(count/_workGroupSize + 1, 1, 1);
}

void
OsdD3D11ComputeKernelBundle::ApplyBilinearFaceVerticesKernel(
    int vertexOffset, int tableOffset, int start, int end) {

    KernelCB args;
    ZeroMemory(&args, sizeof(args));
    args.vertexOffset = vertexOffset;
    args.tableOffset = tableOffset;
    args.indexStart = start;
    args.indexEnd = end;
    dispatchCompute(_kernelComputeFace, args);
}

void
OsdD3D11ComputeKernelBundle::ApplyBilinearEdgeVerticesKernel(
    int vertexOffset, int tableOffset, int start, int end) {

    KernelCB args;
    ZeroMemory(&args, sizeof(args));
    args.vertexOffset = vertexOffset;
    args.tableOffset = tableOffset;
    args.indexStart = start;
    args.indexEnd = end;
    dispatchCompute(_kernelComputeBilinearEdge, args);
}

void
OsdD3D11ComputeKernelBundle::ApplyBilinearVertexVerticesKernel(
    int vertexOffset, int tableOffset, int start, int end) {

    KernelCB args;
    ZeroMemory(&args, sizeof(args));
    args.vertexOffset = vertexOffset;
    args.tableOffset = tableOffset;
    args.indexStart = start;
    args.indexEnd = end;
    dispatchCompute(_kernelComputeVertex, args);
}


void
OsdD3D11ComputeKernelBundle::ApplyCatmarkFaceVerticesKernel(
    int vertexOffset, int tableOffset, int start, int end) {

    KernelCB args;
    ZeroMemory(&args, sizeof(args));
    args.vertexOffset = vertexOffset;
    args.tableOffset = tableOffset;
    args.indexStart = start;
    args.indexEnd = end;
    dispatchCompute(_kernelComputeFace, args);
}

void
OsdD3D11ComputeKernelBundle::ApplyCatmarkEdgeVerticesKernel(
    int vertexOffset, int tableOffset, int start, int end) {

    KernelCB args;
    ZeroMemory(&args, sizeof(args));
    args.vertexOffset = vertexOffset;
    args.tableOffset = tableOffset;
    args.indexStart = start;
    args.indexEnd = end;
    dispatchCompute(_kernelComputeEdge, args);
}

void
OsdD3D11ComputeKernelBundle::ApplyCatmarkVertexVerticesKernelB(
    int vertexOffset, int tableOffset, int start, int end) {

    KernelCB args;
    ZeroMemory(&args, sizeof(args));
    args.vertexOffset = vertexOffset;
    args.tableOffset = tableOffset;
    args.indexStart = start;
    args.indexEnd = end;
    dispatchCompute(_kernelComputeCatmarkVertexB, args);
}

void
OsdD3D11ComputeKernelBundle::ApplyCatmarkVertexVerticesKernelA(
    int vertexOffset, int tableOffset, int start, int end, bool pass) {

    KernelCB args;
    ZeroMemory(&args, sizeof(args));
    args.vertexOffset = vertexOffset;
    args.tableOffset = tableOffset;
    args.indexStart = start;
    args.indexEnd = end;
    args.vertexPass = pass ? 1 : 0;
    dispatchCompute(_kernelComputeVertexA, args);
}

void
OsdD3D11ComputeKernelBundle::ApplyLoopEdgeVerticesKernel(
    int vertexOffset, int tableOffset, int start, int end) {

    KernelCB args;
    ZeroMemory(&args, sizeof(args));
    args.vertexOffset = vertexOffset;
    args.tableOffset = tableOffset;
    args.indexStart = start;
    args.indexEnd = end;
    dispatchCompute(_kernelComputeEdge, args);
}

void
OsdD3D11ComputeKernelBundle::ApplyLoopVertexVerticesKernelB(
    int vertexOffset, int tableOffset, int start, int end) {

    KernelCB args;
    ZeroMemory(&args, sizeof(args));
    args.vertexOffset = vertexOffset;
    args.tableOffset = tableOffset;
    args.indexStart = start;
    args.indexEnd = end;
    dispatchCompute(_kernelComputeLoopVertexB, args);
}

void
OsdD3D11ComputeKernelBundle::ApplyLoopVertexVerticesKernelA(
    int vertexOffset, int tableOffset, int start, int end, bool pass) {

    KernelCB args;
    ZeroMemory(&args, sizeof(args));
    args.vertexOffset = vertexOffset;
    args.tableOffset = tableOffset;
    args.indexStart = start;
    args.indexEnd = end;
    args.vertexPass = pass ? 1 : 0;
    dispatchCompute(_kernelComputeVertexA, args);
}

void
OsdD3D11ComputeKernelBundle::ApplyEditAdd(
    int primvarOffset, int primvarWidth,
    int vertexOffset, int tableOffset, int start, int end) {

    KernelCB args;
    ZeroMemory(&args, sizeof(args));
    args.vertexOffset = vertexOffset;
    args.tableOffset = tableOffset;
    args.indexStart = start;
    args.indexEnd = end;
    args.editPrimVarOffset = primvarOffset;
    args.editPrimVarWidth = primvarWidth;
    dispatchCompute(_kernelEditAdd, args);
}

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
