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

#include "../osd/d3d11ComputeController.h"
#include "../osd/error.h"
#include "../osd/vertexDescriptor.h"

#define INITGUID        // for IID_ID3D11ShaderReflection
#include <D3D11.h>
#include <D3D11shader.h>
#include <D3Dcompiler.h>

#include <algorithm>
#include <cassert>
#include <sstream>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

#define SAFE_RELEASE(p) { if(p) { (p)->Release(); (p)=NULL; } }

static const char *shaderSource =
#include "../osd/hlslComputeKernel.gen.h"
;

// ----------------------------------------------------------------------------

// must match constant buffer declaration in hlslComputeKernel.hlsl
__declspec(align(16))

struct KernelUniformArgs {

    int uniformStart,     // batch
        uniformEnd,

        uniformOffset,    // primvar buffer descriptor
        uniformNumCVs;    // number of const control vertices padded at
};

// ----------------------------------------------------------------------------

class D3D11ComputeController::KernelBundle :
    NonCopyable<D3D11ComputeController::KernelBundle> {

public:

    KernelBundle() :
        _computeShader(0),
        _classLinkage(0),
        _subStencilKernel(0),
        _uniformArgs(0),
        _workGroupSize(64) { }

    ~KernelBundle() {
        SAFE_RELEASE(_computeShader);
        SAFE_RELEASE(_classLinkage);
        SAFE_RELEASE(_subStencilKernel);
        SAFE_RELEASE(_uniformArgs);
    }


    bool Compile(ID3D11DeviceContext *deviceContext,
        VertexBufferDescriptor const &desc) {

        _desc = VertexBufferDescriptor(0, desc.length, desc.stride);

        DWORD dwShaderFlags = D3DCOMPILE_ENABLE_STRICTNESS;
    #ifdef _DEBUG
        dwShaderFlags |= D3DCOMPILE_DEBUG;
    #endif

        std::ostringstream ss;

        ss << _desc.offset;   std::string offsetValue(ss.str()); ss.str("");
        ss << _desc.length;   std::string lengthValue(ss.str()); ss.str("");
        ss << _desc.stride;   std::string strideValue(ss.str()); ss.str("");
        ss << _workGroupSize; std::string workgroupSizeValue(ss.str()); ss.str("");

        D3D_SHADER_MACRO defines[] =
            { "OFFSET", offsetValue.c_str(),
              "LENGTH", lengthValue.c_str(),
              "STRIDE", strideValue.c_str(),
              "WORK_GROUP_SIZE", workgroupSizeValue.c_str(),
              0, 0 };

        ID3DBlob * computeShaderBuffer = NULL;
        ID3DBlob * errorBuffer = NULL;

        HRESULT hr = D3DCompile(shaderSource, strlen(shaderSource),
                                NULL, &defines[0], NULL,
                                "cs_main", "cs_5_0",
                                dwShaderFlags, 0,
                                &computeShaderBuffer, &errorBuffer);
        if (FAILED(hr)) {
            if (errorBuffer != NULL) {
                Error(OSD_D3D11_COMPILE_ERROR,
                         "Error compiling HLSL shader: %s\n",
                         (CHAR*)errorBuffer->GetBufferPointer());
                errorBuffer->Release();
                return false;
            }
        }

        ID3D11Device *device = NULL;
        deviceContext->GetDevice(&device);
        assert(device);

        device->CreateClassLinkage(&_classLinkage);
        assert(_classLinkage);

        device->CreateComputeShader(computeShaderBuffer->GetBufferPointer(),
                                    computeShaderBuffer->GetBufferSize(),
                                    _classLinkage,
                                    &_computeShader);
        assert(_computeShader);

        ID3D11ShaderReflection *reflector;
        D3DReflect(computeShaderBuffer->GetBufferPointer(),
                   computeShaderBuffer->GetBufferSize(),
                   IID_ID3D11ShaderReflection, (void**) &reflector);
        assert(reflector);

        assert(reflector->GetNumInterfaceSlots() == 1);
        reflector->Release();

        computeShaderBuffer->Release();

        _classLinkage->GetClassInstance("computeStencil", 0, &_subStencilKernel);
        assert(_subStencilKernel);

        return true;
    }

    void ApplyStencilTableKernel(ID3D11DeviceContext *deviceContext,
        Far::KernelBatch const &batch, int offset, int numCVs) {

        KernelUniformArgs args;
        args.uniformStart = batch.start;
        args.uniformEnd = batch.end;
        args.uniformOffset = offset;
        args.uniformNumCVs = numCVs;

        dispatchCompute(deviceContext, _subStencilKernel, args);
    }


    struct Match {

        Match(VertexBufferDescriptor const & d) : desc(d) { }

        bool operator() (KernelBundle const * kernel) {
            return (desc.length==kernel->_desc.length and
                    desc.stride==kernel->_desc.stride);
        }

        VertexBufferDescriptor desc;
    };

private:

    void dispatchCompute(ID3D11DeviceContext *deviceContext,
        ID3D11ClassInstance * kernel, KernelUniformArgs const & args) {

        assert(deviceContext);

        int count = args.uniformEnd - args.uniformStart;
        if (count <= 0) return;

        if (not _uniformArgs) {
            ID3D11Device *device = NULL;
            deviceContext->GetDevice(&device);
            assert(device);

            D3D11_BUFFER_DESC cbDesc;
            ZeroMemory(&cbDesc, sizeof(cbDesc));
            cbDesc.Usage = D3D11_USAGE_DYNAMIC;
            cbDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
            cbDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
            cbDesc.MiscFlags = 0;
            cbDesc.ByteWidth = sizeof(KernelUniformArgs);
            device->CreateBuffer(&cbDesc, NULL, &_uniformArgs);
        }
        assert(_uniformArgs);

        D3D11_MAPPED_SUBRESOURCE mappedResource;
        deviceContext->Map(_uniformArgs, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
        CopyMemory(mappedResource.pData, &args, sizeof(KernelUniformArgs));

        deviceContext->Unmap(_uniformArgs, 0);
        deviceContext->CSSetConstantBuffers(0, 1, &_uniformArgs); // b0

        deviceContext->CSSetShader(_computeShader, &kernel, 1);
        deviceContext->Dispatch(count/_workGroupSize + 1, 1, 1);
    }


private:


    ID3D11ComputeShader * _computeShader;

    ID3D11ClassLinkage * _classLinkage;

    ID3D11ClassInstance * _subStencilKernel; // stencil compute kernel HLSL subroutine

    ID3D11Buffer * _uniformArgs; // uniform paramaeters for kernels

    VertexBufferDescriptor _desc; // primvar buffer descriptor

    int _workGroupSize;
};

// ----------------------------------------------------------------------------
void
D3D11ComputeController::Synchronize() {

    if (not _query) {
        ID3D11Device *device = NULL;
        _deviceContext->GetDevice(&device);
        assert(device);

        D3D11_QUERY_DESC desc;
        desc.Query = D3D11_QUERY_EVENT;
        desc.MiscFlags = 0;
        device->CreateQuery(&desc, &_query);
    }
    _deviceContext->Flush();
    _deviceContext->End(_query);
    while (S_OK != _deviceContext->GetData(_query, NULL, 0, 0));
}

// ----------------------------------------------------------------------------

D3D11ComputeController::KernelBundle const *
D3D11ComputeController::getKernel(VertexBufferDescriptor const &desc) {

    KernelRegistry::iterator it =
        std::find_if(_kernelRegistry.begin(), _kernelRegistry.end(),
            KernelBundle::Match(desc));

    if (it != _kernelRegistry.end()) {
        return *it;
    } else {
        assert(_deviceContext);
        KernelBundle * kernelBundle = new KernelBundle();
        kernelBundle->Compile(_deviceContext, desc);
        _kernelRegistry.push_back(kernelBundle);
        return kernelBundle;
    }
}

void
D3D11ComputeController::bindBuffer() {

    // Unbind the vertexBuffer from the input assembler
    ID3D11Buffer *NULLBuffer = 0;
    UINT voffset = 0, vstride = 0;
    _deviceContext->IASetVertexBuffers(0, 1, &NULLBuffer, &voffset, &vstride);

    // Unbind the vertexBuffer from the vertex shader
    ID3D11ShaderResourceView *NULLSRV = 0;
    _deviceContext->VSSetShaderResources(0, 1, &NULLSRV);

    if (_currentBindState.buffer)
        _deviceContext->CSSetUnorderedAccessViews(0, 1, &_currentBindState.buffer, 0); // u0
}

void
D3D11ComputeController::unbindBuffer() {
    assert(_deviceContext);
    ID3D11UnorderedAccessView *UAViews[] = { 0 };
    _deviceContext->CSSetUnorderedAccessViews(0, 1, UAViews, 0); // u0
}

// ----------------------------------------------------------------------------

void
D3D11ComputeController::ApplyStencilTableKernel(
    Far::KernelBatch const &batch, D3D11ComputeContext const *context) const {

    assert(context);

    // XXXX manuelk messy const drop forced by D3D API - could use better solution
    D3D11ComputeController::KernelBundle * bundle =
        const_cast<D3D11ComputeController::KernelBundle *>(_currentBindState.kernelBundle);

    bundle->ApplyStencilTableKernel(_deviceContext,
        batch, _currentBindState.desc.offset, context->GetNumControlVertices());
}


// ----------------------------------------------------------------------------

D3D11ComputeController::D3D11ComputeController(
    ID3D11DeviceContext *deviceContext)
    : _deviceContext(deviceContext), _query(0) {
}

D3D11ComputeController::~D3D11ComputeController() {

    for (KernelRegistry::iterator it = _kernelRegistry.begin();
        it != _kernelRegistry.end(); ++it) {
        delete *it;
    }
    SAFE_RELEASE(_query);
}

}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
