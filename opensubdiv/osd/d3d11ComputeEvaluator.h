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

#ifndef OPENSUBDIV3_OSD_D3D11_COMPUTE_EVALUATOR_H
#define OPENSUBDIV3_OSD_D3D11_COMPUTE_EVALUATOR_H

#include "../version.h"

struct ID3D11DeviceContext;
struct ID3D11Buffer;
struct ID3D11ComputeShader;
struct ID3D11ClassLinkage;
struct ID3D11ClassInstance;
struct ID3D11ShaderResourceView;
struct ID3D11UnorderedAccessView;

#include "../osd/vertexDescriptor.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Far {
    class StencilTables;
}

namespace Osd {

/// \brief D3D11 stencil tables
///
/// This class is a D3D11 Shader Resource View representation of
/// Far::StencilTables.
///
/// D3D11ComputeEvaluator consumes this table to apply stencils
///
class D3D11StencilTables {
public:
    template <typename DEVICE_CONTEXT>
    static D3D11StencilTables *Create(Far::StencilTables const *stencilTables,
                                      DEVICE_CONTEXT context) {
        return new D3D11StencilTables(stencilTables, context->GetDeviceContext());
    }

    static D3D11StencilTables *Create(Far::StencilTables const *stencilTables,
                                      ID3D11DeviceContext *deviceContext) {
        return new D3D11StencilTables(stencilTables, deviceContext);
    }

    D3D11StencilTables(Far::StencilTables const *stencilTables,
                       ID3D11DeviceContext *deviceContext);

    ~D3D11StencilTables();

    // interfaces needed for D3D11ComputeEvaluator
    ID3D11ShaderResourceView *GetSizesSRV() const { return _sizes; }
    ID3D11ShaderResourceView *GetOffsetsSRV() const { return _offsets; }
    ID3D11ShaderResourceView *GetIndicesSRV() const { return _indices; }
    ID3D11ShaderResourceView *GetWeightsSRV() const { return _weights; }
    int GetNumStencils() const { return _numStencils; }

private:
    ID3D11ShaderResourceView *_sizes;
    ID3D11ShaderResourceView *_offsets;
    ID3D11ShaderResourceView *_indices;
    ID3D11ShaderResourceView *_weights;
    ID3D11Buffer *_sizesBuffer;
    ID3D11Buffer *_offsetsBuffer;
    ID3D11Buffer *_indicesBuffer;
    ID3D11Buffer *_weightsBuffer;

    int _numStencils;
};

// ---------------------------------------------------------------------------

class D3D11ComputeEvaluator {
public:
    typedef bool Instantiatable;
    static D3D11ComputeEvaluator * Create(VertexBufferDescriptor const &srcDesc,
                                          VertexBufferDescriptor const &dstDesc,
                                          ID3D11DeviceContext *deviceContext);

    /// Constructor.
    D3D11ComputeEvaluator();

    /// Destructor.
    ~D3D11ComputeEvaluator();

    /// \brief Generic static compute function. This function has a same
    ///        signature as other device kernels have so that it can be called
    ///        transparently from OsdMesh template interface.
    ///
    /// @param srcBuffer      Input primvar buffer.
    ///                       must have BindVBO() method returning a
    ///                       const float pointer for read
    ///
    /// @param srcDesc        vertex buffer descriptor for the input buffer
    ///
    /// @param dstBuffer      Output primvar buffer
    ///                       must have BindVBO() method returning a
    ///                       float pointer for write
    ///
    /// @param dstDesc        vertex buffer descriptor for the output buffer
    ///
    /// @param stencilTables  stencil table to be applied. The table must have
    ///                       SSBO interfaces.
    ///
    /// @param instance       cached compiled instance. Clients are supposed to
    ///                       pre-compile an instance of this class and provide
    ///                       to this function. If it's null the kernel still
    ///                       compute by instantiating on-demand kernel although
    ///                       it may cause a performance problem.
    ///
    /// @param deviceContext  ID3D11DeviceContext.
    ///
    template <typename VERTEX_BUFFER, typename STENCIL_TABLE>
    static bool EvalStencils(VERTEX_BUFFER *srcVertexBuffer,
                             VertexBufferDescriptor const &srcDesc,
                             VERTEX_BUFFER *dstVertexBuffer,
                             VertexBufferDescriptor const &dstDesc,
                             STENCIL_TABLE const *stencilTable,
                             D3D11ComputeEvaluator const *instance,
                             ID3D11DeviceContext * deviceContext) {
        if (instance) {
            return instance->EvalStencils(srcVertexBuffer, srcDesc,
                                          dstVertexBuffer, dstDesc,
                                          stencilTable,
                                          deviceContext);
        } else {
            // Create an instace on demand (slow)
            (void)deviceContext;  // unused
            instance = Create(srcDesc, dstDesc, deviceContext);
            if (instance) {
                bool r = instance->EvalStencils(srcVertexBuffer, srcDesc,
                                                dstVertexBuffer, dstDesc,
                                                stencilTable,
                                                deviceContext);
                delete instance;
                return r;
            }
            return false;
        }
    }

    /// Dispatch the DX compute kernel on GPU asynchronously.
    /// returns false if the kernel hasn't been compiled yet.
    template <typename VERTEX_BUFFER, typename STENCIL_TABLE>
    bool EvalStencils(VERTEX_BUFFER *srcVertexBuffer,
                      VertexBufferDescriptor const &srcDesc,
                      VERTEX_BUFFER *dstVertexBuffer,
                      VertexBufferDescriptor const &dstDesc,
                      STENCIL_TABLE const *stencilTable,
                      ID3D11DeviceContext *deviceContext) const {
        return EvalStencils(srcVertexBuffer->BindD3D11UAV(deviceContext),
                            srcDesc,
                            dstVertexBuffer->BindD3D11UAV(deviceContext),
                            dstDesc,
                            stencilTable->GetSizesSRV(),
                            stencilTable->GetOffsetsSRV(),
                            stencilTable->GetIndicesSRV(),
                            stencilTable->GetWeightsSRV(),
                            /* start = */ 0,
                            /* end   = */ stencilTable->GetNumStencils(),
                            deviceContext);
    }

    /// Dispatch the DX compute kernel on GPU asynchronously.
    /// returns false if the kernel hasn't been compiled yet.
    bool EvalStencils(ID3D11UnorderedAccessView *srcSRV,
                      VertexBufferDescriptor const &srcDesc,
                      ID3D11UnorderedAccessView *dstUAV,
                      VertexBufferDescriptor const &dstDesc,
                      ID3D11ShaderResourceView *sizesSRV,
                      ID3D11ShaderResourceView *offsetsSRV,
                      ID3D11ShaderResourceView *indicesSRV,
                      ID3D11ShaderResourceView *weightsSRV,
                      int start,
                      int end,
                      ID3D11DeviceContext *deviceContext) const;

    /// Configure DX kernel. Returns false if it fails to compile the kernel.
    bool Compile(VertexBufferDescriptor const &srcDesc,
                 VertexBufferDescriptor const &dstDesc,
                 ID3D11DeviceContext *deviceContext);

    /// Wait the dispatched kernel finishes.
    static void Synchronize(ID3D11DeviceContext *deviceContext);

private:
    ID3D11ComputeShader * _computeShader;
    ID3D11ClassLinkage  * _classLinkage;
    ID3D11ClassInstance * _singleBufferKernel;
    ID3D11ClassInstance * _separateBufferKernel;
    ID3D11Buffer        * _uniformArgs; // uniform paramaeters for kernels

    int _workGroupSize;
};

}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv


#endif  // OPENSUBDIV3_OSD_D3D11_COMPUTE_EVALUATOR_H
