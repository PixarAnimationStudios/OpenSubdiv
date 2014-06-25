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

#ifndef OSD_D3D11_COMPUTE_KERNEL_BUNDLE_H
#define OSD_D3D11_COMPUTE_KERNEL_BUNDLE_H

#include "../version.h"

#include "../osd/nonCopyable.h"
#include "../osd/vertexDescriptor.h"

struct ID3D11Buffer;
struct ID3D11ClassInstance;
struct ID3D11ClassLinkage;
struct ID3D11ComputeShader;
struct ID3D11DeviceContext;

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

class OsdD3D11ComputeKernelBundle : OsdNonCopyable<OsdD3D11ComputeKernelBundle> {

public:
    /// Constructor
    OsdD3D11ComputeKernelBundle(ID3D11DeviceContext *deviceContext);
    
    /// Destructor
    ~OsdD3D11ComputeKernelBundle();

    bool Compile(OsdVertexBufferDescriptor const &vertexDesc,
                 OsdVertexBufferDescriptor const &varyingDesc);

    void ApplyBilinearFaceVerticesKernel(
        int vertexOffset, int tableOffset, int start, int end,
        int vertexBaseOffset, int varyingBaseOffset);

    void ApplyBilinearEdgeVerticesKernel(
        int vertexOffset, int tableOffset, int start, int end,
        int vertexBaseOffset, int varyingBaseOffset);

    void ApplyBilinearVertexVerticesKernel(
        int vertexOffset, int tableOffset, int start, int end,
        int vertexBaseOffset, int varyingBaseOffset);

    void ApplyCatmarkFaceVerticesKernel(
        int vertexOffset, int tableOffset, int start, int end,
        int vertexBaseOffset, int varyingBaseOffset);

    void ApplyCatmarkQuadFaceVerticesKernel(
        int vertexOffset, int tableOffset, int start, int end,
        int vertexBaseOffset, int varyingBaseOffset);

    void ApplyCatmarkTriQuadFaceVerticesKernel(
        int vertexOffset, int tableOffset, int start, int end,
        int vertexBaseOffset, int varyingBaseOffset);

    void ApplyCatmarkEdgeVerticesKernel(
        int vertexOffset, int tableOffset, int start, int end,
        int vertexBaseOffset, int varyingBaseOffset);

    void ApplyCatmarkRestrictedEdgeVerticesKernel(
        int vertexOffset, int tableOffset, int start, int end,
        int vertexBaseOffset, int varyingBaseOffset);

    void ApplyCatmarkVertexVerticesKernelB(
        int vertexOffset, int tableOffset, int start, int end,
        int vertexBaseOffset, int varyingBaseOffset);

    void ApplyCatmarkVertexVerticesKernelA(
        int vertexOffset, int tableOffset, int start, int end, bool pass,
        int vertexBaseOffset, int varyingBaseOffset);

    void ApplyCatmarkRestrictedVertexVerticesKernelB1(
        int vertexOffset, int tableOffset, int start, int end,
        int vertexBaseOffset, int varyingBaseOffset);

    void ApplyCatmarkRestrictedVertexVerticesKernelB2(
        int vertexOffset, int tableOffset, int start, int end,
        int vertexBaseOffset, int varyingBaseOffset);

    void ApplyCatmarkRestrictedVertexVerticesKernelA(
        int vertexOffset, int tableOffset, int start, int end,
        int vertexBaseOffset, int varyingBaseOffset);

    void ApplyLoopEdgeVerticesKernel(
        int vertexOffset, int tableOffset, int start, int end,
        int vertexBaseOffset, int varyingBaseOffset);

    void ApplyLoopVertexVerticesKernelB(
        int vertexOffset, int tableOffset, int start, int end,
        int vertexBaseOffset, int varyingBaseOffset);

    void ApplyLoopVertexVerticesKernelA(
        int vertexOffset, int tableOffset, int start, int end, bool pass,
        int vertexBaseOffset, int varyingBaseOffset);

    void ApplyEditAdd(int primvarOffset, int primvarWidth,
                      int vertexOffset, int tableOffset, int start, int end,
                      int vertexBaseOffset, int varyingBaseOffset);

    struct Match {
        /// Constructor
        Match(OsdVertexBufferDescriptor const &vertex,
              OsdVertexBufferDescriptor const &varying)
            : vertexDesc(vertex), varyingDesc(varying) {
        }

        bool operator() (OsdD3D11ComputeKernelBundle const *kernel) {
            // offset is dynamic. just comparing length and stride here,
            // returns true if they are equal
            return (vertexDesc.length == kernel->_numVertexElements and
                    vertexDesc.stride == kernel->_vertexStride and
                    varyingDesc.length == kernel->_numVaryingElements and
                    varyingDesc.stride == kernel->_varyingStride);
        }

        OsdVertexBufferDescriptor vertexDesc;
        OsdVertexBufferDescriptor varyingDesc;
    };

    friend struct Match;

protected:
    struct KernelCB;
    void dispatchCompute(ID3D11ClassInstance * kernel, KernelCB const & args);

    ID3D11DeviceContext * _deviceContext;

    ID3D11ComputeShader * _computeShader;
    ID3D11ClassLinkage * _classLinkage;


    ID3D11Buffer * _kernelCB; // constant buffer


    ID3D11ClassInstance * _kernelComputeFace; // general face-vertex kernel (all schemes)

    ID3D11ClassInstance * _kernelComputeQuadFace; // quad face-vertex kernel (catmark scheme)

    ID3D11ClassInstance * _kernelComputeTriQuadFace; // tri-quad face-vertex kernel (catmark scheme)

    ID3D11ClassInstance * _kernelComputeEdge; // edge-vertex kernel (catmark + loop schemes)

    ID3D11ClassInstance * _kernelComputeRestrictedEdge; // edge-vertex kernel (catmark scheme)

    ID3D11ClassInstance * _kernelComputeBilinearEdge; // edge-vertex kernel (bilinear scheme)

    ID3D11ClassInstance * _kernelComputeVertex; // vertex-vertex kernel (bilinear scheme)

    ID3D11ClassInstance * _kernelComputeVertexA; // vertex-vertex kernel A (catmark + loop schemes)

    ID3D11ClassInstance * _kernelComputeCatmarkVertexB; // vertex-vertex kernel B (catmark scheme)

    ID3D11ClassInstance * _kernelComputeCatmarkRestrictedVertexA; // restricted vertex-vertex kernel A (catmark scheme)

    ID3D11ClassInstance * _kernelComputeCatmarkRestrictedVertexB1; // restricted vertex-vertex kernel B1 (catmark scheme)

    ID3D11ClassInstance * _kernelComputeCatmarkRestrictedVertexB2; // restricted vertex-vertex kernel B2 (catmark scheme)

    ID3D11ClassInstance * _kernelComputeLoopVertexB; // vertex-vertex kernel B (loop scheme)

    ID3D11ClassInstance * _kernelEditAdd; // hedit kernel (add)

    int _workGroupSize;

    int _numVertexElements;
    int _vertexStride;
    int _numVaryingElements;
    int _varyingStride;
};

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OSD_D3D11_COMPUTE_KERNEL_BUNDLE_H
