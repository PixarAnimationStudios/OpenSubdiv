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

    bool Compile(int numVertexElements, int numVaryingElements);

    void ApplyBilinearFaceVerticesKernel(
        int vertexOffset, int tableOffset, int start, int end);

    void ApplyBilinearEdgeVerticesKernel(
        int vertexOffset, int tableOffset, int start, int end);

    void ApplyBilinearVertexVerticesKernel(
        int vertexOffset, int tableOffset, int start, int end);

    void ApplyCatmarkFaceVerticesKernel(
        int vertexOffset, int tableOffset, int start, int end);

    void ApplyCatmarkEdgeVerticesKernel(
        int vertexOffset, int tableOffset, int start, int end);

    void ApplyCatmarkVertexVerticesKernelB(
        int vertexOffset, int tableOffset, int start, int end);

    void ApplyCatmarkVertexVerticesKernelA(
        int vertexOffset, int tableOffset, int start, int end, bool pass);

    void ApplyLoopEdgeVerticesKernel(
        int vertexOffset, int tableOffset, int start, int end);

    void ApplyLoopVertexVerticesKernelB(
        int vertexOffset, int tableOffset, int start, int end);

    void ApplyLoopVertexVerticesKernelA(
        int vertexOffset, int tableOffset, int start, int end, bool pass);

    void ApplyEditAdd(int primvarOffset, int primvarWidth,
                      int vertexOffset, int tableOffset, int start, int end);

    struct Match {

        /// Constructor
        Match(int numVertexElements, int numVaryingElements)
            : vdesc(numVertexElements, numVaryingElements) {
        }

        bool operator() (OsdD3D11ComputeKernelBundle const *kernel) {
            return vdesc == kernel->_vdesc;
        }

        OsdVertexDescriptor vdesc;
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

    ID3D11ClassInstance * _kernelComputeEdge; // edge-vertex kernel (catmark + loop schemes)

    ID3D11ClassInstance * _kernelComputeBilinearEdge; // edge-vertex kernel (bilinear scheme)

    ID3D11ClassInstance * _kernelComputeVertex; // vertex-vertex kernel (bilinear scheme)

    ID3D11ClassInstance * _kernelComputeVertexA; // vertex-vertex kernel A (catmark + loop schemes)

    ID3D11ClassInstance * _kernelComputeCatmarkVertexB; // vertex-vertex kernel B (catmark scheme)

    ID3D11ClassInstance * _kernelComputeLoopVertexB; // vertex-vertex kernel B (loop scheme)

    ID3D11ClassInstance * _kernelEditAdd; // hedit kernel (add)

    int _workGroupSize;

    OsdVertexDescriptor _vdesc;
};

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OSD_D3D11_COMPUTE_KERNEL_BUNDLE_H
