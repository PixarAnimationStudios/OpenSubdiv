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
#ifndef OSD_CUDA_D3D11_VERTEX_BUFFER_H
#define OSD_CUDA_D3D11_VERTEX_BUFFER_H

#include "../version.h"

struct cudaGraphicsResource;

struct ID3D11Buffer;
struct ID3D11Device;
struct ID3D11DeviceContext;

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

/// \brief Concrete vertex buffer class for cuda subvision and D3D11 drawing.
///
/// OsdCudaD3D11VertexBuffer implements OsdCudaVertexBufferInterface and
/// OsdD3D11VertexBufferInterface.
///
/// The buffer interop between Cuda and D3D is handled automatically when a 
/// client calls BindCudaBuffer and BindVBO methods.
///
class OsdCudaD3D11VertexBuffer {
public:
    /// Creator. Returns NULL if error.
    static OsdCudaD3D11VertexBuffer * Create(int numElements, 
                                             int numVertices, 
                                             ID3D11Device *device);

    /// Destructor.
    virtual ~OsdCudaD3D11VertexBuffer();

    /// This method is meant to be used in client code in order to provide coarse
    /// vertices data to Osd.
    void UpdateData(const float *src, int startVertex, int numVertices, void *param);

    /// Returns how many elements defined in this vertex buffer.
    int GetNumElements() const;

    /// Returns how many vertices allocated in this vertex buffer.
    int GetNumVertices() const;

    /// Returns cuda memory. DX buffer will be mapped to cuda resource
    /// if necessary.
    float * BindCudaBuffer();

    /// Returns the D3D11 buffer object.
    ID3D11Buffer *BindD3D11Buffer(ID3D11DeviceContext *deviceContext);

protected:
    /// Constructor.
    OsdCudaD3D11VertexBuffer(int numElements, 
                             int numVertices,
                             ID3D11Device *device);

    bool allocate(ID3D11Device *device);

    // Acqures a cuda resource from DX11
    void map();

    // Releases a cuda resource to DX11
    void unmap();

private:
    int _numElements;
    int _numVertices;
    ID3D11Buffer *_d3d11Buffer;
    void *_cudaBuffer;
    cudaGraphicsResource *_cudaResource;
};

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OSD_CUDA_D3D11_VERTEX_BUFFER_H
