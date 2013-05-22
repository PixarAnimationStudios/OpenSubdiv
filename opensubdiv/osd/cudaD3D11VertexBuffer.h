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
