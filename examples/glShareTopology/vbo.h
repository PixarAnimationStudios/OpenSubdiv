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

#ifndef OPENSUBDIV_EXAMPLES_GL_SHARE_TOPOLOGY_VBO_H
#define OPENSUBDIV_EXAMPLES_GL_SHARE_TOPOLOGY_VBO_H

#include "glLoader.h"

#include <opensubdiv/osd/bufferDescriptor.h>

#include <vector>

template <class VERTEX_BUFFER, class DEVICE_CONTEXT>
class VBO {
public:
    VBO(OpenSubdiv::Osd::BufferDescriptor const &vertexDesc,
        OpenSubdiv::Osd::BufferDescriptor const &varyingDesc,
        bool interleaved, int numVertices, DEVICE_CONTEXT *deviceContext) :
        _vertexDesc(vertexDesc),
        _varyingDesc(varyingDesc),
        _numVertices(numVertices),
        _vertexBuffer(NULL), _varyingBuffer(NULL), _interleaved(interleaved),
        _deviceContext(deviceContext) {

        if (interleaved) {
            assert(vertexDesc.stride == varyingDesc.stride);
            _vertexBuffer = createVertexBuffer(
                vertexDesc.stride, numVertices);
        } else {
            if (vertexDesc.stride > 0) {
                _vertexBuffer = createVertexBuffer(
                    vertexDesc.stride, numVertices);
            }
            if (varyingDesc.stride > 0) {
                _varyingBuffer = createVertexBuffer(
                    varyingDesc.stride, numVertices);
            }
        }
    }

    ~VBO() {
        delete _vertexBuffer;
        delete _varyingBuffer;
    }

    OpenSubdiv::Osd::BufferDescriptor const &GetVertexDesc() const {
        return _vertexDesc;
    }
    OpenSubdiv::Osd::BufferDescriptor const &GetVaryingDesc() const {
        return _varyingDesc;
    }

    void UpdateVertexBuffer(int vertsOffset, std::vector<float> const &src) {
        updateVertexBuffer(_vertexBuffer, &src[0], vertsOffset,
                           (int)src.size()/_vertexBuffer->GetNumElements());
    }
    void UpdateVaryingBuffer(int vertsOffset, std::vector<float> const &src) {
        updateVertexBuffer(_varyingBuffer, &src[0], vertsOffset,
                           (int)src.size()/_varyingBuffer->GetNumElements());
    }

    GLuint BindVertexBuffer() {
        return _vertexBuffer->BindVBO();
    }

    GLuint BindVaryingBuffer() {
        return _varyingBuffer->BindVBO();
    }

    VERTEX_BUFFER *GetVertexBuffer() const {
        return _vertexBuffer;
    }

    VERTEX_BUFFER *GetVaryingBuffer() const {
        return _interleaved ? _vertexBuffer : _varyingBuffer;
    }

    size_t GetSize() const {
        size_t size = _numVertices * _vertexDesc.stride;
        if (_varyingBuffer) size += _numVertices * _varyingDesc.stride;
        return size * sizeof(float);
    }

private:
    VERTEX_BUFFER *createVertexBuffer(int numElements, int numVertices) {
        return VERTEX_BUFFER::Create(numElements, numVertices, _deviceContext);
    }

    void updateVertexBuffer(VERTEX_BUFFER *vertexBuffer,
                            const float *src, int startVertex,
                            int numVertices) {
        vertexBuffer->UpdateData(src, startVertex, numVertices, _deviceContext);
    }

    OpenSubdiv::Osd::BufferDescriptor _vertexDesc;
    OpenSubdiv::Osd::BufferDescriptor _varyingDesc;

    // # of vertices total, including both control verts and refined verts.
    int _numVertices;

    VERTEX_BUFFER *_vertexBuffer;
    VERTEX_BUFFER *_varyingBuffer;
    bool _interleaved;
    DEVICE_CONTEXT *_deviceContext;
};

#endif  //  OPENSUBDIV_EXAMPLES_GL_SHARE_TOPOLOGY_VBO_H
