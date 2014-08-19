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

#ifndef OSD_GLSL_TRANSFORM_FEEDBACK_KERNEL_BUNDLE_H
#define OSD_GLSL_TRANSFORM_FEEDBACK_KERNEL_BUNDLE_H

#include "../version.h"

#include "../osd/nonCopyable.h"
#include "../osd/vertex.h"
#include "../osd/vertexDescriptor.h"
#include "../far/subdivisionTables.h"

#include "../osd/opengl.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

class OsdGLSLTransformFeedbackKernelBundle : OsdNonCopyable<OsdGLSLTransformFeedbackKernelBundle> {
public:
    /// Constructor
    OsdGLSLTransformFeedbackKernelBundle();
    
    ~OsdGLSLTransformFeedbackKernelBundle();

    bool Compile(OsdVertexBufferDescriptor const &vertexDesc,
                 OsdVertexBufferDescriptor const &varyingDesc,
                 bool interleaved);

    void ApplyBilinearFaceVerticesKernel(
        GLuint vertexBuffer, GLuint varyingBuffer,
        int vertexOffset, int varyingOffset,
        int offset, int tableOffset, int start, int end);

    void ApplyBilinearEdgeVerticesKernel(
        GLuint vertexBuffer, GLuint varyingBuffer,
        int vertexOffset, int varyingOffset,
        int offset, int tableOffset, int start, int end);

    void ApplyBilinearVertexVerticesKernel(
        GLuint vertexBuffer, GLuint varyingBuffer,
        int vertexOffset, int varyingOffset,
        int offset, int tableOffset, int start, int end);

    void ApplyCatmarkFaceVerticesKernel(
        GLuint vertexBuffer, GLuint varyingBuffer,
        int vertexOffset, int varyingOffset,
        int offset, int tableOffset, int start, int end);

    void ApplyCatmarkQuadFaceVerticesKernel(
        GLuint vertexBuffer, GLuint varyingBuffer,
        int vertexOffset, int varyingOffset,
        int offset, int tableOffset, int start, int end);

    void ApplyCatmarkTriQuadFaceVerticesKernel(
        GLuint vertexBuffer, GLuint varyingBuffer,
        int vertexOffset, int varyingOffset,
        int offset, int tableOffset, int start, int end);

    void ApplyCatmarkEdgeVerticesKernel(
        GLuint vertexBuffer, GLuint varyingBuffer,
        int vertexOffset, int varyingOffset,
        int offset, int tableOffset, int start, int end);

    void ApplyCatmarkRestrictedEdgeVerticesKernel(
        GLuint vertexBuffer, GLuint varyingBuffer,
        int vertexOffset, int varyingOffset,
        int offset, int tableOffset, int start, int end);

    void ApplyCatmarkVertexVerticesKernelB(
        GLuint vertexBuffer, GLuint varyingBuffer,
        int vertexOffset, int varyingOffset,
        int offset, int tableOffset, int start, int end);

    void ApplyCatmarkVertexVerticesKernelA(
        GLuint vertexBuffer, GLuint varyingBuffer,
        int vertexOffset, int varyingOffset,
        int offset, int tableOffset, int start, int end, bool pass);

    void ApplyCatmarkRestrictedVertexVerticesKernelB1(
        GLuint vertexBuffer, GLuint varyingBuffer,
        int vertexOffset, int varyingOffset,
        int offset, int tableOffset, int start, int end);

    void ApplyCatmarkRestrictedVertexVerticesKernelB2(
        GLuint vertexBuffer, GLuint varyingBuffer,
        int vertexOffset, int varyingOffset,
        int offset, int tableOffset, int start, int end);

    void ApplyCatmarkRestrictedVertexVerticesKernelA(
        GLuint vertexBuffer, GLuint varyingBuffer,
        int vertexOffset, int varyingOffset,
        int offset, int tableOffset, int start, int end);

    void ApplyLoopEdgeVerticesKernel(
        GLuint vertexBuffer, GLuint varyingBuffer,
        int vertexOffset, int varyingOffset,
        int offset, int tableOffset, int start, int end);

    void ApplyLoopVertexVerticesKernelB(
        GLuint vertexBuffer, GLuint varyingBuffer,
        int vertexOffset, int varyingOffset,
        int offset, int tableOffset, int start, int end);

    void ApplyLoopVertexVerticesKernelA(
        GLuint vertexBuffer, GLuint varyingBuffer,
        int vertexOffset, int varyingOffset,
        int offset, int tableOffset, int start, int end, bool pass);

    void ApplyEditAdd(
        GLuint vertexBuffer, GLuint varyingBuffer,
        int vertexOffset, int varyingOffset,
        int primvarOffset, int primvarWidth,
        int offset, int tableOffset, int start, int end);

    void UseProgram(int vertexBaseOffset, int varyingBaseOffset) const;

    GLint GetTableUniformLocation(int tableIndex) const {
        return _uniformTables[tableIndex];
    }
    GLint GetVertexUniformLocation() const {
        return _uniformVertexBuffer;
    }
    GLint GetVaryingUniformLocation() const {
        return _uniformVaryingBuffer;
    }
    GLint GetEditIndicesUniformLocation() const {
        return _uniformEditIndices;
    }
    GLint GetEditValuesUniformLocation() const {
        return _uniformEditValues;
    }
    GLint GetVertexBufferImageUniformLocation() const {
        return _uniformVertexBufferImage;
    }

    struct Match {
        /// Constructor
        Match(OsdVertexBufferDescriptor const &vertex,
              OsdVertexBufferDescriptor const &varying,
              bool interleaved)
            : vertexDesc(vertex), varyingDesc(varying), interleaved(interleaved) {
        }

        bool operator() (OsdGLSLTransformFeedbackKernelBundle const *kernel) {
            // offset is dynamic. just comparing length and stride here,
            // returns true if they are equal
            return (vertexDesc.length == kernel->_numVertexElements and
                    vertexDesc.stride == kernel->_vertexStride and
                    varyingDesc.length == kernel->_numVaryingElements and
                    varyingDesc.stride == kernel->_varyingStride and
                    interleaved == kernel->_interleaved);
        }

        OsdVertexBufferDescriptor vertexDesc;
        OsdVertexBufferDescriptor varyingDesc;
        bool interleaved;
    };

    friend struct Match;

protected:
    void transformGpuBufferData(
        GLuint vertexBuffer, GLuint varyingBuffer,
        int vertexOffset, int varyingOffset,
        int offset, int tableOffset, int start, int end) const;

    GLuint _program;

    // uniform locations
    GLint _uniformTables[FarSubdivisionTables::TABLE_TYPES_COUNT];
    GLint _uniformVertexPass;
    GLint _uniformVertexOffset;
    GLint _uniformTableOffset;
    GLint _uniformIndexStart;
    GLint _uniformVertexBaseOffset;
    GLint _uniformVaryingBaseOffset;

    GLint _uniformVertexBuffer;
    GLint _uniformVaryingBuffer;

    GLint _uniformEditPrimVarOffset;
    GLint _uniformEditPrimVarWidth;

    GLint _uniformEditIndices;
    GLint _uniformEditValues;
    GLint _uniformVertexBufferImage;

    // subroutines

    GLuint _subComputeFace; // general face-vertex kernel (all schemes)

    GLuint _subComputeQuadFace; // quad face-vertex kernel (catmark scheme)

    GLuint _subComputeTriQuadFace; // tri-quad face-vertex kernel (catmark scheme)

    GLuint _subComputeEdge; // edge-vertex kernel (catmark + loop schemes)

    GLuint _subComputeRestrictedEdge; // restricted edge-vertex kernel (catmark scheme)

    GLuint _subComputeBilinearEdge; // edge-vertex kernel (bilinear scheme)

    GLuint _subComputeVertex; // vertex-vertex kernel (bilinear scheme)

    GLuint _subComputeVertexA; // vertex-vertex kernel A (catmark + loop schemes)

    GLuint _subComputeCatmarkVertexB;// vertex-vertex kernel B (catmark scheme)

    GLuint _subComputeRestrictedVertexA; // restricted vertex-vertex kernel A (catmark scheme)

    GLuint _subComputeRestrictedVertexB1; // restricted vertex-vertex kernel B1 (catmark scheme)

    GLuint _subComputeRestrictedVertexB2; // restricted vertex-vertex kernel B2 (catmark scheme)

    GLuint _subComputeLoopVertexB; // vertex-vertex kernel B (loop scheme)

    GLuint _subEditAdd; // hedit kernel (add)

    // kernelbundle discriminators
    int _numVertexElements;
    int _vertexStride;
    int _numVaryingElements;
    int _varyingStride;
    int _vertexOffsetMod;
    int _varyingOffsetMod;
    bool _interleaved;
};

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OSD_GLSL_TRANSFORM_FEEDBACK_KERNEL_BUNDLE_H
