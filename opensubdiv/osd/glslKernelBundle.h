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

#ifndef OSD_GLSL_COMPUTE_KERNEL_BUNDLE_H
#define OSD_GLSL_COMPUTE_KERNEL_BUNDLE_H

#include "../version.h"

#include "../far/subdivisionTables.h"
#include "../osd/nonCopyable.h"
#include "../osd/vertex.h"
#include "../osd/vertexDescriptor.h"

#include "../osd/opengl.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

class OsdGLSLComputeKernelBundle : OsdNonCopyable<OsdGLSLComputeKernelBundle> {
public:
    OsdGLSLComputeKernelBundle();
    ~OsdGLSLComputeKernelBundle();

    bool Compile(OsdVertexBufferDescriptor const &vertexDesc,
                 OsdVertexBufferDescriptor const &varyingDesc);

    void ApplyBilinearFaceVerticesKernel(
        int vertexOffset, int tableOffset, int start, int end);

    void ApplyBilinearEdgeVerticesKernel(
        int vertexOffset, int tableOffset, int start, int end);

    void ApplyBilinearVertexVerticesKernel(
        int vertexOffset, int tableOffset, int start, int end);

    void ApplyCatmarkFaceVerticesKernel(
        int vertexOffset, int tableOffset, int start, int end);

    void ApplyCatmarkQuadFaceVerticesKernel(
        int vertexOffset, int tableOffset, int start, int end);

    void ApplyCatmarkTriQuadFaceVerticesKernel(
        int vertexOffset, int tableOffset, int start, int end);

    void ApplyCatmarkEdgeVerticesKernel(
        int vertexOffset, int tableOffset, int start, int end);

    void ApplyCatmarkRestrictedEdgeVerticesKernel(
        int vertexOffset, int tableOffset, int start, int end);

    void ApplyCatmarkVertexVerticesKernelB(
        int vertexOffset, int tableOffset, int start, int end);

    void ApplyCatmarkVertexVerticesKernelA(
        int vertexOffset, int tableOffset, int start, int end, bool pass);

    void ApplyLoopEdgeVerticesKernel(
        int vertexOffset, int tableOffset, int start, int end);

    void ApplyCatmarkRestrictedVertexVerticesKernelB1(
        int vertexOffset, int tableOffset, int start, int end);

    void ApplyCatmarkRestrictedVertexVerticesKernelB2(
        int vertexOffset, int tableOffset, int start, int end);

    void ApplyCatmarkRestrictedVertexVerticesKernelA(
        int vertexOffset, int tableOffset, int start, int end);

    void ApplyLoopVertexVerticesKernelB(
        int vertexOffset, int tableOffset, int start, int end);

    void ApplyLoopVertexVerticesKernelA(
        int vertexOffset, int tableOffset, int start, int end, bool pass);

    void ApplyEditAdd(int primvarOffset, int primvarWidth,
                      int vertexOffset, int tableOffset,
                      int start, int end);

    void UseProgram(int vertexBaseOffset, int varyingBaseOffset) const;

    GLuint GetTableUniformLocation(int tableIndex) const {
        return _tableUniforms[tableIndex];
    }

    struct Match {
        /// Constructor
        Match(OsdVertexBufferDescriptor const &vertex,
              OsdVertexBufferDescriptor const &varying)
            : vertexDesc(vertex), varyingDesc(varying) {
        }

        bool operator() (OsdGLSLComputeKernelBundle const *kernel) {
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
    void dispatchCompute(int vertexOffset, int tableOffset,
                         int start, int end) const ;

    GLuint _program;

    // uniform locations for compute
    GLuint _tableUniforms[FarSubdivisionTables::TABLE_TYPES_COUNT];
    GLuint _uniformVertexPass;
    GLuint _uniformVertexOffset;
    GLuint _uniformTableOffset;
    GLuint _uniformIndexStart;
    GLuint _uniformIndexEnd;
    GLuint _uniformVertexBaseOffset;
    GLuint _uniformVaryingBaseOffset;

    // uniform locations for vertex edit
    GLuint _uniformEditPrimVarOffset;
    GLuint _uniformEditPrimVarWidth;

    
    GLuint _subComputeFace; // general face-vertex kernel (all schemes)

    GLuint _subComputeQuadFace; // quad face-vertex kernel (catmark scheme)

    GLuint _subComputeTriQuadFace; // tri-quad face-vertex kernel (catmark scheme)

    GLuint _subComputeEdge; // edge-vertex kernel (catmark + loop schemes)

    GLuint _subComputeRestrictedEdge; // restricted edge-vertex kernel (catmark scheme)

    GLuint _subComputeRestrictedVertexA; // restricted vertex-vertex kernel A (catmark scheme)

    GLuint _subComputeRestrictedVertexB1; // restricted vertex-vertex kernel B1 (catmark scheme)

    GLuint _subComputeRestrictedVertexB2; // restricted vertex-vertex kernel B2 (catmark scheme)

    GLuint _subComputeBilinearEdge; // edge-vertex kernel (bilinear scheme)

    GLuint _subComputeVertex; // vertex-vertex kernel (bilinear scheme)

    GLuint _subComputeVertexA; // vertex-vertex kernel A (catmark + loop schemes)

    GLuint _subComputeCatmarkVertexB; // vertex-vertex kernel B (catmark scheme)

    GLuint _subComputeLoopVertexB; // vertex-vertex kernel B (loop scheme)

    GLuint _subEditAdd; // hedit kernel (add)

    int _workGroupSize;

    int _numVertexElements;
    int _vertexStride;
    int _numVaryingElements;
    int _varyingStride;
};

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OSD_GLSL_COMPUTE_KERNEL_BUNDLE_H
