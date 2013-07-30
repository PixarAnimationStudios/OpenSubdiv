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

    bool Compile(int numVertexElements, int numVaryingElements);

    void ApplyBilinearFaceVerticesKernel(
        GLuint vertexBuffer, int numVertexElements,
        GLuint varyingBuffer, int numVaryingElements,
        int vertexOffset, int tableOffset, int start, int end);

    void ApplyBilinearEdgeVerticesKernel(
        GLuint vertexBuffer, int numVertexElements,
        GLuint varyingBuffer, int numVaryingElements,
        int vertexOffset, int tableOffset, int start, int end);

    void ApplyBilinearVertexVerticesKernel(
        GLuint vertexBuffer, int numVertexElements,
        GLuint varyingBuffer, int numVaryingElements,
        int vertexOffset, int tableOffset, int start, int end);

    void ApplyCatmarkFaceVerticesKernel(
        GLuint vertexBuffer, int numVertexElements,
        GLuint varyingBuffer, int numVaryingElements,
        int vertexOffset, int tableOffset, int start, int end);

    void ApplyCatmarkEdgeVerticesKernel(
        GLuint vertexBuffer, int numVertexElements,
        GLuint varyingBuffer, int numVaryingElements,
        int vertexOffset, int tableOffset, int start, int end);

    void ApplyCatmarkVertexVerticesKernelB(
        GLuint vertexBuffer, int numVertexElements,
        GLuint varyingBuffer, int numVaryingElements,
        int vertexOffset, int tableOffset, int start, int end);

    void ApplyCatmarkVertexVerticesKernelA(
        GLuint vertexBuffer, int numVertexElements,
        GLuint varyingBuffer, int numVaryingElements,
        int vertexOffset, int tableOffset, int start, int end, bool pass);

    void ApplyLoopEdgeVerticesKernel(
        GLuint vertexBuffer, int numVertexElements,
        GLuint varyingBuffer, int numVaryingElements,
        int vertexOffset, int tableOffset, int start, int end);

    void ApplyLoopVertexVerticesKernelB(
        GLuint vertexBuffer, int numVertexElements,
        GLuint varyingBuffer, int numVaryingElements,
        int vertexOffset, int tableOffset, int start, int end);

    void ApplyLoopVertexVerticesKernelA(
        GLuint vertexBuffer, int numVertexElements,
        GLuint varyingBuffer, int numVaryingElements,
        int vertexOffset, int tableOffset, int start, int end, bool pass);

    void ApplyEditAdd(
        GLuint vertexBuffer, int numVertexElements,
        GLuint varyingBuffer, int numVaryingElements,
        int primvarOffset, int primvarWidth,
        int vertexOffset, int tableOffset, int start, int end);

    void UseProgram() const;

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
        Match(int numVertexElements, int numVaryingElements)
            : vdesc(numVertexElements, numVaryingElements) {
        }

        bool operator() (OsdGLSLTransformFeedbackKernelBundle const *kernel) {
            return vdesc == kernel->_vdesc;
        }

        OsdVertexDescriptor vdesc;
    };

    friend struct Match;

protected:
    void transformGpuBufferData(
        GLuint vertexBuffer, int numVertexElements,
        GLuint varyingBuffer, int numVaryingElements,
        int vertexOffset, int tableOffset, int start, int end) const;

    GLuint _program;

    // uniform locations
    GLint _uniformTables[FarSubdivisionTables<OsdVertex>::TABLE_TYPES_COUNT];
    GLint _uniformVertexPass;
    GLint _uniformVertexOffset;
    GLint _uniformTableOffset;
    GLint _uniformIndexStart;

    GLint _uniformVertexBuffer;
    GLint _uniformVaryingBuffer;

    GLint _uniformEditPrimVarOffset;
    GLint _uniformEditPrimVarWidth;

    GLint _uniformEditIndices;
    GLint _uniformEditValues;
    GLint _uniformVertexBufferImage;

    // subroutines

    GLuint _subComputeFace; // general face-vertex kernel (all schemes)

    GLuint _subComputeEdge; // edge-vertex kernel (catmark + loop schemes)

    GLuint _subComputeBilinearEdge; // edge-vertex kernel (bilinear scheme)

    GLuint _subComputeVertex; // vertex-vertex kernel (bilinear scheme)

    GLuint _subComputeVertexA; // vertex-vertex kernel A (catmark + loop schemes)

    GLuint _subComputeCatmarkVertexB;// vertex-vertex kernel B (catmark scheme)

    GLuint _subComputeLoopVertexB; // vertex-vertex kernel B (loop scheme)

    GLuint _subEditAdd; // hedit kernel (add)

    OsdVertexDescriptor _vdesc;
};

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OSD_GLSL_TRANSFORM_FEEDBACK_KERNEL_BUNDLE_H
