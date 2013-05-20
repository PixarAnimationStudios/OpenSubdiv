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
#ifndef OSD_GLSL_TRANSFORM_FEEDBACK_KERNEL_BUNDLE_H
#define OSD_GLSL_TRANSFORM_FEEDBACK_KERNEL_BUNDLE_H

#if defined(__APPLE__)
    #include "TargetConditionals.h"
    #if TARGET_OS_IPHONE or TARGET_IPHONE_SIMULATOR
        #include <OpenGLES/ES2/gl.h>
    #else
        #include <OpenGL/gl3.h>
    #endif
#elif defined(ANDROID)
    #include <GLES2/gl2.h>
#else
    #if defined(_WIN32)
        #include <windows.h>
    #endif
    #include <GL/gl.h>
#endif

#include "../version.h"
#include "../osd/nonCopyable.h"
#include "../osd/vertex.h"
#include "../osd/vertexDescriptor.h"
#include "../far/subdivisionTables.h"

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

    GLuint GetTableUniformLocation(int tableIndex) const {
        return _uniformTables[tableIndex];
    }
    GLuint GetVertexUniformLocation() const {
        return _uniformVertexBuffer;
    }
    GLuint GetVaryingUniformLocation() const {
        return _uniformVaryingBuffer;
    }
    GLuint GetEditIndicesUniformLocation() const {
        return _uniformEditIndices;
    }
    GLuint GetEditValuesUniformLocation() const {
        return _uniformEditValues;
    }
    GLuint GetVertexBufferImageUniformLocation() const {
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
    GLuint _uniformTables[FarSubdivisionTables<OsdVertex>::TABLE_TYPES_COUNT];
    GLuint _uniformVertexPass;
    GLuint _uniformVertexOffset;
    GLuint _uniformTableOffset;
    GLuint _uniformIndexStart;

    GLuint _uniformVertexBuffer;
    GLuint _uniformVaryingBuffer;

    GLuint _uniformEditPrimVarOffset;
    GLuint _uniformEditPrimVarWidth;

    GLuint _uniformEditIndices;
    GLuint _uniformEditValues;
    GLuint _uniformVertexBufferImage;

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
