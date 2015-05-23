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

#ifndef OPENSUBDIV3_OSD_GL_COMPUTE_EVALUATOR_H
#define OPENSUBDIV3_OSD_GL_COMPUTE_EVALUATOR_H

#include "../version.h"

#include "../osd/opengl.h"
#include "../osd/vertexDescriptor.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Far {
    class StencilTable;
}

namespace Osd {

/// \brief GL stencil table (Shader Storage buffer)
///
/// This class is a GLSL SSBO representation of Far::StencilTable.
///
/// GLSLComputeKernel consumes this table to apply stencils
///
class GLStencilTableSSBO {
public:
    static GLStencilTableSSBO *Create(Far::StencilTable const *stencilTable,
                                       void *deviceContext = NULL) {
        (void)deviceContext;  // unused
        return new GLStencilTableSSBO(stencilTable);
    }

    explicit GLStencilTableSSBO(Far::StencilTable const *stencilTable);
    ~GLStencilTableSSBO();

    // interfaces needed for GLSLComputeKernel
    GLuint GetSizesBuffer() const { return _sizes; }
    GLuint GetOffsetsBuffer() const { return _offsets; }
    GLuint GetIndicesBuffer() const { return _indices; }
    GLuint GetWeightsBuffer() const { return _weights; }
    int GetNumStencils() const { return _numStencils; }

private:
    GLuint _sizes;
    GLuint _offsets;
    GLuint _indices;
    GLuint _weights;
    int _numStencils;
};

// ---------------------------------------------------------------------------

class GLComputeEvaluator {
public:
    typedef bool Instantiatable;
    static GLComputeEvaluator * Create(VertexBufferDescriptor const &srcDesc,
                                       VertexBufferDescriptor const &dstDesc,
                                       void * deviceContext = NULL) {
        (void)deviceContext;  // not used
        GLComputeEvaluator *instance = new GLComputeEvaluator();
        if (instance->Compile(srcDesc, dstDesc)) return instance;
        delete instance;
        return NULL;
    }

    /// Constructor.
    GLComputeEvaluator();

    /// Destructor. note that the GL context must be made current.
    ~GLComputeEvaluator();

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
    /// @param stencilTable   stencil table to be applied. The table must have
    ///                       SSBO interfaces.
    ///
    /// @param evaluator      cached compiled instance. Clients are supposed to
    ///                       pre-compile an instance of this class and provide
    ///                       to this function. If it's null the kernel still
    ///                       compute by instantiating on-demand kernel although
    ///                       it may cause a performance problem.
    ///
    /// @param deviceContext  not used in the GLSL kernel
    ///
    template <typename VERTEX_BUFFER, typename STENCIL_TABLE>
    static bool EvalStencils(VERTEX_BUFFER *srcVertexBuffer,
                             VertexBufferDescriptor const &srcDesc,
                             VERTEX_BUFFER *dstVertexBuffer,
                             VertexBufferDescriptor const &dstDesc,
                             STENCIL_TABLE const *stencilTable,
                             GLComputeEvaluator const *instance,
                             void * deviceContext = NULL) {
        if (instance) {
            return instance->EvalStencils(srcVertexBuffer, srcDesc,
                                          dstVertexBuffer, dstDesc,
                                          stencilTable);
        } else {
            // Create a kernel on demand (slow)
            (void)deviceContext;  // unused
            instance = Create(srcDesc, dstDesc);
            if (instance) {
                bool r = instance->EvalStencils(srcVertexBuffer, srcDesc,
                                                dstVertexBuffer, dstDesc,
                                                stencilTable);
                delete instance;
                return r;
            }
            return false;
        }
    }

    /// Dispatch the GLSL compute kernel on GPU asynchronously.
    /// returns false if the kernel hasn't been compiled yet.
    template <typename VERTEX_BUFFER, typename STENCIL_TABLE>
    bool EvalStencils(VERTEX_BUFFER *srcVertexBuffer,
                      VertexBufferDescriptor const &srcDesc,
                      VERTEX_BUFFER *dstVertexBuffer,
                      VertexBufferDescriptor const &dstDesc,
                      STENCIL_TABLE const *stencilTable) const {
        return EvalStencils(srcVertexBuffer->BindVBO(),
                            srcDesc,
                            dstVertexBuffer->BindVBO(),
                            dstDesc,
                            stencilTable->GetSizesBuffer(),
                            stencilTable->GetOffsetsBuffer(),
                            stencilTable->GetIndicesBuffer(),
                            stencilTable->GetWeightsBuffer(),
                            /* start = */ 0,
                            /* end   = */ stencilTable->GetNumStencils());
    }

    /// Dispatch the GLSL compute kernel on GPU asynchronously.
    /// returns false if the kernel hasn't been compiled yet.
    bool EvalStencils(GLuint srcBuffer,
                      VertexBufferDescriptor const &srcDesc,
                      GLuint dstBuffer,
                      VertexBufferDescriptor const &dstDesc,
                      GLuint sizesBuffer,
                      GLuint offsetsBuffer,
                      GLuint indicesBuffer,
                      GLuint weightsBuffer,
                      int start,
                      int end) const;

    /// Configure GLSL kernel. A valid GL context must be made current before
    /// calling this function. Returns false if it fails to compile the kernel.
    bool Compile(VertexBufferDescriptor const &srcDesc,
                 VertexBufferDescriptor const &dstDesc);

    /// Wait the dispatched kernel finishes.
    static void Synchronize(void *deviceContext);

private:
    GLuint _program;

    GLuint _uniformSizes,        // stencil table
           _uniformOffsets,
           _uniformIndices,
           _uniformWeights,

           _uniformStart,        // range
           _uniformEnd,

           _uniformSrcOffset,    // src buffer offset (in elements)
           _uniformDstOffset;    // dst buffer offset (in elements)

    int _workGroupSize;
};

}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv


#endif  // OPENSUBDIV3_OSD_GL_COMPUTE_EVALUATOR_H
