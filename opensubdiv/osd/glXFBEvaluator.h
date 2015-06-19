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

#ifndef OPENSUBDIV3_OSD_GL_XFB_EVALUATOR_H
#define OPENSUBDIV3_OSD_GL_XFB_EVALUATOR_H

#include "../version.h"

#include "../osd/opengl.h"
#include "../osd/types.h"
#include "../osd/bufferDescriptor.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Far {
    class PatchTable;
    class StencilTable;
    class LimitStencilTable;
}

namespace Osd {

/// \brief GL TextureBuffer stencil table
///
/// This class is a GL Texture Buffer representation of Far::StencilTable.
///
/// GLSLTransformFeedback consumes this table to apply stencils
///
///
class GLStencilTableTBO {
public:
    static GLStencilTableTBO *Create(
        Far::StencilTable const *stencilTable, void *deviceContext = NULL) {
        (void)deviceContext;  // unused
        return new GLStencilTableTBO(stencilTable);
    }

    static GLStencilTableTBO *Create(
        Far::LimitStencilTable const *limitStencilTable,
        void *deviceContext = NULL) {
        (void)deviceContext;  // unused
        return new GLStencilTableTBO(limitStencilTable);
    }

    explicit GLStencilTableTBO(Far::StencilTable const *stencilTable);
    explicit GLStencilTableTBO(Far::LimitStencilTable const *limitStencilTable);
    ~GLStencilTableTBO();

    // interfaces needed for GLSLTransformFeedbackKernel
    GLuint GetSizesTexture() const { return _sizes; }
    GLuint GetOffsetsTexture() const { return _offsets; }
    GLuint GetIndicesTexture() const { return _indices; }
    GLuint GetWeightsTexture() const { return _weights; }
    GLuint GetDuWeightsTexture() const { return _duWeights; }
    GLuint GetDvWeightsTexture() const { return _dvWeights; }
    int GetNumStencils() const { return _numStencils; }

private:
    GLuint _sizes;
    GLuint _offsets;
    GLuint _indices;
    GLuint _weights;
    GLuint _duWeights;
    GLuint _dvWeights;
    int _numStencils;
};

// ---------------------------------------------------------------------------

class GLXFBEvaluator {
public:
    typedef bool Instantiatable;
    static GLXFBEvaluator * Create(BufferDescriptor const &srcDesc,
                                   BufferDescriptor const &dstDesc,
                                   BufferDescriptor const &duDesc,
                                   BufferDescriptor const &dvDesc,
                                   void * deviceContext = NULL) {
        (void)deviceContext;  // not used
        GLXFBEvaluator *instance = new GLXFBEvaluator();
        if (instance->Compile(srcDesc, dstDesc, duDesc, dvDesc))
            return instance;
        delete instance;
        return NULL;
    }

    /// Constructor.
    GLXFBEvaluator();

    /// Destructor. note that the GL context must be made current.
    ~GLXFBEvaluator();

    /// ----------------------------------------------------------------------
    ///
    ///   Stencil evaluations with StencilTable
    ///
    /// ----------------------------------------------------------------------

    /// \brief Generic static stencil function. This function has a same
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
    ///                       Texture Buffer Object interfaces.
    ///
    /// @param instance       cached compiled instance. Clients are supposed to
    ///                       pre-compile an instance of this class and provide
    ///                       to this function. If it's null the kernel still
    ///                       compute by instantiating on-demand kernel although
    ///                       it may cause a performance problem.
    ///
    /// @param deviceContext  not used in the GLSLTransformFeedback kernel
    ///
    template <typename SRC_BUFFER, typename DST_BUFFER, typename STENCIL_TABLE>
    static bool EvalStencils(
        SRC_BUFFER *srcBuffer, BufferDescriptor const &srcDesc,
        DST_BUFFER *dstBuffer, BufferDescriptor const &dstDesc,
        STENCIL_TABLE const *stencilTable,
        GLXFBEvaluator const *instance,
        void * deviceContext = NULL) {

        if (instance) {
            return instance->EvalStencils(srcBuffer, srcDesc,
                                          dstBuffer, dstDesc,
                                          stencilTable);
        } else {
            // Create an instance on demand (slow)
            (void)deviceContext;  // unused
            instance = Create(srcDesc, dstDesc,
                              BufferDescriptor(),
                              BufferDescriptor());
            if (instance) {
                bool r = instance->EvalStencils(srcBuffer, srcDesc,
                                                dstBuffer, dstDesc,
                                                stencilTable);
                delete instance;
                return r;
            }
            return false;
        }
    }

    /// \brief Generic static stencil function. This function has a same
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
    /// @param duBuffer       Output U-derivative buffer
    ///                       must have BindVBO() method returning a
    ///                       float pointer for write
    ///
    /// @param duDesc         vertex buffer descriptor for the du output buffer
    ///
    /// @param dvBuffer       Output V-derivative buffer
    ///                       must have BindVBO() method returning a
    ///                       float pointer for write
    ///
    /// @param dvDesc         vertex buffer descriptor for the dv output buffer
    ///
    /// @param stencilTable   stencil table to be applied. The table must have
    ///                       Texture Buffer Object interfaces.
    ///
    /// @param instance       cached compiled instance. Clients are supposed to
    ///                       pre-compile an instance of this class and provide
    ///                       to this function. If it's null the kernel still
    ///                       compute by instantiating on-demand kernel although
    ///                       it may cause a performance problem.
    ///
    /// @param deviceContext  not used in the GLSLTransformFeedback kernel
    ///
    template <typename SRC_BUFFER, typename DST_BUFFER, typename STENCIL_TABLE>
    static bool EvalStencils(
        SRC_BUFFER *srcBuffer, BufferDescriptor const &srcDesc,
        DST_BUFFER *dstBuffer, BufferDescriptor const &dstDesc,
        DST_BUFFER *duBuffer, BufferDescriptor const &duDesc,
        DST_BUFFER *dvBuffer, BufferDescriptor const &dvDesc,
        STENCIL_TABLE const *stencilTable,
        GLXFBEvaluator const *instance,
        void * deviceContext = NULL) {

        if (instance) {
            return instance->EvalStencils(srcBuffer, srcDesc,
                                          dstBuffer, dstDesc,
                                          duBuffer,  duDesc,
                                          dvBuffer,  dvDesc,
                                          stencilTable);
        } else {
            // Create an instance on demand (slow)
            (void)deviceContext;  // unused
            instance = Create(srcDesc, dstDesc, duDesc, dvDesc);
            if (instance) {
                bool r = instance->EvalStencils(srcBuffer, srcDesc,
                                                dstBuffer, dstDesc,
                                                duBuffer, duDesc,
                                                dvBuffer, dvDesc,
                                                stencilTable);
                delete instance;
                return r;
            }
            return false;
        }
    }

    /// \brief Generic eval stencils function.
    ///
    /// @param srcBuffer      Input primvar buffer.
    ///                       must have BindVBO() method returning a GL
    ///                       buffer object of source data
    ///
    /// @param srcDesc        vertex buffer descriptor for the input buffer
    ///
    /// @param dstBuffer      Output primvar buffer
    ///                       must have BindVBO() method returning a GL
    ///                       buffer object for destination data
    ///
    /// @param dstDesc        vertex buffer descriptor for the output buffer
    ///
    /// @param stencilTable   stencil table to be applied.
    ///
    template <typename SRC_BUFFER, typename DST_BUFFER, typename STENCIL_TABLE>
    bool EvalStencils(
        SRC_BUFFER *srcBuffer, BufferDescriptor const &srcDesc,
        DST_BUFFER *dstBuffer, BufferDescriptor const &dstDesc,
        STENCIL_TABLE const *stencilTable) const {

        return EvalStencils(srcBuffer->BindVBO(), srcDesc,
                            dstBuffer->BindVBO(), dstDesc,
                            0, BufferDescriptor(),
                            0, BufferDescriptor(),
                            stencilTable->GetSizesTexture(),
                            stencilTable->GetOffsetsTexture(),
                            stencilTable->GetIndicesTexture(),
                            stencilTable->GetWeightsTexture(),
                            0,
                            0,
                            /* start = */ 0,
                            /* end   = */ stencilTable->GetNumStencils());
    }

    /// \brief Generic eval stencils function with derivative evaluation.
    ///
    /// @param srcBuffer      Input primvar buffer.
    ///                       must have BindVBO() method returning a GL
    ///                       buffer object of source data
    ///
    /// @param srcDesc        vertex buffer descriptor for the input buffer
    ///
    /// @param dstBuffer      Output primvar buffer
    ///                       must have BindVBO() method returning a GL
    ///                       buffer object for destination data
    ///
    /// @param dstDesc        vertex buffer descriptor for the output buffer
    ///
    /// @param duBuffer       GL buffer of output U-derivatives.
    ///
    /// @param duDesc         vertex buffer descriptor for the duBuffer
    ///
    /// @param dvBuffer       GL buffer of output V-derivatives.
    ///
    /// @param dvDesc         vertex buffer descriptor for the dvBuffer
    ///
    /// @param stencilTable   stencil table to be applied.
    ///
    template <typename SRC_BUFFER, typename DST_BUFFER, typename STENCIL_TABLE>
    bool EvalStencils(
        SRC_BUFFER *srcBuffer, BufferDescriptor const &srcDesc,
        DST_BUFFER *dstBuffer, BufferDescriptor const &dstDesc,
        DST_BUFFER *duBuffer,  BufferDescriptor const &duDesc,
        DST_BUFFER *dvBuffer,  BufferDescriptor const &dvDesc,
        STENCIL_TABLE const *stencilTable) const {

        return EvalStencils(srcBuffer->BindVBO(), srcDesc,
                            dstBuffer->BindVBO(), dstDesc,
                            duBuffer->BindVBO(),  duDesc,
                            dvBuffer->BindVBO(),  dvDesc,
                            stencilTable->GetSizesTexture(),
                            stencilTable->GetOffsetsTexture(),
                            stencilTable->GetIndicesTexture(),
                            stencilTable->GetWeightsTexture(),
                            stencilTable->GetDuWeightsTexture(),
                            stencilTable->GetDvWeightsTexture(),
                            /* start = */ 0,
                            /* end   = */ stencilTable->GetNumStencils());
    }

    /// \brief dispatch eval stencils function with derivatives.
    ///        dispatch the GLSL XFB kernel on on GPU asynchronously.
    ///
    /// @param srcBuffer       GL buffer of input primvars.
    ///
    /// @param srcDesc         vertex buffer descriptor for the srcBuffer
    ///
    /// @param dstBuffer       GL buffer of output primvars.
    ///
    /// @param dstDesc         vertex buffer descriptor for the dstBuffer
    ///
    /// @param duBuffer        GL buffer of output U-derivatives.
    ///
    /// @param duDesc          vertex buffer descriptor for the duBuffer
    ///
    /// @param dvBuffer        GL buffer of output V-derivatives.
    ///
    /// @param dvDesc          vertex buffer descriptor for the dvBuffer
    ///
    /// @param sizesBuffer     GL buffer of the sizes in the stencil table
    ///
    /// @param offsetsBuffer   GL buffer of the offsets in the stencil table
    ///
    /// @param indicesBuffer   GL buffer of the indices in the stencil table
    ///
    /// @param weightsBuffer   GL buffer of the weights in the stencil table
    ///
    /// @param duWeightsBuffer GL buffer of the du weights in the stencil table
    ///
    /// @param dvWeightsBuffer GL buffer of the dv weights in the stencil table
    ///
    /// @param start           start index of stencil table
    ///
    /// @param end             end index of stencil table
    ///
    bool EvalStencils(GLuint srcBuffer, BufferDescriptor const &srcDesc,
                      GLuint dstBuffer, BufferDescriptor const &dstDesc,
                      GLuint duBuffer,  BufferDescriptor const &duDesc,
                      GLuint dvBuffer,  BufferDescriptor const &dvDesc,
                      GLuint sizesBuffer,
                      GLuint offsetsBuffer,
                      GLuint indicesBuffer,
                      GLuint weightsBuffer,
                      GLuint duWeightsBuffer,
                      GLuint dvWeightsBuffer,
                      int start,
                      int end) const;

    /// ----------------------------------------------------------------------
    ///
    ///   Limit evaluations with PatchTable
    ///
    /// ----------------------------------------------------------------------
    ///
    /// \brief Generic limit eval function. This function has a same
    ///        signature as other device kernels have so that it can be called
    ///        in the same way.
    ///
    /// @param srcBuffer      Input primvar buffer.
    ///                       must have BindVBO() method returning a GL
    ///                       buffer object of source data
    ///
    /// @param srcDesc        vertex buffer descriptor for the input buffer
    ///
    /// @param dstBuffer      Output primvar buffer
    ///                       must have BindVBO() method returning a GL
    ///                       buffer object of destination data
    ///
    /// @param dstDesc        vertex buffer descriptor for the output buffer
    ///
    /// @param numPatchCoords number of patchCoords.
    ///
    /// @param patchCoords    array of locations to be evaluated.
    ///                       must have BindVBO() method returning an
    ///                       array of PatchCoord struct in VBO.
    ///
    /// @param patchTable     GLPatchTable or equivalent
    ///
    /// @param instance       cached compiled instance. Clients are supposed to
    ///                       pre-compile an instance of this class and provide
    ///                       to this function. If it's null the kernel still
    ///                       compute by instantiating on-demand kernel although
    ///                       it may cause a performance problem.
    ///
    /// @param deviceContext  not used in the GLXFB evaluator
    ///
    template <typename SRC_BUFFER, typename DST_BUFFER,
              typename PATCHCOORD_BUFFER, typename PATCH_TABLE>
    static bool EvalPatches(
        SRC_BUFFER *srcBuffer, BufferDescriptor const &srcDesc,
        DST_BUFFER *dstBuffer, BufferDescriptor const &dstDesc,
        int numPatchCoords,
        PATCHCOORD_BUFFER *patchCoords,
        PATCH_TABLE *patchTable,
        GLXFBEvaluator const *instance,
        void * deviceContext = NULL) {

        if (instance) {
            return instance->EvalPatches(srcBuffer, srcDesc,
                                         dstBuffer, dstDesc,
                                         numPatchCoords, patchCoords,
                                         patchTable);
        } else {
            // Create an instance on demand (slow)
            (void)deviceContext;  // unused
            instance = Create(srcDesc, dstDesc,
                              BufferDescriptor(),
                              BufferDescriptor());
            if (instance) {
                bool r = instance->EvalPatches(srcBuffer, srcDesc,
                                               dstBuffer, dstDesc,
                                               numPatchCoords, patchCoords,
                                               patchTable);
                delete instance;
                return r;
            }
            return false;
        }
    }

    /// \brief Generic limit eval function. This function has a same
    ///        signature as other device kernels have so that it can be called
    ///        in the same way.
    ///
    /// @param srcBuffer      Input primvar buffer.
    ///                       must have BindVBO() method returning a GL
    ///                       buffer object of source data
    ///
    /// @param srcDesc        vertex buffer descriptor for the input buffer
    ///
    /// @param dstBuffer      Output primvar buffer
    ///                       must have BindVBO() method returning a GL
    ///                       buffer object of destination data
    ///
    /// @param dstDesc        vertex buffer descriptor for the output buffer
    ///
    /// @param duBuffer
    ///
    /// @param duDesc
    ///
    /// @param dvBuffer
    ///
    /// @param dvDesc
    ///
    /// @param numPatchCoords number of patchCoords.
    ///
    /// @param patchCoords    array of locations to be evaluated.
    ///                       must have BindVBO() method returning an
    ///                       array of PatchCoord struct in VBO.
    ///
    /// @param patchTable     GLPatchTable or equivalent
    ///
    /// @param instance       cached compiled instance. Clients are supposed to
    ///                       pre-compile an instance of this class and provide
    ///                       to this function. If it's null the kernel still
    ///                       compute by instantiating on-demand kernel although
    ///                       it may cause a performance problem.
    ///
    /// @param deviceContext  not used in the GLXFB evaluator
    ///
    template <typename SRC_BUFFER, typename DST_BUFFER,
              typename PATCHCOORD_BUFFER, typename PATCH_TABLE>
    static bool EvalPatches(
        SRC_BUFFER *srcBuffer, BufferDescriptor const &srcDesc,
        DST_BUFFER *dstBuffer, BufferDescriptor const &dstDesc,
        DST_BUFFER *duBuffer,  BufferDescriptor const &duDesc,
        DST_BUFFER *dvBuffer,  BufferDescriptor const &dvDesc,
        int numPatchCoords,
        PATCHCOORD_BUFFER *patchCoords,
        PATCH_TABLE *patchTable,
        GLXFBEvaluator const *instance,
        void * deviceContext = NULL) {

        if (instance) {
            return instance->EvalPatches(srcBuffer, srcDesc,
                                         dstBuffer, dstDesc,
                                         duBuffer, duDesc,
                                         dvBuffer, dvDesc,
                                         numPatchCoords, patchCoords,
                                         patchTable);
        } else {
            // Create an instance on demand (slow)
            (void)deviceContext;  // unused
            instance = Create(srcDesc, dstDesc, duDesc, dvDesc);
            if (instance) {
                bool r = instance->EvalPatches(srcBuffer, srcDesc,
                                               dstBuffer, dstDesc,
                                               duBuffer, duDesc,
                                               dvBuffer, dvDesc,
                                               numPatchCoords, patchCoords,
                                               patchTable);
                delete instance;
                return r;
            }
            return false;
        }
    }

    /// \brief Generic limit eval function. This function has a same
    ///        signature as other device kernels have so that it can be called
    ///        in the same way.
    ///
    /// @param srcBuffer      Input primvar buffer.
    ///                       must have BindCudaBuffer() method returning a
    ///                       const float pointer for read
    ///
    /// @param srcDesc        vertex buffer descriptor for the input buffer
    ///
    /// @param dstBuffer      Output primvar buffer
    ///                       must have BindCudaBuffer() method returning a
    ///                       float pointer for write
    ///
    /// @param dstDesc        vertex buffer descriptor for the output buffer
    ///
    /// @param numPatchCoords number of patchCoords.
    ///
    /// @param patchCoords    array of locations to be evaluated.
    ///                       must have BindCudaBuffer() method returning an
    ///                       array of PatchCoord struct in cuda memory.
    ///
    /// @param patchTable     GLPatchTable or equivalent
    ///
    template <typename SRC_BUFFER, typename DST_BUFFER,
              typename PATCHCOORD_BUFFER, typename PATCH_TABLE>
    bool EvalPatches(
        SRC_BUFFER *srcBuffer, BufferDescriptor const &srcDesc,
        DST_BUFFER *dstBuffer, BufferDescriptor const &dstDesc,
        int numPatchCoords,
        PATCHCOORD_BUFFER *patchCoords,
        PATCH_TABLE *patchTable) const {

        return EvalPatches(srcBuffer->BindVBO(), srcDesc,
                           dstBuffer->BindVBO(), dstDesc,
                           0, BufferDescriptor(),
                           0, BufferDescriptor(),
                           numPatchCoords,
                           patchCoords->BindVBO(),
                           patchTable->GetPatchArrays(),
                           patchTable->GetPatchIndexTextureBuffer(),
                           patchTable->GetPatchParamTextureBuffer());
    }

    /// \brief Generic limit eval function with derivatives. This function has
    ///        a same signature as other device kernels have so that it can be
    ///        called in the same way.
    ///
    /// @param srcBuffer        Input primvar buffer.
    ///                         must have BindCudaBuffer() method returning a
    ///                         const float pointer for read
    ///
    /// @param srcDesc          vertex buffer descriptor for the input buffer
    ///
    /// @param dstBuffer        Output primvar buffer
    ///                         must have BindCudaBuffer() method returning a
    ///                         float pointer for write
    ///
    /// @param dstDesc          vertex buffer descriptor for the output buffer
    ///
    /// @param duBuffer         Output s-derivatives buffer
    ///                         must have BindCudaBuffer() method returning a
    ///                         float pointer for write
    ///
    /// @param duDesc           vertex buffer descriptor for the duBuffer
    ///
    /// @param dvBuffer         Output t-derivatives buffer
    ///                         must have BindCudaBuffer() method returning a
    ///                         float pointer for write
    ///
    /// @param dvDesc           vertex buffer descriptor for the dvBuffer
    ///
    /// @param numPatchCoords   number of patchCoords.
    ///
    /// @param patchCoords      array of locations to be evaluated.
    ///
    /// @param patchTable       GLPatchTable or equivalent
    ///
    template <typename SRC_BUFFER, typename DST_BUFFER,
              typename PATCHCOORD_BUFFER, typename PATCH_TABLE>
    bool EvalPatches(
        SRC_BUFFER *srcBuffer, BufferDescriptor const &srcDesc,
        DST_BUFFER *dstBuffer, BufferDescriptor const &dstDesc,
        DST_BUFFER *duBuffer,  BufferDescriptor const &duDesc,
        DST_BUFFER *dvBuffer,  BufferDescriptor const &dvDesc,
        int numPatchCoords,
        PATCHCOORD_BUFFER *patchCoords,
        PATCH_TABLE *patchTable) const {

        return EvalPatches(srcBuffer->BindVBO(), srcDesc,
                           dstBuffer->BindVBO(), dstDesc,
                           duBuffer->BindVBO(),  duDesc,
                           dvBuffer->BindVBO(),  dvDesc,
                           numPatchCoords,
                           patchCoords->BindVBO(),
                           patchTable->GetPatchArrays(),
                           patchTable->GetPatchIndexTextureBuffer(),
                           patchTable->GetPatchParamTextureBuffer());
    }

    bool EvalPatches(GLuint srcBuffer, BufferDescriptor const &srcDesc,
                     GLuint dstBuffer, BufferDescriptor const &dstDesc,
                     GLuint duBuffer, BufferDescriptor const &duDesc,
                     GLuint dvBuffer, BufferDescriptor const &dvDesc,
                     int numPatchCoords,
                     GLuint patchCoordsBuffer,
                     const PatchArrayVector &patchArrays,
                     GLuint patchIndexBuffer,
                     GLuint patchParamsBuffer) const;

    /// ----------------------------------------------------------------------
    ///
    ///   Other methods
    ///
    /// ----------------------------------------------------------------------

    /// Configure GLSL kernel. A valid GL context must be made current before
    /// calling this function. Returns false if it fails to compile the kernel.
    bool Compile(BufferDescriptor const &srcDesc,
                 BufferDescriptor const &dstDesc,
                 BufferDescriptor const &duDesc,
                 BufferDescriptor const &dvDesc);

    /// Wait the dispatched kernel finishes.
    static void Synchronize(void *kernel);

private:
    GLuint _srcBufferTexture;

    struct _StencilKernel {
        _StencilKernel();
        ~_StencilKernel();
        bool Compile(BufferDescriptor const &srcDesc,
                     BufferDescriptor const &dstDesc,
                     BufferDescriptor const &duDesc,
                     BufferDescriptor const &dvDesc);
        GLuint program;
        GLint uniformSrcBufferTexture;
        GLint uniformSrcOffset;    // src buffer offset (in elements)

        GLint uniformSizesTexture;
        GLint uniformOffsetsTexture;
        GLint uniformIndicesTexture;
        GLint uniformWeightsTexture;
        GLint uniformDuWeightsTexture;
        GLint uniformDvWeightsTexture;
        GLint uniformStart;     // range
        GLint uniformEnd;
    } _stencilKernel;

    struct _PatchKernel {
        _PatchKernel();
        ~_PatchKernel();
        bool Compile(BufferDescriptor const &srcDesc,
                     BufferDescriptor const &dstDesc,
                     BufferDescriptor const &duDesc,
                     BufferDescriptor const &dvDesc);
        GLuint program;
        GLint uniformSrcBufferTexture;
        GLint uniformSrcOffset;    // src buffer offset (in elements)

        GLint uniformPatchArray;
        GLint uniformPatchParamTexture;
        GLint uniformPatchIndexTexture;
    } _patchKernel;

};

}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv


#endif  // OPENSUBDIV3_OSD_GL_XFB_EVALUATOR_H
