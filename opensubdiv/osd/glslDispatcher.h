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
#ifndef OSD_GLSL_DISPATCHER_H
#define OSD_GLSL_DISPATCHER_H

#if not defined(__APPLE__)
    #include <GL/gl.h>
#else
    #include <OpenGL/gl3.h>
#endif

#include "../version.h"
#include "../osd/kernelDispatcher.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

class OsdGlslKernelDispatcher : public OsdKernelDispatcher {

public:
    OsdGlslKernelDispatcher(int levels);

    virtual ~OsdGlslKernelDispatcher();

    virtual void ApplyBilinearFaceVerticesKernel(FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const;

    virtual void ApplyBilinearEdgeVerticesKernel(FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const;

    virtual void ApplyBilinearVertexVerticesKernel(FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const;


    virtual void ApplyCatmarkFaceVerticesKernel(FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const;

    virtual void ApplyCatmarkEdgeVerticesKernel(FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const;

    virtual void ApplyCatmarkVertexVerticesKernelB(FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const;

    virtual void ApplyCatmarkVertexVerticesKernelA(FarMesh<OsdVertex> * mesh, int offset, bool pass, int level, int start, int end, void * data) const;


    virtual void ApplyLoopEdgeVerticesKernel(FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const;

    virtual void ApplyLoopVertexVerticesKernelB(FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const;

    virtual void ApplyLoopVertexVerticesKernelA(FarMesh<OsdVertex> * mesh, int offset, bool pass, int level, int start, int end, void * data) const;

    virtual void ApplyVertexEdits(FarMesh<OsdVertex> *mesh, int offset, int level, void * clientdata) const {}


    virtual void CopyTable(int tableIndex, size_t size, const void *ptr);

    virtual void AllocateEditTables(int n) {}

    virtual void UpdateEditTable(int tableIndex, const FarTable<unsigned int> &offsets, const FarTable<float> &values,
                                 int operation, int primVarOffset, int primVarWidth) {}

    virtual void OnKernelLaunch();

    virtual void OnKernelFinish();

    virtual OsdVertexBuffer *InitializeVertexBuffer(int numElements, int numVertices);

    virtual void BindVertexBuffer(OsdVertexBuffer *vertex, OsdVertexBuffer *varying);

    virtual void UnbindVertexBuffer();

    virtual void Synchronize();

    static OsdKernelDispatcher * Create(int levels) {
        return new OsdGlslKernelDispatcher(levels);
    }
    static void Register() {
        Factory::GetInstance().Register(Create, kGLSL);
    }

protected:

    class ComputeShader {
    public:
        ComputeShader();
        ~ComputeShader();

        bool Compile(int numVertexElements, int numVaryingElements);

        GLuint GetTableUniform(int table) const {
            return _tableUniforms[table];
        }
        GLuint GetVertexUniform() const { return _vertexUniform; }
        GLuint GetVaryingUniform() const { return _varyingUniform; }



        void ApplyBilinearFaceVerticesKernel(OsdGpuVertexBuffer *vertex, OsdGpuVertexBuffer *varying, int F_IT_ofs, int F_ITa_ofs, int offset, int start, int end);

        void ApplyBilinearEdgeVerticesKernel(OsdGpuVertexBuffer *vertex, OsdGpuVertexBuffer *varying, int E_IT_ofs, int offset, int start, int end);

        void ApplyBilinearVertexVerticesKernel(OsdGpuVertexBuffer *vertex, OsdGpuVertexBuffer *varying, int V_ITa_ofs, int offset, int start, int end);



        void ApplyCatmarkFaceVerticesKernel(OsdGpuVertexBuffer *vertex, OsdGpuVertexBuffer *varying, int F_IT_ofs, int F_ITa_ofs, int offset, int start, int end);

        void ApplyCatmarkEdgeVerticesKernel(OsdGpuVertexBuffer *vertex, OsdGpuVertexBuffer *varying, int E_IT_ofs, int E_W_ofs, int offset, int start, int end);

        void ApplyCatmarkVertexVerticesKernelB(OsdGpuVertexBuffer *vertex, OsdGpuVertexBuffer *varying, int V_IT_ofs, int V_ITa_ofs, int V_W_ofs, int offset, int start, int end);

        void ApplyCatmarkVertexVerticesKernelA(OsdGpuVertexBuffer *vertex, OsdGpuVertexBuffer *varying, int V_ITa_ofs, int V_W_ofs, int offset, bool pass, int start, int end);



        void ApplyLoopEdgeVerticesKernel(OsdGpuVertexBuffer *vertex, OsdGpuVertexBuffer *varying, int E_IT_ofs, int E_W_ofs, int offset, int start, int end);

        void ApplyLoopVertexVerticesKernelB(OsdGpuVertexBuffer *vertex, OsdGpuVertexBuffer *varying, int V_IT_ofs, int V_ITa_ofs, int V_W_ofs, int offset, int start, int end);

        void ApplyLoopVertexVerticesKernelA(OsdGpuVertexBuffer *vertex, OsdGpuVertexBuffer *varying, int V_ITa_ofs, int V_W_ofs, int offset, bool pass, int start, int end);


        void UseProgram () const;

        struct Match {
            Match(int numVertexElements, int numVaryingElements) :
                _numVertexElements(numVertexElements), _numVaryingElements(numVaryingElements) { }

            bool operator() (ComputeShader const & shader) {
                return (shader._numVertexElements == _numVertexElements
                        && shader._numVaryingElements == _numVaryingElements);
            }

            int _numVertexElements,
                _numVaryingElements;
        };

        friend struct Match;

    private:
        void transformGpuBufferData(OsdGpuVertexBuffer *vertex, OsdGpuVertexBuffer *varying,
                                    GLint offset, int start, int end) const;

        int _numVertexElements;
        int _numVaryingElements;

        GLuint _program;

        GLuint _uniformVertexPass;
        GLuint _uniformIndexStart;
        GLuint _uniformIndexOffset;

        GLuint _vertexUniform,
               _varyingUniform;

        // shader locations
        GLuint _subComputeFace,           // general face-vertex kernel (all schemes)
               _subComputeEdge,           // edge-vertex kernel (catmark + loop schemes)
               _subComputeBilinearEdge,   // edge-vertex kernel (bilinear scheme)
               _subComputeVertex,         // vertex-vertex kernel (bilinear scheme)
               _subComputeVertexA,        // vertex-vertex kernel A (catmark + loop schemes)
               _subComputeCatmarkVertexB, // vertex-vertex kernel B (catmark scheme)
               _subComputeLoopVertexB;    // vertex-vertex kernel B (loop scheme)

        std::vector<GLuint> _tableUniforms;
        std::vector<GLuint> _tableOffsetUniforms;

    };

    void bindTextureBuffer(GLuint sampler, GLuint buffer, GLuint texture, GLenum type, int unit) const;

    void unbindTextureBuffer(int unit) const;

    ComputeShader * _shader;

    // texture for vertex
    GLuint _vertexTexture,
           _varyingTexture;

    OsdGpuVertexBuffer *_currentVertexBuffer,
                       *_currentVaryingBuffer;

    // table buffers
    std::vector<GLuint> _tableBuffers;
    std::vector<GLuint> _tableTextures;

    // static shader registry (XXX tentative..)
    static std::vector<ComputeShader> shaderRegistry;
};

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif // OSD_GLSL_DISPATCHER_H
