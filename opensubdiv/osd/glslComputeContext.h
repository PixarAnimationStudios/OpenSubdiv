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
#ifndef OSD_GLSL_COMPUTE_CONTEXT_H
#define OSD_GLSL_COMPUTE_CONTEXT_H

#if not defined(__APPLE__)
    #include <GL/gl.h>
#else
    #include <OpenGL/gl3.h>
#endif

#include "../version.h"

#include "../far/table.h"
#include "../far/vertexEditTables.h"
#include "../osd/computeContext.h"

#include <vector>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

class OsdGLSLComputeKernelBundle;

// ----------------------------------------------------------------------------

class OsdGLSLComputeTable : OsdNonCopyable<OsdGLSLComputeTable> {
public:
    OsdGLSLComputeTable(const FarTable<int> &farTable);
    OsdGLSLComputeTable(const FarTable<unsigned int> &farTable);
    OsdGLSLComputeTable(const FarTable<float> &farTable);

    virtual ~OsdGLSLComputeTable();

    GLuint GetBuffer() const;

    int GetMarker(int level) const;

    int GetNumElements(int level) const;

private:
    void createBuffer(int size, const void *ptr);

    GLuint _buffer;
    FarTableMarkers _marker;
};

// ----------------------------------------------------------------------------

class OsdGLSLComputeHEditTable : OsdNonCopyable<OsdGLSLComputeHEditTable> {
public:
    OsdGLSLComputeHEditTable(const FarVertexEditTables<OsdVertex>::
                      VertexEditBatch &batch);

    virtual ~OsdGLSLComputeHEditTable();

    const OsdGLSLComputeTable * GetPrimvarIndices() const;

    const OsdGLSLComputeTable * GetEditValues() const;

    int GetOperation() const;

    int GetPrimvarOffset() const;

    int GetPrimvarWidth() const;

private:
    OsdGLSLComputeTable *_primvarIndicesTable;
    OsdGLSLComputeTable *_editValuesTable;

    int _operation;
    int _primvarOffset;
    int _primvarWidth;
};

// ----------------------------------------------------------------------------

class OsdGLSLComputeContext : public OsdComputeContext {
public:
    static OsdGLSLComputeContext * Create(FarMesh<OsdVertex> *farmesh);

    virtual ~OsdGLSLComputeContext();

    template<class VERTEX_BUFFER, class VARYING_BUFFER>
    void Bind(VERTEX_BUFFER *vertex, VARYING_BUFFER *varying) {

        _currentVertexBuffer = vertex ? vertex->BindVBO() : 0;
        _currentVaryingBuffer = varying ? varying->BindVBO() : 0;

        _numVertexElements = vertex ? vertex->GetNumElements() : 0;
        _numVaryingElements = varying ? varying->GetNumElements() : 0;

        bindShaderStorageBuffers();
    }

    void Unbind() {
        _currentVertexBuffer = 0;
        _currentVaryingBuffer = 0;

        unbindShaderStorageBuffers();
    }

    const OsdGLSLComputeTable * GetTable(int tableIndex) const;

    int GetNumEditTables() const;

    const OsdGLSLComputeHEditTable * GetEditTable(int tableIndex) const;

    GLuint GetCurrentVertexBuffer() const;

    GLuint GetCurrentVaryingBuffer() const;

    int GetNumCurrentVertexElements() const;

    int GetNumCurrentVaryingElements() const;

    OsdGLSLComputeKernelBundle * GetKernelBundle() const;

    void SetKernelBundle(OsdGLSLComputeKernelBundle *kernelBundle);

    void BindUniformBlockBilinearFace(GLuint program, int level);

    void BindUniformBlockBilinearEdge(GLuint program, int level);

    void BindUniformBlockBilinearVertex(GLuint program, int level);

    void BindUniformBlockCatmarkFace(GLuint program, int level);

    void BindUniformBlockCatmarkEdge(GLuint program, int level);

    void BindUniformBlockCatmarkVertexA0(GLuint program, int level);

    void BindUniformBlockCatmarkVertexA1(GLuint program, int level);

    void BindUniformBlockCatmarkVertexB(GLuint program, int level);

    void BindUniformBlockLoopEdge(GLuint program, int level);

    void BindUniformBlockLoopVertexA(GLuint program, int level);

    void BindUniformBlockLoopVertexB(GLuint program, int level);

    void BindEditShaderStorageBuffers(int editIndex);

    void UnbindEditShaderStorageBuffers();

protected:
    explicit OsdGLSLComputeContext(FarMesh<OsdVertex> *farMesh);

    void bindShaderStorageBuffers();

    void unbindShaderStorageBuffers();

private:
    std::vector<OsdGLSLComputeTable*> _tables;
    std::vector<OsdGLSLComputeHEditTable*> _editTables;

    GLuint _vertexTexture;
    GLuint _varyingTexture;

    int _numVertexElements;
    int _numVaryingElements;

    GLuint _currentVertexBuffer, _currentVaryingBuffer;

    OsdGLSLComputeKernelBundle * _kernelBundle;
};

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OSD_GLSL_COMPUTE_CONTEXT_H
