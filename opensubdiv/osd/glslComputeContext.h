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
#ifndef OSD_GLSL_COMPUTE_CONTEXT_H
#define OSD_GLSL_COMPUTE_CONTEXT_H

#include "../version.h"

#include "../far/vertexEditTables.h"
#include "../osd/vertex.h"
#include "../osd/vertexDescriptor.h"
#include "../osd/nonCopyable.h"

#include "../osd/opengl.h"

#include <vector>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

class OsdGLSLComputeKernelBundle;


class OsdGLSLComputeTable : OsdNonCopyable<OsdGLSLComputeTable> {
public:
    template<typename T>
    explicit OsdGLSLComputeTable(const std::vector<T> &table) {
        createBuffer(table.size() * sizeof(unsigned int), table.empty() ? NULL : &table[0]);
    }

    virtual ~OsdGLSLComputeTable();

    GLuint GetBuffer() const;

private:
    void createBuffer(size_t size, const void *ptr);

    GLuint _devicePtr;
};


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


///
/// \brief GLSL-Compute Refine Context
///
/// The GLSL-Compute implementation of the Refine module contextual functionality. 
///
/// Contexts interface the serialized topological data pertaining to the 
/// geometric primitives with the capabilities of the selected discrete 
/// compute device.
///
class OsdGLSLComputeContext {

public:
    /// Creates an OsdGLSLComputeContext instance
    ///
    /// @param farmesh the FarMesh used for this Context.
    ///
    static OsdGLSLComputeContext * Create(FarMesh<OsdVertex> const *farmesh);

    /// Destructor
    virtual ~OsdGLSLComputeContext();

    /// Binds a vertex and a varying data buffers to the context. Binding ensures
    /// that data buffers are properly inter-operated between Contexts and 
    /// Controllers operating across multiple devices.
    ///
    /// @param vertex   a buffer containing vertex-interpolated primvar data
    ///
    /// @param varying  a buffer containing varying-interpolated primvar data
    ///
    template<class VERTEX_BUFFER, class VARYING_BUFFER>
    void Bind(VERTEX_BUFFER *vertex, VARYING_BUFFER *varying) {

        _currentVertexBuffer = vertex ? vertex->BindVBO() : 0;
        _currentVaryingBuffer = varying ? varying->BindVBO() : 0;

        _vdesc.numVertexElements = vertex ? vertex->GetNumElements() : 0;
        _vdesc.numVaryingElements = varying ? varying->GetNumElements() : 0;

        bindShaderStorageBuffers();
    }

    /// Unbinds any previously bound vertex and varying data buffers.
    void Unbind() {
        _currentVertexBuffer = 0;
        _currentVaryingBuffer = 0;

        unbindShaderStorageBuffers();
    }

    /// Returns one of the vertex refinement tables.
    ///
    /// @param tableIndex the type of table
    ///
    const OsdGLSLComputeTable * GetTable(int tableIndex) const;

    /// Returns the number of hierarchical edit tables
    int GetNumEditTables() const;

    /// Returns a specific hierarchical edit table
    ///
    /// @param tableIndex the index of the table
    ///
    const OsdGLSLComputeHEditTable * GetEditTable(int tableIndex) const;

    /// Returns a handle to the vertex-interpolated buffer
    GLuint GetCurrentVertexBuffer() const;

    /// Returns a handle to the varying-interpolated buffer
    GLuint GetCurrentVaryingBuffer() const;

    /// Returns an OsdVertexDescriptor if vertex buffers have been bound.
    ///
    /// @return a descriptor for the format of the vertex data currently bound
    ///
    OsdVertexDescriptor const & GetVertexDescriptor() const {
        return _vdesc;
    }

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
    explicit OsdGLSLComputeContext(FarMesh<OsdVertex> const *farMesh);

    void bindShaderStorageBuffers();

    void unbindShaderStorageBuffers();

private:
    std::vector<OsdGLSLComputeTable*> _tables;
    std::vector<OsdGLSLComputeHEditTable*> _editTables;

    GLuint _vertexTexture,
           _varyingTexture;

    OsdVertexDescriptor _vdesc;

    GLuint _currentVertexBuffer, 
           _currentVaryingBuffer;

    OsdGLSLComputeKernelBundle * _kernelBundle;
};

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OSD_GLSL_COMPUTE_CONTEXT_H
