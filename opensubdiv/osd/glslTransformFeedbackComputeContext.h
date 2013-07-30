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
#ifndef OSD_GLSL_TRANSFORM_FEEDBACK_COMPUTE_CONTEXT_H
#define OSD_GLSL_TRANSFORM_FEEDBACK_COMPUTE_CONTEXT_H

#include "../version.h"

#include "../far/vertexEditTables.h"
#include "../osd/vertex.h"
#include "../osd/vertexDescriptor.h"
#include "../osd/nonCopyable.h"

#include "../osd/opengl.h"

#include <vector>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

class OsdGLSLTransformFeedbackKernelBundle;

class OsdGLSLTransformFeedbackTable : OsdNonCopyable<OsdGLSLTransformFeedbackTable> {
public:
    template<typename T>
    OsdGLSLTransformFeedbackTable(const std::vector<T> &table, GLenum type) {
        createTextureBuffer(table.size() * sizeof(unsigned int), table.empty() ? NULL : &table[0], type);
    }

    virtual ~OsdGLSLTransformFeedbackTable();

    GLuint GetTexture() const;

private:
    void createTextureBuffer(size_t size, const void *ptr, GLenum type);

    GLuint _devicePtr, 
           _texture;
};

class OsdGLSLTransformFeedbackHEditTable : OsdNonCopyable<OsdGLSLTransformFeedbackHEditTable> {
public:
    OsdGLSLTransformFeedbackHEditTable(const FarVertexEditTables<OsdVertex>::
                      VertexEditBatch &batch);

    virtual ~OsdGLSLTransformFeedbackHEditTable();

    const OsdGLSLTransformFeedbackTable * GetPrimvarIndices() const;

    const OsdGLSLTransformFeedbackTable * GetEditValues() const;

    int GetOperation() const;

    int GetPrimvarOffset() const;

    int GetPrimvarWidth() const;

private:
    OsdGLSLTransformFeedbackTable *_primvarIndicesTable;
    OsdGLSLTransformFeedbackTable *_editValuesTable;

    int _operation;
    int _primvarOffset;
    int _primvarWidth;
};

///
/// \brief GLSL (transform-feedback) Refine Context
///
/// The GLSL (transform-feedback) implementation of the Refine module contextual functionality. 
///
/// Contexts interface the serialized topological data pertaining to the 
/// geometric primitives with the capabilities of the selected discrete 
/// compute device.
///
class OsdGLSLTransformFeedbackComputeContext {
public:
    /// Creates an OsdGLSLTransformFeedbackComputeContext instance
    ///
    /// @param farmesh the FarMesh used for this Context.
    ///
    static OsdGLSLTransformFeedbackComputeContext * Create(FarMesh<OsdVertex> const *farmesh);

    /// Destructor
    virtual ~OsdGLSLTransformFeedbackComputeContext();

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

        bind();
    }

    /// Unbinds any previously bound vertex and varying data buffers.
    void Unbind() {
        _currentVertexBuffer = 0;
        _currentVaryingBuffer = 0;
        unbind();
    }

    /// Returns one of the vertex refinement tables.
    ///
    /// @param tableIndex the type of table
    ///
    const OsdGLSLTransformFeedbackTable * GetTable(int tableIndex) const;

    /// Returns the number of hierarchical edit tables
    int GetNumEditTables() const;

    /// Returns a specific hierarchical edit table
    ///
    /// @param tableIndex the index of the table
    ///
    const OsdGLSLTransformFeedbackHEditTable * GetEditTable(int tableIndex) const;

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

    OsdGLSLTransformFeedbackKernelBundle * GetKernelBundle() const;

    void SetKernelBundle(OsdGLSLTransformFeedbackKernelBundle *kernelBundle);

    void BindEditTextures(int editIndex);

    void UnbindEditTextures();

protected:
    explicit OsdGLSLTransformFeedbackComputeContext(FarMesh<OsdVertex> const *farMesh);

    void bindTexture(GLint samplerUniform, GLuint texture, int unit);

    void unbindTexture(GLuint unit);

    void bind();

    void unbind();

private:
    std::vector<OsdGLSLTransformFeedbackTable*> _tables;
    std::vector<OsdGLSLTransformFeedbackHEditTable*> _editTables;

    GLuint _vertexTexture,
           _varyingTexture;

    OsdVertexDescriptor _vdesc;

    GLuint _currentVertexBuffer, 
           _currentVaryingBuffer;

    OsdGLSLTransformFeedbackKernelBundle * _kernelBundle;
};

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OSD_GLSL_TRANSFORM_FEEDBACK_COMPUTE_CONTEXT_H
