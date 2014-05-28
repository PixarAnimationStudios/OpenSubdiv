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

#ifndef OSD_GLSL_TRANSFORM_FEEDBACK_COMPUTE_CONTEXT_H
#define OSD_GLSL_TRANSFORM_FEEDBACK_COMPUTE_CONTEXT_H

#include "../version.h"

#include "../far/subdivisionTables.h"
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

    GLuint _texture;
};

class OsdGLSLTransformFeedbackHEditTable : OsdNonCopyable<OsdGLSLTransformFeedbackHEditTable> {
public:
    OsdGLSLTransformFeedbackHEditTable(
        const FarVertexEditTables::VertexEditBatch &batch);

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
    /// @param subdivisionTables the FarSubdivisionTables used for this Context.
    ///
    /// @param vertexEditTables the FarVertexEditTables used for this Context.
    ///
    static OsdGLSLTransformFeedbackComputeContext * Create(FarSubdivisionTables const *subdivisionTables,
                                                           FarVertexEditTables const *vertexEditTables);

    /// Destructor
    virtual ~OsdGLSLTransformFeedbackComputeContext();

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

    void BindTableTextures(
        OsdGLSLTransformFeedbackKernelBundle const *kernelBundle) const;

    void UnbindTableTextures() const;

    void BindEditTextures(
        int editIndex,
        OsdGLSLTransformFeedbackKernelBundle const *kernelBundle) const;

    void UnbindEditTextures() const;

protected:
    explicit OsdGLSLTransformFeedbackComputeContext(FarSubdivisionTables const *subdivisionTables,
                                                    FarVertexEditTables const *vertexEditTabes);

    void bindTexture(GLint samplerUniform, GLuint texture, int unit) const;

    void unbindTexture(GLuint unit) const;

private:
    std::vector<OsdGLSLTransformFeedbackTable*> _tables;
    std::vector<OsdGLSLTransformFeedbackHEditTable*> _editTables;
};

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OSD_GLSL_TRANSFORM_FEEDBACK_COMPUTE_CONTEXT_H
