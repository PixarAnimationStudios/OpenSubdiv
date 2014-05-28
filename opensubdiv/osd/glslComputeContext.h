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

#ifndef OSD_GLSL_COMPUTE_CONTEXT_H
#define OSD_GLSL_COMPUTE_CONTEXT_H

#include "../version.h"

#include "../far/subdivisionTables.h"
#include "../far/vertexEditTables.h"
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
    OsdGLSLComputeHEditTable(const FarVertexEditTables::
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
    /// @param subdivisionTables the FarSubdivisionTables used for this Context.
    ///
    /// @param vertexEditTables the FarVertexEditTables used for this Context.
    ///
    static OsdGLSLComputeContext * Create(FarSubdivisionTables const *subdivisionTables,
                                          FarVertexEditTables const *vertexEditTables);

    /// Destructor
    virtual ~OsdGLSLComputeContext();

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

    void BindShaderStorageBuffers() const;

    void UnbindShaderStorageBuffers() const;

    void BindEditShaderStorageBuffers(int editIndex) const;

    void UnbindEditShaderStorageBuffers() const;

protected:
    explicit OsdGLSLComputeContext(FarSubdivisionTables const *subdivisionTables,
                                   FarVertexEditTables const *vertexEditTables);

private:
    std::vector<OsdGLSLComputeTable*> _tables;
    std::vector<OsdGLSLComputeHEditTable*> _editTables;
};

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OSD_GLSL_COMPUTE_CONTEXT_H
