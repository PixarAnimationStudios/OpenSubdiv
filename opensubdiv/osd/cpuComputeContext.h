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
#ifndef OSD_CPU_COMPUTE_CONTEXT_H
#define OSD_CPU_COMPUTE_CONTEXT_H

#include "../version.h"

#include "../far/subdivisionTables.h"
#include "../far/vertexEditTables.h"
#include "../osd/vertex.h"
#include "../osd/vertexDescriptor.h"
#include "../osd/nonCopyable.h"

#include <stdlib.h>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

struct OsdVertexDescriptor;

class OsdCpuTable : OsdNonCopyable<OsdCpuTable> {
public:
    template<typename T>
    explicit OsdCpuTable(const std::vector<T> &table) {
        createCpuBuffer(table.size() * sizeof(T), table.empty() ? NULL : &table[0]);
    }

    virtual ~OsdCpuTable();

    void * GetBuffer() const;

private:
    void createCpuBuffer(size_t size, const void *ptr);

    void *_devicePtr;
};

class OsdCpuHEditTable : OsdNonCopyable<OsdCpuHEditTable> {
public:
    OsdCpuHEditTable(const FarVertexEditTables<OsdVertex>::
                      VertexEditBatch &batch);

    virtual ~OsdCpuHEditTable();

    const OsdCpuTable * GetPrimvarIndices() const;

    const OsdCpuTable * GetEditValues() const;

    int GetOperation() const;

    int GetPrimvarOffset() const;

    int GetPrimvarWidth() const;

private:
    OsdCpuTable *_primvarIndicesTable;
    OsdCpuTable *_editValuesTable;

    int _operation;
    int _primvarOffset;
    int _primvarWidth;
};

///
/// \brief CPU Refine Context
///
/// The CPU implementation of the Refine module contextual functionality. 
///
/// Contexts interface the serialized topological data pertaining to the 
/// geometric primitives with the capabilities of the selected discrete 
/// compute device.
///
class OsdCpuComputeContext : OsdNonCopyable<OsdCpuComputeContext> {

public:
    /// Creates an OsdCpuComputeContext instance
    ///
    /// @param farmesh the FarMesh used for this Context.
    ///
    static OsdCpuComputeContext * Create(FarMesh<OsdVertex> const *farmesh);

    /// Destructor
    virtual ~OsdCpuComputeContext();

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

        _currentVertexBuffer = vertex ? vertex->BindCpuBuffer() : 0;
        _currentVaryingBuffer = varying ? varying->BindCpuBuffer() : 0;

        int numVertexElements = vertex ? vertex->GetNumElements() : 0;
        int numVaryingElements = varying ? varying->GetNumElements() : 0;
        _vdesc.Set(numVertexElements, numVaryingElements);
    }

    /// Unbinds any previously bound vertex and varying data buffers.
    void Unbind() {
        _currentVertexBuffer = 0;
        _currentVaryingBuffer = 0;
        _vdesc.Reset();
    }

    /// Returns one of the vertex refinement tables.
    ///
    /// @param tableIndex the type of table
    ///
    const OsdCpuTable * GetTable(int tableIndex) const;

    /// Returns an OsdVertexDescriptor if vertex buffers have been bound.
    ///
    /// @return a descriptor for the format of the vertex data currently bound
    ///
    OsdVertexDescriptor const & GetVertexDescriptor() const {
        return _vdesc;
    }

    /// Returns the number of hierarchical edit tables
    int GetNumEditTables() const;

    /// Returns a specific hierarchical edit table
    ///
    /// @param tableIndex the index of the table
    ///
    const OsdCpuHEditTable * GetEditTable(int tableIndex) const;

    /// Returns a pointer to the vertex-interpolated data
    float * GetCurrentVertexBuffer() const;

    /// Returns a pointer to the varying-interpolated data
    float * GetCurrentVaryingBuffer() const;

protected:
    explicit OsdCpuComputeContext(FarMesh<OsdVertex> const *farMesh);

private:
    std::vector<OsdCpuTable*> _tables;
    std::vector<OsdCpuHEditTable*> _editTables;

    float *_currentVertexBuffer, 
          *_currentVaryingBuffer;

    OsdVertexDescriptor _vdesc;
};

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OSD_CPU_COMPUTE_CONTEXT_H


