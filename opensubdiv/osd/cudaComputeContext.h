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
#ifndef OSD_CUDA_COMPUTE_CONTEXT_H
#define OSD_CUDA_COMPUTE_CONTEXT_H

#include "../version.h"

#include "../far/vertexEditTables.h"
#include "../osd/vertex.h"
#include "../osd/vertexDescriptor.h"
#include "../osd/nonCopyable.h"

#include <stdlib.h>
#include <vector>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

class OsdCudaTable : OsdNonCopyable<OsdCudaTable> {
public:
    template<typename T>
    static OsdCudaTable * Create(const std::vector<T> &table) {
        OsdCudaTable *result = new OsdCudaTable();
        if (not result->createCudaBuffer(table.size() * sizeof(T), table.empty() ? NULL : &table[0])) {
            delete result;
            return NULL;
        }
        return result;
    }

    virtual ~OsdCudaTable();

    void * GetCudaMemory() const;

private:
    OsdCudaTable() : _devicePtr(NULL) {}

    bool createCudaBuffer(size_t size, const void *ptr);

    void *_devicePtr;
};

class OsdCudaHEditTable : OsdNonCopyable<OsdCudaHEditTable> {
public:
    static OsdCudaHEditTable * Create(const FarVertexEditTables<OsdVertex>::
                                      VertexEditBatch &batch);

    virtual ~OsdCudaHEditTable();

    const OsdCudaTable * GetPrimvarIndices() const;

    const OsdCudaTable * GetEditValues() const;

    int GetOperation() const;

    int GetPrimvarOffset() const;

    int GetPrimvarWidth() const;

private:
    OsdCudaHEditTable();

    OsdCudaTable *_primvarIndicesTable;
    OsdCudaTable *_editValuesTable;

    int _operation;
    int _primvarOffset;
    int _primvarWidth;
};

///
/// \brief CUDA Refine Context
///
/// The CUDA implementation of the Refine module contextual functionality. 
///
/// Contexts interface the serialized topological data pertaining to the 
/// geometric primitives with the capabilities of the selected discrete 
/// compute device.
///
class OsdCudaComputeContext : public OsdNonCopyable<OsdCudaComputeContext> {

public:
    /// Creates an OsdCudaComputeContext instance
    ///
    /// @param farmesh the FarMesh used for this Context.
    ///
    static OsdCudaComputeContext * Create(FarMesh<OsdVertex> const *farmesh);

    /// Destructor
    virtual ~OsdCudaComputeContext();

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

        if (vertex) {
            _currentVertexBuffer = static_cast<float*>(vertex->BindCudaBuffer());
            _vdesc.numVertexElements = vertex->GetNumElements();
        } else {
            _currentVertexBuffer = 0;
            _vdesc.numVertexElements = 0;
        }

        if (varying) {
            _currentVaryingBuffer = static_cast<float*>(varying->BindCudaBuffer());
            _vdesc.numVaryingElements = varying->GetNumElements();
        } else {
            _currentVaryingBuffer = 0;
            _vdesc.numVaryingElements = 0;
        }
    }

    /// Unbinds any previously bound vertex and varying data buffers.
    void Unbind() {
        _currentVertexBuffer = 0;
        _currentVaryingBuffer = 0;
    }

    /// Returns one of the vertex refinement tables.
    ///
    /// @param tableIndex the type of table
    ///
    const OsdCudaTable * GetTable(int tableIndex) const;

    /// Returns the number of hierarchical edit tables
    int GetNumEditTables() const;

    /// Returns a specific hierarchical edit table
    ///
    /// @param tableIndex the index of the table
    ///
    const OsdCudaHEditTable * GetEditTable(int tableIndex) const;

    /// Returns a pointer to the vertex-interpolated data
    float * GetCurrentVertexBuffer() const;

    /// Returns a pointer to the varying-interpolated data
    float * GetCurrentVaryingBuffer() const;

    /// Returns an OsdVertexDescriptor if vertex buffers have been bound.
    ///
    /// @return a descriptor for the format of the vertex data currently bound
    ///
    OsdVertexDescriptor const & GetVertexDescriptor() const {
        return _vdesc;
    }


protected:
    OsdCudaComputeContext();

    bool initialize(FarMesh<OsdVertex> const *farMesh);

private:
    std::vector<OsdCudaTable*> _tables;
    std::vector<OsdCudaHEditTable*> _editTables;

    
    float *_currentVertexBuffer, // cuda buffers
          *_currentVaryingBuffer;

    OsdVertexDescriptor _vdesc;
};

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OSD_CUDA_COMPUTE_CONTEXT_H
