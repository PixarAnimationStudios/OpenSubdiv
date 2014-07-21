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

#ifndef OSD_CUDA_COMPUTE_CONTEXT_H
#define OSD_CUDA_COMPUTE_CONTEXT_H

#include "../version.h"

#include "../far/subdivisionTables.h"
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
    static OsdCudaTable * Create(cudaStream_t stream, const std::vector<T> &table) {
        OsdCudaTable *result = new OsdCudaTable();
        if (not result->createCudaBuffer(stream, table.size() * sizeof(T), table.empty() ? NULL : &table[0])) {
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
    static OsdCudaHEditTable * Create(const FarVertexEditTables::VertexEditBatch &batch);

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
    /// @param subdivisionTables the FarSubdivisionTables used for this Context.
    ///
    /// @param vertexEditTables the FarVertexEditTables used for this Context.
    ///
    static OsdCudaComputeContext * Create(FarSubdivisionTables const *subdivisionTables,
                                          FarVertexEditTables const *vertexEditTables);

    /// Destructor
    virtual ~OsdCudaComputeContext();

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

    cudaStream_t GetStream();

protected:
    OsdCudaComputeContext();

    bool initialize(FarSubdivisionTables const *subdivisionTables,
                    FarVertexEditTables const *vertexEditTables);

private:
    std::vector<OsdCudaTable*> _tables;
    std::vector<OsdCudaHEditTable*> _editTables;
	cudaStream_t *_stream;
};

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OSD_CUDA_COMPUTE_CONTEXT_H
