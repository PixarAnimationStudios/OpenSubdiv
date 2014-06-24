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

#ifndef OSD_CL_COMPUTE_CONTEXT_H
#define OSD_CL_COMPUTE_CONTEXT_H

#include "../version.h"

#include "../far/subdivisionTables.h"
#include "../far/vertexEditTables.h"
#include "../osd/vertex.h"
#include "../osd/nonCopyable.h"
#include "../osd/opencl.h"

#include <vector>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

class OsdCLKernelBundle;


class OsdCLTable : OsdNonCopyable<OsdCLTable> {
public:
    template<typename T>
    OsdCLTable(const std::vector<T> &table, cl_context clContext) {
        createCLBuffer(table.size() * sizeof(T), table.empty() ? NULL : &table[0], clContext);
    }

    virtual ~OsdCLTable();

    cl_mem GetDevicePtr() const;

private:
    void createCLBuffer(size_t size, const void *ptr, cl_context clContext);
    cl_mem _devicePtr;
};


class OsdCLHEditTable : OsdNonCopyable<OsdCLHEditTable> {
public:
    OsdCLHEditTable(const FarVertexEditTables::
                    VertexEditBatch &batch, cl_context clContext);

    virtual ~OsdCLHEditTable();

    const OsdCLTable * GetPrimvarIndices() const;

    const OsdCLTable * GetEditValues() const;

    int GetOperation() const;

    int GetPrimvarOffset() const;

    int GetPrimvarWidth() const;

private:
    OsdCLTable *_primvarIndicesTable;
    OsdCLTable *_editValuesTable;

    int _operation;
    int _primvarOffset;
    int _primvarWidth;
};

///
/// \brief OpenCL Refine Context
///
/// The OpenCL implementation of the Refine module contextual functionality. 
///
/// Contexts interface the serialized topological data pertaining to the 
/// geometric primitives with the capabilities of the selected discrete 
/// compute device.
///
class OsdCLComputeContext : public OsdNonCopyable<OsdCLComputeContext> {

public:
    /// Creates an OsdCLComputeContext instance
    ///
    /// @param subdivisionTables the FarSubdivisionTables used for this Context.
    ///
    /// @param vertexEditTables the FarVertexEditTables used for this Context.
    ///
    /// @param clContext  a valid active OpenCL context
    ///
    static OsdCLComputeContext * Create(FarSubdivisionTables const *subdivisionTables,
                                        FarVertexEditTables const *vertexEditTables,
                                        cl_context clContext);

    /// Destructor
    virtual ~OsdCLComputeContext();

    /// Returns one of the vertex refinement tables.
    ///
    /// @param tableIndex the type of table
    ///
    const OsdCLTable * GetTable(int tableIndex) const;

    /// Returns the number of hierarchical edit tables
    int GetNumEditTables() const;

    /// Returns a specific hierarchical edit table
    ///
    /// @param tableIndex the index of the table
    ///
    const OsdCLHEditTable * GetEditTable(int tableIndex) const;

protected:
    explicit OsdCLComputeContext(FarSubdivisionTables const *subdivisionTables,
                                 FarVertexEditTables const *vertexEditTables,
                                 cl_context clContext);

private:
    std::vector<OsdCLTable*> _tables;
    std::vector<OsdCLHEditTable*> _editTables;
};

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OSD_CL_COMPUTE_CONTEXT_H
