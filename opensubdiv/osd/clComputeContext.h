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
#ifndef OSD_CL_COMPUTE_CONTEXT_H
#define OSD_CL_COMPUTE_CONTEXT_H

#include "../version.h"

#include "../far/vertexEditTables.h"
#include "../osd/vertex.h"
#include "../osd/nonCopyable.h"

#if defined(__APPLE__)
    #include <OpenCL/opencl.h>
#else
    #include <CL/opencl.h>
#endif

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
    OsdCLHEditTable(const FarVertexEditTables<OsdVertex>::
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
    /// @param farmesh    the FarMesh used for this Context.
    ///
    /// @param clContext  a valid active OpenCL context
    ///
    static OsdCLComputeContext * Create(FarMesh<OsdVertex> const *farmesh,
                                        cl_context clContext);

    /// Destructor
    virtual ~OsdCLComputeContext();

    /// Binds a vertex and a varying data buffers to the context. Binding ensures
    /// that data buffers are properly inter-operated between Contexts and 
    /// Controllers operating across multiple devices.
    ///
    /// @param vertex   a buffer containing vertex-interpolated primvar data
    ///
    /// @param varying  a buffer containing varying-interpolated primvar data
    ///
    /// @param clQueue  OpenCL command queue associated with the primvar data
    ///
    template<class VERTEX_BUFFER, class VARYING_BUFFER>
        void Bind(VERTEX_BUFFER *vertex, VARYING_BUFFER *varying, cl_command_queue clQueue) {

        _currentVertexBuffer = vertex ? vertex->BindCLBuffer(clQueue) : NULL;
        _currentVaryingBuffer = varying ? varying->BindCLBuffer(clQueue) : NULL;

        _clQueue = clQueue;
    }

    /// Unbinds any previously bound vertex and varying data buffers.
    void Unbind() {
        _currentVertexBuffer = NULL;
        _currentVaryingBuffer = NULL;
        _clQueue = NULL;
        _kernelBundle = NULL;
    }

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

    /// Returns a CL handle to the vertex-interpolated data
    cl_mem GetCurrentVertexBuffer() const;

    /// Returns a CL handle to the varying-interpolated data
    cl_mem GetCurrentVaryingBuffer() const;

    OsdCLKernelBundle * GetKernelBundle() const;

    void SetKernelBundle(OsdCLKernelBundle *kernelBundle);

    cl_command_queue GetCommandQueue() const;

    void SetCommandQueue(cl_command_queue queue);

protected:
    explicit OsdCLComputeContext(FarMesh<OsdVertex> const *farMesh,
                                 cl_context clContext);

private:
    std::vector<OsdCLTable*> _tables;
    std::vector<OsdCLHEditTable*> _editTables;

    cl_mem _currentVertexBuffer, 
           _currentVaryingBuffer;

    cl_command_queue _clQueue;

    OsdCLKernelBundle *_kernelBundle;
};

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OSD_CL_COMPUTE_CONTEXT_H
