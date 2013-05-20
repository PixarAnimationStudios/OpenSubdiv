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
        createCLBuffer(table.size() * sizeof(T), &table[0], clContext);
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
