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
        createCpuBuffer(table.size() * sizeof(T), &table[0]);
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


