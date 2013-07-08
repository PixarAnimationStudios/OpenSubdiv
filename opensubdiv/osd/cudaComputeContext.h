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
