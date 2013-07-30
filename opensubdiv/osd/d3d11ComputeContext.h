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
#ifndef OSD_D3D11_COMPUTE_CONTEXT_H
#define OSD_D3D11_COMPUTE_CONTEXT_H

#include "../version.h"

#include "../far/vertexEditTables.h"
#include "../osd/vertex.h"
#include "../osd/vertexDescriptor.h"
#include "../osd/nonCopyable.h"

#include <D3D11.h>

#include <vector>

struct ID3D11Buffer;
struct ID3D11Device;
struct ID3D11DeviceContext;
struct ID3D11ShaderResourceView;

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

class OsdD3D11ComputeKernelBundle;

class OsdD3D11ComputeTable : OsdNonCopyable<OsdD3D11ComputeTable> {
public:
    template<typename T>
        OsdD3D11ComputeTable(const std::vector<T> &table, ID3D11DeviceContext *deviceContext, DXGI_FORMAT format) {
        createBuffer((int)table.size() * sizeof(T), table.empty() ? NULL : &table[0], format, (int)table.size(), deviceContext);
    }

    virtual ~OsdD3D11ComputeTable();

    ID3D11Buffer * GetBuffer() const;
    ID3D11ShaderResourceView * GetSRV() const;

private:
    void createBuffer(int size, const void *ptr, DXGI_FORMAT format, int numElements, ID3D11DeviceContext *deviceContext);

    ID3D11Buffer * _buffer;
    ID3D11ShaderResourceView * _srv;
};

class OsdD3D11ComputeHEditTable : OsdNonCopyable<OsdD3D11ComputeHEditTable> {
public:
    OsdD3D11ComputeHEditTable(const FarVertexEditTables<OsdVertex>::
                      VertexEditBatch &batch, ID3D11DeviceContext *deviceContext);

    virtual ~OsdD3D11ComputeHEditTable();

    const OsdD3D11ComputeTable * GetPrimvarIndices() const;

    const OsdD3D11ComputeTable * GetEditValues() const;

    int GetOperation() const;

    int GetPrimvarOffset() const;

    int GetPrimvarWidth() const;

private:
    OsdD3D11ComputeTable *_primvarIndicesTable;
    OsdD3D11ComputeTable *_editValuesTable;

    int _operation;
    int _primvarOffset;
    int _primvarWidth;
};

///
/// \brief D3D Refine Context
///
/// The D3D implementation of the Refine module contextual functionality. 
///
/// Contexts interface the serialized topological data pertaining to the 
/// geometric primitives with the capabilities of the selected discrete 
/// compute device.
///
class OsdD3D11ComputeContext : public OsdNonCopyable<OsdD3D11ComputeContext> {
public:
    /// Creates an OsdD3D11ComputeContext instance
    ///
    /// @param farmesh        the FarMesh used for this Context.
    ///
    /// @param deviceContext  D3D device
    ///
    static OsdD3D11ComputeContext * Create(FarMesh<OsdVertex> const *farmesh,
                                           ID3D11DeviceContext *deviceContext);

    /// Destructor
    virtual ~OsdD3D11ComputeContext();

    /// Binds a vertex and a varying data buffers to the context. Binding ensures
    /// that data buffers are properly inter-operated between Contexts and 
    /// Controllers operating across multiple devices.
    ///
    /// @param vertex a buffer containing vertex-interpolated primvar data
    ///
    /// @param varying a buffer containing varying-interpolated primvar data
    ///
    template<class VERTEX_BUFFER, class VARYING_BUFFER>
    void Bind(VERTEX_BUFFER *vertex, VARYING_BUFFER *varying) {

        _currentVertexBufferUAV = vertex ? vertex->BindD3D11UAV(_deviceContext) : 0;
        _currentVaryingBufferUAV = varying ? varying->BindD3D11UAV(_deviceContext) : 0;

        _vdesc.numVertexElements = vertex ? vertex->GetNumElements() : 0;
        _vdesc.numVaryingElements = varying ? varying->GetNumElements() : 0;

        bindShaderStorageBuffers();
    }

    /// Unbinds any previously bound vertex and varying data buffers.
    void Unbind() {
        _currentVertexBufferUAV = 0;
        _currentVaryingBufferUAV = 0;

        unbindShaderStorageBuffers();
    }

    /// Returns one of the vertex refinement tables.
    ///
    /// @param tableIndex the type of table
    ///
    const OsdD3D11ComputeTable * GetTable(int tableIndex) const;

    /// Returns the number of hierarchical edit tables
    int GetNumEditTables() const;

    /// Returns a specific hierarchical edit table
    ///
    /// @param tableIndex the index of the table
    ///
    const OsdD3D11ComputeHEditTable * GetEditTable(int tableIndex) const;

    /// Returns a handle to the vertex-interpolated buffer
    ID3D11UnorderedAccessView * GetCurrentVertexBufferUAV() const;

    /// Returns a handle to the varying-interpolated buffer
    ID3D11UnorderedAccessView * GetCurrentVaryingBufferUAV() const;

    /// Returns an OsdVertexDescriptor if vertex buffers have been bound.
    ///
    /// @return a descriptor for the format of the vertex data currently bound
    ///
    OsdVertexDescriptor const & GetVertexDescriptor() const {
        return _vdesc;
    }

    OsdD3D11ComputeKernelBundle * GetKernelBundle() const;

    void SetKernelBundle(OsdD3D11ComputeKernelBundle *kernelBundle);

    ID3D11DeviceContext * GetDeviceContext() const;

    void SetDeviceContext(ID3D11DeviceContext *deviceContext);

    void BindEditShaderStorageBuffers(int editIndex);

    void UnbindEditShaderStorageBuffers();

protected:
    explicit OsdD3D11ComputeContext(FarMesh<OsdVertex> const *farMesh, ID3D11DeviceContext *deviceContext);

    void bindShaderStorageBuffers();

    void unbindShaderStorageBuffers();

private:
    std::vector<OsdD3D11ComputeTable*> _tables;
    std::vector<OsdD3D11ComputeHEditTable*> _editTables;

    ID3D11DeviceContext *_deviceContext;

    OsdVertexDescriptor _vdesc;

    ID3D11UnorderedAccessView * _currentVertexBufferUAV,
                              * _currentVaryingBufferUAV;

    OsdD3D11ComputeKernelBundle * _kernelBundle;
};

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OSD_D3D11_COMPUTE_CONTEXT_H
