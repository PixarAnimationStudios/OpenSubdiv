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

#include "../far/stencilTables.h"

#include "../osd/error.h"
#include "../osd/clComputeContext.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

// -----------------------------------------------------------------------------

template <class T> cl_mem
createCLBuffer(std::vector<T> const & src, cl_context clContext) {

    cl_mem devicePtr=0; cl_int errNum;

    devicePtr = clCreateBuffer(clContext, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,
            src.size()*sizeof(T), (void*)(&src.at(0)), &errNum);

    if (errNum!=CL_SUCCESS) {
        Error(OSD_CL_RUNTIME_ERROR, "clCreateBuffer: %d", errNum);
    }

    return devicePtr;
}

// -----------------------------------------------------------------------------

class CLComputeContext::CLStencilTables {

public:

    CLStencilTables(Far::StencilTables const & stencilTables, cl_context clContext) {
        _sizes = createCLBuffer(stencilTables.GetSizes(), clContext);
        _offsets = createCLBuffer(stencilTables.GetOffsets(), clContext);
        _indices = createCLBuffer(stencilTables.GetControlIndices(), clContext);
        _weights = createCLBuffer(stencilTables.GetWeights(), clContext);
    }

    ~CLStencilTables() {
        if (_sizes) clReleaseMemObject(_sizes);
        if (_offsets) clReleaseMemObject(_offsets);
        if (_indices) clReleaseMemObject(_indices);
        if (_weights) clReleaseMemObject(_weights);
    }

    bool IsValid() const {
        return _sizes and _offsets and _indices and _weights;
    }

    cl_mem GetSizes() const {
        return _sizes;
    }

    cl_mem GetOffsets() const {
        return _offsets;
    }

    cl_mem GetIndices() const {
        return _indices;
    }

    cl_mem GetWeights() const {
        return _weights;
    }

private:

    cl_mem _sizes,
           _offsets,
           _indices,
           _weights;
};

// -----------------------------------------------------------------------------

CLComputeContext::CLComputeContext(
    Far::StencilTables const * vertexStencilTables,
        Far::StencilTables const * varyingStencilTables,
            cl_context clContext) :
                _vertexStencilTables(0), _varyingStencilTables(0),
                     _numControlVertices(0) {

    if (vertexStencilTables) {
        _vertexStencilTables = new CLStencilTables(*vertexStencilTables, clContext);
        _numControlVertices = vertexStencilTables->GetNumControlVertices();
    }

    if (varyingStencilTables) {
        _varyingStencilTables = new CLStencilTables(*varyingStencilTables, clContext);

        if (_numControlVertices) {
            assert(_numControlVertices==varyingStencilTables->GetNumControlVertices());
        } else {
            _numControlVertices = varyingStencilTables->GetNumControlVertices();
        }
    }
}

CLComputeContext::~CLComputeContext() {
    delete _vertexStencilTables;
    delete _varyingStencilTables;
}

// ----------------------------------------------------------------------------

bool
CLComputeContext::HasVertexStencilTables() const {
    return _vertexStencilTables ? _vertexStencilTables->IsValid() : false;
}

bool
CLComputeContext::HasVaryingStencilTables() const {
    return _varyingStencilTables ? _varyingStencilTables->IsValid() : false;
}

// ----------------------------------------------------------------------------

cl_mem
CLComputeContext::GetVertexStencilTablesSizes() const {
    return _vertexStencilTables ? _vertexStencilTables->GetSizes() : 0;
}

cl_mem
CLComputeContext::GetVertexStencilTablesOffsets() const {
    return _vertexStencilTables ? _vertexStencilTables->GetOffsets() : 0;
}

cl_mem
CLComputeContext::GetVertexStencilTablesIndices() const {
    return _vertexStencilTables ? _vertexStencilTables->GetIndices() : 0;
}

cl_mem
CLComputeContext::GetVertexStencilTablesWeights() const {
    return _vertexStencilTables ? _vertexStencilTables->GetWeights() : 0;
}

// ----------------------------------------------------------------------------

cl_mem
CLComputeContext::GetVaryingStencilTablesSizes() const {
    return _varyingStencilTables ? _varyingStencilTables->GetSizes() : 0;
}

cl_mem
CLComputeContext::GetVaryingStencilTablesOffsets() const {
    return _varyingStencilTables ? _varyingStencilTables->GetOffsets() : 0;
}

cl_mem
CLComputeContext::GetVaryingStencilTablesIndices() const {
    return _varyingStencilTables ? _varyingStencilTables->GetIndices() : 0;
}

cl_mem
CLComputeContext::GetVaryingStencilTablesWeights() const {
    return _varyingStencilTables ? _varyingStencilTables->GetWeights() : 0;
}


// -----------------------------------------------------------------------------

CLComputeContext *
CLComputeContext::Create(cl_context clContext,
    Far::StencilTables const * vertexStencilTables,
        Far::StencilTables const * varyingStencilTables) {

    CLComputeContext *result =
        new CLComputeContext(
            vertexStencilTables, varyingStencilTables, clContext);

    return result;
}

// -----------------------------------------------------------------------------
}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
