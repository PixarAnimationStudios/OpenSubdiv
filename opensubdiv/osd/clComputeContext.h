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

#include "../osd/nonCopyable.h"
#include "../osd/opencl.h"

#include <vector>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Far{ class StencilTables; }

namespace Osd {

///
/// \brief OpenCL Refine Context
///
/// The OpenCL-Compute implementation of the Refine module contextual functionality.
///
/// Contexts interface the serialized topological data pertaining to the
/// geometric primitives with the capabilities of the selected discrete
/// compute device.
///
class CLComputeContext : public NonCopyable<CLComputeContext> {

public:
    /// Creates an CLComputeContext instance
    ///
    /// @param clContext             An active OpenCL compute context
    ///
    /// @param vertexStencilTables   The Far::StencilTables used for vertex
    ///                              interpolation
    ///
    /// @param varyingStencilTables  The Far::StencilTables used for varying
    ///                              interpolation
    ///
    static CLComputeContext * Create(cl_context clContext,
                                        Far::StencilTables const * vertexStencilTables,
                                        Far::StencilTables const * varyingStencilTables=0);

    /// Destructor
    virtual ~CLComputeContext();


    /// Returns true if the Context has a 'vertex' interpolation stencil table
    bool HasVertexStencilTables() const;

    /// Returns true if the Context has a 'varying' interpolation stencil table
    bool HasVaryingStencilTables() const;

    /// Returns the number of control vertices
    int GetNumControlVertices() const {
        return _numControlVertices;
    }

    /// Returns the Cuda buffer containing vertex-stencil stencil sizes
    cl_mem GetVertexStencilTablesSizes() const;

    /// Returns the Cuda buffer containing vertex-stencil stencil offsets
    cl_mem GetVertexStencilTablesOffsets() const;

    /// Returns the Cuda buffer containing vertex-stencil stencil indices
    cl_mem GetVertexStencilTablesIndices() const;

    /// Returns the Cuda buffer containing vertex-stencil stencil weights
    cl_mem GetVertexStencilTablesWeights() const;


    /// Returns the Cuda buffer containing Varying-stencil stencil sizes
    cl_mem GetVaryingStencilTablesSizes() const;

    /// Returns the Cuda buffer containing Varying-stencil stencil offsets
    cl_mem GetVaryingStencilTablesOffsets() const;

    /// Returns the Cuda buffer containing Varying-stencil stencil indices
    cl_mem GetVaryingStencilTablesIndices() const;

    /// Returns the Cuda buffer containing Varying-stencil stencil weights
    cl_mem GetVaryingStencilTablesWeights() const;


protected:
    explicit CLComputeContext(Far::StencilTables const * vertexStencilTables,
                                 Far::StencilTables const * varyingStencilTables,
                                 cl_context clContext);

private:

    class CLStencilTables;

    CLStencilTables * _vertexStencilTables,
                    * _varyingStencilTables;

    int _numControlVertices;
};

}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OSD_CL_COMPUTE_CONTEXT_H
