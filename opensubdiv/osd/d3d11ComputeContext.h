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

#ifndef OSD_D3D11_COMPUTE_CONTEXT_H
#define OSD_D3D11_COMPUTE_CONTEXT_H

#include "../version.h"

#include "../osd/nonCopyable.h"

struct ID3D11DeviceContext;

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Far{ class StencilTables; }

namespace Osd {

///
/// \brief D3D Refine Context
///
/// The D3D implementation of the Refine module contextual functionality.
///
/// Contexts interface the serialized topological data pertaining to the
/// geometric primitives with the capabilities of the selected discrete
/// compute device.
///
class D3D11ComputeContext : public NonCopyable<D3D11ComputeContext> {
public:

    /// Creates an D3D11ComputeContext instance
    ///
    /// @param vertexStencilTables   The Far::StencilTables used for vertex
    ///                              interpolation
    ///
    /// @param varyingStencilTables  The Far::StencilTables used for varying
    ///                              interpolation
    ///
    /// @param deviceContext         The D3D device
    ///
    static D3D11ComputeContext * Create(ID3D11DeviceContext *deviceContext,
                                           Far::StencilTables const * vertexStencilTables,
                                           Far::StencilTables const * varyingStencilTables=0);

    /// Destructor
    virtual ~D3D11ComputeContext();

    /// Returns true if the Context has a 'vertex' interpolation stencil table
    bool HasVertexStencilTables() const;

    /// Returns true if the Context has a 'varying' interpolation stencil table
    bool HasVaryingStencilTables() const;

    /// Returns the number of control vertices
    int GetNumControlVertices() const {
        return _numControlVertices;
    }

    /// Binds D3D11 buffers containing stencils for 'vertex' interpolation
    ///
    /// @param deviceContext         The D3D device
    ///
    void BindVertexStencilTables(ID3D11DeviceContext *deviceContext) const;

    /// Binds D3D11 buffers containing stencils for 'varying' interpolation
    ///
    /// @param deviceContext         The D3D device
    ///
    void BindVaryingStencilTables(ID3D11DeviceContext *deviceContext) const;

    /// Unbinds D3D11 stencil buffers
    ///
    /// @param deviceContext         The D3D device
    ///
    void UnbindStencilTables(ID3D11DeviceContext *deviceContext) const;

protected:

    explicit D3D11ComputeContext(ID3D11DeviceContext *deviceContext,
                                    Far::StencilTables const * vertexStencilTables,
                                    Far::StencilTables const * varyingStencilTables);

private:

    class D3D11StencilTables;

    D3D11StencilTables * _vertexStencilTables,
                       * _varyingStencilTables;

    int _numControlVertices;
};

}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OSD_D3D11_COMPUTE_CONTEXT_H
