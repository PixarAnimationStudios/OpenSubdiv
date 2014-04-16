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

#ifndef FAR_MESH_H
#define FAR_MESH_H

#include "../version.h"

#include "../far/subdivisionTables.h"
#include "../far/patchTables.h"
#include "../far/vertexEditTables.h"
#include "../far/kernelBatch.h"

#include <cassert>
#include <vector>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

/// \brief Feature Adaptive Mesh class.
///
/// FarMesh is a serialized instantiation of an HbrMesh. The HbrMesh contains
/// all the topological data in a highly interconnected data structure for 
/// ease of access and modification. When instantiating a FarMesh, the factory
/// analyzes this data structure and serializes the topology into a linear
/// buffers that are ready for efficient parallel processing.
///
template <class U> class FarMesh {
public:
    typedef U VertexType;

    /// \brief Constructor. FarMesh takes ownership of input tables.
    FarMesh(FarSubdivisionTables *subdivisionTables, FarPatchTables *patchTables,
            FarVertexEditTables *vertexEditTables, FarKernelBatchVector const &batches) :
        _subdivisionTables(subdivisionTables), _patchTables(patchTables),
        _vertexEditTables(vertexEditTables), _batches(batches) { }

    ~FarMesh();

    /// \brief Returns the subdivision method
    FarSubdivisionTables const * GetSubdivisionTables() const { return _subdivisionTables; }

    /// \brief Returns patch tables
    FarPatchTables const * GetPatchTables() const { return _patchTables; }

    /// \brief Returns the total number of vertices in the mesh across across all depths
    int GetNumVertices() const { return GetSubdivisionTables()->GetNumVertices(); }

    /// \brief Returns the list of vertices in the mesh (from subdiv level 0 to N)
    std::vector<U> & GetVertices() { return _vertices; }

    /// \brief Returns a reference to the vertex at the given index
    ///
    /// @param index the index fo the vertex
    ///
    U & GetVertex(int index) { return _vertices[index]; }

    /// \brief Returns vertex edit tables
    FarVertexEditTables const * GetVertexEditTables() const { return _vertexEditTables; }

    /// \brief True if the mesh tables support the feature-adaptive mode.
    bool IsFeatureAdaptive() const { return _patchTables->IsFeatureAdaptive(); }

    /// \brief Returns an ordered vector of batches of compute kernels. 
    /// The kernels describe the sequence of computations required to apply the 
    /// subdivision scheme to the vertices in the mesh.    
    const FarKernelBatchVector & GetKernelBatches() const { return _batches; }

private:
    // Note : the vertex classes are renamed <X,Y> so as not to shadow the 
    // declaration of the templated vertex class U.
    template <class X, class Y> friend class FarMeshFactory;

    FarMesh() : _subdivisionTables(0), _patchTables(0), _vertexEditTables(0) { }

    // non-copyable, so these are not implemented:
    FarMesh(FarMesh<U> const &);
    FarMesh<U> & operator = (FarMesh<U> const &);

    // subdivision method used in this mesh
    FarSubdivisionTables * _subdivisionTables;

    // tables of vertex indices for feature adaptive patches
    FarPatchTables * _patchTables;

    // hierarchical vertex edit tables
    FarVertexEditTables * _vertexEditTables;

    // kernel execution batches
    FarKernelBatchVector _batches;

    // list of vertices (up to N levels of subdivision)
    std::vector<U> _vertices;
};

template <class U>
FarMesh<U>::~FarMesh()
{
    delete _subdivisionTables;
    delete _patchTables;
    delete _vertexEditTables;
}

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* FAR_MESH_H */
