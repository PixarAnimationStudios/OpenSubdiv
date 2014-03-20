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

#ifndef FAR_MULTI_MESH_FACTORY_H
#define FAR_MULTI_MESH_FACTORY_H

#include "../version.h"

#include "../far/mesh.h"

#include "../far/meshFactory.h"
#include "../far/patchTablesFactory.h"
#include "../far/vertexEditTablesFactory.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

/// \brief A specialized factory for batching meshes
///
/// Because meshes require multiple draw calls in order to process the different
/// types of patches, it is useful to have the ability of grouping the tables of
/// multiple meshes into a single set of tables. This factory builds upon the
/// specialized Far factories in order to provide this batching functionality.
///
template <class T, class U=T> class FarMultiMeshFactory  {

public:

    typedef std::vector<FarMesh<U> const *> FarMeshVector;

    /// \brief Constructor.
    FarMultiMeshFactory();
    
    /// \brief Splices a vector of Far meshes into a single Far mesh
    ///
    /// @param meshes  a vector of Far meshes to splice
    ///
    /// @return        the resulting spliced Far mesh
    ///
    FarMesh<U> * Create(std::vector<FarMesh<U> const *> const &meshes);

    std::vector<FarPatchTables::PatchArrayVector> const & GetMultiPatchArrays() {
        return _multiPatchArrays; 
    }

private:
    // patch arrays for each mesh
    std::vector<FarPatchTables::PatchArrayVector> _multiPatchArrays;
};

template <class T, class U>
FarMultiMeshFactory<T, U>::FarMultiMeshFactory() {
}

template <class T, class U> FarMesh<U> *
FarMultiMeshFactory<T, U>::Create(std::vector<FarMesh<U> const *> const &meshes) {

    if (meshes.empty()) return NULL;

    int totalFVarWidth = meshes[0]->GetTotalFVarWidth();
    FarSubdivisionTables::Scheme scheme = meshes[0]->GetSubdivisionTables()->GetScheme();

    for (size_t i = 0; i < meshes.size(); ++i) {
        FarMesh<U> const *mesh = meshes[i];

        // meshes have to have a same subdivision scheme
        if (scheme != mesh->GetSubdivisionTables()->GetScheme()) {
            assert(false);
            return NULL;
        }

        // meshes have to have a same fvardata width
        if (totalFVarWidth != mesh->GetTotalFVarWidth()) {
            assert(false);
            return NULL;
        }
    }

    FarMesh<U> * result = new FarMesh<U>();

    // splice subdivision tables
    result->_subdivisionTables = FarSubdivisionTablesFactory<T, U>::Splice(meshes, &result->_batches);

    // splice patch/quad index tables
    result->_patchTables = FarPatchTablesFactory<T>::Splice(meshes,
                                                            &_multiPatchArrays,
                                                            totalFVarWidth > 0);

    // splice vertex edit tables
    result->_vertexEditTables = FarVertexEditTablesFactory<T, U>::Splice(meshes);

    // count total num vertices, numptex faces
    int numVertices = 0, numPtexFaces = 0;
    for (size_t i = 0; i < meshes.size(); ++i) {
        numVertices += meshes[i]->GetNumVertices();
        numPtexFaces += meshes[i]->GetNumPtexFaces();
    }
    result->_vertices.resize(numVertices);
    result->_numPtexFaces = numPtexFaces;
    result->_totalFVarWidth = totalFVarWidth;

    return result;
}


} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* FAR_MULTI_MESH_FACTORY_H */
