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

#ifndef FAR_KERNEL_BATCH_H
#define FAR_KERNEL_BATCH_H

#include "../version.h"

#include <vector>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {


/// \brief A GP Compute Kernel descriptor.
///
/// Vertex refinement through subdivision schemes requires the successive
/// application of dedicated compute kernels. OpenSubdiv groups these vertices
/// in batches based on their topology in order to minimize the number of kernel
/// switches to process a given primitive.
/// 
///       [Subdivision table for kernel k]
///        ------+---------------------------------+-----
///              |   Prim p, Level n               |
///        ------+---------------------------------+-----
///              |             | batch range |     |
///        ------+---------------------------------+-----
///              ^             ^             ^     
///        tableOffset       start          end
///                            .             .
///                            .             .
///                            .             .
///      [VBO]                 .             .
///        ------+---------------------------------+-----
///              |   Prim p, Kernel k, Level n     |
///        ------+---------------------------------+-----
///              |             | batch range |     | 
///        ------+---------------------------------+-----
///              ^             ^             ^
///        vertexOffset      start          end
///
class FarKernelBatch {

public:

    enum KernelType {
        FIRST_KERNEL_TYPE = 1,
        CATMARK_FACE_VERTEX = FIRST_KERNEL_TYPE,
        CATMARK_QUAD_FACE_VERTEX,
        CATMARK_TRI_QUAD_FACE_VERTEX,
        CATMARK_EDGE_VERTEX,
        CATMARK_RESTRICTED_EDGE_VERTEX,
        CATMARK_VERT_VERTEX_A1,
        CATMARK_VERT_VERTEX_A2,
        CATMARK_VERT_VERTEX_B,
        CATMARK_RESTRICTED_VERT_VERTEX_A,
        CATMARK_RESTRICTED_VERT_VERTEX_B1,
        CATMARK_RESTRICTED_VERT_VERTEX_B2,
        LOOP_EDGE_VERTEX,
        LOOP_VERT_VERTEX_A1,
        LOOP_VERT_VERTEX_A2,
        LOOP_VERT_VERTEX_B,
        BILINEAR_FACE_VERTEX,
        BILINEAR_EDGE_VERTEX,
        BILINEAR_VERT_VERTEX,
        HIERARCHICAL_EDIT,
        NUM_KERNEL_TYPES,
        USER_DEFINED_KERNEL_START = NUM_KERNEL_TYPES
    };

    /// \brief Constructor.
    /// 
    /// @param kernelType    the type of compute kernel kernel
    ///
    /// @param level         the level of subdivision of the vertices in the batch
    ///
    /// @param tableIndex    edit index (for the hierarchical edit kernels only)
    ///
    /// @param start         index of the first vertex in the batch
    ///
    /// @param end           index of the last vertex in the batch
    ///
    /// @param tableOffset   XXXX
    ///
    /// @param vertexOffset  XXXX
    ///
    /// @param meshIndex     XXXX
    ///
    FarKernelBatch( int kernelType,
                    int level,
                    int tableIndex,
                    int start,
                    int end,
                    int tableOffset,
                    int vertexOffset,
                    int meshIndex=0) :
        _kernelType(kernelType),
        _level(level),
        _tableIndex(tableIndex),
        _start(start),
        _end(end),
        _tableOffset(tableOffset),
        _vertexOffset(vertexOffset),
        _meshIndex(meshIndex) {
    }

    /// \brief Returns the type of kernel to apply to the vertices in the batch.
    int GetKernelType() const {
        return _kernelType;
    }


    /// \brief Returns the subdivision level of the vertices in the batch
    int GetLevel() const {
        return _level;
    }


    
    /// \brief Returns the index of the first vertex in the batch
    int GetStart() const {
        return _start;
    }

    /// \brief Returns the index of the first vertex in the batch
    const int * GetStartPtr() const {
        return & _start;
    }


    
    /// \brief Returns the index of the last vertex in the batch
    int GetEnd() const {
        return _end;
    }

    /// \brief Returns the index of the last vertex in the batch
    const int * GetEndPtr() const {
        return & _end;
    }



    /// \brief Returns the edit index (for the hierarchical edit kernels only)
    int GetTableIndex() const {
        return _tableIndex;
    }
    


    /// \brief Returns
    int GetTableOffset() const {
        return _tableOffset;
    }

    /// \brief Returns
    const int * GetTableOffsetPtr() const {
        return & _tableOffset;
    }



    /// \brief Returns
    int GetVertexOffset() const {
        return _vertexOffset;
    }

    /// \brief Returns
    const int * GetVertexOffsetPtr() const {
        return & _vertexOffset;
    }


    /// \brief Returns the mesh index (used in batching)
    int GetMeshIndex() const {
        return _meshIndex;
    }

private:
    friend class FarKernelBatchFactory;
    template <class X, class Y> friend class FarSubdivisionTablesFactory;

    int _kernelType;
    int _level;
    int _tableIndex;
    int _start;
    int _end;
    int _tableOffset;
    int _vertexOffset;
    int _meshIndex;
};

typedef std::vector<FarKernelBatch> FarKernelBatchVector;

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif  /* FAR_KERNEL_BATCH_H */
