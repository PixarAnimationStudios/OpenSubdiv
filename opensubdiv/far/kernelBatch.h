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
        CATMARK_FACE_VERTEX,
        CATMARK_EDGE_VERTEX,
        CATMARK_VERT_VERTEX_A1,
        CATMARK_VERT_VERTEX_A2,
        CATMARK_VERT_VERTEX_B,
        LOOP_EDGE_VERTEX,
        LOOP_VERT_VERTEX_A1,
        LOOP_VERT_VERTEX_A2,
        LOOP_VERT_VERTEX_B,
        BILINEAR_FACE_VERTEX,
        BILINEAR_EDGE_VERTEX,
        BILINEAR_VERT_VERTEX,
        HIERARCHICAL_EDIT,
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
    FarKernelBatch( KernelType kernelType,
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
    KernelType GetKernelType() const {
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
    template <class X, class Y> friend class FarMultiMeshFactory;

    KernelType _kernelType;
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
