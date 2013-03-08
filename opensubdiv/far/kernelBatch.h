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

struct FarKernelBatch {
    FarKernelBatch(int level_,
                   int kernelType_,
                   int tableIndex_,
                   int start_,
                   int end_,
                   int tableOffset_,
                   int vertexOffset_) :
        level(level_),
        kernelType(kernelType_),
        tableIndex(tableIndex_),
        start(start_),
        end(end_),
        tableOffset(tableOffset_),
        vertexOffset(vertexOffset_) {
    }

    /*
      [Subdivision table for kernel k]
        ------+---------------------------------+-----
              |   Prim p, Level n               |
        ------+---------------------------------+-----
              |             |<batch range>|     |
        ------+---------------------------------+-----
              ^             ^             ^     
        tableOffset       start          end
                            .             .
                            .             .
                            .             .
      [VBO]                 .             .
        ------+---------------------------------+-----
              |   Prim p, Kernel k, Level n     |
        ------+---------------------------------+-----
              |             |<batch range>|     | 
        ------+---------------------------------+-----
              ^             ^             ^
        vertexOffset      start          end
    */

    int level;
    int kernelType;
    int tableIndex;  // edit index (for the h-edit kernel only)

    int start;
    int end;
    int tableOffset;
    int vertexOffset;
};

typedef std::vector<FarKernelBatch> FarKernelBatchVector;

struct FarVertexKernelBatchFactory {
    FarVertexKernelBatchFactory(int a, int b) {
        kernelB.first = kernelA1.first = kernelA2.first = a;
        kernelB.second = kernelA1.second = kernelA2.second = b;
    }
    
    void AddVertex( int index, int rank ) {
        // expand the range of kernel batches based on vertex index and rank
        if (rank<7) {
            if (index < kernelB.first)
                kernelB.first=index;
            if (index > kernelB.second)
                kernelB.second=index;
        }
        if ((rank>2) and (rank<8)) {
            if (index < kernelA2.first)
                kernelA2.first=index;
            if (index > kernelA2.second)
                kernelA2.second=index;
        }
        if (rank>6) {
            if (index < kernelA1.first)
                kernelA1.first=index;
            if (index > kernelA1.second)
                kernelA1.second=index;
        }
    }

    void AppendCatmarkBatches(FarKernelBatchVector *result, int level, int tableOffset, int vertexOffset) {
        if (kernelB.second >= kernelB.first)
            result->push_back(FarKernelBatch(level, CATMARK_VERT_VERTEX_B, 0,
                                             kernelB.first, kernelB.second+1,
                                             tableOffset, vertexOffset));
        if (kernelA1.second >= kernelA1.first)
            result->push_back(FarKernelBatch(level, CATMARK_VERT_VERTEX_A1, 0,
                                             kernelA1.first, kernelA1.second+1,
                                             tableOffset, vertexOffset));
        if (kernelA2.second >= kernelA2.first)
            result->push_back(FarKernelBatch(level, CATMARK_VERT_VERTEX_A2, 0,
                                             kernelA2.first, kernelA2.second+1,
                                             tableOffset, vertexOffset));
    }

    void AppendLoopBatches(FarKernelBatchVector *result, int level, int tableOffset, int vertexOffset) {
        if (kernelB.second >= kernelB.first)
            result->push_back(FarKernelBatch(level, LOOP_VERT_VERTEX_B, 0,
                                             kernelB.first, kernelB.second+1,
                                             tableOffset, vertexOffset));
        if (kernelA1.second >= kernelA1.first)
            result->push_back(FarKernelBatch(level, LOOP_VERT_VERTEX_A1, 0,
                                             kernelA1.first, kernelA1.second+1,
                                             tableOffset, vertexOffset));
        if (kernelA2.second >= kernelA2.first)
            result->push_back(FarKernelBatch(level, LOOP_VERT_VERTEX_A2, 0,
                                             kernelA2.first, kernelA2.second+1,
                                             tableOffset, vertexOffset));
    }

    std::pair<int,int> kernelB;  // first / last vertex vertex batch (kernel B)
    std::pair<int,int> kernelA1; // first / last vertex vertex batch (kernel A pass 1)
    std::pair<int,int> kernelA2; // first / last vertex vertex batch (kernel A pass 2)
};

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif  /* FAR_KERNEL_BATCH_H */
