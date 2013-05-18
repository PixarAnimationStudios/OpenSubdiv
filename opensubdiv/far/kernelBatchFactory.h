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
#ifndef FAR_KERNEL_BATCH_FACTORY_H
#define FAR_KERNEL_BATCH_FACTORY_H

#include "../version.h"

#include "../far/kernelBatch.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {


class FarVertexKernelBatchFactory {

public:

    /// Constructor.
    ///
    /// @param start index of the first vertex in the batch
    ///
    /// @param end   index of the last vertex in the batch
    ///
    FarVertexKernelBatchFactory(int start, int end) {
        kernelB.start = kernelA1.start = kernelA2.start = start;
        kernelB.end = kernelA1.end = kernelA2.end = end;
    }


    /// Adds a vertex-vertex to the appropriate compute batch based on "Rank". 
    /// Ranking is based on the interpolation required (Smooth, Dart, Crease,
    /// or Corner). With semi-sharp creases, two passes of interpolation are
    /// required, from which we derive a matrix of compute kernel combinations.
    ///
    /// The kernel combinatorial matrix :
    ///
    /// Rules     +----+----+----+----+----+----+----+----+----+----+
    ///   Pass 0  | Dt | Sm | Sm | Dt | Sm | Dt | Sm | Cr | Co | Cr |
    ///   Pass 1  |    |    |    | Co | Co | Cr | Cr | Co |    |    |
    /// Kernel    +----+----+----+----+----+----+----+----+----+----+
    ///   Pass 0  | B  | B  | B  | B  | B  | B  | B  | A  | A  | A  |
    ///   Pass 1  |    |    |    | A  | A  | A  | A  | A  |    |    |
    ///           +----+----+----+----+----+----+----+----+----+----+
    /// Rank      | 0  | 1  | 2  | 3  | 4  | 5  | 6  | 7  | 8  | 9  |
    ///           +----+----+----+----+----+----+----+----+----+----+
    /// with :
    ///     - A : compute kernel applying k_Crease / k_Corner rules
    ///     - B : compute kernel applying k_Smooth / k_Dart rules
    ///
    /// @param index the index of the vertex
    ///
    /// @param rank  the rank of the vertex (see 
    ///              FarSubdivisionTables::GetMaskRanking())
    ///
    void AddVertex( int index, int rank );



    /// Appends a FarKernelBatch to a vector of batches for Catmark subdivision
    ///
    /// @param level         the subdivision level of the vertices in the batch
    ///
    /// @param tableOffset   XXXX
    ///
    /// @param vertexOffset  XXXX
    ///
    /// @param result        the expanded batch vector
    ///
    void AppendCatmarkBatches(int level, int tableOffset, int vertexOffset, FarKernelBatchVector *result);



    /// Appends a FarKernelBatch to a vector of batches for Loop subdivision
    ///
    /// @param level         the subdivision level of the vertices in the batch
    ///
    /// @param tableOffset   XXXX
    ///
    /// @param vertexOffset  XXXX
    ///
    /// @param result        the expanded batch vector
    ///
    void AppendLoopBatches(int level, int tableOffset, int vertexOffset, FarKernelBatchVector *result);

private:

    struct Range {
        int start,
            end;
    };

    Range kernelB;  // vertex batch reange (kernel B)
    Range kernelA1; // vertex batch reange (kernel A pass 1)
    Range kernelA2; // vertex batch reange (kernel A pass 2)
};

inline void 
FarVertexKernelBatchFactory::AddVertex( int index, int rank ) {

    // expand the range of kernel batches based on vertex index and rank
    if (rank<7) {
        if (index < kernelB.start)
            kernelB.start=index;
        if (index > kernelB.end)
            kernelB.end=index;
    }
    if ((rank>2) and (rank<8)) {
        if (index < kernelA2.start)
            kernelA2.start=index;
        if (index > kernelA2.end)
            kernelA2.end=index;
    }
    if (rank>6) {
        if (index < kernelA1.start)
            kernelA1.start=index;
        if (index > kernelA1.end)
            kernelA1.end=index;
    }
}

inline void 
FarVertexKernelBatchFactory::AppendCatmarkBatches(int level, 
                                                  int tableOffset, 
                                                  int vertexOffset, 
                                                  FarKernelBatchVector *result) {

    if (kernelB.end >= kernelB.start)
        result->push_back(FarKernelBatch( FarKernelBatch::CATMARK_VERT_VERTEX_B, level, 0,
                                          kernelB.start, kernelB.end+1,
                                          tableOffset, vertexOffset) );
    if (kernelA1.end >= kernelA1.start)
        result->push_back(FarKernelBatch( FarKernelBatch::CATMARK_VERT_VERTEX_A1, level, 0,
                                          kernelA1.start, kernelA1.end+1,
                                          tableOffset, vertexOffset));
    if (kernelA2.end >= kernelA2.start)
        result->push_back(FarKernelBatch( FarKernelBatch::CATMARK_VERT_VERTEX_A2, level, 0,
                                          kernelA2.start, kernelA2.end+1,
                                          tableOffset, vertexOffset) );
}

inline void 
FarVertexKernelBatchFactory::AppendLoopBatches(int level, 
                                               int tableOffset, 
                                               int vertexOffset, 
                                               FarKernelBatchVector *result) {
    if (kernelB.end >= kernelB.start)
        result->push_back(FarKernelBatch( FarKernelBatch::LOOP_VERT_VERTEX_B, level, 0,
                                          kernelB.start, kernelB.end+1,
                                          tableOffset, vertexOffset) );
    if (kernelA1.end >= kernelA1.start)
        result->push_back(FarKernelBatch( FarKernelBatch::LOOP_VERT_VERTEX_A1, level, 0,
                                          kernelA1.start, kernelA1.end+1,
                                          tableOffset, vertexOffset) );
    if (kernelA2.end >= kernelA2.start)
        result->push_back(FarKernelBatch( FarKernelBatch::LOOP_VERT_VERTEX_A2, level, 0,
                                          kernelA2.start, kernelA2.end+1,
                                          tableOffset, vertexOffset) );
}

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif  /* FAR_KERNEL_BATCH_FACTORY_H */
