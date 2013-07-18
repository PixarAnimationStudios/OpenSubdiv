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
#ifndef FAR_KERNEL_BATCH_FACTORY_H
#define FAR_KERNEL_BATCH_FACTORY_H

#include "../version.h"

#include "../far/kernelBatch.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {


class FarVertexKernelBatchFactory {

public:

    /// \brief Constructor.
    ///
    /// @param start index of the first vertex in the batch
    ///
    /// @param end   index of the last vertex in the batch
    ///
    FarVertexKernelBatchFactory(int start, int end) {
        kernelB.start = kernelA1.start = kernelA2.start = start;
        kernelB.end = kernelA1.end = kernelA2.end = end;
    }


    /// \brief Adds a vertex-vertex to the appropriate compute batch based on "Rank". 
    ///
    /// Ranking is based on the interpolation required (Smooth, Dart, Crease,
    /// or Corner). With semi-sharp creases, two passes of interpolation are
    /// required, from which we derive a matrix of compute kernel combinations.
    ///
    /// The kernel combinatorial matrix :
    ///
#ifndef doxygen    
    /// Rules     +----+----+----+----+----+----+----+----+----+----+
    ///   Pass 0  | Dt | Sm | Sm | Dt | Sm | Dt | Sm | Cr | Co | Cr |
    ///   Pass 1  |    |    |    | Co | Co | Cr | Cr | Co |    |    |
    /// Kernel    +----+----+----+----+----+----+----+----+----+----+
    ///   Pass 0  | B  | B  | B  | B  | B  | B  | B  | A  | A  | A  |
    ///   Pass 1  |    |    |    | A  | A  | A  | A  | A  |    |    |
    ///           +----+----+----+----+----+----+----+----+----+----+
    /// Rank      | 0  | 1  | 2  | 3  | 4  | 5  | 6  | 7  | 8  | 9  |
    ///           +----+----+----+----+----+----+----+----+----+----+
#else
    /// <table class="doxtable">
    /// <tr> <th colspan="11" align="left">Rules</th></tr>
    /// <tr align="center"> <td>Pass 0</td> <td>Dart</td> <td>Smooth</td> 
    ///     <td>Smooth</td> <td>Dart  </td> <td>Smooth</td> <td>Dart  </td> 
    ///     <td>Smooth</td> <td>Crease</td> <td>Corner</td> <td>Crease</td> </tr>
    /// <tr align="center"> <td>Pass 1</td> <td>    </td> <td>      </td> 
    ///     <td>      </td> <td>Corner</td> <td>Corner</td> <td>Crease</td> 
    ///     <td>Crease</td> <td>Corner</td> <td>      </td> <td>      </td> </tr>
    /// <tr> <th colspan="11" align="left">Kernel</th></tr>
    /// <tr align="center"> <td>Pass 0</td> <td>  B </td> <td>  B   </td> 
    ///      <td>  B   </td> <td>  B   </td> <td>  B   </td> <td>  B   </td> 
    ///      <td>  B   </td> <td>  A   </td> <td>  A   </td> <td>  A   </td> </tr>
    /// <tr align="center"> <td>Pass 1</td> <td>    </td> <td>      </td> 
    ///      <td>      </td> <td>  A   </td> <td>  A   </td> <td>  A   </td> 
    ///      <td>  A   </td> <td>  A   </td> <td>      </td> <td>      </td> </tr>    
    /// </table>
#endif    
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



    /// \brief Appends a FarKernelBatch to a vector of batches for Catmark subdivision
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



    /// \brief Appends a FarKernelBatch to a vector of batches for Loop subdivision
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
