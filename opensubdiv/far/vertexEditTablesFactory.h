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

#ifndef FAR_VERTEX_EDIT_TABLES_FACTORY_H
#define FAR_VERTEX_EDIT_TABLES_FACTORY_H

#include "../version.h"

#include "../hbr/vertexEdit.h"

#include "../far/vertexEditTables.h"
#include "../far/kernelBatch.h"

#include <cassert>
#include <vector>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

/// \brief A specialized factory for FarVertexEditTables
///
/// This factory is private to Far and should not be used by client code.
///
template <class T, class U> class FarVertexEditTablesFactory {

protected:
    template <class X, class Y> friend class FarMeshFactory;

    /// \brief Compares the number of subfaces in an edit (for sorting purposes)
    static bool compareEdits(HbrVertexEdit<T> const *a, HbrVertexEdit<T> const *b);
    
    static void insertHEditBatch(FarKernelBatchVector *batches, int batchIndex, int batchLevel, int batchCount, int tableOffset);

    /// \brief Creates a FarVertexEditTables instance.
    static FarVertexEditTables<U> * Create( FarMeshFactory<T,U> const * factory, FarMesh<U> * mesh, FarKernelBatchVector *batches, int maxlevel );
};

template <class T, class U> bool
FarVertexEditTablesFactory<T,U>::compareEdits(HbrVertexEdit<T> const *a, HbrVertexEdit<T> const *b) {

    return a->GetNSubfaces() < b->GetNSubfaces();
}

template <class T, class U> void
FarVertexEditTablesFactory<T,U>::insertHEditBatch(FarKernelBatchVector *batches, int batchIndex, int batchLevel, int batchCount, int tableOffset) {

    FarKernelBatchVector::iterator it = batches->begin();

    while (it != batches->end()) {
        if (it->GetLevel() > batchLevel+1) 
            break;
        ++it;
    }
    
    batches->insert(it, FarKernelBatch( FarKernelBatch::HIERARCHICAL_EDIT, batchLevel+1, batchIndex, 0, batchCount, tableOffset, 0) );
}

template <class T, class U> FarVertexEditTables<U> * 
FarVertexEditTablesFactory<T,U>::Create( FarMeshFactory<T,U> const * factory, FarMesh<U> * mesh, FarKernelBatchVector *batches, int maxlevel ) {

    assert( factory and mesh );

    FarVertexEditTables<U> * result = new FarVertexEditTables<U>(mesh);

    std::vector<HbrHierarchicalEdit<T>*> const & hEdits = factory->_hbrMesh->GetHierarchicalEdits();

    std::vector<HbrVertexEdit<T> const *> vertexEdits;
    vertexEdits.reserve(hEdits.size());

    for (int i=0; i<(int)hEdits.size(); ++i) {
        HbrVertexEdit<T> *vedit = dynamic_cast<HbrVertexEdit<T> *>(hEdits[i]);
        if (vedit) {
            int editlevel = vedit->GetNSubfaces();
            if (editlevel > maxlevel)
                continue;   // far table doesn't contain such level

            vertexEdits.push_back(vedit);
        }
    }

    // sort vertex edits by level
    std::sort(vertexEdits.begin(), vertexEdits.end(), compareEdits);

    // First pass : count batches based on operation and primvar being edited
    std::vector<int> batchIndices;
    std::vector<int> batchSizes;
    for(int i=0; i<(int)vertexEdits.size(); ++i) {
        HbrVertexEdit<T> const *vedit = vertexEdits[i];

        // translate operation enum
        FarVertexEdit::Operation op = (vedit->GetOperation() == HbrHierarchicalEdit<T>::Set) ?
            FarVertexEdit::Set : FarVertexEdit::Add;

        // determine which batch this edit belongs to (create it if necessary)
        // XXXX manuelk - if the number of edits becomes large, we may need to switch this
        // to a map.
        int batchIndex = -1;
        for(int i = 0; i<(int)result->_batches.size(); ++i) {
            if(result->_batches[i]._primvarIndex == vedit->GetIndex() &&
               result->_batches[i]._primvarWidth == vedit->GetWidth() &&
               result->_batches[i]._op == op) {
                batchIndex = i;
                break;
            }
        }
        if (batchIndex == -1) {
            // create new batch
            batchIndex = (int)result->_batches.size();
            result->_batches.push_back(typename FarVertexEditTables<U>::VertexEditBatch(vedit->GetIndex(), vedit->GetWidth(), op));
            batchSizes.push_back(0);
        }
        batchSizes[batchIndex]++;
        batchIndices.push_back(batchIndex);
    }

    // Second pass : populate the batches
    int numBatches = result->GetNumBatches();
    for(int i=0; i<numBatches; ++i) {
        result->_batches[i]._vertIndices.resize(batchSizes[i]);
        result->_batches[i]._edits.resize(batchSizes[i] * result->_batches[i].GetPrimvarWidth());
    }

    // Resolve vertexedits path to absolute offset and put them into corresponding batch
    std::vector<int> currentLevels(numBatches);
    std::vector<int> currentCounts(numBatches);
    std::vector<int> currentOffsets(numBatches);
    for(int i=0; i<(int)vertexEdits.size(); ++i){
        HbrVertexEdit<T> const *vedit = vertexEdits[i];

        HbrFace<T> * f = factory->_hbrMesh->GetFace(vedit->GetFaceID());

        int level = vedit->GetNSubfaces();
        for (int j=0; j<level; ++j)
            f = f->GetChild(vedit->GetSubface(j));

        int vertexID = f->GetVertex(vedit->GetVertexID())->GetID();

        // Remap vertex ID
        vertexID = factory->_remapTable[vertexID];

        int batchIndex = batchIndices[i];
        int & batchLevel = currentLevels[batchIndex];
        int & batchCount = currentCounts[batchIndex];
        typename FarVertexEditTables<U>::VertexEditBatch &batch = result->_batches[batchIndex];

        // if a new batch is needed
        if (batchLevel != level-1) {
            // if the current batch isn't empty, emit a kernelBatch
            if (batchCount != currentOffsets[batchIndex]) {
                insertHEditBatch(batches, batchIndex, batchLevel, batchCount-currentOffsets[batchIndex], currentOffsets[batchIndex]);
            }
            // prepare a next batch
            batchLevel = level-1;
            currentOffsets[batchIndex] = batchCount;
        }

        // Set absolute vertex index
        batch._vertIndices[batchCount] = vertexID;

        // Copy edit values : Subtract edits are optimized into Add edits (fewer batches)
        const float *edit = vedit->GetEdit();

        bool negate = (vedit->GetOperation() == HbrHierarchicalEdit<T>::Subtract);
        
        for(int i=0; i<batch.GetPrimvarWidth(); ++i)
            batch._edits[batchCount * batch.GetPrimvarWidth() + i] = negate ? -edit[i] : edit[i];

        batchCount++;
    }
    
    for(int i=0; i<numBatches; ++i) {
        int & batchLevel = currentLevels[i];
        int & batchCount = currentCounts[i];

        if (batchCount > 0) {
            insertHEditBatch(batches, i, batchLevel, batchCount-currentOffsets[i], currentOffsets[i]);
        }
    }

    return result;
}

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* FAR_VERTEX_EDIT_TABLES_FACTORY_H */
