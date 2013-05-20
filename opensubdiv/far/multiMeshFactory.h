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

#ifndef FAR_MULTI_MESH_FACTORY_H
#define FAR_MULTI_MESH_FACTORY_H

#include "../version.h"

#include "../far/mesh.h"
#include "../far/bilinearSubdivisionTablesFactory.h"
#include "../far/catmarkSubdivisionTablesFactory.h"
#include "../far/loopSubdivisionTablesFactory.h"
#include "../far/patchTablesFactory.h"
#include "../far/vertexEditTablesFactory.h"

#include <typeinfo>

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

    /// Constructor.
    FarMultiMeshFactory() {}
    
    /// Splices a vector of Far meshes into a single Far mesh
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

    // splice subdivision tables
    FarSubdivisionTables<U> * spliceSubdivisionTables(FarMesh<U> *farmesh, FarMeshVector const &meshes);

    // splice patch tables
    FarPatchTables * splicePatchTables(FarMeshVector const &meshes);

    // splice patch array
    FarPatchTables::PTable::iterator splicePatch(FarPatchTables::Descriptor desc,
                                                 FarMeshVector const &meshes,
                                                 FarPatchTables::PatchArrayVector &result,
                                                 FarPatchTables::PTable::iterator dstIndexIt,
                                                 int *voffset, int *poffset, int *qoffset,
                                                 std::vector<int> const &vertexOffsets);

    // splice hierarchical edit tables
    FarVertexEditTables<U> * spliceVertexEditTables(FarMesh<U> *farmesh, FarMeshVector const &meshes);

    int _maxlevel;
    int _maxvalence;

    // patch arrays for each mesh
    std::vector<FarPatchTables::PatchArrayVector> _multiPatchArrays;
};


template <class T, class U> FarMesh<U> *
FarMultiMeshFactory<T, U>::Create(std::vector<FarMesh<U> const *> const &meshes) {

    if (meshes.empty()) return NULL;

    bool adaptive = (meshes[0]->GetPatchTables() != NULL);
    const std::type_info &scheme = typeid(*(meshes[0]->GetSubdivisionTables()));
    _maxlevel = 0;
    _maxvalence = 0;

    for (size_t i = 0; i < meshes.size(); ++i) {
        FarMesh<U> const *mesh = meshes[i];
        // XXX: once uniform quads are integrated into patch tables,
        // this restriction can be relaxed so that we can merge adaptive and uniform meshes together.
        if (adaptive ^ (mesh->GetPatchTables() != NULL)) {
            assert(false);
            return NULL;
        }

        // meshes have to have a same subdivision scheme
        if (scheme != typeid(*(mesh->GetSubdivisionTables()))) {
            assert(false);
            return NULL;
        }

        _maxlevel = std::max(_maxlevel, mesh->GetSubdivisionTables()->GetMaxLevel()-1);
        if (mesh->GetPatchTables()) {
            _maxvalence = std::max(_maxvalence, mesh->GetPatchTables()->GetMaxValence());
        }
    }

    FarMesh<U> * result = new FarMesh<U>();

    // splice subdivision tables
    result->_subdivisionTables = spliceSubdivisionTables(result, meshes);

    // splice patch/quad index tables
    result->_patchTables = splicePatchTables(meshes);

    // splice vertex edit tables
    result->_vertexEditTables = spliceVertexEditTables(result, meshes);

    // count total num vertices, numptex faces
    int numVertices = 0, numPtexFaces = 0;
    for (size_t i = 0; i < meshes.size(); ++i) {
        numVertices += meshes[i]->GetNumVertices();
        numPtexFaces += meshes[i]->GetNumPtexFaces();
    }
    result->_vertices.resize(numVertices);
    result->_numPtexFaces = numPtexFaces;
    result->_totalFVarWidth = 0;  // XXX: fvar for multimesh hasn't been implemented yet.

    return result;
}

template <typename V, typename IT> static IT
copyWithOffset(IT dst_iterator, V const &src, int offset) {
    return std::transform(src.begin(), src.end(), dst_iterator,
                          std::bind2nd(std::plus<typename V::value_type>(), offset));
}

template <typename V, typename IT> static IT
copyWithOffset(IT dst_iterator, V const &src, int start, int count, int offset) {
    return std::transform(src.begin()+start, src.begin()+start+count, dst_iterator,
                          std::bind2nd(std::plus<typename V::value_type>(), offset));
}

template <typename V, typename IT> static IT
copyWithPtexFaceOffset(IT dst_iterator, V const &src, int start, int count, int offset) {
    for (typename V::const_iterator it = src.begin()+start; it != src.begin()+start+count; ++it) {
        typename V::value_type ptexCoord = *it;
        ptexCoord.faceIndex += offset;
        *dst_iterator++ = ptexCoord;
    }
    return dst_iterator;
}

template <typename V, typename IT> static IT
copyWithOffsetF_ITa(IT dst_iterator, V const &src, int offset) {
    for (typename V::const_iterator it = src.begin(); it != src.end();) {
        *dst_iterator++ = *it++ + offset;   // offset to F_IT
        *dst_iterator++ = *it++;            // valence
    }
    return dst_iterator;
}

template <typename V, typename IT> static IT
copyWithOffsetE_IT(IT dst_iterator, V const &src, int offset) {
    for (typename V::const_iterator it = src.begin(); it != src.end(); ++it) {
        *dst_iterator++ = (*it == -1) ? -1 : (*it + offset);
    }
    return dst_iterator;
}

template <typename V, typename IT> static IT
copyWithOffsetV_ITa(IT dst_iterator, V const &src, int tableOffset, int vertexOffset) {
    for (typename V::const_iterator it = src.begin(); it != src.end();) {
        *dst_iterator++ = *it++ + tableOffset;   // offset to V_IT
        *dst_iterator++ = *it++;                 // valence
        *dst_iterator++ = (*it == -1) ? -1 : (*it + vertexOffset); ++it;
        *dst_iterator++ = (*it == -1) ? -1 : (*it + vertexOffset); ++it;
        *dst_iterator++ = (*it == -1) ? -1 : (*it + vertexOffset); ++it;
    }
    return dst_iterator;
}

template <typename V, typename IT> static IT
copyWithOffsetVertexValence(IT dst_iterator, V const &src, int srcMaxValence, int dstMaxValence, int offset) {
    for (typename V::const_iterator it = src.begin(); it != src.end(); ) {
        int valence = *it++;
        *dst_iterator++ = valence;
        valence = abs(valence);
        for (int i = 0; i < 2*dstMaxValence; ++i) {
            if (i < 2*srcMaxValence) {
                *dst_iterator++ = (i < 2*valence) ? *it + offset : 0;
                ++it;
            } else {
                *dst_iterator++ = 0;
            }
        }
    }
    return dst_iterator;
}

template <class T, class U> FarSubdivisionTables<U> *
FarMultiMeshFactory<T, U>::spliceSubdivisionTables(FarMesh<U> *farMesh, FarMeshVector const &meshes) {

    // count total table size
    size_t total_F_ITa = 0, total_F_IT = 0;
    size_t total_E_IT = 0, total_E_W = 0;
    size_t total_V_ITa = 0, total_V_IT = 0, total_V_W = 0;
    for (size_t i = 0; i < meshes.size(); ++i) {
        FarSubdivisionTables<U> const * tables = meshes[i]->GetSubdivisionTables();
        assert(tables);

        total_F_ITa += tables->Get_F_ITa().size();
        total_F_IT  += tables->Get_F_IT().size();
        total_E_IT  += tables->Get_E_IT().size();
        total_E_W   += tables->Get_E_W().size();
        total_V_ITa += tables->Get_V_ITa().size();
        total_V_IT  += tables->Get_V_IT().size();
        total_V_W   += tables->Get_V_W().size();
    }

    FarSubdivisionTables<U> *result = NULL;
    const std::type_info &scheme = typeid(*(meshes[0]->GetSubdivisionTables()));

    if (scheme == typeid(FarCatmarkSubdivisionTables<U>) ) {
        result = new FarCatmarkSubdivisionTables<U>(farMesh, _maxlevel);
    } else if (scheme == typeid(FarBilinearSubdivisionTables<U>) ) {
        result = new FarBilinearSubdivisionTables<U>(farMesh, _maxlevel);
    } else if (scheme == typeid(FarLoopSubdivisionTables<U>) ) {
        result = new FarLoopSubdivisionTables<U>(farMesh, _maxlevel);
    }

    result->_F_ITa.resize(total_F_ITa);
    result->_F_IT.resize(total_F_IT);
    result->_E_IT.resize(total_E_IT);
    result->_E_W.resize(total_E_W);
    result->_V_ITa.resize(total_V_ITa);
    result->_V_IT.resize(total_V_IT);
    result->_V_W.resize(total_V_W);

    // compute table offsets;
    std::vector<int> vertexOffsets;
    std::vector<int> fvOffsets;
    std::vector<int> evOffsets;
    std::vector<int> vvOffsets;
    std::vector<int> F_IToffsets;
    std::vector<int> V_IToffsets;

    {
        int vertexOffset = 0;
        int F_IToffset = 0;
        int V_IToffset = 0;
        int fvOffset = 0;
        int evOffset = 0;
        int vvOffset = 0;
        for (size_t i = 0; i < meshes.size(); ++i) {
            FarSubdivisionTables<U> const * tables = meshes[i]->GetSubdivisionTables();
            assert(tables);
            
            vertexOffsets.push_back(vertexOffset);
            F_IToffsets.push_back(F_IToffset);
            V_IToffsets.push_back(V_IToffset);
            fvOffsets.push_back(fvOffset);
            evOffsets.push_back(evOffset);
            vvOffsets.push_back(vvOffset);

            vertexOffset += meshes[i]->GetNumVertices();
            F_IToffset += (int)tables->Get_F_IT().size();
            fvOffset += (int)tables->Get_F_ITa().size()/2;
            V_IToffset += (int)tables->Get_V_IT().size();

            if (scheme == typeid(FarCatmarkSubdivisionTables<U>) ||
                scheme == typeid(FarLoopSubdivisionTables<U>)) {
                evOffset += (int)tables->Get_E_IT().size()/4;
                vvOffset += (int)tables->Get_V_ITa().size()/5;
            } else {
                evOffset += (int)tables->Get_E_IT().size()/2;
                vvOffset += (int)tables->Get_V_ITa().size();
            }
        }
    }

    // concat F_IT and V_IT
    std::vector<unsigned int>::iterator F_IT = result->_F_IT.begin();
    std::vector<unsigned int>::iterator V_IT = result->_V_IT.begin();

    for (size_t i = 0; i < meshes.size(); ++i) {
        FarSubdivisionTables<U> const * tables = meshes[i]->GetSubdivisionTables();

        int vertexOffset = vertexOffsets[i];
        // remap F_IT, V_IT tables
        F_IT = copyWithOffset(F_IT, tables->Get_F_IT(), vertexOffset);
        V_IT = copyWithOffset(V_IT, tables->Get_V_IT(), vertexOffset);
    }

    // merge other tables
    std::vector<int>::iterator F_ITa = result->_F_ITa.begin();
    std::vector<int>::iterator E_IT  = result->_E_IT.begin();
    std::vector<float>::iterator E_W = result->_E_W.begin();
    std::vector<float>::iterator V_W = result->_V_W.begin();
    std::vector<int>::iterator V_ITa = result->_V_ITa.begin();

    for (size_t i = 0; i < meshes.size(); ++i) {
        FarSubdivisionTables<U> const * tables = meshes[i]->GetSubdivisionTables();

        // copy face tables
        F_ITa = copyWithOffsetF_ITa(F_ITa, tables->Get_F_ITa(), F_IToffsets[i]);

        // copy edge tables
        E_IT = copyWithOffsetE_IT(E_IT, tables->Get_E_IT(), vertexOffsets[i]);
        E_W = copyWithOffset(E_W, tables->Get_E_W(), 0);

        // copy vert tables
        if (scheme == typeid(FarCatmarkSubdivisionTables<U>) ||
            scheme == typeid(FarLoopSubdivisionTables<U>)) {
            V_ITa = copyWithOffsetV_ITa(V_ITa, tables->Get_V_ITa(), V_IToffsets[i], vertexOffsets[i]);
        } else {
            V_ITa = copyWithOffset(V_ITa, tables->Get_V_ITa(), vertexOffsets[i]);
        }
        V_W = copyWithOffset(V_W, tables->Get_V_W(), 0);
    }

    // merge batch, model by model
    FarKernelBatchVector &batches = farMesh->_batches;
    int editTableIndexOffset = 0;
    for (size_t i = 0; i < meshes.size(); ++i) {
        for (int j = 0; j < (int)meshes[i]->_batches.size(); ++j) {
            FarKernelBatch batch = meshes[i]->_batches[j];
            batch._vertexOffset += vertexOffsets[i];
            
            if (batch._kernelType == FarKernelBatch::CATMARK_FACE_VERTEX or
                batch._kernelType == FarKernelBatch::BILINEAR_FACE_VERTEX) {
                
                batch._tableOffset += fvOffsets[i];
                
            } else if (batch._kernelType == FarKernelBatch::CATMARK_EDGE_VERTEX or
                       batch._kernelType == FarKernelBatch::LOOP_EDGE_VERTEX or
                       batch._kernelType == FarKernelBatch::BILINEAR_EDGE_VERTEX) {
                       
                batch._tableOffset += evOffsets[i];
                
            } else if (batch._kernelType == FarKernelBatch::CATMARK_VERT_VERTEX_A1 or
                       batch._kernelType == FarKernelBatch::CATMARK_VERT_VERTEX_A2 or
                       batch._kernelType == FarKernelBatch::CATMARK_VERT_VERTEX_B or
                       batch._kernelType == FarKernelBatch::LOOP_VERT_VERTEX_A1 or
                       batch._kernelType == FarKernelBatch::LOOP_VERT_VERTEX_A2 or
                       batch._kernelType == FarKernelBatch::LOOP_VERT_VERTEX_B or
                       batch._kernelType == FarKernelBatch::BILINEAR_VERT_VERTEX) {
                       
                batch._tableOffset += vvOffsets[i];
                
            } else if (batch._kernelType == FarKernelBatch::HIERARCHICAL_EDIT) {
            
                batch._tableIndex += editTableIndexOffset;
            }
            batches.push_back(batch);
        }
        editTableIndexOffset += meshes[i]->_vertexEditTables ? meshes[i]->_vertexEditTables->GetNumBatches() : 0;
    }

    // count verts offsets
    result->_vertsOffsets.resize(_maxlevel+2);
    for (size_t i = 0; i < meshes.size(); ++i) {
        FarSubdivisionTables<U> const * tables = meshes[i]->GetSubdivisionTables();
        for (size_t j = 0; j < tables->_vertsOffsets.size(); ++j) {
            result->_vertsOffsets[j] += tables->_vertsOffsets[j];
        }
    }

    return result;
}        

template <class T, class U> FarPatchTables::PTable::iterator
FarMultiMeshFactory<T, U>::splicePatch(FarPatchTables::Descriptor desc,
                                       FarMeshVector const &meshes,
                                       FarPatchTables::PatchArrayVector &result,
                                       FarPatchTables::PTable::iterator dstIndexIt,
                                       int *voffset, int *poffset, int *qoffset,
                                       std::vector<int> const &vertexOffsets)
{
    for (size_t i = 0; i < meshes.size(); ++i) {
        FarPatchTables const *patchTables = meshes[i]->GetPatchTables();
        FarPatchTables::PatchArray const *srcPatchArray = patchTables->GetPatchArray(desc);
        if (not srcPatchArray) continue;

        // create new patcharray with offset
        int vindex = srcPatchArray->GetVertIndex();
        int npatch = srcPatchArray->GetNumPatches();
        int nvertex = npatch * desc.GetNumControlVertices();

        FarPatchTables::PatchArray patchArray(desc,
                                              *voffset,
                                              *poffset,
                                              npatch,
                                              *qoffset);
        // append patch array
        result.push_back(patchArray);

        // also store into multiPatchArrays, will be used for partial drawing
        // XXX: can be stored as indices. revisit here later
        _multiPatchArrays[i].push_back(patchArray);

        // increment offset
        *voffset += nvertex;
        *poffset += npatch;
        *qoffset += (desc.GetType() == FarPatchTables::GREGORY ||
                     desc.GetType() == FarPatchTables::GREGORY_BOUNDARY) ? npatch * 4 : 0;

        // copy index arrays [vindex, vindex+nvertex]
        dstIndexIt = copyWithOffset(dstIndexIt,
                                    patchTables->GetPatchTable(),
                                    vindex,
                                    nvertex,
                                    vertexOffsets[i]);
    }
    return dstIndexIt;
}

template <class T, class U> FarPatchTables *
FarMultiMeshFactory<T, U>::splicePatchTables(FarMeshVector const &meshes) {

    FarPatchTables *result = new FarPatchTables(_maxvalence);

    int total_quadOffset0 = 0;
    int total_quadOffset1 = 0;

    std::vector<int> vertexOffsets;
    std::vector<int> gregoryQuadOffsets;
    std::vector<int> numGregoryPatches;
    int vertexOffset = 0;
    int maxValence = 0;
    int numTotalIndices = 0;

    //result->_patchCounts.reserve(meshes.size());
    //FarPatchCount totalCount;
    typedef FarPatchTables::Descriptor Descriptor;

    // count how many patches exist on each mesh
    for (size_t i = 0; i < meshes.size(); ++i) {
        const FarPatchTables *ptables = meshes[i]->GetPatchTables();
        assert(ptables);
        
        vertexOffsets.push_back(vertexOffset);
        vertexOffset += meshes[i]->GetNumVertices();

        // need to align maxvalence with the highest value
        maxValence = std::max(maxValence, ptables->_maxValence);

        FarPatchTables::PatchArray const *gregory = ptables->GetPatchArray(Descriptor(FarPatchTables::GREGORY, FarPatchTables::NON_TRANSITION, /*rot*/ 0));
        FarPatchTables::PatchArray const *gregoryBoundary = ptables->GetPatchArray(Descriptor(FarPatchTables::GREGORY_BOUNDARY, FarPatchTables::NON_TRANSITION, /*rot*/ 0));

        int nGregory = gregory ? gregory->GetNumPatches() : 0;
        int nGregoryBoundary = gregoryBoundary ? gregoryBoundary->GetNumPatches() : 0;
        total_quadOffset0 += nGregory * 4;
        total_quadOffset1 += nGregoryBoundary * 4;
        numGregoryPatches.push_back(nGregory);
        gregoryQuadOffsets.push_back(total_quadOffset0);

        numTotalIndices += ptables->GetNumControlVertices();
    }

    // Allocate full patches
    result->_patches.resize(numTotalIndices);

    // Allocate vertex valence table, quad offset table
    if (total_quadOffset0 + total_quadOffset1 > 0) {
        result->_vertexValenceTable.resize((2*maxValence+1) * vertexOffset);
        result->_quadOffsetTable.resize(total_quadOffset0 + total_quadOffset1);
    }

    // splice tables
    // assuming input farmeshes have dense patchtables

    _multiPatchArrays.resize(meshes.size());

    int voffset = 0, poffset = 0, qoffset = 0;
    FarPatchTables::PTable::iterator dstIndexIt = result->_patches.begin();

    // splice patches : iterate from POINTS
    for (FarPatchTables::Descriptor::iterator it(FarPatchTables::Descriptor(FarPatchTables::POINTS, FarPatchTables::NON_TRANSITION, 0));
         it != FarPatchTables::Descriptor::end(); ++it) {
        dstIndexIt = splicePatch(*it, meshes, result->_patchArrays, dstIndexIt, &voffset, &poffset, &qoffset, vertexOffsets);
    }

    // merge vertexvalence and quadoffset tables
    std::vector<unsigned int>::iterator Q0_IT = result->_quadOffsetTable.begin();
    std::vector<unsigned int>::iterator Q1_IT = Q0_IT + total_quadOffset0;

    std::vector<int>::iterator VV_IT = result->_vertexValenceTable.begin();
    for (size_t i = 0; i < meshes.size(); ++i) {
        const FarPatchTables *ptables = meshes[i]->GetPatchTables();

        // merge vertex valence
        // note: some prims may not have vertex valence table, but still need a space
        // in order to fill following prim's data at appropriate location.
        copyWithOffsetVertexValence(VV_IT,
                                    ptables->_vertexValenceTable,
                                    ptables->_maxValence,
                                    maxValence,
                                    vertexOffsets[i]);

        VV_IT += meshes[i]->GetNumVertices() * (2 * maxValence + 1);

        // merge quad offsets
//        int nGregoryQuads = (int)ptables->_full._G_IT.first.size();
        int nGregoryQuads = numGregoryPatches[i] * 4;
        if (nGregoryQuads > 0) {
            Q0_IT = std::copy(ptables->_quadOffsetTable.begin(),
                              ptables->_quadOffsetTable.begin()+nGregoryQuads,
                              Q0_IT);
        }
        if (nGregoryQuads < (int)ptables->_quadOffsetTable.size()) {
            Q1_IT = std::copy(ptables->_quadOffsetTable.begin()+nGregoryQuads,
                              ptables->_quadOffsetTable.end(),
                              Q1_IT);
        }
    }

    // merge ptexCoord table
    for (FarPatchTables::Descriptor::iterator it(FarPatchTables::Descriptor(FarPatchTables::POINTS, FarPatchTables::NON_TRANSITION, 0));
         it != FarPatchTables::Descriptor::end(); ++it) {
        int ptexFaceOffset = 0;
        for (size_t i = 0; i < meshes.size(); ++i) {
            FarPatchTables const *ptables = meshes[i]->GetPatchTables();
            FarPatchTables::PatchArray const *parray = ptables->GetPatchArray(*it);
            if (not parray) continue;

            copyWithPtexFaceOffset(std::back_inserter(result->_paramTable),
                                                      ptables->_paramTable,
                                                      parray->GetPatchIndex(),
                                                      parray->GetNumPatches(), ptexFaceOffset);

            ptexFaceOffset += meshes[i]->GetNumPtexFaces();
        }
    }

    return result;
}

template <class T, class U> FarVertexEditTables<U> *
FarMultiMeshFactory<T, U>::spliceVertexEditTables(FarMesh<U> *farMesh, FarMeshVector const &meshes) {

    FarVertexEditTables<U> * result = new FarVertexEditTables<U>(farMesh);

    // at this moment, don't merge vertex edit tables (separate batch)
    for (size_t i = 0; i < meshes.size(); ++i) {
        const FarVertexEditTables<U> *vertexEditTables = meshes[i]->GetVertexEdit();
        if (not vertexEditTables) continue;

        // copy each edit batch  XXX:inefficient copy
        result->_batches.insert(result->_batches.end(),
                                vertexEditTables->_batches.begin(),
                                vertexEditTables->_batches.end());
    }

    if (result->_batches.empty()) {
        delete result;
        return NULL;
    }
    return result;
}

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* FAR_MULTI_MESH_FACTORY_H */
