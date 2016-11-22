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

#include "../far/stencilTableFactory.h"
#include "../far/stencilBuilder.h"
#include "../far/endCapGregoryBasisPatchFactory.h"
#include "../far/patchTable.h"
#include "../far/patchTableFactory.h"
#include "../far/patchMap.h"
#include "../far/topologyRefiner.h"
#include "../far/primvarRefiner.h"

#include <cassert>
#include <algorithm>
#include <iostream>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Far {

namespace {
#ifdef __INTEL_COMPILER
#pragma warning (push)
#pragma warning disable 1572
#endif
template<class FD>
inline bool isWeightZero(FD w) { return (w == (FD)0.0); }

#ifdef __INTEL_COMPILER
#pragma warning (pop)
#endif
}

//------------------------------------------------------------------------------

template<class FD>
void
StencilTableFactoryG<FD>::generateControlVertStencils(
    int numControlVerts, StencilG<FD> & dst) {

    // Control vertices contribute a single index with a weight of 1.0
    for (int i=0; i<numControlVerts; ++i) {
        *dst._size = 1;
        *dst._indices = i;
        *dst._weights = 1.0;
        dst.Next();
    }
}

//
// StencilTable factory
//
template<class FD>
StencilTableG<FD> const *
StencilTableFactoryG<FD>::Create(TopologyRefiner const & refiner,
    Options options) {

    int maxlevel = std::min(int(options.maxLevel), refiner.GetMaxLevel());
    if (maxlevel==0 && (! options.generateControlVerts)) {
        StencilTableG<FD> * result = new StencilTableG<FD>;
        result->_numControlVertices = refiner.GetLevel(0).GetNumVertices();
        return result;
    }

    bool interpolateVarying = options.interpolationMode==INTERPOLATE_VARYING;
    internal::StencilBuilderG<FD> builder(refiner.GetLevel(0).GetNumVertices(),
                                /*genControlVerts*/ true,
                                /*compactWeights*/  true);

    //
    // Interpolate stencils for each refinement level using
    // PrimvarRefiner::InterpolateLevel<>() for vertex or varying
    //
    PrimvarRefinerG<FD> primvarRefiner(refiner);

    typename internal::StencilBuilderG<FD>::Index srcIndex(&builder, 0);
    typename internal::StencilBuilderG<FD>::Index dstIndex(&builder, 
                                    refiner.GetLevel(0).GetNumVertices());

    for (int level=1; level<=maxlevel; ++level) {
        if (! interpolateVarying) {
            primvarRefiner.Interpolate(level, srcIndex, dstIndex);
        } else {
            primvarRefiner.InterpolateVarying(level, srcIndex, dstIndex);
        }

        if (options.factorizeIntermediateLevels) {
            srcIndex = dstIndex;
        }

        dstIndex = dstIndex[refiner.GetLevel(level).GetNumVertices()];

        if (! options.factorizeIntermediateLevels) {
            // All previous verts are considered as coarse verts, as a
            // result, we don't update the srcIndex and update the coarse
            // vertex count.
            builder.SetCoarseVertCount(dstIndex.GetOffset());
        }
    }

    size_t firstOffset = refiner.GetLevel(0).GetNumVertices();
    if (! options.generateIntermediateLevels)
        firstOffset = srcIndex.GetOffset();
 
    // Copy stencils from the StencilBuilder into the StencilTable.
    // Always initialize numControlVertices (useful for torus case)
    StencilTableG<FD> * result = 
                        new StencilTableG<FD>(refiner.GetLevel(0).GetNumVertices(),
                                          builder.GetStencilOffsets(),
                                          builder.GetStencilSizes(),
                                          builder.GetStencilSources(),
                                          builder.GetStencilWeights(),
                                          options.generateControlVerts,
                                          firstOffset);
    return result;
}

//------------------------------------------------------------------------------

template<class FD>
StencilTableG<FD> const *
StencilTableFactoryG<FD>::Create(int numTables, StencilTableG<FD> const ** tables) {

    // XXXtakahito:
    // This function returns NULL for empty inputs or erroneous condition.
    // It's convenient for skipping varying stencils etc, however,
    // other Create() API returns an empty stencil instead of NULL.
    // They need to be consistent.

    if ( (numTables<=0) || (! tables)) {
        return NULL;
    }

    int ncvs = -1,
        nstencils = 0,
        nelems = 0;

    for (int i=0; i<numTables; ++i) {

        StencilTableG<FD> const * st = tables[i];
        // allow the tables could have a null entry.
        if (!st) continue;

        if (ncvs >= 0 && st->GetNumControlVertices() != ncvs) {
            return NULL;
        }
        ncvs = st->GetNumControlVertices();
        nstencils += st->GetNumStencils();
        nelems += (int)st->GetControlIndices().size();
    }

    if (ncvs == -1) {
        return NULL;
    }

    StencilTableG<FD> * result = new StencilTableG<FD>;
    result->resize(nstencils, nelems);

    int * sizes = &result->_sizes[0];
    Index * indices = &result->_indices[0];
    FD * weights = &result->_weights[0];
    for (int i=0; i<numTables; ++i) {
        StencilTableG<FD> const * st = tables[i];
        if (!st) continue;

        int st_nstencils = st->GetNumStencils(),
            st_nelems = (int)st->_indices.size();
        memcpy(sizes, &st->_sizes[0], st_nstencils*sizeof(int));
        memcpy(indices, &st->_indices[0], st_nelems*sizeof(Index));
        memcpy(weights, &st->_weights[0], st_nelems*sizeof(FD));

        sizes += st_nstencils;
        indices += st_nelems;
        weights += st_nelems;
    }

    result->_numControlVertices = ncvs;

    // have to re-generate offsets from scratch
    result->generateOffsets();

    return result;
}

//------------------------------------------------------------------------------

template<class FD>
StencilTableG<FD> const *
StencilTableFactoryG<FD>::AppendLocalPointStencilTable(
    TopologyRefiner const &refiner,
    StencilTableG<FD> const * baseStencilTable,
    StencilTableG<FD> const * localPointStencilTable,
    bool factorize) {

    // factorize and append.
    if (baseStencilTable == NULL ||
        localPointStencilTable == NULL ||
        localPointStencilTable->GetNumStencils() == 0) return NULL;

    // baseStencilTable can be built with or without singular stencils
    // (single weight of 1.0f) as place-holders for coarse mesh vertices.

    int controlVertsIndexOffset = 0;
    int nBaseStencils = baseStencilTable->GetNumStencils();
    int nBaseStencilsElements = (int)baseStencilTable->_indices.size();
    {
        int nverts = refiner.GetNumVerticesTotal();
        if (nBaseStencils == nverts) {

            // the table contain stencils for the control vertices
            //
            //  <-----------------  nverts ------------------>
            //
            //  +---------------+----------------------------+-----------------+
            //  | control verts | refined verts   : (max lv) |   local points  |
            //  +---------------+----------------------------+-----------------+
            //  |          base stencil table                |    LP stencils  |
            //  +--------------------------------------------+-----------------+
            //                         ^                           /
            //                          \_________________________/
            //
            //
            controlVertsIndexOffset = 0;

        } else if (nBaseStencils == (nverts -refiner.GetLevel(0).GetNumVertices())) {

            // the table does not contain stencils for the control vertices
            //
            //  <-----------------  nverts ------------------>
            //                  <------ nBaseStencils ------->
            //  +---------------+----------------------------+-----------------+
            //  | control verts | refined verts   : (max lv) |   local points  |
            //  +---------------+----------------------------+-----------------+
            //                  |     base stencil table     |    LP stencils  |
            //                  +----------------------------+-----------------+
            //                                 ^                   /
            //                                  \_________________/
            //  <-------------->
            //                 controlVertsIndexOffset
            //
            controlVertsIndexOffset = refiner.GetLevel(0).GetNumVertices();

        } else {
            // these are not the stencils you are looking for.
            assert(0);
            return NULL;
        }
    }

    // copy all local points stencils to proto stencils, and factorize if needed.
    int nLocalPointStencils = localPointStencilTable->GetNumStencils();
    int nLocalPointStencilsElements = 0;

    internal::StencilBuilderG<FD> builder(refiner.GetLevel(0).GetNumVertices(),
                                /*genControlVerts*/ false,
                                /*compactWeights*/  factorize);
    typename internal::StencilBuilderG<FD>::Index origin(&builder, 0);
    typename internal::StencilBuilderG<FD>::Index dst = origin;
    typename internal::StencilBuilderG<FD>::Index srcIdx = origin;

    for (int i = 0 ; i < nLocalPointStencils; ++i) {
        StencilG<FD> src = localPointStencilTable->GetStencil(i);
        dst = origin[i];
        for (int j = 0; j < src.GetSize(); ++j) {
            Index index = src.GetVertexIndices()[j];
            FD weight = src.GetWeights()[j];
            if (isWeightZero(weight)) continue;

            if (factorize) {
                dst.AddWithWeight(
                    // subtracting controlVertsIndex if the baseStencil doesn't
                    // include control vertices (see above diagram)
                    // since currently local point stencils are created with
                    // absolute indices including control (level=0) vertices.
                    baseStencilTable->GetStencil(index - controlVertsIndexOffset),
                    weight);
            } else {
                srcIdx = origin[index + controlVertsIndexOffset];
                dst.AddWithWeight(srcIdx, weight);
            }
        }
        nLocalPointStencilsElements += builder.GetNumVertsInStencil(i);
    }

    // create new stencil table
    StencilTableG<FD> * result = new StencilTableG<FD>;
    result->_numControlVertices = refiner.GetLevel(0).GetNumVertices();
    result->resize(nBaseStencils + nLocalPointStencils,
                   nBaseStencilsElements + nLocalPointStencilsElements);

    int* sizes = &result->_sizes[0];
    Index * indices = &result->_indices[0];
    FD * weights = &result->_weights[0];

    // put base stencils first
    memcpy(sizes, &baseStencilTable->_sizes[0],
           nBaseStencils*sizeof(int));
    memcpy(indices, &baseStencilTable->_indices[0],
           nBaseStencilsElements*sizeof(Index));
    memcpy(weights, &baseStencilTable->_weights[0],
           nBaseStencilsElements*sizeof(FD));

    sizes += nBaseStencils;
    indices += nBaseStencilsElements;
    weights += nBaseStencilsElements;

    // endcap stencils second
    for (int i = 0 ; i < nLocalPointStencils; ++i) {
        int size = builder.GetNumVertsInStencil(i);
        int idx = builder.GetStencilOffsets()[i];
        for (int j = 0; j < size; ++j) {
            *indices++ = builder.GetStencilSources()[idx+j];
            *weights++ = builder.GetStencilWeights()[idx+j];
        }
        *sizes++ = size;
    }

    // have to re-generate offsets from scratch
    result->generateOffsets();

    return result;
}

//------------------------------------------------------------------------------
template<class FD>
LimitStencilTableG<FD> const *
LimitStencilTableFactoryG<FD>::Create(TopologyRefiner const & refiner,
    LocationArrayVec const & locationArrays,
   StencilTableG<FD> const * cvStencilsIn,
     PatchTableG<FD> const * patchTableIn,
                     Options options) {

    // Compute the total number of stencils to generate
    int numStencils=0, numLimitStencils=0;
    for (int i=0; i<(int)locationArrays.size(); ++i) {
        assert(locationArrays[i].numLocations>=0);
        numStencils += locationArrays[i].numLocations;
    }
    if (numStencils<=0) {
        return 0;
    }

    bool uniform = refiner.IsUniform();

    int maxlevel = refiner.GetMaxLevel();

    StencilTableG<FD> const * cvstencils = cvStencilsIn;
    if (! cvstencils) {
        // Generate stencils for the control vertices - this is necessary to
        // properly factorize patches with control vertices at level 0 (natural
        // regular patches, such as in a torus)
        // note: the control vertices of the mesh are added as single-index
        //       stencils of weight 1.0f
        typename StencilTableFactoryG<FD>::Options options;
        options.generateIntermediateLevels = uniform ? false :true;
        options.generateControlVerts = true;
        options.generateOffsets = true;

        // PERFORMANCE: We could potentially save some mem-copies by not
        // instantiating the stencil tables and work directly off the source
        // data.
        cvstencils = StencilTableFactoryG<FD>::Create(refiner, options);
    } else {
        // Sanity checks
        //
        // Note that the input cvStencils could be larger than the number of
        // refiner's vertices, due to the existence of the end cap stencils.
        if (cvstencils->GetNumStencils() < (uniform ?
            refiner.GetLevel(maxlevel).GetNumVertices() :
                refiner.GetNumVerticesTotal())) {
                return 0;
        }
    }

    // If a stencil table was given, use it, otherwise, create a new one
    PatchTableG<FD> const * patchtable = patchTableIn;

    if (! patchtable) {
        // XXXX (manuelk) If no patch-table was passed, we should be able to
        // infer the patches fairly easily from the refiner. Once more tags
        // have been added to the refiner, maybe we can remove the need for the
        // patch table.

        typename PatchTableFactoryG<FD>::Options options;
        options.SetEndCapType(
            Far::PatchTableFactoryG<FD>::Options::ENDCAP_GREGORY_BASIS);
        options.useInfSharpPatch = !uniform &&
            refiner.GetAdaptiveOptions().useInfSharpPatch;

        patchtable = PatchTableFactoryG<FD>::Create(refiner, options);

        if (! cvStencilsIn) {
            // if cvstencils is just created above, append endcap stencils
            if (StencilTableG<FD> const *localPointStencilTable =
                patchtable->GetLocalPointStencilTable()) {
                StencilTableG<FD> const *table =
                    StencilTableFactoryG<FD>::AppendLocalPointStencilTable(
                        refiner, cvstencils, localPointStencilTable);
                delete cvstencils;
                cvstencils = table;
            }
        }
    } else {
        // Sanity checks
        if (patchtable->IsFeatureAdaptive()==uniform) {
            if (! cvStencilsIn) {
                assert(cvstencils && cvstencils!=cvStencilsIn);
                delete cvstencils;
            }
            return 0;
        }
    }

    assert(patchtable && cvstencils);

    // Create a patch-map to locate sub-patches faster
    PatchMapG<FD> patchmap( *patchtable );

    //
    // Generate limit stencils for locations
    //

    internal::StencilBuilderG<FD> builder(refiner.GetLevel(0).GetNumVertices(),
                                /*genControlVerts*/ false,
                                /*compactWeights*/  true);
    typename internal::StencilBuilderG<FD>::Index origin(&builder, 0);
    typename internal::StencilBuilderG<FD>::Index dst = origin;

    FD wP[20], wDs[20], wDt[20], wDss[20], wDst[20], wDtt[20];

    for (size_t i=0; i<locationArrays.size(); ++i) {
        LocationArray const & array = locationArrays[i];
        assert(array.ptexIdx>=0);

        for (int j=0; j<array.numLocations; ++j) { // for each face we're working on
            FD s = array.s[j],
                  t = array.t[j]; // for each target (s,t) point on that face

            typename PatchMapG<FD>::Handle const * handle = 
                                        patchmap.FindPatch(array.ptexIdx, s, t);
            if (handle) {
                ConstIndexArray cvs = patchtable->GetPatchVertices(*handle);

                StencilTableG<FD> const & src = *cvstencils;
                dst = origin[numLimitStencils];

                if (options.generate2ndDerivatives) {
                    patchtable->EvaluateBasis(*handle, s, t, wP, wDs, wDt, wDss, wDst, wDtt);

                    dst.Clear();
                    for (int k = 0; k < cvs.size(); ++k) {
                        dst.AddWithWeight(src[cvs[k]], wP[k], wDs[k], wDt[k], wDss[k], wDst[k], wDtt[k]);
                    }
                } else if (options.generate1stDerivatives) {
                    patchtable->EvaluateBasis(*handle, s, t, wP, wDs, wDt);

                    dst.Clear();
                    for (int k = 0; k < cvs.size(); ++k) {
                        dst.AddWithWeight(src[cvs[k]], wP[k], wDs[k], wDt[k]);
                    }
                } else {
                    patchtable->EvaluateBasis(*handle, s, t, wP);

                    dst.Clear();
                    for (int k = 0; k < cvs.size(); ++k) {
                        dst.AddWithWeight(src[cvs[k]], wP[k]);
                    }
                }

                ++numLimitStencils;
            }
        }
    }

    if (! cvStencilsIn) {
        delete cvstencils;
    }

    if (! patchTableIn) {
        delete patchtable;
    }

    //
    // Copy the proto-stencils into the limit stencil table
    //
    LimitStencilTableG<FD> * result = new LimitStencilTableG<FD>(
                                          refiner.GetLevel(0).GetNumVertices(),
                                          builder.GetStencilOffsets(),
                                          builder.GetStencilSizes(),
                                          builder.GetStencilSources(),
                                          builder.GetStencilWeights(),
                                          builder.GetStencilDuWeights(),
                                          builder.GetStencilDvWeights(),
                                          builder.GetStencilDuuWeights(),
                                          builder.GetStencilDuvWeights(),
                                          builder.GetStencilDvvWeights(),
                                          /*ctrlVerts*/false,
                                          /*fristOffset*/0);
    return result;
}

template class StencilTableFactoryG<float>;
template class StencilTableFactoryG<double>;
template class LimitStencilTableFactoryG<float>;
template class LimitStencilTableFactoryG<double>;

} // end namespace Far

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv
