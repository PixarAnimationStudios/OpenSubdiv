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

#include "../far/stencilTablesFactory.h"
#include "../far/patchTablesFactory.h"
#include "../far/patchMap.h"
#include "../far/protoStencil.h"
#include "../far/topologyRefiner.h"

#include <cassert>
#include <algorithm>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Far {

//------------------------------------------------------------------------------

void
StencilTablesFactory::generateControlVertStencils(
    int numControlVerts, Stencil & dst) {

    // Control vertices contribute a single index with a weight of 1.0
    for (int i=0; i<numControlVerts; ++i) {
        *dst._size = 1;
        *dst._indices = i;
        *dst._weights = 1.0f;
        dst.Next();
    }
}

//
// StencilTables factory
//
StencilTables const *
StencilTablesFactory::Create(TopologyRefiner const & refiner,
    Options options) {

    StencilTables * result = new StencilTables;

    int maxlevel = std::min(int(options.maxLevel), refiner.GetMaxLevel());
    if (maxlevel==0 and (not options.generateControlVerts)) {
        return result;
    }

    // 'maxsize' reflects the size of the default supporting basis factorized
    // in the stencils, with a little bit of head-room. Each subdivision scheme
    // has a set valence for 'regular' vertices, which drives the size of the
    // supporting basis of control-vertices. The goal is to reduce the number
    // of incidences where the pool allocator has to switch to dynamically
    // allocated heap memory when encountering extraordinary vertices that
    // require a larger supporting basis.
    //
    // The maxsize settings we use follow the assumption that the vast
    // majority of the vertices in a mesh are regular, and that the valence
    // of the extraordinary vertices is only higher by 1 edge.
    int maxsize = 0;
    bool interpolateVarying = false;
    switch (options.interpolationMode) {
        case INTERPOLATE_VERTEX: {
                Sdc::SchemeType type = refiner.GetSchemeType();
                switch (type) {
                    case Sdc::SCHEME_BILINEAR : maxsize = 5; break;
                    case Sdc::SCHEME_CATMARK  : maxsize = 17; break;
                    case Sdc::SCHEME_LOOP     : maxsize = 10; break;
                    default:
                        assert(0);
                }
            } break;
        case INTERPOLATE_VARYING: maxsize = 5; interpolateVarying=true; break;
        default:
            assert(0);
    }

    std::vector<StencilAllocator> allocators(
        options.generateIntermediateLevels ? maxlevel+1 : 2,
            StencilAllocator(maxsize, interpolateVarying));

    StencilAllocator * srcAlloc = &allocators[0],
                     * dstAlloc = &allocators[1];

    //
    // Interpolate stencils for each refinement level using
    // TopologyRefiner::InterpolateLevel<>()
    //
    for (int level=1;level<=maxlevel; ++level) {

        dstAlloc->Resize(refiner.GetNumVertices(level));

        if (options.interpolationMode==INTERPOLATE_VERTEX) {
            refiner.Interpolate(level, *srcAlloc, *dstAlloc);
        } else {
            refiner.InterpolateVarying(level, *srcAlloc, *dstAlloc);
        }

        if (options.generateIntermediateLevels) {
            if (level<maxlevel) {
                if (options.factorizeIntermediateLevels) {
                    srcAlloc = &allocators[level];
                } else {
                    // if the stencils are dependent on the previous level of
                    // subdivision, pass an empty allocator to treat all parent
                    // vertices as control vertices
                    assert(allocators[0].GetNumStencils()==0);
                }
                dstAlloc = &allocators[level+1];
            }
        } else {
            std::swap(srcAlloc, dstAlloc);
        }
    }

    // Copy stencils from the pool allocator into the tables
    {
        // Add total number of stencils, weights & indices
        int nelems = 0, nstencils=0;
        if (options.generateIntermediateLevels) {
            for (int level=0; level<=maxlevel; ++level) {
                nstencils += allocators[level].GetNumStencils();
                nelems += allocators[level].GetNumVerticesTotal();
            }
        } else {
            nstencils = (int)srcAlloc->GetNumStencils();
            nelems = srcAlloc->GetNumVerticesTotal();
        }

        // Allocate
        result->_numControlVertices = refiner.GetNumVertices(0);

        if (options.generateControlVerts) {
            nstencils += result->_numControlVertices;
            nelems += result->_numControlVertices;
        }
        result->resize(nstencils, nelems);

        // Copy stencils
        Stencil dst(&result->_sizes.at(0),
            &result->_indices.at(0), &result->_weights.at(0));

        if (options.generateControlVerts) {
            generateControlVertStencils(result->_numControlVertices, dst);
        }

        if (options.generateIntermediateLevels) {
            for (int level=1; level<=maxlevel; ++level) {
                for (int i=0; i<allocators[level].GetNumStencils(); ++i) {
                    *dst._size = allocators[level].CopyStencil(i, dst._indices, dst._weights);
                    dst.Next();
                }
            }
        } else {
            for (int i=0; i<srcAlloc->GetNumStencils(); ++i) {
                *dst._size = srcAlloc->CopyStencil(i, dst._indices, dst._weights);
                dst.Next();
            }
        }

        if (options.generateOffsets) {
            result->generateOffsets();
        }
    }

    return result;
}

//------------------------------------------------------------------------------

StencilTables const *
StencilTablesFactory::Create(int numTables, StencilTables const ** tables) {

    StencilTables * result = new StencilTables;

    if ( (numTables<=0) or (not tables)) {
        return result;
    }

    int ncvs = tables[0]->GetNumControlVertices(),
        nstencils = 0,
        nelems = 0;

    for (int i=0; i<numTables; ++i) {

        StencilTables const & st = *tables[i];

        if (st.GetNumControlVertices()!=ncvs) {
            return result;
        }
        nstencils += st.GetNumStencils();
        nelems += (int)st.GetControlIndices().size();
    }

    result->resize(nstencils, nelems);

    unsigned char * sizes = &result->_sizes[0];
    Index * indices = &result->_indices[0];
    float * weights = &result->_weights[0];
    for (int i=0; i<numTables; ++i) {
        StencilTables const & st = *tables[i];

        int st_nstencils = st.GetNumStencils(),
            st_nelems = (int)st._indices.size();
        memcpy(sizes, &st._sizes[0], st_nstencils*sizeof(unsigned char));
        memcpy(indices, &st._indices[0], st_nelems*sizeof(Index));
        memcpy(weights, &st._weights[0], st_nelems*sizeof(float));

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

LimitStencilTables const *
LimitStencilTablesFactory::Create(TopologyRefiner const & refiner,
    LocationArrayVec const & locationArrays, StencilTables const * cvStencils,
        PatchTables const * patchTables) {

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

    int maxlevel = refiner.GetMaxLevel(), maxsize=17;

    StencilTables const * cvstencils = cvStencils;
    if (not cvstencils) {
        // Generate stencils for the control vertices - this is necessary to
        // properly factorize patches with control vertices at level 0 (natural
        // regular patches, such as in a torus)
        // note: the control vertices of the mesh are added as single-index
        //       stencils of weight 1.0f
        StencilTablesFactory::Options options;
        options.generateIntermediateLevels = uniform ? false :true;
        options.generateControlVerts = true;
        options.generateOffsets = true;

        // XXXX (manuelk) We could potentially save some mem-copies by not
        // instanciating the stencil tables and work directly off the pool
        // allocators.
        cvstencils = StencilTablesFactory::Create(refiner, options);
    } else {
        // Sanity checks
        if (cvstencils->GetNumStencils() != (uniform ?
            refiner.GetNumVertices(maxlevel) :
                refiner.GetNumVerticesTotal())) {
                return 0;
        }
    }

    // If a stencil table was given, use it, otherwise, create a new one
    PatchTables const * patchtables = patchTables;
    if (not patchTables) {
        // XXXX (manuelk) If no patch-tables was passed, we should be able to
        // infer the patches fairly easily from the refiner. Once more tags
        // have been added to the refiner, maybe we can remove the need for the
        // patch tables.

        OpenSubdiv::Far::PatchTablesFactory::Options options;
        options.adaptiveStencilTables = cvstencils;

        patchtables = PatchTablesFactory::Create(refiner, options);
    } else {
        // Sanity checks
        if (patchTables->IsFeatureAdaptive()==uniform) {
            if (not cvStencils) {
                assert(cvstencils and cvstencils!=cvStencils);
                delete cvstencils;
            }
            return 0;
        }
    }

    assert(patchtables and cvstencils);

    // Create a patch-map to locate sub-patches faster
    PatchMap patchmap( *patchtables );

    //
    // Generate limit stencils for locations
    //

    // Create a pool allocator to accumulate ProtoLimitStencils
    LimitStencilAllocator alloc(maxsize);
    alloc.Resize(numStencils);

    // XXXX (manuelk) we can make uniform (bilinear) stencils faster with a
    //       dedicated code path that does not use PatchTables or the PatchMap
    for (int i=0, currentStencil=0; i<(int)locationArrays.size(); ++i) {

        LocationArray const & array = locationArrays[i];

        assert(array.ptexIdx>=0);

        for (int j=0; j<array.numLocations; ++j, ++currentStencil) {

            float s = array.s[j],
                  t = array.t[j];

            PatchMap::Handle const * handle =
                patchmap.FindPatch(array.ptexIdx, s, t);

            if (handle) {
                ProtoLimitStencil dst = alloc[currentStencil];
                if (uniform) {
                    patchtables->Interpolate(*handle, s, t, *cvstencils, dst);
                } else {
                    patchtables->Limit(*handle, s, t, *cvstencils, dst);
                }
                ++numLimitStencils;
            }
        }
    }

    if (not cvStencils) {
        delete cvstencils;
    }

    //
    // Copy the proto-stencils into the limit stencil tables
    //
    LimitStencilTables * result = new LimitStencilTables;

    int nelems = alloc.GetNumVerticesTotal();
    if (nelems>0) {

        // Allocate
        result->resize(numLimitStencils, nelems);

        // Copy stencils
        LimitStencil dst(&result->_sizes.at(0), &result->_indices.at(0),
            &result->_weights.at(0), &result->_duWeights.at(0),
                &result->_dvWeights.at(0));

        for (int i=0; i<alloc.GetNumStencils(); ++i) {
            *dst._size = alloc.CopyLimitStencil(i, dst._indices, dst._weights,
                dst._duWeights, dst._dvWeights);
            dst.Next();
        }

        // XXXX manuelk should offset creation be optional ?
        result->generateOffsets();
    }
    result->_numControlVertices = refiner.GetNumVertices(0);

    return result;
}

//------------------------------------------------------------------------------

KernelBatch
StencilTablesFactory::Create(StencilTables const &stencilTables) {

    return KernelBatch( KernelBatch::KERNEL_STENCIL_TABLE,
        -1, 0, stencilTables.GetNumStencils());
}

} // end namespace Far

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv
