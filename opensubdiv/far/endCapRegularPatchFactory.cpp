//
//   Copyright 2015 Pixar
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

#include "../far/gregoryBasis.h"
#include "../far/endCapRegularPatchFactory.h"
#include "../far/endCapGregoryBasisPatchFactory.h"
#include "../far/error.h"
#include "../far/stencilTablesFactory.h"
#include "../far/topologyRefiner.h"

#include <cassert>
#include <cmath>
#include <cstring>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Far {

struct EndCapRegularPatchFactory::Context {
    Context(TopologyRefiner const *refiner) :
        refiner(refiner), numVertices(0), numPatches(0) { }

    TopologyRefiner const *refiner;
    std::vector<GregoryBasis::Point> vertexStencils;
    std::vector<GregoryBasis::Point> varyingStencils;
    int numVertices;
    int numPatches;
    std::vector<Index> vertIndices;
};

EndCapRegularPatchFactory::EndCapRegularPatchFactory(
    TopologyRefiner const & refiner) {

    _context = new Context(&refiner);
}

EndCapRegularPatchFactory::~EndCapRegularPatchFactory() {

    delete _context;
}

PatchDescriptor::Type
EndCapRegularPatchFactory::GetPatchType(
    PatchTablesFactoryBase::PatchFaceTag const &) const {

    return PatchDescriptor::REGULAR;
}

ConstIndexArray
EndCapRegularPatchFactory::GetTopology(
    Vtr::Level const& level, Index faceIndex,
    PatchTablesFactoryBase::PatchFaceTag const * /*levelPatchTags*/,
    int /*levelVertOffset*/) {

    // XXX: For now, always create new 16 indices for each patch.
    // we'll optimize later to share all regular control points with
    // other patches as well as to try to make extra ordinary verts watertight.

    int vertexOffset = _context->refiner->GetNumVerticesTotal();
    for (int i = 0; i < 16; ++i) {
        _context->vertIndices.push_back(_context->numVertices + vertexOffset);
        ++_context->numVertices;
    }

    // XXX: temporary hack. we should traverse topology and find existing
    //      vertices if available
    //
    // Reorder gregory basis stencils into regular bezier
    GregoryBasis::ProtoBasis basis(level, faceIndex);
    std::vector<GregoryBasis::Point> bezierCP;
    bezierCP.reserve(16);

    bezierCP.push_back(basis.P[0]);
    bezierCP.push_back(basis.Ep[0]);
    bezierCP.push_back(basis.Em[1]);
    bezierCP.push_back(basis.P[1]);

    bezierCP.push_back(basis.Em[0]);
    bezierCP.push_back(basis.Fp[0]); // arbitrary
    bezierCP.push_back(basis.Fp[1]); // arbitrary
    bezierCP.push_back(basis.Ep[1]);

    bezierCP.push_back(basis.Ep[3]);
    bezierCP.push_back(basis.Fp[3]); // arbitrary
    bezierCP.push_back(basis.Fp[2]); // arbitrary
    bezierCP.push_back(basis.Em[2]);

    bezierCP.push_back(basis.P[3]);
    bezierCP.push_back(basis.Em[3]);
    bezierCP.push_back(basis.Ep[2]);
    bezierCP.push_back(basis.P[2]);

    // Apply basis conversion from bezier to b-spline
    float Q[4][4] = {{ 6, -7,  2, 0},
                     { 0,  2, -1, 0},
                     { 0, -1,  2, 0},
                     { 0,  2, -7, 6} };
    std::vector<GregoryBasis::Point> H(16);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 4; ++k) {
                if (Q[i][k] != 0) H[i*4+j] += bezierCP[j+k*4] * Q[i][k];
            }
        }
    }
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            GregoryBasis::Point p;
            for (int k = 0; k < 4; ++k) {
                if (Q[j][k] != 0) p += H[i*4+k] * Q[j][k];
            }
            _context->vertexStencils.push_back(p);
        }
    }

    int varyingIndices[] = { 0, 0, 1, 1,
                             0, 0, 1, 1,
                             3, 3, 2, 2,
                             3, 3, 2, 2,};
    for (int i = 0; i < 16; ++i) {
        _context->varyingStencils.push_back(basis.V[varyingIndices[i]]);
    }

    ++_context->numPatches;
    return ConstIndexArray(
        &_context->vertIndices[(_context->numPatches-1)*16], 16);
}

StencilTables const *
EndCapRegularPatchFactory::createStencilTables(StencilAllocator &alloc,
                                               StencilTables const *baseStencils,
                                               bool append) {

    int nStencils = alloc.GetNumStencils();
    int nelems = alloc.GetNumVerticesTotal();

    // return NULL if empty
    if (nStencils==0 or nelems==0) {
        return NULL;
    }

    // Finalize the stencil tables from the temporary pool allocator
    StencilTables * result = new StencilTables;

    result->_numControlVertices = _context->refiner->GetNumVertices(0);

    result->resize(nStencils, nelems);

    Stencil dst(&result->_sizes.at(0),
        &result->_indices.at(0), &result->_weights.at(0));

    for (int i = 0; i < nStencils; ++i) {

        Index index = i;
        *dst._size = alloc.CopyStencil(index, dst._indices, dst._weights);

        dst.Next();
    }

    result->generateOffsets();

    if (append) {
        StencilTables const *inStencilTables[] = {
            baseStencils, result
        };
        StencilTables const *concatStencilTables =
            StencilTablesFactory::Create(2, inStencilTables);
        delete result;
        return concatStencilTables;
    } else {
        return result;
    }
}

static void
createStencil(StencilAllocator &alloc,
              StencilTables const *baseStencils,
              TopologyRefiner const *refiner,
              std::vector<GregoryBasis::Point> const &gregoryStencils) {

    // Gregory limit stencils have indices that are relative to the level
    // (maxlevel) of subdivision. These indices need to be offset to match
    // the indices from the multi-level adaptive stencil tables.
    // In addition: stencil tables can be built with singular stencils
    // (single weight of 1.0f) as place-holders for coarse mesh vertices,
    // which also needs to be accounted for.
    int stencilsIndexOffset = 0;
    {
        int maxlevel = refiner->GetMaxLevel();
        int nverts = refiner->GetNumVerticesTotal();
        int nBaseStencils = baseStencils->GetNumStencils();
        if (nBaseStencils == nverts) {

            // the table contain stencils for the control vertices
            stencilsIndexOffset = nverts - refiner->GetNumVertices(maxlevel);

        } else if (nBaseStencils == (nverts -refiner->GetNumVertices(0))) {

            // the table does not contain stencils for the control vertices
            stencilsIndexOffset = nverts - refiner->GetNumVertices(maxlevel)
                - refiner->GetNumVertices(0);

        } else {
            // these are not the stencils you are looking for...
            assert(0);
            return;
        }
    }

    int nStencils = (int)gregoryStencils.size();

    alloc.Resize(nStencils);
    for (int i = 0; i < nStencils; ++i) {
        GregoryBasis::Point p = gregoryStencils[i];
        p.OffsetIndices(stencilsIndexOffset);

        p.FactorizeBasisVertex(baseStencils, alloc[i]);
    }
}

StencilTables const *
EndCapRegularPatchFactory::CreateVertexStencilTables(
    StencilTables const *baseStencils,
    bool append) {

    // Factorize the basis CVs with the stencil tables: the basis is now
    // expressed as a linear combination of vertices from the coarse control
    // mesh with no data dependencies
    int maxvalence = _context->refiner->GetMaxValence();
    StencilAllocator alloc(GregoryBasis::getNumMaxElements(maxvalence));

    createStencil(alloc, baseStencils,
                  _context->refiner, _context->vertexStencils);

    return createStencilTables(alloc, baseStencils, append);
}

StencilTables const *
EndCapRegularPatchFactory::CreateVaryingStencilTables(
    StencilTables const *baseStencils,
    bool append) {

    int maxvalence = _context->refiner->GetMaxValence();
    StencilAllocator alloc(GregoryBasis::getNumMaxElements(maxvalence));

    createStencil(alloc, baseStencils,
                  _context->refiner, _context->varyingStencils);

    return createStencilTables(alloc, baseStencils, append);
}


} // end namespace Far

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv
