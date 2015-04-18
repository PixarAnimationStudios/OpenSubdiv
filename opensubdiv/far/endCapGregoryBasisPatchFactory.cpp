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

#include "../far/gregoryBasis.h"
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


static inline bool
checkMaxValence(Vtr::Level const & level) {
    if (level.getMaxValence()>EndCapGregoryBasisPatchFactory::GetMaxValence()) {
        // The proto-basis closed-form table limits valence to 'MAX_VALENCE'
        Error(FAR_RUNTIME_ERROR,
            "Vertex valence %d exceeds maximum %d supported",
                level.getMaxValence(), EndCapGregoryBasisPatchFactory::GetMaxValence());
        return false;
    }
    return true;
}


// ---------------------------------------------------------------------------
//
//  Factory context
//

struct EndCapGregoryBasisPatchFactory::Context {
    Context(TopologyRefiner const *refiner, bool shareBoundaryVertices) :
        refiner(refiner), shareBoundaryVertices(shareBoundaryVertices),
        numGregoryBasisVertices(0), numGregoryBasisPatches(0) { }

    std::vector<GregoryBasis::Point> vertexStencils;
    std::vector<GregoryBasis::Point> varyingStencils;

    TopologyRefiner const *refiner;
    bool shareBoundaryVertices;
    int numGregoryBasisVertices;
    int numGregoryBasisPatches;
    std::vector<Index> basisIndices;
    std::vector<Index> topology;
};

// ---------------------------------------------------------------------------

//
// EndCapGregoryBasisPatchFactory for Vertex StencilTables
//
EndCapGregoryBasisPatchFactory::EndCapGregoryBasisPatchFactory(
    TopologyRefiner const & refiner, bool shareBoundaryVertices) :
    _context(NULL) {

    // Sanity check: the mesh must be adaptively refined
    assert(not refiner.IsUniform());

    // create context
    _context = new EndCapGregoryBasisPatchFactory::Context(
        &refiner, shareBoundaryVertices);
}

EndCapGregoryBasisPatchFactory::~EndCapGregoryBasisPatchFactory() {
    delete _context;
}

int
EndCapGregoryBasisPatchFactory::GetMaxValence() {

    return GregoryBasis::MAX_VALENCE;
}

//
// Stateless EndCapGregoryBasisPatchFactory
//
GregoryBasis const *
EndCapGregoryBasisPatchFactory::Create(TopologyRefiner const & refiner,
    Index faceIndex, int fvarChannel) {

    // Gregory patches are end-caps: they only exist on max-level
    Vtr::Level const & level = refiner.getLevel(refiner.GetMaxLevel());

    if (not checkMaxValence(level)) {
        return 0;
    }

    GregoryBasis::ProtoBasis basis(level, faceIndex, fvarChannel);

    int nelems= basis.GetNumElements();

    GregoryBasis * result = new GregoryBasis;

    result->_indices.resize(nelems);
    result->_weights.resize(nelems);

    basis.Copy(result->_sizes, &result->_indices[0], &result->_weights[0]);

    // note: this function doesn't create varying stencils.

    for (int i=0, offset=0; i<20; ++i) {
        result->_offsets[i] = offset;
        offset += result->_sizes[i];
    }

    return result;
}

static void factorizeBasisVertex(StencilTables const * stencils,
                                 GregoryBasis::Point const & p,
                                 ProtoStencil dst) {
    // Use the Allocator to factorize the Gregory patch influence CVs with the
    // supporting CVs from the stencil tables.
    if (!stencils) return;

    dst.Clear();
    for (int j=0; j<p.GetSize(); ++j) {
        dst.AddWithWeight(*stencils,
            p.GetIndices()[j], p.GetWeights()[j]);
    }
}

bool
EndCapGregoryBasisPatchFactory::addPatchBasis(Index faceIndex,
                                   bool verticesMask[4][5]) {

    // Gregory patches only exist on the hight
    Vtr::Level const & level = _context->refiner->getLevel(
        _context->refiner->GetMaxLevel());

    if (not checkMaxValence(level)) {
        return false;
    }

    // Gather the CVs that influence the Gregory patch and their relative
    // weights in a basis
    GregoryBasis::ProtoBasis basis(level, faceIndex);

    for (int i = 0; i < 4; ++i) {
        if (verticesMask[i][0]) {
            _context->vertexStencils.push_back(basis.P[i]);
            _context->varyingStencils.push_back(basis.V[i]);
        }
        if (verticesMask[i][1]) {
            _context->vertexStencils.push_back(basis.Ep[i]);
            _context->varyingStencils.push_back(basis.V[i]);
        }
        if (verticesMask[i][2]) {
            _context->vertexStencils.push_back(basis.Em[i]);
            _context->varyingStencils.push_back(basis.V[i]);
        }
        if (verticesMask[i][3]) {
            _context->vertexStencils.push_back(basis.Fp[i]);
            _context->varyingStencils.push_back(basis.V[i]);
        }
        if (verticesMask[i][4]) {
            _context->vertexStencils.push_back(basis.Fm[i]);
            _context->varyingStencils.push_back(basis.V[i]);
        }
    }
    return true;
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

        factorizeBasisVertex(baseStencils, p, alloc[i]);
    }
}

StencilTables const *
EndCapGregoryBasisPatchFactory::createStencilTables(StencilAllocator &alloc,
                                                    StencilTables const *baseStencils,
                                                    bool append,
                                                    int const permute[20]) {

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
        if (permute) {
            int localIndex = i % 20,
                baseIndex = i - localIndex;
            index = baseIndex + permute[localIndex];
        }

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

PatchDescriptor::Type
EndCapGregoryBasisPatchFactory::GetPatchType(PatchTablesFactoryBase::PatchFaceTag const &) const {

    return PatchDescriptor::GREGORY_BASIS;
}

//
// Populates the topology table used by Gregory-basis patches
//
// Note  : 'faceIndex' values are expected to be sorted in ascending order !!!
// Note 2: this code attempts to identify basis vertices shared along
//         gregory patch edges
ConstIndexArray
EndCapGregoryBasisPatchFactory::GetTopology(
    Vtr::Level const& level, Index faceIndex,
    PatchTablesFactoryBase::PatchFaceTag const * levelPatchTags,
    int /*not used: levelVertsOffset*/)
{
    // allocate indices (awkward)
    // assert(Vtr::INDEX_INVALID==0xFFFFFFFF);
    for (int i = 0; i < 20; ++i) {
        _context->topology.push_back(Vtr::INDEX_INVALID);
    }
    Index * dest = &_context->topology[_context->numGregoryBasisPatches * 20];

    int gregoryVertexOffset = _context->refiner->GetNumVerticesTotal();

    if (_context->shareBoundaryVertices) {
        ConstIndexArray fedges = level.getFaceEdges(faceIndex);
        assert(fedges.size()==4);

        for (int i=0; i<4; ++i) {
            Index edge = fedges[i], adjface = 0;

            { // Gather adjacent faces
                ConstIndexArray adjfaces = level.getEdgeFaces(edge);
                for (int i=0; i<adjfaces.size(); ++i) {
                    if (adjfaces[i]==faceIndex) {
                        // XXXX manuelk if 'edge' is non-manifold, arbitrarily pick the
                        // next face in the list of adjacent faces
                        adjface = (adjfaces[(i+1)%adjfaces.size()]);
                        break;
                    }
                }
            }
            // We are looking for adjacent faces that:
            // - exist (no boundary)
            // - have already been processed (known CV indices)
            // - are also Gregory basis patches
            if (adjface!=Vtr::INDEX_INVALID and (adjface < faceIndex) and
                (not levelPatchTags[adjface]._isRegular)) {

                ConstIndexArray aedges = level.getFaceEdges(adjface);
                int aedge = aedges.FindIndexIn4Tuple(edge);
                assert(aedge!=Vtr::INDEX_INVALID);

                // Find index of basis in the list of basis already generated
                struct compare {
                    static int op(void const * a, void const * b) {
                        return *(Index *)a - *(Index *)b;
                    }
                };

                Index * ptr = (Index *)std::bsearch(&adjface,
                                                    &_context->basisIndices[0],
                                                    _context->basisIndices.size(),
                                                    sizeof(Index), compare::op);

                int srcBasisIdx = (int)(ptr - &_context->basisIndices[0]);

                if (!ptr) {
                    // if the adjface is hole, it won't be found
                    break;
                }
                assert(ptr
                       and srcBasisIdx>=0
                       and srcBasisIdx<(int)_context->basisIndices.size());

                // Copy the indices of CVs from the face on the other side of the
                // shared edge
                static int const gregoryEdgeVerts[4][4] = { { 0,  1,  7,  5},
                                                            { 5,  6, 12, 10},
                                                            {10, 11, 17, 15},
                                                            {15, 16,  2,  0} };
                Index * src = &_context->topology[srcBasisIdx*20];
                for (int j=0; j<4; ++j) {
                    // invert direction
                    // note that src  indices have already been offsetted.
                    dest[gregoryEdgeVerts[i][3-j]] = src[gregoryEdgeVerts[aedge][j]];
                }
            }
        }
    }

    bool newVerticesMask[4][5];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 5; ++j) {
            if (dest[i*5+j]==Vtr::INDEX_INVALID) {
                // assign new vertex
                dest[i*5+j] =
                    _context->numGregoryBasisVertices + gregoryVertexOffset;
                ++_context->numGregoryBasisVertices;
                newVerticesMask[i][j] = true;
            } else {
                // share vertex
                newVerticesMask[i][j] = false;
            }
        }
    }
    _context->basisIndices.push_back(faceIndex);

    // add basis
    addPatchBasis(faceIndex, newVerticesMask);

    ++_context->numGregoryBasisPatches;

    // return cvs;
    return ConstIndexArray(dest, 20);
}

Index
EndCapGregoryBasisPatchFactory::GetFaceIndex(Index patchIndex) const {

    return _context->basisIndices[patchIndex];
}

int
EndCapGregoryBasisPatchFactory::GetNumGregoryBasisPatches() const {

    return _context->numGregoryBasisPatches;
}

int
EndCapGregoryBasisPatchFactory::GetNumGregoryBasisVertices() const {

    return _context->numGregoryBasisVertices;
}

StencilTables const *
EndCapGregoryBasisPatchFactory::CreateVertexStencilTables(
    StencilTables const *baseStencils,
    bool append,
    int const permute[20]) {

    // Factorize the basis CVs with the stencil tables: the basis is now
    // expressed as a linear combination of vertices from the coarse control
    // mesh with no data dependencies
    int maxvalence = _context->refiner->GetMaxValence();
    StencilAllocator alloc(GregoryBasis::getNumMaxElements(maxvalence));

    createStencil(alloc, baseStencils,
                  _context->refiner, _context->vertexStencils);

    return createStencilTables(alloc, baseStencils, append, permute);
}

StencilTables const *
EndCapGregoryBasisPatchFactory::CreateVaryingStencilTables(
    StencilTables const *baseStencils,
    bool append,
    int const permute[20]) {

    int maxvalence = _context->refiner->GetMaxValence();
    StencilAllocator alloc(GregoryBasis::getNumMaxElements(maxvalence));

    createStencil(alloc, baseStencils,
                  _context->refiner, _context->varyingStencils);

    return createStencilTables(alloc, baseStencils, append, permute);
}

} // end namespace Far

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv
