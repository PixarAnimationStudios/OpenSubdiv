//
//   Copyright 2014 DreamWorks Animation LLC.
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
#include "../far/stencilTables.h"
#include "../far/topologyRefiner.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <vector>
#include <map>

namespace {

class ProtoStencilAllocator;

typedef OpenSubdiv::Far::Index Index;

//------------------------------------------------------------------------------

//
// ProtoStencil
//
// ProtoStencils are used to interpolate stencils from supporting vertices.
// These stencils are backed by a pool allocator (ProtoStencilAllocator) to
// allow for fast push-back of additional vertex weights & indices.
//
class ProtoStencil {

public:

    // Return stencil unique ID in pool allocator
    Index GetID() const {
        return _ID;
    }

    // Set stencil weights to 0.0
    void Clear();

    // Weighted add for coarse vertices (size=1, weight=1.0f)
    void AddWithWeight(int, float weight);

    // Weighted add of a Stencil
    void AddWithWeight(ProtoStencil const & src, float weight);

    // Weighted add for coarse vertices (size=1, weight=1.0f)
    void AddVaryingWithWeight(int, float);

    // Weighted add of a Stencil
    void AddVaryingWithWeight(ProtoStencil const &, float);

    // Returns the current size of the Stencil
    int GetSize() const;

    // Returns a pointer to the vertex indices of the stencil
    Index const * GetIndices() const;

    // Returns a pointer to the vertex weights of the stencil
    float const * GetWeights() const;

    // Debug output
    void Print() const;

    // Comparison operator to sort stencils by size
    static bool CompareSize(ProtoStencil const & a, ProtoStencil const & b) {
        return (a.GetSize() < b.GetSize());
    }

protected:

    friend class ProtoStencilAllocator;

    // Returns the location of vertex 'vertex' in the stencil indices or -1
    Index findVertex(Index vertex);

protected:

    Index _ID;                        // Stencil ID in allocator
    ProtoStencilAllocator * _alloc; // Pool allocator
};

typedef std::vector<ProtoStencil> ProtoStencilVec;



//------------------------------------------------------------------------------

//
// ProtoLimitStencil
//
// ProtoStencil class extended to support interpolation of derivatives.
//
class ProtoLimitStencil : public ProtoStencil {

public:

    typedef OpenSubdiv::Far::Stencil Stencil;

    // Returns a pointer to the vertex U derivative weights of the stencil
    float const * GetDuWeights() const;

    // Returns a pointer to the vertex U derivative weights of the stencil
    float const * GetDvWeights() const;

    // Set stencil weights to 0.0
    void Clear();

    // Weighted add of a LimitStencil
    void AddWithWeight(Stencil const & src, float w, float wDu, float wDv);
};

typedef std::vector<ProtoLimitStencil> ProtoLimitStencilVec;



//------------------------------------------------------------------------------

//
// Stencil pool allocator
//
// Strategy: allocate up-front a data pool for supporting stencils of a size
// slightly above average. For the (rare) stencils that require more support
// vertices, switch to (slow) heap allocation.
//
class ProtoStencilAllocator {

public:

    enum Mode {
        INTERPOLATE_VERTEX,
        INTERPOLATE_VARYING,
        INTERPOLATE_LIMITS,
    };

    typedef OpenSubdiv::Far::TopologyRefiner TopologyRefiner;

    // Constructor
    ProtoStencilAllocator(TopologyRefiner const & refiner, Mode mode);

    // Destructor
    ~ProtoStencilAllocator() ;

    // Returns an array of all the Stencils in the allocator
    ProtoStencilVec & GetStencils() {
        return _stencils;
    }

    // Returns an array of all the Stencils in the allocator
    ProtoLimitStencilVec & GetLimitStencils() {
        return reinterpret_cast<ProtoLimitStencilVec &>(_stencils);
    }

    // Returns the stencil interpolation mode
    Mode GetMode() const {
        return _mode;
    }

    // Append a support vertex of index 'index' and weight 'weight' to the
    // Stencil 'stencil' (use findVertex() to make sure it does not exist
    // yet)
    void PushBackVertex(ProtoStencil & stencil, Index index, float weight);

    // Append a support vertex of index 'index' and weight 'weight' to the
    // LimitStencil 'stencil' (use findVertex() to make sure it does not exist
    // yet)
    void PushBackVertex(ProtoLimitStencil & stencil, Index index,
        float weight, float duweight, float dvweight);

    // Allocate enough memory to hold 'numStencils' Stencils
    void Resize(int numStencils);

    // Returns the number of stencil vertices that have been pushed back
    int GetNumVertices() const;

private:

    friend class ProtoStencil;
    friend class ProtoLimitStencil;

    // returns the size of the stencil
    unsigned char * getSize(Index stencilID) {
        assert(stencilID<(int)_sizes.size());
        return &_sizes[stencilID];
    }

    // returns the indices of the stencil
    Index * getIndices(Index stencilID) {
        if (*getSize(stencilID)<_maxsize) {
            return &_indices[stencilID*_maxsize];
        } else {
            if (GetMode()==INTERPOLATE_LIMITS) {
                return &_biglimitstencils[stencilID]->indices[0];
            } else {
                return &_bigstencils[stencilID]->indices[0];
            }
        }
    }

    // returns the weights of the stencil
    float * getWeights(Index stencilID) {
        if (*getSize(stencilID)<_maxsize) {
            return &_weights[stencilID*_maxsize];
        } else {
            if (GetMode()==INTERPOLATE_LIMITS) {
                return &_biglimitstencils[stencilID]->weights[0];
            } else {
                return &_bigstencils[stencilID]->weights[0];
            }
        }
    }

    // returns the U derivative weights of the stencil
    float * getDuWeights(Index stencilID) {
        assert(GetMode()==INTERPOLATE_LIMITS);
        if (*getSize(stencilID)<_maxsize) {
            return &_duWeights[stencilID*_maxsize];
        } else {
            return &_biglimitstencils[stencilID]->duWeights[0];
        }
    }

    // returns the V derivative weights of the stencil
    float * getDvWeights(Index stencilID) {
        assert(GetMode()==INTERPOLATE_LIMITS);
        if (*getSize(stencilID)<_maxsize) {
            return &_dvWeights[stencilID*_maxsize];
        } else {
            return &_biglimitstencils[stencilID]->dvWeights[0];
        }
    }

private:

    Mode _mode;

    int _maxsize; // maximum size of a pre-allocated stencil

    ProtoStencilVec _stencils;

    std::vector<unsigned char> _sizes;    // temp stencils data (as SOA)
    std::vector<Index>    _indices;
    std::vector<float>         _weights;
    std::vector<float>         _duWeights;
    std::vector<float>         _dvWeights;

    //
    // When proto-stencils exceed _maxsize, fall back to heap allocated
    // "BigStencils"
    //
    struct BigStencil {

        BigStencil(int size, Index const * iindices, float const * iweights) {
            indices.reserve(size+5); indices.resize(size);
            weights.reserve(size+5); weights.resize(size);
            memcpy(&indices.at(0), iindices, size*sizeof(int) );
            memcpy(&weights.at(0), iweights, size*sizeof(int) );
        }

        std::vector<int>   indices;
        std::vector<float> weights;
    };

    typedef std::map<int, BigStencil *> BigStencilMap;

    BigStencilMap _bigstencils;


    //
    // Same as "BigStencil", except with limit derivatives
    //
    struct BigLimitStencil : public BigStencil {

        BigLimitStencil(int size, Index const * iindices, float const * iweights,
            float const * iduWeights, float const * idvWeights) :
                BigStencil(size, iindices, iweights) {

            duWeights.reserve(size+10); duWeights.resize(size);
            dvWeights.reserve(size+10); dvWeights.resize(size);

            memcpy(&duWeights.at(0), iduWeights, size*sizeof(int) );
            memcpy(&dvWeights.at(0), idvWeights, size*sizeof(int) );
        }

        std::vector<float> duWeights,
                           dvWeights;
    };

    typedef std::map<int, BigLimitStencil *> BigLimitStencilMap;

    BigLimitStencilMap _biglimitstencils;
};

// Constructor
ProtoStencilAllocator::ProtoStencilAllocator(
    TopologyRefiner const & refiner, Mode mode) : _mode(mode) {

    using namespace OpenSubdiv;

    // Make an educated guess as to what the max size should be
    switch (mode) {
        case INTERPOLATE_VERTEX : {
            Sdc::Type type = refiner.GetSchemeType();
            switch (type) {
                case Sdc::TYPE_BILINEAR : _maxsize = 5; break;
                case Sdc::TYPE_CATMARK  : _maxsize = 17; break;
                case Sdc::TYPE_LOOP     : _maxsize = 10; break;
                default:
                    assert(0);
            }
        } break;
        case INTERPOLATE_VARYING : _maxsize = 5; break;
        case INTERPOLATE_LIMITS : _maxsize = 17; break;
    }
}

// Destructor
ProtoStencilAllocator::~ProtoStencilAllocator() {


    for (BigStencilMap::iterator it=_bigstencils.begin();
        it!=_bigstencils.end(); ++it) {

        delete it->second;
    }

    for (BigLimitStencilMap::iterator it=_biglimitstencils.begin();
        it!=_biglimitstencils.end(); ++it) {

        delete it->second;
    }
}

// Allocate enough memory to hold 'numStencils' Stencils
void
ProtoStencilAllocator::Resize(int numStencils) {

    int currentSize = (int)_stencils.size();

    // Pre-allocate the Stencils
    _stencils.resize(numStencils);

    for (Index i=currentSize; i<numStencils; ++i) {
        _stencils[i]._ID = i;
        _stencils[i]._alloc = this;
    }

    int nelems = numStencils * _maxsize;
    _sizes.clear();
    _sizes.resize(numStencils);
    _indices.resize(nelems);
    _weights.resize(nelems);
    if (_mode==INTERPOLATE_LIMITS) {
        _duWeights.resize(nelems);
        _dvWeights.resize(nelems);
    }

    for (BigStencilMap::iterator it=_bigstencils.begin();
        it!=_bigstencils.end(); ++it) {

        delete it->second;
    }
    _bigstencils.clear();

    for (BigLimitStencilMap::iterator it=_biglimitstencils.begin();
        it!=_biglimitstencils.end(); ++it) {

        delete it->second;
    }
    _biglimitstencils.clear();
}

// Append a support vertex of index 'index' and weight 'weight' to the
// Stencil 'stencil' (use findVertex() to make sure it does not exist
// yet)
void
ProtoStencilAllocator::PushBackVertex(
    ProtoStencil & stencil, Index index, float weight) {

    assert(weight!=0.0f);

    unsigned char * size    = getSize(stencil.GetID());
    Index    * indices = getIndices(stencil.GetID());
    float         * weights = getWeights(stencil.GetID());


    if (*size<(_maxsize-1)) {

        // The stencil still fits in pool memory, just copy the data
        indices[*size]=index;
        weights[*size]=weight;
    } else {

        // The stencil is now too big: fall back to heap memory
        BigStencil * dst=0;

        // Is this stencil already a BigStencil or do we need a new one ?
        if (*size==(_maxsize-1)) {
            dst = new BigStencil(*size, indices, weights);
            assert(_bigstencils.find(stencil.GetID())==_bigstencils.end());
            _bigstencils[stencil.GetID()]=dst;
        } else {
            assert(_bigstencils.find(stencil.GetID())!=_bigstencils.end());
            dst = _bigstencils[stencil.GetID()];
        }
        assert(dst);

        // push back the new vertex
        dst->indices.push_back(index);
        dst->weights.push_back(weight);
    }
    ++(*size);
}

// Append a support vertex of index 'index' and weight 'weight' to the
// LimitStencil 'stencil' (use findVertex() to make sure it does not exist
// yet)
void
ProtoStencilAllocator::PushBackVertex(ProtoLimitStencil & stencil,
    Index index, float weight, float duweight, float dvweight) {

    assert(weight!=0.0f);

    unsigned char * size    = getSize(stencil.GetID());
    Index    * indices = getIndices(stencil.GetID());
    float         * weights = getWeights(stencil.GetID()),
                  * duweights = getDuWeights(stencil.GetID()),
                  * dvweights = getDvWeights(stencil.GetID());

    if (*size<(_maxsize-1)) {

        // The stencil still fits in pool memory, just copy the data
        indices[*size]=index;
        weights[*size]=weight;
        duweights[*size]=duweight;
        dvweights[*size]=dvweight;
    } else {

        // The stencil is now too big: fall back to heap memory
        BigLimitStencil * dst=0;

        // Is this stencil already a BigLimitStencil or do we need a new one ?
        if (*size==(_maxsize-1)) {
            dst = new BigLimitStencil(*size, indices, weights, duweights, dvweights);
            assert(_biglimitstencils.find(stencil.GetID())==_biglimitstencils.end());
            _biglimitstencils[stencil.GetID()]=dst;
        } else {
            assert(_biglimitstencils.find(stencil.GetID())!=_biglimitstencils.end());
            dst = _biglimitstencils[stencil.GetID()];
        }
        assert(dst);

        // push back the new vertex
        dst->indices.push_back(index);
        dst->weights.push_back(weight);
        dst->duWeights.push_back(duweight);
        dst->dvWeights.push_back(dvweight);
    }
    ++(*size);
}

// Returns the number of stencil vertices that have been pushed back
int
ProtoStencilAllocator::GetNumVertices() const {

    int nverts=0;
    for (int i=0; i<(int)_stencils.size(); ++i) {
        nverts+=_stencils[i].GetSize();
    }
    return nverts;
}

// Returns the current size of the Stencil
int
ProtoStencil::GetSize() const {
    return (int)*_alloc->getSize(this->GetID());
}

// Returns a pointer to the vertex indices of the stencil
Index const *
ProtoStencil::GetIndices() const {
    return _alloc->getIndices(this->GetID());
}

// Returns a pointer to the vertex weights of the stencil
float const *
ProtoStencil::GetWeights() const {
    return _alloc->getWeights(this->GetID());
}

// Returns a pointer to the vertex weights of the stencil
float const *
ProtoLimitStencil::GetDuWeights() const {
    return _alloc->getDuWeights(this->GetID());
}

// Returns a pointer to the vertex weights of the stencil
float const *
ProtoLimitStencil::GetDvWeights() const {
    return _alloc->getDvWeights(this->GetID());
}

// Debug dump
void
ProtoStencil::Print() const {

    printf("tempStencil size=%d indices={ ", GetSize());
    for (int i=0; i<GetSize(); ++i) {
        printf("%d ", GetIndices()[i]);
    }
    printf("} weights={ ");
    for (int i=0; i<GetSize(); ++i) {
        printf("%f ", GetWeights()[i]);
    }
    printf("}\n");
}

// Find the location of vertex 'vertex' in the stencil indices.
inline Index
ProtoStencil::findVertex(Index vertex) {

    // XXXX manuelk serial search -> we can figure out something better ?
    unsigned char * size    = _alloc->getSize(this->GetID());
    Index    * indices = _alloc->getIndices(this->GetID());
    for (int i=0; i<*size; ++i) {
        if (indices[i]==vertex)
            return i;
    }
    return -1;
}

// Set stencil weights to 0.0
void
ProtoStencil::Clear() {
    float * weights = _alloc->getWeights(this->GetID());
    for (int i=0; i<*_alloc->getSize(this->GetID()); ++i) {
        weights[i]=0.0f;
    }
}

// Weighted add of a coarse vertex
inline void
ProtoStencil::AddWithWeight(Index vertIndex, float weight) {

    if (weight==0.0f) {
        return;
    }

    Index n = findVertex(vertIndex);
    if (n<0) {
        _alloc->PushBackVertex(*this, vertIndex, weight);
    } else {
        float * dstWeights = _alloc->getWeights(this->GetID());
        dstWeights[n] += weight;
        assert(dstWeights[n]>0.0f);
    }
}

// Weighted add of a Stencil
inline void
ProtoStencil::AddWithWeight(ProtoStencil const & src, float weight) {

    if (weight==0.0f) {
        return;
    }

    unsigned char const * srcSize    = src._alloc->getSize(src.GetID());
    Index const    * srcIndices = src._alloc->getIndices(src.GetID());
    float const         * srcWeights = src._alloc->getWeights(src.GetID());

    for (int i=0; i<*srcSize; ++i) {

        assert(srcWeights[i]!=0.0f);
        float w = weight * srcWeights[i];

        if (w==0.0f) {
            continue;
        }

        Index vertIndex = srcIndices[i];

        // Attempt to locate the vertex index in the list of supporting vertices
        // of the destination stencil.
        Index n = findVertex(vertIndex);
        if (n<0) {
            _alloc->PushBackVertex(*this, vertIndex, w);
        } else {
            float * dstWeights = _alloc->getWeights(this->GetID());
            dstWeights[n] += w;
            assert(dstWeights[n]!=0.0f);
        }
    }
}

inline void
ProtoStencil::AddVaryingWithWeight(Index vertIndex, float weight) {

    if (_alloc->GetMode()==ProtoStencilAllocator::INTERPOLATE_VARYING) {
        AddWithWeight(vertIndex, weight);
    }
}

inline void
ProtoStencil::AddVaryingWithWeight(ProtoStencil const & src, float weight) {

    if (_alloc->GetMode()==ProtoStencilAllocator::INTERPOLATE_VARYING) {
        AddWithWeight(src, weight);
    }
}

// Clear ProtoLimitStencil
void
ProtoLimitStencil::Clear() {
    float * weights = _alloc->getWeights(this->GetID()),
          * duweights = _alloc->getDuWeights(this->GetID()),
          * dvweights = _alloc->getDvWeights(this->GetID());
    for (int i=0; i<*_alloc->getSize(this->GetID()); ++i) {
        weights[i]=0.0f;
        duweights[i]=0.0f;
        dvweights[i]=0.0f;
    }
}

// Weighted add on a ProtoLimitStencil
inline void
ProtoLimitStencil::AddWithWeight(Stencil const & src,
    float weight, float duWeight, float dvWeight) {

    if (weight==0.0f) {
        return;
    }

    unsigned char const * srcSize    = src.GetSizePtr();
    Index const    * srcIndices = src.GetVertexIndices();
    float const         * srcWeights = src.GetWeights();

    for (int i=0; i<*srcSize; ++i) {

        Index vertIndex = srcIndices[i];

        float srcWeight = srcWeights[i];

        if (srcWeight==0.0f) {
            continue;
        }

        Index n = findVertex(vertIndex);

        if (n<0) {
            _alloc->PushBackVertex(*this, vertIndex,
                weight * srcWeight, duWeight*srcWeight, dvWeight*srcWeight);
        } else {
            float   * dstWeights = _alloc->getWeights(this->GetID()),
                  * dstDuWeights = _alloc->getDuWeights(this->GetID()),
                  * dstDvWeights = _alloc->getDvWeights(this->GetID());

              dstWeights[n] +=   weight * srcWeight;
            dstDuWeights[n] += duWeight * srcWeight;
            dstDvWeights[n] += dvWeight * srcWeight;
        }
    }
}

} // end namespace unnamed

//------------------------------------------------------------------------------

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Far {

static void
generateOffsets(std::vector<unsigned char> const & sizes,
    std::vector<Index> & offsets ) {

    Index offset=0;
    for (int i=0; i<(int)sizes.size(); ++i ) {
        //assert(sizes[i]!=0);
        offsets[i]=offset;
        offset+=sizes[i];
    }
}

// Copy a stencil into StencilTables
template <> void
StencilTablesFactory::copyStencil(ProtoStencil const & src, Stencil & dst) {

    unsigned char size = (unsigned char)src.GetSize();
    Index const * indices = src.GetIndices();
    float const * weights = src.GetWeights();

    *dst._size = size;
    for (unsigned char i=0; i<size; ++i) {
        memcpy(dst._indices, indices, size*sizeof(int));
        memcpy(dst._weights, weights, size*sizeof(float));
    }
}

// (Sort &) Copy a vector of stencils into StencilTables
template <> void
StencilTablesFactory::copyStencils(ProtoStencilVec & src,
    Stencil & dst, bool sortBySize) {

    if (sortBySize) {
        std::sort(src.begin(), src.end(), ProtoStencil::CompareSize);
    }

    for (int i=0; i<(int)src.size(); ++i) {
        copyStencil(src[i], dst);
        dst.Next();
    }
}

void
StencilTablesFactory::generateControlVertStencils(
    int numControlVerts, Stencil & dst) {

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

    int maxlevel = refiner.GetMaxLevel();
    if (maxlevel==0) {
        return result;
    }

    ProtoStencilAllocator::Mode mode=ProtoStencilAllocator::INTERPOLATE_VERTEX;
    switch (options.interpolationMode) {
        case INTERPOLATE_VERTEX:
            mode = ProtoStencilAllocator::INTERPOLATE_VERTEX; break;
        case INTERPOLATE_VARYING:
            mode = ProtoStencilAllocator::INTERPOLATE_VARYING; break;
        default:
            assert(0);
    }

    std::vector<ProtoStencilAllocator> allocators(
        options.generateAllLevels ? maxlevel : 2,
            ProtoStencilAllocator(refiner, mode));

    ProtoStencilAllocator * srcAlloc, * dstAlloc;
    if (options.generateAllLevels) {
        srcAlloc = 0;
        dstAlloc = &allocators[0];
    } else {
        srcAlloc = &allocators[0];
        dstAlloc = &allocators[1];
    }

    // Interpolate stencils for each refinement level using
    // TopologyRefiner::InterpolateLevel<>()

    for (int level=1;level<=maxlevel; ++level) {

        dstAlloc->Resize(refiner.GetNumVertices(level));

        if (level==1) {

            // coarse vertices have a single index and a weight of 1.0f
            Index * srcStencils = new Index[refiner.GetNumVertices(0)];
            for (int i=0; i<refiner.GetNumVertices(0); ++i) {
                srcStencils[i]=i;
            }

            ProtoStencil * dstStencils = &(dstAlloc->GetStencils()).at(0);

            if (mode==ProtoStencilAllocator::INTERPOLATE_VERTEX) {
                refiner.Interpolate(level, srcStencils, dstStencils);
            } else {
                refiner.InterpolateVarying(level, srcStencils, dstStencils);
            }

            delete [] srcStencils;
        } else {

            ProtoStencil * srcStencils = &(srcAlloc->GetStencils()).at(0),
                         * dstStencils = &(dstAlloc->GetStencils()).at(0);

            if (mode==ProtoStencilAllocator::INTERPOLATE_VERTEX) {
                refiner.Interpolate(level, srcStencils, dstStencils);
            } else {
                refiner.InterpolateVarying(level, srcStencils, dstStencils);
            }
        }

        if (options.generateAllLevels) {
            if (level<maxlevel) {
                srcAlloc = &allocators[level-1];
                dstAlloc = &allocators[level];
            }
        } else {
            std::swap(srcAlloc, dstAlloc);
        }
    }

    // Sort & Copy stencils into tables
    {
        result->_numControlVertices = refiner.GetNumVertices(0);

        // Add total number of stencils, weights & indices
        int nelems = 0, nstencils=0;
        if (options.generateAllLevels) {

            for (int level=0; level<maxlevel; ++level) {
                nstencils += (int)allocators[level].GetStencils().size();
                nelems += allocators[level].GetNumVertices();
            }
        } else {
            nstencils = (int)srcAlloc->GetStencils().size();
            nelems = srcAlloc->GetNumVertices();
        }


        { // Allocate
            if (options.generateControlVerts) {
                nstencils += result->_numControlVertices;
                nelems += result->_numControlVertices;
            }

            result->_sizes.resize(nstencils);
            if (options.generateOffsets) {
                result->_offsets.resize(nstencils);
            }
            result->_indices.resize(nelems);
            result->_weights.resize(nelems);
        }

        // Copy stencils
        Stencil dst(&result->_sizes.at(0),
            &result->_indices.at(0), &result->_weights.at(0));

        if (options.generateControlVerts) {
            generateControlVertStencils(result->_numControlVertices, dst);
        }

        bool doSort = options.sortBySize!=0;

        if (options.generateAllLevels) {
            for (int level=0; level<maxlevel; ++level) {
                copyStencils(allocators[level].GetStencils(), dst, doSort);
            }
        } else {
            copyStencils(srcAlloc->GetStencils(), dst, doSort);
        }

        if (options.generateOffsets) {
            generateOffsets(result->_sizes, result->_offsets);
        }
    }
    return result;
}


//------------------------------------------------------------------------------

// Copy a stencil into StencilTables
template <> void
LimitStencilTablesFactory::copyLimitStencil(
    ProtoLimitStencil const & src, LimitStencil & dst) {

    unsigned char size = (unsigned char)src.GetSize();
    Index const * indices = src.GetIndices();
    float const * weights = src.GetWeights(),
                * duWeights = src.GetDuWeights(),
                * dvWeights = src.GetDvWeights();

    *dst._size = size;
    for (unsigned char i=0; i<size; ++i) {
        memcpy(dst._indices, indices, size*sizeof(int));
        memcpy(dst._weights, weights, size*sizeof(float));
        memcpy(dst._duWeights, duWeights, size*sizeof(float));
        memcpy(dst._dvWeights, dvWeights, size*sizeof(float));
    }
}

// (Sort &) Copy a vector of stencils into StencilTables
template <> void
LimitStencilTablesFactory::copyLimitStencils(
    ProtoLimitStencilVec & src, LimitStencil & dst) {

    for (int i=0; i<(int)src.size(); ++i) {
        copyLimitStencil(src[i], dst);
        dst.Next();
    }
}

//------------------------------------------------------------------------------

LimitStencilTables const *
LimitStencilTablesFactory::Create(TopologyRefiner const & refiner,
    LocationArrayVec const & locationArrays, StencilTables const * cvStencils,
        PatchTables const * patchTables) {

    bool uniform = refiner.IsUniform();

    // Compute the total number of stencils to generate
    int numStencils=0, numLimitStencils=0;
    for (int i=0; i<(int)locationArrays.size(); ++i) {
        assert(locationArrays[i].numLocations>=0);
        numStencils += locationArrays[i].numLocations;
    }

    if (numStencils<=0) {
        return 0;
    }

    int maxlevel = refiner.GetMaxLevel();

    StencilTables const * _cvStencils = cvStencils;
    if (not _cvStencils) {
        // Generate stencils for the control vertices
        // note: the control vertices of the mesh are added as single-index
        //       stencils of weight 1.0f
        StencilTablesFactory::Options options;
        options.generateAllLevels = uniform ? false :true;
        options.generateControlVerts = true;
        options.generateOffsets = true;

        _cvStencils = StencilTablesFactory::Create(refiner, options);
    } else {
        // sanity checks
        if (_cvStencils->GetNumStencils() != (uniform ?
            refiner.GetNumVertices(maxlevel) :
                refiner.GetNumVerticesTotal())) {
                return 0;
        }
    }
    
    PatchTables const * _patchTables = patchTables;
    if (not patchTables) {
        _patchTables = PatchTablesFactory::Create(refiner);
    } else {
        // sanity checks
        if (patchTables->IsFeatureAdaptive()==uniform) {
            // XXXX smart pointers /grumble
            if (not cvStencils) {
                delete _cvStencils;
            }
            return 0;
        }
    }
    
    assert( _patchTables and _cvStencils );

    // Create a patch-map to locate sub-patches faster
    PatchMap patchmap( *_patchTables );

    // Create a pool allocator to accumulate ProtoLimitStencils
    ProtoStencilAllocator alloc(refiner,
        ProtoStencilAllocator::INTERPOLATE_LIMITS);

    alloc.Resize(numStencils);

    // Generate limit stencils for locations
    
    // XXXX manuelk we can make uniform much faster with a dedicated 
    //              code path that does not use PatchTables
    for (int i=0, currentStencil=0; i<(int)locationArrays.size(); ++i) {

        LocationArray const & array = locationArrays[i];

        assert(array.ptexIdx>=0);

        for (int j=0; j<array.numLocations; ++j, ++currentStencil) {

            float s = array.s[j],
                  t = array.t[j];

            PatchMap::Handle const * handle =
                patchmap.FindPatch(array.ptexIdx, s, t);

            if (handle) {

                ProtoLimitStencil & dst =
                    alloc.GetLimitStencils()[currentStencil];

                if (uniform) {
                    _patchTables->Interpolate(*handle, s, t, *_cvStencils, &dst);
                } else {
                    _patchTables->Limit(*handle, s, t, *_cvStencils, &dst);
                }
                ++numLimitStencils;
            }
        }
    }

    // Sort & Copy the proto stencils into the limit stencil tables
    LimitStencilTables * result = new LimitStencilTables;

    int nelems = alloc.GetNumVertices();
    if (nelems>0) {

        // Allocate
        result->_sizes.resize(numLimitStencils);
        result->_offsets.resize(numLimitStencils);
        result->_indices.resize(nelems);
        result->_weights.resize(nelems);
        result->_duWeights.resize(nelems);
        result->_dvWeights.resize(nelems);

        // Copy stencils
        LimitStencil dst(&result->_sizes.at(0),
                         &result->_indices.at(0),
                         &result->_weights.at(0),
                         &result->_duWeights.at(0),
                         &result->_dvWeights.at(0));

        copyLimitStencils(alloc.GetLimitStencils(), dst);

        generateOffsets(result->_sizes, result->_offsets);
    }
    result->_numControlVertices = refiner.GetNumVertices(0);

    if (not cvStencils) {
        delete _cvStencils;
    }

    if (not patchTables) {
        delete _patchTables;
    }

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
