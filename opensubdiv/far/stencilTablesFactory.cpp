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

#include "../far/stencilTables.h"
#include "../far/stencilTablesFactory.h"
#include "../far/topologyRefiner.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <vector>
#include <map>

namespace {

class StencilAllocator;

//
// Stencil
//
// Temporary stencil used to interpolate stencils from supporting vertices.
// These stencils are backed by a pool allocator to allow for fast push-back
// of additional vertex weights & indices.
//
class Stencil {

public:

    // Return stencil unique ID in pool allocator
    int GetID() const {
        return _ID;
    }

    // Set stencil weights to 0.0
    void Clear();

    // Weighted add for coarse vertices (size=1, weight=1.0f)
    void AddWithWeight(int, float weight);

    // Weighted add of a Stencil
    void AddWithWeight(Stencil const & src, float weight);

    // Weighted add for coarse vertices (size=1, weight=1.0f)
    void AddVaryingWithWeight(int, float);

    // Weighted add of a Stencil
    void AddVaryingWithWeight(Stencil const &, float);

    // Returns the current size of the Stencil
    int GetSize() const;

    // Returns a pointer to the vertex indices of the stencil
    int const * GetIndices() const;

    // Returns a pointer to the vertex weights of the stencil
    float const * GetWeights() const;

    // Debug output
    void Print() const;

    // Comparison operator to sort stencils by size
    static bool CompareSize(Stencil const & a, Stencil const & b) {
        return (a.GetSize() < b.GetSize());
    }

private:

    friend class StencilAllocator;

    // Returns the location of vertex 'vertex' in the stencil indices or -1
    int findVertex(int vertex);

private:

    int _ID;                   // Stencil ID in allocator
    StencilAllocator * _alloc; // Pool allocator
};

typedef std::vector<Stencil> StencilVec;

//
// Stencil pool allocator
//
// Strategy: allocate up-front a data pool for supporting stencils of a size
// slightly above average. For the (rare) stencils that require more support
// vertices, switch to (slow) heap allocation.
//
class StencilAllocator {

public:

    // Constructor
    StencilAllocator(OpenSubdiv::Far::TopologyRefiner const & refiner,
        OpenSubdiv::Far::StencilTablesFactory::Mode mode);

    // Destructor
    ~StencilAllocator() ;

    // Returns the stencil interpolation mode
    // Returns an array of all the Stencils in the allocator
    StencilVec & GetStencils() {
        return _stencils;
    }
    
    // Returns true if the allocator is generating varying interpolation
    // stencils
    bool InterpolateVarying() const {
        return _interpolateVarying;
    }

    // Append a support vertex of index 'index' and weight 'weight' to the
    // Stencil 'stencil' (use findVertex() to make sure it does not exist
    // yet)
    void PushBackVertex(Stencil & stencil, int index, float weight);

    // Allocate enough memory to hold 'numStencils' Stencils
    void Resize(int numStencils);

    // Returns the number of stencil vertices that have been pushed back
    int GetNumVertices() const;

private:

    friend class Stencil;

    // returns the size of the stencil
    unsigned char * getSize(Stencil const & stencil) {
        assert(stencil.GetID()<(int)_sizes.size());
        return &_sizes[stencil.GetID()];
    }

    // returns the indices of the stencil
    int * getIndices(Stencil const & stencil) {
        if (*getSize(stencil)<_maxsize) {
            return &_indices[stencil.GetID()*_maxsize];
        } else {
            return &_bigstencils[stencil.GetID()]->indices[0];
        }
    }

    // returns the weights of the stencil
    float * getWeights(Stencil const & stencil) {
        if (*getSize(stencil)<_maxsize) {
            return &_weights[stencil.GetID()*_maxsize];
        } else {
            return &_bigstencils[stencil.GetID()]->weights[0];
        }
    }

private:

    bool _interpolateVarying;

    int _maxsize; // maximum size of a pre-allocated stencil

    StencilVec _stencils;

    std::vector<unsigned char> _sizes;    // temp stencils data (as SOA)
    std::vector<int>           _indices;
    std::vector<float>         _weights;

    // When stencils exceed _maxsize, fall back to heap allocated "BigStencils"
    struct BigStencil {

        BigStencil(int size, int const * iindices, float const * iweights) {
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
};

// Find the location of vertex 'vertex' in the stencil indices.
inline int
Stencil::findVertex(int vertex) {

    // XXXX manuelk serial serial search we can figure out something better
    unsigned char * size    = _alloc->getSize(*this);
    int           * indices = _alloc->getIndices(*this);
    for (int i=0; i<*size; ++i) {
        if (indices[i]==vertex)
            return i;
    }
    return -1;
}

// Set stencil weights to 0.0
void
Stencil::Clear() {
    for (int i=0; i<*_alloc->getSize(*this); ++i) {
        float * weights = _alloc->getWeights(*this);
        weights[i]=0.0f;
    }
}

// Weighted add of a coarse vertex
inline void
Stencil::AddWithWeight(int vertIndex, float weight) {

    if (weight==0.0f) {
        return;
    }

    int n = findVertex(vertIndex);
    if (n<0) {
        _alloc->PushBackVertex(*this, vertIndex, weight);
    } else {
        float * dstWeights = _alloc->getWeights(*this);
        dstWeights[n] += weight;
        assert(dstWeights[n]>0.0f);
    }
}

// Weighted add of a Stencil
inline void
Stencil::AddWithWeight(Stencil const & src, float weight) {

    if (weight==0.0f) {
        return;
    }

    unsigned char const * srcSize    = src._alloc->getSize(src);
    int                 * srcIndices = src._alloc->getIndices(src);
    float const         * srcWeights = src._alloc->getWeights(src);

    for (int i=0; i<*srcSize; ++i) {

        int vertIndex = srcIndices[i];

        // Attempt to locate the vertex index in the list of supporting vertices
        // of the destination stencil.
        int n = findVertex(vertIndex);
        if (n<0) {
            _alloc->PushBackVertex(*this, vertIndex, weight * srcWeights[i]);
        } else {
            float * dstWeights = _alloc->getWeights(*this);
            assert(srcWeights[i]>0.0f);
            dstWeights[n] += weight * srcWeights[i];
            assert(dstWeights[n]>0.0f);
        }
    }
}

inline void
Stencil::AddVaryingWithWeight(int vertIndex, float weight) {

    if (_alloc->InterpolateVarying()) {
        AddWithWeight(vertIndex, weight);
    }
}

inline void
Stencil::AddVaryingWithWeight(Stencil const & src, float weight) {

    if (_alloc->InterpolateVarying()) {
        AddWithWeight(src, weight);
    }
}


// Returns the current size of the Stencil
int
Stencil::GetSize() const {
    return (int)*_alloc->getSize(*this);
}

// Returns a pointer to the vertex indices of the stencil
int const *
Stencil::GetIndices() const {
    return _alloc->getIndices(*this);
}

// Returns a pointer to the vertex weights of the stencil
float const *
Stencil::GetWeights() const {
    return _alloc->getWeights(*this);
}

// Debug dump
void
Stencil::Print() const {

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

// Constructor
StencilAllocator::StencilAllocator(
    OpenSubdiv::Far::TopologyRefiner const & refiner,
        OpenSubdiv::Far::StencilTablesFactory::Mode mode) :
            _interpolateVarying(false) {

    if (mode == OpenSubdiv::Far::StencilTablesFactory::INTERPOLATE_VARYING) {
        _interpolateVarying = true;
    }

    // Make an educated guess as to what the max size should be

    OpenSubdiv::Sdc::Type type = refiner.GetSchemeType();
    switch (type) {
        case OpenSubdiv::Sdc::TYPE_BILINEAR :
            _maxsize = _interpolateVarying ? 5 : 5; break;
        case OpenSubdiv::Sdc::TYPE_CATMARK :
            _maxsize = _interpolateVarying ? 5 : 10; break;
        case OpenSubdiv::Sdc::TYPE_LOOP :
            _maxsize = _interpolateVarying ? 5 : 10; break;
        default:
            assert(0);
    }
}

// Destructor
StencilAllocator::~StencilAllocator() {

    for (BigStencilMap::iterator it=_bigstencils.begin(); it!=_bigstencils.end(); ++it) {
        delete it->second;
    }
}

// Allocate enough memory to hold 'numStencils' Stencils
void
StencilAllocator::Resize(int numStencils) {

    int currentSize = (int)_stencils.size();

    // Pre-allocate the Stencils
    _stencils.resize(numStencils);

    for (int i=currentSize; i<numStencils; ++i) {
        _stencils[i]._ID = i;
        _stencils[i]._alloc = this;
    }

    int nelems = numStencils * _maxsize;
    _sizes.clear();
    _sizes.resize(numStencils);
    _indices.resize(nelems);
    _weights.resize(nelems);

    for (BigStencilMap::iterator it=_bigstencils.begin(); it!=_bigstencils.end(); ++it) {
        delete it->second;
    }
    _bigstencils.clear();
}

// Append a support vertex of index 'index' and weight 'weight' to the
// Stencil 'stencil' (use findVertex() to make sure it does not exist
// yet)
void
StencilAllocator::PushBackVertex(Stencil & stencil, int index, float weight) {

    assert(weight>0.0f);

    unsigned char * size    = getSize(stencil);
    int           * indices = getIndices(stencil);
    float         * weights = getWeights(stencil);

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

// Returns the number of stencil vertices that have been pushed back
int
StencilAllocator::GetNumVertices() const {

    int nverts=0;
    for (int i=0; i<(int)_stencils.size(); ++i) {
        nverts+=_stencils[i].GetSize();
    }
    return nverts;
}

} // end namespace unnamed

//------------------------------------------------------------------------------

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Far {

// Copy a stencil into StencilTables
template <> void
StencilTablesFactory::copyStencil(::Stencil const & src, Stencil & dst) {

    unsigned char size = (unsigned char)src.GetSize();
    int const * indices = src.GetIndices();
    float const * weights = src.GetWeights();

    *dst._size = size;
    for (unsigned char i=0; i<size; ++i) {
        memcpy(dst._indices, indices, size*sizeof(int));
        memcpy(dst._weights, weights, size*sizeof(float));
    }
}

// (Sort &) Copy a vector of stencils into StencilTables
template <> void
StencilTablesFactory::copyStencils(::StencilVec & src,
    Stencil & dst, bool sortBySize) {

    if (sortBySize) {
        std::sort(src.begin(), src.end(), ::Stencil::CompareSize);
    }

    for (int i=0; i<(int)src.size(); ++i) {
        copyStencil(src[i], dst);
        dst.Next();
    }
}

//
// StencilTables factory
//
StencilTables const *
StencilTablesFactory::Create(TopologyRefiner const & refiner,
    Options options) {


    int maxlevel = refiner.GetMaxLevel();

    if (maxlevel==0) {
        return new StencilTables;
    }

    Mode mode = (Mode)options.interpolationMode;

    std::vector<StencilAllocator> allocators(
        options.generateAllLevels ? maxlevel : 2,
            StencilAllocator(refiner, mode));

    StencilAllocator * srcAlloc, * dstAlloc;
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
            int * srcStencils = new int[refiner.GetNumVertices(0)];
            for (int i=0; i<refiner.GetNumVertices(0); ++i) {
                srcStencils[i]=i;
            }

            ::Stencil * dstStencils = &(dstAlloc->GetStencils()).at(0);

            if (mode==INTERPOLATE_VERTEX) {
                refiner.Interpolate(level, srcStencils, dstStencils);
            } else {
                refiner.InterpolateVarying(level, srcStencils, dstStencils);
            }

            delete [] srcStencils;
        } else {

            ::Stencil * srcStencils = &(srcAlloc->GetStencils()).at(0),
                      * dstStencils = &(dstAlloc->GetStencils()).at(0);

            if (mode==INTERPOLATE_VERTEX) {
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

    StencilTables * result = new StencilTables;
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

        bool doSort = options.sortBySize!=0;

        if (options.generateAllLevels) {
            for (int level=0; level<maxlevel; ++level) {
                copyStencils(allocators[level].GetStencils(), dst, doSort);
            }
        } else {
            copyStencils(srcAlloc->GetStencils(), dst, doSort);
        }

        if (options.generateOffsets) {
            for (int i=0, ofs=0; i<nstencils; ++i ) {
                result->_offsets[i]=ofs;
                assert(result->_sizes[i]!=0 and
                    result->_sizes[i]<(int)result->_weights.size());
                ofs+=result->_sizes[i];
            }
        }
    }
    return result;
}

KernelBatch
StencilTablesFactory::Create(StencilTables const &stencilTables) {

    return KernelBatch( KernelBatch::KERNEL_STENCIL_TABLE,
        -1, 0, stencilTables.GetNumStencils());
}

} // end namespace Far

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv
