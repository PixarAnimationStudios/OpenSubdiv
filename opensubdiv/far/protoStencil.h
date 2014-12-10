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

#ifndef FAR_PROTOSTENCIL_H
#define FAR_PROTOSTENCIL_H

#include "../far/stencilTables.h"

#include <cstring>
#include <map>
#include <vector>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Far {

//
// Proto-stencil Pool Allocator classes
//
// Strategy: allocate up-front a data pool for supporting PROTOSTENCILS of a size
// (maxsize) slightly above average. For the (rare) BIG_PROTOSTENCILS that
// require more support vertices, switch to (slow) heap allocation.
//
template <typename PROTOSTENCIL, class BIG_PROTOSTENCIL>
class Allocator {

public:

    // Constructor
    Allocator(int maxSize, bool interpolateVarying=false) :
        _maxsize(maxSize), _interpolateVarying(interpolateVarying) { }

	~Allocator() {
		clearBigStencils();
	}

    // Returns the number of stencils in the allocator
    int GetNumStencils() const {
        return (int)_sizes.size();
    }

    // Returns the total number of control vertices used by the all the stencils
    int GetNumVerticesTotal() const {
        int nverts=0;
        for (int i=0; i<GetNumStencils(); ++i) {
            nverts += _sizes[i];
        }
        return nverts;
    }

    // Returns true if the pool allocator executes AddVaryingWithWeight
    // factorization
    bool GetInterpolateVarying() const {
        return _interpolateVarying;
    }

    // Allocates storage for 'size' stencils with a fixed '_maxsize' supporting
    // basis of control-vertices
    void Resize(int numStencils) {
        clearBigStencils();
        int nelems = numStencils * _maxsize;
        _sizes.clear();
        _sizes.resize(numStencils);
        _indices.resize(nelems);
        _weights.resize(nelems);
    }

    // Adds the contribution of a supporting vertex that was not yet
    // in the stencil
    void PushBackVertex(Index protoStencil, Index vert, float weight) {
        assert(weight!=0.0f);
        unsigned char & size = _sizes[protoStencil];
        Index idx = protoStencil*_maxsize;
        if (size < (_maxsize-1)) {
            idx += size;
            _indices[idx] = vert;
            _weights[idx] = weight;
        } else {
            BIG_PROTOSTENCIL * dst = 0;
            if (size==(_maxsize-1)) {
                dst = new BIG_PROTOSTENCIL(size, &_indices[idx], &_weights[idx]);
                assert(_bigStencils.find(protoStencil)==_bigStencils.end());
                _bigStencils[protoStencil] = dst;
            } else {
                assert(_bigStencils.find(protoStencil)!=_bigStencils.end());
                dst = _bigStencils[protoStencil];
            }
            dst->_indices.push_back(vert);
            dst->_weights.push_back(weight);
        }
        ++size;
    }

    // Returns the local index in 'stencil' of a given vertex index, or
    // INDEX_INVALID if the stencil does not contain this vertex
    int FindVertex(Index protoStencil, Index vert) {
        int size = _sizes[protoStencil];
        Index const * indices = GetIndices(protoStencil);
        for (int i=0; i<size; ++i) {
            if (indices[i]==vert) {
                return i;
            }
        }
        return Vtr::INDEX_INVALID;
    }

    // Returns true of the stencil does not fit in the pool allocator and
    // has been moved to the 'big' (slow) allocation pool
    bool IsBigStencil(Index protoStencil) const {
        assert(protoStencil<(int)_sizes.size());
        return _sizes[protoStencil]>=_maxsize;
    }

    // Returns the size of a given proto-stencil
    unsigned char GetSize(Index protoStencil) const {
        assert(protoStencil<(int)_sizes.size());
        return _sizes[protoStencil];
    }

    // Resolve memory pool and return a pointer to the indices of a given
    // proto-stencil
    Index * GetIndices(Index protoStencil) {
        if (not IsBigStencil(protoStencil)) {
            return &_indices[protoStencil*_maxsize];
        } else {
            assert(_bigStencils.find(protoStencil)!=_bigStencils.end());
            return &_bigStencils[protoStencil]->_indices[0];
        }
    }

    // Resolve memory pool and return a pointer to the weights of a given
    // proto-stencil
    float * GetWeights(Index protoStencil) {
        if (not IsBigStencil(protoStencil)) {
            return &_weights[protoStencil*_maxsize];
        } else {
            assert(_bigStencils.find(protoStencil)!=_bigStencils.end());
            return &_bigStencils[protoStencil]->_weights[0];
        }
    }

    // Returns the proto-stencil at a given index
    PROTOSTENCIL operator[] (Index protoStencil) {
        // If the allocator is empty, AddWithWeight() expects a coarse control
        // vertex instead of a stencil and we only need to pass the index
        return PROTOSTENCIL(protoStencil, this->GetNumStencils()>0 ? this : 0);
    }

    // Returns the proto-stencil at a given index
    PROTOSTENCIL operator[] (Index protoStencil) const {
        // If the allocator is empty, AddWithWeight() expects a coarse control
        // vertex instead of a stencil and we only need to pass the index
        return PROTOSTENCIL(protoStencil, this->GetNumStencils()>0 ?
            const_cast<Allocator<PROTOSTENCIL, BIG_PROTOSTENCIL> *>(this) : 0);
    }

    // Copy the proto-stencil out of the pool
    unsigned char CopyStencil(Index protoStencil,
        Index * indices, float * weights) {
        unsigned char size = GetSize(protoStencil);
        memcpy(indices, this->GetIndices(protoStencil), size*sizeof(Index));
        memcpy(weights, this->GetWeights(protoStencil), size*sizeof(float));
        return size;
    }

protected:

    // delete 'slow' memory pool
    void clearBigStencils() {
        typename BigStencilMap::iterator it;
        for (it=_bigStencils.begin(); it!=_bigStencils.end(); ++it) {
            delete it->second;
        }
        _bigStencils.clear();
    }

protected:

    int _maxsize; // max size of stencil that fits in the 'fast' pool

    bool _interpolateVarying;             // true for varying interpolation

    std::vector<unsigned char> _sizes;    // 'fast' memory pool
    std::vector<int>           _indices;
    std::vector<float>         _weights;

    typedef std::map<int, BIG_PROTOSTENCIL *> BigStencilMap;
    BigStencilMap _bigStencils;           // 'slow' memory pool
};

//
// Specialization of the Allocator for stencils with tangents that require
// additional derivative weights.
//
template <typename PROTOSTENCIL, class BIG_PROTOSTENCIL>
class LimitAllocator : public Allocator<PROTOSTENCIL, BIG_PROTOSTENCIL> {

public:

    // Constructor
    LimitAllocator(int maxSize) :
        Allocator<PROTOSTENCIL, BIG_PROTOSTENCIL>(maxSize) { }

    void Resize(int size) {
        Allocator<PROTOSTENCIL, BIG_PROTOSTENCIL>::Resize(size);
        int nelems = (int)this->_weights.size();
        _tan1Weights.resize(nelems);
        _tan2Weights.resize(nelems);
    }

    void PushBackVertex(Index protoStencil,
        Index vert, float weight, float tan1Weight, float tan2Weight) {
        assert(weight!=0.0f or tan1Weight!=0.0f or tan2Weight!=0.0f);
        unsigned char & size = this->_sizes[protoStencil];
        Index idx = protoStencil*this->_maxsize;
        if (size < (this->_maxsize-1)) {
            idx += size;
            this->_indices[idx] = vert;
            this->_weights[idx] = weight;
            this->_tan1Weights[idx] = tan1Weight;
            this->_tan2Weights[idx] = tan2Weight;
        } else {
            BIG_PROTOSTENCIL * dst = 0;
            if (size==(this->_maxsize-1)) {
                dst = new BIG_PROTOSTENCIL(size,
                    &this->_indices[idx], &this->_weights[idx],
                        &this->_tan1Weights[idx], &this->_tan2Weights[idx]);
                assert(this->_bigStencils.find(protoStencil)==this->_bigStencils.end());
                this->_bigStencils[protoStencil] = dst;
            } else {
                assert(this->_bigStencils.find(protoStencil)!=this->_bigStencils.end());
                dst = this->_bigStencils[protoStencil];
            }
            dst->_indices.push_back(vert);
            dst->_weights.push_back(weight);
            dst->_tan1Weights.push_back(tan1Weight);
            dst->_tan2Weights.push_back(tan2Weight);
        }
        ++size;
    }

    float * GetTan1Weights(Index protoStencil) {
        if (not this->IsBigStencil(protoStencil)) {
            return &_tan1Weights[protoStencil*this->_maxsize];
        } else {
            assert(this->_bigStencils.find(protoStencil)!=this->_bigStencils.end());
            return &this->_bigStencils[protoStencil]->_tan1Weights[0];
        }
    }

    float * GetTan2Weights(Index protoStencil) {
        if (not this->IsBigStencil(protoStencil)) {
            return &_tan2Weights[protoStencil*this->_maxsize];
        } else {
            assert(this->_bigStencils.find(protoStencil)!=this->_bigStencils.end());
            return &this->_bigStencils[protoStencil]->_tan2Weights[0];
        }
    }

    PROTOSTENCIL operator[] (Index protoStencil) {
        assert(this->GetNumStencils()>0);
        return PROTOSTENCIL(protoStencil, this);
    }

    void ClearStencil(Index protoStencil) {
        Allocator<PROTOSTENCIL, BIG_PROTOSTENCIL>::ClearStencil(protoStencil);
        memset(GetTan1Weights(protoStencil), 0, this->_sizes[protoStencil]*sizeof(float));
        memset(GetTan2Weights(protoStencil), 0, this->_sizes[protoStencil]*sizeof(float));
    }

    unsigned char CopyLimitStencil(Index protoStencil,
        Index * indices, float * weights, float * tan1Weights, float * tan2Weights) {
        unsigned char size =
            Allocator<PROTOSTENCIL, BIG_PROTOSTENCIL>::CopyStencil(
                protoStencil, indices, weights);
        memcpy(tan1Weights, this->GetTan1Weights(protoStencil), size*sizeof(Index));
        memcpy(tan2Weights, this->GetTan2Weights(protoStencil), size*sizeof(float));
        return size;
    }

private:
    std::vector<float> _tan1Weights,
                       _tan2Weights;
};

//
// 'Big' Proto stencil classes
//
// When proto-stencils exceed _maxsize, fall back to dynamically allocated
// "BigStencils" (with 'Limit' specialization to handle tangents)
//
struct BigStencil {

    BigStencil(unsigned char size, Index const * indices,
        float const * weights) {
        _indices.reserve(size+5); _indices.resize(size);
        memcpy(&_indices.at(0), indices, size*sizeof(int));
        _weights.reserve(size+5); _weights.resize(size);
        memcpy(&_weights.at(0), weights, size*sizeof(float));
    }

    std::vector<Index> _indices;
    std::vector<float> _weights;
};
struct BigLimitStencil : public BigStencil {

    BigLimitStencil(unsigned char size,
        Index const * indices, float const * weights,
            float const * tan1Weights,  float const * tan2Weights) :
                BigStencil(size, indices, weights) {

        _tan1Weights.reserve(size+5); _tan1Weights.resize(size);
        memcpy(&_tan1Weights.at(0), tan1Weights, size*sizeof(float));
        _tan2Weights.reserve(size+5); _tan2Weights.resize(size);
        memcpy(&_tan2Weights.at(0), tan2Weights, size*sizeof(float));
    }

    std::vector<float> _tan1Weights,
                       _tan2Weights;
};

//
// ProtoStencils
//
// Proto-stencils are used to interpolate stencils from supporting vertices.
// These stencils are backed by a pool allocator to allow for fast push-back
// of contributing control-vertices weights & indices as they are discovered.
//
class ProtoStencil {

public:

    ProtoStencil(Index id, Allocator<ProtoStencil, BigStencil> * alloc) :
        _id(id), _alloc(alloc) { }

    void Clear() {
        // Clear() can only ever be called on an empty stencil: nothing to do
        assert(_alloc->GetSize(_id)==0);
    }

    // Factorize from a proto-stencil allocator
    void AddWithWeight(ProtoStencil const & src, float weight) {

        if(weight==0.0f) {
            return;
        }

        if (src._alloc) {
            // Stencil contribution
            unsigned char srcSize = src._alloc->GetSize(src._id);
            Index const * srcIndices = src._alloc->GetIndices(src._id);
            float const * srcWeights = src._alloc->GetWeights(src._id);

            addWithWeight(weight, srcSize, srcIndices, srcWeights);
        } else {
            // Coarse vertex contribution
            Index n = _alloc->FindVertex(_id, src._id);
            if (Vtr::IndexIsValid(n)) {
                _alloc->GetWeights(_id)[n] += weight;
                assert(_alloc->GetWeights(_id)[n]>0.0f);
            } else {
                _alloc->PushBackVertex(_id, src._id, weight);
            }
        }
    }

    // Factorize from a finished stencil table
    void AddWithWeight(StencilTables const & table, Index idx, float weight) {

        assert(idx<table.GetNumStencils());

        if(weight==0.0f) {
            return;
        }

        unsigned char srcSize = table.GetSizes()[idx];
        Index offset = table.GetOffsets()[idx];
        Index const * srcIndices = &table.GetControlIndices()[offset];
        float const * srcWeights = &table.GetWeights()[offset];

        addWithWeight(weight, srcSize, srcIndices, srcWeights);
    }

    void AddVaryingWithWeight(ProtoStencil const & src, float weight) {
        if (_alloc->GetInterpolateVarying()) {
            AddWithWeight(src, weight);
        }
    }

protected:

    friend class ProtoLimitStencil;

    void addWithWeight(float weight, unsigned char srcSize,
        Index const * srcIndices, float const * srcWeights) {

        for (unsigned char i=0; i<srcSize; ++i) {

            assert(srcWeights[i]!=0.0f);

            float w = weight * srcWeights[i];
            if (w==0.0f) {
                continue;
            }

            Index vertIndex = srcIndices[i],
                  n = _alloc->FindVertex(_id, vertIndex);
            if (Vtr::IndexIsValid(n)) {
                _alloc->GetWeights(_id)[n] += w;
                assert(_alloc->GetWeights(_id)[n]!=0.0f);
            } else {
                _alloc->PushBackVertex(_id, vertIndex, w);
            }
        }
    }

    Index _id;
    Allocator<ProtoStencil, BigStencil> * _alloc;
};

typedef Allocator<ProtoStencil, BigStencil> StencilAllocator;


//
// ProtoLimitStencil
//
class ProtoLimitStencil {

public:

    ProtoLimitStencil(Index id,
        LimitAllocator<ProtoLimitStencil, BigLimitStencil> * alloc) :
            _id(id), _alloc(alloc) { }

    void Clear() {
        // Clear() can only ever be called on an empty stencil: nothing to do
        assert(_alloc->GetSize(_id)==0);
    }

    void AddWithWeight(Stencil const & src,
        float weight, float tan1Weight, float tan2Weight) {

        if(weight==0.0f and tan1Weight==0.0f and tan2Weight==0.0f) {
            return;
        }

        unsigned char srcSize = *src.GetSizePtr();
        Index const * srcIndices = src.GetVertexIndices();
        float const * srcWeights = src.GetWeights();

        for (unsigned char i=0; i<srcSize; ++i) {

            float w = srcWeights[i];
            if (w==0.0f) {
                continue;
            }

            Index vertIndex = srcIndices[i],
                  n = _alloc->FindVertex(_id, vertIndex);
            if (Vtr::IndexIsValid(n)) {
                _alloc->GetWeights(_id)[n] += weight*w;
                _alloc->GetTan1Weights(_id)[n] += tan1Weight*w;
                _alloc->GetTan2Weights(_id)[n] += tan2Weight*w;

            } else {
                _alloc->PushBackVertex(_id, vertIndex,
                    weight*w, tan1Weight*w, tan2Weight*w);
            }
        }
    }

private:
    Index _id;
    LimitAllocator<ProtoLimitStencil, BigLimitStencil> * _alloc;
};

typedef LimitAllocator<ProtoLimitStencil, BigLimitStencil> LimitStencilAllocator;



} // end namespace Far

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv

#endif // FAR_PROTOSTENCIL_H
