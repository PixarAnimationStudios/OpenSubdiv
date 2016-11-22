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

#include "../far/stencilBuilder.h"
#include "../far/topologyRefiner.h"
 
namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Far {
namespace internal {

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

template<class FD>
struct Point1stDerivWeight {
    FD p;
    FD du;
    FD dv;

    Point1stDerivWeight()
        : p(0.0f), du(0.0f), dv(0.0f)
    { }
    Point1stDerivWeight(FD w)
        : p(w), du(w), dv(w)
    { }
    Point1stDerivWeight(FD w, FD wDu, FD wDv)
        : p(w), du(wDu), dv(wDv)
    { }

    friend Point1stDerivWeight operator*(Point1stDerivWeight lhs,
                                         Point1stDerivWeight const& rhs) {
        lhs.p *= rhs.p;
        lhs.du *= rhs.du;
        lhs.dv *= rhs.dv;
        return lhs;
    }
    Point1stDerivWeight& operator+=(Point1stDerivWeight const& rhs) {
        p += rhs.p;
        du += rhs.du;
        dv += rhs.dv;
        return *this;
    }
};

template<class FD>
struct Point2ndDerivWeight {
    FD p;
    FD du;
    FD dv;
    FD duu;
    FD duv;
    FD dvv;

    Point2ndDerivWeight()
        : p((FD)0.0), du((FD)0.0), dv((FD)0.0), duu((FD)0.0), duv((FD)0.0), dvv((FD)0.0)
    { }
    Point2ndDerivWeight(FD w)
        : p(w), du(w), dv(w), duu(w), duv(w), dvv(w)
    { }
    Point2ndDerivWeight(FD w, FD wDu, FD wDv,
                        FD wDuu, FD wDuv, FD wDvv)
        : p(w), du(wDu), dv(wDv), duu(wDuu), duv(wDuv), dvv(wDvv)
    { }

    friend Point2ndDerivWeight operator*(Point2ndDerivWeight lhs,
                                         Point2ndDerivWeight const& rhs) {
        lhs.p *= rhs.p;
        lhs.du *= rhs.du;
        lhs.dv *= rhs.dv;
        lhs.duu *= rhs.duu;
        lhs.duv *= rhs.duv;
        lhs.dvv *= rhs.dvv;
        return lhs;
    }
    Point2ndDerivWeight& operator+=(Point2ndDerivWeight const& rhs) {
        p += rhs.p;
        du += rhs.du;
        dv += rhs.dv;
        duu += rhs.duu;
        duv += rhs.duv;
        dvv += rhs.dvv;
        return *this;
    }
};

/// Stencil table constructor set.
///
template<class FD>
class WeightTable {
public:
    WeightTable(int coarseVerts,
                bool genCtrlVertStencils,
                bool compactWeights)
        : _size(0)
        , _lastOffset(0)
        , _coarseVertCount(coarseVerts)
        , _compactWeights(compactWeights)
    {
        // These numbers were chosen by profiling production assets at uniform
        // level 3.
        size_t n = std::max(coarseVerts,
                      std::min(int(5*1024*1024),
                          coarseVerts*2));
        _dests.reserve(n);
        _sources.reserve(n);
        _weights.reserve(n);
        
        if (!genCtrlVertStencils)
            return;

        // Generate trivial control vert stencils
        _sources.resize(coarseVerts);
        _weights.resize(coarseVerts);
        _dests.resize(coarseVerts);
        _indices.resize(coarseVerts);
        _sizes.resize(coarseVerts);

        for (int i = 0; i < coarseVerts; i++) {
            _indices[i] = i;
            _sizes[i] = 1;
            _dests[i] = i;
            _sources[i] = i;
            _weights[i] = 1.0;
        }

        _size = static_cast<int>(_sources.size());
        _lastOffset = _size - 1;
    }

    template <class W, class WACCUM>
    void AddWithWeight(int src, int dest, W weight, WACCUM weights)
    {
        // Factorized stencils are expressed purely in terms of the control
        // mesh verts. Without this flattening, level_i's weights would point
        // to level_i-1, which would point to level_i-2, until the final level
        // points to the control verts.
        //
        // So here, we check if the incoming vert (src) is in the control mesh,
        // if it is, we can simply merge it without attempting to resolve it
        // first.
        if (src < _coarseVertCount) {
            merge(src, dest, weight, W(1.0), _lastOffset, _size, weights);
            return;
        }

        // src is not in the control mesh, so resolve all contributing coarse
        // verts (src itself is made up of many control vert weights). 
        //
        // Find the src stencil and number of contributing CVs.
        int len = _sizes[src];
        int start = _indices[src];

        for (int i = start; i < start+len; i++) {
            // Invariant: by processing each level in order and each vertex in
            // dependent order, any src stencil vertex reference is guaranteed
            // to consist only of coarse verts: therefore resolving src verts
            // must yield verts in the coarse mesh.
            assert(_sources[i] < _coarseVertCount);

            // Merge each of src's contributing verts into this stencil.
            merge(_sources[i], dest, weights.Get(i), weight,
                                _lastOffset, _size, weights);
        }
    }

    class Point1stDerivAccumulator {
        WeightTable* _tbl;
    public:
        Point1stDerivAccumulator(WeightTable* tbl) : _tbl(tbl)
        { }
        void PushBack(Point1stDerivWeight<FD> weight) {
            _tbl->_weights.push_back(weight.p);
            _tbl->_duWeights.push_back(weight.du);
            _tbl->_dvWeights.push_back(weight.dv);
        }
        void Add(size_t i, Point1stDerivWeight<FD> weight) {
            _tbl->_weights[i] += weight.p;
            _tbl->_duWeights[i] += weight.du;
            _tbl->_dvWeights[i] += weight.dv;
        }
        Point1stDerivWeight<FD> Get(size_t index) {
            return Point1stDerivWeight<FD>(_tbl->_weights[index],
                                       _tbl->_duWeights[index],
                                       _tbl->_dvWeights[index]);
        }
    };
    Point1stDerivAccumulator GetPoint1stDerivAccumulator() {
        return Point1stDerivAccumulator(this);
    };

    class Point2ndDerivAccumulator {
        WeightTable* _tbl;
    public:
        Point2ndDerivAccumulator(WeightTable* tbl) : _tbl(tbl)
        { }
        void PushBack(Point2ndDerivWeight<FD> weight) {
            _tbl->_weights.push_back(weight.p);
            _tbl->_duWeights.push_back(weight.du);
            _tbl->_dvWeights.push_back(weight.dv);
            _tbl->_duuWeights.push_back(weight.duu);
            _tbl->_duvWeights.push_back(weight.duv);
            _tbl->_dvvWeights.push_back(weight.dvv);
        }
        void Add(size_t i, Point2ndDerivWeight<FD> weight) {
            _tbl->_weights[i] += weight.p;
            _tbl->_duWeights[i] += weight.du;
            _tbl->_dvWeights[i] += weight.dv;
            _tbl->_duuWeights[i] += weight.duu;
            _tbl->_duvWeights[i] += weight.duv;
            _tbl->_dvvWeights[i] += weight.dvv;
        }
        Point2ndDerivWeight<FD> Get(size_t index) {
            return Point2ndDerivWeight<FD>(_tbl->_weights[index],
                                       _tbl->_duWeights[index],
                                       _tbl->_dvWeights[index],
                                       _tbl->_duuWeights[index],
                                       _tbl->_duvWeights[index],
                                       _tbl->_dvvWeights[index]);
        }
    };
    Point2ndDerivAccumulator GetPoint2ndDerivAccumulator() {
        return Point2ndDerivAccumulator(this);
    };

    class ScalarAccumulator {
        WeightTable* _tbl;
    public:
        ScalarAccumulator(WeightTable* tbl) : _tbl(tbl)
        { }
        void PushBack(FD weight) {
            _tbl->_weights.push_back(weight);
        }
        void Add(size_t i, FD w) {
            _tbl->_weights[i] += w;
        }
        FD Get(size_t index) {
            return _tbl->_weights[index];
        }
    };
    ScalarAccumulator GetScalarAccumulator() {
        return ScalarAccumulator(this);
    };

    std::vector<int> const&
    GetOffsets() const { return _indices; }

    std::vector<int> const& 
    GetSizes() const { return _sizes; }

    std::vector<int> const&
    GetSources() const { return _sources; }

    std::vector<FD> const&
    GetWeights() const { return _weights; }

    std::vector<FD> const&
    GetDuWeights() const { return _duWeights; }

    std::vector<FD> const&
    GetDvWeights() const { return _dvWeights; }

    std::vector<FD> const&
    GetDuuWeights() const { return _duuWeights; }

    std::vector<FD> const&
    GetDuvWeights() const { return _duvWeights; }

    std::vector<FD> const&
    GetDvvWeights() const { return _dvvWeights; }

    void SetCoarseVertCount(int numVerts) {
        _coarseVertCount = numVerts;
    }
private:

    // Merge a vertex weight into the stencil table, if there is an existing
    // weight for a given source vertex it will be combined.
    //
    // PERFORMANCE: caution, this function is super hot.
    template <class W, class WACCUM>
    void merge(int src, int dst, W weight,
               // Delaying weight*factor multiplication hides memory latency of
               // accessing weight[i], yielding more stable performance.
               W weightFactor, 
               // Similarly, passing offset & tableSize as params yields higher
               // performance than accessing the class members directly.
               int lastOffset, int tableSize, WACCUM weights)
    {
        // The lastOffset is the vertex we're currently processing, by
        // leveraging this we need not lookup the dest stencil size or offset.
        //
        // Additionally, if the client does not want the resulting verts
        // compacted, do not attempt to combine weights.
        if (_compactWeights && !_dests.empty() && _dests[lastOffset] == dst) {

            // tableSize is exactly _sources.size(), but using tableSize is
            // significantly faster.
            for (int i = lastOffset; i < tableSize; i++) {

                // If we find an existing vertex that matches src, we need to
                // combine the weights to avoid duplicate entries for src.
                if (_sources[i] == src) {
                    weights.Add(i, weight*weightFactor);
                    return;
                }
            }
        }

        // We haven't seen src yet, insert it as a new vertex weight.
        add(src, dst, weight*weightFactor, weights);
    }

    // Add a new vertex weight to the stencil table.
    template <class W, class WACCUM>
    void add(int src, int dst, W weight, WACCUM weights)
    {
        // The _dests array has num(weights) elements mapping each individual
        // element back to a specific stencil. The array is constructed in such
        // a way that the current stencil being built is always at the end of
        // the array, so if the dests array is empty or back() doesn't match
        // dst, then we just started building a new stencil.
        if (_dests.empty() || dst != _dests.back()) {
            // _indices and _sizes always have num(stencils) elements so that
            // stencils can be directly looked up by their index in these
            // arrays. So here, ensure that they are large enough to hold the
            // new stencil about to be built.
            if (dst+1 > (int)_indices.size()) {
                _indices.resize(dst+1);
                _sizes.resize(dst+1);
            }
            // Initialize the new stencil's meta-data (offset, size).
            _indices[dst] = static_cast<int>(_sources.size());
            _sizes[dst] = 0;
            // Keep track of where the current stencil begins, which lets us
            // avoid having to look it up later.
            _lastOffset = static_cast<int>(_sources.size());
        }
        // Cache the number of elements as an optimization, it's faster than
        // calling size() on any of the vectors.
        _size++;

        // Increment the current stencil element size.
        _sizes[dst]++;
        // Track this element as belonging to the stencil "dst".
        _dests.push_back(dst);

        // Store the actual stencil data.
        _sources.push_back(src);
        weights.PushBack(weight);
    }

    // The following vectors are explicitly stored as non-interleaved elements
    // to reduce cache misses.

    // Stencil to destination vertex map.
    std::vector<int> _dests;

    // The actual stencil data.
    std::vector<int> _sources;
    std::vector<FD> _weights;
    std::vector<FD> _duWeights;
    std::vector<FD> _dvWeights;
    std::vector<FD> _duuWeights;
    std::vector<FD> _duvWeights;
    std::vector<FD> _dvvWeights;

    // Index data used to recover stencil-to-vertex mapping.
    std::vector<int> _indices;
    std::vector<int> _sizes;

    // Acceleration members to avoid pointer chasing and reverse loops.
    int _size;
    int _lastOffset;
    int _coarseVertCount;
    bool _compactWeights;
};

template<class FD>
StencilBuilderG<FD>::StencilBuilderG(int coarseVertCount,
                                    bool genCtrlVertStencils,
                                    bool compactWeights)
        : _weightTable(new WeightTable<FD>(coarseVertCount,
                                   genCtrlVertStencils,
                                   compactWeights))
{
}

template<class FD>
StencilBuilderG<FD>::~StencilBuilderG()
{
    delete _weightTable;
}

template<class FD>
size_t
StencilBuilderG<FD>::GetNumVerticesTotal() const
{
    return _weightTable->GetWeights().size();
}


template<class FD>
int 
StencilBuilderG<FD>::GetNumVertsInStencil(size_t stencilIndex) const
{
    if (stencilIndex > _weightTable->GetSizes().size() - 1)
        return 0;

    return (int)_weightTable->GetSizes()[stencilIndex];
}

template<class FD>
void
StencilBuilderG<FD>::SetCoarseVertCount(int numVerts)
{
    _weightTable->SetCoarseVertCount(numVerts);
}

template<class FD>
std::vector<int> const&
StencilBuilderG<FD>::GetStencilOffsets() const {
    return _weightTable->GetOffsets();
}

template<class FD>
std::vector<int> const& 
StencilBuilderG<FD>::GetStencilSizes() const {
    return _weightTable->GetSizes();
}

template<class FD>
std::vector<int> const&
StencilBuilderG<FD>::GetStencilSources() const {
    return _weightTable->GetSources();
}

template<class FD>
std::vector<FD> const&
StencilBuilderG<FD>::GetStencilWeights() const {
    return _weightTable->GetWeights();
}

template<class FD>
std::vector<FD> const&
StencilBuilderG<FD>::GetStencilDuWeights() const {
    return _weightTable->GetDuWeights();
}

template<class FD>
std::vector<FD> const&
StencilBuilderG<FD>::GetStencilDvWeights() const {
    return _weightTable->GetDvWeights();
}

template<class FD>
std::vector<FD> const&
StencilBuilderG<FD>::GetStencilDuuWeights() const {
    return _weightTable->GetDuuWeights();
}

template<class FD>
std::vector<FD> const&
StencilBuilderG<FD>::GetStencilDuvWeights() const {
    return _weightTable->GetDuvWeights();
}

template<class FD>
std::vector<FD> const&
StencilBuilderG<FD>::GetStencilDvvWeights() const {
    return _weightTable->GetDvvWeights();
}

template<class FD>
void
StencilBuilderG<FD>::Index::AddWithWeight(Index const & src, FD weight)
{
    // Ignore no-op weights.
    if (isWeightZero(weight)) {
        return;
    }
    _owner->_weightTable->AddWithWeight(src._index, _index, weight,
                                _owner->_weightTable->GetScalarAccumulator());
}

template<class FD>
void
StencilBuilderG<FD>::Index::AddWithWeight(StencilG<FD> const& src, FD weight)
{
    if (isWeightZero(weight)) {
        return;
    }

    int srcSize = *src.GetSizePtr();
    Vtr::Index const * srcIndices = src.GetVertexIndices();
    FD const * srcWeights = src.GetWeights();

    for (int i = 0; i < srcSize; ++i) {
        FD w = srcWeights[i];
        if (isWeightZero(w)) {
            continue;
        }

        Vtr::Index srcIndex = srcIndices[i];

        FD wgt = weight * w;
        _owner->_weightTable->AddWithWeight(srcIndex, _index, wgt,
                            _owner->_weightTable->GetScalarAccumulator());
    }  
}

template<class FD>
void
StencilBuilderG<FD>::Index::AddWithWeight(StencilG<FD> const& src,
    FD weight, FD du, FD dv)
{
    if (isWeightZero(weight) && isWeightZero(du) && isWeightZero(dv)) {
        return;
    }

    int srcSize = *src.GetSizePtr();
    Vtr::Index const * srcIndices = src.GetVertexIndices();
    FD const * srcWeights = src.GetWeights();

    for (int i = 0; i < srcSize; ++i) {
        FD w = srcWeights[i];
        if (isWeightZero(w)) {
            continue;
        }

        Vtr::Index srcIndex = srcIndices[i];

        Point1stDerivWeight<FD> wgt = Point1stDerivWeight<FD>(weight, du, dv) * w;
        _owner->_weightTable->AddWithWeight(srcIndex, _index, wgt,
                           _owner->_weightTable->GetPoint1stDerivAccumulator());
    }
}

template<class FD>
void
StencilBuilderG<FD>::Index::AddWithWeight(StencilG<FD> const& src,
    FD weight, FD du, FD dv, FD duu, FD duv, FD dvv)
{
    if (isWeightZero(weight) && isWeightZero(du) && isWeightZero(dv) &&
        isWeightZero(duu) && isWeightZero(duv) && isWeightZero(dvv)) {
        return;
    }

    int srcSize = *src.GetSizePtr();
    Vtr::Index const * srcIndices = src.GetVertexIndices();
    FD const * srcWeights = src.GetWeights();

    for (int i = 0; i < srcSize; ++i) {
        FD w = srcWeights[i];
        if (isWeightZero(w)) {
            continue;
        }

        Vtr::Index srcIndex = srcIndices[i];

        Point2ndDerivWeight<FD> wgt = Point2ndDerivWeight<FD>(weight, du, dv, duu, duv, dvv) * w;
        _owner->_weightTable->AddWithWeight(srcIndex, _index, wgt,
                           _owner->_weightTable->GetPoint2ndDerivAccumulator());
    }
}

template class StencilBuilderG<float>;
template class StencilBuilderG<double>;

} // end namespace internal
} // end namespace Far
} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv

