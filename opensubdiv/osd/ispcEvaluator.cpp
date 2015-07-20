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

#include "ispcEvaluator.h"
#include "cpuKernel.h"
#include "../far/patchBasis.h"
#include "ispcEvalLimitKernel.isph"

#include <tbb/parallel_for.h>
#include <cstdlib>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

#define grain_size  512

/* static */
bool
IspcEvaluator::EvalStencils(const float *src, BufferDescriptor const &srcDesc,
                           float *dst,       BufferDescriptor const &dstDesc,
                           const int * sizes,
                           const int * offsets,
                           const int * indices,
                           const float * weights,
                           int start, int end) {

    if (end <= start) return true;
    if (srcDesc.length != dstDesc.length) return false;

    // XXX: we can probably expand cpuKernel.cpp to here.
    CpuEvalStencils(src, srcDesc, dst, dstDesc,
                    sizes, offsets, indices, weights, start, end);

    return true;
}

/* static */
bool
IspcEvaluator::EvalStencils(const float *src, BufferDescriptor const &srcDesc,
                           float *dst,       BufferDescriptor const &dstDesc,
                           float *du,        BufferDescriptor const &duDesc,
                           float *dv,        BufferDescriptor const &dvDesc,
                           const int * sizes,
                           const int * offsets,
                           const int * indices,
                           const float * weights,
                           const float * duWeights,
                           const float * dvWeights,
                           int start, int end) {
    if (end <= start) return true;
    if (srcDesc.length != dstDesc.length) return false;
    if (srcDesc.length != duDesc.length) return false;
    if (srcDesc.length != dvDesc.length) return false;

    CpuEvalStencils(src, srcDesc,
                    dst, dstDesc,
                    du,  duDesc,
                    dv,  dvDesc,
                    sizes, offsets, indices,
                    weights, duWeights, dvWeights,
                    start, end);

    return true;
}

template <typename T>
struct BufferAdapter {
    BufferAdapter(T *p, int length, int stride) :
        _p(p), _length(length), _stride(stride) { }
    void Clear() {
        for (int i = 0; i < _length; ++i) _p[i] = 0;
    }
    void AddWithWeight(T const *src, float w) {
        if (_p) {
            for (int i = 0; i < _length; ++i) {
                _p[i] += src[i] * w;
            }
        }
    }
    const T *operator[] (int index) const {
        return _p + _stride * index;
    }
    BufferAdapter<T> & operator ++() {
        if (_p) {
            _p += _stride;
        }
        return *this;
    }

    T *_p;
    int _length;
    int _stride;
};

/* static */
bool
IspcEvaluator::EvalPatches(const float *src, BufferDescriptor const &srcDesc,
                           float *dst,       BufferDescriptor const &dstDesc,
                           int numPatchCoords,
                           const PatchCoord *patchCoords,
                           const PatchArray *patchArrays,
                           const int *patchIndexBuffer,
                           const PatchParam *patchParamBuffer) { 
    if (srcDesc.length != dstDesc.length) return false;
        
    // Copy BufferDescriptor to ispc version
    // Since memory alignment in ISPC may be different from C++,
    // we use the assignment for each field instead of the assignment for 
    // the whole struct
    ispc::BufferDescriptor ispcSrcDesc;
    ispcSrcDesc.offset = srcDesc.offset;
    ispcSrcDesc.length = srcDesc.length;
    ispcSrcDesc.stride = srcDesc.stride;                                           
                          
    tbb::blocked_range<int> range = tbb::blocked_range<int>(0, numPatchCoords, grain_size);
    tbb::parallel_for(range, [&](const tbb::blocked_range<int> &r)
    {    
    uint i = r.begin();
        
    ispc::BufferDescriptor ispcDstDesc, ispcDuDesc, ispcDvDesc;                               
    ispcDstDesc.offset = dstDesc.offset + dstDesc.offset + i * dstDesc.stride;
    ispcDstDesc.length = dstDesc.length;
    ispcDstDesc.stride = dstDesc.stride;
    
    while (i < r.end()) {
        // the patch coordinates are sorted by patch handle
        // the following code searches the coordinates that
        // belongs to the same patch so that they can be evalauated 
        // with ISPC
        int nCoord = 1;
        Far::PatchTable::PatchHandle handle = patchCoords[i].handle;
        while(i + nCoord < r.end() && 
              handle.isEqual(patchCoords[i + nCoord].handle) )
              nCoord ++;
              
        PatchArray const &array = patchArrays[handle.arrayIndex];
        int patchType = array.GetPatchType();
        Far::PatchParam const & param = patchParamBuffer[handle.patchIndex];

        unsigned int bitField = param.field1;

        const int *cvs = &patchIndexBuffer[array.indexBase + handle.vertIndex];

        __declspec( align(64) ) float u[nCoord];
        __declspec( align(64) ) float v[nCoord];        
        
        for(int n=0; n<nCoord; n++) {
            u[n] = patchCoords[i + n].s;
            v[n] = patchCoords[i + n].t;            
        }
        
        if (patchType == Far::PatchDescriptor::REGULAR) {
            ispc::evalBSplineNoDerivative(bitField, nCoord, u, v, cvs, ispcSrcDesc, src, 
                              ispcDstDesc, dst);
        } else if (patchType == Far::PatchDescriptor::GREGORY_BASIS) {
            ispc::evalGregoryNoDerivative(bitField, nCoord, u, v, cvs, ispcSrcDesc, src, 
                              ispcDstDesc, dst);        
        } else if (patchType == Far::PatchDescriptor::QUADS) {
            ispc::evalBilinearNoDerivative(bitField, nCoord, u, v, cvs, ispcSrcDesc, src, 
                               ispcDstDesc, dst);           
        } else {
            assert(0);
        }
        
        i += nCoord;
        ispcDstDesc.offset = dstDesc.offset + i * dstDesc.stride;                                                  
    }
    });
    
    return true;
}

/* static */
bool
IspcEvaluator::EvalPatches(const float *src, BufferDescriptor const &srcDesc,
                           float *dst,       BufferDescriptor const &dstDesc,
                           float *du,        BufferDescriptor const &duDesc,
                           float *dv,        BufferDescriptor const &dvDesc,
                           int numPatchCoords,
                           const PatchCoord *patchCoords,
                           const PatchArray *patchArrays,
                           const int *patchIndexBuffer,
                           const PatchParam *patchParamBuffer) {
    if (srcDesc.length != dstDesc.length) return false;
        
    // Copy BufferDescriptor to ispc version
    // Since memory alignment in ISPC may be different from C++,
    // we use the assignment for each field instead of the assignment for 
    // the whole struct
    ispc::BufferDescriptor ispcSrcDesc;
    ispcSrcDesc.offset = srcDesc.offset;
    ispcSrcDesc.length = srcDesc.length;
    ispcSrcDesc.stride = srcDesc.stride;                      
                      
    tbb::blocked_range<int> range = tbb::blocked_range<int>(0, numPatchCoords, grain_size);
    tbb::parallel_for(range, [&](const tbb::blocked_range<int> &r)
    {    
    uint i = r.begin();
        
    ispc::BufferDescriptor ispcDstDesc, ispcDuDesc, ispcDvDesc;                               
    ispcDstDesc.offset = dstDesc.offset + dstDesc.offset + i * dstDesc.stride;
    ispcDstDesc.length = dstDesc.length;
    ispcDstDesc.stride = dstDesc.stride;
    
    ispcDuDesc.offset  = duDesc.offset  + i * duDesc.stride;
    ispcDuDesc.length  = duDesc.length;
    ispcDuDesc.stride  = duDesc.stride;
    
    ispcDvDesc.offset  = dvDesc.offset  + i * dvDesc.stride;
    ispcDvDesc.length  = dvDesc.length;
    ispcDvDesc.stride  = dvDesc.stride;
    while (i < r.end()) {
        // the patch coordinates are sorted by patch handle
        // the following code searches the coordinates that
        // belongs to the same patch so that they can be evalauated 
        // with ISPC
        int nCoord = 1;
        Far::PatchTable::PatchHandle handle = patchCoords[i].handle;
        while(i + nCoord < r.end() && 
              handle.isEqual(patchCoords[i + nCoord].handle) )
              nCoord ++;
              
        PatchArray const &array = patchArrays[handle.arrayIndex];
        int patchType = array.GetPatchType();
        Far::PatchParam const & param = patchParamBuffer[handle.patchIndex];

        unsigned int bitField = param.field1;

        const int *cvs = &patchIndexBuffer[array.indexBase + handle.vertIndex];

        __declspec( align(64) ) float u[nCoord];
        __declspec( align(64) ) float v[nCoord];        
        
        for(int n=0; n<nCoord; n++) {
            u[n] = patchCoords[i + n].s;
            v[n] = patchCoords[i + n].t;            
        }
        
        if (patchType == Far::PatchDescriptor::REGULAR) {
            ispc::evalBSpline(bitField, nCoord, u, v, cvs, ispcSrcDesc, src, 
                              ispcDstDesc, dst, ispcDuDesc, du, ispcDvDesc, dv);
        } else if (patchType == Far::PatchDescriptor::GREGORY_BASIS) {
            ispc::evalGregory(bitField, nCoord, u, v, cvs, ispcSrcDesc, src, 
                              ispcDstDesc, dst, ispcDuDesc, du, ispcDvDesc, dv);        
        } else if (patchType == Far::PatchDescriptor::QUADS) {
            ispc::evalBilinear(bitField, nCoord, u, v, cvs, ispcSrcDesc, src, 
                               ispcDstDesc, dst, ispcDuDesc, du, ispcDvDesc, dv);           
        } else {
            assert(0);
        }
        
        i += nCoord;
        ispcDstDesc.offset = dstDesc.offset + i * dstDesc.stride;
        ispcDuDesc.offset  = duDesc.offset  + i * duDesc.stride;
        ispcDvDesc.offset  = dvDesc.offset  + i * dvDesc.stride;                                                        
    }
    });
    
    return true;
}


}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
