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

class ISPCEvalPatchNoDerivativeKernel {

    const float            *_src;
    BufferDescriptor        _srcDesc;
    float                  *_dst;   
    BufferDescriptor        _dstDesc;
    const PatchCoord       *_patchCoords;
    const PatchArray       *_patchArrays;
    const int              *_patchIndexBuffer;
    const PatchParam       *_patchParamBuffer;

public:
    ISPCEvalPatchNoDerivativeKernel
          (const float            *src, 
           BufferDescriptor        srcDesc,
           float                  *dst,       
           BufferDescriptor        dstDesc,
           const PatchCoord       *patchCoords,
           const PatchArray       *patchArrays,
           const int              *patchIndexBuffer,
           const PatchParam       *patchParamBuffer) :
                _src(src), 
                _srcDesc(srcDesc),
                _dst(dst),       
                _dstDesc(dstDesc),
                _patchCoords(patchCoords),
                _patchArrays(patchArrays),
                _patchIndexBuffer(patchIndexBuffer),
                _patchParamBuffer(patchParamBuffer)                       
          { }

    ISPCEvalPatchNoDerivativeKernel(ISPCEvalPatchNoDerivativeKernel const & other) {
                _src              = other._src;
                _srcDesc          = other._srcDesc;
                _dst              = other._dst;
                _dstDesc          = other._dstDesc;
                _patchCoords      = other._patchCoords;
                _patchArrays      = other._patchArrays;
                _patchIndexBuffer = other._patchIndexBuffer;
                _patchParamBuffer = other._patchParamBuffer; 
    }

    void operator() (tbb::blocked_range<int> const &r) const {
        // Copy BufferDescriptor to ispc version
        // Since memory alignment in ISPC may be different from C++,
        // we use the assignment for each field instead of the assignment for 
        // the whole struct
        ispc::BufferDescriptor ispcSrcDesc;
        ispcSrcDesc.offset = _srcDesc.offset;
        ispcSrcDesc.length = _srcDesc.length;
        ispcSrcDesc.stride = _srcDesc.stride;                                           
                          
        uint i = r.begin();
        
        ispc::BufferDescriptor ispcDstDesc;
        ispcDstDesc.offset = _dstDesc.offset + _dstDesc.offset + i * _dstDesc.stride;
        ispcDstDesc.length = _dstDesc.length;
        ispcDstDesc.stride = _dstDesc.stride;
    
        while (i < r.end()) {
            // the patch coordinates are sorted by patch handle
            // the following code searches the coordinates that
            // belongs to the same patch so that they can be evalauated 
            // with ISPC
            int nCoord = 1;
            Far::PatchTable::PatchHandle handle = _patchCoords[i].handle;
            while(i + nCoord < r.end() && 
                  handle == _patchCoords[i + nCoord].handle )
                  nCoord ++;
              
            PatchArray const &array = _patchArrays[handle.arrayIndex];
            int patchType = array.GetPatchType();
            Far::PatchParam const & param = _patchParamBuffer[handle.patchIndex];

            unsigned int bitField = param.field1;

            const int *cvs = &_patchIndexBuffer[array.indexBase + handle.vertIndex];

            __declspec( align(64) ) float u[nCoord];
            __declspec( align(64) ) float v[nCoord];        
        
            for(int n=0; n<nCoord; n++) {
                u[n] = _patchCoords[i + n].s;
                v[n] = _patchCoords[i + n].t;            
            }
        
            if (patchType == Far::PatchDescriptor::REGULAR) {
                ispc::evalBSplineNoDerivative(bitField, nCoord, u, v, cvs, ispcSrcDesc, _src, 
                              ispcDstDesc, _dst);
            } else if (patchType == Far::PatchDescriptor::GREGORY_BASIS) {
                ispc::evalGregoryNoDerivative(bitField, nCoord, u, v, cvs, ispcSrcDesc, _src, 
                              ispcDstDesc, _dst);        
            } else if (patchType == Far::PatchDescriptor::QUADS) {
                ispc::evalBilinearNoDerivative(bitField, nCoord, u, v, cvs, ispcSrcDesc, _src, 
                               ispcDstDesc, _dst);           
            } else {
                assert(0);
            }
        
            i += nCoord;
            ispcDstDesc.offset = _dstDesc.offset + i * _dstDesc.stride;                                                  
        }
    }
};

class ISPCEvalPatchKernel {

    const float            *_src;
    BufferDescriptor        _srcDesc;
    float                  *_dst;   
    BufferDescriptor        _dstDesc;
    float                  *_du;
    BufferDescriptor        _duDesc;
    float                  *_dv;    
    BufferDescriptor        _dvDesc;
    const PatchCoord       *_patchCoords;
    const PatchArray       *_patchArrays;
    const int              *_patchIndexBuffer;
    const PatchParam       *_patchParamBuffer;

public:
    ISPCEvalPatchKernel(const float            *src, 
                        BufferDescriptor        srcDesc,
                        float                  *dst,       
                        BufferDescriptor        dstDesc,
                        float                  *du,     
                        BufferDescriptor        duDesc,
                        float                  *dv,        
                        BufferDescriptor        dvDesc,                        
                        const PatchCoord       *patchCoords,
                        const PatchArray       *patchArrays,
                        const int              *patchIndexBuffer,
                        const PatchParam       *patchParamBuffer) :
                _src(src), 
                _srcDesc(srcDesc),
                _dst(dst),       
                _dstDesc(dstDesc),
                _du(du),       
                _duDesc(duDesc),
                _dv(dv),       
                _dvDesc(dvDesc),                                
                _patchCoords(patchCoords),
                _patchArrays(patchArrays),
                _patchIndexBuffer(patchIndexBuffer),
                _patchParamBuffer(patchParamBuffer)                       
          { }

    ISPCEvalPatchKernel(ISPCEvalPatchKernel const & other) {
                _src              = other._src;
                _srcDesc          = other._srcDesc;
                _dst              = other._dst;
                _dstDesc          = other._dstDesc;
                _du               = other._du;       
                _duDesc           = other._duDesc;
                _dv               = other._dv;       
                _dvDesc           = other._dvDesc;
                _patchCoords      = other._patchCoords;
                _patchArrays      = other._patchArrays;
                _patchIndexBuffer = other._patchIndexBuffer;
                _patchParamBuffer = other._patchParamBuffer; 
    }

    void operator() (tbb::blocked_range<int> const &r) const {
        // Copy BufferDescriptor to ispc version
        // Since memory alignment in ISPC may be different from C++,
        // we use the assignment for each field instead of the assignment for 
        // the whole struct
        ispc::BufferDescriptor ispcSrcDesc, ispcDstDesc, ispcDuDesc, ispcDvDesc;
        ispcSrcDesc.offset = _srcDesc.offset;
        ispcSrcDesc.length = _srcDesc.length;
        ispcSrcDesc.stride = _srcDesc.stride;                                           
                          
        uint i = r.begin();
                                       
        ispcDstDesc.offset = _dstDesc.offset + _dstDesc.offset + i * _dstDesc.stride;
        ispcDstDesc.length = _dstDesc.length;
        ispcDstDesc.stride = _dstDesc.stride;
    
        ispcDuDesc.offset  = _duDesc.offset  + i * _duDesc.stride;
        ispcDuDesc.length  = _duDesc.length;
        ispcDuDesc.stride  = _duDesc.stride;
    
        ispcDvDesc.offset  = _dvDesc.offset  + i * _dvDesc.stride;
        ispcDvDesc.length  = _dvDesc.length;
        ispcDvDesc.stride  = _dvDesc.stride;
    
        while (i < r.end()) {
            // the patch coordinates are sorted by patch handle
            // the following code searches the coordinates that
            // belongs to the same patch so that they can be evalauated 
            // with ISPC
            int nCoord = 1;
            Far::PatchTable::PatchHandle handle = _patchCoords[i].handle;
            while(i + nCoord < r.end() && 
                  handle == _patchCoords[i + nCoord].handle )
                  nCoord ++;
              
            PatchArray const &array = _patchArrays[handle.arrayIndex];
            int patchType = array.GetPatchType();
            Far::PatchParam const & param = _patchParamBuffer[handle.patchIndex];

            unsigned int bitField = param.field1;

            const int *cvs = &_patchIndexBuffer[array.indexBase + handle.vertIndex];

            __declspec( align(64) ) float u[nCoord];
            __declspec( align(64) ) float v[nCoord];        
        
            for(int n=0; n<nCoord; n++) {
                u[n] = _patchCoords[i + n].s;
                v[n] = _patchCoords[i + n].t;            
            }
        
            if (patchType == Far::PatchDescriptor::REGULAR) {
                ispc::evalBSpline(bitField, nCoord, u, v, cvs, ispcSrcDesc, _src, 
                              ispcDstDesc, _dst, ispcDuDesc, _du, ispcDvDesc, _dv);
            } else if (patchType == Far::PatchDescriptor::GREGORY_BASIS) {
                ispc::evalGregory(bitField, nCoord, u, v, cvs, ispcSrcDesc, _src, 
                              ispcDstDesc, _dst, ispcDuDesc, _du, ispcDvDesc, _dv);        
            } else if (patchType == Far::PatchDescriptor::QUADS) {
                ispc::evalBilinear(bitField, nCoord, u, v, cvs, ispcSrcDesc, _src, 
                               ispcDstDesc, _dst, ispcDuDesc, _du, ispcDvDesc, _dv);           
            } else {
                assert(0);
            }
        
            i += nCoord;
            ispcDstDesc.offset = _dstDesc.offset + i * _dstDesc.stride;                                                  
        }
    }
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
                          
    ISPCEvalPatchNoDerivativeKernel kernel(src, srcDesc,
                                           dst, dstDesc,
                                           patchCoords,
                                           patchArrays,
                                           patchIndexBuffer,
                                           patchParamBuffer);
                          
    tbb::blocked_range<int> range = tbb::blocked_range<int>(0, numPatchCoords, grain_size);
    tbb::parallel_for(range, kernel);

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
    
    ISPCEvalPatchKernel kernel(src, srcDesc,
                               dst, dstDesc,
                               du,  duDesc,
                               dv, dvDesc,
                               patchCoords,
                               patchArrays,
                               patchIndexBuffer,
                               patchParamBuffer);
                          
    tbb::blocked_range<int> range = tbb::blocked_range<int>(0, numPatchCoords, grain_size);
    tbb::parallel_for(range, kernel);
    
    return true;
}


}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
