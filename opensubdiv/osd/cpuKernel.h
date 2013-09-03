//
//     Copyright 2013 Pixar
//
//     Licensed under the Apache License, Version 2.0 (the "License");
//     you may not use this file except in compliance with the License
//     and the following modification to it: Section 6 Trademarks.
//     deleted and replaced with:
//
//     6. Trademarks. This License does not grant permission to use the
//     trade names, trademarks, service marks, or product names of the
//     Licensor and its affiliates, except as required for reproducing
//     the content of the NOTICE file.
//
//     You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
//     Unless required by applicable law or agreed to in writing,
//     software distributed under the License is distributed on an
//     "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
//     either express or implied.  See the License for the specific
//     language governing permissions and limitations under the
//     License.
//
#ifndef OSD_CPU_KERNEL_H
#define OSD_CPU_KERNEL_H

#if defined ( __INTEL_COMPILER ) or defined ( __ICC )
    #define __ALIGN_DATA __declspec(align(32))
#else
    #define __ALIGN_DATA 
#endif

#include <string.h>
#include <math.h>
#include "../version.h"

#include "../osd/vertexDescriptor.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

struct OsdVertexDescriptor;

template<int numVertexElements>
void ComputeFaceKernel(float     *vertex, 
                       const int *F_IT, 
                       const int *F_ITa, 
                             int  vertexOffset, 
                             int  tableOffset,
                             int  start, 
                             int  end) {

    __ALIGN_DATA float result [numVertexElements];
    __ALIGN_DATA float result1[numVertexElements];                
    float *src, *des;        
    for (int i = start + tableOffset; i < end + tableOffset; i++) {
        int h = F_ITa[2*i];
        int n = F_ITa[2*i+1];
        float weight = 1.0f/n;

#if defined ( __INTEL_COMPILER ) or defined ( __ICC )
    #pragma simd       
    #pragma vector aligned
#endif
        for (int k = 0; k < numVertexElements; ++k)
            result[k] = 0.0f;
            
        int dstIndex = i + vertexOffset - tableOffset;
        
        for (int j = 0; j < n; ++j) {
            int index = F_IT[h+j];
            src = vertex + index  * numVertexElements;
#if defined ( __INTEL_COMPILER ) or defined ( __ICC )
    #pragma simd       
    #pragma vector aligned
#endif
            for (int k = 0; k < numVertexElements; ++k)
                result[k] += src[k] * weight;
        }
#if defined ( __INTEL_COMPILER ) or defined ( __ICC )
    #pragma simd       
    #pragma vector aligned
#endif
        for (int k = 0; k < numVertexElements; ++k)
            result1[k] = result[k];                 
        des = vertex + dstIndex * numVertexElements;
        memcpy(des, result1, sizeof(float)*numVertexElements);        
    }
}
void OsdCpuComputeFace(OsdVertexDescriptor const &vdesc,
                       float * vertex, float * varying,
                       const int *F_IT, const int *F_ITa,
                       int vertexOffset, int tableOffset,
                       int start, int end);

template<int numVertexElements>
void ComputeEdgeKernel(      float *vertex,
                       const int   *E_IT, 
                       const float *E_W, 
                             int    vertexOffset, 
                             int    tableOffset,
                             int    start, 
                             int    end) 
{
    __ALIGN_DATA float result[numVertexElements];    
    __ALIGN_DATA float result1[numVertexElements]; 
    
    float *src, *src2, *des;
    for (int i = start + tableOffset; i < end + tableOffset; i++) {
        int eidx0 = E_IT[4*i+0];
        int eidx1 = E_IT[4*i+1];
        int eidx2 = E_IT[4*i+2];
        int eidx3 = E_IT[4*i+3];

        float vertWeight = E_W[i*2+0];
      
        src  = vertex + eidx0 * numVertexElements;
        src2 = vertex + eidx1 * numVertexElements;
#if defined ( __INTEL_COMPILER ) or defined ( __ICC )
    #pragma simd       
    #pragma vector aligned
#endif
        for (int j = 0; j < numVertexElements; ++j)
            result[j] = (src[j]+src2[j]) * vertWeight;

        if (eidx2 != -1) {
            float faceWeight = E_W[i*2+1];
            src  = vertex + eidx2 * numVertexElements;
            src2 = vertex + eidx3 * numVertexElements;            
#if defined ( __INTEL_COMPILER ) or defined ( __ICC )
    #pragma simd       
    #pragma vector aligned
#endif
            for (int j = 0; j < numVertexElements; ++j)
                result[j] += (src[j]+src2[j]) * faceWeight;
        }
#if defined ( __INTEL_COMPILER ) or defined ( __ICC )
    #pragma simd       
    #pragma vector aligned
#endif
        for (int j = 0; j < numVertexElements; ++j)
            result1[j] = result[j]; 

        int dstIndex = i + vertexOffset - tableOffset;
        des = vertex + dstIndex * numVertexElements;
        memcpy(des, result1, sizeof(float)*numVertexElements);
    }
}
void OsdCpuComputeEdge(OsdVertexDescriptor const &vdesc,
                       float *vertex, float * varying,
                       const int *E_IT, const float *E_ITa,
                       int vertexOffset, int tableOffset,
                       int start, int end);

template<int numVertexElements>
void ComputeVertexAKernel(      float *vertex, 
                          const int   *V_ITa, 
                          const float *V_W, 
                                int vertexOffset,
                                int tableOffset,
                                int start,
                                int end,
                                int pass) {
    __ALIGN_DATA float result [numVertexElements];
    __ALIGN_DATA float result1[numVertexElements];        
    float *src, *src2, *src3, *des;        
    for (int i = start + tableOffset; i < end + tableOffset; i++) {
        int n     = V_ITa[5*i+1];
        int p     = V_ITa[5*i+2];
        int eidx0 = V_ITa[5*i+3];
        int eidx1 = V_ITa[5*i+4];

        float weight = (pass == 1) ? V_W[i] : 1.0f - V_W[i];

        // In the case of fractional weight, the weight must be inverted since
        // the value is shared with the k_Smooth kernel (statistically the
        // k_Smooth kernel runs much more often than this one)
        if (weight > 0.0f && weight < 1.0f && n > 0)
            weight = 1.0f - weight;

        int dstIndex = i + vertexOffset - tableOffset;

        if (not pass) {
#if defined ( __INTEL_COMPILER ) or defined ( __ICC )
    #pragma simd       
    #pragma vector aligned
#endif
            for (int k = 0; k < numVertexElements; ++k)
                result[k] = 0.0f;
        }
        else {
            memcpy(result1, vertex+dstIndex*numVertexElements,  
                   sizeof(float)*numVertexElements);
#if defined ( __INTEL_COMPILER ) or defined ( __ICC )
    #pragma simd       
    #pragma vector aligned
#endif
            for (int k = 0; k < numVertexElements; ++k)
                result[k] = result1[k];                   
        }
        
        if (eidx0 == -1 || (pass == 0 && (n == -1))) {
            src = vertex + p * numVertexElements;
#if defined ( __INTEL_COMPILER ) or defined ( __ICC )
    #pragma simd       
    #pragma vector aligned
#endif
            for (int j = 0; j < numVertexElements; ++j)
                result[j] += src[j] * weight;
        } else {
            src  = vertex + p     * numVertexElements;
            src2 = vertex + eidx0 * numVertexElements;            
            src3 = vertex + eidx1 * numVertexElements;            
#if defined ( __INTEL_COMPILER ) or defined ( __ICC )
    #pragma simd       
    #pragma vector aligned
#endif
            for (int j = 0; j < numVertexElements; ++j)
                result[j] += (src[j]*0.75f + src2[j]*0.125f + src3[j]*0.125f) * weight;
        }    
#if defined ( __INTEL_COMPILER ) or defined ( __ICC )
    #pragma simd       
    #pragma vector aligned
#endif
        for (int k = 0; k < numVertexElements; ++k)
            result1[k] = result[k]; 

        des = vertex + dstIndex * numVertexElements;
        memcpy(des, result1, sizeof(float)*numVertexElements);
    }
}
void OsdCpuComputeVertexA(OsdVertexDescriptor const &vdesc,
                          float *vertex, float * varying,
                          const int *V_ITa, const float *V_IT,
                          int vertexOffset, int tableOffset,
                          int start, int end, int pass);

template<int numVertexElements>
void ComputeVertexBKernel(      float *vertex,
                          const   int *V_ITa, 
                          const   int *V_IT,
                          const float *V_W,
                                  int  vertexOffset, 
                                  int  tableOffset, 
                                  int  start,
                                  int  end) {
    __ALIGN_DATA float result [numVertexElements];
    __ALIGN_DATA float result1[numVertexElements];        
    float *src, *src1, *src2, *des;  
    for (int i = start + tableOffset; i < end + tableOffset; i++) {
        int h = V_ITa[5*i];
        int n = V_ITa[5*i+1];
        int p = V_ITa[5*i+2];

        float weight = V_W[i];
        float wp = 1.0f/static_cast<float>(n*n);
        float wv = (n-2.0f) * n * wp; 

        int dstIndex = i + vertexOffset - tableOffset;
                                  
        src = vertex + p * numVertexElements;
#if defined ( __INTEL_COMPILER ) or defined ( __ICC )
    #pragma simd       
    #pragma vector aligned
#endif
        for (int j = 0; j < numVertexElements; ++j)
            result[j] = src[j] * weight * wv; 

        for (int j = 0; j < n; ++j) {
            int id1 = V_IT[h+2*j];
            int id2 = V_IT[h+2*j+1];
            src1 = vertex + id1 * numVertexElements;
            src2 = vertex + id2 * numVertexElements;
#if defined ( __INTEL_COMPILER ) or defined ( __ICC )
    #pragma simd       
    #pragma vector aligned
#endif
            for (int k = 0; k < numVertexElements; ++k)
                result[k] += (src1[k]+src2[k]) * weight * wp; 
        }
#if defined ( __INTEL_COMPILER ) or defined ( __ICC )
    #pragma simd       
    #pragma vector aligned
#endif
        for (int j = 0; j < numVertexElements; ++j)
            result1[j] = result[j]; 

        des = vertex + dstIndex * numVertexElements;
        memcpy(des, result1, sizeof(float)*numVertexElements);
    }
}
        
void OsdCpuComputeVertexB(OsdVertexDescriptor const &vdesc,
                          float *vertex, float * varying,
                          const int *V_ITa, const int *V_IT, const float *V_W,
                          int vertexOffset, int tableOffset,
                          int start, int end);

template<int numVertexElements>
void ComputeLoopVertexBKernel(      float *vertex, 
                              const   int *V_ITa, 
                              const   int *V_IT, 
                              const float *V_W,
                                      int  vertexOffset, 
                                      int  tableOffset, 
                                      int  start, 
                                      int  end) {
    __ALIGN_DATA float result [numVertexElements];
    __ALIGN_DATA float result1[numVertexElements];        
    float *src, *des;  
    for (int i = start + tableOffset; i < end + tableOffset; i++) {
        int h = V_ITa[5*i];
        int n = V_ITa[5*i+1];
        int p = V_ITa[5*i+2];

        float weight = V_W[i];
        float wp = 1.0f/static_cast<float>(n);
        float beta = 0.25f * cosf(static_cast<float>(M_PI) * 2.0f * wp) + 0.375f;
        beta = beta * beta;
        beta = (0.625f - beta) * wp;

        int dstIndex = i + vertexOffset - tableOffset;
        src = vertex + p * numVertexElements;
#if defined ( __INTEL_COMPILER ) or defined ( __ICC )
    #pragma simd       
    #pragma vector aligned
#endif
        for (int k = 0; k < numVertexElements; ++k)
            result[k] = src[k] * weight * (1.0f - (beta * n));

        for (int j = 0; j < n; ++j) {
            src = vertex + V_IT[h+j] * numVertexElements;
#if defined ( __INTEL_COMPILER ) or defined ( __ICC )
    #pragma simd       
    #pragma vector aligned
#endif
            for (int k = 0; k < numVertexElements; ++k)
                result[k] += src[k] * weight * beta;
        }

#if defined ( __INTEL_COMPILER ) or defined ( __ICC )
    #pragma simd       
    #pragma vector aligned
#endif
        for (int j = 0; j < numVertexElements; ++j)
            result1[j] = result[j]; 

        des = vertex + dstIndex * numVertexElements;                
        memcpy(des, result1, sizeof(float)*numVertexElements);
    }
}
void OsdCpuComputeLoopVertexB(OsdVertexDescriptor const &vdesc,
                              float *vertex, float * varying,
                              const int *V_ITa, const int *V_IT,
                              const float *V_W,
                              int vertexOffset, int tableOffset,
                              int start, int end);

template<int numVertexElements>
void ComputeBilinearEdgeKernel(    float *vertex,
                               const int *E_IT,
                                     int  vertexOffset, 
                                     int  tableOffset, 
                                     int  start, 
                                     int  end) 
{
    __ALIGN_DATA float result [numVertexElements];        
    float *src1, *src2, *des;      
    for (int i = start + tableOffset; i < end + tableOffset; i++) {
        int eidx0 = E_IT[2*i+0];
        int eidx1 = E_IT[2*i+1];

        src1 = vertex + eidx0 * numVertexElements;
        src2 = vertex + eidx1 * numVertexElements;            
#if defined ( __INTEL_COMPILER ) or defined ( __ICC )
    #pragma simd       
    #pragma vector aligned
#endif
        for (int j = 0; j < numVertexElements; ++j)
            result[j] = 0.5f * (src1[j]+src2[j]);     
                
        int dstIndex = i + vertexOffset - tableOffset;        
        des = vertex + dstIndex * numVertexElements;
        memcpy(des, result, sizeof(float)*numVertexElements);                    
    }
}
void OsdCpuComputeBilinearEdge(OsdVertexDescriptor const &vdesc,
                               float *vertex, float * varying,
                               const int *E_IT,
                               int vertexOffset, int tableOffset,
                               int start, int end);

void OsdCpuComputeBilinearVertex(OsdVertexDescriptor const &vdesc,
                                 float *vertex, float * varying,
                                 const int *V_ITa,
                                 int vertexOffset, int tableOffset,
                                 int start, int end);

void OsdCpuEditVertexAdd(OsdVertexDescriptor const &vdesc, float *vertex,
                         int primVarOffset, int primVarWidth,
                         int vertexOffset, int tableOffset,
                         int start, int end,
                         const unsigned int *editIndices,
                         const float *editValues);

void OsdCpuEditVertexSet(OsdVertexDescriptor const &vdesc, float *vertex,
                         int primVarOffset, int primVarWidth,
                         int vertexOffset, int tableOffset,
                         int start, int end,
                         const unsigned int *editIndices,
                         const float *editValues);

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OSD_CPU_KERNEL_H
