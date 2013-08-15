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
#ifndef OSD_CPU_VERTEX_DESCRIPTOR_H
#define OSD_CPU_VERTEX_DESCRIPTOR_H

#include "../version.h"
#include <string.h>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

struct OsdVertexDescriptor {

    /// Constructor
    OsdVertexDescriptor() : numVertexElements(0), numVaryingElements(0) {}

    /// Constructor
    ///
    /// @param numVertexElem   number of vertex-interpolated data elements (floats)
    ///
    /// @param numVaryingElem  number of varying-interpolated data elements (floats)
    ///
    OsdVertexDescriptor(int numVertexElem, int numVaryingElem)
        : numVertexElements(numVertexElem),
        numVaryingElements(numVaryingElem) { }

    /// Sets descriptor
    ///
    /// @param numVertexElem   number of vertex-interpolated data elements (floats)
    ///
    /// @param numVaryingElem  number of varying-interpolated data elements (floats)
    ///
    void Set(int numVertexElem, int numVaryingElem) {
        numVertexElements = numVertexElem;
        numVaryingElements = numVaryingElem;
    }
    
    /// Resets the descriptor
    void Reset() {
        numVertexElements = numVaryingElements = 0;
    }
    
    /// Returns the total number of elements (vertex + varying)
    int GetNumElements() const {
        return numVertexElements + numVaryingElements;
    }

    bool operator == (OsdVertexDescriptor const & other) {
        return (numVertexElements == other.numVertexElements and
                numVaryingElements == other.numVaryingElements);
    }

    /// Resets the contents of vertex & varying primvar data buffers for a given
    /// vertex.
    ///
    /// @param vertex  The float array containing the vertex-interpolated primvar 
    ///                data that needs to be reset.
    ///
    /// @param varying The float array containing the varying-interpolated primvar
    ///                data that needs to be reset.
    ///
    /// @param index   Vertex index in the buffer.
    ///
    void Clear(float *vertex, float *varying, int index) const {
        if (vertex) {
            memset(vertex+index*numVertexElements, 0, sizeof(float)*numVertexElements);               
        }

        if (varying) {
            memset(varying+index*numVaryingElements, 0, sizeof(float)*numVaryingElements);       
               
        }
    }
    
    /// Applies "dst += src*weight" to "vertex" primvar data in a vertex buffer.
    ///
    /// @param vertex The VertexData buffer
    ///
    /// @param dstIndex Index of the destination vertex.
    ///
    /// @param srcIndex Index of the origin vertex.
    ///
    /// @param weight Weight applied to the primvar data.
    ///
    inline 
    void AddWithWeight(float *vertex, int dstIndex, int srcIndex, float weight) const {
        int d = dstIndex * numVertexElements;
        int s = srcIndex * numVertexElements;       
#if defined ( __INTEL_COMPILER ) or defined ( __ICC )
    #pragma ivdep  
    #pragma vector aligned
#endif 
        for (int i = 0; i < numVertexElements; ++i)
            vertex[d++] += vertex[s++] * weight;
    }

    /// Applies "dst += src*weight" to "varying" primvar data in a vertex buffer.
    ///
    /// @param varying The VaryingData buffer
    ///
    /// @param dstIndex Index of the destination vertex.
    ///
    /// @param srcIndex Index of the source vertex.
    ///
    /// @param weight Weight applied to the primvar data.
    ///
    inline
    void AddVaryingWithWeight(float *varying, int dstIndex, int srcIndex, float weight) const {
        int d = dstIndex * numVaryingElements;
        int s = srcIndex * numVaryingElements;
#if defined ( __INTEL_COMPILER ) or defined ( __ICC )
    #pragma ivdep  
    #pragma vector aligned
#endif 
        for (int i = 0; i < numVaryingElements; ++i)
            varying[d++] += varying[s++] * weight;
    }

    /// Applies an "add" vertex edit
    ///
    /// @param vertex The primvar data buffer.
    ///
    /// @param primVarOffset Offset to the primvar datum.
    ///
    /// @param primVarWidth Length of the primvar datum.
    ///
    /// @param editIndex The location of the vertex in the buffer.
    ///
    /// @param editValues The values to add to the primvar datum.
    ///
    void ApplyVertexEditAdd(float *vertex, int primVarOffset, int primVarWidth, int editIndex, const float *editValues) const {
        int d = editIndex * numVertexElements + primVarOffset;
        for (int i = 0; i < primVarWidth; ++i) {
            vertex[d++] += editValues[i];
        }
    }

    /// Applies a "set" vertex edit
    ///
    /// @param vertex The primvar data buffer.
    ///
    /// @param primVarOffset Offset to the primvar datum.
    ///
    /// @param primVarWidth Length of the primvar datum.
    ///
    /// @param editIndex The location of the vertex in the buffer.
    ///
    /// @param editValues The values to add to the primvar datum.
    ///
    void ApplyVertexEditSet(float *vertex, int primVarOffset, int primVarWidth, int editIndex, const float *editValues) const {
        int d = editIndex * numVertexElements + primVarOffset;
        for (int i = 0; i < primVarWidth; ++i) {
            vertex[d++] = editValues[i];
        }
    }

    int numVertexElements;
    int numVaryingElements;
};

/// \brief Describes vertex elements in interleaved data buffers
struct OsdVertexBufferDescriptor {

    /// Default Constructor
    OsdVertexBufferDescriptor() : offset(0), length(0), stride(0) { }

    /// Constructor
    OsdVertexBufferDescriptor(int o, int l, int s) : offset(o), length(l), stride(s) { }

    /// True if the descriptor values are internally consistent
    bool IsValid() const {
        return (length>0) and (offset<length) and (stride>=length);
    }
    
    /// True if the 'other' descriptor can be used as a destination for
    /// data evaluations.
    bool CanEval( OsdVertexBufferDescriptor const & other ) const {
        return IsValid() and 
               other.IsValid() and 
               (length==other.length) and 
               (other.length <= (stride-offset));
    }

    /// Resets the descriptor to default
    void Reset() {
        offset = length = stride = 0;
    }

    int offset;  // offset to desired element data
    int length;  // number or length of the data
    int stride;  // stride to the next element
};

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* OSD_CPU_VERTEX_DESCRIPTRO_H */
