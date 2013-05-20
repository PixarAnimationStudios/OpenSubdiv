//
//     Copyright (C) Pixar. All rights reserved.
//
//     This license governs use of the accompanying software. If you
//     use the software, you accept this license. If you do not accept
//     the license, do not use the software.
//
//     1. Definitions
//     The terms "reproduce," "reproduction," "derivative works," and
//     "distribution" have the same meaning here as under U.S.
//     copyright law.  A "contribution" is the original software, or
//     any additions or changes to the software.
//     A "contributor" is any person or entity that distributes its
//     contribution under this license.
//     "Licensed patents" are a contributor's patent claims that read
//     directly on its contribution.
//
//     2. Grant of Rights
//     (A) Copyright Grant- Subject to the terms of this license,
//     including the license conditions and limitations in section 3,
//     each contributor grants you a non-exclusive, worldwide,
//     royalty-free copyright license to reproduce its contribution,
//     prepare derivative works of its contribution, and distribute
//     its contribution or any derivative works that you create.
//     (B) Patent Grant- Subject to the terms of this license,
//     including the license conditions and limitations in section 3,
//     each contributor grants you a non-exclusive, worldwide,
//     royalty-free license under its licensed patents to make, have
//     made, use, sell, offer for sale, import, and/or otherwise
//     dispose of its contribution in the software or derivative works
//     of the contribution in the software.
//
//     3. Conditions and Limitations
//     (A) No Trademark License- This license does not grant you
//     rights to use any contributor's name, logo, or trademarks.
//     (B) If you bring a patent claim against any contributor over
//     patents that you claim are infringed by the software, your
//     patent license from such contributor to the software ends
//     automatically.
//     (C) If you distribute any portion of the software, you must
//     retain all copyright, patent, trademark, and attribution
//     notices that are present in the software.
//     (D) If you distribute any portion of the software in source
//     code form, you may do so only under this license by including a
//     complete copy of this license with your distribution. If you
//     distribute any portion of the software in compiled or object
//     code form, you may only do so under a license that complies
//     with this license.
//     (E) The software is licensed "as-is." You bear the risk of
//     using it. The contributors give no express warranties,
//     guarantees or conditions. You may have additional consumer
//     rights under your local laws which this license cannot change.
//     To the extent permitted under your local laws, the contributors
//     exclude the implied warranties of merchantability, fitness for
//     a particular purpose and non-infringement.
//
#ifndef OSD_CPU_VERTEX_DESCRIPTOR_H
#define OSD_CPU_VERTEX_DESCRIPTOR_H

#include "../version.h"

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
            for (int i = 0; i < numVertexElements; ++i)
                vertex[index*numVertexElements+i] = 0.0f;
        }

        if (varying) {
            for (int i = 0; i < numVaryingElements; ++i)
                varying[index*numVaryingElements+i] = 0.0f;
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
    void AddWithWeight(float *vertex, int dstIndex, int srcIndex, float weight) const {
        int d = dstIndex * numVertexElements;
        int s = srcIndex * numVertexElements;
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
    void AddVaryingWithWeight(float *varying, int dstIndex, int srcIndex, float weight) const {
        int d = dstIndex * numVaryingElements;
        int s = srcIndex * numVaryingElements;
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
        return (offset<length) and (stride>=length);
    }
    
    /// True if the 'other' descriptor can be used as a destination for
    /// data evaluations.
    bool CanEval( OsdVertexBufferDescriptor const & other ) const {
        return IsValid() and 
               other.IsValid() and 
               (length==other.length) and 
               (other.length <= (stride-offset));
    }

    int offset;  // offset to desired element data
    int length;  // number or length of the data
    int stride;  // stride to the next element
};

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* OSD_CPU_VERTEX_DESCRIPTRO_H */
