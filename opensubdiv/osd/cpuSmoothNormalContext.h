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

#ifndef OSD_CPU_SMOOTHNORMAL_CONTEXT_H
#define OSD_CPU_SMOOTHNORMAL_CONTEXT_H

#include "../version.h"

#include "../osd/nonCopyable.h"
#include "../osd/vertexDescriptor.h"
#include "../osd/vertex.h"

#include "../far/patchTables.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

class CpuSmoothNormalContext :  private NonCopyable<CpuSmoothNormalContext> {

public:

    /// Creates an CpuComputeContext instance
    ///
    /// @param patchTables  The Far::PatchTables used for this Context.
    ///
    /// @param resetMemory  Set to true if the target vertex buffer needs its
    ///                     memory reset before accumulating the averaged normals.
    ///                     If the SmoothNormal Controller runs after a Computer
    ///                     Controller, then the vertex buffer will already have
    ///                     been reset and this step can be skipped to save time.
    ///
    static CpuSmoothNormalContext * Create(
        Far::PatchTables const *patchTables, bool resetMemory=false);

    /// Binds a vertex and a varying data buffers to the context. Binding ensures
    /// that data buffers are properly inter-operated between Contexts and
    /// Controllers operating across multiple devices.
    ///
    /// @param in   a buffer containing input vertex-interpolated primvar data
    ///
    /// @param iOfs offset to the buffer element describing the vertex position
    ///
    /// @param out  a buffer where the smooth normals will be output
    ///
    /// @param oOfs offset to the buffer element describing the normal position
    ///
    template<class VERTEX_BUFFER>
    void Bind(VERTEX_BUFFER * in, int iOfs,
              VERTEX_BUFFER * out, int oOfs) {

        assert( ((iOfs+3)<=in->GetNumElements()) and
            ((oOfs+3)<=out->GetNumElements()) and
            out->GetNumVertices()>=in->GetNumVertices());

        _iBuffer = in ? in->BindCpuBuffer() : 0;
        _oBuffer = out ? out->BindCpuBuffer() : 0;

        _iDesc = VertexBufferDescriptor( iOfs, 3, in->GetNumElements() );
        _oDesc = VertexBufferDescriptor( oOfs, 3, out->GetNumElements() );

        _numVertices = out->GetNumVertices();
    }

    /// Unbinds any previously bound vertex and varying data buffers.
    void Unbind() {

        _iBuffer = _oBuffer = 0;
        _iDesc.Reset();
        _oDesc.Reset();
        _numVertices = 0;
    }

    /// Returns the vector of patch arrays
    const Far::PatchTables::PatchArrayVector & GetPatchArrayVector() const {
        return _patchArrays;
    }

    /// The ordered array of control vertex indices for all the patches
    const std::vector<unsigned int> & GetControlVertices() const {
        return _patches;
    }

    /// Returns a pointer to the data of the input buffer
    float const * GetCurrentInputVertexBuffer() const {
        return _iBuffer;
    }

    /// Returns a pointer to the data of the output buffer
    float * GetCurrentOutputVertexBuffer() {
        return _oBuffer;
    }

    /// Returns an VertexDescriptor for the input vertex data
    VertexBufferDescriptor const & GetInputVertexDescriptor() const {
        return _iDesc;
    }

    /// Returns an VertexDescriptor for the buffer where the normals data
    /// will be stored
    VertexBufferDescriptor const & GetOutputVertexDescriptor() const {
        return _oDesc;
    }

    /// Returns the number of vertices in output vertex buffer
    int GetNumVertices() const {
        return _numVertices;
    }

    /// Returns whether the controller needs to reset the vertex buffer before
    /// accumulating smooth normals
    bool GetResetMemory() const {
        return _resetMemory;
    }

    /// Set to true if the controller needs to reset the vertex buffer before
    /// accumulating smooth normals
    void SetResetMemory(bool resetMemory) {
        _resetMemory = resetMemory;
    }

protected:
    // Constructor
    explicit CpuSmoothNormalContext(
        Far::PatchTables const *patchTables, bool resetMemory);

private:

    // Topology data for a mesh
    Far::PatchTables::PatchArrayVector     _patchArrays;    // patch descriptor for each patch in the mesh
    Far::PatchTables::PTable               _patches;        // patch control vertices

    VertexBufferDescriptor _iDesc,
                              _oDesc;

    int _numVertices;

    float * _iBuffer,
          * _oBuffer;

    bool _resetMemory;  // set to true if the output buffer needs to be reset to 0
};


}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OSD_CPU_SMOOTHNORMAL_CONTEXT_H
