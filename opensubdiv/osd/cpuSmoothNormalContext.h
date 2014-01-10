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

class OsdCpuSmoothNormalContext :  OsdNonCopyable<OsdCpuSmoothNormalContext> {

public:

    /// Creates an OsdCpuComputeContext instance
    ///
    /// @param farmesh the FarMesh used for this Context.
    ///
    static OsdCpuSmoothNormalContext * Create(FarPatchTables const *patchTables);

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
            ((oOfs+3)<=out->GetNumElements()));

        _iBuffer = in ? in->BindCpuBuffer() : 0;
        _oBuffer = out ? out->BindCpuBuffer() : 0;

        _iDesc = OsdVertexBufferDescriptor( iOfs, 3, in->GetNumElements() );
        _oDesc = OsdVertexBufferDescriptor( oOfs, 3, out->GetNumElements() );
    }

    /// Unbinds any previously bound vertex and varying data buffers.
    void Unbind() {

        _iBuffer = _oBuffer = 0;
        _iDesc.Reset();
        _oDesc.Reset();
    }

    /// Returns the vector of patch arrays
    const FarPatchTables::PatchArrayVector & GetPatchArrayVector() const {
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

    /// Returns an OsdVertexDescriptor for the input vertex data
    OsdVertexBufferDescriptor const & GetInputVertexDescriptor() const {
        return _iDesc;
    }

    /// Returns an OsdVertexDescriptor for the buffer where the normals data
    /// will be stored
    OsdVertexBufferDescriptor const & GetOutputVertexDescriptor() const {
        return _oDesc;
    }

protected:
    // Constructor
    explicit OsdCpuSmoothNormalContext(FarPatchTables const *patchTables);

private:

    // Topology data for a mesh
    FarPatchTables::PatchArrayVector     _patchArrays;    // patch descriptor for each patch in the mesh
    FarPatchTables::PTable               _patches;        // patch control vertices

    OsdVertexBufferDescriptor _iDesc,
                              _oDesc;

    float * _iBuffer,
          * _oBuffer;

};


}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OSD_CPU_SMOOTHNORMAL_CONTEXT_H
