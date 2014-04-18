//
//     Copyright (c) 2013 DigitalFish, Inc.
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

#ifndef OSD_CPU_DRAW_CONTEXT_H
#define OSD_CPU_DRAW_CONTEXT_H

#include "../version.h"

#include "../far/mesh.h"
#include "../osd/drawContext.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

/// \brief CPU-only DrawContext class
///
/// OsdCpuDrawContext implements the OSD drawing interface using shared memory.
/// This is useful for off-line geometry processing that is independent from the
/// real-time graphics capabilities.
///
/// Contexts interface the serialized topological data pertaining to the 
/// geometric primitives with the capabilities of the selected discrete 
/// compute device.
///
class OsdCpuDrawContext : public OsdDrawContext
{
public:
    typedef float * VertexBufferBinding;

    /// \brief Create an OsdCpuDrawContext from FarPatchTables
    ///
    /// @param patchTables      a valid set of FarPatchTables
    ///
    /// @param requireFVarData  set to true to enable face-varying data to be 
    ///                         carried over from the Far data structures.
    ///
    static OsdCpuDrawContext * Create(FarPatchTables const * patchTables,
        bool requireFVarData);

    virtual ~OsdCpuDrawContext();

    /// Returns a pointer to the fvar data
    const float * GetFVarDataBuffer() const {
        return !_fvarDataBuffer.empty() ? &_fvarDataBuffer.front() : NULL;
    }

    /// Returns a pointer to the patch control vertices array
    const unsigned int * GetPatchIndexBuffer() const {
        return &_patchIndexBuffer.front();
    }

    /// Returns a pointer to the patch local parameterization data
    const unsigned int * GetPatchParamBuffer() const {
        return &_patchParamBuffer.front();
    }

    /// Returns a pointer to the patch quad offsets data (onlyused by Gregory
    /// patches)
    const unsigned int * GetQuadOffsetsBuffer() const {
        return !_quadOffsetBuffer.empty() ? &_quadOffsetBuffer.front() : NULL;
    }

    /// Returns a pointer to the patch vertex valence data (only used by Gregory
    // patches)
    const int * GetVertexValenceBuffer() const {
        return !_vertexValenceBuffer.empty() ?
            &_vertexValenceBuffer.front() : NULL;
    }

protected:
    OsdCpuDrawContext();

    // allocate buffers from patchTables
    bool create(FarPatchTables const * patchTables, bool requireFVarData);

    std::vector<float> _fvarDataBuffer;

    std::vector<unsigned int> _patchIndexBuffer;

    std::vector<unsigned int> _patchParamBuffer;

    std::vector<unsigned int> _quadOffsetBuffer;

    std::vector<int> _vertexValenceBuffer;
};

} // namespace OPENSUBDIV_VERSION

using namespace OPENSUBDIV_VERSION;

}  // namespace OpenSubdiv

#endif
