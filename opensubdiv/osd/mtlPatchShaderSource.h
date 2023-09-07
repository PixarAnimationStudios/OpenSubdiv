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

#ifndef OPENSUBDIV3_OSD_MTL_PATCH_SHADER_SOURCE_H
#define OPENSUBDIV3_OSD_MTL_PATCH_SHADER_SOURCE_H

#import "../version.h"

#import "../far/patchDescriptor.h"

#import <string>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

/// \brief Provides shader source which can be used by client code.
class MTLPatchShaderSource {
public:
    /// \brief Returns shader source which can be used to evaluate
    /// position and first and second derivatives on piecewise parametric
    /// patches resulting from subdivision refinement.
    static std::string GetPatchBasisShaderSource();

    /// \brief Returns shader source which can be used while drawing
    /// piecewise parametric patches resulting from subdivision refinement,
    /// e.g. while using GPU HW tessellation.
    static std::string GetPatchDrawingShaderSource();

    /// \name Alternative methods
    /// \{
    /// These methods return shader source which can be used
    /// while drawing. Unlike the methods above, the source returned
    /// by these methods includes support for legacy patch types along
    /// with dependencies on specific resource bindings and interstage
    /// shader variable declarations.

    static std::string GetCommonShaderSource();

    static std::string GetVertexShaderSource(Far::PatchDescriptor::Type type);

    static std::string GetHullShaderSource(Far::PatchDescriptor::Type type);

    static std::string GetDomainShaderSource(Far::PatchDescriptor::Type type);

    /// These methods are deprecated. Clients should determine the
    /// patch type of a face-varying patch by inspecting the
    /// face-varying patch array descriptors.
    /// \brief Deprecated
    static std::string GetVertexShaderSource(
        Far::PatchDescriptor::Type type,
        Far::PatchDescriptor::Type fvarType);
    static std::string GetHullShaderSource(
        Far::PatchDescriptor::Type type,
        Far::PatchDescriptor::Type fvarType);
    static std::string GetDomainShaderSource(
        Far::PatchDescriptor::Type type,
        Far::PatchDescriptor::Type fvarType);

    /// @}

};

}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;
    
} // end namespace OpenSubdiv

#endif  // OPENSUBDIV3_OSD_MTL_PATCH_SHADER_SOURCE
