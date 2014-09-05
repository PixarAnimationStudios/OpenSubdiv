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

#ifndef OSD_ERROR_H
#define OSD_ERROR_H

#include "../version.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

typedef enum {
    OSD_NO_ERROR,
    OSD_INTERNAL_CODING_ERROR,
    OSD_CL_PROGRAM_BUILD_ERROR,
    OSD_CL_KERNEL_CREATE_ERROR,
    OSD_CL_RUNTIME_ERROR,
    OSD_CUDA_GL_ERROR,
    OSD_GL_ERROR,
    OSD_GLSL_COMPILE_ERROR,
    OSD_GLSL_LINK_ERROR,
    OSD_D3D11_COMPILE_ERROR,
    OSD_D3D11_COMPUTE_BUFFER_CREATE_ERROR,
    OSD_D3D11_VERTEX_BUFFER_CREATE_ERROR,
    OSD_D3D11_BUFFER_MAP_ERROR
} ErrorType;



typedef void (*ErrorCallbackFunc)(ErrorType err, const char *message);

/// Sets the error callback function (default is "printf")
///
/// @param func function pointer to the callback function
///
void SetErrorCallback(ErrorCallbackFunc func);

/// Sends an OSD error 
///
/// @param err the error type
///
void Error(ErrorType err);

/// Sends an OSD error with a message
///
/// @param err     the error type
///
/// @param format  the format of the message (followed by arguments)
///
void Error(ErrorType err, const char *format, ...);


/// Sets the warning callback function (default is "printf")
typedef void (*WarningCallbackFunc)(const char *message);

/// Sets the warning callback function (default is "printf")
///
/// @param func function pointer to the callback function
///
void SetWarningCallback(WarningCallbackFunc func);

/// Sends an OSD warning message
///
/// @param format  the format of the message (followed by arguments)
///
void Warning(const char *format, ...);


} // end namespace 

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif // OSD_ERROR_H
