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

#include "../osd/error.h"

#include <stdarg.h>
#include <stdio.h>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

static ErrorCallbackFunc errorFunc = 0;

static char const * errors[] = {
    "OSD_NO_ERROR",
    "OSD_INTERNAL_CODING_ERROR",
    "OSD_CL_PROGRAM_BUILD_ERROR",
    "OSD_CL_KERNEL_CREATE_ERROR",
    "OSD_CL_RUNTIME_ERROR",
    "OSD_CUDA_GL_ERROR",
    "OSD_GL_ERROR",
    "OSD_GLSL_COMPILE_ERROR",
    "OSD_GLSL_LINK_ERROR",
    "OSD_D3D11_COMPILE_ERROR",
    "OSD_D3D11_COMPUTE_BUFFER_CREATE_ERROR",
    "OSD_D3D11_VERTEX_BUFFER_CREATE_ERROR",
    "OSD_D3D11_BUFFER_MAP_ERROR"
};

void SetErrorCallback(ErrorCallbackFunc func) {

    errorFunc = func;
}

void Error(ErrorType err) {

    if (errorFunc) {
        errorFunc(err, NULL);
    } else {
        printf("%s\n",errors[err]);
    }
}

void Error(ErrorType err, const char *format, ...) {

    char message[10240];
    va_list argptr;
    va_start(argptr, format);
    vsnprintf(message, 10240, format, argptr);
    va_end(argptr);
    
    if (errorFunc) {
        errorFunc(err, message);
    } else {
        printf("%s : %s\n",errors[err], message);
    }
}

static WarningCallbackFunc warningFunc = 0;

void SetWarningCallback(WarningCallbackFunc func) {

    warningFunc = func;
}

void Warning(const char *format, ...) {

    char message[10240];
    va_list argptr;
    va_start(argptr, format);
    vsnprintf(message, 10240, format, argptr);
    va_end(argptr);
    
    if (warningFunc) {
        warningFunc(message);
    } else {
        printf("OSD_WARNING : %s\n", message);
    }
}

} // end namespace 

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv
