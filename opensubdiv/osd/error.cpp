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

#include "../osd/error.h"

#include <stdarg.h>
#include <stdio.h>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

static OsdErrorCallbackFunc errorFunc = 0;

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

void OsdSetErrorCallback(OsdErrorCallbackFunc func) {

    errorFunc = func;
}

void OsdError(OsdErrorType err) {

    if (errorFunc) {
        errorFunc(err, NULL);
    } else {
        printf("%s\n",errors[err]);
    }
}

void OsdError(OsdErrorType err, const char *format, ...) {

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

static OsdWarningCallbackFunc warningFunc = 0;

void OsdSetWarningCallback(OsdWarningCallbackFunc func) {

    warningFunc = func;
}

void OsdWarning(const char *format, ...) {

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

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv
