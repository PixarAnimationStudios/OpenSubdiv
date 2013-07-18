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

#ifdef OPENSUBDIV_HAS_CUDA

#if not defined(__APPLE__)
    #include <GL/glew.h>
    #if defined(WIN32)
        #include <GL/wglew.h>
    #endif
#endif

#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include <stdio.h>
#include <algorithm>
#include "../common/cudaInit.h"

// needed to split this function just because of 'short2' conflict
// between Maya includes and cuda includes...

void cudaInit() {
    cudaGLSetGLDevice(cutGetMaxGflopsDeviceId());
}

#endif
