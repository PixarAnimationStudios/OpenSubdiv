//
//   Copyright 2020 Pixar
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

#include "glLoader.h"

#include <stdio.h>
#include <stdlib.h>


namespace OpenSubdiv {
namespace internal {
namespace GLLoader {


bool
applicationInitializeGL()
{
#if defined(OSD_USES_INTERNAL_GLAPILOADER)
    // -- GLAPILOADER
    return OpenSubdiv::internal::GLApi::glApiLoad();

#elif defined(OSD_USES_GLEW)
    // -- GLEW
#define CORE_PROFILE
#ifdef CORE_PROFILE
    // this is the only way to initialize GLEW (before GLEW 1.13)
    // correctly under core profile context.
    glewExperimental = true;
#endif
    GLenum status = glewInit();
    if (status != GLEW_OK) {
        printf("Failed to initialize glew. Error = %s\n",
               glewGetErrorString(status));
        return false;
    }
#ifdef CORE_PROFILE
    // clear GL errors which were generated during glewInit()
    glGetError();
#endif
#endif
    return true;
}

bool
libraryInitializeGL()
{
#if defined(OSD_USES_INTERNAL_GLAPILOADER)
    return OpenSubdiv::internal::GLApi::glApiLoad();
#else
    // otherwise do nothing
    return true;
#endif
}


}  // namespace GLLoader
}  // namespace internal
}  // namespace OpenSubdiv
