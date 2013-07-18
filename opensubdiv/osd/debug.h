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
#ifndef OSD_DEBUG_H
#define OSD_DEBUG_H

#ifdef OSD_DEBUG_BUILD

//  XXX: put gl includes into OPENSUBDIV_HAS_GL
#include <GL/gl.h>
#include <stdio.h>
#endif

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

#ifdef OSD_DEBUG_BUILD

#define OSD_DEBUG(...) \
    printf(__VA_ARGS__);

#define OSD_DEBUG_CHECK_GL_ERROR(...) \
    if (GLuint err = glGetError()) {   \
      printf("GL error %x :", err); \
      printf(__VA_ARGS__); \
    }

#else

#define OSD_DEBUG(...)
#define OSD_DEBUG_CHECK_GL_ERROR(...)

#endif

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif // OSD_DEBUG_H
