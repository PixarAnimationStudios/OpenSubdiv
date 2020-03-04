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

#ifndef OPENSUBDIV3_GLLOADER_H
#define OPENSUBDIV3_GLLOADER_H

/// OpenGL Loading Library Support
///
/// An OpenGL loading library is required in order to load extended GL API
/// function entry points at run-time and to define types, enum values,
/// and function prototypes needed at compile-time.
///
/// There are three components to this library:
///   glLoader.h        -- interface (this header file)
///   glApi.h           -- low-level generated OpenGL API loader
///   khrplatform.h     -- khronos.org abstraction for low-level platform types
///
/// To use:
///   - Include "glloader.h"
///
///   - Initialize from example applications or from the library by calling;
///
///     internal::GLLoader::applicationInitializeGL();
///     -or-
///     internal::GLLoader::libraryInitializeGL();
///
/// There are distinct compile-time and run-time requirements for developing
/// OpenGL software which will build and run in a variety of situations.
///
/// 1) Compile-time definitions of core types
///
/// These are the core data types used as the argument and return value types
/// of GL API functions. In this implementation, they are defined in terms
/// of the types defined by Khronos Group in "khrplatform.h"
///
/// These types are defined in the global namespace.
///
/// 2) Compile-time definitions of enum values
///
/// These are GLenum data values used by the GL API. They are implemented
/// as preprocessor definitions, e.g.
///
///     #define GL_UNIFORM_BUFFER 0x8A0F
///
/// These preprocessor symbols are defined in the global namespace.
///
/// 3) Compile-time definitions for versions and extensions
///
/// The GL API and extension specifications specify standard preprocessor
/// symbols which can be tested at compile-time to protect sections of source
/// code which require a specific version or extension, e.g.
///
///     #if defined(GL_VERSION_4_0)
///         // ... GL_VERSION_4_0 specific code
///     #endif
///
///     #if defined(GL_ARB_tessellation_shader)
///         // ... ARB_tessellation_shader specific code
///     #endif
///
/// These preprocessor symbols are defined in the global namespace.
///
/// 4) Compile-time declarations of function prototypes
///
/// Each GL API function has a typedef for the function prototype and
/// a function pointer (initialized to nullptr) which implements the
/// function, e.g.
///
///     typedef void (GLAPIENTRY * PFNGLBINDBUFFERBASEPROC)(
///                             GLenum  target, GLuint  index, GLuint  buffer);
///     PFNGLBINDBUFFERBASEPROC glBindBufferBase;
///
/// These types and function pointers are defined in an internal namespace.
///
/// 5) Run-time loading of supported functions
///
/// Because a GL program may be compiled and run in different environments,
/// GL API functions must be loaded dynamically at run time. This is taken
/// care of by executing the low-level OpenGL API loader, e.g.
///
///     bool internal::GLLoader::libraryInitializeGL() {
///         ...
///         glBindBufferBase = (PFNGLBINDBUFFERBASEPROC)
///                                 loadFunction("glBindBufferBase");
///         ...
///     }
///
/// The loaded function pointers are defined in an internal namespace.
///
/// 6) Run-time queries for supported versions and extensions
///
/// Finally, a GL program should check for the availability of required
/// features at run-time. Typically, this requires inspection of the
/// strings or string lists returned by calling the glGetString functions.
///
/// This loader library processes these queries when loading the GL API
/// and saves the results in boolean variables which follow a naming
/// convention similar to the preprocessor symbols described above.
///
/// This loader further provides macros which wrap evaluation of these
/// boolean variables in order to help isolate source code from details
/// of the specific naming (prefixes, etc) of these variables, e.g.
///
///     #if defined(GL_VERSION_4_0)
///     if (OSD_OPENGL_HAS(VERSION_4_0)) {
///         // ... GL_VERSION_4_0 specific code
///     }
///     #endif
///
///     #if defined(GL_ARB_tessellation_shader)
///     if (OSD_OPENGL_HAS(ARB_tessellation_shader)) {
///         // ... ARB_tessellation_shader specific code
///     }
///     #endif
///
/// The boolean variables described here are defined in an internal namespace.
///

#if defined(OSD_USES_INTERNAL_GLAPILOADER)
    // -- GLAPILOADER
    #include "glApi.h"

    #define OSD_OPENGL_HAS(token) (GLAPILOADER_GL_##token)

#elif defined(OSD_USES_GLEW)
    // -- GLEW
    #include <GL/glew.h>

    #define OSD_OPENGL_HAS(token) (GLEW_##token)

#endif

namespace OpenSubdiv {
namespace internal {
namespace GLLoader {


// Initialize OpenGL loader library from the application. This is used
// only by examples and tests in this code base.
extern bool applicationInitializeGL();

// Initialize OpenGL loader library from the library. This does nothing
// for external loader libraries like GLEW, since in that case, it is
// the application's responsibility to initialize the loader library.
extern bool libraryInitializeGL();


}  // namespace GLLoader
}  // namespace internal
}  // namespace OpenSubdiv


#if defined(OSD_USES_INTERNAL_GLAPILOADER)
using namespace OpenSubdiv::internal::GLApi;
#endif


#endif  // OPENSUBDIV3_GLLOADER_H
