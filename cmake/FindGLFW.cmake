#
#     Copyright (C) Pixar. All rights reserved.
#
#     This license governs use of the accompanying software. If you
#     use the software, you accept this license. If you do not accept
#     the license, do not use the software.
#
#     1. Definitions
#     The terms "reproduce," "reproduction," "derivative works," and
#     "distribution" have the same meaning here as under U.S.
#     copyright law.  A "contribution" is the original software, or
#     any additions or changes to the software.
#     A "contributor" is any person or entity that distributes its
#     contribution under this license.
#     "Licensed patents" are a contributor's patent claims that read
#     directly on its contribution.
#
#     2. Grant of Rights
#     (A) Copyright Grant- Subject to the terms of this license,
#     including the license conditions and limitations in section 3,
#     each contributor grants you a non-exclusive, worldwide,
#     royalty-free copyright license to reproduce its contribution,
#     prepare derivative works of its contribution, and distribute
#     its contribution or any derivative works that you create.
#     (B) Patent Grant- Subject to the terms of this license,
#     including the license conditions and limitations in section 3,
#     each contributor grants you a non-exclusive, worldwide,
#     royalty-free license under its licensed patents to make, have
#     made, use, sell, offer for sale, import, and/or otherwise
#     dispose of its contribution in the software or derivative works
#     of the contribution in the software.
#
#     3. Conditions and Limitations
#     (A) No Trademark License- This license does not grant you
#     rights to use any contributor's name, logo, or trademarks.
#     (B) If you bring a patent claim against any contributor over
#     patents that you claim are infringed by the software, your
#     patent license from such contributor to the software ends
#     automatically.
#     (C) If you distribute any portion of the software, you must
#     retain all copyright, patent, trademark, and attribution
#     notices that are present in the software.
#     (D) If you distribute any portion of the software in source
#     code form, you may do so only under this license by including a
#     complete copy of this license with your distribution. If you
#     distribute any portion of the software in compiled or object
#     code form, you may only do so under a license that complies
#     with this license.
#     (E) The software is licensed "as-is." You bear the risk of
#     using it. The contributors give no express warranties,
#     guarantees or conditions. You may have additional consumer
#     rights under your local laws which this license cannot change.
#     To the extent permitted under your local laws, the contributors
#     exclude the implied warranties of merchantability, fitness for
#     a particular purpose and non-infringement.
#
#

# Try to find GLFW library and include path.
# Once done this will define
#
# GLFW_FOUND
# GLFW_INCLUDE_DIR
# GLFW_LIBRARIES
#

if (WIN32)
    if(CYGWIN)
        find_path( GLFW_INCLUDE_DIR GL/glfw.h
          ${GLFW_LOCATION}/include
          $ENV{GLFW_LOCATION}/include
          /usr/include
        )
        find_library( GLFW_glfw_LIBRARY glfw32
          ${GLFW_LOCATION}/lib
          ${GLFW_LOCATION}/lib/x64
          $ENV{GLFW_LOCATION}/lib
          ${OPENGL_LIBRARY_DIR}
          /usr/lib
          /usr/lib/w32api
          /usr/local/lib
          /usr/X11R6/lib
        )
    else()
        find_path( GLFW_INCLUDE_DIR GL/glfw.h
            ${GLFW_LOCATION}/include
            $ENV{GLFW_LOCATION}/include
            ${PROJECT_SOURCE_DIR}/extern/glfw/include
            $ENV{PROGRAMFILES}/GLFW/include
            ${OPENGL_INCLUDE_DIR}
            DOC "The directory where GL/glfw.h resides")
        find_library( GLFW_glfw_LIBRARY
            NAMES glfw32 glfw32s glfw
            PATHS
            ${GLFW_LOCATION}/lib
            ${GLFW_LOCATION}/lib/x64
            ${GLFW_LOCATION}/lib-msvc110
            $ENV{GLFW_LOCATION}/lib
            ${PROJECT_SOURCE_DIR}/extern/glfw/bin
            ${PROJECT_SOURCE_DIR}/extern/glfw/lib
            $ENV{PROGRAMFILES}/GLFW/lib
            ${OPENGL_LIBRARY_DIR}
            DOC "The GLFW library")

    endif()
else ()
    if (APPLE)
        # These values for Apple could probably do with improvement.
        find_path( GLFW_INCLUDE_DIR GL/glfw.h
            ${GLFW_LOCATION}/include
            /usr/local/include
        )
        find_library( GLFW_glfw_LIBRARY glfw
            NAMES glfw
            PATHS
            ${GLFW_LOCATION}/lib
            ${GLFW_LOCATION}/lib/cocoa
            /usr/local/lib
        )
        set(GLFW_cocoa_LIBRARY "-framework Cocoa" CACHE STRING "Cocoa framework for OSX")
        set(GLFW_iokit_LIBRARY "-framework IOKit" CACHE STRING "IOKit framework for OSX")
    else ()
        find_path( GLFW_INCLUDE_DIR GL/glfw.h
            ${GLFW_LOCATION}/include
            $ENV{GLFW_LOCATION}/include
            /usr/include
            /usr/include/GL
            /usr/local/include
            /usr/openwin/share/include
            /usr/openwin/include
            /usr/X11R6/include
            /usr/include/X11
            /opt/graphics/OpenGL/include
            /opt/graphics/OpenGL/contrib/libglfw
        )
        find_library( GLFW_glfw_LIBRARY glfw
            ${GLFW_LOCATION}/lib
            $ENV{GLFW_LOCATION}/lib
            ${GLFW_LOCATION}/lib/x11
            $ENV{GLFW_LOCATION}/lib/x11
            /usr/lib
            /usr/local/lib
            /usr/openwin/lib
            /usr/X11R6/lib
        )
    endif (APPLE)
endif (WIN32)

set( GLFW_FOUND "NO" )

if(GLFW_INCLUDE_DIR)
  if(GLFW_glfw_LIBRARY)
    set( GLFW_LIBRARIES
      ${GLFW_glfw_LIBRARY}
      ${GLFW_cocoa_LIBRARY}
      ${GLFW_iokit_LIBRARY}
    )
    set( GLFW_FOUND "YES" )

    set (GLFW_LIBRARY ${GLFW_LIBRARIES})
    set (GLFW_INCLUDE_PATH ${GLFW_INCLUDE_DIR})

  endif(GLFW_glfw_LIBRARY)
endif(GLFW_INCLUDE_DIR)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(GLFW DEFAULT_MSG
    GLFW_INCLUDE_DIR
    GLFW_LIBRARIES
)

mark_as_advanced(
  GLFW_INCLUDE_DIR
  GLFW_glfw_LIBRARY
  GLFW_cocoa_LIBRARY
)

