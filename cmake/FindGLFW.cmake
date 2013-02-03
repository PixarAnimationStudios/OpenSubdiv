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

find_path( GLFW_INCLUDE_DIR 
    NAMES
        GL/glfw.h
        GL/glfw3.h
    PATHS
        ${GLFW_LOCATION}/include
        $ENV{GLFW_LOCATION}/include
        $ENV{PROGRAMFILES}/GLFW/include
        ${OPENGL_INCLUDE_DIR}
        /usr/openwin/share/include
        /usr/openwin/include
        /usr/X11R6/include
        /usr/include/X11
        /opt/graphics/OpenGL/include
        /opt/graphics/OpenGL/contrib/libglfw
        /usr/local/include
        /usr/include/GL
        /usr/include
    DOC 
        "The directory where GL/glfw.h resides"
)

if (WIN32)
    if(CYGWIN)
        find_library( GLFW_glfw_LIBRARY 
            NAMES
                glfw32
            PATHS
                ${GLFW_LOCATION}/lib
                ${GLFW_LOCATION}/lib/x64
                $ENV{GLFW_LOCATION}/lib
                ${OPENGL_LIBRARY_DIR}
                /usr/lib
                /usr/lib/w32api
                /usr/local/lib
                /usr/X11R6/lib
            DOC 
                "The GLFW library"
        )
    else()
        find_library( GLFW_glfw_LIBRARY
            NAMES 
                glfw32 
                glfw32s 
                glfw
                glfw3
            PATHS
                ${GLFW_LOCATION}/lib
                ${GLFW_LOCATION}/lib/x64
                ${GLFW_LOCATION}/lib-msvc110
                $ENV{GLFW_LOCATION}/lib
                ${PROJECT_SOURCE_DIR}/extern/glfw/bin
                ${PROJECT_SOURCE_DIR}/extern/glfw/lib
                $ENV{PROGRAMFILES}/GLFW/lib
                ${OPENGL_LIBRARY_DIR}
            DOC 
                "The GLFW library"
        )
    endif()
else ()
    if (APPLE)
        find_library( GLFW_glfw_LIBRARY glfw
            NAMES 
                glfw
                glfw3
            PATHS
                ${GLFW_LOCATION}/lib
                ${GLFW_LOCATION}/lib/cocoa
                $ENV{GLFW_LOCATION}/lib
                $ENV{GLFW_LOCATION}/lib/cocoa
                /usr/local/lib
        )
        set(GLFW_cocoa_LIBRARY "-framework Cocoa" CACHE STRING "Cocoa framework for OSX")
        set(GLFW_iokit_LIBRARY "-framework IOKit" CACHE STRING "IOKit framework for OSX")
    else ()
        # (*)NIX
        
        find_package(X11 REQUIRED)
        
        if(NOT X11_Xrandr_FOUND)
            message(FATAL_ERROR "Xrandr library not found")
        endif()
        
        find_library( GLFW_glfw_LIBRARY
            NAMES 
                glfw
                glfw3
            PATHS
                ${GLFW_LOCATION}/lib
                $ENV{GLFW_LOCATION}/lib
                ${GLFW_LOCATION}/lib/x11
                $ENV{GLFW_LOCATION}/lib/x11
                /usr/lib
                /usr/local/lib
                /usr/openwin/lib
                /usr/X11R6/lib
            DOC 
                "The GLFW library"
        )
    endif (APPLE)
endif (WIN32)

set( GLFW_FOUND "NO" )

if(GLFW_INCLUDE_DIR)

    if(GLFW_glfw_LIBRARY)
        set( GLFW_LIBRARIES ${GLFW_glfw_LIBRARY} ${GLFW_cocoa_LIBRARY} ${GLFW_iokit_LIBRARY} )
        set( GLFW_FOUND "YES" )
        set (GLFW_LIBRARY ${GLFW_LIBRARIES})
        set (GLFW_INCLUDE_PATH ${GLFW_INCLUDE_DIR})
    endif(GLFW_glfw_LIBRARY)


    # Tease the GLFW_VERSION numbers from the lib headers
    function(parseVersion FILENAME VARNAME)
            
        set(PATTERN "^#define ${VARNAME}.*$")
        
        file(STRINGS "${GLFW_INCLUDE_DIR}/GL/${FILENAME}" TMP REGEX ${PATTERN})
        
        string(REGEX MATCHALL "[0-9]+" TMP ${TMP})
        
        set(${VARNAME} ${TMP} PARENT_SCOPE)
        
    endfunction()


    if(EXISTS "${GLFW_INCLUDE_DIR}/GL/glfw.h")

        parseVersion(glfw.h GLFW_VERSION_MAJOR)
        parseVersion(glfw.h GLFW_VERSION_MINOR)
        parseVersion(glfw.h GLFW_VERSION_REVISION)

    elseif(EXISTS "${GLFW_INCLUDE_DIR}/GL/glfw3.h")

        parseVersion(glfw3.h GLFW_VERSION_MAJOR)
        parseVersion(glfw3.h GLFW_VERSION_MINOR)
        parseVersion(glfw3.h GLFW_VERSION_REVISION)
 
    endif()
     
    if(${GLFW_VERSION_MAJOR} OR ${GLFW_VERSION_MINOR} OR ${GLFW_VERSION_REVISION})
        set(GLFW_VERSION "${GLFW_VERSION_MAJOR}.${GLFW_VERSION_MINOR}.${GLFW_VERSION_REVISION}")
        set(GLFW_VERSION_STRING "${GLFW_VERSION}")
        mark_as_advanced(GLFW_VERSION)
    endif()
    
    # static builds of glfw require Xrandr
    if( NOT WIN32 AND NOT APPLE AND GLFW_FOUND)
        list(APPEND GLFW_LIBRARIES -lXrandr -lXxf86vm)
    endif()
endif(GLFW_INCLUDE_DIR)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(GLFW 
    REQUIRED_VARS
        GLFW_INCLUDE_DIR
        GLFW_LIBRARIES
    VERSION_VAR
        GLFW_VERSION
)

mark_as_advanced(
  GLFW_INCLUDE_DIR
  GLFW_LIBRARIES
  GLFW_glfw_LIBRARY
  GLFW_cocoa_LIBRARY
)


