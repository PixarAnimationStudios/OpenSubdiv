#
#     Copyright 2013 Pixar
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License
#     and the following modification to it: Section 6 Trademarks.
#     deleted and replaced with:
#
#     6. Trademarks. This License does not grant permission to use the
#     trade names, trademarks, service marks, or product names of the
#     Licensor and its affiliates, except as required for reproducing
#     the content of the NOTICE file.
#
#     You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing,
#     software distributed under the License is distributed on an
#     "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
#     either express or implied.  See the License for the specific
#     language governing permissions and limitations under the
#     License.
#

# Try to find GLEW library and include path.
# Once done this will define
#
# GLEW_FOUND
# GLEW_INCLUDE_DIR
# GLEW_LIBRARY
#

include(FindPackageHandleStandardArgs)

if (WIN32)
    find_path( GLEW_INCLUDE_DIR
        NAMES
            GL/glew.h
        PATHS
            ${GLEW_LOCATION}/include
            $ENV{GLEW_LOCATION}/include
            $ENV{PROGRAMFILES}/GLEW/include
            ${PROJECT_SOURCE_DIR}/extern/glew/include
            DOC "The directory where GL/glew.h resides" )

    find_library( GLEW_LIBRARY
        NAMES
            glew GLEW glew32s glew32
        PATHS
            ${GLEW_LOCATION}/lib
            $ENV{GLEW_LOCATION}/lib
            $ENV{PROGRAMFILES}/GLEW/lib
            ${PROJECT_SOURCE_DIR}/extern/glew/bin
            ${PROJECT_SOURCE_DIR}/extern/glew/lib
            DOC "The GLEW library")
endif ()

if (${CMAKE_HOST_UNIX})
    find_path( GLEW_INCLUDE_DIR
        NAMES
            GL/glew.h
        PATHS
            ${GLEW_LOCATION}/include
            $ENV{GLEW_LOCATION}/include
            /usr/include
            /usr/local/include
            /sw/include
            /opt/local/include
            NO_DEFAULT_PATH
            DOC "The directory where GL/glew.h resides"
    )
    find_library( GLEW_LIBRARY
        NAMES
            GLEW glew
        PATHS
            ${GLEW_LOCATION}/lib
            $ENV{GLEW_LOCATION}/lib
            /usr/lib64
            /usr/lib
            /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}
            /usr/local/lib64
            /usr/local/lib
            /sw/lib
            /opt/local/lib
            NO_DEFAULT_PATH
            DOC "The GLEW library")
endif ()

if (GLEW_INCLUDE_DIR AND EXISTS "${GLEW_INCLUDE_DIR}/GL/glew.h")

   file(STRINGS "${GLEW_INCLUDE_DIR}/GL/glew.h" GLEW_4_2 REGEX "^#define GL_VERSION_4_2.*$")
   if (GLEW_4_2)
       set(OPENGL_4_2_FOUND TRUE)
   else ()
       message(WARNING
       "glew-1.7.0 or newer needed for supporting OpenGL 4.2 dependent features"
       )
   endif ()

   file(STRINGS "${GLEW_INCLUDE_DIR}/GL/glew.h" GLEW_4_3 REGEX "^#define GL_VERSION_4_3.*$")
   if (GLEW_4_3)
       SET(OPENGL_4_3_FOUND TRUE)
   else ()
       message(WARNING
       "glew-1.9.0 or newer needed for supporting OpenGL 4.3 dependent features"
       )
   endif ()

endif ()

find_package_handle_standard_args(GLEW
    REQUIRED_VARS
        GLEW_INCLUDE_DIR
        GLEW_LIBRARY
)
