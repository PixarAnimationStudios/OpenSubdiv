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

# Try to find PTex library and include path.
# Once done this will define
#
# PTEX_FOUND
# PTEX_INCLUDE_DIR
# PTEX_LIBRARY
#

if (WIN32)
    find_path( PTEX_INCLUDE_DIR
        NAMES
            Ptexture.h
        PATHS
            ${PTEX_LOCATION}/include
            $ENV{PTEX_LOCATION}/include
            $ENV{PROGRAMFILES}/Ptex/include
            /usr/include
            DOC "The directory where Ptexture.h resides")
    find_library( PTEX_LIBRARY
        NAMES
            Ptex32 Ptex32s Ptex
        PATHS
            ${PTEX_LOCATION}/lib
            $ENV{PTEX_LOCATION}/lib
            $ENV{PROGRAMFILES}/Ptex/lib
            /usr/lib
            /usr/lib/w32api
            /usr/local/lib
            /usr/X11R6/lib
            DOC "The Ptex library")
elseif (APPLE)
    find_path( PTEX_INCLUDE_DIR
        NAMES
            Ptexture.h
        PATHS
            ${PTEX_LOCATION}/include
            $ENV{PTEX_LOCATION}/include
            DOC "The directory where Ptexture.h resides")
    find_library( PTEX_LIBRARY
        NAMES
            Ptex libPtex.a
        PATHS
            ${PTEX_LOCATION}/lib
            $ENV{PTEX_LOCATION}/lib
            DOC "The Ptex Library")
else ()
    find_path( PTEX_INCLUDE_DIR
        NAMES
            Ptexture.h
        PATHS
            ${PTEX_LOCATION}/include
            ${PTEX_LOCATION}/include/wdas
            $ENV{PTEX_LOCATION}/include
            $ENV{PTEX_LOCATION}/include/wdas
            /usr/include
            /usr/local/include
            /usr/openwin/share/include
            /usr/openwin/include
            /usr/X11R6/include
            /usr/include/X11
            DOC "The directory where Ptexture.h resides")
    find_library( PTEX_LIBRARY
        NAMES
            Ptex wdasPtex
        PATHS
            ${PTEX_LOCATION}/lib
            $ENV{PTEX_LOCATION}/lib
            /usr/lib
            /usr/local/lib
            /usr/openwin/lib
            /usr/X11R6/lib
            DOC "The Ptex library")
endif ()

if (PTEX_INCLUDE_DIR AND EXISTS "${PTEX_INCLUDE_DIR}/Ptexture.h" )

    file(STRINGS "${PTEX_INCLUDE_DIR}/Ptexture.h" TMP REGEX "^#define PtexAPIVersion.*$")
    string(REGEX MATCHALL "[0-9]+" API ${TMP})
    
    file(STRINGS "${PTEX_INCLUDE_DIR}/Ptexture.h" TMP REGEX "^#define PtexFileMajorVersion.*$")
    string(REGEX MATCHALL "[0-9]+" MAJOR ${TMP})

    file(STRINGS "${PTEX_INCLUDE_DIR}/Ptexture.h" TMP REGEX "^#define PtexFileMinorVersion.*$")
    string(REGEX MATCHALL "[0-9]+" MINOR ${TMP})

    set(PTEX_VERSION ${API}.${MAJOR}.${MINOR})

endif()

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(PTex 
    REQUIRED_VARS
        PTEX_INCLUDE_DIR
        PTEX_LIBRARY
    VERSION_VAR
        PTEX_VERSION
)

mark_as_advanced(
  PTEX_INCLUDE_DIR
  PTEX_LIBRARY
)

