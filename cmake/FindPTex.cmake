#
#   Copyright 2013 Pixar
#
#   Licensed under the Apache License, Version 2.0 (the "Apache License")
#   with the following modification; you may not use this file except in
#   compliance with the Apache License and the following modification to it:
#   Section 6. Trademarks. is deleted and replaced with:
#
#   6. Trademarks. This License does not grant permission to use the trade
#      names, trademarks, service marks, or product names of the Licensor
#      and its affiliates, except as required to comply with Section 4(c) of
#      the License and to reproduce the content of the NOTICE file.
#
#   You may obtain a copy of the Apache License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the Apache License with the above modification is
#   distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#   KIND, either express or implied. See the Apache License for the specific
#   language governing permissions and limitations under the Apache License.
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
        HINTS
            "${PTEX_LOCATION}/include"
            "$ENV{PTEX_LOCATION}/include"
        PATHS
            "$ENV{PROGRAMFILES}/Ptex/include"
            /usr/include
            DOC "The directory where Ptexture.h resides")
    find_library( PTEX_LIBRARY
        NAMES
            Ptex32 Ptex32s Ptex
        HINTS
            "${PTEX_LOCATION}/lib"
            "$ENV{PTEX_LOCATION}/lib"
        PATHS
            "$ENV{PROGRAMFILES}/Ptex/lib"
            /usr/lib
            /usr/lib/w32api
            /usr/local/lib
            /usr/X11R6/lib
            DOC "The Ptex library")
elseif (APPLE)
    find_path( PTEX_INCLUDE_DIR
        NAMES
            Ptexture.h
        HINTS
            "${PTEX_LOCATION}/include"
            "$ENV{PTEX_LOCATION}/include"
        PATHS
            DOC "The directory where Ptexture.h resides")
    if (IOS)
        #IOS needs to link with the static version of ptex
        find_library( PTEX_LIBRARY
            NAMES
                libPtex.a
            PATHS
                "${PTEX_LOCATION}/lib"
                "$ENV{PTEX_LOCATION}/lib"
                DOC "The Ptex Library")
    else ()
        find_library( PTEX_LIBRARY
            NAMES
                Ptex libPtex.a
            PATHS
                "${PTEX_LOCATION}/lib"
                "$ENV{PTEX_LOCATION}/lib"
                DOC "The Ptex Library")
    endif()
else ()
    find_path( PTEX_INCLUDE_DIR
        NAMES
            Ptexture.h
        HINTS
            "${PTEX_LOCATION}/include"
            "${PTEX_LOCATION}/include/wdas"
            "$ENV{PTEX_LOCATION}/include"
            "$ENV{PTEX_LOCATION}/include/wdas"
        PATHS
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
        HINTS
            "${PTEX_LOCATION}/lib"
            "$ENV{PTEX_LOCATION}/lib"
        PATHS
            /usr/lib
            /usr/local/lib
            /usr/openwin/lib
            /usr/X11R6/lib
            DOC "The Ptex library")
endif ()

if (PTEX_INCLUDE_DIR AND EXISTS "${PTEX_INCLUDE_DIR}/PtexVersion.h")
    set (PTEX_VERSION_FILE "${PTEX_INCLUDE_DIR}/PtexVersion.h")
elseif (PTEX_INCLUDE_DIR AND EXISTS "${PTEX_INCLUDE_DIR}/Ptexture.h")    
    set (PTEX_VERSION_FILE "${PTEX_INCLUDE_DIR}/Ptexture.h")
endif()

if (PTEX_VERSION_FILE)

    file(STRINGS "${PTEX_VERSION_FILE}" TMP REGEX "^#define PtexAPIVersion.*$")
    string(REGEX MATCHALL "[0-9]+" API ${TMP})
    
    file(STRINGS "${PTEX_VERSION_FILE}" TMP REGEX "^#define PtexFileMajorVersion.*$")
    string(REGEX MATCHALL "[0-9]+" MAJOR ${TMP})

    file(STRINGS "${PTEX_VERSION_FILE}" TMP REGEX "^#define PtexFileMinorVersion.*$")
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

