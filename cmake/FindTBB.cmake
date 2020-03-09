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

# - Try to find Intel's Threading Building Blocks
# Once done this will define
#
#  TBB_FOUND - System has OPENCL
#  TBB_INCLUDE_DIR - The TBB include directory
#  TBB_LIBRARIES - The libraries needed to use TBB

# Obtain include directory
if (WIN32)
    find_path(TBB_INCLUDE_DIR
        NAMES
            tbb/tbb.h
        HINTS
            "${TBB_LOCATION}/include"
            "$ENV{TBB_LOCATION}/include"
        PATHS
            "$ENV{PROGRAMFILES}/Intel/TBB/include"
            /usr/include
            DOC "The directory where TBB headers reside")
elseif (APPLE)
    find_path(TBB_INCLUDE_DIR
        NAMES
            tbb/tbb.h
        HINTS
            "${TBB_LOCATION}/include"
            "$ENV{TBB_LOCATION}/include"
        PATHS
            DOC "The directory where TBB headers reside")
else ()
    find_path(TBB_INCLUDE_DIR
        NAMES
            tbb/tbb.h
        HINTS
            "${TBB_LOCATION}/include"
            "$ENV{TBB_LOCATION}/include"
        PATHS
            /usr/include
            /usr/local/include
            /usr/openwin/share/include
            /usr/openwin/include
            DOC "The directory where TBB headers reside")
endif ()

set (TBB_LIB_ARCH "")

if (WIN32)

    if ("${CMAKE_GENERATOR}" MATCHES "[Ww]in64" OR
        "${CMAKE_GENERATOR_PLATFORM}" MATCHES "x64")
        set(WINPATH intel64)
    else ()
        set(WINPATH ia32)
    endif()

    if (MSVC80)
        set(WINPATH "${WINPATH}/vc8")
    elseif (MSVC90)
        set(WINPATH "${WINPATH}/vc9")
    elseif (MSVC10)
        set(WINPATH "${WINPATH}/vc10")
    elseif (MSVC11)
        set(WINPATH "${WINPATH}/vc11")
    elseif (MSVC12)
        set(WINPATH "${WINPATH}/vc12")
    elseif (MSVC14)
        set(WINPATH "${WINPATH}/vc14")
    endif()

    list(APPEND TBB_LIB_ARCH ${WINPATH})

elseif(ANDROID)

    list(APPEND TBB_LIB_ARCH "android")

else()

    if(CMAKE_SIZEOF_VOID_P MATCHES "8")
        list(APPEND TBB_LIB_ARCH "intel64")
    elseif(CMAKE_SIZEOF_VOID_P MATCHES "4")
        list(APPEND TBB_LIB_ARCH "ia32")
    endif()
endif()

# List library files
foreach(TBB_LIB tbb             tbb_debug)

    find_library(TBB_${TBB_LIB}_LIBRARY
        NAMES
            ${TBB_LIB}
        HINTS
            "${TBB_LOCATION}/lib"
            "${TBB_LOCATION}/bin"
            "$ENV{TBB_LOCATION}/lib"
            "$ENV{TBB_LOCATION}/bin"
            "$ENV{PROGRAMFILES}/TBB/lib"
            /usr/lib
            /usr/lib/w32api
            /usr/local/lib
            /usr/X11R6/lib
        PATH_SUFFIXES
            "${TBB_LIB_ARCH}"
            "${TBB_LIB_ARCH}/${TBB_COMPILER}"
            "${TBB_LIB_ARCH}/gcc4.4"
            "${TBB_LIB_ARCH}/gcc4.1"
        DOC "Intel's Threading Building Blocks library")

    if (TBB_${TBB_LIB}_LIBRARY)
        list(APPEND TBB_LIBRARIES ${TBB_${TBB_LIB}_LIBRARY})
    endif()

endforeach()

# Obtain version information
if(TBB_INCLUDE_DIR)

    # Tease the TBB version numbers from the lib headers
    function(parseVersion FILENAME VARNAME)

        set(PATTERN "^#define ${VARNAME}.*$")

        file(STRINGS "${TBB_INCLUDE_DIR}/${FILENAME}" TMP REGEX ${PATTERN})

        string(REGEX MATCHALL "[0-9]+" TMP ${TMP})

        set(${VARNAME} ${TMP} PARENT_SCOPE)

    endfunction()

    if(EXISTS "${TBB_INCLUDE_DIR}/tbb/tbb_stddef.h")
        parseVersion(tbb/tbb_stddef.h TBB_VERSION_MAJOR)
        parseVersion(tbb/tbb_stddef.h TBB_VERSION_MINOR)
    endif()

    if(${TBB_VERSION_MAJOR} OR ${TBB_VERSION_MINOR})
        set(TBB_VERSION "${TBB_VERSION_MAJOR}.${TBB_VERSION_MINOR}")
        set(TBB_VERSION_STRING "${TBB_VERSION}")
        mark_as_advanced(TBB_VERSION)
    endif()

endif(TBB_INCLUDE_DIR)




include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(TBB
    REQUIRED_VARS
        TBB_INCLUDE_DIR
        TBB_LIBRARIES
    VERSION_VAR
        TBB_VERSION
)

mark_as_advanced(
  TBB_INCLUDE_DIR
  TBB_LIBRARIES
)

