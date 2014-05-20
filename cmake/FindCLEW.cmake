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

# Try to find CLEW library and include path.
# Once done this will define
#
# CLEW_FOUND
# CLEW_INCLUDE_DIR
# CLEW_LIBRARY
#

include(FindPackageHandleStandardArgs)

if (WIN32)
    find_path( CLEW_INCLUDE_DIR
        NAMES
            clew.h
        PATHS
            "${CLEW_LOCATION}/include"
            "$ENV{CLEW_LOCATION}/include"
            "$ENV{PROGRAMFILES}/CLEW/include"
            "${PROJECT_SOURCE_DIR}/extern/clew/include"
            DOC "The directory where clew.h resides" )

    find_library( CLEW_LIBRARY
        NAMES
            clew CLEW clew32s clew32
        PATHS
            "${CLEW_LOCATION}/lib"
            "$ENV{CLEW_LOCATION}/lib"
            "$ENV{PROGRAMFILES}/CLEW/lib"
            "${PROJECT_SOURCE_DIR}/extern/clew/bin"
            "${PROJECT_SOURCE_DIR}/extern/clew/lib"
            DOC "The CLEW library")
endif ()

if (${CMAKE_HOST_UNIX})
    find_path( CLEW_INCLUDE_DIR
        NAMES
            clew.h
        PATHS
            "${CLEW_LOCATION}/include"
            "$ENV{CLEW_LOCATION}/include"
            /usr/include
            /usr/local/include
            /sw/include
            /opt/local/include
            NO_DEFAULT_PATH
            DOC "The directory where clew.h resides"
    )
    find_library( CLEW_LIBRARY
        NAMES
            CLEW clew
        PATHS
            "${CLEW_LOCATION}/lib"
            "$ENV{CLEW_LOCATION}/lib"
            /usr/lib64
            /usr/lib
            /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}
            /usr/local/lib64
            /usr/local/lib
            /sw/lib
            /opt/local/lib
            NO_DEFAULT_PATH
            DOC "The CLEW library")
endif ()

find_package_handle_standard_args(CLEW
    REQUIRED_VARS
        CLEW_INCLUDE_DIR
        CLEW_LIBRARY
)
