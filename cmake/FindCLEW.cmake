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

include (FindPackageHandleStandardArgs)

if(WIN32)
  set(_clew_SEARCH_DIRS
    "${CLEW_LOCATION}/include"
    "$ENV{CLEW_LOCATION}/include"
    "$ENV{PROGRAMFILES}/CLEW/include"
    "${PROJECT_SOURCE_DIR}/extern/clew/include"
  )
else()
  set(_clew_SEARCH_DIRS
      "${CLEW_LOCATION}"
      "$ENV{CLEW_LOCATION}"
      /usr
      /usr/local
      /sw
      /opt/local
      /opt/lib/clew
  )
endif()

find_path(CLEW_INCLUDE_DIR
  NAMES
    clew.h
  HINTS
    ${_clew_SEARCH_DIRS}
  PATH_SUFFIXES
    include
  NO_DEFAULT_PATH
  DOC "The directory where clew.h resides")

find_library(CLEW_LIBRARY
  NAMES
    CLEW clew
  PATHS
    ${_clew_SEARCH_DIRS}
  PATH_SUFFIXES
    lib lib64
  NO_DEFAULT_PATH
  DOC "The CLEW library")

find_package_handle_standard_args(CLEW
    REQUIRED_VARS
        CLEW_INCLUDE_DIR
        CLEW_LIBRARY
)
