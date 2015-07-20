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

# - Try to find Intel's ISPC
# Once done this will define
#
#  ISPC_FOUND - System has ISPC
#  ISPC_DIR - The ISPC directory

# Obtain ISPC directory
if (WIN32)
    #NOT IMPLEMENTED
elseif (APPLE)
    #NOT IMPLEMENTED
else ()
    find_path(ISPC_DIR
        NAMES
            ispc
        PATHS
            ${ISPC_LOCATION}  
        NO_DEFAULT_PATH NO_SYSTEM_ENVIRONMENT_PATH
        DOC "The directory where ISPC reside")
endif ()

if (ISPC_DIR)
    execute_process(COMMAND ${ISPC_DIR}/ispc --version OUTPUT_VARIABLE ISPC_VERSION)
    string(REGEX MATCH "[0-9].[0-9].[0-9]" ISPC_VERSION ${ISPC_VERSION})
endif ()

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(ISPC
    REQUIRED_VARS
        ISPC_DIR
    VERSION_VAR
        ISPC_VERSION
)

mark_as_advanced( ISPC_DIR )

MACRO (ispc_compile)
  
    SET(ISPC_TARGET_DIR ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/osd_ispc_obj.dir)

    SET(ISPC_OBJECTS "")
    
    FOREACH(src ${ARGN})
    
        GET_FILENAME_COMPONENT(fname ${src} NAME_WE)
        
        SET(results "${ISPC_TARGET_DIR}/${fname}.dev.o")
  
        ADD_CUSTOM_COMMAND(
            OUTPUT ${results} ${ISPC_TARGET_DIR}/${fname}_ispc.h
            COMMAND  ${ISPC_DIR}/ispc  
            --pic
            -O1
            --wno-perf
            --woff
            -h ${ISPC_TARGET_DIR}/${fname}_ispc.h
            -MMM  ${ISPC_TARGET_DIR}/${fname}.dev.idep 
            -o ${ISPC_TARGET_DIR}/${fname}.dev.o
            ${CMAKE_CURRENT_SOURCE_DIR}/${src} 
            \;
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${src} 
        )

        SET(ISPC_OBJECTS ${ISPC_OBJECTS} ${results})

    ENDFOREACH()
    
ENDMACRO()

