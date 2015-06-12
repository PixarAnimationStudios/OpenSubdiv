#
#   Copyright 2013 Autodesk, Inc.
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

# *** mayaPolySmooth ***

set(MAYA_FIND_QUIETLY TRUE)

if(NOT MAYA_FOUND)
    message(STATUS
        "Maya could not be found, so the OpenSubdiv mayaPolySmooth plugin "
        "will not be available. If you do have Maya installed and see this message, "
        "please add your Maya path to cmake/FindMaya.cmake or set it in "
        "the MAYA_LOCATION environment variable."
    )
    return()
endif()

set(PLATFORM_LIBRARIES
    "${OSD_LINK_TARGET}"
)

include_directories(
    "${OPENSUBDIV_INCLUDE_DIR}"
    "${MAYA_INCLUDE_DIRS}"
)

set(SOURCE_FILES
    mayaPolySmooth.cpp
)

set(HEADER_FILES
    mayaPolySmooth.h
)

if(UNIX)
    set(PLATFORM_COMPILE_FLAGS
        -D_BOOL
        -DREQUIRE_IOSTREAM
        -DLINUX
    )
    set(PLATFORM_PLUGIN_EXTENSION
        .so
    )
    set(PLATFORM_LINK_FLAGS
    )
elseif(WIN32)
    set(PLATFORM_COMPILE_FLAGS
        /D_AFXDLL
        /DNT_PLUGIN
        /DREQUIRE_IOSTREAM
    )
    set(PLATFORM_PLUGIN_EXTENSION
        .mll
    )
    set(PLATFORM_LINK_FLAGS
        "/export:initializePlugin /export:uninitializePlugin"
    )
endif()

add_definitions(
    ${PLATFORM_COMPILE_FLAGS}
)

add_library(maya_polySmoothNode SHARED
    ${SOURCE_FILES}
    ${HEADER_FILES}
    $<TARGET_OBJECTS:sdc_obj>
    $<TARGET_OBJECTS:vtr_obj>
    $<TARGET_OBJECTS:far_obj>
)

set_target_properties(maya_polySmoothNode
    PROPERTIES
    OUTPUT_NAME "mayaPolySmooth"
    PREFIX ""
    SUFFIX ${PLATFORM_PLUGIN_EXTENSION}
    LINK_FLAGS "${PLATFORM_LINK_FLAGS}"
)

target_link_libraries(maya_polySmoothNode
    "${MAYA_Foundation_LIBRARY}"
    "${MAYA_OpenMaya_LIBRARY}"
    "${MAYA_OpenMayaRender_LIBRARY}"
    "${MAYA_OpenMayaUI_LIBRARY}"
)

install(TARGETS maya_polySmoothNode DESTINATION "${CMAKE_PLUGINDIR_BASE}")

add_custom_target(maya_polySmoothNode_melScripts
        COMMAND
            ${CMAKE_COMMAND} -E copy
            "${CMAKE_CURRENT_SOURCE_DIR}/mayaPolySmooth.mel"
            "${CMAKE_BINARY_DIR}/${CMAKE_LIBDIR_BASE}/mayaPolySmooth.mel"
        DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/mayaPolySmooth.mel"
)

install(
    FILES
        "${CMAKE_BINARY_DIR}/${CMAKE_LIBDIR_BASE}/mayaPolySmooth.mel"
    DESTINATION
        "${CMAKE_PLUGINDIR_BASE}"
)

add_custom_target(mayaPolySmooth
    ALL
    DEPENDS
        maya_polySmoothNode maya_polySmoothNode_melScripts
)
