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

# *** mayaPtexViewer ***

set(MAYA_FIND_QUIETLY TRUE)

if(NOT MAYA_FOUND OR NOT PTEX_FOUND)
    message(STATUS
        "Maya or Ptex could not be found, so the OpenSubdiv mayaPtexViwer plugin "
		"will not be available. If you do have Maya installed and see this message, "
        "please add your Maya path to cmake/FindMaya.cmake or set it in "
        "the MAYA_LOCATION environment variable."
    )
    return()
endif()

set(PLATFORM_LIBRARIES
    ${OSD_LINK_TARGET}
    ${OPENGL_LIBRARY}
    ${GLEW_LIBRARY}
)

include_directories(
    ${PROJECT_SOURCE_DIR}/opensubdiv
    ${MAYA_INCLUDE_DIRS}
    ${GLEW_INCLUDE_DIR}
    ${PTEX_INCLUDE_DIR}
)

set(SHADER_FILES
     shader.glsl
)

set(SOURCE_FILES
    OpenSubdivPtexShaderOverride.cpp
    OpenSubdivPtexShader.cpp
    osdPtexMeshData.cpp
    hbrUtil.cpp
)

set(HEADER_FILES
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
endif(UNIX)

if(WIN32)
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
endif(WIN32)

#-------------------------------------------------------------------------------
# CUDA code & dependencies
if( CUDA_FOUND )
    list(APPEND SOURCE_FILES
        cudaUtil.cpp
    )
    include_directories( ${CUDA_INCLUDE_DIRS} 
    )
endif()

#-------------------------------------------------------------------------------
# Shader Stringification
# We want to use preprocessor include directives to include GLSL and OpenCL
# shader source files in cpp files, but since the sources contain newline
# characters we would need raw string literals from C++11 to do this directly.
# To avoid depending on C++11 we instead use a small tool called "line_quote"
# to generate source files that are suitable for direct inclusion.
foreach(shader_file ${SHADER_FILES})

    string(REGEX REPLACE ".*[.](.*)" "\\1" extension ${shader_file})

    string(REGEX REPLACE "(.*)[.].*" "\\1.inc" inc_file ${shader_file})
    list(APPEND INC_FILES ${inc_file})

    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/${inc_file}
        COMMAND stringify ${CMAKE_CURRENT_SOURCE_DIR}/${shader_file}
            ${CMAKE_CURRENT_SOURCE_DIR}/${inc_file}
        DEPENDS stringify ${CMAKE_CURRENT_SOURCE_DIR}/${shader_file}
    )
endforeach()


add_definitions(
    ${PLATFORM_COMPILE_FLAGS}
)

add_library(maya_ptex_plugin SHARED
    ${SOURCE_FILES}
    ${HEADER_FILES}
    ${SHADER_FILES}
    ${INC_FILES}
)

set_target_properties(maya_ptex_plugin
    PROPERTIES
    OUTPUT_NAME "MayaPtexViewer"
    PREFIX "osd"
    SUFFIX ${PLATFORM_PLUGIN_EXTENSION}
    LINK_FLAGS "${PLATFORM_LINK_FLAGS}"
)

target_link_libraries(maya_ptex_plugin
    ${MAYA_Foundation_LIBRARY}
    ${MAYA_OpenMaya_LIBRARY}
    ${MAYA_OpenMayaRender_LIBRARY}
    ${MAYA_OpenMayaUI_LIBRARY}
    ${PLATFORM_LIBRARIES}
)

install(TARGETS maya_ptex_plugin DESTINATION ${CMAKE_PLUGINDIR_BASE})
