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

set(EXAMPLES_COMMON_SHADER_FILES
     framebuffer.glsl
     ssao.glsl
)

set(EXAMPLES_COMMON_SOURCE_FILES
    font_image.cpp
    hdr_reader.cpp
    hud.cpp
    objAnim.cpp
    patchColors.cpp
)

set(EXAMPLES_COMMON_HEADER_FILES
    clInit.h
    cudaInit.h
    font_image.h
    hdr_reader.h
    hud.h
    objAnim.h
    patchColors.h
    simple_math.h
    stopwatch.h
)

if( OPENGL_FOUND AND (GLEW_FOUND AND GLFW_FOUND) OR (APPLE AND GLFW_FOUND))

    list(APPEND EXAMPLES_COMMON_SOURCE_FILES
        gl_common.cpp
        gl_framebuffer.cpp
        gl_hud.cpp
    )

    list(APPEND EXAMPLES_COMMON_HEADER_FILES
        gl_common.h
        gl_framebuffer.h
        gl_hud.h
    )

    if (NOT "${OPENGL_INCLUDE_DIR}" STREQUAL "")
        include_directories("${OPENGL_INCLUDE_DIR}")
    endif()

    if (GLEW_FOUND)
        include_directories("${GLEW_INCLUDE_DIR}")
    endif()

endif()


if(DXSDK_FOUND)

    list(APPEND EXAMPLES_COMMON_SOURCE_FILES
        d3d11_hud.cpp
    )

    list(APPEND EXAMPLES_COMMON_HEADER_FILES
        d3d11_hud.h
    )

    include_directories("${DXSDK_INCLUDE_DIR}")

endif()

if( OPENCL_FOUND )
    include_directories("${OPENCL_INCLUDE_DIRS}")
endif()

include_directories(
    "${PROJECT_SOURCE_DIR}/opensubdiv"
    "${CMAKE_CURRENT_BINARY_DIR}"
)

set(INC_FILES )

_stringify("${EXAMPLES_COMMON_SHADER_FILES}" INC_FILES)

source_group("Shaders" FILES ${EXAMPLES_COMMON_SHADER_FILES})
source_group("Inc" FILES ${INC_FILES})

add_library(examples_common_obj
    OBJECT
        ${EXAMPLES_COMMON_SOURCE_FILES}
        ${EXAMPLES_COMMON_HEADER_FILES}
        ${INC_FILES}        
)

