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

set(EXAMPLES_COMMON_SOURCE_FILES
    font_image.cpp
    hdr_reader.cpp
    hud.cpp
    objAnim.cpp
    patchColors.cpp
    viewerArgsUtils.cpp
)

set(EXAMPLES_COMMON_HEADER_FILES
    font_image.h
    hdr_reader.h
    hud.h
    objAnim.h
    patchColors.h
    shaderCache.h
    simple_math.h
    stb_image_write.h 
    stopwatch.h
    viewerArgsUtils.h
)

include_directories(
    "${OPENSUBDIV_INCLUDE_DIR}"
    "${CMAKE_CURRENT_BINARY_DIR}"
)

if(OPENCL_FOUND)
    include_directories("${OPENCL_INCLUDE_DIRS}")

    list(APPEND EXAMPLES_COMMON_SOURCE_FILES
        clDeviceContext.cpp
    )
    list(APPEND EXAMPLES_COMMON_HEADER_FILES
        clDeviceContext.h
    )
endif()

if(CUDA_FOUND)
    include_directories("${CUDA_INCLUDE_DIRS}")

    list(APPEND EXAMPLES_COMMON_SOURCE_FILES
        cudaDeviceContext.cpp
    )
    list(APPEND EXAMPLES_COMMON_HEADER_FILES
        cudaDeviceContext.h
    )
endif()

if(DEFINED OPENGL_LOADER_INCLUDE_DIRS)
    include_directories(${OPENGL_LOADER_INCLUDE_DIRS})
endif()

if(OPENGL_FOUND AND GLFW_FOUND)

    list(APPEND EXAMPLES_COMMON_GL_SOURCE_FILES
        glControlMeshDisplay.cpp
        glHud.cpp
        glUtils.cpp
        glShaderCache.cpp
    )

    list(APPEND EXAMPLES_COMMON_GL_HEADER_FILES
        glControlMeshDisplay.h
        glHud.h
        glUtils.h
        glShaderCache.h
    )

    if (GLFW_FOUND)
        include_directories("${GLFW_INCLUDE_DIR}")
    endif()

    add_library(examples_common_gl_obj
        OBJECT
            ${EXAMPLES_COMMON_SOURCE_FILES}
            ${EXAMPLES_COMMON_HEADER_FILES}
            ${EXAMPLES_COMMON_GL_SOURCE_FILES}
            ${EXAMPLES_COMMON_GL_HEADER_FILES}
    )

    set_target_properties(examples_common_gl_obj
        PROPERTIES
            FOLDER "examples"
    )

endif()


if(DXSDK_FOUND)

    list(APPEND EXAMPLES_COMMON_DX11_SOURCE_FILES
        d3d11ControlMeshDisplay.cpp
        d3d11Hud.cpp
        d3d11Utils.cpp
        d3d11ShaderCache.cpp
    )

    list(APPEND EXAMPLES_COMMON_DX11_HEADER_FILES
        d3d11ControlMeshDisplay.h
        d3d11Hud.h
        d3d11Utils.h
        d3d11ShaderCache.h
    )

    include_directories("${DXSDK_INCLUDE_DIR}")

    add_library(examples_common_dx11_obj
        OBJECT
            ${EXAMPLES_COMMON_SOURCE_FILES}
            ${EXAMPLES_COMMON_HEADER_FILES}
            ${EXAMPLES_COMMON_DX11_SOURCE_FILES}
            ${EXAMPLES_COMMON_DX11_HEADER_FILES}
    )

    set_target_properties(examples_common_dx11_obj
        PROPERTIES
            FOLDER "examples"
    )

endif()

if(METAL_FOUND)
    include_directories("${METAL_INCLUDE_DIR}")
    list(APPEND EXAMPLES_COMMON_MTL_SOURCE_FILES
        mtlControlMeshDisplay.mm
        mtlUtils.mm
        mtlHud.mm
    )
    list(APPEND EXAMPLES_COMMON_MTL_HEADER_FILES
        mtlControlMeshDisplay.h
        mtlUtils.h
        mtlHud.h
    )

    add_library(examples_common_mtl_obj
        OBJECT
            ${EXAMPLES_COMMON_SOURCE_FILES}
            ${EXAMPLES_COMMON_HEADER_FILES}
            ${EXAMPLES_COMMON_MTL_SOURCE_FILES}
            ${EXAMPLES_COMMON_MTL_HEADER_FILES}
    )

    set_target_properties(examples_common_mtl_obj
        PROPERTIES
            FOLDER "examples"
    )

endif()

#-------------------------------------------------------------------------------
if( PTEX_FOUND )
    list(APPEND EXAMPLES_COMMON_PTEX_HEADER_FILES
        ptexMipmapTextureLoader.h
    )
    list(APPEND EXAMPLES_COMMON_PTEX_SOURCE_FILES
        ptexMipmapTextureLoader.cpp
    )
    if( OPENGL_FOUND )
        list(APPEND EXAMPLES_COMMON_PTEX_SOURCE_FILES
            glPtexMipmapTexture.cpp
        )
        list(APPEND EXAMPLES_COMMON_PTEX_HEADER_FILES
            glPtexMipmapTexture.h
        )
    endif()
    if( DXSDK_FOUND )
        list(APPEND EXAMPLES_COMMON_PTEX_SOURCE_FILES
            d3d11PtexMipmapTexture.cpp
        )
        list(APPEND EXAMPLES_COMMON_PTEX_HEADER_FILES
            d3d11PtexMipmapTexture.h
        )
    endif()
    if( METAL_FOUND )
        list(APPEND EXAMPLES_COMMON_PTEX_SOURCE_FILES
            mtlPtexMipmapTexture.mm
        )
        list(APPEND EXAMPLES_COMMON_PTEX_HEADER_FILES
            mtlPtexMipmapTexture.h
        )
    endif()

    set(EXAMPLES_COMMON_PTEX_SHADER_FILES
         glslPtexCommon.glsl
         hlslPtexCommon.hlsl
         mtlPtexCommon.metal
    )

set(EXAMPLES_COMMON_PTEX_INC_FILES )
    if(EXAMPLES_COMMON_PTEX_SHADER_FILES)
        osd_stringify("${EXAMPLES_COMMON_PTEX_SHADER_FILES}"
                      EXAMPLES_COMMON_PTEX_INC_FILES)

        source_group("Shaders" FILES ${EXAMPLES_COMMON_PTEX_SHADER_FILES})
        source_group("Inc" FILES ${EXAMPLES_COMMON_PTEX_INC_FILES})
    endif()

    include_directories( "${PTEX_INCLUDE_DIR}" )

    add_library(examples_common_ptex_obj
        OBJECT
            ${EXAMPLES_COMMON_PTEX_SOURCE_FILES}
            ${EXAMPLES_COMMON_PTEX_HEADER_FILES}
            ${EXAMPLES_COMMON_PTEX_INC_FILES}
    )

    set_target_properties(examples_common_ptex_obj
        PROPERTIES
            FOLDER "examples"
    )

endif()
