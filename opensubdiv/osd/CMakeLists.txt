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

#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# source & headers
set(CPU_SOURCE_FILES
    cpuKernel.cpp
    cpuComputeController.cpp
    cpuComputeContext.cpp
    cpuEvalLimitContext.cpp
    cpuEvalLimitController.cpp
    cpuEvalLimitKernel.cpp
    cpuSmoothNormalContext.cpp
    cpuSmoothNormalController.cpp
    cpuVertexBuffer.cpp
    drawContext.cpp
    drawRegistry.cpp
    error.cpp
    evalLimitContext.cpp
)

set(GPU_SOURCE_FILES )

set(INC_FILES )

set(PRIVATE_HEADER_FILES
    debug.h
    cpuKernel.h
    cpuEvalLimitKernel.h
)

set(PUBLIC_HEADER_FILES
    computeController.h
    cpuComputeContext.h
    cpuComputeController.h
    cpuEvalLimitContext.h
    cpuEvalLimitController.h
    cpuSmoothNormalContext.h
    cpuSmoothNormalController.h
    cpuVertexBuffer.h
    error.h
    evalLimitContext.h
    mesh.h
    nonCopyable.h
    opengl.h
    drawContext.h
    drawRegistry.h
    vertex.h
    vertexDescriptor.h
)

set(DOXY_HEADER_FILES ${PUBLIC_HEADER_FILES})

#-------------------------------------------------------------------------------
set(GL_PTEX_PUBLIC_HEADERS
    glPtexTexture.h
    glPtexMipmapTexture.h
)
set(DX_PTEX_PUBLIC_HEADERS
    d3d11PtexTexture.h
    d3d11PtexMipmapTexture.h
)

if( PTEX_FOUND )
    list(APPEND PUBLIC_HEADER_FILES
        ptexTextureLoader.h
        ptexMipmapTextureLoader.h
    )
    list(APPEND CPU_SOURCE_FILES
        ptexTextureLoader.cpp
        ptexMipmapTextureLoader.cpp
    )
    if( OPENGL_FOUND )
        list(APPEND GPU_SOURCE_FILES
            glPtexTexture.cpp
            glPtexMipmapTexture.cpp
        )
        list(APPEND PUBLIC_HEADER_FILES
            ${GL_PTEX_PUBLIC_HEADERS}
        )
    endif()
    if( DXSDK_FOUND )
        list(APPEND GPU_SOURCE_FILES
            d3d11PtexTexture.cpp
            d3d11PtexMipmapTexture.cpp
        )
        list(APPEND PUBLIC_HEADER_FILES
            ${DX_PTEX_PUBLIC_HEADERS}
        )
    endif()
    include_directories( "${PTEX_INCLUDE_DIR}" )
    list(APPEND PLATFORM_CPU_LIBRARIES
        ${PTEX_LIBRARY}
    )
endif()

list(APPEND DOXY_HEADER_FILES ${GL_PTEX_PUBLIC_HEADERS} ${DX_PTEX_PUBLIC_HEADERS})

#-------------------------------------------------------------------------------
set(OPENMP_PUBLIC_HEADERS
    ompKernel.h
    ompComputeController.h
#    ompEvalStencilsController.h
    ompSmoothNormalController.h
)

if(OPENMP_FOUND )
    list(APPEND CPU_SOURCE_FILES
        ompKernel.cpp
        ompComputeController.cpp
#        ompEvalStencilsController.cpp
        ompSmoothNormalController.cpp
    )

    list(APPEND PUBLIC_HEADER_FILES ${OPENMP_PUBLIC_HEADERS})

    if (CMAKE_COMPILER_IS_GNUCXX)
        list(APPEND PLATFORM_CPU_LIBRARIES gomp)
    endif()
endif()

list(APPEND DOXY_HEADER_FILES ${OPENMP_PUBLIC_HEADERS})

#-------------------------------------------------------------------------------
set(TBB_PUBLIC_HEADERS
    tbbKernel.h
    tbbComputeController.h
#    tbbEvalStencilsController.h
    tbbSmoothNormalController.h
)

if( TBB_FOUND )
    include_directories("${TBB_INCLUDE_DIR}")

    list(APPEND CPU_SOURCE_FILES
        tbbKernel.cpp
        tbbComputeController.cpp
#        tbbEvalStencilsController.cpp
        tbbSmoothNormalController.cpp
    )

    list(APPEND PUBLIC_HEADER_FILES ${TBB_PUBLIC_HEADERS})

    if (CMAKE_COMPILER_IS_GNUCXX)
        list(APPEND PLATFORM_CPU_LIBRARIES gomp)
    endif()

    list(APPEND PLATFORM_CPU_LIBRARIES
        ${TBB_LIBRARIES}
    )
endif()

list(APPEND DOXY_HEADER_FILES ${TBB_PUBLIC_HEADERS})

#-------------------------------------------------------------------------------
# GL code & dependencies
set(GL_PUBLIC_HEADERS
    cpuGLVertexBuffer.h
    glDrawContext.h
    glDrawRegistry.h
    glVertexBuffer.h
    glMesh.h
)

if( OPENGL_FOUND OR OPENGLES_FOUND )
    list(APPEND GPU_SOURCE_FILES
        cpuGLVertexBuffer.cpp
        glDrawContext.cpp
        glDrawRegistry.cpp
        glVertexBuffer.cpp
    )
    list(APPEND PUBLIC_HEADER_FILES ${GL_PUBLIC_HEADERS})
    if ( OPENGL_FOUND )
        list(APPEND KERNEL_FILES
            glslPatchCommon.glsl
            glslPatchBSpline.glsl
            glslPatchGregory.glsl
            glslPatchTransition.glsl
            glslPtexCommon.glsl
        )
    endif()
endif()

list(APPEND DOXY_HEADER_FILES ${GL_PUBLIC_HEADERS})

#-------------------------------------------------------------------------------
# OpenGL 4.2 dependencies
# note : (GLSL transform feedback kernels require GL 4.2)
set(GL_4_2_PUBLIC_HEADERS
    glslTransformFeedbackComputeContext.h
    glslTransformFeedbackComputeController.h
)

if( OPENGL_4_2_FOUND )
    list(APPEND GPU_SOURCE_FILES
        glslTransformFeedbackComputeContext.cpp
        glslTransformFeedbackComputeController.cpp
    )
    list(APPEND PUBLIC_HEADER_FILES ${GL_4_2_PUBLIC_HEADERS})
    list(APPEND KERNEL_FILES
        glslTransformFeedbackKernel.glsl
    )
    list(APPEND PLATFORM_GPU_LIBRARIES
        ${GLEW_LIBRARY}
    )
endif()

list(APPEND DOXY_HEADER_FILES ${GL_4_2_PUBLIC_HEADERS})


#-------------------------------------------------------------------------------
# OpenGL 4.3 dependencies
# note : (GLSL compute shader kernels require GL 4.3)
set(GL_4_3_PUBLIC_HEADERS
        glslComputeContext.h
        glslComputeController.h
)

if( OPENGL_4_3_FOUND )
    list(APPEND GPU_SOURCE_FILES
        glslComputeContext.cpp
        glslComputeController.cpp
    )
    list(APPEND PUBLIC_HEADER_FILES ${GL_4_3_PUBLIC_HEADERS})
    list(APPEND KERNEL_FILES
        glslComputeKernel.glsl
    )
    list(APPEND PLATFORM_GPU_LIBRARIES
        ${GLEW_LIBRARY}
    )
endif()

list(APPEND DOXY_HEADER_FILES ${GL_4_3_PUBLIC_HEADERS})

#-------------------------------------------------------------------------------
# DX11 code & dependencies
set(DXSDK_PUBLIC_HEADERS
    cpuD3D11VertexBuffer.h
    d3d11ComputeContext.h
    d3d11ComputeController.h
    d3d11DrawContext.h
    d3d11DrawRegistry.h
    d3d11VertexBuffer.h
    d3d11Mesh.h
)
if( DXSDK_FOUND )
    list(APPEND GPU_SOURCE_FILES
        cpuD3D11VertexBuffer.cpp
        d3d11ComputeContext.cpp
        d3d11ComputeController.cpp
        d3d11DrawContext.cpp
        d3d11DrawRegistry.cpp
        d3d11VertexBuffer.cpp
    )
    list(APPEND PUBLIC_HEADER_FILES ${DXSDK_PUBLIC_HEADERS})
    list(APPEND KERNEL_FILES
        hlslComputeKernel.hlsl
        hlslPatchCommon.hlsl
        hlslPatchBSpline.hlsl
        hlslPatchGregory.hlsl
        hlslPatchTransition.hlsl
        hlslPtexCommon.hlsl
    )
    list(APPEND PLATFORM_GPU_LIBRARIES
        ${DXSDK_LIBRARIES}
    )
endif()

list(APPEND DOXY_HEADER_FILES ${DXSDK_PUBLIC_HEADERS})

#-------------------------------------------------------------------------------
# OpenCL code & dependencies
set(OPENCL_PUBLIC_HEADERS
    clComputeContext.h
    clComputeController.h
    clVertexBuffer.h
    opencl.h
)

if ( OPENCL_FOUND )
    list(APPEND GPU_SOURCE_FILES
        clComputeContext.cpp
        clComputeController.cpp
        clVertexBuffer.cpp
    )
    list(APPEND PUBLIC_HEADER_FILES ${OPENCL_PUBLIC_HEADERS})
    list(APPEND KERNEL_FILES
        clKernel.cl
    )
    list(APPEND PLATFORM_GPU_LIBRARIES
        ${OPENCL_LIBRARIES}
    )
    include_directories( "${OPENCL_INCLUDE_DIRS}" )
    if ( OPENGL_FOUND )
        list(APPEND GPU_SOURCE_FILES clGLVertexBuffer.cpp)
        list(APPEND PUBLIC_HEADER_FILES clGLVertexBuffer.h)
    endif()

    # OpenCL D3D11 interop needs work...
    #if ( DXSDK_FOUND )
    #    list(APPEND GPU_SOURCE_FILES
    #         clD3D11VertexBuffer.cpp
    #    )
    #    list(APPEND PUBLIC_HEADER_FILES
    #         clD3D11VertexBuffer.h)
    #    )
    #endif()
endif()

list(APPEND DOXY_HEADER_FILES ${OPENCL_PUBLIC_HEADERS})

#-------------------------------------------------------------------------------
# CUDA code & dependencies
set(CUDA_PUBLIC_HEADERS
    cudaComputeContext.h
    cudaComputeController.h
    cudaVertexBuffer.h
)

if( CUDA_FOUND )
    list(APPEND GPU_SOURCE_FILES
        cudaComputeController.cpp
        cudaComputeContext.cpp
        cudaVertexBuffer.cpp
    )
    list(APPEND PUBLIC_HEADER_FILES ${CUDA_PUBLIC_HEADERS})
    list(APPEND CUDA_KERNEL_FILES
        osd/cudaKernel.cu
    )
    set (CUDA_KERNEL_FILES ${CUDA_KERNEL_FILES} PARENT_SCOPE)
    if ( OPENGL_FOUND )
        list(APPEND GPU_SOURCE_FILES
            cudaGLVertexBuffer.cpp
        )
        list(APPEND PUBLIC_HEADER_FILES
            cudaGLVertexBuffer.h
        )
    endif()

    if( DXSDK_FOUND )
        list(APPEND GPU_SOURCE_FILES
            cudaD3D11VertexBuffer.cpp
        )
        list(APPEND PUBLIC_HEADER_FILES
            cudaD3D11VertexBuffer.h
        )
    endif()

    if (UNIX)
        list( APPEND CUDA_NVCC_FLAGS -Xcompiler -fPIC --gpu-architecture compute_11 )
    endif()
endif()

list(APPEND DOXY_HEADER_FILES ${CUDA_PUBLIC_HEADERS})

#-------------------------------------------------------------------------------

_stringify("${KERNEL_FILES}" INC_FILES)

include_directories("${CMAKE_CURRENT_BINARY_DIR}")

#-------------------------------------------------------------------------------
source_group("Kernels" FILES ${KERNEL_FILES})

source_group("Inc" FILES ${INC_FILES})

# Compile objs first for both the CPU and GPU libs -----
add_library(osd_cpu_obj
    OBJECT
        ${CPU_SOURCE_FILES}
        ${PRIVATE_HEADER_FILES}
        ${PUBLIC_HEADER_FILES}
)

add_library(osd_gpu_obj
    OBJECT
        ${GPU_SOURCE_FILES}
        ${PRIVATE_HEADER_FILES}
        ${PUBLIC_HEADER_FILES}
        ${INC_FILES}
)

_add_doxy_headers( "${DOXY_HEADER_FILES}" )

install(
    FILES
        ${PUBLIC_HEADER_FILES}
    DESTINATION
        "${CMAKE_INCDIR_BASE}/osd"
    PERMISSIONS
        OWNER_READ
        GROUP_READ
        WORLD_READ )


if (ANDROID)
    install(
        FILES
            Android.mk
        DESTINATION
            "${LIBRARY_OUTPUT_PATH_ROOT}"
        PERMISSIONS
            OWNER_READ
            GROUP_READ
            WORLD_READ )
endif()

#-------------------------------------------------------------------------------
