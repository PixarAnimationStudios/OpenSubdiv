#
#     Copyright (C) Pixar. All rights reserved.
#
#     This license governs use of the accompanying software. If you
#     use the software, you accept this license. If you do not accept
#     the license, do not use the software.
#
#     1. Definitions
#     The terms "reproduce," "reproduction," "derivative works," and
#     "distribution" have the same meaning here as under U.S.
#     copyright law.  A "contribution" is the original software, or
#     any additions or changes to the software.
#     A "contributor" is any person or entity that distributes its
#     contribution under this license.
#     "Licensed patents" are a contributor's patent claims that read
#     directly on its contribution.
#
#     2. Grant of Rights
#     (A) Copyright Grant- Subject to the terms of this license,
#     including the license conditions and limitations in section 3,
#     each contributor grants you a non-exclusive, worldwide,
#     royalty-free copyright license to reproduce its contribution,
#     prepare derivative works of its contribution, and distribute
#     its contribution or any derivative works that you create.
#     (B) Patent Grant- Subject to the terms of this license,
#     including the license conditions and limitations in section 3,
#     each contributor grants you a non-exclusive, worldwide,
#     royalty-free license under its licensed patents to make, have
#     made, use, sell, offer for sale, import, and/or otherwise
#     dispose of its contribution in the software or derivative works
#     of the contribution in the software.
#
#     3. Conditions and Limitations
#     (A) No Trademark License- This license does not grant you
#     rights to use any contributor's name, logo, or trademarks.
#     (B) If you bring a patent claim against any contributor over
#     patents that you claim are infringed by the software, your
#     patent license from such contributor to the software ends
#     automatically.
#     (C) If you distribute any portion of the software, you must
#     retain all copyright, patent, trademark, and attribution
#     notices that are present in the software.
#     (D) If you distribute any portion of the software in source
#     code form, you may do so only under this license by including a
#     complete copy of this license with your distribution. If you
#     distribute any portion of the software in compiled or object
#     code form, you may only do so under a license that complies
#     with this license.
#     (E) The software is licensed "as-is." You bear the risk of
#     using it. The contributors give no express warranties,
#     guarantees or conditions. You may have additional consumer
#     rights under your local laws which this license cannot change.
#     To the extent permitted under your local laws, the contributors
#     exclude the implied warranties of merchantability, fitness for
#     a particular purpose and non-infringement.
#
#

#-------------------------------------------------------------------------------
include_directories(
    ${DXSDK_INCLUDE_DIR}
    ${PROJECT_SOURCE_DIR}/opensubdiv
)

if( OPENGL_FOUND )
    include_directories(
        ${OPENGL_INCLUDE_DIR}
        ${GLEW_INCLUDE_DIR}
    )
elseif( OPENGLES_FOUND )
    include_directories(
        ${OPENGLES_INCLUDE_DIR}
    )
endif()

#-------------------------------------------------------------------------------
# source & headers
set(CPU_SOURCE_FILES
    cpuKernel.cpp
    cpuComputeController.cpp
    cpuComputeContext.cpp
    cpuEvalLimitContext.cpp
    cpuEvalLimitController.cpp
    cpuEvalLimitKernel.cpp
    cpuVertexBuffer.cpp
    error.cpp
    evalLimitContext.cpp
    drawContext.cpp
    drawRegistry.cpp
)

set(GPU_SOURCE_FILES
)

set(KERNEL_FILES
)

set(INC_FILES
)

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
    cpuVertexBuffer.h
    error.h
    evalLimitContext.h
    mesh.h
    nonCopyable.h
    drawContext.h
    drawRegistry.h
    vertex.h
    vertexDescriptor.h
)

#-------------------------------------------------------------------------------
# platform dependent tweaks
if(APPLE OR UNIX)
    set(PLATFORM_COMPILE_FLAGS
        -fPIC
    )
elseif(WIN32)

endif()

add_definitions(
    ${PLATFORM_COMPILE_FLAGS}
)

#-------------------------------------------------------------------------------
if( PTEX_FOUND )
    list(APPEND CPU_SOURCE_FILES
        ptexTextureLoader.cpp
    )
    if( OPENGL_FOUND )
        list(APPEND GPU_SOURCE_FILES
            glPtexTexture.cpp
        )
        list(APPEND PUBLIC_HEADER_FILES
            glPtexTexture.h
        )
    endif()
    if( DXSDK_FOUND )
        list(APPEND GPU_SOURCE_FILES
            d3d11PtexTexture.cpp
        )
        list(APPEND PUBLIC_HEADER_FILES
            d3d11PtexTexture.h
        )
    endif()
    include_directories( ${PTEX_INCLUDE_DIR} )
    list(APPEND PLATFORM_LIBRARIES
        ${PTEX_LIBRARY}
    )
endif()

#-------------------------------------------------------------------------------
if( OPENMP_FOUND )
    list(APPEND CPU_SOURCE_FILES
        ompKernel.cpp
        ompComputeController.cpp
    )
    list(APPEND PUBLIC_HEADER_FILES
        ompKernel.h
        ompComputeController.h
    )   
    if (CMAKE_COMPILER_IS_GNUCXX)
        list(APPEND PLATFORM_LIBRARIES
            gomp
        )
    endif()
endif()

#-------------------------------------------------------------------------------
if( GCD_FOUND )
    list(APPEND CPU_SOURCE_FILES
        gcdKernel.cpp
        gcdComputeController.cpp
    )
    list(APPEND PUBLIC_HEADER_FILES
        gcdKernel.h
        gcdComputeController.h
    )   
endif()

#-------------------------------------------------------------------------------
# GL code & dependencies
if( OPENGL_FOUND OR OPENGLES_FOUND )
    list(APPEND GPU_SOURCE_FILES
        cpuGLVertexBuffer.cpp
        glDrawContext.cpp
        glDrawRegistry.cpp
        glVertexBuffer.cpp
    )
    list(APPEND PUBLIC_HEADER_FILES
        cpuGLVertexBuffer.h
        glDrawContext.h
        glDrawRegistry.h
        glVertexBuffer.h
        glMesh.h
    )
    if ( OPENGL_FOUND )
        list(APPEND KERNEL_FILES
            glslPatchCommon.glsl
            glslPatchRegular.glsl
            glslPatchBoundary.glsl
            glslPatchCorner.glsl
            glslPatchGregory.glsl
            glslPatchBoundaryGregory.glsl
            glslPatchTransition.glsl
        )
    endif()
    list(APPEND PLATFORM_LIBRARIES
        ${OPENGL_LIBRARY}
        ${OPENGLES_LIBRARIES}
    )
endif()

#-------------------------------------------------------------------------------
# OpenGL 4.2 dependencies
# note : (GLSL transform feedback kernels require GL 4.2)
if( OPENGL_4_2_FOUND )
    list(APPEND GPU_SOURCE_FILES
        glslTransformFeedbackComputeController.cpp
        glslTransformFeedbackComputeContext.cpp
        glslTransformFeedbackKernelBundle.cpp
    )
    list(APPEND PUBLIC_HEADER_FILES
        glslTransformFeedbackComputeContext.h
        glslTransformFeedbackComputeController.h
        glslTransformFeedbackKernelBundle.h
    )
    list(APPEND KERNEL_FILES
        glslTransformFeedbackKernel.glsl
    )
    list(APPEND PLATFORM_LIBRARIES
        ${GLEW_LIBRARY}
    )
endif()

#-------------------------------------------------------------------------------
# OpenGL 4.3 dependencies
# note : (GLSL compute shader kernels require GL 4.3)
if( OPENGL_4_3_FOUND )
    list(APPEND GPU_SOURCE_FILES
        glslComputeController.cpp
        glslComputeContext.cpp
        glslKernelBundle.cpp
    )
    list(APPEND PUBLIC_HEADER_FILES
        glslComputeContext.h
        glslComputeController.h
        glslKernelBundle.h
    )
    list(APPEND KERNEL_FILES
        glslComputeKernel.glsl
    )
    list(APPEND PLATFORM_LIBRARIES
        ${GLEW_LIBRARY}
    )
endif()

#-------------------------------------------------------------------------------
# DX11 code & dependencies
if( DXSDK_FOUND )
    list(APPEND GPU_SOURCE_FILES
        cpuD3D11VertexBuffer.cpp
        d3d11ComputeContext.cpp
        d3d11ComputeController.cpp
        d3d11DrawContext.cpp
        d3d11DrawRegistry.cpp
        d3d11KernelBundle.cpp
        d3d11VertexBuffer.cpp
    )
    list(APPEND PUBLIC_HEADER_FILES
        cpuD3D11VertexBuffer.h
        d3d11ComputeContext.h
        d3d11ComputeController.h
        d3d11DrawContext.h
        d3d11DrawRegistry.h
        d3d11KernelBundle.h
        d3d11VertexBuffer.h
        d3d11Mesh.h
    )
    list(APPEND KERNEL_FILES
        hlslComputeKernel.hlsl
        hlslPatchCommon.hlsl
        hlslPatchRegular.hlsl
        hlslPatchBoundary.hlsl
        hlslPatchCorner.hlsl
        hlslPatchGregory.hlsl
        hlslPatchBoundaryGregory.hlsl
        hlslPatchTransition.hlsl
    )
    list(APPEND PLATFORM_LIBRARIES
        ${DXSDK_LIBRARIES}
    )
endif()

#-------------------------------------------------------------------------------
# OpenCL code & dependencies
if ( OPENCL_FOUND )
    list(APPEND GPU_SOURCE_FILES
        clComputeController.cpp
        clComputeContext.cpp
        clKernelBundle.cpp
        clVertexBuffer.cpp
    )
    list(APPEND PUBLIC_HEADER_FILES
        clComputeContext.h
        clComputeController.h
        clVertexBuffer.h
    )
    list(APPEND KERNEL_FILES
        clKernel.cl
    )
    list(APPEND PLATFORM_LIBRARIES
        ${OPENCL_LIBRARIES}
    )
    include_directories( ${OPENCL_INCLUDE_DIRS} )
    if ( OPENGL_FOUND )
        list(APPEND GPU_SOURCE_FILES
            clGLVertexBuffer.cpp
        )
        list(APPEND PUBLIC_HEADER_FILES
            clGLVertexBuffer.h
        )
    endif()
endif()

#-------------------------------------------------------------------------------
# CUDA code & dependencies
if( CUDA_FOUND )
    list(APPEND GPU_SOURCE_FILES
        cudaComputeController.cpp
        cudaComputeContext.cpp
        cudaVertexBuffer.cpp
    )
    list(APPEND PUBLIC_HEADER_FILES
        cudaComputeContext.h
        cudaComputeController.h
        cudaVertexBuffer.h
    )
    list(APPEND KERNEL_FILES
        cudaKernel.cu
    )
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


#-------------------------------------------------------------------------------
# Kernel Stringification
# We want to use preprocessor include directives to include GLSL and OpenCL
# kernel source files in cpp files, but since the sources contain newline
# characters we would need raw string literals from C++11 to do this directly.
# To avoid depending on C++11 we instead use a small tool called "line_quote"
# to generate source files that are suitable for direct inclusion.
foreach(kernel_file ${KERNEL_FILES})

    string(REGEX REPLACE ".*[.](.*)" "\\1" extension ${kernel_file})

    if(NOT ${extension} STREQUAL "cu")

        string(REGEX REPLACE "(.*)[.].*" "\\1.inc" inc_file ${kernel_file})
        list(APPEND INC_FILES ${inc_file})

        add_custom_command(
            OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/${inc_file}
            COMMAND stringify ${CMAKE_CURRENT_SOURCE_DIR}/${kernel_file}
                ${CMAKE_CURRENT_SOURCE_DIR}/${inc_file}
            DEPENDS stringify ${CMAKE_CURRENT_SOURCE_DIR}/${kernel_file}
        )

    endif()
endforeach()

#-------------------------------------------------------------------------------
source_group("Kernels" FILES ${KERNEL_FILES})

source_group("Inc" FILES ${INC_FILES})

_add_possibly_cuda_library(osd_static_cpu STATIC
    ${CPU_SOURCE_FILES}
    ${PRIVATE_HEADER_FILES}
    ${PUBLIC_HEADER_FILES}
)
set_target_properties(osd_static_cpu PROPERTIES OUTPUT_NAME osdCPU CLEAN_DIRECT_OUTPUT 1)

target_link_libraries(osd_static_cpu
    ${PLATFORM_LIBRARIES}
)

_add_possibly_cuda_library(osd_static_gpu STATIC
    ${GPU_SOURCE_FILES}
    ${PRIVATE_HEADER_FILES}
    ${PUBLIC_HEADER_FILES}
    ${KERNEL_FILES}
    ${INC_FILES}
)
set_target_properties(osd_static_gpu PROPERTIES OUTPUT_NAME osdGPU CLEAN_DIRECT_OUTPUT 1)

target_link_libraries(osd_static_gpu
    ${PLATFORM_LIBRARIES}
)

install( TARGETS osd_static_cpu DESTINATION ${CMAKE_LIBDIR_BASE} )
install( TARGETS osd_static_gpu DESTINATION ${CMAKE_LIBDIR_BASE} )

if (NOT WIN32 AND NOT IOS)
    _add_possibly_cuda_library(osd_dynamic_cpu SHARED
        ${CPU_SOURCE_FILES}
        ${PRIVATE_HEADER_FILES}
        ${PUBLIC_HEADER_FILES}
    )

    set_target_properties(osd_dynamic_cpu PROPERTIES OUTPUT_NAME osdCPU CLEAN_DIRECT_OUTPUT 1)

    target_link_libraries(osd_dynamic_cpu
        ${PLATFORM_LIBRARIES}
    )

    _add_possibly_cuda_library(osd_dynamic_gpu SHARED
        ${GPU_SOURCE_FILES}
        ${PRIVATE_HEADER_FILES}
        ${PUBLIC_HEADER_FILES}
        ${KERNEL_FILES}
        ${INC_FILES}
    )

    set_target_properties(osd_dynamic_gpu PROPERTIES OUTPUT_NAME osdGPU CLEAN_DIRECT_OUTPUT 1)

    target_link_libraries(osd_dynamic_gpu
        osd_dynamic_cpu
        ${PLATFORM_LIBRARIES}
    )
    
    install( TARGETS osd_dynamic_cpu DESTINATION ${CMAKE_LIBDIR_BASE} )
    install( TARGETS osd_dynamic_gpu DESTINATION ${CMAKE_LIBDIR_BASE} )
endif()


install( FILES ${PUBLIC_HEADER_FILES}
         DESTINATION ${CMAKE_INCDIR_BASE}/osd
         PERMISSIONS OWNER_READ GROUP_READ WORLD_READ )


if (ANDROID)
    install( FILES Android.mk 
             DESTINATION ${LIBRARY_OUTPUT_PATH_ROOT}
             PERMISSIONS OWNER_READ GROUP_READ WORLD_READ )
endif()

#-------------------------------------------------------------------------------
