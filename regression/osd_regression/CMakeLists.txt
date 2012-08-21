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

include_directories(
    ${PROJECT_SOURCE_DIR}/opensubdiv
    ${GLEW_INCLUDE_DIR}
    ${GLUT_INCLUDE_DIR}
)

set(SOURCE_FILES
    main.cpp
)


add_executable(osd_regression
    ${SOURCE_FILES}
)

#-------------------------------------------------------------------------------
if( PTEX_FOUND )
    list(APPEND SOURCE_FILES
        pTexture.cpp
    )
    list(APPEND PUBLIC_HEADER_FILES
        pTexture.h
    )
    include_directories( ${PTEX_INCLUDE_DIR} )
    list(APPEND PLATFORM_LIBRARIES
        ${PTEX_LIBRARY}
    )
endif()

#-------------------------------------------------------------------------------
if( OPENMP_FOUND )
    if (CMAKE_COMPILER_IS_GNUCXX)
        list(APPEND PLATFORM_LIBRARIES
            gomp
        )
    endif()
endif()

#-------------------------------------------------------------------------------
# GL code & dependencies
# note : (GLSL compute kernels require GL 4.2, which excludes APPLE)
if( OPENGL_FOUND AND GLEW_FOUND AND (NOT APPLE) )
    list(APPEND SOURCE_FILES
        glslDispatcher.cpp
    )
    list(APPEND PUBLIC_HEADER_FILES
        glslDispatcher.h
    )
    list(APPEND KERNEL_FILES
        glslKernel.glsl
    )
    list(APPEND PLATFORM_LIBRARIES
        ${OPENGL_LIBRARY}
        ${GLEW_LIBRARY}
    )
else( OPENGL_FOUND AND APPLE )
    list(APPEND PLATFORM_LIBRARIES
        ${OPENGL_LIBRARY}
    )
endif()

#-------------------------------------------------------------------------------
# OpenCL code & dependencies
if ( OPENCL_FOUND )
    list(APPEND SOURCE_FILES
        clDispatcher.cpp
    )
    list(APPEND PUBLIC_HEADER_FILES
        clDispatcher.h
    )
    list(APPEND KERNEL_FILES
        clKernel.cl
    )
    list(APPEND PLATFORM_LIBRARIES
        ${OPENCL_LIBRARIES}
    )
    include_directories(${OPENCL_INCLUDE_DIRS})
endif()

#-------------------------------------------------------------------------------
# CUDA code & dependencies
if( CUDA_FOUND )
    list(APPEND SOURCE_FILES
        cudaDispatcher.cpp
    )
    list(APPEND PUBLIC_HEADER_FILES
        cudaDispatcher.h
    )
    list(APPEND KERNEL_FILES
        cudaKernel.cu
    )
    if (UNIX)
        list( APPEND CUDA_NVCC_FLAGS -Xcompiler -fPIC )
    endif()
endif()



target_link_libraries(osd_regression
    ${OSD_LINK_TARGET}
    ${OPENGL_LIBRARY}
    ${GLEW_LIBRARY}
    ${GLUT_LIBRARIES}
)
