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

if (NOT NO_LIB)

    #---------------------------------------------------------------------------
    include_directories(
        "${PROJECT_SOURCE_DIR}/opensubdiv"
    )

    #---------------------------------------------------------------------------
    # platform dependent tweaks
    if(APPLE OR UNIX)
        set(PLATFORM_COMPILE_FLAGS -fPIC)
    elseif(WIN32)

    endif()

    add_definitions(
        ${PLATFORM_COMPILE_FLAGS}
    )

    #---------------------------------------------------------------------------
    if( PTEX_FOUND )
        include_directories( "${PTEX_INCLUDE_DIR}" )
        list(APPEND PLATFORM_CPU_LIBRARIES
            ${PTEX_LIBRARY}
        )
    endif()

    if( OPENMP_FOUND )
        if (CMAKE_COMPILER_IS_GNUCXX)
            list(APPEND PLATFORM_CPU_LIBRARIES gomp)
        endif()
    endif()

    if( TBB_FOUND )
        include_directories("${TBB_INCLUDE_DIR}")
        list(APPEND PLATFORM_CPU_LIBRARIES
            ${TBB_LIBRARIES}
        )
    endif()

    if(OPENGL_FOUND OR OPENCL_FOUND OR DXSDK_FOUND)
        add_subdirectory(tools/stringify)
    endif()

    if( OPENGL_FOUND )
        if (NOT "${OPENGL_INCLUDE_DIR}" STREQUAL "")
            include_directories("${OPENGL_INCLUDE_DIR}")
        endif()
        if (GLEW_FOUND)
            include_directories("${GLEW_INCLUDE_DIR}")
        endif()
        list(APPEND PLATFORM_GPU_LIBRARIES
            ${OPENGL_LIBRARY}
        )
    elseif( OPENGLES_FOUND )
        include_directories("${OPENGLES_INCLUDE_DIR}")
        list(APPEND PLATFORM_GPU_LIBRARIES
            ${OPENGLES_LIBRARIES}
        )
    endif()

    if( OPENGL_4_2_FOUND OR OPENGL_4_3_FOUND)
        list(APPEND PLATFORM_GPU_LIBRARIES
            ${GLEW_LIBRARY}
        )
    endif()

    if( DXSDK_FOUND )
        include_directories( "${DXSDK_INCLUDE_DIR}" )
        list(APPEND PLATFORM_GPU_LIBRARIES
            ${DXSDK_LIBRARIES}
        )
    endif()

    if ( OPENCL_FOUND )
        include_directories( "${OPENCL_INCLUDE_DIRS}" )
        list(APPEND PLATFORM_GPU_LIBRARIES
            ${OPENCL_LIBRARIES}
        )
    endif()

    if( CUDA_FOUND )
        include_directories( "${CUDA_INCLUDE_DIRS}" )
    endif()



    #---------------------------------------------------------------------------

    add_subdirectory(hbr)

    add_subdirectory(sdc)

    add_subdirectory(vtr)

    add_subdirectory(far)

    add_subdirectory(osd)

    #---------------------------------------------------------------------------

    install( FILES version.h
        DESTINATION
            "${CMAKE_INCDIR_BASE}"
        PERMISSIONS
            OWNER_READ
            GROUP_READ
            WORLD_READ )

    #---------------------------------------------------------------------------

    # Build static libs ------------------------------------
    add_library(osd_static_cpu
        STATIC
        version.cpp
        $<TARGET_OBJECTS:sdc_obj>
        $<TARGET_OBJECTS:vtr_obj>
        $<TARGET_OBJECTS:far_obj>
        $<TARGET_OBJECTS:osd_cpu_obj>
    )
    set_target_properties(osd_static_cpu 
        PROPERTIES 
            OUTPUT_NAME osdCPU 
            CLEAN_DIRECT_OUTPUT 1
            FOLDER "opensubdiv"
    )

    target_link_libraries(osd_static_cpu
        ${PLATFORM_CPU_LIBRARIES}
    )

    install( TARGETS osd_static_cpu DESTINATION "${CMAKE_LIBDIR_BASE}" )

    if( OSD_GPU )
        # this macro uses FindCUDA.cmake to compile .cu kernel files
        # the target then adds the other obj dependencies and include files
        _add_possibly_cuda_library(osd_static_gpu "opensubdiv"
            STATIC
                version.cpp
                $<TARGET_OBJECTS:osd_gpu_obj>
                ${CUDA_KERNEL_FILES}
        )
        set_target_properties(osd_static_gpu PROPERTIES OUTPUT_NAME osdGPU CLEAN_DIRECT_OUTPUT 1)

        target_link_libraries(osd_static_gpu
            ${PLATFORM_CPU_LIBRARIES} ${PLATFORM_GPU_LIBRARIES}
        )

        install( TARGETS osd_static_gpu DESTINATION "${CMAKE_LIBDIR_BASE}" )
    endif()


    # Build dynamic libs  ----------------------------------
    if (NOT WIN32 AND NOT IOS)

        # generate dynamic-link targets

        #---------------------------------------------------
        add_library(osd_dynamic_cpu
            SHARED
            version.cpp
            $<TARGET_OBJECTS:sdc_obj>
            $<TARGET_OBJECTS:vtr_obj>
            $<TARGET_OBJECTS:far_obj>
            $<TARGET_OBJECTS:osd_cpu_obj>
        )

        if (NOT ANDROID)
            set_target_properties(osd_dynamic_cpu
                PROPERTIES
                    OUTPUT_NAME osdCPU
                    CLEAN_DIRECT_OUTPUT 1
                    SOVERSION ${OSD_SONAME}
                )
        else()
            set_target_properties(osd_dynamic_cpu
                PROPERTIES
                    OUTPUT_NAME osdCPU
                    CLEAN_DIRECT_OUTPUT 1
                )
        endif()

        target_link_libraries(osd_dynamic_cpu
            ${PLATFORM_CPU_LIBRARIES}
        )

        install( TARGETS osd_dynamic_cpu LIBRARY DESTINATION "${CMAKE_LIBDIR_BASE}" )

        #---------------------------------------------------
        if( OSD_GPU )
            _add_possibly_cuda_library(osd_dynamic_gpu "opensubdiv"
                SHARED
                version.cpp
                $<TARGET_OBJECTS:osd_gpu_obj>
                ${CUDA_KERNEL_FILES}
            )

            if (NOT ANDROID)
                set_target_properties(osd_dynamic_gpu
                    PROPERTIES
                        OUTPUT_NAME osdGPU
                        CLEAN_DIRECT_OUTPUT 1
                        SOVERSION ${OSD_SONAME}
                    )
            else()
                set_target_properties(osd_dynamic_gpu
                    PROPERTIES
                        OUTPUT_NAME osdGPU
                        CLEAN_DIRECT_OUTPUT 1
                    )
            endif()

            target_link_libraries(osd_dynamic_gpu
                osd_dynamic_cpu
                ${PLATFORM_CPU_LIBRARIES} ${PLATFORM_GPU_LIBRARIES}
            )

            install( TARGETS osd_dynamic_gpu LIBRARY DESTINATION "${CMAKE_LIBDIR_BASE}" )
        endif()

    endif()

endif()

#-------------------------------------------------------------------------------
