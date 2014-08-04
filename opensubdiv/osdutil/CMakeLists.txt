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
include_directories("${PROJECT_SOURCE_DIR}/opensubdiv")

#-------------------------------------------------------------------------------
# source & headers

set(PUBLIC_HEADER_FILES
    adaptiveEvaluator.h
    batch.h
    drawItem.h
    drawController.h
    evaluator_capi.h    
    mesh.h
    multiMeshFactory.h
    patchPartitioner.h
    refiner.h
    topology.h
    uniformEvaluator.h
    vertexSplit.h
)

add_library(osdutil
    adaptiveEvaluator.h
    adaptiveEvaluator.cpp
    evaluator_capi.h
    evaluator_capi.cpp            
    mesh.h
    mesh.cpp
    refiner.h
    refiner.cpp
    topology.h
    topology.cpp
    uniformEvaluator.h
    uniformEvaluator.cpp            
    ${INC_FILES}
)

set_target_properties(osdutil PROPERTIES OUTPUT_NAME osdutil CLEAN_DIRECT_OUTPUT 1)

target_link_libraries(
    osdutil    
    "${PLATFORM_LIBRARIES}"
)
        
                
set(DOXY_HEADER_FILES ${PUBLIC_HEADER_FILES})

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
# OpenCL code & dependencies

set(OPENCL_PUBLIC_HEADER_FILES
    batchCL.h
)

if ( OPENCL_FOUND )
    list(APPEND PUBLIC_HEADER_FILES ${OPENCL_PUBLIC_HEADER_FILES} )
endif()

list(APPEND DOXY_HEADER_FILES ${OPENCL_PUBLIC_HEADER_FILES} )

#-------------------------------------------------------------------------------

_add_doxy_headers( "${DOXY_HEADER_FILES}" )

install(
    FILES 
        ${PUBLIC_HEADER_FILES}
    DESTINATION
        "${CMAKE_INCDIR_BASE}/osdutil"
    PERMISSIONS
        OWNER_READ
        GROUP_READ
        WORLD_READ )

install( TARGETS osdutil DESTINATION "${CMAKE_LIBDIR_BASE}")
                  
#-------------------------------------------------------------------------------
