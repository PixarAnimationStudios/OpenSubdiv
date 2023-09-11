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

# *** glEvalLimit ***

include_directories(
    "${OPENSUBDIV_INCLUDE_DIR}"
    "${OPENGL_LOADER_INCLUDE_DIRS}"
    "${GLFW_INCLUDE_DIR}"
)

list(APPEND PLATFORM_LIBRARIES
    "${OSD_LINK_TARGET}"
    "${OPENGL_LOADER_LIBRARIES}"
    "${GLFW_LIBRARIES}"
)

if( OPENCL_FOUND )
    include_directories("${OPENCL_INCLUDE_DIRS}")
endif()

if( TBB_FOUND )
    list(APPEND PLATFORM_LIBRARIES
        TBB::tbb
    )
endif()

osd_add_glfw_executable(glEvalLimit "examples"
    glEvalLimit.cpp
    particles.cpp
    particles.h
    "${INC_FILES}"
    $<TARGET_OBJECTS:regression_common_obj>
    $<TARGET_OBJECTS:regression_far_utils_obj>
    $<TARGET_OBJECTS:examples_common_gl_obj>
)

target_link_libraries(glEvalLimit
    ${PLATFORM_LIBRARIES}
)

install(TARGETS glEvalLimit DESTINATION "${CMAKE_BINDIR_BASE}")
