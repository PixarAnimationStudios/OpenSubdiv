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

# *** glViewer ***

set(PLATFORM_LIBRARIES
    ${OSD_LINK_TARGET}
    ${GLFW_LIBRARIES}
)

include_directories(
    ${PROJECT_SOURCE_DIR}/opensubdiv
    ${PROJECT_SOURCE_DIR}/regression
    ${GLFW_INCLUDE_DIR}
)

if ( GLEW_FOUND )
    include_directories(${GLEW_INCLUDE_DIR})
    list(APPEND PLATFORM_LIBRARIES ${GLEW_LIBRARY})
endif()

_add_possibly_cuda_executable(limitEval
    main.cpp
    ../common/font_image.cpp
    ../common/hud.cpp
    ../common/gl_hud.cpp
    ${INC_FILES}
)

target_link_libraries(limitEval
    ${PLATFORM_LIBRARIES}
)

install(TARGETS limitEval DESTINATION ${CMAKE_BINDIR_BASE})
