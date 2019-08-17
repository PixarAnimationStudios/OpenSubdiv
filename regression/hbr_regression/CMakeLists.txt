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

include_directories("${OPENSUBDIV_INCLUDE_DIR}")

osd_add_possibly_cuda_executable(hbr_regression "regression"
   main.cpp
    $<TARGET_OBJECTS:regression_common_obj>
)

# This is what we "should" be using but we need to escape the backslash and cmake
# regex doesn't allow that apparently
#file(TO_NATIVE_PATH "${PROJECT_SOURCE_DIR}/regression/hbr_regression/baseline/" HBR_BASELINE_DIR)

set(HBR_BASELINE_DIR "${PROJECT_SOURCE_DIR}/regression/hbr_regression/baseline/" )

if (WIN32)
    string(REGEX REPLACE "/" "\\\\\\\\" HBR_BASELINE_DIR "${HBR_BASELINE_DIR}")
endif()

add_definitions(
    -DHBR_BASELINE_DIR="${HBR_BASELINE_DIR}"
)


osd_add_executable(hbr_baseline "regression"
   baseline.cpp
    $<TARGET_OBJECTS:regression_common_obj>
)

install(TARGETS hbr_baseline DESTINATION "${CMAKE_BINDIR_BASE}")
install(TARGETS hbr_regression DESTINATION "${CMAKE_BINDIR_BASE}")
