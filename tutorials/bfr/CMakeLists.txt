#
#   Copyright 2021 Pixar
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
macro(osd_add_bfr_tutorial NAME)

     osd_add_executable(${NAME} "tutorials/bfr"
        ${ARGN}
        $<TARGET_OBJECTS:sdc_obj>
        $<TARGET_OBJECTS:vtr_obj>
        $<TARGET_OBJECTS:far_obj>
        $<TARGET_OBJECTS:bfr_obj>
        $<TARGET_OBJECTS:regression_common_obj>
     )

     install(TARGETS ${NAME} DESTINATION "${CMAKE_BINDIR_BASE}/tutorials")

endmacro()


set(TUTORIALS
    tutorial_1_1
    tutorial_1_2
    tutorial_1_3
    tutorial_1_4
    tutorial_1_5
    tutorial_2_1
    tutorial_2_2
    tutorial_3_1
    tutorial_3_2
)

foreach(tutorial ${TUTORIALS})

    add_subdirectory("${tutorial}")

    list(APPEND TUTORIAL_TARGETS "bfr_${tutorial}")

    add_test(bfr_${tutorial} ${EXECUTABLE_OUTPUT_PATH}/bfr_${tutorial})

endforeach()

add_custom_target(bfr_tutorials DEPENDS ${TUTORIAL_TARGETS})

set_target_properties(bfr_tutorials
    PROPERTIES 
        FOLDER "tutorials/bfr"
)
