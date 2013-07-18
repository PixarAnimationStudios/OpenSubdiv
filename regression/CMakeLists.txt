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

add_subdirectory(hbr_regression)

add_subdirectory(far_regression)

if(OPENGL_FOUND AND (GLEW_FOUND OR APPLE) AND GLFW_FOUND)
    add_subdirectory(osd_regression)
else()
    set(MISSING "")

    if (NOT OPENGL_FOUND)
        list(APPEND MISSING OpenGL)
    endif()

    if (NOT GLEW_FOUND)
        list(APPEND MISSING glew)
    endif()

    if (NOT GLFW_FOUND)
        list(APPEND MISSING glfw)
    endif()

    message(WARNING
        "The following libraries could not be found : ${MISSING}.  "
        "The osd regression test will not be available.  "
        "If you have these libraries installed, please specify their "
        "path to cmake (through the GLEW_LOCATION and GLFW_LOCATION "
        "command line arguments or environment variables)."
    )
endif()
