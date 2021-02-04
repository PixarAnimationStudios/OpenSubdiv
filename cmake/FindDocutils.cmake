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

# - Try to find the restructured text to HTML converter
#
# Once done this will define
#
#  DOCUTILS_FOUND - System has Docutils
#
#  RST2HTML_EXECUTABLE

find_package(PackageHandleStandardArgs)

find_program( RST2HTML_EXECUTABLE
    NAMES
        rst2html.py
        rst2html
    DOC
        "The Python Docutils reStructuredText to HTML converter"
)

if (RST2HTML_EXECUTABLE)
    # Note we only check for a python interpreter and use it for the command
    # on Windows.  It would be cleaner if we could agree that this is the
    # right way to do it for all platforms, or find a way that works for all
    # platforms uniformly.
    if (WIN32)
        find_package(PythonInterp)
        if (NOT PYTHON_EXECUTABLE)
            message(FATAL_ERROR "Could not find Python interpreter, which is required for Docutils")
        endif()
        execute_process(COMMAND ${PYTHON_EXECUTABLE} ${RST2HTML_EXECUTABLE} --version OUTPUT_VARIABLE VERSION_STRING )
    else()
        execute_process(COMMAND ${RST2HTML_EXECUTABLE} --version OUTPUT_VARIABLE VERSION_STRING )
    endif()

    # The above code may fail to find a version. Which will make the REGEX
    # REPLACE below fail and break the build. So we check for an empty
    # VERSION_STRING.
    if (NOT "${VERSION_STRING}" STREQUAL "")

        # find the version
        # ex : rst2html (Docutils 0.6 [release], Python 2.6.6, on linux2)
        string(REGEX MATCHALL "([^\ ]+\ |[^\ ]+$)" VERSION_TOKENS "${VERSION_STRING}")

        # isolate the 3rd. word in the string
        list (GET VERSION_TOKENS 2 VERSION_STRING)

        # remove white space
        string(REGEX REPLACE "[ \t]+$" "" VERSION_STRING ${VERSION_STRING})
        string(REGEX REPLACE "^[ \t]+" "" VERSION_STRING ${VERSION_STRING})

        set(DOCUTILS_VERSION ${VERSION_STRING})

    endif()

endif()

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(Docutils
    REQUIRED_VARS
        RST2HTML_EXECUTABLE
        DOCUTILS_VERSION
    VERSION_VAR
        DOCUTILS_VERSION
)

mark_as_advanced(
    RST2HTML_EXECUTABLE
)
