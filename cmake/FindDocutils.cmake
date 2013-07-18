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

# - Try to find the restructured text to HTML converter
#
# Once done this will define
#
#  DOCUTILS_FOUND - System has Docutils
#
#  RST2HTML_EXECUTALE

find_package(PackageHandleStandardArgs)

find_program( RST2HTML_EXECUTABLE 
    NAMES 
        rst2html.py 
        rst2html
    DOC 
        "The Python Docutils reStructuredText HTML converter"
)

if (RST2HTML_EXECUTABLE)

    set(DOCUTILS_FOUND "YES")

    # find the version
    execute_process( COMMAND ${RST2HTML_EXECUTABLE} --version OUTPUT_VARIABLE VERSION_STRING )

    # ex : rst2html (Docutils 0.6 [release], Python 2.6.6, on linux2)
    string(REGEX MATCHALL "([^\ ]+\ |[^\ ]+$)" VERSION_TOKENS "${VERSION_STRING}")  
    
    # isolate the 3rd. word in the string
    list (GET VERSION_TOKENS 2 VERSION_STRING)
    
    # remove white space
    string(REGEX REPLACE "[ \t]+$" "" VERSION_STRING ${VERSION_STRING})
    string(REGEX REPLACE "^[ \t]+" "" VERSION_STRING ${VERSION_STRING})
    
    set(DOCUTILS_VERSION ${VERSION_STRING})
        
else()

    set(DOCUTILS_FOUND "NO")

endif()

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(Docutils 
    REQUIRED_VARS
        RST2HTML_EXECUTABLE
    VERSION_VAR
        DOCUTILS_VERSION
)

mark_as_advanced(
    RST2HTML_EXECUTABLE
)
