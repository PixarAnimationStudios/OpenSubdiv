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

if(PYTHONINTERP_FOUND AND SWIG_FOUND)
    message(STATUS "Python and SWIG found.  Looking for numpy...")
    execute_process(
        COMMAND
        ${PYTHON_EXECUTABLE} -c "import numpy; print numpy.get_include()"
        OUTPUT_VARIABLE NUMPY_INCLUDE_PATH
        RESULT_VARIABLE NUMPY_ERR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NUMPY_ERR)
        message(WARNING "Unable to import numpy.")
    else()
        message(STATUS "Numpy package has been located.")
        set(PYCMD ${PYTHON_EXECUTABLE} setup.py build )
        list(APPEND PYCMD --osddir=${LIBRARY_OUTPUT_PATH} )
        list(APPEND PYCMD --build-platlib=${PROJECT_BINARY_DIR}/python )
        list(APPEND PYCMD --build-temp=${PROJECT_BINARY_DIR}/temp )
        
        # grab all compiler definitions and add '-D'
        get_directory_property( TMP DIRECTORY ${CMAKE_SOURCE_DIR} COMPILE_DEFINITIONS )
        foreach(FLAG ${TMP})
            list(APPEND SWIG_COMPILE_FLAGS "-D${FLAG}")
        endforeach()
        
        # append platform specific compiler flags
        list(APPEND SWIG_COMPILE_FLAGS ${OSD_COMPILER_FLAGS})
        list(APPEND PYCMD --cxxflags="${SWIG_COMPILE_FLAGS}" )
        
        # add Swig -builtin optimization build flag 
        if(SWIG_VERSION VERSION_GREATER 2.0.4)
            list(APPEND SWIG_OPTS "-builtin")
        endif()
        list(APPEND PYCMD --swigopts="${SWIG_OPTS}" )
        
        add_custom_command(
            OUTPUT 
                ${CURRENT_BINARY_DIR}/osd
            COMMAND 
                ${PYCMD}
            WORKING_DIRECTORY 
                ${CMAKE_SOURCE_DIR}/python
            DEPENDS 
                osd_static_cpu 
                osd_dynamic_cpu
            COMMENT 
                "Building Python bindings with distutils"
        )
        add_custom_target(python ALL
            DEPENDS 
                ${CURRENT_BINARY_DIR}/osd
        )
        install(CODE "execute_process(
                          WORKING_DIRECTORY 
                              ../python 
                          COMMAND 
                              ${PYCMD} install --user)"
        )
    endif()
endif()
