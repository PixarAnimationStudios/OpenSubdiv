#
#     Copyright (C) Pixar. All rights reserved.
#
#     This license governs use of the accompanying software. If you
#     use the software, you accept this license. If you do not accept
#     the license, do not use the software.
#
#     1. Definitions
#     The terms "reproduce," "reproduction," "derivative works," and
#     "distribution" have the same meaning here as under U.S.
#     copyright law.  A "contribution" is the original software, or
#     any additions or changes to the software.
#     A "contributor" is any person or entity that distributes its
#     contribution under this license.
#     "Licensed patents" are a contributor's patent claims that read
#     directly on its contribution.
#
#     2. Grant of Rights
#     (A) Copyright Grant- Subject to the terms of this license,
#     including the license conditions and limitations in section 3,
#     each contributor grants you a non-exclusive, worldwide,
#     royalty-free copyright license to reproduce its contribution,
#     prepare derivative works of its contribution, and distribute
#     its contribution or any derivative works that you create.
#     (B) Patent Grant- Subject to the terms of this license,
#     including the license conditions and limitations in section 3,
#     each contributor grants you a non-exclusive, worldwide,
#     royalty-free license under its licensed patents to make, have
#     made, use, sell, offer for sale, import, and/or otherwise
#     dispose of its contribution in the software or derivative works
#     of the contribution in the software.
#
#     3. Conditions and Limitations
#     (A) No Trademark License- This license does not grant you
#     rights to use any contributor's name, logo, or trademarks.
#     (B) If you bring a patent claim against any contributor over
#     patents that you claim are infringed by the software, your
#     patent license from such contributor to the software ends
#     automatically.
#     (C) If you distribute any portion of the software, you must
#     retain all copyright, patent, trademark, and attribution
#     notices that are present in the software.
#     (D) If you distribute any portion of the software in source
#     code form, you may do so only under this license by including a
#     complete copy of this license with your distribution. If you
#     distribute any portion of the software in compiled or object
#     code form, you may only do so under a license that complies
#     with this license.
#     (E) The software is licensed "as-is." You bear the risk of
#     using it. The contributors give no express warranties,
#     guarantees or conditions. You may have additional consumer
#     rights under your local laws which this license cannot change.
#     To the extent permitted under your local laws, the contributors
#     exclude the implied warranties of merchantability, fitness for
#     a particular purpose and non-infringement.
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
