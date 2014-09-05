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


# Doxygen documentation
if (DOXYGEN_FOUND)

    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/OpenSubdiv.doxy" "${CMAKE_CURRENT_BINARY_DIR}/OpenSubdiv.doxy" @ONLY)

    add_custom_target(doc_doxy
            ${DOXYGEN_EXECUTABLE} "${CMAKE_CURRENT_BINARY_DIR}/OpenSubdiv.doxy"
        WORKING_DIRECTORY
            "${CMAKE_BINARY_DIR}/public_headers/"
        DEPENDS
            public_headers
        COMMENT
            "Generating API documentation with Doxygen" VERBATIM
    )

    list(APPEND DOC_TARGETS doc_doxy)

    install(
        DIRECTORY
            "${CMAKE_CURRENT_BINARY_DIR}/doxy_html"
        DESTINATION
            "${CMAKE_DOCDIR_BASE}"
    )

else()

    message(WARNING
        "Doxyen was not found : support for Doxygen automated API documentation is disabled.")

endif()

find_package(PythonInterp 2.6)

# ReST - HTML documentation
if (DOCUTILS_FOUND AND PYTHONINTERP_FOUND)

    set(HTML_FILES
         search.html
    )

    set(RST_FILES
        additional_resources.rst
        api_overview.rst
        cmake_build.rst
        code_examples.rst
        dxviewer.rst
        dxptexviewer.rst
        far_overview.rst
        getting_started.rst
        glevallimit.rst
        glfvarviewer.rst
        glpainttest.rst
        glptexviewer.rst
        glsharetopology.rst
        glstencilviewer.rst
        glviewer.rst
        hbr_overview.rst
        intro.rst
        maya_osdpolysmooth.rst
        osd_overview.rst
        release_notes.rst
        release_notes_2x.rst
        sdc_overview.rst
        subdivision_surfaces.rst
        tutorials.rst
        using_osd.rst
        using_osd_compile.rst
        using_osd_hbr.rst
        using_osd_textures.rst
        vtr_overview.rst
    )


    # Configure release number in RST template file

    # Replace '_' with '.'
    string(REGEX REPLACE "(_)" "." RELEASE_STRING ${OpenSubdiv_VERSION})

    # Remove starting 'v' character
    string(REGEX REPLACE "^v" "" RELEASE_STRING ${RELEASE_STRING})

    # Format API version string
    set(RELEASE_STRING "Release ${RELEASE_STRING}")

    # Replace string in navigation bar
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/rst_template.txt"
                   "${CMAKE_CURRENT_BINARY_DIR}/rst_template.txt" )

    # process rst markup files
    foreach(src ${RST_FILES})
        get_filename_component(BASENAME ${src} NAME_WE)

        set(infile "${CMAKE_CURRENT_SOURCE_DIR}/${src}")
        set(outfile "${CMAKE_CURRENT_BINARY_DIR}/${BASENAME}.html")

        add_custom_command(
            OUTPUT
                "${outfile}"
            COMMAND
                "${RST2HTML_EXECUTABLE}"
            ARGS
                --date
                --time
                --no-xml-declaration
                --initial-header-level=3
                --strip-comments
                --template="${CMAKE_CURRENT_BINARY_DIR}/rst_template.txt"
                --stylesheet=css/rst.css
                --link-stylesheet
                "${infile}" "${outfile}"
            DEPENDS
                "${infile}" rst_template.txt nav_template.txt
        )

        add_custom_target(${src} DEPENDS "${outfile}")

        list(APPEND RST_TARGETS ${src})

        install(
            FILES
                "${outfile}"
            DESTINATION
                "${CMAKE_DOCDIR_BASE}"
            PERMISSIONS
                OWNER_READ GROUP_READ WORLD_READ
        )

    endforeach()

    # copy html files to the build area
    foreach(src ${HTML_FILES})
        set(infile "${CMAKE_CURRENT_SOURCE_DIR}/${src}")
        set(outfile "${CMAKE_CURRENT_BINARY_DIR}/${src}")

        add_custom_command(
            OUTPUT
                "${outfile}"
            COMMAND
                "${CMAKE_COMMAND}"
            ARGS
                -E copy "${infile}" "${outfile}"
        )

        add_custom_target( ${src} DEPENDS "${outfile}")

        list(APPEND HTML_TARGETS ${src})

        install(
            FILES
                "${outfile}"
            DESTINATION
                "${CMAKE_DOCDIR_BASE}"
        )
    endforeach()

    # copy the site resources to the build area so that the
    # documentation can be read without an install
    add_custom_target(doc_html_images
        COMMAND
            "${CMAKE_COMMAND}" -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/images" "${CMAKE_CURRENT_BINARY_DIR}/images"
    )

    add_custom_target(doc_html_css
        COMMAND
            ${CMAKE_COMMAND} -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/css" "${CMAKE_CURRENT_BINARY_DIR}/css"
    )

    add_custom_target(doc_tipuesearch
        COMMAND
            ${CMAKE_COMMAND} -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/tipuesearch" "${CMAKE_CURRENT_BINARY_DIR}/tipuesearch"
    )

    # build search index and insert navigation tab
    add_custom_target(search_index
        COMMAND
            "${PYTHON_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/processHtml.py"
            "${CMAKE_CURRENT_BINARY_DIR}"
            "${CMAKE_CURRENT_SOURCE_DIR}/nav_template.txt"
        DEPENDS
            ${HTML_TARGETS}
            ${RST_TARGETS}
            doc_tipuesearch
    )

    add_custom_target(doc_html
        DEPENDS
            search_index
            ${HTML_TARGETS}
            ${RST_TARGETS}
            doc_html_images
            doc_html_css
            doc_tipuesearch
    )

    install(
        DIRECTORY
            "${CMAKE_CURRENT_BINARY_DIR}/tipuesearch"
        DESTINATION
            "${CMAKE_DOCDIR_BASE}"
    )


    install(
        DIRECTORY
            "${CMAKE_CURRENT_BINARY_DIR}/images"
        DESTINATION
            "${CMAKE_DOCDIR_BASE}"
    )

    install(
        DIRECTORY
            "${CMAKE_CURRENT_BINARY_DIR}/css"
        DESTINATION
            "${CMAKE_DOCDIR_BASE}"
    )

    list(APPEND DOC_TARGETS doc_html)
endif()

add_custom_target(doc ALL DEPENDS ${DOC_TARGETS})
