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
            "${OpenSubdiv_BINARY_DIR}/public_headers/"
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

# ReST - HTML documentation
if (DOCUTILS_FOUND AND Python_Interpreter_FOUND)

    set(HTML_FILES
         search.html
    )

    set(RST_FILES
        additional_resources.rst
        api_overview.rst
        bfr_overview.rst
        cmake_build.rst
        code_examples.rst
        compatibility.rst
        contributing.rst
        downloads.rst
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
        hedits.rst
        hbr_overview.rst
        intro.rst
        license.rst
        mod_notes.rst
        mtlviewer.rst
        mtlptexviewer.rst
        osd_overview.rst
        osd_shader_interface.rst
        porting.rst
        references.rst
        release_30.rst
        release_31.rst
        release_32.rst
        release_33.rst
        release_34.rst
        release_35.rst
        release_36.rst
        release_notes.rst
        release_notes_2x.rst
        sdc_overview.rst
        subdivision_surfaces.rst
        tutorials.rst
        using_osd_hbr.rst
        vtr_overview.rst
    )


    # Configure release number in RST template file

    # Replace '_' with '.'
    string(REGEX REPLACE "(_)" "." RELEASE_STRING ${OpenSubdiv_VERSION})

    # Remove starting 'v' character
    string(REGEX REPLACE "^v" "" RELEASE_STRING ${RELEASE_STRING})

    # Format API version string
    set(RELEASE_STRING "${RELEASE_STRING}")

    # Replace string in navigation bar
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/rst_template.txt"
                   "${CMAKE_CURRENT_BINARY_DIR}/rst_template.txt" )

    # Process rst markup files
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

    # Generate tutorials pages
    
    # XXXX manuelk we should push more of this logic into the python script
    set(TUTORIAL_CODE
        hbr/tutorial_0/hbr_tutorial_0.cpp
        hbr/tutorial_1/hbr_tutorial_1.cpp
        hbr/tutorial_2/hbr_tutorial_2.cpp
        far/tutorial_1_1/far_tutorial_1_1.cpp
        far/tutorial_1_2/far_tutorial_1_2.cpp
        far/tutorial_2_1/far_tutorial_2_1.cpp
        far/tutorial_2_2/far_tutorial_2_2.cpp
        far/tutorial_2_3/far_tutorial_2_3.cpp
        far/tutorial_3_1/far_tutorial_3_1.cpp
        far/tutorial_4_1/far_tutorial_4_1.cpp
        far/tutorial_4_2/far_tutorial_4_2.cpp
        far/tutorial_4_3/far_tutorial_4_3.cpp
        far/tutorial_5_1/far_tutorial_5_1.cpp
        far/tutorial_5_2/far_tutorial_5_2.cpp
        far/tutorial_5_3/far_tutorial_5_3.cpp
        bfr/tutorial_1_1/bfr_tutorial_1_1.cpp
        bfr/tutorial_1_2/bfr_tutorial_1_2.cpp
        bfr/tutorial_1_3/bfr_tutorial_1_3.cpp
        bfr/tutorial_1_4/bfr_tutorial_1_4.cpp
        bfr/tutorial_1_5/bfr_tutorial_1_5.cpp
        bfr/tutorial_2_1/bfr_tutorial_2_1.cpp
        bfr/tutorial_2_2/bfr_tutorial_2_2.cpp
        bfr/tutorial_3_1/bfr_tutorial_3_1.cpp
        bfr/tutorial_3_2/bfr_tutorial_3_2.cpp
        osd/tutorial_0/osd_tutorial_0.cpp
    )

    foreach(tutorial ${TUTORIAL_CODE})
        get_filename_component(BASENAME ${tutorial} NAME_WE)

        string(REGEX REPLACE "(/)" "_" TARGET_NAME ${tutorial})
        set(infile "${OpenSubdiv_SOURCE_DIR}/tutorials/${tutorial}")
        set(rstfile "${CMAKE_CURRENT_BINARY_DIR}/${BASENAME}.rst")
        set(htmlfile "${CMAKE_CURRENT_BINARY_DIR}/${BASENAME}.html")

        # Process C++ code to ReST
        add_custom_command(
            OUTPUT
                "${rstfile}"
            COMMAND
                "${Python_EXECUTABLE}"
            ARGS
                "${CMAKE_CURRENT_SOURCE_DIR}/processTutorials.py"
                "${infile}"
                "${rstfile}"
                "${tutorial}"
            DEPENDS
                ${infile}
                "${CMAKE_CURRENT_SOURCE_DIR}/processTutorials.py"
        )


        # Convert ReST to HTML
        add_custom_command(
            OUTPUT
                "${htmlfile}"
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
                "${rstfile}" "${htmlfile}"
            DEPENDS
                "${rstfile}" rst_template.txt nav_template.txt
        )

        list(APPEND RST_TARGETS ${TARGET_NAME})

        add_custom_target(${TARGET_NAME} DEPENDS "${htmlfile}")

        install(
            FILES
                "${htmlfile}"
            DESTINATION
                "${CMAKE_DOCDIR_BASE}"
            PERMISSIONS
                OWNER_READ GROUP_READ WORLD_READ
        )

    endforeach()

    # Copy html files to the build area
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

        # Exclude generated search.html
        if (NOT ${src} STREQUAL "search.html")
            add_custom_target( ${src} DEPENDS "${outfile}")
        endif()

        list(APPEND HTML_TARGETS ${src})

        install(
            FILES
                "${outfile}"
            DESTINATION
                "${CMAKE_DOCDIR_BASE}"
        )
    endforeach()

    # Copy the site resources to the build area so that the
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
            "${Python_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/processHtml.py"
            "${CMAKE_CURRENT_BINARY_DIR}"
            "${CMAKE_CURRENT_SOURCE_DIR}/nav_template.txt"
        DEPENDS
            ${HTML_TARGETS}
            ${RST_TARGETS}
            "${CMAKE_CURRENT_SOURCE_DIR}/processHtml.py"
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
